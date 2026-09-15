"""Cached planning must not retain frame callbacks, ownership or stale layouts."""
import gc
import weakref

import pytest
import vulkan as vk

from ordinarylight.pipeline.graph import VulkanGraph
from ordinarylight.pipeline.vulkan import (
    VulkanPass, VulkanPassPipeline, VulkanResource, VulkanResourceUse, _barrier_plan,
)


class Owner:
    def __init__(self, runtime=None):
        self.runtime = runtime
        self.layout = vk.VK_IMAGE_LAYOUT_UNDEFINED
        self.binding_revision = 0
        self.closed = False

    def require_open(self):
        if self.closed:
            raise RuntimeError('closed')


def test_schedule_reuse_keeps_fresh_callbacks_revisions_and_no_owners(monkeypatch):
    owner = Owner()
    resource = VulkanResource(owner, 'buffer', 70001, 16)
    def graph(callback, *, access=vk.VK_ACCESS_SHADER_WRITE_BIT, after=()):
        return VulkanGraph().add('write', VulkanPass('write', (
            VulkanResourceUse(resource, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, access),
        ), callback), after=after)
    calls = []
    graph(lambda c: calls.append('old')).compile()
    original = VulkanGraph._compile_uncached
    def forbidden(self):
        raise AssertionError('Unchanged scheduling declarations were recompiled')
    monkeypatch.setattr(VulkanGraph, '_compile_uncached', forbidden)
    owner.binding_revision = 1
    compiled = graph(lambda c: calls.append('new')).compile()
    compiled.nodes[0].operation.passes[0].record(None)
    assert calls == ['new']
    assert compiled._generations[owner] == 1
    monkeypatch.setattr(VulkanGraph, '_compile_uncached', original)
    with pytest.raises(ValueError, match='Unknown dependency'):
        graph(lambda c: None, after=('absent',)).compile()
    with pytest.raises(ValueError, match='Unsupported access'):
        graph(lambda c: None, access=1 << 31).compile()
    reference = weakref.ref(owner)
    del owner, resource, compiled, graph
    gc.collect()
    assert reference() is None


def test_barrier_replay_reads_current_layout_and_preserves_range_hazards(monkeypatch):
    runtime = object()
    image_owner, buffer_owner = Owner(runtime), Owner(runtime)
    image = VulkanResource(image_owner, 'image', 70002)
    buffer = VulkanResource(buffer_owner, 'buffer', 70003, 64)
    calls, recorded = [], []
    monkeypatch.setattr(vk, 'vkCmdPipelineBarrier', lambda *args: recorded.append(args))
    def pipeline(label):
        return VulkanPassPipeline([
            VulkanPass('write', (
                VulkanResourceUse(buffer.byte_range(0, 16), vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                                  vk.VK_ACCESS_TRANSFER_WRITE_BIT),
                VulkanResourceUse(image, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                  vk.VK_ACCESS_SHADER_WRITE_BIT, vk.VK_IMAGE_LAYOUT_GENERAL),
            ), lambda c: calls.append(label)),
            VulkanPass('disjoint', (VulkanResourceUse(buffer.byte_range(32, 16),
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.VK_ACCESS_SHADER_READ_BIT),), lambda c: None),
            VulkanPass('read', (VulkanResourceUse(buffer.byte_range(0, 16),
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.VK_ACCESS_SHADER_READ_BIT),), lambda c: None),
        ])
    _barrier_plan.cache_clear()
    for iteration in range(3):
        record, owners, commit = pipeline(iteration)._prepare_recording(runtime)
        recorded.clear()
        record(None)
        image_barrier = recorded[0][-1][0]
        assert image_barrier.oldLayout == (vk.VK_IMAGE_LAYOUT_UNDEFINED if iteration == 0
                                           else vk.VK_IMAGE_LAYOUT_GENERAL)
        # Disjoint access must not erase the earlier overlapping write dependency.
        assert recorded[2][7][0].srcAccessMask & vk.VK_ACCESS_TRANSFER_WRITE_BIT
        assert recorded[2][7][0].offset == 0
        assert recorded[2][7][0].size == 16
        commit()
    assert calls == [0, 1, 2]
    assert _barrier_plan.cache_info().hits == 1
    # Read entry layouts when recording, even if preparation happened earlier.
    record, _, commit = pipeline('changed')._prepare_recording(runtime)
    image_owner.layout = vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
    recorded.clear()
    record(None)
    assert recorded[0][-1][0].oldLayout == vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
    assert image_owner.layout == vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
    commit()
    assert image_owner.layout == vk.VK_IMAGE_LAYOUT_GENERAL
    image_owner.closed = True
    with pytest.raises(RuntimeError, match='closed'):
        pipeline('closed')._prepare_recording(runtime)
