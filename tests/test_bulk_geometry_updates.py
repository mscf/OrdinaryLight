"""CPU-only bulk update command, shadow and ownership validation."""

from threading import RLock
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import vulkan as vk

from ordinarylight.geometry import CustomGeometryBatch, IntersectionProgram
from ordinarylight.pipeline.vulkan import VulkanResource
from ordinarylight.transport import (
    TransportMaterial,
    VulkanCustomGeometryUpdate,
    VulkanTransportScene,
)
from ordinarylight.transport._custom_batch import CustomSlots, prepare_custom_geometry
from ordinarylight.transport._dynamic_scene import _pack
from ordinarylight.transport.bulk_updates import _plan

PROGRAM = IntersectionProgram("bulkHit", "// callback")


def batch(count=3, identities=None):
    lower = np.arange(count * 3).reshape(count, 3)
    return CustomGeometryBatch(
        np.stack((lower, lower + 1), axis=1),
        PROGRAM,
        identities=np.arange(count) if identities is None else identities,
    )


class Allocation:
    def __init__(self, size, data=b"", handle=700):
        self.size = self.byte_size = size
        self.data = bytes(data)
        self.buffer = handle
        self.closed = False

    def require_open(self):
        if self.closed:
            raise RuntimeError("Allocation closed")

    def close(self):
        self.closed = True


class SceneStub(SimpleNamespace):
    __hash__ = object.__hash__


@pytest.fixture
def scene():
    s = SceneStub(
        custom_capacity=10000,
        programs={PROGRAM.name: PROGRAM},
        _custom_materials=(TransportMaterial(),),
        triangle_count=0,
        boundaries=(),
        boundary_indices={},
        binding_revision=0,
        geometry_revision=0,
        last_completion=None,
        _custom_update_clients=set(),
        require_open=lambda: None,
        _custom_bounds=np.tile([0, 0, 0, 1, 1, 1], (10000, 1)).astype(np.float32),
        _buffers={"custom": Allocation(10000 * 64, handle=100)},
        _bounds_buffer=Allocation(10000 * 24, handle=200),
        _instance_buffer=Allocation(64, handle=300),
        _custom_shape=object(),
        _instance_count=1,
        _builder=SimpleNamespace(_tlas_geometry=lambda buffer: object()),
        _custom_blas=SimpleNamespace(handle=400, scratch=Allocation(64, handle=401)),
        tlas=SimpleNamespace(handle=500, scratch=Allocation(64, handle=501)),
    )
    s.runtime = SimpleNamespace(
        lock=RLock(),
        buffer=Mock(side_effect=lambda size, data, usage: Allocation(size, data)),
    )
    s._boundary_index = lambda identity, material: VulkanTransportScene._boundary_index(
        s, identity, material
    )
    s.resource = lambda name: (
        VulkanResource(s, "acceleration_structure", s.tlas.handle)
        if name == "tlas"
        else VulkanResource(s, "buffer", s._buffers[name].buffer, s._buffers[name].size)
    )
    s.custom_geometry = CustomSlots(batch(), s.custom_capacity)
    return s


def test_unsorted_slots_keep_row_association_and_merge_adjacent_copies(
    scene, monkeypatch
):
    slots = np.array([9, 3, 4])
    geometry = batch(identities=[90, 30, 40])
    update = VulkanCustomGeometryUpdate(scene, slots, geometry)
    slots[:] = 0  # snapshot, not a borrowed input
    calls = []
    monkeypatch.setattr(
        vk,
        "vkCmdCopyBuffer",
        lambda command, src, dst, count, regions: calls.append(
            (dst, [(r.srcOffset, r.dstOffset, r.size) for r in regions])
        ),
    )
    completion = SimpleNamespace(wait=Mock())
    with update:
        op = update.operation()
        assert [p.name for p in op.passes] == [
            "bulk_custom_slot_updates",
            "custom_blas_update",
            "custom_tlas_update",
        ]
        op.validate()
        op.passes[0].record(None)
        assert calls == [
            (100, [(0, 3 * 64, 2 * 64), (2 * 64, 9 * 64, 64)]),
            (200, [(3 * 64, 3 * 24, 2 * 24), (3 * 64 + 2 * 24, 9 * 24, 24)]),
        ]
        assert scene.geometry_revision == 0
        op.submitted(completion)
        assert scene.geometry_revision == 1
        assert scene.custom_geometry[3].identity == 30
        assert scene.custom_geometry[4].identity == 40
        assert scene.custom_geometry[9].identity == 90
        np.testing.assert_array_equal(
            scene._custom_bounds[9], geometry.bounds[0].reshape(6)
        )
        assert op.dependencies() == (completion,)
    completion.wait.assert_called_once()
    assert not scene._custom_update_clients and update._staging.closed
    with pytest.raises(RuntimeError, match="closed"):
        op.validate()


def test_large_update_is_two_copies_and_does_not_expand_geometry(scene, monkeypatch):
    geometry = batch(4096)
    monkeypatch.setattr(
        CustomGeometryBatch, "__getitem__", lambda *args: pytest.fail("expanded batch")
    )
    with VulkanCustomGeometryUpdate(scene, np.arange(4096), geometry) as update:
        assert update._record_ranges == ((0, 0, 4096 * 64),)
        assert update._bound_ranges == ((4096 * 64, 0, 4096 * 24),)
        op = update.operation(mode="rebuild")
        op.submitted(SimpleNamespace(wait=lambda: None))
        packed, _, shadow = prepare_custom_geometry(scene, scene.custom_geometry, 10002)
        assert len(shadow.bulk.indices) == 4096
        np.testing.assert_array_equal(packed["metadata"][:4096, 3], np.arange(4096))
        assert shadow.base is scene.custom_geometry.base


def test_removal_keeps_bounds_and_skips_acceleration(scene):
    original_bounds = scene._custom_bounds.copy()
    with VulkanCustomGeometryUpdate(scene, [2, 0], None) as update:
        op = update.operation(mode="rebuild")
        assert len(op.passes) == 1
        assert update._bound_ranges == ()
        op.submitted(SimpleNamespace(wait=lambda: None))
        assert scene.custom_geometry[0] is None and scene.custom_geometry[2] is None
        np.testing.assert_array_equal(scene._custom_bounds, original_bounds)
        packed, _, _ = prepare_custom_geometry(scene, scene.custom_geometry, 10000)
        assert packed[0].tobytes() == _pack(scene, None)


def test_overlapping_object_and_bulk_updates_are_last_write_wins(scene):
    initial = scene.custom_geometry
    first, *_ = _plan(scene, [0, 2], batch(2, [11, 22]))
    state = initial.bulk_updated(first)
    state = state.updated({0: batch(1, [33])[0]})
    assert state[0].identity == 33
    second, *_ = _plan(scene, [0], batch(1, [44]))
    state = state.bulk_updated(second)
    assert state[0].identity == 44 and state[2].identity == 22
    for _ in range(10):
        state = state.bulk_updated(second)
    assert len(state.bulk.indices) == 2  # no retained per-frame history
    assert not state.updates
    assert initial.bulk is None
    packed, _, grown = prepare_custom_geometry(scene, state, 10001)
    assert packed[0].tobytes() == _pack(scene, state[0])
    assert grown[2].identity == 22


@pytest.mark.parametrize(
    "slots", [[], [1, 1], [-1], [10000], [1.5], [True], [[0]], [2**64 - 1]]
)
def test_bad_slots_fail_before_allocation(scene, slots):
    with pytest.raises(ValueError):
        VulkanCustomGeometryUpdate(scene, slots, None)
    scene.runtime.buffer.assert_not_called()


def test_bad_geometry_and_program_fail_before_allocation(scene):
    for geometry, error in [(batch(2), ValueError), ([batch(1)[0]], TypeError)]:
        with pytest.raises(error):
            VulkanCustomGeometryUpdate(scene, [0], geometry)
    other = CustomGeometryBatch(
        batch(1).bounds, IntersectionProgram("newHit", "// new")
    )
    with pytest.raises(ValueError, match="replacement scene"):
        VulkanCustomGeometryUpdate(scene, [0], other)
    scene.runtime.buffer.assert_not_called()
    assert tuple(scene.programs) == (PROGRAM.name,)


def test_dependencies_stale_bindings_and_close(scene):
    update = VulkanCustomGeometryUpdate(scene, [0], batch(1))
    explicit = object()
    op = update.operation(after=[explicit])
    assert op.dependencies() == (explicit,)
    with pytest.raises(ValueError, match="mode"):
        update.operation(mode="wrong")
    scene.binding_revision += 1
    with pytest.raises(ValueError, match="bindings changed"):
        op.validate()
    update.close()  # stale clients must still be closeable
    update.close()
    assert not scene._custom_update_clients


def test_failed_allocation_does_not_register_client(scene):
    scene.runtime.buffer.side_effect = RuntimeError("allocation failed")
    with pytest.raises(RuntimeError, match="allocation failed"):
        VulkanCustomGeometryUpdate(scene, [0], batch(1))
    assert not scene._custom_update_clients


def test_boundary_and_material_mapping_match_single_updates(scene):
    scene.triangle_count = 7
    scene._custom_materials = (TransportMaterial(), TransportMaterial("dielectric"))
    scene.boundaries = (SimpleNamespace(identity=42),)
    scene.boundary_indices = {42: 0}
    geometry = CustomGeometryBatch(
        batch(2).bounds,
        PROGRAM,
        materials=[1, 0],
        boundaries=[42, 0xFFFFFFFF],
        identities=[0xFFFFFFFF, 0x80000001],
    )
    change, _, payload, *_ = _plan(scene, [7, 1], geometry)
    assert payload[:64] == _pack(scene, geometry[1])
    assert payload[64:128] == _pack(scene, geometry[0])
    assert change.geometry(1) == geometry[0]


def test_fragmented_command_regions_are_bounded(scene, monkeypatch):
    geometry = batch(5000)
    calls = []
    monkeypatch.setattr(
        vk, "vkCmdCopyBuffer", lambda cmd, src, dst, count, regions: calls.append(count)
    )
    with VulkanCustomGeometryUpdate(scene, np.arange(0, 10000, 2), geometry) as update:
        update.operation().passes[0].record(None)
    assert calls == [4096, 904, 4096, 904]


def test_bulk_copy_graph_orders_read_after_transfer_and_build(scene):
    from ordinarylight.pipeline.graph import VulkanGraph, VulkanOperation
    from ordinarylight.pipeline.vulkan import VulkanPass, VulkanResourceUse

    with VulkanCustomGeometryUpdate(scene, [0], batch(1)) as update:
        read = VulkanOperation(
            [
                VulkanPass(
                    "read",
                    (
                        VulkanResourceUse(
                            scene.resource("custom"),
                            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            vk.VK_ACCESS_SHADER_READ_BIT,
                        ),
                        VulkanResourceUse(
                            scene.resource("tlas"),
                            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            vk.VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
                        ),
                    ),
                    lambda command: None,
                )
            ]
        )
        schedule = (
            VulkanGraph()
            .add("edit", update.operation())
            .add("read", read, after=("edit",))
            .compile()
        )
        assert [node.name for node in schedule.nodes] == ["edit", "read"]


def test_public_bulk_sphere_example_when_gpu_validation_is_enabled(monkeypatch):
    import os
    from pathlib import Path

    if os.environ.get("ORDINARYLIGHT_TEST_VULKAN_TRANSPORT") != "1":
        pytest.skip("GPU validation explicitly deferred/opt-in")
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[1] / "examples/transport_client/src")
    )
    from ordinarylight_transport_demo.bulk_updates import run

    assert run(8) == dict(spheres=8, moved=8, removed=4, invalid_paths=0)


def test_repacking_shadow_remaps_new_scene_indices(scene):
    scene.triangle_count = 7
    scene._custom_materials = (TransportMaterial(), TransportMaterial("dielectric"))
    scene.boundaries = (SimpleNamespace(identity=42),)
    scene.boundary_indices = {42: 0}
    geometry = CustomGeometryBatch(batch(1).bounds, PROGRAM, materials=1, boundaries=42)
    change, *_ = _plan(scene, [0], geometry)
    shadow = scene.custom_geometry.bulk_updated(change)
    scene.triangle_count = 12
    scene.programs = {
        "other": IntersectionProgram("other", "// other"),
        PROGRAM.name: PROGRAM,
    }
    scene.boundaries = (SimpleNamespace(identity=99), SimpleNamespace(identity=42))
    scene.boundary_indices = {99: 0, 42: 1}
    packed, _, _ = prepare_custom_geometry(scene, shadow, 10000)
    assert packed[0].tobytes() == _pack(scene, geometry[0])
    scene._custom_materials = (TransportMaterial(), TransportMaterial())
    with pytest.raises(ValueError, match="dielectric material"):
        prepare_custom_geometry(scene, shadow, 10000)
