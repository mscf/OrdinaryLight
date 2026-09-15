"""Compare separate and combined resolve with samples, history, resize and reuse."""
from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest


@pytest.mark.parametrize("reuse",(False,True))
def test_fused_resolve_samples_history_resize(reuse):
    import vulkan as vk
    from ordinarylight.integrations.raster_workbench import _gi_config
    from ordinarylight.pipeline.graph import VulkanGraph
    from ordinarylight.pipeline.vulkan import VulkanResource, VulkanResourceUse, VulkanPass
    from ordinarylight.runtime import VulkanRuntime, VulkanWavefrontPipeline
    from tools.denoiser_motion.glass_detail import fixture, pose
    scene, glass, bars = fixture()
    config = replace(_gi_config(SimpleNamespace(id='glass-detail', renderer={}),
        render_scale=1.0, upscale_filter='bilinear', capture=True),
        samples_per_pixel=2, max_bounces=4, wavefront_primary_hits=True,
        wavefront_tile_capacity=1024, wavefront_raw_hdr_output=True)
    camera = replace(pose(scene, glass, bars, 'camera', 0), vertical_fov_degrees=70)
    enabled = False
    rows = {}
    with ExitStack() as stack:
        runtime = stack.enter_context(VulkanRuntime(config=config))
        resident = stack.enter_context(runtime.upload_scene(scene))
        for mode in ('fused', 'split'):
            enabled = mode == 'split'
            pipeline = stack.enter_context(VulkanWavefrontPipeline(runtime, resident, config=replace(config,denoiser_fused_resolve=enabled,wavefront_indirect_reuse_candidates=reuse,wavefront_indirect_reuse_storage=reuse)))
            rows[mode] = []
            for extent in ((65, 49), (65, 49), (81, 57), (65, 49)):
                frame = pipeline.prepare(camera, extent)
                hits = frame.buffers['primary_hits']
                with runtime.buffer(hits.byte_size) as staging:
                    graph = VulkanGraph().add('lighting', frame.operation)
                    graph.add('read', VulkanPass('copy', (
                        VulkanResourceUse(VulkanResource.buffer(hits), vk.VK_PIPELINE_STAGE_TRANSFER_BIT, vk.VK_ACCESS_TRANSFER_READ_BIT),
                        VulkanResourceUse(VulkanResource.buffer(staging), vk.VK_PIPELINE_STAGE_TRANSFER_BIT, vk.VK_ACCESS_TRANSFER_WRITE_BIT),
                    ), lambda command: vk.vkCmdCopyBuffer(command, hits.buffer, staging.buffer, 1, [vk.VkBufferCopy(size=hits.byte_size)])))
                    graph.compile().execute(runtime).wait()
                    records = np.frombuffer(staging.read(), hits.record_dtype).copy()
                assert len(records) == 2 * extent[0] * extent[1]
                rows[mode].append((records, pipeline.capture_wavefront_hdr()))
    for (a, ahdr), (b, bhdr) in zip(rows['fused'], rows['split']):
        assert (a['position_distance'][:,3] >= 0).any()
        np.testing.assert_array_equal(a, b)
        assert np.isfinite(ahdr).all() and np.isfinite(bhdr).all()
        np.testing.assert_array_equal(ahdr, bhdr)
