"""Public hit record selection, lifetime and same-ray identity."""
import os
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest
from ordinarylight.targets.vulkan.api import RendererConfig
from ordinarylight.wavefront import PRIMARY_HIT_DTYPE, PRIMARY_HIT_IDENTITY_DTYPE, primary_hit_dtype


def test_primary_hit_format_contract():
    assert RendererConfig().wavefront_primary_hit_format == 'full'
    assert primary_hit_dtype('full') == PRIMARY_HIT_DTYPE
    assert primary_hit_dtype('identity') == PRIMARY_HIT_IDENTITY_DTYPE
    assert PRIMARY_HIT_IDENTITY_DTYPE.itemsize == 20
    assert PRIMARY_HIT_IDENTITY_DTYPE.fields['valid'][1] == 16
    for invalid in ('compact', '', None):
        with pytest.raises(ValueError):
            RendererConfig(wavefront_primary_hit_format=invalid)
        with pytest.raises(ValueError):
            primary_hit_dtype(invalid)


@pytest.mark.skipif(os.environ.get('ORDINARYLIGHT_TEST_VULKAN_GRAPH') != '1', reason='opt-in GPU')
def test_identity_records_match_full_across_samples_and_resize():
    import vulkan as vk
    from ordinarylight.integrations.raster_workbench import _gi_config
    from ordinarylight.pipeline.graph import VulkanGraph
    from ordinarylight.pipeline.vulkan import VulkanResource, VulkanResourceUse, VulkanPass
    from ordinarylight.runtime import VulkanRuntime, VulkanWavefrontPipeline
    from tools.denoiser_motion.glass_detail import fixture, pose
    scene, glass, bars = fixture()
    config = replace(_gi_config(SimpleNamespace(id='glass-detail', renderer={}),
        render_scale=0.5, upscale_filter='bilinear', capture=True), max_bounces=4,
        samples_per_pixel=2, wavefront_primary_hits=True)
    camera = replace(pose(scene, glass, bars, 'camera', 0), vertical_fov_degrees=70)
    results = {}
    with VulkanRuntime(config=config) as runtime, runtime.upload_scene(scene) as resident:
        for format in ('full', 'identity'):
            results[format] = []
            with VulkanWavefrontPipeline(runtime, resident, config=replace(config,
                    wavefront_primary_hit_format=format)) as pipeline:
                with pytest.raises(ValueError, match='recreation'):
                    pipeline.reconfigure(wavefront_primary_hit_format='identity')
                previous = None
                for extent in ((130, 98), (162, 114), (130, 98)):
                    frame = pipeline.prepare(camera, extent)
                    hits = frame.buffers['primary_hits']
                    w, h = frame.render_extent
                    assert frame.sample_count == 2
                    assert hits.record_dtype == primary_hit_dtype(format)
                    assert hits.byte_size == 2 * w * h * hits.record_dtype.itemsize
                    with runtime.buffer(hits.byte_size) as staging:
                        graph = VulkanGraph().add('lighting', frame.operation)
                        graph.add('read', VulkanPass('copy', (
                            VulkanResourceUse(VulkanResource.buffer(hits), vk.VK_PIPELINE_STAGE_TRANSFER_BIT, vk.VK_ACCESS_TRANSFER_READ_BIT),
                            VulkanResourceUse(VulkanResource.buffer(staging), vk.VK_PIPELINE_STAGE_TRANSFER_BIT, vk.VK_ACCESS_TRANSFER_WRITE_BIT),
                        ), lambda command: vk.vkCmdCopyBuffer(command, hits.buffer, staging.buffer, 1, [vk.VkBufferCopy(size=hits.byte_size)])))
                        graph.compile().execute(runtime).wait()
                        data = np.frombuffer(staging.read(), hits.record_dtype).copy().reshape(2,h,w)
                    if previous is not None:
                        with pytest.raises(RuntimeError, match='retired'):
                            previous.require_open()
                    previous = hits
                    hdr = pipeline.capture_wavefront_hdr()
                    assert np.isfinite(hdr).all()
                    results[format].append((data, hdr))
            with pytest.raises(RuntimeError):
                previous.require_open()
    for (full, full_hdr), (compact, compact_hdr) in zip(results['full'], results['identity']):
        valid = full['position_distance'][...,3] >= 0
        assert valid.any() and (~valid).any()
        np.testing.assert_array_equal(full['identity'], compact['identity'])
        np.testing.assert_array_equal(valid, compact['valid'] != 0)
        np.testing.assert_allclose(full_hdr, compact_hdr, atol=0.002, rtol=0.002)
