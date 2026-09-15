"""A fused primary dispatch may own initialization only when it visits every record."""
import os
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import pytest


@pytest.mark.parametrize('signals,fused,strategy,stride,needed', [
    (False,True,'wavefront',0,False), (True,False,'wavefront',0,True),
    (True,True,'wavefront',0,False), (True,True,'wavefront',1,False),
    (True,True,'wavefront',2,True), (True,True,'wavefront',4,True),
    (True,True,'hybrid',0,True), (True,True,'megakernel',0,True),
])
def test_secondary_initialization_owner(signals,fused,strategy,stride,needed):
    from ordinarylight.targets.vulkan.core import VulkanWavefrontExecutor
    executor = SimpleNamespace(core=SimpleNamespace(resolved_execution_strategy=strategy),
        _denoiser_signals_active=lambda:signals, _indirect_capture_stride=lambda:stride)
    assert VulkanWavefrontExecutor._needs_secondary_transfer_clear(executor,fused) == needed


@pytest.mark.skipif(os.environ.get('ORDINARYLIGHT_TEST_VULKAN_GRAPH') != '1', reason='opt-in GPU')
@pytest.mark.parametrize('samples,stride,bounces',[(1,0,4),(2,0,4),(1,0,1),(1,1,4),(1,2,4),(1,4,4)])
def test_reused_secondary_tiles_match_transfer_clear(monkeypatch,samples,stride,bounces):
    from ordinarylight.integrations.raster_workbench import _gi_config
    from ordinarylight.runtime import VulkanRuntime, VulkanWavefrontPipeline
    from ordinarylight.targets.vulkan.core import VulkanWavefrontExecutor
    from tools.denoiser_motion.glass_detail import fixture, pose
    scene, glass, bars = fixture()
    config = replace(_gi_config(SimpleNamespace(id='glass-detail',renderer={}),
        render_scale=1,upscale_filter='bilinear',capture=True),
        max_bounces=bounces,samples_per_pixel=samples,wavefront_tile_capacity=128,
        wavefront_primary_hits=True,wavefront_primary_hit_format='identity',
        wavefront_indirect_reuse_storage=bool(stride),wavefront_indirect_reuse_candidates=bool(stride),wavefront_indirect_reuse_scale=1/stride if stride else .5)
    camera = pose(scene,glass,bars,'camera',0)
    miss = replace(camera,target=(camera.position[0],camera.position[1],camera.position[2]-10))
    results=[]
    with VulkanRuntime(config=config) as runtime, runtime.upload_scene(scene) as resident:
        for baseline in (True,False):
            with monkeypatch.context() as patch:
                if baseline:
                    patch.setattr(VulkanWavefrontExecutor,'_needs_secondary_transfer_clear',lambda executor,fused:executor._denoiser_signals_active())
                with VulkanWavefrontPipeline(runtime,resident,config=config) as pipeline:
                    frames=[]
                    for view,extent in ((camera,(33,25)),(miss,(33,25)),(camera,(35,27)),(miss,(35,27))):
                        frame=pipeline.render(view,extent)
                        frame.completion.wait()
                        hdr=pipeline.capture_wavefront_hdr()
                        assert np.isfinite(hdr).all()
                        frames.append(hdr.copy())
                    results.append(frames)
    for before,after in zip(*results):
        np.testing.assert_array_equal(before,after)
