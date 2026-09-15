"""Primary continuation thinning keeps its expectation and validates its API."""
import os
import numpy as np
import ordinaryshade as osh
import pytest
from ordinarylight.targets.vulkan.api import RendererConfig
from ordinarylight.shaders.fused_primary_programs import primaryDiffuseContinuationWeight

@pytest.mark.parametrize('value',(0.,1e-12,-.1,1.1,float('nan'),float('inf'),True))
def test_invalid_probability(value):
    with pytest.raises(ValueError,match='primary_diffuse_probability'):
        RendererConfig(wavefront_primary_diffuse_probability=value)


def test_default_probability_preserves_existing_policy():
    assert RendererConfig().wavefront_primary_diffuse_probability==1.
    assert RendererConfig(wavefront_primary_diffuse_probability=.25).max_bounces>=4

@osh.compute(workgroup_size=(64,1,1))
def weights(output: osh.storage_buffer(osh.vec4,binding=0)):
    i=osh.global_invocation_id.x
    if i>=osh.array_length(output):return
    u=(osh.f32(i)+0.5)/osh.f32(osh.array_length(output))
    output[i]=osh.vec4(primaryDiffuseContinuationWeight(1.0,u),
                      primaryDiffuseContinuationWeight(0.5,u),
                      primaryDiffuseContinuationWeight(0.25,u),
                      primaryDiffuseContinuationWeight(0.125,u))

@pytest.mark.skipif(os.environ.get('ORDINARYLIGHT_TEST_VULKAN_GRAPH')!='1',reason='Opt-in Vulkan')
def test_gpu_weights_preserve_expected_energy():
    from ordinarylight.runtime import VulkanRuntime,VulkanKernel,compile_compute
    from ordinarylight.pipeline.graph import reflected_operation
    from ordinarylight.pipeline.vulkan import VulkanResource
    with VulkanRuntime() as runtime,runtime.buffer(4096*16) as output:
        program=osh.compile(weights,helpers=(primaryDiffuseContinuationWeight,))
        with VulkanKernel(runtime,compile_compute(program.source),{0:VulkanResource.buffer(output)}) as kernel:
            reflected_operation(kernel,program.reflection,workgroups=(64,1,1)).execute(runtime).wait()
        values=np.frombuffer(output.read(),np.float32).reshape(-1,4)
        np.testing.assert_array_equal(values.mean(axis=0),np.ones(4))
        np.testing.assert_array_equal(np.count_nonzero(values,axis=0),[4096,2048,1024,512])


def test_unsupported_builtin_pipeline_rejects_thinning_before_device_creation():
    from ordinarylight.targets.vulkan.core import VulkanRayQueryCore
    with pytest.raises(ValueError,match='native custom geometry'):
        VulkanRayQueryCore(config=RendererConfig(wavefront_primary_diffuse_probability=.5))
