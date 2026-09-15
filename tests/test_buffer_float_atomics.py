import os
import numpy as np
import ordinaryshade as osh
import pytest

@osh.compute(workgroup_size=(64,1,1),capabilities=('buffer_float32_atomic_add',))
def accumulate(sums: osh.storage_buffer(osh.f32,access='read_write',binding=0)):
    osh.atomic_add(sums[0],0.125)
    osh.atomic_add(sums[1],-0.25)
    osh.atomic_add(sums[2],1024.0)

@pytest.mark.skipif(os.environ.get('ORDINARYLIGHT_TEST_VULKAN_GRAPH')!='1',reason='Opt-in Vulkan')
def test_contended_float_atomic_add_preserves_hdr():
    from ordinarylight.runtime import VulkanRuntime,VulkanKernel,compile_compute
    from ordinarylight.pipeline.graph import reflected_operation
    from ordinarylight.pipeline.vulkan import VulkanResource
    with VulkanRuntime() as runtime,runtime.buffer(12) as output:
        if not runtime.capabilities.buffer_float32_atomic_add:
            pytest.skip('Device has no buffer float32 atomic addition')
        output.upload(np.zeros(3,dtype=np.float32).tobytes())
        program=osh.compile(accumulate)
        with VulkanKernel(runtime,compile_compute(program.source),{0:VulkanResource.buffer(output)}) as kernel:
            reflected_operation(kernel,program.reflection,workgroups=(64,1,1)).execute(runtime).wait()
        np.testing.assert_array_equal(np.frombuffer(output.read(),np.float32),[512.,-1024.,4194304.])
