"""Public independently updateable acceleration structures and GPU range copies."""
import os
import gc
from contextlib import ExitStack
import numpy as np
import pytest
import ordinaryshade as osh

pytestmark=pytest.mark.skipif(os.environ.get('ORDINARYLIGHT_TEST_VULKAN_GRAPH')!='1',reason='Opt-in Vulkan graph test')

@osh.compute(workgroup_size=(1,1,1))
def hit_instances(scene: osh.acceleration_structure(binding=0), output: osh.storage_buffer(osh.uvec4,binding=1)):
    i=osh.global_invocation_id.x
    query=osh.ray_query()
    query.initialize(scene,osh.u32(0),osh.u32(255),osh.vec3(0.5+2.0*osh.f32(i),0.5,-2.0),0.001,osh.vec3(0,0,1),100.0)
    while query.proceed():
        query.generate_intersection(2.0)
    result=osh.uvec4(4294967295)
    if query.intersection_type(True)!=osh.u32(0):
        result=osh.uvec4(query.instance_id(True),query.instance_custom_index(True),query.primitive_index(True),osh.u32(1))
    output[i]=result


def test_chunk_refit_ranges_leases_and_deferred_build(monkeypatch):
    import vulkan as vk
    from ordinarylight.runtime import (VulkanRuntime,VulkanAabbBlas,VulkanTlas,VulkanBlasInstance,
        VulkanKernel,compile_compute,buffer_copy_operation)
    from ordinarylight.pipeline.vulkan import VulkanResource
    from ordinarylight.pipeline.graph import VulkanGraph,reflected_operation
    with ExitStack() as resources:
        own=resources.enter_context
        runtime=own(VulkanRuntime())
        boxes=np.array([[0,0,0,1,1,1],[2,0,0,3,1,1]],np.float32)
        usage=vk.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR|vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        bounds=own(runtime.buffer(boxes.nbytes,data=boxes,memory='device',device_address=True,usage=usage))
        a=own(VulkanAabbBlas(runtime,VulkanResource.buffer(bounds).byte_range(0,24),1))
        b=own(VulkanAabbBlas(runtime,VulkanResource.buffer(bounds).byte_range(24,24),1))
        with pytest.raises(ValueError,match='submitted'):
            b.operation(mode='refit')
        packed=VulkanBlasInstance(a,custom_index=7).pack()+VulkanBlasInstance(b,custom_index=9).pack()
        instances=own(runtime.buffer(128,data=packed,memory='device',device_address=True,usage=usage))
        tlas=own(VulkanTlas(runtime,instances,2,referenced_blas=(a,b)))
        output=own(runtime.buffer(32))
        compiled=osh.compile(hit_instances)
        kernel=own(VulkanKernel(runtime,compile_compute(compiled.source),{0:tlas.resource,1:VulkanResource.buffer(output)}))
        trace=lambda:reflected_operation(kernel,compiled.reflection,workgroups=(2,1,1))
        graph=VulkanGraph().add('a',a.operation()).add('b',b.operation()).add('tlas',tlas.operation()).add('trace',trace())
        gc.collect()  # Vulkan range structs must survive deferred recording.
        graph.compile().execute(runtime).wait()
        values=np.frombuffer(output.read(),np.uint32).reshape(2,4)
        np.testing.assert_array_equal(values,[[0,7,0,1],[1,9,0,1]])
        with pytest.raises(RuntimeError,match='borrowers'):a.close()
        with pytest.raises(RuntimeError,match='borrowers'):bounds.close()
        before=a.last_completion
        moved=own(runtime.buffer(24,data=np.array([4,0,0,5,1,1],np.float32),memory='device'))
        copy=buffer_copy_operation(moved,bounds,[(0,24,24)])
        graph=VulkanGraph().add('copy',copy).add('b',b.operation(mode='refit')).add('tlas',tlas.operation(mode='refit')).add('trace',trace())
        def forbidden(*args,**kwargs):raise AssertionError('Stage allocated or waited on CPU')
        with monkeypatch.context() as patch:
            patch.setattr(runtime,'buffer',forbidden)
            patch.setattr(vk,'vkQueueWaitIdle',forbidden)
            patch.setattr(vk,'vkDeviceWaitIdle',forbidden)
            completion=graph.compile().execute(runtime)
        completion.wait()
        assert a.last_completion is before
        values=np.frombuffer(output.read(),np.uint32).reshape(2,4)
        np.testing.assert_array_equal(values[0],[0,7,0,1])
        assert np.all(values[1]==0xffffffff)
        with pytest.raises(ValueError,match='exceeds'):buffer_copy_operation(moved,bounds,[(0,47,24)])
        with pytest.raises(ValueError,match='overlap'):buffer_copy_operation(moved,bounds,[(0,0,16),(0,8,16)])
