"""Complete payload bits survive independent planar GPU write/read dispatches."""
from contextlib import ExitStack
import numpy as np
import ordinaryshade as osh
from ordinarylight.shaders.native_intersection_programs import NativeIntersection
from ordinarylight.shaders.visibility_cache_programs import storeVisibilityPlanes,loadVisibilityPlanes


@osh.compute(workgroup_size=(64,1,1))
def pack(source:osh.storage_buffer(NativeIntersection,access='read',binding=0),
         visibility_planes:osh.storage_buffer(osh.uvec4,access='write',binding=1)):
    i=osh.global_invocation_id.x
    if i>=osh.array_length(source):return
    storeVisibilityPlanes(source[i],i%osh.u32(221),osh.u32(221),i/osh.u32(221))


@osh.compute(workgroup_size=(64,1,1))
def unpack(visibility_planes:osh.storage_buffer(osh.uvec4,access='read',binding=0),
           restored:osh.storage_buffer(NativeIntersection,access='write',binding=1)):
    i=osh.global_invocation_id.x
    if i>=osh.array_length(restored):return
    restored[i]=loadVisibilityPlanes(i%osh.u32(221),osh.u32(221),i/osh.u32(221))


def test_planar_payload_bit_roundtrip():
    from ordinarylight.runtime import VulkanRuntime,VulkanKernel,compile_compute
    from ordinarylight.pipeline.graph import VulkanGraph,reflected_operation
    from ordinarylight.pipeline.vulkan import VulkanResource
    data=np.random.default_rng(8293).integers(0,2**32,(3,221,7,4),dtype=np.uint32)
    with ExitStack() as stack:
        own=stack.enter_context
        runtime=own(VulkanRuntime())
        source=own(runtime.buffer(data.nbytes,data=data,memory='device'))
        planes=own(runtime.buffer(data.nbytes,memory='device'))
        restored=own(runtime.buffer(data.nbytes,memory='device'))
        graph=VulkanGraph()
        for name,shader,helpers,buffers in (
            ('pack',pack,(storeVisibilityPlanes,),(source,planes)),
            ('unpack',unpack,(loadVisibilityPlanes,),(planes,restored))):
            compiled=osh.compile(shader,helpers=helpers)
            kernel=own(VulkanKernel(runtime,compile_compute(compiled.source),{i:VulkanResource.buffer(b) for i,b in enumerate(buffers)}))
            graph.add(name,reflected_operation(kernel,compiled.reflection,workgroups=(11,1,1)),after=() if name=='pack' else ('pack',))
        graph.compile().execute(runtime).wait()
        np.testing.assert_array_equal(np.frombuffer(planes.read(),np.uint32).reshape(3,7,221,4),data.transpose(0,2,1,3))
        np.testing.assert_array_equal(np.frombuffer(restored.read(),np.uint32).reshape(data.shape),data)
