"""GPU ABI/bit-preservation check for a distance-only position cache."""
from contextlib import ExitStack
import numpy as np
import ordinaryshade as osh
from ordinarylight.shaders.visibility_cache_programs import storeDistancePlanes,loadDistancePlanes
from ordinarylight.shaders.native_intersection_programs import NativeIntersection


@osh.compute(workgroup_size=(64,1,1))
def roundtrip(source:osh.storage_buffer(NativeIntersection,access='read',binding=0),
              origins:osh.storage_buffer(osh.vec4,access='read',binding=1),
              directions:osh.storage_buffer(osh.vec4,access='read',binding=2),
              reference:osh.storage_buffer(NativeIntersection,access='write',binding=3),
              visibility_planes:osh.storage_buffer(osh.uvec4,access='write',binding=4),
              restored:osh.storage_buffer(NativeIntersection,access='write',binding=5)):
    i=osh.global_invocation_id.x
    if i>=osh.array_length(source):return
    hit=source[i]
    hit.position_distance.xyz=osh.vec3(0.0)
    if hit.address.w!=osh.u32(0):
        hit.position_distance.xyz=origins[i].xyz+hit.position_distance.w*directions[i].xyz
    reference[i]=hit
    storeDistancePlanes(hit,i%osh.u32(221),osh.u32(221),i/osh.u32(221))



@osh.compute(workgroup_size=(64,1,1))
def restore(visibility_planes:osh.storage_buffer(osh.uvec4,access='read',binding=0),
            origins:osh.storage_buffer(osh.vec4,access='read',binding=1),
            directions:osh.storage_buffer(osh.vec4,access='read',binding=2),
            restored:osh.storage_buffer(NativeIntersection,access='write',binding=3)):
    i=osh.global_invocation_id.x
    if i>=osh.array_length(restored):return
    restored[i]=loadDistancePlanes(i%osh.u32(221),osh.u32(221),i/osh.u32(221),origins[i].xyz,directions[i].xyz)


def test_distance_cache_preserves_payload_bits_and_position():
    from ordinarylight.runtime import VulkanRuntime,VulkanKernel,compile_compute
    from ordinarylight.pipeline.graph import VulkanGraph,reflected_operation
    from ordinarylight.pipeline.vulkan import VulkanResource
    count=3*221
    plane_words=6*221*4+((221+3)//4)*4
    initial=np.full((3,plane_words),0xDEADBEEF,dtype=np.uint32)
    rng=np.random.default_rng(2365)
    words=rng.integers(0,2**32,(count,28),dtype=np.uint32)
    # Keep all payload bits, including NaNs and high-bit identity values.
    words[:,3]=rng.uniform(.001,1e5,count).astype(np.float32).view(np.uint32)
    words[:,19]=np.arange(count,dtype=np.uint32)%3
    origins=rng.uniform(-1e5,1e5,(count,4)).astype(np.float32)
    directions=rng.normal(size=(count,4)).astype(np.float32)
    directions[:,:3]/=np.linalg.norm(directions[:,:3],axis=1)[:,None]
    compiled=osh.compile(roundtrip,helpers=(storeDistancePlanes,))
    with ExitStack() as stack:
        own=stack.enter_context;runtime=own(VulkanRuntime())
        inputs=[own(runtime.buffer(a.nbytes,data=a,memory='device')) for a in (words,origins,directions)]
        outputs=[own(runtime.buffer(count*112,memory='device')),own(runtime.buffer(initial.nbytes,data=initial,memory='device')),own(runtime.buffer(count*112,memory='device'))]
        kernel=own(VulkanKernel(runtime,compile_compute(compiled.source),{i:VulkanResource.buffer(b) for i,b in enumerate(inputs+outputs)}))
        restore_compiled=osh.compile(restore,helpers=(loadDistancePlanes,))
        restore_kernel=own(VulkanKernel(runtime,compile_compute(restore_compiled.source),
            {i:VulkanResource.buffer(b) for i,b in enumerate((outputs[1],inputs[1],inputs[2],outputs[2]))}))
        graph=VulkanGraph().add('pack',reflected_operation(kernel,compiled.reflection,workgroups=((count+63)//64,1,1)))
        graph.add('restore',reflected_operation(restore_kernel,restore_compiled.reflection,workgroups=((count+63)//64,1,1)),after=('pack',)).compile().execute(runtime).wait()
        reference=np.frombuffer(outputs[0].read(),np.uint32).reshape(count,28)
        packed=np.frombuffer(outputs[1].read(),np.uint32).reshape(3,plane_words)
        restored=np.frombuffer(outputs[2].read(),np.uint32).reshape(count,28)
        np.testing.assert_array_equal(reference[:,4:],words[:,4:])
        np.testing.assert_array_equal(packed[:,:6*221*4].reshape(3,6,221,4),words[:,4:].reshape(3,221,6,4).transpose(0,2,1,3))
        np.testing.assert_array_equal(packed[:,6*221*4:][:,:221],words[:,3].reshape(3,221))
        np.testing.assert_array_equal(packed[:,6*221*4:][:,221:],initial[:,6*221*4:][:,221:])
        np.testing.assert_array_equal(restored,reference)
