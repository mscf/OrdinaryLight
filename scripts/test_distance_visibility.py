"""GPU ABI/bit-preservation check for a distance-only position cache."""
from contextlib import ExitStack
import numpy as np
import ordinaryshade as osh
from ordinarylight.shaders.visibility_cache_programs import DistanceVisibility,packDistanceVisibility,unpackDistanceVisibility
from ordinarylight.shaders.native_intersection_programs import NativeIntersection


@osh.compute(workgroup_size=(64,1,1))
def roundtrip(source:osh.storage_buffer(NativeIntersection,access='read',binding=0),
              origins:osh.storage_buffer(osh.vec4,access='read',binding=1),
              directions:osh.storage_buffer(osh.vec4,access='read',binding=2),
              reference:osh.storage_buffer(NativeIntersection,access='write',binding=3),
              packed:osh.storage_buffer(DistanceVisibility,access='write',binding=4),
              restored:osh.storage_buffer(NativeIntersection,access='write',binding=5)):
    i=osh.global_invocation_id.x
    if i>=osh.array_length(source):return
    hit=source[i]
    hit.position_distance.xyz=osh.vec3(0.0)
    if hit.address.w!=osh.u32(0):
        hit.position_distance.xyz=origins[i].xyz+hit.position_distance.w*directions[i].xyz
    reference[i]=hit
    value=packDistanceVisibility(hit)
    packed[i]=value



@osh.compute(workgroup_size=(64,1,1))
def restore(packed:osh.storage_buffer(DistanceVisibility,access='read',binding=0),
            origins:osh.storage_buffer(osh.vec4,access='read',binding=1),
            directions:osh.storage_buffer(osh.vec4,access='read',binding=2),
            restored:osh.storage_buffer(NativeIntersection,access='write',binding=3)):
    i=osh.global_invocation_id.x
    if i>=osh.array_length(packed):return
    restored[i]=unpackDistanceVisibility(packed[i],origins[i].xyz,directions[i].xyz)


def test_distance_cache_preserves_payload_bits_and_position():
    from ordinarylight.runtime import VulkanRuntime,VulkanKernel,compile_compute
    from ordinarylight.pipeline.graph import VulkanGraph,reflected_operation
    from ordinarylight.pipeline.vulkan import VulkanResource
    count=4099
    rng=np.random.default_rng(2365)
    words=rng.integers(0,2**32,(count,28),dtype=np.uint32)
    # Keep all payload bits, including NaNs and high-bit identity values.
    words[:,3]=rng.uniform(.001,1e5,count).astype(np.float32).view(np.uint32)
    words[:,19]=np.arange(count,dtype=np.uint32)%3
    origins=rng.uniform(-1e5,1e5,(count,4)).astype(np.float32)
    directions=rng.normal(size=(count,4)).astype(np.float32)
    directions[:,:3]/=np.linalg.norm(directions[:,:3],axis=1)[:,None]
    compiled=osh.compile(roundtrip,helpers=(packDistanceVisibility,unpackDistanceVisibility))
    with ExitStack() as stack:
        own=stack.enter_context;runtime=own(VulkanRuntime())
        inputs=[own(runtime.buffer(a.nbytes,data=a,memory='device')) for a in (words,origins,directions)]
        outputs=[own(runtime.buffer(size,memory='device')) for size in (count*112,count*100,count*112)]
        kernel=own(VulkanKernel(runtime,compile_compute(compiled.source),{i:VulkanResource.buffer(b) for i,b in enumerate(inputs+outputs)}))
        restore_compiled=osh.compile(restore,helpers=(unpackDistanceVisibility,))
        restore_kernel=own(VulkanKernel(runtime,compile_compute(restore_compiled.source),
            {i:VulkanResource.buffer(b) for i,b in enumerate((outputs[1],inputs[1],inputs[2],outputs[2]))}))
        graph=VulkanGraph().add('pack',reflected_operation(kernel,compiled.reflection,workgroups=((count+63)//64,1,1)))
        graph.add('restore',reflected_operation(restore_kernel,restore_compiled.reflection,workgroups=((count+63)//64,1,1)),after=('pack',)).compile().execute(runtime).wait()
        reference=np.frombuffer(outputs[0].read(),np.uint32).reshape(count,28)
        packed=np.frombuffer(outputs[1].read(),np.uint32).reshape(count,25)
        restored=np.frombuffer(outputs[2].read(),np.uint32).reshape(count,28)
        np.testing.assert_array_equal(reference[:,4:],words[:,4:])
        np.testing.assert_array_equal(packed[:,0],words[:,3])
        np.testing.assert_array_equal(packed[:,1:],words[:,4:])
        np.testing.assert_array_equal(restored,reference)
