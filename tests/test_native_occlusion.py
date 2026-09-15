"""Boolean visibility preserves filtering while avoiding opaque hit payloads."""
import os
from contextlib import ExitStack
import numpy as np
import ordinaryshade as osh
import pytest
from test_native_triangles import box_hit, triangle_hit
from ordinarylight.shaders.native_intersection_programs import (
    NativeIntersection, NativeOpticalBoundary, nativeIntersectionMiss,
    nativeTraceSurface, nativeOccluded, nativeBoundaryEnabled,
)

@osh.function(name='nativeEvaluateBoundary')
def boundary(hit: NativeIntersection) -> NativeOpticalBoundary:
    return NativeOpticalBoundary(osh.vec2(1.0), hit.identity.w == osh.u32(71))

@osh.compute(workgroup_size=(1,1,1))
def compare(scene_tlas: osh.acceleration_structure(binding=0),
            results: osh.storage_buffer(osh.uvec4,binding=1)):
    i = osh.global_invocation_id.x
    origin = osh.vec3(0.25 if i % osh.u32(2) == osh.u32(0) else 0.75, 0.25, 0)
    direction = osh.vec3(0,0,1)
    limit = 2.0 if i < osh.u32(2) else 10.0
    full = nativeTraceSurface(origin, 0.001, direction, limit, True, osh.u32(255))
    blocked = nativeOccluded(origin, 0.001, direction, limit, osh.u32(255))
    results[i] = osh.uvec4(osh.u32(full.address.w != osh.u32(0)), osh.u32(blocked), 0, 0)

def compiled():
    return osh.compile(compare,helpers=(box_hit,triangle_hit,boundary,nativeIntersectionMiss,
        nativeBoundaryEnabled,nativeTraceSurface,nativeOccluded))

def test_boolean_visibility_is_typed_and_has_no_committed_payload_lookup():
    result=compiled()
    body=result.source.split('bool nativeOccluded(',2)[-1]
    assert 'return (rayQueryGetIntersectionTypeEXT(query, true) != uint(0));' in body

@pytest.mark.skipif(os.environ.get('ORDINARYLIGHT_TEST_VULKAN_GRAPH')!='1',reason='Opt-in Vulkan')
@pytest.mark.parametrize('opaque,neutral',((False,False),(True,False),(False,True)))
def test_visibility_matches_surface_query(opaque,neutral):
    import vulkan as vk
    from ordinarylight.runtime import (VulkanRuntime,VulkanTriangleBlas,VulkanAabbBlas,
        VulkanBlasInstance,VulkanTlas,VulkanKernel,compile_compute)
    from ordinarylight.pipeline.graph import VulkanGraph,reflected_operation
    from ordinarylight.pipeline.vulkan import VulkanResource
    with ExitStack() as stack:
        own=stack.enter_context;runtime=own(VulkanRuntime())
        usage=vk.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
        vertices=np.array([[(0,0,z),(2,0,z),(0,2,z)] for z in (1,5)],np.float32)
        buffer=own(runtime.buffer(vertices.nbytes,data=vertices,memory='device',device_address=True,usage=usage))
        triangles=own(VulkanTriangleBlas(runtime,buffer,2,opaque=opaque))
        bounds=np.array([0,0,3,1,1,4],np.float32)
        boxes=own(runtime.buffer(bounds.nbytes,data=bounds,memory='device',device_address=True,usage=usage))
        box=own(VulkanAabbBlas(runtime,boxes,1))
        packed=VulkanBlasInstance(triangles,custom_index=300 if opaque else 100).pack()+VulkanBlasInstance(box,custom_index=200).pack()
        instances=own(runtime.buffer(len(packed),data=packed,memory='device',device_address=True,usage=usage))
        tlas=own(VulkanTlas(runtime,instances,2,referenced_blas=(triangles,box)))
        output=own(runtime.buffer(64));program=compiled()
        definitions='#define WAVE_CUSTOM_GEOMETRY 1\n#define WAVE_CUSTOM_TRIANGLES 1\n'
        if neutral:definitions+='#define WAVE_NATIVE_OPTICAL_BOUNDARIES 1\n'
        source=program.source.replace('#version 460','#version 460\n'+definitions,1)
        kernel=own(VulkanKernel(runtime,compile_compute(source),{0:tlas.resource,1:VulkanResource.buffer(output)}))
        graph=VulkanGraph().add('triangles',triangles.operation()).add('box',box.operation()).add('tlas',tlas.operation())
        graph.add('visibility',reflected_operation(kernel,program.reflection,workgroups=(4,1,1))).compile().execute(runtime).wait()
        values=np.frombuffer(output.read(),np.uint32).reshape(4,4)
        np.testing.assert_array_equal(values[:,0],values[:,1])
        np.testing.assert_array_equal(values[:,1],[0,0,1,1] if neutral else [1,1,1,1] if opaque else [1,0,1,1])
