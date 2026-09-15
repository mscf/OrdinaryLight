"""Hardware triangle hits and typed payloads coexist with procedural geometry."""
import os
from contextlib import ExitStack
import numpy as np
import pytest
import ordinaryshade as osh
from ordinarylight.shaders.native_intersection_programs import (
    NativeIntersection, nativeIntersectionMiss, nativeTraceSurface,
    nativeBoundaryEnabled, nativeEvaluateBoundary,
)

@osh.function(name='nativeIntersectCandidate')
def box_hit(origin: osh.vec3, direction: osh.vec3, t_min: osh.f32, t_max: osh.f32,
            primitive: osh.u32, instance: osh.u32, offset: osh.u32) -> NativeIntersection:
    hit = nativeIntersectionMiss()
    distance = (3.0 - origin.z) / direction.z
    if distance < t_min or distance > t_max:
        return hit
    hit.position_distance.w = distance
    hit.geometric_normal = osh.vec4(0, 0, -1, 0)
    hit.shading_normal = hit.geometric_normal
    hit.identity = osh.uvec4(instance, offset, primitive, 42)
    return hit

@osh.function(name='nativeEvaluateTriangle')
def triangle_hit(origin: osh.vec3, direction: osh.vec3, distance: osh.f32,
                 barycentrics: osh.vec2, primitive: osh.u32, instance: osh.u32,
                 offset: osh.u32) -> NativeIntersection:
    hit = nativeIntersectionMiss()
    # Reject the near triangle in lane x > 0.5, so traversal must continue.
    if primitive == osh.u32(0) and origin.x > 0.5 and offset < osh.u32(300):
        return hit
    hit.position_distance.w = distance
    hit.geometric_normal = osh.vec4(0, 0, -1, 0)
    hit.shading_normal = hit.geometric_normal
    hit.identity = osh.uvec4(instance, offset + osh.u32(origin.x * 10.0), primitive, 71)
    hit.texcoord = osh.vec4(barycentrics, 0, 0)
    hit.previous_position = osh.vec4(origin + distance * direction, 1)
    return hit

@osh.compute(workgroup_size=(1,1,1))
def trace(scene_tlas: osh.acceleration_structure(binding=0),
          hits: osh.storage_buffer(NativeIntersection, binding=1)):
    i = osh.global_invocation_id.x
    x = 0.25 if i % osh.u32(2) == osh.u32(0) else 0.75
    hits[i] = nativeTraceSurface(osh.vec3(x, 0.25, 0), 0.001, osh.vec3(0,0,1),
                                10.0, i >= osh.u32(2), osh.u32(255))


def compiled_trace():
    return osh.compile(trace, helpers=(nativeIntersectionMiss, box_hit, triangle_hit,
                       nativeBoundaryEnabled, nativeTraceSurface), externals=(nativeEvaluateBoundary,))


def test_triangle_callback_and_confirmation_compile():
    result = compiled_trace()
    assert 'rayQueryConfirmIntersectionEXT' in result.source
    assert 'rayQueryGenerateIntersectionEXT' in result.source
    from ordinarylight.geometry import NativeGeometryProgram
    from ordinarylight.shaders.native_surface_programs import nativeEvaluateMaterial
    # Reject untyped/incorrect callback contracts before creating GPU resources.
    with pytest.raises(TypeError, match='typed'):
        NativeGeometryProgram(box_hit, nativeEvaluateMaterial, triangle=object())


@pytest.mark.skipif(os.environ.get('ORDINARYLIGHT_TEST_VULKAN_GRAPH') != '1', reason='opt-in GPU query test')
@pytest.mark.parametrize('opaque',(False,True))
def test_mixed_triangles_boxes_nearest_and_rejected_visibility(opaque):
    import vulkan as vk
    from ordinarylight.runtime import (VulkanRuntime, VulkanTriangleBlas, VulkanAabbBlas,
        VulkanTlas, VulkanBlasInstance, VulkanKernel, compile_compute)
    from ordinarylight.pipeline.graph import VulkanGraph, reflected_operation
    from ordinarylight.pipeline.vulkan import VulkanResource
    with ExitStack() as stack:
        own = stack.enter_context
        runtime = own(VulkanRuntime())
        usage = vk.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
        vertices = np.array([[(0,0,z),(2,0,z),(0,2,z)] for z in (1,5)], np.float32)
        buffer = own(runtime.buffer(vertices.nbytes, data=vertices, memory='device', device_address=True, usage=usage))
        with pytest.raises(ValueError, match='Invalid triangle'):
            VulkanTriangleBlas(runtime, buffer, 3)
        triangles = own(VulkanTriangleBlas(runtime, buffer, 2, opaque=opaque))
        boxes_data = np.array([0,0,3,1,1,4], np.float32)
        boxes = own(runtime.buffer(boxes_data.nbytes, data=boxes_data, memory='device', device_address=True, usage=usage))
        box = own(VulkanAabbBlas(runtime, boxes, 1))
        packed = VulkanBlasInstance(triangles, custom_index=300 if opaque else 100).pack() + VulkanBlasInstance(box, custom_index=200).pack()
        instances = own(runtime.buffer(len(packed), data=packed, memory='device', device_address=True, usage=usage))
        tlas = own(VulkanTlas(runtime, instances, 2, referenced_blas=(triangles, box)))
        output = own(runtime.buffer(4*112))
        compiled = compiled_trace()
        source = compiled.source.replace('#version 460', '#version 460\n#define WAVE_CUSTOM_GEOMETRY 1\n#define WAVE_CUSTOM_TRIANGLES 1', 1)
        kernel = own(VulkanKernel(runtime, compile_compute(source), {0:tlas.resource,1:VulkanResource.buffer(output)}))
        graph = VulkanGraph().add('triangles', triangles.operation()).add('boxes', box.operation()).add('tlas', tlas.operation())
        graph.add('trace', reflected_operation(kernel, compiled.reflection, workgroups=(4,1,1))).compile().execute(runtime).wait()
        values = np.frombuffer(output.read(), np.float32).reshape(4,7,4)
        ids = values.view(np.uint32)
        np.testing.assert_allclose(values[:2,0,3], [1,1] if opaque else [1,3])
        np.testing.assert_array_equal(ids[:2,3], [[0,302,0,71],[0,307,0,71]] if opaque else [[0,102,0,71],[1,200,0,42]])
        np.testing.assert_array_equal(ids[:2,4,3], [2,2])
        assert np.all(values[2:,0,3] > 0)
        # Any accepted occluder is legal; the rejected near triangle is not.
        if not opaque:
            assert not (ids[3,3,3] == 71 and ids[3,3,2] == 0)
        with pytest.raises(RuntimeError, match='borrowers'):
            buffer.close()
        with pytest.raises(RuntimeError, match='borrowers'):
            triangles.close()
