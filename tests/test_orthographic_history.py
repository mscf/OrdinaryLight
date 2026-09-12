"""Known camera points verify native denoiser and ReSTIR projection together."""
import os
import numpy as np
import ordinaryshade as osh
import pytest
from ordinarylight.denoising.kernels import PrepareCamera, prepare_previous_pixel
from ordinarylight.shaders.fused_primary_programs import reprojectRestir, restirPreviousWorldPosition


@osh.structure
class ProjectionConstants:
    image_tile: osh.uvec4


@osh.compute(workgroup_size=(1, 1, 1))
def project_history(previous_camera: osh.storage_record(PrepareCamera, access='read', binding=0),
                    points: osh.storage_buffer(osh.vec4, access='read', binding=1),
                    result: osh.storage_buffer(osh.vec4, access='write', binding=2),
                    push: osh.push_constants(ProjectionConstants)):
    i = osh.global_invocation_id.x
    projected = prepare_previous_pixel(points[i].xyz, osh.ivec2(push.image_tile.xy))
    pixel = osh.ivec2(0)
    valid = reprojectRestir(points[i].xyz, pixel)
    restored = restirPreviousWorldPosition(pixel, points[i].w)
    result[i * osh.u32(3)] = osh.vec4(projected, 1.0)
    result[i * osh.u32(3) + osh.u32(1)] = osh.vec4(osh.vec2(pixel), 1.0 if valid else 0.0, 0.0)
    result[i * osh.u32(3) + osh.u32(2)] = osh.vec4(restored, 1.0)


def test_history_projection_compiles_from_typed_sources():
    compiled = osh.compile(project_history, helpers=(prepare_previous_pixel, reprojectRestir, restirPreviousWorldPosition))
    assert 'projection_scale' in compiled.source


@pytest.mark.skipif(os.environ.get('ORDINARYLIGHT_TEST_VULKAN_GRAPH') != '1', reason='Opt-in projection GPU test')
def test_parallel_rays_reproject_independently_of_depth():
    from ordinarylight.runtime import VulkanRuntime, VulkanKernel, compile_compute
    from ordinarylight.pipeline.graph import reflected_operation
    from ordinarylight.pipeline.vulkan import VulkanResource
    import struct
    compiled = osh.compile(project_history, helpers=(prepare_previous_pixel, reprojectRestir, restirPreviousWorldPosition))
    camera = np.array([(0, 0, 4, 0), (0, 0, -1, 0), (2, 0, 0, 0), (0, 2, 0, 1)], '<f4')
    values = np.array([(1.05, .05, 0, 4), (1.05, .05, -4, 8)], '<f4')
    with VulkanRuntime() as runtime:
        with runtime.buffer(camera.nbytes, data=camera) as previous, runtime.buffer(values.nbytes, data=values) as points, runtime.buffer(2 * 3 * 16) as result:
            b = VulkanResource.buffer
            with VulkanKernel(runtime, compile_compute(compiled.source), {0: b(previous), 1: b(points), 2: b(result)}, push_constant_size=16) as kernel:
                for projection in (1, 0):
                    camera[3, 3] = projection
                    previous.upload(camera)
                    reflected_operation(kernel, compiled.reflection, workgroups=(2, 1, 1),
                                        push_constants=struct.pack('<4I', 80, 40, 0, 0)).execute(runtime).wait()
                    actual = np.frombuffer(result.read(), '<f4').reshape(2, 3, 4)
                    if projection == 1:
                        np.testing.assert_allclose(actual[:, 0, :2], [[50, 19], [50, 19]], atol=1e-5)
                        np.testing.assert_allclose(actual[:, 1, :3], [[50, 19, 1], [50, 19, 1]], atol=1e-5)
                        np.testing.assert_allclose(actual[:, 2, :3], values[:, :3], atol=1e-5)
                    else:
                        np.testing.assert_allclose(actual[:, 0, :2], [[42.125, 19.375], [40.8125, 19.4375]], atol=1e-5)
                    np.testing.assert_allclose(actual[:, 0, 2], [4, 8], atol=1e-5)
