"""Shared native query semantics with application-defined voxel slot/face IDs."""
import os
import numpy as np
import ordinaryshade as osh
from ordinarylight.shaders.native_intersection_programs import nativeBoundaryEnabled, nativeEvaluateBoundary
import pytest

from ordinarylight.shaders.native_intersection_programs import (
    NativeIntersection, nativeIntersectionMiss, nativeTraceSurface,
)


@osh.structure
class CellRecord:
    lower: osh.vec4
    upper: osh.vec4
    parameters: osh.vec4
    metadata: osh.uvec4


@osh.structure
class TestRay:
    origin_tmin: osh.vec4
    direction_tmax: osh.vec4
    metadata: osh.uvec4


@osh.function(name='nativeIntersectCandidate')
def intersect_cell(origin: osh.vec3, direction: osh.vec3, t_min: osh.f32,
                   t_max: osh.f32, primitive: osh.u32, instance: osh.u32,
                   instance_offset: osh.u32) -> NativeIntersection:
    hit = nativeIntersectionMiss()
    cell = cells[primitive]
    if cell.metadata.x == osh.u32(4294967295):
        return hit
    near_t = -1e30
    far_t = 1e30
    near_n = osh.vec3(0)
    far_n = osh.vec3(0)
    near_face = osh.u32(0)
    far_face = osh.u32(0)
    axis = 0
    while axis < 3:
        if osh.absolute(direction[axis]) < 1e-20:
            if origin[axis] < cell.lower[axis] or origin[axis] > cell.upper[axis]:
                return hit
        else:
            a = (cell.lower[axis] - origin[axis]) / direction[axis]
            b = (cell.upper[axis] - origin[axis]) / direction[axis]
            n = osh.vec3(0)
            n[axis] = -1.0 if direction[axis] > 0.0 else 1.0
            face = osh.u32(axis * 2) + (osh.u32(0) if direction[axis] > 0.0 else osh.u32(1))
            if osh.minimum(a, b) > near_t:
                near_t = osh.minimum(a, b)
                near_n = n
                near_face = face
            if osh.maximum(a, b) < far_t:
                far_t = osh.maximum(a, b)
                far_n = -n
                far_face = face ^ osh.u32(1)
        axis = axis + 1
    if near_t > far_t:
        return hit
    distance = near_t if near_t >= t_min else far_t
    if distance < t_min or distance > t_max:
        return hit
    normal = near_n if near_t >= t_min else far_n
    face = near_face if near_t >= t_min else far_face
    hit.position_distance.w = distance
    hit.geometric_normal = osh.vec4(normal, 0)
    hit.shading_normal = osh.vec4(normal, 0)
    hit.identity = osh.uvec4(instance, cell.metadata.w, face, osh.u32(71))
    position = origin + distance * direction
    previous_valid = 1.0
    if cell.parameters.w > 0.5 and position.x < 0.5:
        previous_valid = 0.0
    hit.previous_position = osh.vec4(position + cell.parameters.xyz, previous_valid)
    return hit


@osh.compute(workgroup_size=(1, 1, 1))
def trace_cells(scene_tlas: osh.acceleration_structure(binding=0),
                cells: osh.storage_buffer(CellRecord, access='read', binding=1),
                rays: osh.storage_buffer(TestRay, access='read', binding=2),
                hits: osh.storage_buffer(NativeIntersection, access='write', binding=3)):
    i = osh.global_invocation_id.x
    ray = rays[i]
    hits[i] = nativeTraceSurface(ray.origin_tmin.xyz, ray.origin_tmin.w,
                                ray.direction_tmax.xyz, ray.direction_tmax.w,
                                ray.metadata.x != osh.u32(0), osh.u32(2))


def shader():
    return osh.compile(trace_cells, helpers=(
        nativeIntersectionMiss, intersect_cell, nativeBoundaryEnabled, nativeTraceSurface,
    ), externals=(nativeEvaluateBoundary,))


def test_shared_query_and_callback_compile_from_typed_sources():
    source = shader().source
    assert 'NativeIntersection nativeIntersectCandidate' in source
    assert 'rayQueryGenerateIntersectionEXT' in source
    assert 'selected_instance' in source
    assert '#if WAVE_CUSTOM_GEOMETRY' in source


@pytest.mark.skipif(os.environ.get('ORDINARYLIGHT_TEST_VULKAN_GRAPH') != '1', reason='opt-in GPU query test')
def test_custom_nearest_hit_slot_face_and_visibility():
    from ordinarylight.geometry import CustomGeometry, IntersectionProgram
    from ordinarylight.scene import Scene
    from ordinarylight.transport import VulkanTransportScene, TransportMaterial
    from ordinarylight.runtime import VulkanRuntime, VulkanKernel, compile_compute
    from ordinarylight.pipeline.graph import reflected_operation
    from ordinarylight.pipeline.vulkan import VulkanResource

    # Use existing AS ownership/building only; the shader under test supplies its
    # own typed box intersection, independently of the scene's sphere callback.
    program = IntersectionProgram.sdf_sphere()
    boxes = [CustomGeometry(((0, 0, z), (1, 1, z + 1)), program,
                            (0.5, 0.5, z + 0.5, 0.5), identity=slot)
             for z, slot in ((4, 900), (1, 42), (7, 123))]
    ray_dtype = np.dtype([('origin', '<f4', (4,)), ('direction', '<f4', (4,)), ('metadata', '<u4', (4,))])
    rays = np.zeros(6, ray_dtype)
    rays['origin'] = [(0.5, 0.5, 0, .001), (0.5, 0.5, 10, .001),
                      (0.5, 0.5, 1.5, .001), (2, 2, 0, .001),
                      (0.5, 0.5, 0, .001), (0.5, 0.5, 0, .001)]
    rays['direction'] = [(0, 0, 1, 20), (0, 0, -1, 20), (0, 0, 1, 20),
                         (0, 0, 1, 20), (0, 0, 1, .5), (0, 0, 1, 20)]
    rays['metadata'][-1, 0] = 1
    compiled = shader()
    source = compiled.source.replace('#version 460', '#version 460\n#define WAVE_CUSTOM_GEOMETRY 1', 1)
    binary = compile_compute(source)
    with VulkanRuntime() as runtime:
        with VulkanTransportScene(runtime, custom_geometry=boxes,
                                  custom_materials=[TransportMaterial()]) as scene:
            with runtime.import_scene(Scene(), acceleration=scene.resource('tlas')) as imported:
                assert not imported.blases and not imported.instances
                assert not imported._structures, 'Import must not allocate acceleration structures'
                assert imported.primitive_ids.size == 0
                assert imported.tlas.handle == scene.resource('tlas').handle
                with pytest.raises(RuntimeError):
                    scene.close()
                with runtime.buffer(rays.nbytes, data=rays) as inputs, runtime.buffer(6 * 112) as outputs:
                    with VulkanKernel(runtime, binary, {0: imported.resource('tlas'), 1: scene.resource('custom'),
                                      2: VulkanResource.buffer(inputs), 3: VulkanResource.buffer(outputs)}) as kernel:
                        reflected_operation(kernel, compiled.reflection, workgroups=(6, 1, 1)).execute(runtime).wait()
                        floats = np.frombuffer(outputs.read(), '<f4').reshape(6, 7, 4)
                        ints = floats.view('<u4')
                        np.testing.assert_allclose(floats[:5, 0, 3], [1, 2, .5, -1, -1])
                        np.testing.assert_array_equal(ints[:3, 3, 1:3], [[42, 4], [123, 5], [42, 5]])
                        np.testing.assert_array_equal(ints[3:5, 3], np.uint32(0xffffffff))
                        assert floats[5, 0, 3] >= 0, 'Visibility rays must include procedural geometry'
                        assert ints[5, 3, 1] in (42, 900, 123), 'Any-hit visibility may terminate before nearest hit'


from ordinarylight.shaders.transport_programs import MaterialData
from ordinarylight.shaders.native_surface_programs import nativeEvaluateMaterial


@osh.function(name='nativeEvaluateMaterial')
def evaluate_cell_material(hit: NativeIntersection, cone_width: osh.f32) -> MaterialData:
    material = cell_materials[hit.address.y]
    # Face data is independent of the enclosing acceleration primitive.
    material.base_roughness.w = osh.f32(hit.identity.z) * 0.1 + cone_width
    return material


def test_native_geometry_program_compiles_typed_resources_and_callbacks():
    from ordinarylight.geometry.native import NativeGeometryBuffer, NativeGeometryProgram
    program = NativeGeometryProgram(intersect_cell, evaluate_cell_material, buffers=(
        NativeGeometryBuffer('cells', CellRecord),
        NativeGeometryBuffer('cell_materials', MaterialData),
    ))
    assert 'nativeIntersectCandidate(' in program.source
    assert 'nativeEvaluateMaterial(' in program.source
    assert 'struct NativeIntersection' not in program.source
    assert 'struct MaterialData' not in program.source
    assert 'void main()' not in program.source
    assert 'set = 2' in program.source
    with pytest.raises(TypeError, match='typed OrdinaryShade'):
        NativeGeometryProgram(nativeEvaluateMaterial, evaluate_cell_material)
    with pytest.raises(ValueError, match='unique'):
        NativeGeometryProgram(intersect_cell, evaluate_cell_material,
                              buffers=(NativeGeometryBuffer('cells', CellRecord),) * 2)


@osh.compute(workgroup_size=(8, 8, 1))
def read_native_history(
    history: osh.storage_image('r32f', access='read', binding=0),
    motion: osh.storage_image('rgba16f', access='read', binding=1),
    identity: osh.storage_image('r32ui', access='read', binding=2),
    values: osh.storage_buffer(osh.vec4, access='write', binding=3),
):
    pixel = osh.ivec2(osh.global_invocation_id.xy)
    size = history.size()
    if osh.any_value(pixel >= size):
        return
    index = pixel.y * size.x + pixel.x
    values[index * 2] = osh.vec4(history.load(pixel).x, motion.load(pixel).xyz)
    values[index * 2 + 1] = osh.uint_bits_to_float(identity.load(pixel))


@pytest.mark.skipif(os.environ.get('ORDINARYLIGHT_TEST_VULKAN_GRAPH') != '1', reason='opt-in native GI test')
def test_custom_surface_in_native_camera_gi():
    from dataclasses import replace
    from ordinarylight import PerspectiveCamera
    from ordinarylight.geometry import CustomGeometry, IntersectionProgram
    from ordinarylight.geometry.native import NativeGeometryBuffer, NativeGeometryProgram, VulkanNativeGeometryResources
    from ordinarylight.transport import VulkanTransportScene, TransportMaterial
    from ordinarylight.runtime import VulkanRuntime, VulkanWavefrontPipeline
    from ordinarylight.targets.vulkan.api import RendererConfig
    from ordinarylight.scene import Scene

    base = RendererConfig(max_bounces=4, samples_per_pixel=1, wavefront_hdr_capture=True,
                          wavefront_primary_hits=True, wavefront_restir_di=False,
                          denoiser_enabled=False, temporal_history=False,
                          wavefront_custom_inline=False, wavefront_ordinaryshade_shade=True)
    program = NativeGeometryProgram(intersect_cell, evaluate_cell_material, buffers=(
        NativeGeometryBuffer('cells', CellRecord), NativeGeometryBuffer('cell_materials', MaterialData),
    ))
    cell_data = np.zeros((2, 4, 4), '<f4')
    cell_data[:, 0, :3] = [(-10000, -10000, 2), (-10000, -10000, -3)]
    cell_data[:, 1, :3] = [(10000, 10000, 3), (10000, 10000, -2)]
    cell_data.view('<u4')[:, 3] = [(0, 0, 0xffffffff, 42), (0, 1, 0xffffffff, 900)]
    material_data = np.zeros((2, 12, 4), '<f4')
    material_data[:, 0] = (.8, .8, .8, .5)
    material_data[1, 1] = (2, 1, .5, 0)
    material_data[:, 2] = (1, 1, 1, 0)
    material_data[:, 3] = (1.5, 1, 0, 1)
    material_data[:, 4] = -1
    material_data[:, 5] = (1, 0, 0, 1)
    shapes = [CustomGeometry((tuple(cell_data[i, 0, :3]), tuple(cell_data[i, 1, :3])),
                             IntersectionProgram.sdf_sphere(), (0, 0, 0, 1), identity=slot)
              for i, slot in enumerate((42, 900))]
    with VulkanRuntime(config=base) as runtime:
        with runtime.buffer(cell_data.nbytes, data=cell_data) as cells, runtime.buffer(material_data.nbytes, data=material_data) as materials:
            with VulkanNativeGeometryResources(runtime, program, {'cells': cells, 'cell_materials': materials}) as geometry:
                config = replace(base, geometry_resources=geometry)
                with VulkanTransportScene(runtime, custom_geometry=shapes, custom_materials=[TransportMaterial()]) as owner:
                    with runtime.import_scene(Scene(), acceleration=owner.resource('tlas'), config=config) as resident:
                        from ordinarylight.pipeline.graph import VulkanGraph
                        from ordinarylight.pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse
                        from ordinarylight.wavefront import PRIMARY_HIT_DTYPE
                        import vulkan as vk
                        images = []
                        for bounces in (1, 4):
                            with VulkanWavefrontPipeline(runtime, resident, config=replace(config, max_bounces=bounces)) as pipeline:
                                camera = PerspectiveCamera(position=(.5, .5, 0), target=(.5, .5, 1.5))
                                frame = pipeline.prepare(camera, (32, 32))
                                with pytest.raises(RuntimeError, match='Submit or cancel'):
                                    geometry.notify_content_changed()
                                with pytest.raises(RuntimeError, match='Submit or cancel'):
                                    geometry.replace_buffers({})
                                hits = frame.buffers['primary_hits']
                                with runtime.buffer(hits.byte_size) as readback:
                                    graph = VulkanGraph().add('gi', frame.operation)
                                    graph.add('primary diagnostics', VulkanPass('copy', (
                                        VulkanResourceUse(VulkanResource.buffer(hits), vk.VK_PIPELINE_STAGE_TRANSFER_BIT, vk.VK_ACCESS_TRANSFER_READ_BIT),
                                        VulkanResourceUse(VulkanResource.buffer(readback), vk.VK_PIPELINE_STAGE_TRANSFER_BIT, vk.VK_ACCESS_TRANSFER_WRITE_BIT),
                                    ), lambda command: vk.vkCmdCopyBuffer(command, hits.buffer, readback.buffer, 1, [vk.VkBufferCopy(size=hits.byte_size)])))
                                    graph.compile().execute(runtime).wait()
                                    data = np.frombuffer(readback.read(), PRIMARY_HIT_DTYPE)
                                    np.testing.assert_array_equal(data['identity'][:, 1], 42)
                                    np.testing.assert_array_equal(data['identity'][:, 2], 4)
                                    np.testing.assert_allclose(data['position_distance'][:, 2], 2, atol=1e-5)
                                    assert np.all(data['position_distance'][:, 3] > 0)
                                hdr = pipeline.capture_wavefront_hdr()
                                assert np.isfinite(hdr).all()
                                images.append(hdr)
                                if bounces == 4:
                                    keys = [f.get('wavefront_command_key') for f in pipeline._core.window_frames]
                                    dark = material_data.copy()
                                    dark[:, 1, :3] = 0
                                    materials.upload(dark)
                                    geometry.notify_content_changed(invalidate_history=False)
                                    assert keys == [f.get('wavefront_command_key') for f in pipeline._core.window_frames]
                                    updated = pipeline.render(camera, (32, 32))
                                    updated.completion.wait()
                                    assert pipeline.capture_wavefront_hdr()[..., :3].mean() < hdr[..., :3].mean() * .1
                                    alignment = int(vk.vkGetPhysicalDeviceProperties(runtime.physical_device).limits.minStorageBufferOffsetAlignment)
                                    offset = max(256, alignment)
                                    padded = bytes(offset) + material_data.tobytes()
                                    with runtime.buffer(len(padded), data=padded) as replacement:
                                        view = VulkanResource.buffer(replacement).byte_range(offset, material_data.nbytes)
                                        geometry.replace_buffers({'cell_materials': view})
                                        assert all(f.get('wavefront_command_key') is None for f in pipeline._core.window_frames)
                                        restored = pipeline.render(camera, (32, 32))
                                        restored.completion.wait()
                                        assert pipeline.capture_wavefront_hdr()[..., 0].mean() > .5
                                        with pytest.raises(RuntimeError, match='borrowers'):
                                            replacement.close()
                                        geometry.replace_buffers({'cell_materials': materials})
                                    assert geometry.binding_revision == 2
                        np.testing.assert_allclose(images[0][..., :3], 0, atol=1e-5)
                        assert np.mean(images[1][..., 0]) > .5, 'Secondary custom surfaces must contribute lighting'
                        assert np.mean(images[1][..., 1]) > .25
                        assert np.mean(images[1][..., 2]) > .125
                        materials.upload(material_data)
                        history_config = replace(config, denoiser_enabled=True, temporal_history=True, progressive_accumulation=True,
                                                 denoiser_sampled_indirect=True)
                        from ordinarylight.runtime import VulkanKernel, compile_compute
                        from ordinarylight.pipeline.graph import reflected_operation
                        compiled = osh.compile(read_native_history)
                        binary = compile_compute(compiled.source)
                        with VulkanWavefrontPipeline(runtime, resident, config=history_config) as pipeline:
                            observed = []
                            for tick in range(5):
                                if tick == 4:
                                    cell_data[0, 2, 3] = 1
                                    cells.upload(cell_data)
                                    geometry.notify_content_changed(invalidate_history=False)
                                frame = pipeline.prepare(camera, (32, 32))
                                with runtime.buffer(32 * 32 * 32) as output:
                                    bindings = {i: VulkanResource.image(frame.images[name]) for i, name in enumerate(('diffuse_history', 'motion', 'identity'))}
                                    bindings[3] = VulkanResource.buffer(output)
                                    with VulkanKernel(runtime, binary, bindings) as reader:
                                        graph = VulkanGraph().add('gi', frame.operation)
                                        graph.add('history diagnostics', reflected_operation(reader, compiled.reflection, workgroups=(4, 4, 1)))
                                        graph.compile().execute(runtime).wait()
                                        observed.append(np.frombuffer(output.read(), '<f4').reshape(32, 32, 2, 4).copy())
                                assert np.isfinite(pipeline.capture_wavefront_hdr()).all()
                            assert observed[3][2:-2, 2:-2, 0, 0].mean() > 2
                            world_x = data['position_distance'][:, 0].reshape(32, 32)
                            interior = np.zeros((32, 32), bool)
                            interior[2:-2, 2:-2] = True
                            rejected = (world_x < .3) & interior
                            retained = (world_x > .7) & interior
                            assert np.any(rejected) and np.any(retained)
                            np.testing.assert_allclose(observed[4][..., 0, 0][rejected], 1, atol=1e-3)
                            assert observed[4][..., 0, 0][retained].mean() > 2
                            np.testing.assert_allclose(observed[4][..., 0, 3][rejected], 0, atol=1e-3)
                            assert observed[4][..., 0, 3][retained].mean() > 1


from ordinarylight.geometry import NativeOpticalBoundary


@osh.function(name='nativeEvaluateBoundary')
def evaluate_cell_boundary(hit: NativeIntersection) -> NativeOpticalBoundary:
    cell = cells[hit.address.y]
    return NativeOpticalBoundary(cell.parameters.xy, hit.identity.y == osh.u32(42))


def test_native_boundary_callback_is_typed_and_optional():
    from ordinarylight.geometry import NativeGeometryProgram, NativeGeometryBuffer
    buffers = (NativeGeometryBuffer('cells', CellRecord),
               NativeGeometryBuffer('cell_materials', MaterialData))
    program = NativeGeometryProgram(intersect_cell, evaluate_cell_material,
                                   buffers=buffers, boundary=evaluate_cell_boundary)
    assert program.boundary is evaluate_cell_boundary
    assert 'nativeEvaluateBoundary' in program.source
    assert 'struct NativeOpticalBoundary' not in program.source
    with pytest.raises(TypeError, match='typed OrdinaryShade'):
        NativeGeometryProgram(intersect_cell, evaluate_cell_material,
                              buffers=buffers, boundary=nativeEvaluateBoundary)


@pytest.mark.skipif(os.environ.get('ORDINARYLIGHT_TEST_VULKAN_GRAPH') != '1', reason='opt-in native optical GI test')
def test_native_boundary_primary_secondary_and_visibility():
    from dataclasses import replace
    from ordinarylight import PerspectiveCamera
    from ordinarylight.geometry import (CustomGeometry, IntersectionProgram,
        NativeGeometryBuffer, NativeGeometryProgram, VulkanNativeGeometryResources)
    from ordinarylight.transport import VulkanTransportScene, TransportMaterial
    from ordinarylight.runtime import VulkanRuntime, VulkanWavefrontPipeline, VulkanKernel, compile_compute
    from ordinarylight.pipeline.graph import VulkanGraph, reflected_operation
    from ordinarylight.pipeline.vulkan import VulkanResource
    from ordinarylight.targets.vulkan.api import RendererConfig
    from ordinarylight.scene import Scene

    base = RendererConfig(max_bounces=3, samples_per_pixel=1, wavefront_hdr_capture=True,
                          wavefront_primary_hits=True, wavefront_restir_di=False,
                          denoiser_enabled=False, temporal_history=False,
                          wavefront_custom_inline=False, wavefront_ordinaryshade_shade=True)
    program = NativeGeometryProgram(intersect_cell, evaluate_cell_material,
        boundary=evaluate_cell_boundary, buffers=(NativeGeometryBuffer('cells', CellRecord),
        NativeGeometryBuffer('cell_materials', MaterialData)))
    cell_data = np.zeros((3, 4, 4), '<f4')
    cell_data[:, 0, :3] = [(-1000, -1000, 1), (-1000, -1000, 4), (-1000, -1000, -5)]
    cell_data[:, 1, :3] = [(1000, 1000, 2), (1000, 1000, 5), (1000, 1000, -4)]
    cell_data[0, 2, :2] = (1, 1)
    cell_data.view('<u4')[:, 3] = [(0, 0, 0xffffffff, 42), (0, 1, 0xffffffff, 900), (0, 2, 0xffffffff, 901)]
    material_data = np.zeros((3, 12, 4), '<f4')
    material_data[1:, 1] = (2, 1, .5, 1)
    material_data[:, 2] = (1, 1, 1, 0)
    material_data[:, 3] = (1.5, 1, 0, 1)
    material_data[:, 4] = -1
    material_data[:, 5] = (1, 0, 0, 1)
    shapes = [CustomGeometry((tuple(c[0, :3]), tuple(c[1, :3])),
        IntersectionProgram.sdf_sphere(), (0, 0, 0, 1), identity=slot)
        for c, slot in zip(cell_data, (42, 900, 901))]
    camera = PerspectiveCamera(position=(0, 0, 0), target=(0, 0, 2))
    with VulkanRuntime(config=base) as runtime:
        with runtime.buffer(cell_data.nbytes, data=cell_data) as cells, runtime.buffer(material_data.nbytes, data=material_data) as materials:
            with VulkanNativeGeometryResources(runtime, program, {'cells': cells, 'cell_materials': materials}) as geometry:
                config = replace(base, geometry_resources=geometry)
                with VulkanTransportScene(runtime, custom_geometry=shapes, custom_materials=[TransportMaterial()]) as owner:
                    with runtime.import_scene(Scene(), acceleration=owner.resource('tlas'), config=config) as resident:
                        with VulkanWavefrontPipeline(runtime, resident, config=config) as pipeline:
                            results = []
                            for interior_ior in (1.0, 1.5):
                                cell_data[0, 2, 1] = interior_ior
                                cells.upload(cell_data)
                                geometry.notify_content_changed()
                                frame = pipeline.prepare(camera, (32, 32))
                                VulkanGraph().add('gi', frame.operation).compile().execute(runtime).wait()
                                results.append(pipeline.capture_wavefront_hdr().copy())
                            # Entry and exit are handled by different native stages. The
                            # eta-squared transmission factors cancel across this slab.
                            np.testing.assert_allclose(results[0][..., :3], np.broadcast_to((2, 1, .5), (32, 32, 3)), atol=2e-3)
                            assert .75 < results[1][..., 0].mean() / 2 < 1.01
                            assert np.isfinite(results[1]).all()
                            # With two bounces, only Fresnel-reflected primary rays can
                            # reach the rear emitter. A delta event must not receive
                            # a zero MIS weight against continuous light sampling.
                            pipeline.reconfigure(max_bounces=2)
                            frame = pipeline.prepare(camera, (32, 32))
                            VulkanGraph().add('reflected gi', frame.operation).compile().execute(runtime).wait()
                            reflected_result = pipeline.capture_wavefront_hdr()
                            assert .025 < reflected_result[..., 0].mean() < .2
                            # A camera already inside the solid must use inside-to-outside
                            # IOR at its primary hit, without assuming an air entry first.
                            inside = PerspectiveCamera(position=(0, 0, 1.5), target=(0, 0, 4))
                            pipeline.reconfigure(max_bounces=2)
                            frame = pipeline.prepare(inside, (32, 32))
                            VulkanGraph().add('inside gi', frame.operation).compile().execute(runtime).wait()
                            interior_result = pipeline.capture_wavefront_hdr()
                            assert 3.7 < interior_result[..., 0].mean() < 4.6
                            # Every camera ray exceeds the critical angle. Primary and
                            # secondary hits must keep reflecting inside the slab.
                            pipeline.reconfigure(max_bounces=4)
                            grazing = PerspectiveCamera(position=(0, 0, 1.5), target=(1, 0, 2),
                                                        vertical_fov_degrees=1)
                            frame = pipeline.prepare(grazing, (32, 32))
                            VulkanGraph().add('tir gi', frame.operation).compile().execute(runtime).wait()
                            np.testing.assert_allclose(pipeline.capture_wavefront_hdr()[..., :3], 0, atol=1e-6)


                        query = osh.compile(trace_cells, helpers=(nativeIntersectionMiss,
                            intersect_cell, nativeBoundaryEnabled, evaluate_cell_boundary, nativeTraceSurface))
                        source = query.source.replace('#version 460', '#version 460\n#define WAVE_CUSTOM_GEOMETRY 1\n#define WAVE_NATIVE_OPTICAL_BOUNDARIES 1', 1)
                        binary = compile_compute(source)
                        ray = np.zeros((1, 3, 4), '<f4')
                        ray[0, 0] = (0, 0, 0, .001)
                        ray[0, 1] = (0, 0, 1, 3)
                        ray.view('<u4')[0, 2, 0] = 1
                        with runtime.buffer(ray.nbytes, data=ray) as rays, runtime.buffer(112) as hits:
                            with VulkanKernel(runtime, binary, {0: resident.resource('tlas'), 1: VulkanResource.buffer(cells),
                                2: VulkanResource.buffer(rays), 3: VulkanResource.buffer(hits)}) as kernel:
                                for interior_ior, expected_kind in ((1.0, 0), (1.5, 2)):
                                    cell_data[0, 2, 1] = interior_ior
                                    cells.upload(cell_data)
                                    reflected_operation(kernel, query.reflection, workgroups=(1, 1, 1)).execute(runtime).wait()
                                    result = np.frombuffer(hits.read(), '<u4').reshape(7, 4)
                                    assert result[4, 3] == expected_kind


from ordinarylight.geometry import NativeEmitterSample, NativeEmitterProgram


@osh.function(name='nativeEmitterCount')
def cell_emitter_count() -> osh.u32:
    return osh.u32(1) if cells[1].parameters.w > 0.5 else osh.u32(0)


@osh.function(name='nativeSelectEmitter')
def select_cell_emitter(selector: osh.f32) -> osh.u32:
    return osh.u32(0)


@osh.function(name='nativeEvaluateEmitter')
def evaluate_cell_emitter(emitter: osh.u32, coordinates: osh.vec2) -> NativeEmitterSample:
    cell = cells[1]
    size = cell.upper.xy - cell.lower.xy
    front = cell.parameters.z >= 0.0
    position = osh.vec3(cell.lower.xy + size * coordinates, cell.upper.z if front else cell.lower.z)
    return NativeEmitterSample(position, osh.vec3(0, 0, 1 if front else -1), cell_materials[1].emission_metallic.rgb,
                               1.0 / (size.x * size.y), False)


@osh.function(name='nativeEmitterPdf')
def cell_emitter_pdf(hit: NativeIntersection) -> osh.f32:
    face = osh.u32(5) if cells[1].parameters.z >= 0.0 else osh.u32(4)
    if cells[1].parameters.w <= 0.5 or hit.identity.y != osh.u32(900) or hit.identity.z != face:
        return 0.0
    size = cells[1].upper.xy - cells[1].lower.xy
    return 1.0 / (size.x * size.y)


def cell_emitters():
    return NativeEmitterProgram(cell_emitter_count, select_cell_emitter,
                                evaluate_cell_emitter, cell_emitter_pdf)


def test_native_emitter_program_requires_matching_typed_callbacks():
    from ordinarylight.geometry import NativeGeometryProgram, NativeGeometryBuffer
    buffers = (NativeGeometryBuffer('cells', CellRecord), NativeGeometryBuffer('cell_materials', MaterialData))
    program = NativeGeometryProgram(intersect_cell, evaluate_cell_material,
                                   buffers=buffers, emitters=cell_emitters())
    assert 'nativeEvaluateEmitter' in program.source
    assert 'nativeEmitterPdf' in program.source
    assert 'struct NativeEmitterSample' not in program.source
    with pytest.raises(TypeError, match='NativeEmitterProgram'):
        NativeGeometryProgram(intersect_cell, evaluate_cell_material, emitters=object())
    with pytest.raises(ValueError, match='nativeSelectEmitter'):
        NativeGeometryProgram(intersect_cell, evaluate_cell_material, buffers=buffers,
            emitters=NativeEmitterProgram(cell_emitter_count, cell_emitter_count,
                                          evaluate_cell_emitter, cell_emitter_pdf))


@pytest.mark.skipif(os.environ.get('ORDINARYLIGHT_TEST_VULKAN_GRAPH') != '1', reason='opt-in native emitter GI test')
@pytest.mark.parametrize('secondary', [False, True])
def test_native_emitters_nee_restir_and_path_hit_energy(secondary):
    from dataclasses import replace
    from ordinarylight import PerspectiveCamera
    from ordinarylight.lights import EnvironmentLight
    from ordinarylight.geometry import (CustomGeometry, IntersectionProgram, NativeGeometryBuffer,
        NativeGeometryProgram, VulkanNativeGeometryResources)
    from ordinarylight.transport import VulkanTransportScene, TransportMaterial
    from ordinarylight.runtime import VulkanRuntime, VulkanWavefrontPipeline
    from ordinarylight.pipeline.graph import VulkanGraph
    from ordinarylight.targets.vulkan.api import RendererConfig
    from ordinarylight.scene import Scene

    base = RendererConfig(max_bounces=3 if secondary else 2, samples_per_pixel=64, wavefront_hdr_capture=True,
        wavefront_restir_di=False, denoiser_enabled=False, temporal_history=False,
        wavefront_custom_inline=False, wavefront_ordinaryshade_shade=True,
        wavefront_environment_samples=0, wavefront_profiling=True)
    cell_data = np.zeros((2, 4, 4), '<f4')
    cell_data[:, 0, :3] = [(-1000, -1000, 2), (-2, -2, -2)]
    cell_data[:, 1, :3] = [(1000, 1000, 3), (2, 2, -1)]
    cell_data[1, 2, 3] = 1
    cell_data.view('<u4')[:, 3] = [(0, 0, 0xffffffff, 42), (0, 1, 0xffffffff, 900)]
    material_data = np.zeros((2, 12, 4), '<f4')
    material_data[0, 0] = (.8, .8, .8, .5)
    material_data[1, 1] = (2, 1, .5, 1)
    material_data[:, 2] = (1, 1, 1, 0)
    material_data[:, 3] = (1.5, 1, 0, 1)
    material_data[:, 4] = -1
    material_data[:, 5] = (1, 0, 0, 1)
    slots = (42, 900)
    if secondary:
        # A nearly perfect dielectric reflector hides the diffuse receiver
        # from the camera. Only secondary NEE can illuminate that receiver.
        cell_data = np.concatenate((cell_data, np.zeros((1, 4, 4), '<f4')))
        cell_data[0, 0, :3] = (-1, -1, 1)
        cell_data[0, 1, :3] = (1, 1, 2)
        cell_data[0, 2, :2] = (1, 1e6)
        cell_data[1, 0, :3] = (2, -2, -1)
        cell_data[1, 1, :3] = (4, 2, 0)
        cell_data[1, 2, 2] = -1
        cell_data[2, 0, :3] = (-1000, -1000, -4)
        cell_data[2, 1, :3] = (1000, 1000, -3)
        cell_data.view('<u4')[2, 3] = (0, 2, 0xffffffff, 901)
        material_data = np.concatenate((material_data, material_data[0:1]))
        slots = (42, 900, 901)
    shapes = [CustomGeometry((tuple(c[0, :3]), tuple(c[1, :3])),
        IntersectionProgram.sdf_sphere(), (0, 0, 0, 1), identity=slot)
        for c, slot in zip(cell_data, slots)]
    scene = Scene(environment=EnvironmentLight(intensity=0))
    camera = PerspectiveCamera(position=(0, 0, 0), target=(0, 0, 2))
    means = []
    with VulkanRuntime(config=base) as runtime:
        with runtime.buffer(cell_data.nbytes, data=cell_data) as cells, runtime.buffer(material_data.nbytes, data=material_data) as materials:
            with VulkanTransportScene(runtime, custom_geometry=shapes, custom_materials=[TransportMaterial()]) as owner:
                for mode, area_samples, candidates in (('path', 1, 1), ('nee', 1, 1), ('nee', 4, 1), ('restir', 1, 4)):
                    program = NativeGeometryProgram(intersect_cell, evaluate_cell_material,
                        emitters=None if mode == 'path' else cell_emitters(),
                        boundary=evaluate_cell_boundary if secondary else None, buffers=(
                        NativeGeometryBuffer('cells', CellRecord), NativeGeometryBuffer('cell_materials', MaterialData)))
                    with VulkanNativeGeometryResources(runtime, program, {'cells': cells, 'cell_materials': materials}) as geometry:
                        config = replace(base, geometry_resources=geometry, samples_per_pixel=64 if mode == 'path' else 1 if mode == 'restir' else 16,
                            area_light_samples=area_samples, wavefront_restir_di=mode == 'restir',
                            wavefront_restir_candidates=candidates, wavefront_restir_shared_primary=False,
                            wavefront_restir_reservoirs=8 if mode == 'restir' else 1,
                            wavefront_restir_spatial_reuse=mode == 'restir',
                            wavefront_unified_primary_restir=mode == 'restir',
                            wavefront_environment_samples=1 if mode == 'restir' else 0,
                            temporal_history=mode == 'restir', progressive_accumulation=mode == 'restir')
                        with runtime.import_scene(scene, acceleration=owner.resource('tlas'), config=config) as resident:
                            with VulkanWavefrontPipeline(runtime, resident, config=config) as pipeline:
                                frame = pipeline.prepare(camera, (32, 32))
                                VulkanGraph().add('gi', frame.operation).compile().execute(runtime).wait()
                                hdr = pipeline.capture_wavefront_hdr().copy()
                                assert np.isfinite(hdr).all()
                                # Read-only test diagnostics prove the estimator ran;
                                # energy agreement alone could hide a fallback path.
                                counters = pipeline._core.wavefront_executor.read_work_counters(frame.slot)
                                if mode == 'path':
                                    assert counters['shadow_rays'] == 0
                                else:
                                    assert counters['shadow_rays'] > 0
                                    if secondary:
                                        assert counters['shadow_rays_bounce_1'] > 0

                                means.append(hdr[..., :3].mean(axis=(0, 1)))
                                if mode == 'restir':
                                    for reuse_frame in range(3):
                                        frame = pipeline.prepare(camera, (32, 32))
                                        VulkanGraph().add('reused gi', frame.operation).compile().execute(runtime).wait()
                                        means.append(pipeline.capture_wavefront_hdr()[..., :3].mean(axis=(0, 1)))
                                    if not secondary:
                                        counters = pipeline._core.wavefront_executor.read_work_counters(frame.slot)
                                        assert counters['restir_history_accepted'] > 0
                                    # The emitter distribution lives entirely on the GPU;
                                    # disabling it restores pure path sampling without
                                    # replacing descriptors or restarting the pipeline.
                                    cell_data[1, 2, 3] = 0
                                    cells.upload(cell_data)
                                    geometry.notify_content_changed()
                                    frame = pipeline.prepare(camera, (32, 32))
                                    VulkanGraph().add('updated gi', frame.operation).compile().execute(runtime).wait()
                                    means.append(pipeline.capture_wavefront_hdr()[..., :3].mean(axis=(0, 1)))
    assert means[0][0] > .03
    for estimate in means[1:]:
        np.testing.assert_allclose(estimate, means[0], rtol=.08, atol=.01)


def test_native_buffer_views_require_leased_storage_owners():
    from types import SimpleNamespace
    from ordinarylight.geometry.native import _native_buffer_resource
    from ordinarylight.pipeline.vulkan import VulkanResource
    runtime = object()
    owner = SimpleNamespace(runtime=runtime, require_open=lambda: None,
                            retain=lambda consumer: None, release=lambda consumer: None)
    resource = VulkanResource(owner, 'buffer', object(), 512).byte_range(256, 128)
    assert _native_buffer_resource(runtime, resource) is resource
    with pytest.raises(ValueError, match='same-runtime'):
        _native_buffer_resource(object(), resource)
    with pytest.raises(ValueError, match='storage-buffer'):
        _native_buffer_resource(runtime, VulkanResource(owner, 'image', object()))
    with pytest.raises(ValueError, match='storage-buffer'):
        _native_buffer_resource(runtime, VulkanResource(owner, 'buffer', object(), 16, 'uniform_buffer'))
    owner.release = None
    with pytest.raises(ValueError, match='retain and release'):
        _native_buffer_resource(runtime, resource)


@pytest.mark.skipif(os.environ.get('ORDINARYLIGHT_TEST_VULKAN_GRAPH') != '1', reason='opt-in resource lease test')
def test_native_buffers_lease_public_transport_scene_owner():
    from ordinarylight.geometry.native import NativeGeometryBuffer, NativeGeometryProgram, VulkanNativeGeometryResources
    from ordinarylight.geometry import CustomGeometry, IntersectionProgram
    from ordinarylight.transport import VulkanTransportScene, TransportMaterial
    from ordinarylight.runtime import VulkanRuntime
    program = NativeGeometryProgram(intersect_cell, evaluate_cell_material, buffers=(
        NativeGeometryBuffer('cells', CellRecord), NativeGeometryBuffer('cell_materials', MaterialData),
    ))
    with VulkanRuntime() as runtime:
        shape = CustomGeometry(((-1, -1, -1), (1, 1, 1)), IntersectionProgram.sdf_sphere(), (0, 0, 0, 1))
        with VulkanTransportScene(runtime, custom_geometry=[shape], custom_materials=[TransportMaterial()]) as scene:
            # Lease the public scene owner, independently of any importer/kernel.
            palette = scene.resource('materials')
            with VulkanNativeGeometryResources(runtime, program, {'cells': palette, 'cell_materials': palette}) as geometry:
                with pytest.raises(RuntimeError):
                    scene.close()
                with runtime.buffer(192) as replacement:
                    geometry.replace_buffers({'cell_materials': replacement})
                    with pytest.raises(RuntimeError):
                        scene.close()  # The other alias must still retain the scene.
                    geometry.replace_buffers({'cells': replacement})
                    scene.close()
                    with pytest.raises(RuntimeError):
                        replacement.close()
                    geometry.close()
