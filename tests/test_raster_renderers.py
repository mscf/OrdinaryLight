import sys
import shutil
import importlib.util
import os
import unittest
from pathlib import Path

import numpy as np

shade_root = Path(__file__).parents[2] / "ordinaryshade"
if shade_root.exists():
    sys.path.insert(0, str(shade_root))

import ordinarylight as ol
from ordinarylight.renderers.raster._diagnostics import frame_difference
from ordinarylight.materials.gpu import (
    SurfaceContext, SurfaceParameters, blend_surface_parameters,
)

try:
    import ordinaryshade as osh
except ImportError as error:
    raise unittest.SkipTest("raster compiler tests require sibling ordinaryshade") from error


@osh.vertex
def raster_vertex(position: osh.location(osh.vec2, 0)) -> osh.builtin(osh.vec4, "position"):
    return osh.vec4(position, 0.0, 1.0)


@osh.fragment
def raster_fragment() -> osh.location(osh.vec4, 0):
    return osh.vec4(0.95, 0.45, 0.15, 1.0)


@osh.function
def vertex_offset(value: osh.f32) -> osh.f32:
    return value * 0.25


@osh.vertex
def helper_raster_vertex(
    position: osh.location(osh.vec2, 0),
) -> osh.builtin(osh.vec4, "position"):
    return osh.vec4(position.x, position.y + vertex_offset(position.x), 0.0, 1.0)


@ol.raster_material_hook
def striped_material_hook(
    surface: SurfaceParameters, context: SurfaceContext,
) -> SurfaceParameters:
    stripe = osh.maximum(0.0, osh.minimum(1.0, context.uv.x * 2.0))
    layer = SurfaceParameters(
        osh.vec3(0.1, 0.8, 1.0), surface.emission, surface.normal,
        0.0, 0.18, surface.transmission, surface.occlusion,
        0.6, 0.1, osh.vec3(0.05), 0.3, 0.0, surface.thin_walled,
        0.2, osh.vec3(1.0, 0.4, 0.2), 0.35,
    )
    return blend_surface_parameters(surface, layer, stripe)


class RasterBackendTests(unittest.TestCase):
    @staticmethod
    def _overlap_scene():
        vertices = [[-0.8, -0.8, 0], [0.8, -0.8, 0], [0, 0.8, 0]]
        near = ol.Mesh(vertices, [[0, 1, 2]], ol.Material(base_color=(1, 0, 0)),
                       transform=ol.Transform.translation((0, 0, 1)))
        far = ol.Mesh(vertices, [[0, 1, 2]], ol.Material(base_color=(0, 0, 1)))
        return ol.Scene([near, far]), ol.PerspectiveCamera((0, 0, 4), (0, 0, 0))
    def test_raster_program_compiles_for_both_implementations(self):
        target = "spirv" if shutil.which("glslangValidator") else "glsl"
        vulkan = ol.RasterProgram.compile(raster_vertex, raster_fragment, target=target)
        webgpu = ol.RasterProgram.compile(raster_vertex, raster_fragment, target="wgsl", validate=False)
        self.assertTrue(vulkan.vertex.binary if target == "spirv" else vulkan.vertex.source)
        self.assertTrue(webgpu.vertex.source.startswith("@vertex"))
        self.assertEqual(vulkan.reflection.vertex.stage, "vertex")

    def test_raster_program_accepts_vertex_stage_helpers(self):
        program = ol.RasterProgram.compile(
            helper_raster_vertex, raster_fragment, target="wgsl",
            validate=False, vertex_helpers=(vertex_offset,),
        )
        self.assertIn("fn vertex_offset", program.vertex.source)

    def test_fragment_helper_spellings_are_mutually_exclusive(self):
        with self.assertRaisesRegex(TypeError, "helpers or fragment_helpers"):
            ol.RasterProgram.compile(
                raster_vertex, raster_fragment, target="wgsl", validate=False,
                helpers=(vertex_offset,), fragment_helpers=(vertex_offset,),
            )

    def test_vulkan_renderer_exposes_direct_surface_presentation(self):
        renderer = ol.renderers.raster.VulkanRasterRenderer
        self.assertTrue(callable(renderer.present_frame))
        self.assertIsInstance(renderer.direct_presentation, property)

    def test_frame_difference_detects_and_quantifies_changed_pixels(self):
        reference = np.zeros((2, 3, 4), np.float32)
        current = reference.copy()
        current[0, 1, 2] = 4.0
        current[1, 2, 0] = -2.0

        difference = frame_difference(reference, current)

        self.assertEqual(difference["maximum_absolute_difference"], 4.0)
        self.assertEqual(difference["changed_pixels"], 2)
        self.assertEqual(
            difference["changed_bounds"], {"x": [1, 2], "y": [0, 1]},
        )
        self.assertAlmostEqual(
            difference["rmse"], np.sqrt((16.0 + 4.0) / 24.0),
        )
        self.assertEqual(
            frame_difference(reference, reference),
            {
                "maximum_absolute_difference": 0.0,
                "rmse": 0.0,
                "changed_pixels": 0,
                "changed_bounds": None,
            },
        )

    def test_present_cache_key_tracks_draw_shape_not_camera_payload(self):
        renderer = ol.renderers.raster.VulkanRasterRenderer
        first = ol.triangle_mesh()
        second = ol.RasterMesh(
            first.vertices + 0.5, first.indices, first.layout,
            resources=first.resources,
        )
        key = renderer._present_cache_key(first, 1, 1920, 1080)
        self.assertEqual(
            key, renderer._present_cache_key(second, 1, 1920, 1080),
        )
        self.assertNotEqual(
            key, renderer._present_cache_key(first, 2, 1920, 1080),
        )

    def test_present_cache_retires_previous_camera_order_generation(self):
        renderer_type = ol.renderers.raster.VulkanRasterRenderer
        renderer = object.__new__(renderer_type)
        renderer._present_cache_generation = None
        renderer.device = object()
        events = []

        class FakeVulkan:
            @staticmethod
            def vkDeviceWaitIdle(device):
                events.append(("idle", device))

        renderer.vk = FakeVulkan()
        renderer._clear_present_cache = lambda: events.append(("clear",))

        self.assertTrue(renderer._activate_present_cache_generation((1, 2)))
        self.assertFalse(renderer._activate_present_cache_generation((1, 2)))
        self.assertEqual(events, [])
        self.assertTrue(renderer._activate_present_cache_generation((2, 1)))
        self.assertEqual(events, [("idle", renderer.device), ("clear",)])

    def test_present_completion_semaphore_is_owned_by_swapchain_image(self):
        renderer = object.__new__(
            ol.renderers.raster.VulkanRasterRenderer
        )
        renderer._present_render_finished = [object(), object(), object()]

        self.assertIs(
            renderer._render_finished_for_image(0),
            renderer._present_render_finished[0],
        )
        self.assertIs(
            renderer._render_finished_for_image(2),
            renderer._present_render_finished[2],
        )
        with self.assertRaisesRegex(RuntimeError, "completion semaphore"):
            renderer._render_finished_for_image(3)

    def test_opaque_prepass_disables_screen_space_sampling(self):
        camera = np.zeros(1, ol.raster.CAMERA_DTYPE)
        camera["viewport_optics"][0] = (1920, 1080, -1, 32)
        payload = ol.renderers.raster.VulkanRasterRenderer._opaque_camera_payload(
            camera.tobytes(),
        )
        decoded = np.frombuffer(payload, dtype=ol.raster.CAMERA_DTYPE)
        self.assertEqual(decoded["viewport_optics"][0, 2], -2.0)
        self.assertEqual(decoded["viewport_optics"][0, 3], 32.0)

    def test_opaque_prepass_preserves_light_and_shadow_counts(self):
        camera = np.zeros(1, ol.CAMERA_DTYPE)
        camera["viewport_optics"][0, 2] = -1.0
        camera["optical_diagnostic"][0] = (7.0, 3.0, 5.0, 1.0)
        for implementation in (
            ol.renderers.raster.VulkanRasterRenderer,
            ol.renderers.raster.WebGpuRasterRenderer,
        ):
            payload = implementation._opaque_camera_payload(camera.tobytes())
            decoded = np.frombuffer(payload, ol.CAMERA_DTYPE)
            np.testing.assert_array_equal(
                decoded["optical_diagnostic"][0], (0.0, 3.0, 5.0, 1.0),
            )

    def test_builtin_scene_program_compiles_for_both_implementations(self):
        target = "spirv" if shutil.which("glslangValidator") else "glsl"
        native = ol.RasterProgram.scene(target=target)
        web = ol.RasterProgram.scene(target="wgsl", validate=False)
        self.assertEqual(len(native.reflection.varyings), 24)
        self.assertIn("@location(1)", web.vertex.source)

    def test_builtin_sheen_is_a_grazing_angle_lobe(self):
        web = ol.RasterProgram.scene(target="wgsl", validate=False)
        fragment = web.fragment.source
        self.assertIn("pow((1.0 - vdoth), 5.0)", fragment)
        self.assertIn(
            "let sheen: vec3<f32> = (surface_sheen_color * sheen_weight)",
            fragment,
        )
        self.assertNotIn(
            "let surface_sheen: vec3<f32> = (hooked.sheen_color *",
            fragment,
        )

    def test_material_shader_variants_are_cached_by_program_set(self):
        from ordinarylight.showcases.materials import mirror
        first = ol.RasterProgram.scene(
            target="wgsl", validate=False,
            material_programs=(ol.builtin_material,),
        )
        repeated = ol.RasterProgram.scene(
            target="wgsl", validate=False,
            material_programs=(ol.builtin_material,),
        )
        mirror_variant = ol.RasterProgram.scene(
            target="wgsl", validate=False,
            material_programs=(mirror,),
        )
        self.assertIs(first, repeated)
        self.assertNotEqual(first.cache_key, mirror_variant.cache_key)
        self.assertIn("storage_buffer", [
            item["kind"] if isinstance(item, dict) else item.kind
            for item in first.fragment.reflection.resources
        ])

    def test_portable_raster_material_hook_compiles_and_is_cached(self):
        first = ol.RasterProgram.scene(
            target="wgsl", validate=False,
            material_hook=striped_material_hook,
        )
        repeated = ol.RasterProgram.scene(
            target="wgsl", validate=False,
            material_hook=striped_material_hook,
        )
        plain = ol.RasterProgram.scene(target="wgsl", validate=False)
        self.assertIs(first, repeated)
        self.assertNotEqual(first.cache_key, plain.cache_key)
        self.assertIn("ordinarylight_material_modifier", first.fragment.source)
        self.assertIn("blend_surface_parameters", first.fragment.source)
        self.assertIn("context.uv.x", first.fragment.source)

    def test_raster_config_rejects_undecorated_material_hook(self):
        with self.assertRaises(TypeError):
            ol.RasterConfig(material_hook=lambda surface, context: surface)

    def test_material_id_decoding_rounds_interpolated_float_varying(self):
        program = ol.RasterProgram.scene(target="wgsl", validate=False)
        self.assertIn("u32((material_index + 0.5))", program.fragment.source)

    def test_raster_mesh_owns_contiguous_typed_arrays(self):
        mesh = ol.RasterMesh([[0, 0], [1, 0], [0, 1]], [0, 1, 2])
        self.assertEqual(mesh.vertices.dtype, np.float32)
        self.assertEqual(mesh.indices.dtype, np.uint32)
        self.assertTrue(mesh.vertices.flags.c_contiguous)

    def test_general_interleaved_vertex_layout(self):
        layout = ol.RasterVertexLayout(28, (
            ol.RasterVertexAttribute(0, "float32x4", 0, "position"),
            ol.RasterVertexAttribute(1, "float32x3", 16, "color"),
        ))
        mesh = ol.RasterMesh(np.zeros((3, 7), np.float32), [0, 1, 2], layout)
        self.assertEqual(mesh.layout.stride, 28)
        self.assertEqual(mesh.layout.attributes[1].location, 1)

    def test_scene_adapter_applies_camera_and_object_transform(self):
        source = ol.Mesh(
            [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0, 0.5, 0]],
            [[0, 1, 2]], ol.Material(base_color=(0.2, 0.4, 0.8)),
            transform=ol.Transform.translation((1, 0, 0)),
        )
        scene = ol.Scene([source])
        camera = ol.PerspectiveCamera((0, 0, 4), (0, 0, 0))
        mesh = ol.scene_mesh(
            scene, camera, 100, 100,
            ol.RasterConfig(direct_lighting=False),
        )
        self.assertEqual(mesh.vertices.shape, (3, 64))
        np.testing.assert_allclose(
            mesh.vertices[:, 4:7], np.tile((0.2, 0.4, 0.8), (3, 1)),
        )
        self.assertGreater(mesh.vertices[:, 0].mean(), 0.0)

    def test_raster_state_validation(self):
        state = ol.RasterState(cull_mode="none", depth_compare="less-equal")
        self.assertTrue(state.depth_test)
        with self.assertRaises(ValueError):
            ol.RasterState(blend_mode="multiply")

    def test_parameter_grid_exposes_a_generic_vec4_domain(self):
        mesh = ol.parameter_grid(
            3, 2, u_range=(-2.0, 2.0), v_range=(1.0, 3.0),
            parameters=(0.75,),
        )
        self.assertEqual(mesh.vertices.shape, (6, 4))
        self.assertEqual(mesh.indices.shape, (12,))
        np.testing.assert_allclose(mesh.vertices[:, 2], 0.75)
        np.testing.assert_allclose(mesh.vertices[:, 3], 0.0)

    @unittest.skipUnless(
        os.environ.get("ORDINARYLIGHT_RUN_GPU_GATES") == "1",
        "GPU raster validation is opt-in",
    )
    def test_webgpu_implementation_draws_offscreen_when_available(self):
        if importlib.util.find_spec("wgpu") is None:
            self.skipTest("wgpu is unavailable")
        program = ol.RasterProgram.compile(
            raster_vertex, raster_fragment, target="wgsl", validate=False,
        )
        backend = ol.renderers.raster.WebGpuRasterRenderer(program)
        try:
            image = backend.render(ol.triangle_mesh(), 64, 48)
            self.assertEqual(image.shape, (48, 64, 4))
            self.assertEqual(image.dtype, np.float32)
            self.assertGreater(np.count_nonzero(image[..., 0] > 0.5), 100)
        finally:
            backend.close()

    @unittest.skipUnless(
        os.environ.get("ORDINARYLIGHT_RUN_GPU_GATES") == "1",
        "GPU raster validation is opt-in",
    )
    def test_vulkan_implementation_draws_offscreen_when_available(self):
        if importlib.util.find_spec("vulkan") is None:
            self.skipTest("vulkan is unavailable")
        program = ol.RasterProgram.compile(
            raster_vertex, raster_fragment, target="spirv",
        )
        backend = ol.renderers.raster.VulkanRasterRenderer(program)
        try:
            image = backend.render(ol.triangle_mesh(), 64, 48)
            self.assertEqual(image.shape, (48, 64, 4))
            self.assertEqual(image.dtype, np.float32)
            self.assertGreater(np.count_nonzero(image[..., 0] > 0.5), 100)
        finally:
            backend.close()

    @unittest.skipUnless(
        os.environ.get("ORDINARYLIGHT_RUN_GPU_GATES") == "1",
        "GPU raster validation is opt-in",
    )
    def test_screen_space_optics_runs_on_both_native_targets(self):
        from ordinarylight.showcases.optical_materials import (
            build_refraction_scene,
        )
        scene = build_refraction_scene()
        camera = ol.PerspectiveCamera((0, 3, 8), (0, 1, 0))
        target_types = []
        if importlib.util.find_spec("vulkan") is not None:
            target_types.append((
                "spirv", ol.renderers.raster.VulkanRasterRenderer,
            ))
        if importlib.util.find_spec("wgpu") is not None:
            target_types.append((
                "wgsl", ol.renderers.raster.WebGpuRasterRenderer,
            ))
        if not target_types:
            self.skipTest("no GPU raster backend is available")
        screen_images = []
        for target, implementation in target_types:
            with self.subTest(target=target):
                program = ol.RasterProgram.scene(
                    target=target, validate=False,
                    material_programs=scene.material_programs(
                        ol.builtin_material,
                    ),
                )
                images = []
                for tier in ("environment", "screen-space"):
                    backend = implementation(
                        program, config=ol.RasterConfig(
                            optical_quality=tier, shadows=False,
                            state=ol.RasterState(cull_mode="none"),
                        ),
                    )
                    try:
                        images.append(backend.render_frame(
                            scene, camera, 160, 90,
                        ))
                    finally:
                        backend.close()
                self.assertTrue(np.all(np.isfinite(images[1])))
                self.assertGreater(
                    float(np.mean(np.abs(images[1] - images[0]))), 1e-4,
                )
                screen_images.append(images[1])
        if len(screen_images) == 2:
            self.assertLess(
                float(np.mean(np.abs(screen_images[0] - screen_images[1]))),
                0.06,
            )

    @unittest.skipUnless(
        os.environ.get("ORDINARYLIGHT_RUN_GPU_GATES") == "1",
        "GPU raster validation is opt-in",
    )
    def test_scene_rendering_and_depth_match_implementation_contract(self):
        scene, camera = self._overlap_scene()
        choices = []
        if importlib.util.find_spec("vulkan") is not None:
            choices.append(ol.renderers.raster.VulkanRasterRenderer(
                ol.RasterProgram.scene(target="spirv"),
                state=ol.RasterState(cull_mode="none"),
            ))
        if importlib.util.find_spec("wgpu") is not None:
            choices.append(ol.renderers.raster.WebGpuRasterRenderer(
                ol.RasterProgram.scene(target="wgsl", validate=False),
                state=ol.RasterState(cull_mode="none"),
            ))
        if not choices:
            self.skipTest("no GPU raster backend is available")
        for backend in choices:
            with self.subTest(renderer=backend.capabilities.renderer):
                for feature in (
                    "volume-shadowing",
                    "overlapping-volume-extinction",
                    "volume-empty-space-skipping",
                ):
                    self.assertTrue(backend.capabilities.supports(feature))
                renderer = ol.Renderer(implementation=backend)
                try:
                    image = renderer.render(scene, camera, (96, 64))
                    self.assertEqual(image.dtype, np.float32)
                    self.assertEqual(image.shape, (64, 96, 4))
                    center = image[32, 48, :3]
                    self.assertGreater(center[0], 0.8)
                    self.assertLess(center[2], 0.2)
                    products = renderer.render(
                        scene, camera, (48, 32),
                        outputs=("color", "depth", "normal", "object_id"),
                    )
                    self.assertEqual(products["depth"].shape, (32, 48))
                    self.assertGreater(np.count_nonzero(products["object_id"]), 10)
                    mask = products["object_id"] != 0
                    self.assertTrue(np.isfinite(products["depth"][mask]).all())
                    self.assertTrue(np.isinf(products["depth"][~mask]).all())
                    self.assertGreater(
                        float(np.linalg.norm(
                            products["normal"][mask], axis=1,
                        ).mean()), 0.9,
                    )
                    first_motion = renderer.render(
                        scene, camera, (48, 32), outputs=("motion",),
                    )["motion"]
                    moved = ol.PerspectiveCamera((0.2, 0, 4), (0, 0, 0))
                    moved_products = renderer.render(
                        scene, moved, (48, 32),
                        outputs=("motion", "object_id"),
                    )
                    self.assertEqual(float(np.max(np.abs(first_motion))), 0.0)
                    moved_mask = moved_products["object_id"] != 0
                    self.assertGreater(float(np.max(np.abs(
                        moved_products["motion"][moved_mask]
                    ))), 0.01)
                finally:
                    renderer.close()

    @unittest.skipUnless(
        os.environ.get("ORDINARYLIGHT_RUN_GPU_GATES") == "1",
        "GPU raster validation is opt-in",
    )
    def test_scene_base_color_texture_is_sampled_on_both_targets(self):
        texture = ol.Texture(np.array([[[255, 16, 8, 255]]], np.uint8))
        mesh = ol.Mesh(
            [[-0.8, -0.8, 0], [0.8, -0.8, 0], [0, 0.8, 0]],
            [[0, 1, 2]],
            ol.Material(base_color_texture=texture),
        )
        scene = ol.Scene([mesh])
        camera = ol.PerspectiveCamera((0, 0, 4), (0, 0, 0))
        choices = []
        if importlib.util.find_spec("vulkan") is not None:
            choices.append(ol.renderers.raster.VulkanRasterRenderer(
                ol.RasterProgram.scene(target="spirv"),
                config=ol.RasterConfig(
                    direct_lighting=False, ambient_light=1.0,
                    state=ol.RasterState(cull_mode="none"),
                ),
            ))
        if importlib.util.find_spec("wgpu") is not None:
            choices.append(ol.renderers.raster.WebGpuRasterRenderer(
                ol.RasterProgram.scene(target="wgsl", validate=False),
                config=ol.RasterConfig(
                    direct_lighting=False, ambient_light=1.0,
                    state=ol.RasterState(cull_mode="none"),
                ),
            ))
        for backend in choices:
            try:
                image = backend.render_frame(scene, camera, 64, 48)
                cached_pipeline_count = (
                    len(backend._pipelines)
                    if hasattr(backend, "_pipelines") else None
                )
                second = backend.render_frame(scene, camera, 64, 48)
                center = image[24, 32, :3]
                self.assertGreater(float(center[0]), 0.5)
                self.assertLess(float(center[1]), 0.05)
                self.assertLess(float(center[2]), 0.05)
                np.testing.assert_allclose(second, image, atol=2e-3)
                if cached_pipeline_count is not None:
                    self.assertEqual(
                        len(backend._pipelines), cached_pipeline_count,
                    )
            finally:
                backend.close()

    @unittest.skipUnless(
        os.environ.get("ORDINARYLIGHT_RUN_GPU_GATES") == "1",
        "GPU raster validation is opt-in",
    )
    def test_screen_space_optics_executes_with_sampled_scene_depth(self):
        if importlib.util.find_spec("vulkan") is None:
            self.skipTest("Vulkan is unavailable")
        from ordinarylight.showcases.optical_materials import (
            build_environment_reflection_scene,
        )
        backend = ol.renderers.raster.VulkanRasterRenderer(
            ol.RasterProgram.scene(target="spirv"),
            config=ol.RasterConfig(
                optical_quality="screen-space", screen_space_ray_steps=16,
                state=ol.RasterState(cull_mode="none"),
            ),
        )
        try:
            scene = build_environment_reflection_scene()
            camera = ol.PerspectiveCamera((0, 3.0, 9.0), (0, 1.0, 0))
            image = backend.render_frame(scene, camera, 96, 64)
            repeated = backend.render_frame(scene, camera, 96, 64)
            self.assertTrue(np.isfinite(image).all())
            self.assertGreater(float(np.std(image[..., :3])), 0.02)
            np.testing.assert_allclose(repeated, image, atol=2e-3)
        finally:
            backend.close()

    @unittest.skipUnless(
        os.environ.get("ORDINARYLIGHT_RUN_GPU_GATES") == "1",
        "GPU raster validation is opt-in",
    )
    def test_tangent_space_normal_map_changes_fragment_lighting(self):
        if importlib.util.find_spec("vulkan") is None:
            self.skipTest("Vulkan is unavailable")
        vertices = [[-1,-1,0],[1,-1,0],[0,1,0]]
        indices = [[0,1,2]]
        camera = ol.PerspectiveCamera((0,0,4),(0,0,0))
        def render(normal_pixel):
            texture = ol.Texture(np.asarray([[normal_pixel]], np.uint8))
            mesh = ol.Mesh(
                vertices, indices,
                ol.Material(
                    base_color=(0.8,0.8,0.8), roughness=0.8,
                    normal_texture=texture,
                ),
                texcoords=[[0,0],[1,0],[0.5,1]],
            )
            scene = ol.Scene(
                [mesh], [ol.PointLight((3,0,3), intensity=18)],
            )
            backend = ol.renderers.raster.VulkanRasterRenderer(
                ol.RasterProgram.scene(target="spirv"),
                config=ol.RasterConfig(
                    ambient_light=0.0, shadows=False,
                    state=ol.RasterState(cull_mode="none"),
                ),
            )
            try:
                return backend.render_frame(scene, camera, 64, 48)[24,32,:3]
            finally:
                backend.close()
        neutral = render((128,128,255,255))
        tilted = render((255,128,128,255))
        # The framebuffer is already tone mapped, so even a substantial
        # tangent-space normal change can produce a modest display-space
        # delta.  Keep this above quantization noise while avoiding a test
        # that depends on a particular GPU's floating-point rounding.
        self.assertGreater(float(np.linalg.norm(neutral - tilted)), 0.005)

    @unittest.skipUnless(
        os.environ.get("ORDINARYLIGHT_RUN_GPU_GATES") == "1",
        "GPU raster validation is opt-in",
    )
    def test_custom_material_hook_executes_on_vulkan(self):
        if importlib.util.find_spec("vulkan") is None:
            self.skipTest("Vulkan is unavailable")
        mesh = ol.Mesh(
            [[-1,-1,0],[1,-1,0],[0,1,0]], [[0,1,2]],
            ol.Material(base_color=(0.8, 0.1, 0.1)),
            texcoords=[[0,0],[1,0],[0.5,1]],
        )
        scene = ol.Scene([mesh])
        camera = ol.PerspectiveCamera((0,0,4),(0,0,0))
        backend = ol.renderers.raster.VulkanRasterRenderer(
            ol.RasterProgram.scene(
                target="spirv", material_hook=striped_material_hook,
            ),
            config=ol.RasterConfig(
                material_hook=striped_material_hook,
                direct_lighting=False, ambient_light=1.0,
                state=ol.RasterState(cull_mode="none"),
            ),
        )
        try:
            center = backend.render_frame(scene, camera, 64, 48)[24,32,:3]
            self.assertGreater(float(center[2]), 0.3)
            self.assertGreater(float(center[1]), 0.2)
        finally:
            backend.close()

    @unittest.skipUnless(
        os.environ.get("ORDINARYLIGHT_RUN_GPU_GATES") == "1",
        "GPU raster validation is opt-in",
    )
    def test_vulkan_native_volume_ray_march_draws_without_proxy_geometry(self):
        if importlib.util.find_spec("vulkan") is None:
            self.skipTest("Vulkan is unavailable")
        from ordinarylight.showcases.volumes import build_volume_showcase
        scene = build_volume_showcase(16)
        for mesh in tuple(scene.meshes):
            scene.remove_mesh(mesh)
        camera = ol.PerspectiveCamera((5, 3, 6), (0, 1, 0))
        backend = ol.renderers.raster.VulkanRasterRenderer(
            ol.RasterProgram.scene(target="spirv"),
            config=ol.RasterConfig(
                volume_rendering="ray-march", volume_max_steps=256,
                state=ol.RasterState(cull_mode="none"),
            ),
        )
        try:
            image = backend.render_frame(scene, camera, 96, 64)
            self.assertTrue(np.isfinite(image).all())
            self.assertGreater(float(np.max(image[..., :3])), 0.25)
            self.assertGreater(float(np.std(image[..., :3])), 0.01)
        finally:
            backend.close()

    @unittest.skipUnless(
        os.environ.get("ORDINARYLIGHT_RUN_GPU_GATES") == "1",
        "GPU raster validation is opt-in",
    )
    def test_vulkan_native_volume_scattering_responds_to_scene_lights(self):
        if importlib.util.find_spec("vulkan") is None:
            self.skipTest("Vulkan is unavailable")
        from ordinarylight.showcases.volume_scattering import (
            build_volume_scattering_showcase,
        )

        lit = build_volume_scattering_showcase(20)
        unlit = build_volume_scattering_showcase(20)
        for scene in (lit, unlit):
            for mesh in tuple(scene.meshes):
                scene.remove_mesh(mesh)
        for light in tuple(unlit.lights):
            unlit.remove_light(light)
        camera = ol.PerspectiveCamera((5.2, 3.1, 6.2), (-0.1, 1.45, -0.7))
        backend = ol.renderers.raster.VulkanRasterRenderer(
            ol.RasterProgram.scene(target="spirv"),
            config=ol.RasterConfig(
                volume_rendering="ray-march", volume_max_steps=512,
                state=ol.RasterState(cull_mode="none"),
            ),
        )
        try:
            lit_image = backend.render_frame(lit, camera, 128, 96)
            unlit_image = backend.render_frame(unlit, camera, 128, 96)
        finally:
            backend.close()

        contribution = np.maximum(
            lit_image[..., :3] - unlit_image[..., :3], 0.0,
        )
        self.assertGreater(float(np.max(contribution)), 0.01)
        self.assertGreater(int(np.count_nonzero(
            np.max(contribution, axis=2) > 0.002,
        )), 50)

    @unittest.skipUnless(
        os.environ.get("ORDINARYLIGHT_RUN_GPU_GATES") == "1",
        "GPU raster validation is opt-in",
    )
    def test_vulkan_native_volume_empty_space_skipping_preserves_hdr(self):
        if importlib.util.find_spec("vulkan") is None:
            self.skipTest("Vulkan is unavailable")
        coordinates = np.linspace(-1.0, 1.0, 65, dtype=np.float32)
        z, y, x = np.meshgrid(
            coordinates, coordinates, coordinates, indexing="ij",
        )
        density = np.where(
            x * x + y * y + z * z < 0.12 ** 2, 1.0, 0.0,
        ).astype(np.float32)
        scene = ol.Scene(volumes=[ol.Volume(
            density,
            ol.VolumeMaterial(
                ol.Texture1D(((0, 0, 0, 0), (0.8, 0.25, 0.05, 0.2))),
                emission_scale=1.0, step_size=0.004,
            ),
            transform=(ol.Transform.translation((-1, -1, -1))
                       @ ol.Transform.scale((2, 2, 2))),
        )])
        camera = ol.PerspectiveCamera((0, 0, -3.2), (0, 0, 0))
        images = []
        for enabled in (False, True):
            backend = ol.renderers.raster.VulkanRasterRenderer(
                ol.RasterProgram.scene(target="spirv"),
                config=ol.RasterConfig(
                    volume_rendering="ray-march", volume_max_steps=1024,
                    volume_empty_space_skipping=enabled,
                    state=ol.RasterState(cull_mode="none"),
                ),
            )
            try:
                images.append(backend.render_frame(scene, camera, 96, 64))
            finally:
                backend.close()
        np.testing.assert_allclose(images[1], images[0], atol=2e-3)

    @unittest.skipUnless(
        os.environ.get("ORDINARYLIGHT_RUN_GPU_GATES") == "1",
        "GPU raster validation is opt-in",
    )
    def test_vulkan_native_volume_scattering_receives_opaque_shadows(self):
        if importlib.util.find_spec("vulkan") is None:
            self.skipTest("Vulkan is unavailable")

        def scene(blocked, lit=True):
            result = ol.Scene(volumes=[ol.Volume(
                np.ones((12, 12, 12), np.float32),
                ol.VolumeMaterial(
                    ol.Texture1D(((0, 0, 0, 0.055),) * 2),
                    emission_scale=0.0, step_size=0.04,
                    scattering_scale=1.0,
                    scattering_color=(0.45, 0.7, 1.0),
                    phase_function="henyey_greenstein", anisotropy=0.0,
                ),
                transform=(ol.Transform.translation((-0.6, -0.6, 0.0))
                           @ ol.Transform.scale((1.2, 1.2, 1.2))),
            )])
            if lit:
                result.add_point_light((0, 2, 2.0), intensity=38.0)
            if blocked:
                result.add_mesh(
                    ((-3, 1.2, -3), (3, 1.2, -3),
                     (3, 1.2, 3), (-3, 1.2, 3)),
                    ((0, 1, 2), (0, 2, 3)),
                    ol.Material(base_color=(0.01, 0.01, 0.01)),
                )
            return result

        camera = ol.PerspectiveCamera((0, 0, -2.3), (0, 0, 0.6))
        backend = ol.renderers.raster.VulkanRasterRenderer(
            ol.RasterProgram.scene(target="spirv"),
            config=ol.RasterConfig(
                volume_rendering="ray-march", volume_max_steps=512,
                ambient_light=0.0, shadows=True, shadow_map_size=512,
                state=ol.RasterState(cull_mode="none"),
            ),
        )
        try:
            unblocked = backend.render_frame(scene(False), camera, 128, 96)
            unblocked_control = backend.render_frame(
                scene(False, False), camera, 128, 96,
            )
            blocked = backend.render_frame(scene(True), camera, 128, 96)
            blocked_control = backend.render_frame(
                scene(True, False), camera, 128, 96,
            )
        finally:
            backend.close()
        center = np.s_[32:64, 48:80, :3]
        unblocked_effect = float(np.mean(np.maximum(
            unblocked[center] - unblocked_control[center], 0.0,
        )))
        blocked_effect = float(np.mean(np.maximum(
            blocked[center] - blocked_control[center], 0.0,
        )))
        self.assertGreater(unblocked_effect, 0.005)
        self.assertLess(blocked_effect, unblocked_effect * 0.8)

    @unittest.skipUnless(
        os.environ.get("ORDINARYLIGHT_RUN_GPU_GATES") == "1",
        "GPU raster validation is opt-in",
    )
    def test_vulkan_native_overlapping_volumes_accumulate_extinction(self):
        if importlib.util.find_spec("vulkan") is None:
            self.skipTest("Vulkan is unavailable")

        transform = (ol.Transform.translation((-0.6, -0.6, 0.0))
                     @ ol.Transform.scale((1.2, 1.2, 1.2)))
        scattering = ol.VolumeMaterial(
            ol.Texture1D(((0, 0, 0, 0.04),) * 2),
            emission_scale=0.0, step_size=0.04,
            scattering_scale=1.0, scattering_color=(0.6, 0.8, 1.0),
        )
        absorber = ol.VolumeMaterial(
            ol.Texture1D(((0, 0, 0, 0.16),) * 2),
            emission_scale=0.0, step_size=0.04, scattering_scale=0.0,
        )

        def scene(overlap, lit):
            volumes = [ol.Volume(
                np.ones((8, 8, 8), np.float32), scattering,
                transform=transform,
            )]
            if overlap:
                volumes.append(ol.Volume(
                    np.ones((8, 8, 8), np.float32), absorber,
                    transform=transform,
                ))
            result = ol.Scene(volumes=volumes)
            if lit:
                result.add_point_light((0, 2, 2), intensity=38.0)
            return result

        camera = ol.PerspectiveCamera((0, 0, -2.3), (0, 0, 0.6))
        backend = ol.renderers.raster.VulkanRasterRenderer(
            ol.RasterProgram.scene(target="spirv"),
            config=ol.RasterConfig(
                volume_rendering="ray-march", volume_max_steps=512,
                ambient_light=0.0, shadows=True,
                state=ol.RasterState(cull_mode="none"),
            ),
        )
        try:
            single = backend.render_frame(scene(False, True), camera, 96, 64)
            single_control = backend.render_frame(
                scene(False, False), camera, 96, 64,
            )
            overlap = backend.render_frame(scene(True, True), camera, 96, 64)
            overlap_control = backend.render_frame(
                scene(True, False), camera, 96, 64,
            )
        finally:
            backend.close()
        center = np.s_[20:44, 36:60, :3]
        single_effect = float(np.mean(np.maximum(
            single[center] - single_control[center], 0.0,
        )))
        overlap_effect = float(np.mean(np.maximum(
            overlap[center] - overlap_control[center], 0.0,
        )))
        self.assertGreater(single_effect, 0.005)
        self.assertLess(overlap_effect, single_effect * 0.75)

    @unittest.skipUnless(
        os.environ.get("ORDINARYLIGHT_RUN_GPU_GATES") == "1",
        "GPU raster validation is opt-in",
    )
    def test_vulkan_native_volume_composite_preserves_scene_orientation(self):
        if importlib.util.find_spec("vulkan") is None:
            self.skipTest("Vulkan is unavailable")
        from ordinarylight.showcases.volumes import build_volume_showcase
        scene = build_volume_showcase(16)
        camera = ol.PerspectiveCamera((5, 3, 6), (0, 1, 0))
        images = []
        for mode in ("slices", "ray-march"):
            backend = ol.renderers.raster.VulkanRasterRenderer(
                ol.RasterProgram.scene(target="spirv"),
                config=ol.RasterConfig(
                    volume_rendering=mode, volume_slices=48,
                    volume_max_steps=256,
                    state=ol.RasterState(cull_mode="none"),
                ),
            )
            try:
                images.append(backend.render_frame(scene, camera, 96, 64))
            finally:
                backend.close()
        reference, native = (image[..., :3] for image in images)
        # Compare the mostly opaque/background portion.  The two volume
        # algorithms need not match exactly, but an inverted composite is
        # overwhelmingly closer after flipping than in its native layout.
        mask = np.max(reference, axis=2) < 1.0
        direct_error = float(np.mean(np.abs(reference[mask] - native[mask])))
        flipped_error = float(
            np.mean(np.abs(reference[mask] - native[::-1][mask]))
        )
        self.assertLess(direct_error, flipped_error * 0.25)

    @unittest.skipUnless(
        os.environ.get("ORDINARYLIGHT_RUN_GPU_GATES") == "1",
        "GPU raster validation is opt-in",
    )
    def test_webgpu_native_volume_composite_preserves_scene_orientation(self):
        if importlib.util.find_spec("wgpu") is None:
            self.skipTest("WebGPU is unavailable")
        from ordinarylight.showcases.volumes import build_volume_showcase
        scene = build_volume_showcase(16)
        camera = ol.PerspectiveCamera((5, 3, 6), (0, 1, 0))
        images = []
        for mode in ("slices", "ray-march"):
            backend = ol.renderers.raster.WebGpuRasterRenderer(
                ol.RasterProgram.scene(target="wgsl"),
                config=ol.RasterConfig(
                    volume_rendering=mode, volume_slices=48,
                    volume_max_steps=256,
                    state=ol.RasterState(cull_mode="none"),
                ),
            )
            try:
                images.append(backend.render_frame(scene, camera, 96, 64))
            finally:
                backend.close()
        reference, native = (image[..., :3] for image in images)
        mask = np.max(reference, axis=2) < 1.0
        direct_error = float(np.mean(np.abs(reference[mask] - native[mask])))
        flipped_error = float(
            np.mean(np.abs(reference[mask] - native[::-1][mask]))
        )
        self.assertLess(direct_error, flipped_error * 0.25)

    @unittest.skipUnless(
        os.environ.get("ORDINARYLIGHT_RUN_GPU_GATES") == "1",
        "GPU raster validation is opt-in",
    )
    def test_vulkan_native_volume_matches_transformed_voxel_support(self):
        if importlib.util.find_spec("vulkan") is None:
            self.skipTest("Vulkan is unavailable")
        from ordinarylight.raster import camera_matrix
        from ordinarylight.showcases.volumes import build_volume_showcase

        width, height = 320, 180
        camera = ol.PerspectiveCamera(
            (-6.9692433238091684, 3.1, -5.178113013123175),
            (-0.1, 1.45, -0.7), up=(0.0, 1.0, 0.0),
            vertical_fov_degrees=45.0,
        )
        scene = build_volume_showcase(32)
        control = build_volume_showcase(32)
        control.remove_volume(control.volumes[0])
        backend = ol.renderers.raster.VulkanRasterRenderer(
            ol.RasterProgram.scene(target="spirv"),
            config=ol.RasterConfig(
                volume_rendering="ray-march", volume_max_steps=512,
                state=ol.RasterState(cull_mode="none"),
            ),
        )
        try:
            active = backend.render_frame(scene, camera, width, height)
            bare = backend.render_frame(control, camera, width, height)
        finally:
            backend.close()

        difference = np.linalg.norm(
            active[..., :3] - bare[..., :3], axis=2,
        )
        rendered_y, rendered_x = np.where(difference > 0.015)
        self.assertGreater(len(rendered_x), 100)

        volume = scene.volumes[0]
        z, y, x = np.where(volume.normalized_data > 0.02)
        depth, rows, columns = volume.shape
        local = np.column_stack((
            x / (columns - 1), y / (rows - 1), z / (depth - 1),
            np.ones(len(x)),
        ))
        world = local @ volume.transform.matrix.T
        clip = world @ camera_matrix(camera, width, height).T
        ndc = clip[:, :3] / clip[:, 3, None]
        projected_x = (ndc[:, 0] * 0.5 + 0.5) * width
        projected_y = (0.5 - ndc[:, 1] * 0.5) * height

        self.assertLess(abs(rendered_x.mean() - projected_x.mean()), 3.0)
        self.assertLess(abs(rendered_y.mean() - projected_y.mean()), 3.0)
