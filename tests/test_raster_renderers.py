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
                second = backend.render_frame(scene, camera, 64, 48)
                center = image[24, 32, :3]
                self.assertGreater(float(center[0]), 0.5)
                self.assertLess(float(center[1]), 0.05)
                self.assertLess(float(center[2]), 0.05)
                np.testing.assert_allclose(second, image, atol=2e-3)
                if hasattr(backend, "_pipelines"):
                    self.assertEqual(len(backend._pipelines), 1)
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
