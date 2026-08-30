import unittest

import numpy as np

import ordinarylight as ol


class _Backend:
    available_outputs = ("color",)
    config = None
    device = None
    last_timings = {"total_ms": 1.0}

    def __init__(self, value): self.value = value; self.closed = False
    def render_frame(self, _scene, _camera, width, height, **_kwargs):
        return np.full((height, width, 4), self.value, np.float32)
    def close(self): self.closed = True


def _camera():
    return ol.PerspectiveCamera((0, 0, 4), (0, 0, 0))


class RasterFeatureTests(unittest.TestCase):
    def test_shared_material_texture_is_evaluated(self):
        pixels = np.array([[[255, 0, 0, 255], [0, 255, 0, 255]]], np.uint8)
        material = ol.Material(
            base_color=(1, 1, 1), base_color_texture=ol.Texture(pixels),
        )
        mesh = ol.Mesh(
            [[-1, -1, 0], [1, -1, 0], [0, 1, 0]], [[0, 1, 2]], material,
            texcoords=[[0, 0], [0.99, 0], [0, 0]],
        )
        packed = ol.scene_mesh(
            ol.Scene([mesh]), _camera(), 64, 64,
            ol.RasterConfig(direct_lighting=False),
        )
        self.assertGreater(packed.vertices[0, 4], 0.9)
        self.assertGreater(packed.vertices[1, 5], 0.9)
        self.assertEqual(packed.resources["base_color_atlas"].shape, (1, 5, 4))
        uv_offset = next(
            item.offset // 4 for item in packed.layout.attributes
            if item.semantic == "base_color_uv"
        )
        self.assertLess(
            float(packed.vertices[0, uv_offset]),
            float(packed.vertices[1, uv_offset]),
        )

    def test_direct_light_and_hard_shadow_are_composable(self):
        receiver = ol.Mesh(
            [[-1, -1, 0], [1, -1, 0], [0, 1, 0]], [[0, 1, 2]],
            ol.Material(base_color=(1, 1, 1)),
        )
        blocker = ol.Mesh(
            [[-2, -2, 1], [2, -2, 1], [0, 2, 1]], [[0, 1, 2]],
            ol.Material(),
        )
        light = ol.PointLight((0, 0, 2), intensity=8)
        scene = ol.Scene([receiver, blocker], [light])
        lit = ol.scene_mesh(scene, _camera(), 64, 64, ol.RasterConfig(ambient_light=0, shadows=False, shading_model="diffuse"))
        shadowed = ol.scene_mesh(scene, _camera(), 64, 64, ol.RasterConfig(ambient_light=0, shadows=True, shading_model="diffuse"))
        self.assertGreater(float(lit.vertices[:3, 4:7].mean()), 0.1)
        self.assertLess(float(shadowed.vertices[:3, 4:7].mean()), 1e-5)

    def test_directional_shadow_depth_is_packed_and_addressed(self):
        floor = ol.Mesh(
            [[-2, -2, 0], [2, -2, 0], [2, 2, 0], [-2, 2, 0]],
            [[0, 1, 2], [0, 2, 3]],
        )
        blocker = ol.Mesh(
            [[-0.5, -0.5, 1], [0.5, -0.5, 1], [0, 0.5, 1]],
            [[0, 1, 2]],
        )
        packed = ol.scene_mesh(
            ol.Scene(
                [floor, blocker],
                [ol.DirectionalLight((0, 0, -1), intensity=3)],
            ),
            _camera(), 64, 64,
            ol.RasterConfig(shadow_map_size=32),
        )
        atlas = packed.resources["base_color_atlas"]
        self.assertEqual(atlas.shape, (33, 32, 4))
        self.assertLess(int(atlas[1:, :, 3].min()), 255)
        shadow_offset = next(
            item.offset // 4 for item in packed.layout.attributes
            if item.semantic == "shadow_coordinate"
        )
        coordinates = packed.vertices[:, shadow_offset:shadow_offset + 4]
        projected = coordinates[:, :3] / np.maximum(
            np.abs(coordinates[:, 3:4]), 1e-8,
        )
        self.assertTrue(np.all(projected[:, :2] >= 0.0))
        self.assertTrue(np.all(projected[:, :2] <= 1.0))

    def test_native_shadow_map_does_not_allocate_cpu_shadow_pixels(self):
        mesh = ol.Mesh(
            [[-1, -1, 0], [1, -1, 0], [0, 1, 0]], [[0, 1, 2]],
        )
        packed = ol.scene_mesh(
            ol.Scene(
                [mesh], [ol.DirectionalLight((0, 0, -1), intensity=3)],
            ),
            _camera(), 64, 64, ol.RasterConfig(shadow_map_size=2048),
            native_shadow_maps=True,
        )
        self.assertEqual(packed.resources["base_color_atlas"].shape, (1, 3, 4))
        self.assertEqual(
            packed.resources["shadow_rectangle"],
            (0, 0, 2048, 2048, 2048, 2048),
        )

    def test_gpu_camera_mesh_keeps_world_space_vertices_resident(self):
        mesh = ol.Mesh(
            [[-1, -1, 0], [1, -1, 0], [0, 1, 0]], [[0, 1, 2]],
        )
        packed = ol.scene_mesh(
            ol.Scene([mesh]), _camera(), 1920, 1080,
            ol.RasterConfig(shadows=False), gpu_camera=True,
        )
        np.testing.assert_allclose(
            packed.vertices[:, :4],
            np.column_stack((mesh.world_vertices, np.ones(3))),
        )
        self.assertTrue(packed.resources["gpu_camera"])

    def test_native_directional_and_spot_shadow_depths_use_zero_to_one(self):
        from ordinarylight.showcases.raster_features import (
            build_directional_shadow_scene, build_spot_shadow_scene,
        )
        for factory in (build_directional_shadow_scene, build_spot_shadow_scene):
            scene = factory()
            packed = ol.scene_mesh(
                scene, _camera(), 320, 180,
                ol.RasterConfig(shadow_map_size=64),
                native_shadow_maps=True,
            )
            offset = next(
                item.offset // 4 for item in packed.layout.attributes
                if item.semantic == "shadow_coordinate"
            )
            coordinates = packed.vertices[:, offset:offset + 4]
            active = np.abs(coordinates[:, 3]) > 1e-5
            self.assertTrue(np.any(active))
            projected_depth = (
                coordinates[active, 2] / np.abs(coordinates[active, 3])
            )
            self.assertGreaterEqual(float(projected_depth.min()), -1e-5)
            self.assertLessEqual(float(projected_depth.max()), 1.0 + 1e-5)

    def test_shadow_showcase_boxes_have_flat_face_normals(self):
        from ordinarylight.showcases.raster_features import (
            build_directional_shadow_scene,
        )
        scene = build_directional_shadow_scene()
        for mesh in scene.visible_meshes[1:]:
            absolute = np.abs(mesh.normals)
            np.testing.assert_allclose(absolute.sum(axis=1), 1.0, atol=1e-6)
            self.assertTrue(np.all(np.count_nonzero(absolute > 0.5, axis=1) == 1))
            self.assertEqual(len(mesh.vertices), 24)

    def test_shadow_cull_mode_validation(self):
        self.assertEqual(ol.RasterConfig().shadow_cull_mode, "none")
        self.assertEqual(
            ol.RasterConfig(shadow_cull_mode="none").shadow_cull_mode, "none",
        )
        with self.assertRaises(ValueError):
            ol.RasterConfig(shadow_cull_mode="sideways")
        with self.assertRaises(ValueError):
            ol.RasterConfig(shadow_normal_bias=-1.0)

    def test_shadow_receivers_apply_planned_normal_bias(self):
        from ordinarylight.raster.shadows import plan_shadow_maps
        from ordinarylight.showcases.raster_features import (
            build_directional_shadow_scene,
        )
        scene = build_directional_shadow_scene()
        request = plan_shadow_maps(scene, extent=(64, 64), max_maps=1)[0]
        self.assertGreater(request.normal_bias, 0.0)
        packed = ol.scene_mesh(
            scene, _camera(), 64, 64, ol.RasterConfig(shadow_map_size=64),
            native_shadow_maps=True,
        )
        offset = next(
            item.offset // 4 for item in packed.layout.attributes
            if item.semantic == "shadow_coordinate"
        )
        coordinates = packed.vertices[:, offset:offset + 4]
        self.assertTrue(np.all(np.isfinite(coordinates)))

    def test_shadow_normal_bias_scales_with_map_resolution(self):
        from ordinarylight.raster.shadows import plan_shadow_maps
        from ordinarylight.showcases.raster_features import (
            build_directional_shadow_scene,
        )
        scene = build_directional_shadow_scene()
        low = plan_shadow_maps(scene, extent=(512, 512), max_maps=1)[0]
        high = plan_shadow_maps(scene, extent=(4096, 4096), max_maps=1)[0]
        self.assertAlmostEqual(low.normal_bias / high.normal_bias, 8.0)

    def test_complete_material_channels_use_gpu_record_and_atlas(self):
        pixels = lambda rgba: ol.Texture(np.asarray([[rgba]], np.uint8))
        material = ol.Material(
            base_color=(0.2, 0.4, 0.8), roughness=0.7, metallic=0.6,
            emission=(0.1, 0.2, 0.3), transmission=0.5,
            base_color_texture=pixels((250, 20, 10, 255)),
            metallic_roughness_texture=pixels((0, 80, 220, 255)),
            emissive_texture=pixels((10, 240, 20, 255)),
            normal_texture=pixels((180, 100, 240, 255)),
            occlusion_texture=pixels((90, 90, 90, 255)),
            transmission_texture=pixels((130, 130, 130, 255)),
        )
        mesh = ol.Mesh(
            [[-1,-1,0],[1,-1,0],[0,1,0]], [[0,1,2]], material,
            texcoords=[[0,0],[1,0],[0.5,1]],
        )
        packed = ol.scene_mesh(
            ol.Scene([mesh]), _camera(), 64, 64, ol.RasterConfig(),
        )
        self.assertEqual(packed.resources["base_color_atlas"].shape, (1, 9, 4))
        records = np.frombuffer(
            packed.resources["material_buffer"], dtype=ol.MATERIAL_DTYPE,
        )
        self.assertEqual(len(records), 1)
        np.testing.assert_allclose(
            records["base_color_roughness"][0], (0.2,0.4,0.8,0.7),
        )
        semantics = {item.semantic for item in packed.layout.attributes}
        self.assertTrue({
            "tangent", "metallic_roughness_uv", "emissive_uv", "normal_uv",
            "occlusion_uv", "transmission_uv", "material_index",
        }.issubset(semantics))

    def test_material_programs_select_portable_raster_models(self):
        from ordinarylight.showcases.materials import (
            diffuse, fresnel_glass, mirror,
        )
        programs = (diffuse, mirror, fresnel_glass, ol.unlit_material)
        scene = ol.Scene([
            ol.Mesh(
                [[-1,-1,0],[1,-1,0],[0,1,0]], [[0,1,2]],
                ol.Material(program=program),
            ) for program in programs
        ])
        packed = ol.scene_mesh(scene, _camera(), 64, 64)
        records = np.frombuffer(
            packed.resources["material_buffer"], dtype=ol.MATERIAL_DTYPE,
        )
        np.testing.assert_array_equal(
            np.floor(records["ior_distance_program_flags"][:, 3]),
            (1, 2, 3, 4),
        )
        self.assertEqual(packed.resources["material_programs"], programs)

    def test_pbr_raster_material_distinguishes_metal_and_roughness(self):
        vertices = [[-1, -1, 0], [1, -1, 0], [0, 1, 0]]
        indices = [[0, 1, 2]]
        diffuse = ol.Mesh(
            vertices, indices,
            ol.Material(base_color=(0.8, 0.2, 0.1), roughness=1.0),
        )
        metal = ol.Mesh(
            vertices, indices,
            ol.Material(
                base_color=(0.8, 0.2, 0.1), metallic=1.0, roughness=0.15,
            ),
        )
        config = ol.RasterConfig(ambient_light=0.0, shadows=False)
        diffuse_data = ol.scene_mesh(
            ol.Scene([diffuse], [ol.PointLight((0, 0, 2), intensity=12)]),
            _camera(), 64, 64, config,
        ).vertices
        metal_data = ol.scene_mesh(
            ol.Scene([metal], [ol.PointLight((0, 0, 2), intensity=12)]),
            _camera(), 64, 64, config,
        ).vertices
        np.testing.assert_allclose(diffuse_data[:, 4:7], metal_data[:, 4:7])
        self.assertTrue(np.all(diffuse_data[:, 13] == 0.0))
        self.assertTrue(np.all(metal_data[:, 13] == 1.0))
        self.assertTrue(np.all(diffuse_data[:, 14] == 1.0))
        self.assertTrue(np.all(metal_data[:, 14] == 0.15))

    def test_pbr_raster_reads_metallic_roughness_and_emissive_textures(self):
        pixels = np.array([[[0, 128, 255, 255]]], np.uint8)
        emissive = np.array([[[128, 64, 32, 255]]], np.uint8)
        material = ol.Material(
            base_color=(0.5, 0.5, 0.5), metallic=0.5, roughness=0.5,
            emission=(1, 1, 1),
            metallic_roughness_texture=ol.Texture(pixels),
            emissive_texture=ol.Texture(emissive),
        )
        mesh = ol.Mesh(
            [[-1, -1, 0], [1, -1, 0], [0, 1, 0]], [[0, 1, 2]], material,
        )
        channels = ol.raster.material_channels(mesh)
        np.testing.assert_allclose(channels[1], 0.5, atol=1e-6)
        np.testing.assert_allclose(channels[2], 0.5 * 128 / 255, atol=1e-6)
        np.testing.assert_allclose(
            channels[3][0], np.array((128, 64, 32)) / 255, atol=1e-6,
        )

    def test_temporal_history_only_accumulates_compatible_frames(self):
        config = ol.RasterConfig(temporal_history=True, temporal_weight=0.5)
        post = ol.RasterPostProcessor(config)
        scene = ol.Scene(); camera = _camera()
        first = post.process(np.zeros((2, 2, 4), np.float32), scene, camera)
        second = post.process(np.ones((2, 2, 4), np.float32), scene, camera)
        self.assertEqual(post.accumulated_frames, 2)
        np.testing.assert_allclose(second, 0.5)
        moved = ol.PerspectiveCamera((1, 0, 4), (0, 0, 0))
        third = post.process(np.ones((2, 2, 4), np.float32), scene, moved)
        self.assertEqual(post.accumulated_frames, 1)
        np.testing.assert_allclose(third, 1.0)

    def test_render_graph_declares_multipass_dependencies(self):
        pipeline = ol.create_raster_pipeline(ol.RasterConfig(temporal_history=True))
        self.assertEqual(
            pipeline.stage_names,
            ("shadow_maps", "forward_lighting", "temporal", "tone_map"),
        )
        self.assertIn("output", pipeline.output_resources)

    def test_geometry_products_preserve_depth_normal_and_object_id(self):
        mesh = ol.Mesh(
            [[-0.5, -0.5, 0], [0.5, -0.5, 0], [0, 0.5, 0]], [[0, 1, 2]],
        )
        scene = ol.Scene([mesh])
        packed = ol.scene_mesh(scene, _camera(), 32, 24)
        products = ol.rasterize_geometry_products(packed, 32, 24)
        mask = products["depth"] < 1.0
        self.assertGreater(np.count_nonzero(mask), 10)
        self.assertTrue(np.all(products["object_id"][mask] == mesh.id))
        self.assertGreater(float(np.linalg.norm(products["normal"][mask], axis=1).mean()), 0.9)

    def test_volume_slice_geometry_uses_transfer_function_alpha(self):
        data = np.ones((2, 2, 2), np.float32)
        volume = ol.Volume(data)
        mesh = ol.scene_mesh(
            ol.Scene(volumes=[volume]), _camera(), 64, 64,
            ol.RasterConfig(volume_slices=3),
        )
        self.assertEqual(mesh.vertices.shape, (12, 62))
        self.assertGreater(float(mesh.vertices[:, 16].max()), 0.0)

    def test_hybrid_implementation_composes_child_renderers(self):
        raster, lighting = _Backend(0.25), _Backend(0.5)
        renderer = ol.Renderer(implementation=ol.renderers.hybrid.HybridRenderer(raster, lighting, weight=0.5))
        try:
            result = renderer.render(ol.Scene(), _camera(), (3, 2))
            np.testing.assert_allclose(result[..., :3], 0.5)
            np.testing.assert_allclose(result[..., 3], 0.25)
        finally:
            renderer.close()
        self.assertTrue(raster.closed and lighting.closed)

    def test_renderer_presents_to_array_surface(self):
        renderer = ol.Renderer(implementation=_Backend(0.5))
        surface = ol.ArraySurface(3, 2)
        try:
            result = renderer.render_to(ol.Scene(), _camera(), surface)
        finally:
            renderer.close()
        self.assertEqual(result.dtype, np.uint8)
        self.assertEqual(int(result[0, 0, 0]), 128)


if __name__ == "__main__":
    unittest.main()
