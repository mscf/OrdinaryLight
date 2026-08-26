import unittest

import numpy as np

import ordinarylight as ol


class SceneTests(unittest.TestCase):
    def test_rejects_invalid_indices(self):
        with self.assertRaises(ValueError):
            ol.Mesh(
                np.zeros((3, 3), dtype=np.float32),
                np.asarray(((0, 1, 3),), dtype=np.uint32),
            )

    def test_packs_triangle_materials(self):
        scene = ol.Scene()
        scene.add_mesh(
            ((-1, -1, 0), (1, -1, 0), (0, 1, 0)),
            ((0, 1, 2),),
            ol.Material(base_color=(1, 0, 0)),
        )
        triangles, colors, emissions = scene.triangles()
        self.assertEqual(triangles.shape, (1, 3, 3))
        np.testing.assert_array_equal(colors, ((1, 0, 0),))
        np.testing.assert_array_equal(emissions, ((0, 0, 0),))

    def test_generates_and_packs_vertex_attributes(self):
        scene = ol.Scene()
        mesh = scene.add_mesh(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0)), ((0, 1, 2),),
            texcoords=((0, 0), (1, 0), (0, 1)),
            texcoords1=((0.2, 0.3), (0.7, 0.3), (0.2, 0.9)),
        )
        np.testing.assert_allclose(mesh.normals, ((0, 0, 1),) * 3)
        packed = scene.triangle_attribute_data()
        self.assertEqual(packed.shape, (3, 3, 4))
        np.testing.assert_allclose(packed[:, 1, :2], ((0, 0), (1, 0), (0, 1)))
        np.testing.assert_allclose(
            packed[:, 1, 2:], ((0.2, 0.3), (0.7, 0.3), (0.2, 0.9))
        )
        np.testing.assert_allclose(mesh.tangents[:, :3], ((1, 0, 0),) * 3)
        np.testing.assert_allclose(mesh.tangents[:, 3], (1, 1, 1))
        np.testing.assert_allclose(packed[:, 0, 3], (1, 1, 1))

    def test_named_vertex_attributes_are_validated_and_packed(self):
        scene = ol.Scene()
        mesh = scene.add_mesh(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0)), ((0, 1, 2),),
            attributes={
                "color": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                "weight": (0.0, 0.5, 1.0),
            },
        )
        self.assertIn("normal", mesh.vertex_attribute_names)
        self.assertIn("weight", mesh.vertex_attribute_names)
        self.assertEqual(mesh.vertex_attribute("weight").shape, (3, 1))
        self.assertFalse(mesh.vertex_attribute("weight").flags.writeable)
        np.testing.assert_array_equal(
            scene.triangle_vertex_attribute_data("color"),
            ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        )
        with self.assertRaises(ValueError):
            scene.add_mesh(
                ((0, 0, 0), (1, 0, 0), (0, 1, 0)), ((0, 1, 2),),
                attributes={"bad-name": (0, 1, 2)},
            )
        with self.assertRaises(ValueError):
            scene.add_mesh(
                ((0, 0, 0), (1, 0, 0), (0, 1, 0)), ((0, 1, 2),),
                attributes={"position": ((0, 0, 0),) * 3},
            )

    def test_selected_attribute_layout_has_stable_vec4_abi(self):
        scene = ol.Scene()
        for offset in (0, 2):
            scene.add_mesh(
                ((offset, 0, 0), (offset + 1, 0, 0), (offset, 1, 0)),
                ((0, 1, 2),),
                attributes={"weight": (0, 0.5, 1), "color": ((1, 0, 0),) * 3},
            )
        layout = ol.VertexAttributeLayout.from_scene(
            scene, ("weight", "color")
        )
        self.assertEqual(layout.channels, (("weight", 1), ("color", 3)))
        self.assertEqual(layout.slot("color"), 1)
        packed = layout.pack(scene)
        self.assertEqual(packed.shape, (6, 2, 4))
        np.testing.assert_array_equal(packed[:3, 0, 0], (0, 0.5, 1))
        np.testing.assert_array_equal(packed[:, 1, :3], ((1, 0, 0),) * 6)
        np.testing.assert_array_equal(packed[..., 3], 0)

        inconsistent = ol.Scene()
        inconsistent.add_mesh(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0)), ((0, 1, 2),),
            attributes={"value": (0, 1, 2)},
        )
        inconsistent.add_mesh(
            ((0, 0, 1), (1, 0, 1), (0, 1, 1)), ((0, 1, 2),),
            attributes={"value": ((0, 0), (1, 1), (2, 2))},
        )
        with self.assertRaises(ValueError):
            ol.VertexAttributeLayout.from_scene(inconsistent, ("value",))

    def test_packs_point_lights(self):
        scene = ol.Scene()
        scene.add_point_light((1, 2, 3), color=(0.5, 0.75, 1.0), intensity=12.0)
        data = scene.point_light_data()
        self.assertEqual(data.shape, (1, 2, 4))
        np.testing.assert_allclose(data[0, 0, :3], (1, 2, 3))
        np.testing.assert_allclose(data[0, 1], (0.5, 0.75, 1.0, 12.0))

    def test_packs_emissive_triangles(self):
        scene = ol.Scene()
        scene.add_mesh(
            ((0, 0, 0), (2, 0, 0), (0, 2, 0)), ((0, 1, 2),),
            ol.Material(emission=(4, 3, 2)),
        )
        data = scene.emissive_triangle_data()
        self.assertEqual(scene.emissive_triangle_count, 1)
        self.assertEqual(data.shape, (1, 5, 4))
        np.testing.assert_allclose(data[0, 3], (4, 3, 2, 2))
        np.testing.assert_allclose(data[0, 4, :2], (1, 1))

    def test_packs_two_sided_emission_flag(self):
        scene = ol.Scene()
        scene.add_mesh(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0)), ((0, 1, 2),),
            ol.Material(emission=(1, 1, 1), emission_two_sided=True),
        )
        self.assertEqual(scene.emissive_triangle_data()[0, 4, 2], 1.0)
        self.assertEqual(scene.triangle_material_data()[0, 3, 3], 1.0)

    def test_weights_emissive_triangles_by_area_and_power(self):
        scene = ol.Scene()
        scene.add_mesh(
            ((0, 0, 0), (1, 0, 0), (0, 2, 0)), ((0, 1, 2),),
            ol.Material(emission=(1, 1, 1)),
        )
        scene.add_mesh(
            ((0, 0, 1), (2, 0, 1), (0, 2, 1)), ((0, 1, 2),),
            ol.Material(emission=(2, 2, 2)),
        )
        data = scene.emissive_triangle_data()
        np.testing.assert_allclose(data[:, 4, 1], (0.2, 0.8), rtol=1e-6)
        np.testing.assert_allclose(data[:, 4, 0], (0.2, 1.0), rtol=1e-6)
        self.assertAlmostEqual(scene.emissive_light_weight, 5.0)


class ReferenceRendererTests(unittest.TestCase):
    def test_reference_backend_implements_high_level_hdr_contract(self):
        scene = ol.Scene()
        camera = ol.PanoramicCamera((0, 0, -3), (0, 0, 0))
        backend = ol.backends.ReferenceBackend(
            samples_per_pixel=1, max_bounces=2, seed=7
        )
        with ol.Renderer(backend=backend) as renderer:
            image = renderer.render(scene, camera, (8, 4))
            self.assertEqual(image.shape, (4, 8, 4))
            self.assertEqual(image.dtype, np.float32)
            self.assertGreater(float(image[..., :3].max()), 0.0)
            self.assertEqual(renderer.capabilities.backend, "cpu-reference")
            self.assertFalse(renderer.capabilities.supports("hardware_ray_tracing"))
        with self.assertRaises(RuntimeError):
            backend.render_wavefront(scene, camera, 8, 4)

    def test_render_shape_type_and_repeatability(self):
        scene = ol.Scene()
        scene.add_mesh(
            ((-2, -1, 0), (2, -1, 0), (0, 2, 0)),
            ((0, 1, 2),),
            ol.Material(base_color=(0.8, 0.2, 0.1)),
        )
        camera = ol.PerspectiveCamera((0, 0, -3), (0, 0, 0))
        renderer = ol.ReferencePathTracer(seed=4)
        first = renderer.render(scene, camera, 24, 16, samples=2)
        second = renderer.render(scene, camera, 24, 16, samples=2)
        self.assertEqual(first.shape, (16, 24, 4))
        self.assertEqual(first.dtype, np.uint8)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all(first[..., 3] == 255))

    def test_render_to_array_surface(self):
        scene = ol.Scene()
        camera = ol.PerspectiveCamera((0, 0, -3), (0, 0, 0))
        surface = ol.ArraySurface(12, 8)
        result = ol.ReferencePathTracer(seed=2).render_to(
            scene, camera, surface, samples=1
        )
        self.assertIs(result, surface.pixels)
        self.assertEqual(result.shape, (8, 12, 4))
        self.assertGreater(int(result[..., :3].sum()), 0)


class SurfaceTests(unittest.TestCase):
    def test_rejects_wrong_shape_and_type(self):
        surface = ol.ArraySurface(4, 3)
        with self.assertRaises(ValueError):
            surface.present(np.zeros((3, 5, 4), dtype=np.uint8))
        with self.assertRaises(ValueError):
            surface.present(np.zeros((3, 4, 4), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
