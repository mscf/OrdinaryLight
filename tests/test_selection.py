import unittest

import numpy as np

import ordinarylight as ol


class SelectionTests(unittest.TestCase):
    def test_viewport_mapping_handles_dpi_letterbox_and_render_scale(self):
        mapping = ol.ViewportMapping(
            (1000, 800), framebuffer_size=(2000, 1600),
            render_size=(1000, 800), content_rect=(100, 100, 800, 600),
        )
        self.assertEqual(mapping.map_pixel((500, 400)), (500.0, 400.0))
        self.assertEqual(
            mapping.map_pixel((500, 400), target="framebuffer"),
            (1000.0, 800.0),
        )
        self.assertIsNone(mapping.map_pixel((50, 400)))
        with self.assertRaises(ValueError):
            mapping.map_pixel((500, 400), target="texture")

    def test_pick_policy_can_look_through_glass_and_filter_volumes(self):
        scene = ol.Scene()
        glass = scene.add_mesh(
            ((-1, -1, 1), (1, -1, 1), (0, 1, 1)), ((0, 1, 2),),
            ol.Material(transmission=1.0), name="glass",
        )
        opaque = scene.add_mesh(
            ((-1, -1, 0), (1, -1, 0), (0, 1, 0)), ((0, 1, 2),),
            name="opaque",
        )
        camera = ol.PerspectiveCamera((0, 0, 3), (0, 0, 0))
        surface = ol.pick(scene, camera, (101, 101), (50, 50))
        through = ol.pick(
            scene, camera, (101, 101), (50, 50),
            options=ol.PickOptions(transmissive="through"),
        )
        self.assertIs(surface.object, glass)
        self.assertIs(through.object, opaque)
        self.assertIsNone(ol.pick(
            scene, camera, (101, 101), (50, 50),
            options=ol.PickOptions(volumes="only"),
        ))

    def test_pick_options_are_strictly_validated(self):
        with self.assertRaises(ValueError):
            ol.PickOptions(transmissive="alpha")
        with self.assertRaises(ValueError):
            ol.PickOptions(volumes="maybe")
        with self.assertRaises(ValueError):
            ol.PickOptions(transmission_threshold=2)

    def test_center_pixel_picks_nearest_named_object(self):
        scene = ol.Scene()
        near = scene.add_mesh(
            ((-1, -1, 0), (1, -1, 0), (0, 1, 0)), ((0, 1, 2),),
            name="near",
        )
        scene.add_mesh(
            ((-1, -1, -2), (1, -1, -2), (0, 1, -2)), ((0, 1, 2),),
            name="far",
        )
        camera = ol.PerspectiveCamera((0, 0, 3), (0, 0, 0))
        result = ol.pick(scene, camera, (101, 101), (50, 50))
        self.assertIsNotNone(result)
        self.assertIs(result.object, near)
        self.assertEqual(result.object_id, near.id)
        self.assertEqual(result.triangle_index, 0)
        self.assertAlmostEqual(sum(result.barycentric), 1.0)

    def test_empty_pixel_clears_selection(self):
        scene = ol.Scene()
        scene.add_mesh(
            ((-0.1, -0.1, 0), (0.1, -0.1, 0), (0, 0.1, 0)),
            ((0, 1, 2),),
        )
        camera = ol.PerspectiveCamera((0, 0, 3), (0, 0, 0))
        self.assertIsNone(ol.pick(scene, camera, (100, 100), (0, 0)))

    def test_object_triangle_ranges_follow_packed_visible_order(self):
        scene = ol.Scene()
        first = scene.add_mesh(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0)), ((0, 1, 2),),
        )
        second = scene.add_mesh(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)),
            ((0, 1, 2), (1, 3, 2)),
        )
        self.assertEqual(scene.object_triangle_range(first), (0, 1))
        self.assertEqual(scene.object_triangle_range(second.id), (1, 3))
        first.visible = False
        self.assertEqual(scene.object_triangle_range(second), (0, 2))
        with self.assertRaises(KeyError):
            scene.object_triangle_range(first)

    def test_orthographic_camera_ray_shifts_origin(self):
        camera = ol.OrthographicCamera(
            (0, 0, 3), (0, 0, 0), vertical_size=2.0,
        )
        center_origin, center_direction = ol.camera_ray(
            camera, (100, 100), (49.5, 49.5)
        )
        right_origin, right_direction = ol.camera_ray(
            camera, (100, 100), (99.5, 49.5)
        )
        np.testing.assert_allclose(center_origin, (0, 0, 3), atol=1e-6)
        self.assertGreater(right_origin[0], center_origin[0])
        np.testing.assert_allclose(center_direction, right_direction)


if __name__ == "__main__":
    unittest.main()
