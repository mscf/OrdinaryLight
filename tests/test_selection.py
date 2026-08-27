import unittest

import numpy as np

import ordinarylight as ol


class SelectionTests(unittest.TestCase):
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
