import math
import unittest

import numpy as np

import ordinarylight as ol


class ArcballCameraControllerTests(unittest.TestCase):
    def test_preserves_authored_camera(self):
        source = ol.PerspectiveCamera((3.0, 2.0, 4.0), (0.5, 0.25, -1.0))
        camera = ol.ArcballCameraController.from_camera(source).camera()
        np.testing.assert_allclose(camera.position, source.position, atol=1e-12)
        np.testing.assert_allclose(camera.target, source.target, atol=1e-12)

    def test_orbit_and_dolly(self):
        controller = ol.ArcballCameraController(distance=10.0)
        controller.orbit(math.pi * 0.5, 0.0).dolly(1.0)
        camera = controller.camera()
        self.assertGreater(camera.position[0], 0.0)
        self.assertAlmostEqual(camera.position[2], 0.0, places=10)
        self.assertLess(controller.distance, 10.0)

    def test_pan_moves_position_and_target_together(self):
        controller = ol.ArcballCameraController(distance=5.0)
        before = controller.camera()
        controller.pan(0.1, -0.2)
        after = controller.camera()
        np.testing.assert_allclose(
            np.asarray(after.position) - np.asarray(before.position),
            np.asarray(after.target) - np.asarray(before.target),
        )

    def test_elevation_and_distance_are_clamped(self):
        controller = ol.ArcballCameraController(
            distance=2.0, minimum_distance=1.0, maximum_distance=3.0,
        )
        controller.orbit(0.0, 100.0).dolly(1000.0)
        self.assertLess(controller.elevation, math.pi * 0.5)
        self.assertEqual(controller.distance, 1.0)


if __name__ == "__main__":
    unittest.main()
