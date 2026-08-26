import math
import unittest

import numpy as np

import ordinarylight as ol
from ordinarylight.scene import PerspectiveCamera as LegacyCamera
from ordinarylight.scene import PointLight as LegacyPointLight


class SemanticResourceTests(unittest.TestCase):
    def test_canonical_namespaces_retain_compatibility_aliases(self):
        self.assertIs(ol.PerspectiveCamera, ol.cameras.PerspectiveCamera)
        self.assertIs(ol.OrthographicCamera, ol.cameras.OrthographicCamera)
        self.assertIs(ol.PanoramicCamera, ol.cameras.PanoramicCamera)
        self.assertIs(LegacyCamera, ol.cameras.PerspectiveCamera)
        self.assertIs(ol.PointLight, ol.lights.PointLight)
        self.assertIs(LegacyPointLight, ol.lights.PointLight)
        self.assertIs(ol.DirectionalLight, ol.lights.DirectionalLight)
        self.assertIs(ol.SpotLight, ol.lights.SpotLight)

    def test_camera_models_validate_projection_parameters(self):
        with self.assertRaises(ValueError):
            ol.OrthographicCamera((0, 0, 1), (0, 0, 0), vertical_size=0)
        with self.assertRaises(ValueError):
            ol.PanoramicCamera(
                (0, 0, 1), (0, 0, 0), horizontal_fov_degrees=361
            )
        with self.assertRaises(ValueError):
            ol.PerspectiveCamera((0, 0, 1), (0, 0, 0), up=(0, 0, 1))

    def test_reference_camera_models_generate_distinct_rays(self):
        rng = np.random.default_rng(4)
        perspective = ol.ReferencePathTracer._camera_rays(
            ol.PerspectiveCamera((0, 0, 2), (0, 0, 0)), 4, 2, rng
        )
        orthographic = ol.ReferencePathTracer._camera_rays(
            ol.OrthographicCamera((0, 0, 2), (0, 0, 0)), 4, 2,
            np.random.default_rng(4),
        )
        panoramic = ol.ReferencePathTracer._camera_rays(
            ol.PanoramicCamera((0, 0, 2), (0, 0, 0)), 4, 2,
            np.random.default_rng(4),
        )
        self.assertTrue(np.allclose(perspective[0], perspective[0][0]))
        self.assertFalse(np.allclose(orthographic[0], orthographic[0][0]))
        self.assertTrue(np.allclose(orthographic[1], orthographic[1][0]))
        self.assertFalse(np.allclose(panoramic[1], panoramic[1][0]))

    def test_validates_direction_and_spot_cones(self):
        with self.assertRaises(ValueError):
            ol.DirectionalLight((0, 0, 0))
        with self.assertRaises(ValueError):
            ol.SpotLight((0, 0, 0), (0, -1, 0), outer_cone_angle=0)
        with self.assertRaises(ValueError):
            ol.SpotLight(
                (0, 0, 0), (0, -1, 0),
                inner_cone_angle=0.6, outer_cone_angle=0.5,
            )
        with self.assertRaises(ValueError):
            ol.PointLight((0, 0, 0), range=-1)

    def test_scene_packs_all_analytic_light_types(self):
        scene = ol.Scene()
        point = scene.add_point_light((1, 2, 3), intensity=4, range=5)
        directional = scene.add_directional_light((0, -2, 0), intensity=2)
        spot = scene.add_spot_light(
            (3, 2, 1), (0, 0, -4), intensity=7,
            inner_cone_angle=0.2, outer_cone_angle=0.5, range=9,
        )

        data = scene.analytic_light_data()
        self.assertEqual(data.shape, (3, 4, 4))
        np.testing.assert_allclose(data[0, 0], (1, 2, 3, 0))
        np.testing.assert_allclose(data[0, 1], (0, 0, 0, 5))
        np.testing.assert_allclose(data[1, 0], (0, 0, 0, 1))
        np.testing.assert_allclose(data[1, 1], (0, -1, 0, 0))
        np.testing.assert_allclose(data[2, 0], (3, 2, 1, 2))
        np.testing.assert_allclose(data[2, 1], (0, 0, -1, 9))
        np.testing.assert_allclose(
            data[2, 3, :2], (math.cos(0.2), math.cos(0.5))
        )
        self.assertEqual((point.id, directional.id, spot.id), (1, 2, 3))

    def test_typed_updates_and_snapshot(self):
        scene = ol.Scene()
        directional = scene.add_directional_light((0, -1, 0))
        spot = scene.add_spot_light((0, 2, 0), (0, -1, 0))
        revision = scene.shading_revision
        scene.update_directional_light(directional.id, intensity=3)
        scene.update_spot_light(spot.id, range=4, outer_cone_angle=0.6)
        self.assertEqual(scene.shading_revision, revision + 2)
        snapshot = scene.snapshot()
        self.assertEqual(snapshot["directional_lights"][0]["intensity"], 3)
        self.assertEqual(snapshot["spot_lights"][0]["range"], 4)
        self.assertIs(scene.remove_directional_light(directional.id), directional)
        self.assertIs(scene.remove_spot_light(spot.id), spot)

    def test_hdr_environment_is_unique_and_uses_scene_texture_transport(self):
        image = np.asarray((
            ((0.0, 1.0, 8.0), (2.0, 0.5, 0.0)),
            ((1.0, 1.0, 1.0), (16.0, 4.0, 0.25)),
        ), np.float32)
        scene = ol.Scene()
        environment = scene.set_environment(
            image=image, intensity=2.0, rotation=0.25
        )
        self.assertIsInstance(environment, ol.EnvironmentLight)
        self.assertEqual(scene.analytic_light_count, 1)
        self.assertEqual(len(scene.textures), 1)
        data = scene.analytic_light_data()
        self.assertEqual(data.shape, (1, 4, 4))
        self.assertEqual(data[0, 0, 3], 3)
        self.assertEqual(data[0, 3, 0], 0)
        self.assertGreater(data[0, 3, 2], 4)
        self.assertEqual(scene.snapshot()["environment"]["image_shape"], [2, 2, 3])
        scene.set_environment()
        self.assertEqual(scene.analytic_light_count, 0)
        self.assertEqual(scene.textures, ())


if __name__ == "__main__":
    unittest.main()
