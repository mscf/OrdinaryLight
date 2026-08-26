import unittest

import numpy as np

import ordinarylight as ol


class ValidationTests(unittest.TestCase):
    def test_feature_parity_scene_exercises_required_materials(self):
        scene = ol.build_feature_parity_scene()
        materials = [mesh.material for mesh in scene.meshes]
        self.assertTrue(any(material.transmission == 1.0 for material in materials))
        self.assertGreaterEqual(
            sum(material.transmission == 1.0 for material in materials), 3
        )
        self.assertTrue(any(material.metallic == 1.0 for material in materials))
        self.assertTrue(any(any(material.emission) for material in materials))
        self.assertTrue(any(material.base_color_texture for material in materials))
        self.assertTrue(any(
            material.metallic_roughness_texture for material in materials
        ))
        self.assertTrue(any(material.emissive_texture for material in materials))
        self.assertTrue(any(material.normal_texture for material in materials))
        self.assertTrue(any(material.occlusion_texture for material in materials))
        self.assertTrue(any(
            material.base_color_transform != ol.TextureTransform()
            for material in materials
        ))
        self.assertTrue(any(
            material.occlusion_transform.texcoord_set == 1
            for material in materials
        ))

    def test_image_error_metrics(self):
        reference = np.ones((2, 2, 4), dtype=np.float32)
        candidate = reference.copy()
        self.assertEqual(
            ol.image_error_metrics(reference, candidate)["relative_rmse"], 0.0
        )
        candidate[..., :3] += 0.5
        metrics = ol.image_error_metrics(reference, candidate)
        self.assertAlmostEqual(metrics["mae"], 0.5)
        self.assertAlmostEqual(metrics["rmse"], 0.5)
        self.assertAlmostEqual(metrics["relative_rmse"], 0.5)

    def test_image_error_rejects_mismatched_shapes(self):
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            ol.image_error_metrics(np.zeros((2, 2, 3)), np.zeros((3, 2, 3)))


if __name__ == "__main__":
    unittest.main()
