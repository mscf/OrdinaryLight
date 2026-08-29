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

    def test_cross_renderer_metrics_separate_exposure_and_structure(self):
        reference = np.zeros((8, 8, 3), np.float32)
        reference[2:6, 2:6] = (0.8, 0.4, 0.2)
        metrics = ol.renderer_visual_metrics(reference, reference * 0.5)
        self.assertAlmostEqual(metrics["exposure_scale"], 2.0, places=5)
        self.assertLess(metrics["log_color_rmse"], 1e-6)
        self.assertGreater(metrics["edge_correlation"], 0.999)
        self.assertGreater(metrics["coverage_iou"], 0.999)
        shifted = ol.renderer_visual_metrics(
            reference, np.roll(reference * 0.5, 2, axis=1),
        )
        self.assertLess(shifted["edge_correlation"], metrics["edge_correlation"])
        mask = np.zeros((8, 8), bool)
        mask[2:6, 2:6] = True
        masked = ol.renderer_visual_metrics(
            reference, reference * 0.5,
            reference_mask=mask, candidate_mask=mask,
        )
        self.assertEqual(masked["coverage_iou"], 1.0)


if __name__ == "__main__":
    unittest.main()
