import unittest

import numpy as np

from tests.gates.relax_motion_quality import (
    evaluate_against_baseline,
    evaluate_sequence,
)


class RelaxMotionQualityGateTests(unittest.TestCase):
    def test_pixel_evaluation_detects_temporal_ghost(self):
        reference = np.zeros((3, 8, 12, 4), np.float32)
        reference[..., :3] = 0.1
        reference[..., 3] = 1.0
        reference[0, 2:6, 1:4, :3] = 2.0
        reference[1, 2:6, 4:7, :3] = 2.0
        reference[2, 2:6, 7:10, :3] = 2.0
        clean = evaluate_sequence(reference, reference.copy())
        ghosted = reference.copy()
        ghosted[1, 2:6, 1:4, :3] = 1.0
        ghosted[2, 2:6, 4:7, :3] = 1.0
        ghosted[1, 0, 0, :3] = 0.5
        ghosted[2, 0, 0, :3] = 1.0
        dirty = evaluate_sequence(reference, ghosted)
        self.assertEqual(clean["motion_region_rmse"], 0.0)
        self.assertGreater(dirty["motion_region_rmse"], 0.05)
        self.assertGreater(dirty["stationary_residual_rmse"], 0.0)

    def test_baseline_evaluation_catches_quality_and_timing_regression(self):
        metrics = {
            "log_luminance_rmse": 0.1,
            "motion_region_rmse": 0.2,
            "stationary_residual_rmse": 0.05,
            "edge_correlation": 0.8,
            "median_gpu_ms": 2.0,
        }
        baseline = {
            "schema": 1, "configuration": {"fixture": 1},
            "accepted": {"object": metrics},
        }
        self.assertFalse(evaluate_against_baseline(
            {"fixture": 1}, {"object": metrics}, baseline,
        ))
        regressed = dict(metrics, motion_region_rmse=0.4, median_gpu_ms=4.0)
        failures = evaluate_against_baseline(
            {"fixture": 1}, {"object": regressed}, baseline,
        )
        self.assertTrue(any("motion_region_rmse" in item for item in failures))
        self.assertTrue(any("median_gpu_ms" in item for item in failures))


if __name__ == "__main__":
    unittest.main()
