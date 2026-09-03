import copy
from types import SimpleNamespace
import unittest

import numpy as np

from tests.gates.noise_quality import (
    evaluate_against_baseline,
    make_baseline,
    _renderer_config,
    summarize_structural_edges,
)


class NoiseQualityGateTests(unittest.TestCase):
    def setUp(self):
        self.configuration = {"width": 320, "scenarios": [{"name": "diffuse"}]}
        self.summary = {
            "diffuse": {
                "bias_mean": 0.001,
                "structural_edge_error_mean": 0.10,
                "structural_edge_gain_mean": 0.98,
                "horizontal_band_rms_p95": 0.01,
                "low_frequency_energy_ratio_mean": 1.0,
                "positive_outlier_p999_p95": 0.20,
                "relative_rmse_mean": 0.30,
                "relative_rmse_p95": 0.34,
                "temporal_residual_rmse_mean": 0.08,
                "temporal_residual_rmse_p95": 0.10,
                "vertical_band_rms_p95": 0.01,
            }
        }
        self.timings = {"diffuse": 1.0}
        self.baseline = make_baseline(
            self.configuration, self.summary, reason="initial accepted result",
            timings=self.timings,
        )

    def test_identical_and_improved_results_pass(self):
        self.assertEqual(evaluate_against_baseline(
            self.configuration, self.summary, self.baseline,
            timings=self.timings,
        ), [])
        improved = copy.deepcopy(self.summary)
        improved["diffuse"]["relative_rmse_mean"] *= 0.5
        improved["diffuse"]["structural_edge_gain_mean"] = 1.0
        self.assertEqual(evaluate_against_baseline(
            self.configuration, improved, self.baseline,
            timings=self.timings,
        ), [])

    def test_noise_regression_fails(self):
        regressed = copy.deepcopy(self.summary)
        regressed["diffuse"]["relative_rmse_mean"] = 0.5
        failures = evaluate_against_baseline(
            self.configuration, regressed, self.baseline
        )
        self.assertTrue(any("relative_rmse_mean" in item for item in failures))

    def test_lost_edge_contrast_fails(self):
        blurred = copy.deepcopy(self.summary)
        blurred["diffuse"]["structural_edge_gain_mean"] = 0.7
        failures = evaluate_against_baseline(
            self.configuration, blurred, self.baseline
        )
        self.assertTrue(any("structural_edge_gain_mean" in item for item in failures))

    def test_structural_edge_metric_detects_blur(self):
        reference = np.zeros((1, 48, 64, 3), dtype=np.float32)
        reference[:, :, 32:] = 1.0
        identical = summarize_structural_edges(reference, reference)
        blurred = reference.copy()
        for column, value in ((29, 0.2), (30, 0.35), (31, 0.45),
                              (32, 0.55), (33, 0.65), (34, 0.8)):
            blurred[:, :, column] = value
        softened = summarize_structural_edges(reference, blurred)
        self.assertAlmostEqual(
            identical["structural_edge_gain_mean"], 1.0, places=5
        )
        self.assertLess(softened["structural_edge_gain_mean"], 0.9)
        self.assertGreater(softened["structural_edge_error_mean"], 0.1)

    def test_structural_edge_metric_suppresses_sample_scale_noise(self):
        reference = np.zeros((1, 48, 64, 3), dtype=np.float32)
        reference[:, :, 32:] = 1.0
        checker = (np.indices((48, 64)).sum(axis=0) % 2) * 2.0 - 1.0
        noisy = reference + (0.1 * checker)[None, :, :, None]
        result = summarize_structural_edges(reference, noisy)
        self.assertAlmostEqual(result["structural_edge_gain_mean"], 1.0, delta=0.03)
        self.assertLess(result["structural_edge_error_mean"], 0.05)

    def test_configuration_change_requires_reacceptance(self):
        failures = evaluate_against_baseline(
            {"width": 640}, self.summary, self.baseline
        )
        self.assertIn("capture configuration differs", failures[0])

    def test_baseline_records_override_reason(self):
        self.assertEqual(
            self.baseline["override_reason"], "initial accepted result"
        )

    def test_gpu_time_regression_fails(self):
        failures = evaluate_against_baseline(
            self.configuration, self.summary, self.baseline,
            timings={"diffuse": 1.5},
        )
        self.assertTrue(any("median_gpu_ms" in item for item in failures))

    def test_candidate_uses_relax_while_reference_stays_independent(self):
        args = SimpleNamespace(
            bounces=8, reference_samples=16, candidate_samples=1,
            area_light_samples=2, history_limit=32, atrous_iterations=3,
            width=320, height=180,
        )
        reference = _renderer_config(args, reference=True)
        candidate = _renderer_config(args, reference=False)
        self.assertFalse(reference.denoiser_enabled)
        self.assertFalse(reference.temporal_history)
        self.assertTrue(candidate.denoiser_enabled)
        self.assertTrue(candidate.temporal_history)
        self.assertTrue(candidate.wavefront_restir_di)
        self.assertEqual(candidate.samples_per_pixel, 1)


if __name__ == "__main__":
    unittest.main()
