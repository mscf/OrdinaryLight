import copy
from types import SimpleNamespace
import unittest

from tests.gates.noise_quality import (
    evaluate_against_baseline,
    make_baseline,
    _renderer_config,
)


class NoiseQualityGateTests(unittest.TestCase):
    def setUp(self):
        self.configuration = {"width": 320, "scenarios": [{"name": "diffuse"}]}
        self.summary = {
            "diffuse": {
                "bias_mean": 0.001,
                "edge_gradient_error_mean": 0.10,
                "edge_gradient_gain_mean": 0.98,
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
        self.baseline = make_baseline(
            self.configuration, self.summary, reason="initial accepted result"
        )

    def test_identical_and_improved_results_pass(self):
        self.assertEqual(evaluate_against_baseline(
            self.configuration, self.summary, self.baseline
        ), [])
        improved = copy.deepcopy(self.summary)
        improved["diffuse"]["relative_rmse_mean"] *= 0.5
        improved["diffuse"]["edge_gradient_gain_mean"] = 1.0
        self.assertEqual(evaluate_against_baseline(
            self.configuration, improved, self.baseline
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
        blurred["diffuse"]["edge_gradient_gain_mean"] = 0.7
        failures = evaluate_against_baseline(
            self.configuration, blurred, self.baseline
        )
        self.assertTrue(any("edge_gradient_gain_mean" in item for item in failures))

    def test_configuration_change_requires_reacceptance(self):
        failures = evaluate_against_baseline(
            {"width": 640}, self.summary, self.baseline
        )
        self.assertIn("capture configuration differs", failures[0])

    def test_baseline_records_override_reason(self):
        self.assertEqual(
            self.baseline["override_reason"], "initial accepted result"
        )

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
