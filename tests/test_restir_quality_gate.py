import unittest
from types import SimpleNamespace

from tests.gates.restir_quality import _evaluate_gate, _mean_gpu_ms


class RestirQualityGateTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(
            gate_max_abs_bias=0.01,
            gate_max_error_ratio=1.2,
            gate_max_mae_ratio=1.02,
            gate_max_pairwise_gpu_ratio=1.25,
            gate_max_generalized_gpu_ratio=1.5,
            gate_quality_only=False,
        )
        self.summaries = {
            "conventional": {
                "relative_rmse_mean": 0.30, "mae_mean": 0.13,
                "bias_mean": 0.001,
            },
            "canonical": {
                "relative_rmse_mean": 0.33, "mae_mean": 0.125,
                "bias_mean": 0.001,
            },
            "pairwise": {
                "relative_rmse_mean": 0.329, "mae_mean": 0.124,
                "bias_mean": 0.0005,
            },
            "generalized": {
                "relative_rmse_mean": 0.328, "mae_mean": 0.124,
                "bias_mean": 0.0006,
            },
        }
        self.gpu = {
            "conventional": 0.20, "canonical": 0.24,
            "pairwise": 0.25, "generalized": 0.32,
        }

    def test_accepts_strategies_within_limits(self):
        self.assertEqual(
            _evaluate_gate(self.summaries, self.gpu, self.args), []
        )

    def test_reports_quality_and_performance_regressions(self):
        self.summaries["pairwise"]["bias_mean"] = 0.02
        self.gpu["generalized"] = 0.5
        failures = _evaluate_gate(self.summaries, self.gpu, self.args)
        self.assertTrue(any("pairwise: |bias_mean|" in item for item in failures))
        self.assertTrue(any("generalized: canonical GPU ratio" in item
                            for item in failures))

    def test_gpu_mean_ignores_unavailable_warmup_samples(self):
        self.assertEqual(_mean_gpu_ms([
            {"gpu_ms": 0.0}, {"gpu_ms": 2.0}, {"gpu_ms": 4.0},
        ]), 3.0)

    def test_quality_only_gate_records_but_does_not_enforce_gpu_ratio(self):
        self.args.gate_quality_only = True
        self.gpu["generalized"] = 100.0
        self.assertEqual(
            _evaluate_gate(self.summaries, self.gpu, self.args), []
        )


if __name__ == "__main__":
    unittest.main()
