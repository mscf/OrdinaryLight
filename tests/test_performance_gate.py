import unittest

from ordinarylight.validation import performance_gate_result


class PerformanceGateTests(unittest.TestCase):
    def test_passes_at_threshold_and_near_4k_extent(self):
        result = performance_gate_result(
            50.0, 50.0, (3840, 2130), (3840, 2160), 0.98,
        )
        self.assertEqual(result["status"], "pass")
        self.assertGreater(result["pixel_ratio"], 0.98)

    def test_fails_slow_or_undersized_run(self):
        result = performance_gate_result(
            49.9, 50.0, (3072, 1728), (3840, 2160), 0.98,
        )
        self.assertEqual(result["status"], "fail")
        self.assertEqual(len(result["failures"]), 2)

    def test_explicit_reason_can_override_failure(self):
        result = performance_gate_result(
            48.0, 50.0, (3840, 2160), (3840, 2160), 0.98,
            allow_failure=True, override_reason="accepted quality tradeoff",
        )
        self.assertEqual(result["status"], "override")
        self.assertEqual(result["override_reason"], "accepted quality tradeoff")

    def test_empty_reason_does_not_override(self):
        result = performance_gate_result(
            48.0, 50.0, (3840, 2160), (3840, 2160), 0.98,
            allow_failure=True, override_reason=" ",
        )
        self.assertEqual(result["status"], "fail")


if __name__ == "__main__":
    unittest.main()
