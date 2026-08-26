import unittest

from ordinarylight.integrations.dynamic_resolution import (
    DynamicResolutionController,
)


class DynamicResolutionControllerTests(unittest.TestCase):
    def test_reduces_and_recovers_scale_with_bounded_steps(self):
        controller = DynamicResolutionController(
            target_ms=16.0, minimum_scale=0.5, current_scale=1.0,
            smoothing=1.0, update_interval=1, maximum_step=0.125,
            recovery_smoothing=1.0, recovery_updates=1,
            quantization=1.0 / 16.0,
        )
        self.assertEqual(controller.update(64.0), 0.875)
        self.assertEqual(controller.update(64.0), 0.75)
        self.assertEqual(controller.update(8.0), 0.875)

    def test_hysteresis_and_update_interval_prevent_churn(self):
        controller = DynamicResolutionController(
            target_ms=16.0, current_scale=0.75, hysteresis=0.1,
            smoothing=1.0, update_interval=2,
        )
        self.assertEqual(controller.update(30.0), 0.75)
        reduced = controller.update(30.0)
        self.assertLess(reduced, 0.75)
        self.assertEqual(controller.update(16.5), reduced)
        self.assertEqual(controller.update(16.5), reduced)

    def test_rejects_invalid_configuration_and_samples(self):
        with self.assertRaises(ValueError):
            DynamicResolutionController(minimum_scale=0.2)
        controller = DynamicResolutionController(current_scale=0.75)
        self.assertEqual(controller.update(0.0), 0.75)


if __name__ == "__main__":
    unittest.main()
