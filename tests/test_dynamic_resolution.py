import unittest

from ordinarylight.integrations.dynamic_resolution import (
    DynamicResolutionController,
)
from ordinarylight.integrations.dynamic_sampling import DynamicSampleController


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

    def test_resolution_normalizes_changing_sample_work(self):
        controller = DynamicResolutionController(
            target_ms=16.0, minimum_scale=0.5, current_scale=1.0,
            smoothing=1.0, recovery_smoothing=1.0,
            update_interval=1, recovery_updates=1,
        )
        self.assertEqual(
            controller.update(
                32.0, 1.0, work_units=2, target_work_units=1
            ),
            1.0,
        )


class DynamicSampleControllerTests(unittest.TestCase):
    def test_raises_samples_to_use_available_motion_budget(self):
        controller = DynamicSampleController(
            target_ms=16.0, minimum_samples=1, maximum_samples=4,
            current_samples=1, smoothing=1.0, recovery_smoothing=1.0,
            update_interval=1,
        )
        self.assertEqual(controller.update(4.0, 1), 2)
        self.assertEqual(controller.update(8.0, 2), 3)
        self.assertEqual(controller.update(12.0, 3), 3)

    def test_drops_samples_when_over_budget(self):
        controller = DynamicSampleController(
            target_ms=16.0, minimum_samples=1, maximum_samples=8,
            current_samples=8, smoothing=1.0, recovery_smoothing=1.0,
            update_interval=1,
        )
        self.assertEqual(controller.update(32.0, 8), 3)

    def test_resolution_recovery_takes_priority_over_extra_samples(self):
        controller = DynamicSampleController(
            target_ms=16.0, minimum_samples=1, maximum_samples=8,
            current_samples=4, update_interval=1,
        )
        self.assertEqual(
            controller.update(8.0, 4, 0.5, allow_increase=False), 1
        )

    def test_rejects_invalid_sample_range(self):
        with self.assertRaises(ValueError):
            DynamicSampleController(minimum_samples=2, maximum_samples=1)


if __name__ == "__main__":
    unittest.main()
