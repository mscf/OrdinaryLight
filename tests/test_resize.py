import unittest

from ordinarylight.integrations.resize import ResizeRecreationGate


class ResizeRecreationGateTests(unittest.TestCase):
    def test_wayland_maximize_bounce_recreates_only_after_final_extent_settles(self):
        gate = ResizeRecreationGate(0.15)

        self.assertTrue(gate.should_render(
            (1600, 900), 0.0, resources_allocated=False
        ))
        self.assertTrue(gate.should_render(
            (1600, 900), 0.005, resources_allocated=True
        ))
        self.assertFalse(gate.should_render(
            (3840, 2130), 0.01, resources_allocated=True
        ))
        self.assertFalse(gate.should_render(
            (1600, 900), 0.05, resources_allocated=True
        ))
        self.assertFalse(gate.should_render(
            (3840, 2130), 0.09, resources_allocated=True
        ))
        self.assertFalse(gate.should_render(
            (3840, 2130), 0.239, resources_allocated=True
        ))
        self.assertTrue(gate.should_render(
            (3840, 2130), 0.24, resources_allocated=True
        ))
        self.assertEqual(gate.pending_extent, (3840, 2130))

    def test_zero_delay_preserves_immediate_resize_behavior(self):
        gate = ResizeRecreationGate(0.0)
        gate.should_render((800, 600), 0.0, resources_allocated=False)
        self.assertTrue(gate.should_render(
            (1920, 1080), 0.01, resources_allocated=True
        ))

    def test_rejects_invalid_configuration_and_extent(self):
        with self.assertRaises(ValueError):
            ResizeRecreationGate(-0.1)
        gate = ResizeRecreationGate()
        with self.assertRaises(ValueError):
            gate.should_render((0, 720), 0.0, resources_allocated=False)


if __name__ == "__main__":
    unittest.main()
