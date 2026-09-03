"""Fast structural checks for the executable gate package."""

from importlib import import_module
from pathlib import Path
import unittest


PYTHON_GATES = (
    "execution_parity",
    "indirect_quality",
    "noise_quality",
    "path_termination_quality",
    "raster_lighting",
    "relax_motion_quality",
    "relax_tessellation_quality",
    "restir_matrix",
    "restir_quality",
    "ser_quality",
    "temporal_quality",
    "transition_latency",
    "tile_quality",
    "validation_matrix",
    "volume_compositing",
    "volume_empty_space",
    "volume_multiple_scattering",
    "volume_raster_parity",
    "volume_scattering",
)

SHELL_GATES = (
    "run_4k_instancing.sh",
    "run_4k_performance.sh",
    "run_4k_primitives.sh",
    "run_volume.sh",
    "run_volume_compositing.sh",
    "run_volume_empty_space.sh",
)


class GateContractTests(unittest.TestCase):
    def test_python_gates_are_importable_modules_with_main(self):
        for name in PYTHON_GATES:
            with self.subTest(name=name):
                module = import_module(f"tests.gates.{name}")
                self.assertTrue(callable(getattr(module, "main", None)))

    def test_shell_launchers_are_colocated_and_executable(self):
        directory = Path(__file__).parent
        for name in SHELL_GATES:
            with self.subTest(name=name):
                path = directory / name
                self.assertTrue(path.is_file())
                self.assertTrue(path.stat().st_mode & 0o100)

    def test_raster_lighting_baseline_is_checked_in(self):
        path = Path(__file__).parent / "baselines" / "raster_lighting.json"
        self.assertTrue(path.is_file())
        self.assertIn("point", path.read_text())

    def test_relax_motion_baseline_is_checked_in(self):
        path = Path(__file__).parent / "baselines" / "relax_motion_quality.json"
        self.assertTrue(path.is_file())
        payload = __import__("json").loads(path.read_text())
        self.assertEqual(set(payload["accepted"]), {"object", "camera"})

    def test_noise_baseline_uses_relax(self):
        path = Path(__file__).parent / "baselines" / "noise_quality.json"
        payload = __import__("json").loads(path.read_text())
        self.assertEqual(payload["schema"], 5)
        self.assertEqual(payload["configuration"]["candidate_samples"], 1)
        self.assertIn("atrous_iterations", payload["configuration"])
        self.assertEqual(
            set(payload["accepted_timings"]), set(payload["accepted"])
        )

    def test_relax_tessellation_baseline_is_checked_in(self):
        path = (
            Path(__file__).parent / "baselines"
            / "relax_tessellation_quality.json"
        )
        payload = __import__("json").loads(path.read_text())
        self.assertGreater(payload["masked_pixels"], 1000)
        self.assertIn("triangle_edge_dark_excess_p95", payload)


if __name__ == "__main__":
    unittest.main()
