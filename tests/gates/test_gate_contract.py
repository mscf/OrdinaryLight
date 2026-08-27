"""Fast structural checks for the executable gate package."""

from importlib import import_module
from pathlib import Path
import unittest


PYTHON_GATES = (
    "execution_parity",
    "indirect_quality",
    "noise_quality",
    "path_termination_quality",
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


if __name__ == "__main__":
    unittest.main()
