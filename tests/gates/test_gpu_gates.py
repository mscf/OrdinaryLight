"""Opt-in integration of expensive Vulkan gates with unittest discovery."""

import os
from pathlib import Path
import subprocess
import sys
import unittest


RUN_GPU = os.environ.get("ORDINARYLIGHT_RUN_GPU_GATES", "").lower() in {
    "1", "true", "yes", "on",
}
RUN_PERFORMANCE = os.environ.get(
    "ORDINARYLIGHT_RUN_PERFORMANCE_GATES", ""
).lower() in {"1", "true", "yes", "on"}


@unittest.skipUnless(RUN_GPU, "set ORDINARYLIGHT_RUN_GPU_GATES=1")
class VulkanGateTests(unittest.TestCase):
    def _run(self, *arguments):
        subprocess.run(
            (sys.executable, "-m", *arguments),
            cwd=Path(__file__).parents[2], check=True,
        )

    def test_cross_scene_quality_and_parity(self):
        stages = ["indirect", "restir", "parity", "termination"]
        if RUN_PERFORMANCE:
            stages.append("performance")
        self._run(
            "tests.gates.validation_matrix",
            "--stages", *stages,
            "--output", "/tmp/ordinarylight_validation_matrix",
        )

    def test_accepted_noise_quality_baseline(self):
        self._run("tests.gates.noise_quality")

    def test_resident_transition_latency(self):
        self._run("tests.gates.transition_latency")

    def test_gpu_picking(self):
        self._run("tests.gates.gpu_picking")

    def test_raster_backend_parity(self):
        self._run("tests.gates.raster_parity")

    def test_raster_gi_visual_parity(self):
        self._run("tests.gates.renderer_visual_parity")

    def test_volume_compositing(self):
        self._run(
            "tests.gates.volume_compositing",
            "--summary", "/tmp/ordinarylight_volume_compositing.json",
        )

    def test_volume_scattering(self):
        self._run("tests.gates.volume_scattering")
        self._run("tests.gates.volume_multiple_scattering")


if __name__ == "__main__":
    unittest.main()
