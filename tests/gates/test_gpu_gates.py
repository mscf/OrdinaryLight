"""Opt-in integration of expensive Vulkan gates with unittest discovery."""

import os
from pathlib import Path
import subprocess
import sys
import unittest
import json


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

    def _run_optical_pose_matrix(self, scene, filename, object_prefix):
        poses = json.loads(
            Path("tests/gates/poses", filename).read_text()
        )
        for entry in poses:
            pose = dict(entry)
            name = pose.pop("name")
            thresholds = pose.pop("thresholds")
            arguments = [
                "tests.gates.renderer_visual_parity", "--scene", scene,
                "--raster-optics", "screen-space",
                "--camera-pose", json.dumps(pose),
                "--object-prefix", object_prefix,
                "--output", f"/tmp/ordinarylight_{scene}_{name}_parity",
            ]
            option_names = {
                "max_log_color_rmse": "--max-log-color-rmse",
                "max_object_log_luminance_error":
                    "--max-object-log-luminance-error",
                "min_edge_correlation": "--min-edge-correlation",
                "min_coverage_iou": "--min-coverage-iou",
                "min_object_edge_correlation":
                    "--min-object-edge-correlation",
            }
            for key, option in option_names.items():
                if key in thresholds:
                    arguments.extend((option, str(thresholds[key])))
            self._run(*arguments)

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

    def test_native_raster_lighting_and_shadows(self):
        for target in ("vulkan", "webgpu"):
            self._run(
                "tests.gates.raster_lighting", "--target", target,
                "--output", f"/tmp/ordinarylight_raster_lighting_{target}",
            )

    def test_raster_gi_visual_parity(self):
        self._run("tests.gates.renderer_visual_parity")
        self._run(
            "tests.gates.renderer_visual_parity", "--scene", "modifier",
            "--output", "/tmp/ordinarylight_surface_modifier_parity",
        )
        for scene in (
            "clearcoat", "sheen", "anisotropy", "thin-transmission",
            "subsurface",
        ):
            self._run(
                "tests.gates.renderer_visual_parity", "--scene", scene,
                "--output", f"/tmp/ordinarylight_{scene}_parity",
            )
        for scene in (
            "environment-reflection", "refraction", "absorption",
            "nested-dielectric", "transparency",
        ):
            self._run(
                "tests.gates.renderer_visual_parity", "--scene", scene,
                "--output", f"/tmp/ordinarylight_{scene}_parity",
            )

    def test_fixed_pose_optical_visual_parity(self):
        # Keep this independent of the broad/default-pose matrix so a failure
        # in another feature cannot prevent these regression-critical camera
        # poses from running and producing their own evidence.
        self._run_optical_pose_matrix(
            "refraction", "refraction_parity.json", "refraction-",
        )
        self._run_optical_pose_matrix(
            "nested-dielectric", "nested_dielectric_parity.json", "outer-",
        )
        self._run_optical_pose_matrix(
            "thin-transmission", "thin_transmission_parity.json", "material-",
        )

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
