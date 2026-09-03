"""Guard the accepted multi-scene ReLAX denoising quality baseline."""

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path

import numpy as np

import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.integrations.temporal_quality import (
    summarize_temporal_quality,
    write_temporal_quality_csv,
)
from ordinarylight.showcases.rooms import get_restir_scene
from ordinarylight.showcases.volumes import build_volume_showcase


BASELINE_SCHEMA = 2
DEFAULT_BASELINE = Path(__file__).with_name("baselines") / "noise_quality.json"
OVERRIDE_REASON = "ORDINARYLIGHT_NOISE_GATE_OVERRIDE_REASON"

# Max policies allow a relative margin plus a small absolute floor so a metric
# whose accepted value is nearly zero does not become numerically impossible.
MAX_POLICIES = {
    "relative_rmse_mean": (1.05, 0.002),
    "relative_rmse_p95": (1.08, 0.003),
    "temporal_residual_rmse_mean": (1.08, 0.002),
    "temporal_residual_rmse_p95": (1.10, 0.003),
    "low_frequency_energy_ratio_mean": (1.10, 0.05),
    "horizontal_band_rms_p95": (1.10, 0.001),
    "vertical_band_rms_p95": (1.10, 0.001),
    "positive_outlier_p999_p95": (1.15, 0.01),
    "edge_gradient_error_mean": (1.08, 0.005),
}
MIN_POLICIES = {
    # An absolute margin is more meaningful than a ratio around the ideal
    # signed edge-gradient gain of one.
    "edge_gradient_gain_mean": 0.05,
}
BIAS_MARGIN = 0.002


@dataclass(frozen=True)
class NoiseScenario:
    name: str
    scene_name: str
    motion_radians: float
    orbit_radius: float
    camera_height: float
    target: tuple[float, float, float]

    def build(self):
        if self.scene_name == "volume":
            return build_volume_showcase(resolution=64)
        return get_restir_scene(self.scene_name).build()


SCENARIOS = (
    NoiseScenario("diffuse", "diffuse", 0.0, 8.5, 3.2, (0.0, 1.25, 0.0)),
    NoiseScenario(
        "area_light", "area_lights", 0.12, 8.5, 3.2, (0.0, 1.25, 0.0)
    ),
    NoiseScenario(
        "glossy_glass", "glossy_glass", 0.16, 8.5, 3.2,
        (0.0, 1.25, 0.0),
    ),
    NoiseScenario("dense_motion", "dense", 0.40, 8.5, 3.2, (0.0, 1.25, 0.0)),
    NoiseScenario("volume", "volume", 0.12, 8.8, 3.5, (0.0, 1.5, 0.0)),
)


def _configuration(args):
    configuration = {
        "width": args.width,
        "height": args.height,
        "frames": args.frames,
        "reference_samples": args.reference_samples,
        "candidate_samples": args.candidate_samples,
        "max_bounces": args.bounces,
        "area_light_samples": args.area_light_samples,
        "atrous_iterations": args.atrous_iterations,
        "history_limit": args.history_limit,
        "scenarios": [asdict(scenario) for scenario in SCENARIOS],
    }
    # Use the same list/object representation before and after a JSON
    # round-trip so configuration identity is stable at gate time.
    return json.loads(json.dumps(configuration))


def _camera(scenario, index, frames):
    fraction = index / max(frames - 1, 1)
    angle = math.pi + scenario.motion_radians * (fraction - 0.5)
    return ol.PerspectiveCamera(
        position=(
            scenario.orbit_radius * math.sin(angle),
            scenario.camera_height,
            scenario.orbit_radius * math.cos(angle),
        ),
        target=scenario.target,
    )


def _renderer_config(args, *, reference):
    denoise = not reference
    return ol.RendererConfig(
        max_bounces=args.bounces,
        samples_per_pixel=(
            args.reference_samples if reference else args.candidate_samples
        ),
        area_light_samples=args.area_light_samples,
        wavefront_secondary_area_light_samples=1,
        wavefront_environment_samples=1,
        wavefront_restir_di=denoise,
        wavefront_restir_reservoirs=2,
        wavefront_restir_candidates=4,
        wavefront_restir_history_limit=4,
        progressive_accumulation=denoise,
        temporal_history=denoise,
        temporal_history_limit=args.history_limit,
        denoiser_enabled=denoise,
        denoiser_iterations=args.atrous_iterations,
        wavefront_execution_strategy="wavefront",
        wavefront_hdr_capture=True,
        wavefront_tile_capacity=args.width * args.height,
        direct_swapchain_storage=False,
    )


def _capture(window, scenario, scene, args, *, reference, path):
    mode = "reference" if reference else "relax"
    frames = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32,
        shape=(args.frames, args.height, args.width, 4),
    )
    timings = []
    config = _renderer_config(args, reference=reference)
    with ol.VulkanGlfwPresenter(window, config=config) as presenter:
        for index in range(args.frames):
            presenter.present_wavefront(
                scene, _camera(scenario, index, args.frames),
                args.width, args.height,
            )
            frames[index] = presenter.capture_wavefront_hdr()
            timings.append(float(presenter.last_timings.get("gpu_frame_ms", 0.0)))
            print(f"{scenario.name}/{mode}: {index + 1}/{args.frames}")
    frames.flush()
    return frames, timings


def make_baseline(configuration, summaries, *, reason):
    """Create the reviewable baseline payload stored in version control."""
    return {
        "schema": BASELINE_SCHEMA,
        "override_reason": reason,
        "configuration": configuration,
        "policies": {
            "maximum": {
                name: {"ratio": ratio, "floor": floor}
                for name, (ratio, floor) in MAX_POLICIES.items()
            },
            "minimum": {
                name: {"margin": margin}
                for name, margin in MIN_POLICIES.items()
            },
            "bias_margin": BIAS_MARGIN,
        },
        "accepted": summaries,
    }


def evaluate_against_baseline(configuration, summaries, baseline):
    """Return human-readable failures against an accepted baseline payload."""
    failures = []
    if baseline.get("schema") != BASELINE_SCHEMA:
        failures.append("baseline schema does not match this gate")
        return failures
    if baseline.get("configuration") != configuration:
        failures.append(
            "capture configuration differs from the accepted baseline; "
            "use the baseline configuration or explicitly accept a replacement"
        )
        return failures
    accepted_scenes = baseline.get("accepted", {})
    if set(accepted_scenes) != set(summaries):
        failures.append("scenario set differs from the accepted baseline")
        return failures
    policies = baseline["policies"]
    for scene_name, observed in summaries.items():
        accepted = accepted_scenes[scene_name]
        for metric, policy in policies["maximum"].items():
            limit = max(
                accepted[metric] * policy["ratio"],
                accepted[metric] + policy["floor"],
            )
            if observed[metric] > limit:
                failures.append(
                    f"{scene_name}/{metric}: {observed[metric]:.7g} > "
                    f"{limit:.7g} (accepted {accepted[metric]:.7g})"
                )
        for metric, policy in policies["minimum"].items():
            limit = accepted[metric] - policy["margin"]
            if observed[metric] < limit:
                failures.append(
                    f"{scene_name}/{metric}: {observed[metric]:.7g} < "
                    f"{limit:.7g} (accepted {accepted[metric]:.7g})"
                )
        bias_limit = abs(accepted["bias_mean"]) + policies["bias_margin"]
        if abs(observed["bias_mean"]) > bias_limit:
            failures.append(
                f"{scene_name}/|bias_mean|: {abs(observed['bias_mean']):.7g} "
                f"> {bias_limit:.7g}"
            )
    return failures


def _selected_summary(summary):
    names = set(MAX_POLICIES) | set(MIN_POLICIES) | {"bias_mean"}
    return {name: float(summary[name]) for name in sorted(names)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--reference-samples", type=int, default=16)
    parser.add_argument("--candidate-samples", type=int, default=1)
    parser.add_argument("--bounces", type=int, default=8)
    parser.add_argument("--area-light-samples", type=int, default=2)
    parser.add_argument("--atrous-iterations", type=int, default=3)
    parser.add_argument("--history-limit", type=int, default=32)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--output", type=Path,
        default=Path("/tmp/ordinarylight_noise_quality"),
    )
    parser.add_argument(
        "--accept-baseline", action="store_true",
        help=("replace the accepted baseline; requires " + OVERRIDE_REASON),
    )
    args = parser.parse_args()
    if args.frames < 3:
        parser.error("--frames must be at least 3")
    if args.reference_samples <= args.candidate_samples:
        parser.error("reference samples must exceed candidate samples")
    if args.width * args.height > 4194304:
        parser.error("capture extent exceeds wavefront tile capacity")
    reason = os.environ.get(OVERRIDE_REASON, "").strip()
    if args.accept_baseline and not reason:
        parser.error(f"--accept-baseline requires {OVERRIDE_REASON}")

    args.output.mkdir(parents=True, exist_ok=True)
    glfw = load_glfw()
    if not glfw.init():
        raise RuntimeError("GLFW initialization failed")
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.DECORATED, glfw.FALSE)
    window = glfw.create_window(
        args.width, args.height, "Ordinary Light ReLAX noise gate", None, None
    )
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW Vulkan window creation failed")

    summaries = {}
    timings = {}
    comparisons = {}
    try:
        for scenario in SCENARIOS:
            scene = scenario.build()
            reference, reference_times = _capture(
                window, scenario, scene, args, reference=True,
                path=args.output / f"{scenario.name}_reference.npy",
            )
            candidate, candidate_times = _capture(
                window, scenario, scene, args, reference=False,
                path=args.output / f"{scenario.name}_relax.npy",
            )
            comparisons[scenario.name] = (reference, candidate)
            summaries[scenario.name] = _selected_summary(
                summarize_temporal_quality(reference, candidate)
            )
            timings[scenario.name] = {
                "reference_gpu_ms_mean": float(np.mean(reference_times)),
                "candidate_gpu_ms_mean": float(np.mean(candidate_times)),
            }
    finally:
        glfw.destroy_window(window)
        glfw.terminate()

    configuration = _configuration(args)
    report = {
        "configuration": configuration,
        "quality": summaries,
        "timings": timings,
    }
    write_temporal_quality_csv(args.output / "metrics.csv", comparisons)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))

    if args.accept_baseline:
        baseline = make_baseline(configuration, summaries, reason=reason)
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(baseline, indent=2) + "\n")
        print(f"ACCEPTED: ReLAX noise baseline -> {args.baseline}")
        return
    if not args.baseline.is_file():
        raise SystemExit(
            "FAIL: accepted baseline is missing; create it with "
            f"--accept-baseline and {OVERRIDE_REASON}"
        )
    failures = evaluate_against_baseline(
        configuration, summaries, json.loads(args.baseline.read_text())
    )
    if failures:
        print("FAIL: accepted noise-quality baseline regressed")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)
    print("PASS: ReLAX preserves the accepted multi-scene noise baseline")


if __name__ == "__main__":
    main()
