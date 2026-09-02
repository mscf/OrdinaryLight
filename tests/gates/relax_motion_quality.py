"""Guard ReLAX quality across rigid-object and camera motion.

This is an opt-in Vulkan gate.  It compares recurrent one-sample output with
independent high-sample reference frames, deriving motion/disocclusion masks
from the reference pixels rather than from fixture-specific screen regions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics

import numpy as np

import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.showcases.rooms import (
    animate_object_motion_room,
    build_object_motion_room,
)


SCHEMA = 1
DEFAULT_BASELINE = Path(__file__).with_name("baselines") / "relax_motion_quality.json"
OVERRIDE_REASON = "ORDINARYLIGHT_RELAX_MOTION_GATE_OVERRIDE_REASON"
MAXIMUM_POLICIES = {
    "log_luminance_rmse": (1.10, 0.01),
    "motion_region_rmse": (1.12, 0.015),
    "stationary_residual_rmse": (1.12, 0.008),
}
MINIMUM_POLICIES = {"edge_correlation": 0.02}


def _camera(angle=math.pi):
    return ol.PerspectiveCamera(
        position=(8.5 * math.sin(angle), 3.2, 8.5 * math.cos(angle)),
        target=(0.0, 1.25, 0.0),
    )


def _config(args, *, reference):
    denoise = not reference
    return ol.RendererConfig(
        max_bounces=args.bounces,
        samples_per_pixel=(args.reference_samples if reference else 1),
        area_light_samples=2,
        wavefront_restir_di=not reference,
        wavefront_restir_reservoirs=2,
        wavefront_restir_candidates=4,
        wavefront_restir_history_limit=4,
        progressive_accumulation=denoise,
        temporal_history=denoise,
        temporal_history_limit=32,
        denoiser_enabled=denoise,
        denoiser_iterations=args.atrous_iterations,
        wavefront_hdr_capture=True,
        wavefront_tile_capacity=args.width * args.height,
        direct_swapchain_storage=False,
    )


def _trajectory(args, kind):
    fractions = np.linspace(0.0, 1.0, args.frames)
    if kind == "object":
        return [(float(2.7 * value), _camera()) for value in fractions]
    return [
        (0.0, _camera(math.pi + args.camera_arc * (value - 0.5)))
        for value in fractions
    ]


def _capture(window, args, kind, *, reference, output):
    scene = build_object_motion_room()
    frames = np.empty((args.frames, args.height, args.width, 4), np.float32)
    timings = []
    with ol.VulkanGlfwPresenter(
        window, config=_config(args, reference=reference),
    ) as presenter:
        for index, (animation_time, camera) in enumerate(_trajectory(args, kind)):
            if kind == "object":
                animate_object_motion_room(scene, animation_time)
            presenter.present_wavefront(scene, camera, args.width, args.height)
            frames[index] = presenter.capture_wavefront_hdr()
            gpu_ms = float(presenter.last_timings.get("gpu_frame_ms", 0.0))
            if gpu_ms > 0.0:
                timings.append(gpu_ms)
            print(
                f"{kind}/{'reference' if reference else 'relax'}: "
                f"{index + 1}/{args.frames}"
            )
    np.save(output, frames, allow_pickle=False)
    return frames, timings


def _luminance(frames):
    rgb = np.maximum(np.asarray(frames, np.float32)[..., :3], 0.0)
    return np.log1p(
        rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    )


def _gradient(image):
    gx = np.diff(image, axis=1, append=image[:, -1:])
    gy = np.diff(image, axis=0, append=image[-1:, :])
    return np.sqrt(gx * gx + gy * gy)


def evaluate_sequence(reference, candidate):
    """Measure actual inter-frame pixels, including disocclusion regions."""
    ref = _luminance(reference)
    test = _luminance(candidate)
    if ref.shape != test.shape or ref.ndim != 3 or ref.shape[0] < 3:
        raise ValueError("matching reference/candidate sequences need >= 3 frames")
    scale = max(float(np.sqrt(np.mean(ref * ref))), 1.0e-6)
    full_rmse = float(np.sqrt(np.mean((test - ref) ** 2)) / scale)
    motion_errors = []
    stationary_errors = []
    edge_correlations = []
    for index in range(1, ref.shape[0]):
        reference_delta = np.abs(ref[index] - ref[index - 1])
        active = (ref[index] > 1.0e-5) | (ref[index - 1] > 1.0e-5)
        # Include both newly covered and newly revealed pixels. Restricting
        # this to the current foreground would make a trailing history ghost
        # invisible to the metric precisely when an object uncovers it.
        motion = (reference_delta > 0.025) & active
        stationary = (reference_delta < 0.003) & active
        if np.any(motion):
            motion_errors.append(float(np.sqrt(np.mean(
                (test[index][motion] - ref[index][motion]) ** 2
            )) / scale))
        if np.any(stationary):
            candidate_delta = test[index] - test[index - 1]
            reference_signed_delta = ref[index] - ref[index - 1]
            stationary_errors.append(float(np.sqrt(np.mean(
                (candidate_delta[stationary] - reference_signed_delta[stationary]) ** 2
            )) / scale))
        ref_edge = _gradient(ref[index])
        test_edge = _gradient(test[index])
        edge_pixels = ref_edge > max(float(np.percentile(ref_edge, 85.0)), 1e-5)
        if np.count_nonzero(edge_pixels) > 8:
            correlation = np.corrcoef(
                ref_edge[edge_pixels], test_edge[edge_pixels]
            )[0, 1]
            if np.isfinite(correlation):
                edge_correlations.append(float(correlation))
    return {
        "log_luminance_rmse": full_rmse,
        "motion_region_rmse": max(motion_errors, default=full_rmse),
        "stationary_residual_rmse": max(stationary_errors, default=0.0),
        "edge_correlation": min(edge_correlations, default=0.0),
    }


def evaluate_against_baseline(configuration, observed, baseline):
    failures = []
    if baseline.get("schema") != SCHEMA:
        return ["baseline schema does not match this gate"]
    if baseline.get("configuration") != configuration:
        return ["capture configuration differs from the accepted baseline"]
    for scenario, metrics in observed.items():
        accepted = baseline.get("accepted", {}).get(scenario)
        if accepted is None:
            failures.append(f"missing accepted scenario {scenario}")
            continue
        for name, (ratio, floor) in MAXIMUM_POLICIES.items():
            limit = max(accepted[name] * ratio, accepted[name] + floor)
            if metrics[name] > limit:
                failures.append(f"{scenario}/{name}: {metrics[name]:.6g} > {limit:.6g}")
        for name, margin in MINIMUM_POLICIES.items():
            limit = accepted[name] - margin
            if metrics[name] < limit:
                failures.append(f"{scenario}/{name}: {metrics[name]:.6g} < {limit:.6g}")
        timing_limit = max(
            accepted["median_gpu_ms"] * 1.20,
            accepted["median_gpu_ms"] + 0.35,
        )
        if metrics["median_gpu_ms"] > timing_limit:
            failures.append(
                f"{scenario}/median_gpu_ms: {metrics['median_gpu_ms']:.6g} > "
                f"{timing_limit:.6g}"
            )
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--reference-samples", type=int, default=16)
    parser.add_argument("--bounces", type=int, default=8)
    parser.add_argument("--atrous-iterations", type=int, default=3)
    parser.add_argument("--camera-arc", type=float, default=0.20)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--output", type=Path, default=Path("/tmp/ordinarylight_relax_motion"))
    parser.add_argument("--accept-baseline", action="store_true")
    args = parser.parse_args(argv)
    if args.frames < 3:
        parser.error("--frames must be at least 3")
    if args.width * args.height > 4194304:
        parser.error("capture extent exceeds wavefront tile capacity")
    reason = os.environ.get(OVERRIDE_REASON, "").strip()
    if args.accept_baseline and not reason:
        parser.error(f"--accept-baseline requires {OVERRIDE_REASON}")

    args.output.mkdir(parents=True, exist_ok=True)
    configuration = {
        "width": args.width, "height": args.height, "frames": args.frames,
        "reference_samples": args.reference_samples, "bounces": args.bounces,
        "atrous_iterations": args.atrous_iterations,
        "camera_arc": args.camera_arc,
    }
    glfw = load_glfw()
    if not glfw.init():
        raise RuntimeError("GLFW initialization failed")
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.DECORATED, glfw.FALSE)
    window = glfw.create_window(args.width, args.height, "ReLAX motion gate", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW Vulkan window creation failed")
    results = {}
    try:
        for kind in ("object", "camera"):
            reference, _ = _capture(
                window, args, kind, reference=True,
                output=args.output / f"{kind}_reference.npy",
            )
            candidate, timings = _capture(
                window, args, kind, reference=False,
                output=args.output / f"{kind}_relax.npy",
            )
            results[kind] = evaluate_sequence(reference, candidate)
            results[kind]["median_gpu_ms"] = (
                float(statistics.median(timings)) if timings else 0.0
            )
    finally:
        glfw.destroy_window(window)
        glfw.terminate()

    report = {"schema": SCHEMA, "configuration": configuration, "results": results}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.accept_baseline:
        payload = {
            "schema": SCHEMA, "override_reason": reason,
            "configuration": configuration, "accepted": results,
        }
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"ACCEPTED: ReLAX motion baseline -> {args.baseline}")
        return 0
    if not args.baseline.is_file():
        raise SystemExit(
            f"FAIL: missing {args.baseline}; accept it explicitly with {OVERRIDE_REASON}"
        )
    failures = evaluate_against_baseline(
        configuration, results, json.loads(args.baseline.read_text())
    )
    if failures:
        print("FAIL: ReLAX motion quality regressed")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("PASS: ReLAX object/camera motion preserves the accepted baseline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
