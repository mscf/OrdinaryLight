"""Gate conventional versus secondary-reservoir indirect reuse quality."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.integrations.temporal_quality import (
    save_hdr_sequence,
    summarize_temporal_quality,
    write_temporal_quality_csv,
)

from ordinarylight.showcases.rooms import SCENES, get_restir_scene


def camera(index, frames, motion, scene_spec):
    angle = math.pi + motion * index / max(frames - 1, 1)
    return ol.PerspectiveCamera(
        position=(
            scene_spec.orbit_radius * math.sin(angle),
            scene_spec.camera_height,
            scene_spec.orbit_radius * math.cos(angle),
        ),
        target=scene_spec.target,
    )


def capture(window, scene, args, mode, samples):
    reuse = mode == "reuse"
    reuse_infrastructure = mode in {"conventional", "reuse"}
    config = ol.RendererConfig(
        max_bounces=args.bounces,
        samples_per_pixel=samples,
        wavefront_execution_strategy="wavefront",
        wavefront_hdr_capture=True,
        wavefront_profiling=True,
        direct_swapchain_storage=False,
        wavefront_indirect_reuse_storage=reuse_infrastructure,
        wavefront_indirect_reuse_candidates=reuse_infrastructure,
        wavefront_indirect_reuse_temporal=reuse_infrastructure,
        wavefront_indirect_reuse_spatial=reuse_infrastructure,
        wavefront_indirect_reuse_apply=reuse,
        wavefront_indirect_reuse_apply_strength=args.strength,
        wavefront_indirect_reuse_history_limit=args.history_limit,
    )
    frames = np.empty((args.frames, args.height, args.width, 4), np.float32)
    gpu = []
    with ol.VulkanGlfwPresenter(window, config=config) as presenter:
        for index in range(args.frames):
            presenter.present_wavefront(
                scene, camera(index, args.frames, args.motion, args.scene_spec),
                args.width, args.height,
            )
            frames[index] = presenter.capture_wavefront_hdr()
            value = presenter.last_timings.get("gpu_frame_ms", 0.0)
            if value > 0.0:
                gpu.append(value)
            print(f"{mode}: {index + 1}/{args.frames}")
    return frames, float(np.median(gpu)) if gpu else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--reference-samples", type=int, default=32)
    parser.add_argument("--bounces", type=int, default=5)
    parser.add_argument("--motion", type=float, default=0.08)
    parser.add_argument("--strength", type=float, default=0.35)
    parser.add_argument("--history-limit", type=int, default=32)
    parser.add_argument("--scene", choices=tuple(SCENES), default="diffuse")
    parser.add_argument(
        "--output", type=Path,
        default=Path("/tmp/ordinarylight_indirect_quality"),
    )
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--max-error-ratio", type=float, default=1.0)
    parser.add_argument("--max-abs-bias", type=float, default=0.01)
    parser.add_argument("--max-lag-ratio", type=float, default=1.05)
    parser.add_argument("--max-low-frequency-ratio", type=float, default=1.10)
    args = parser.parse_args()
    args.scene_spec = get_restir_scene(args.scene)
    if args.frames < 3:
        raise ValueError("quality capture requires at least three frames")
    glfw = load_glfw()
    if not glfw.init():
        raise RuntimeError("GLFW initialization failed")
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    window = glfw.create_window(args.width, args.height, __doc__, None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW Vulkan window creation failed")
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        scene = args.scene_spec.build()
        reference, reference_gpu = capture(
            window, scene, args, "reference", args.reference_samples)
        conventional, conventional_gpu = capture(
            window, scene, args, "conventional", 1)
        reuse, reuse_gpu = capture(window, scene, args, "reuse", 1)
    finally:
        glfw.destroy_window(window)
        glfw.terminate()
    for name, frames in (
        ("reference", reference), ("conventional", conventional),
        ("reuse", reuse),
    ):
        save_hdr_sequence(args.output / f"{name}.npz", frames)
    comparisons = {
        "conventional": (reference, conventional),
        "reuse": (reference, reuse),
    }
    summaries = {
        name: summarize_temporal_quality(*comparison)
        for name, comparison in comparisons.items()
    }
    gpu = {
        "reference": reference_gpu,
        "conventional": conventional_gpu,
        "reuse": reuse_gpu,
    }
    result = {
        "scene": args.scene,
        "extent": [args.width, args.height],
        "frames": args.frames,
        "quality": summaries,
        "gpu_median_ms": gpu,
    }
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n")
    write_temporal_quality_csv(args.output / "metrics.csv", comparisons)
    print(json.dumps(result, indent=2))
    conventional_error = summaries["conventional"]["relative_rmse_mean"]
    reuse_summary = summaries["reuse"]
    failures = []
    if reuse_summary["relative_rmse_mean"] > (
            conventional_error * args.max_error_ratio):
        failures.append("reuse relative RMSE did not improve")
    if abs(reuse_summary["bias_mean"]) > args.max_abs_bias:
        failures.append("reuse bias exceeded limit")
    conventional_lag = summaries["conventional"]["history_lag_ratio_p95"]
    if reuse_summary["history_lag_ratio_p95"] > (
            conventional_lag * args.max_lag_ratio):
        failures.append("reuse history lag regressed versus conventional")
    conventional_low_frequency = summaries["conventional"][
        "low_frequency_energy_ratio_mean"]
    if reuse_summary["low_frequency_energy_ratio_mean"] > (
            conventional_low_frequency * args.max_low_frequency_ratio):
        failures.append(
            "reuse introduced excessive low-frequency noise structure")
    if args.gate and failures:
        raise SystemExit("FAIL: " + "; ".join(failures))
    if args.gate:
        print("PASS: indirect reuse improves HDR error within stability limits")


if __name__ == "__main__":
    main()
