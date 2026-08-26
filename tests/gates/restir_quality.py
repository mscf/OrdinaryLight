"""Gate deterministic conventional/ReSTIR HDR sequences."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.integrations.temporal_quality import (
    summarize_temporal_quality,
    write_temporal_quality_csv,
)

from ordinarylight.showcases.rooms import SCENES, get_restir_scene


def _camera(index, frames, motion, scene_spec):
    fraction = index / max(frames - 1, 1)
    angle = math.pi + motion * fraction
    return ol.PerspectiveCamera(
        position=(
            scene_spec.orbit_radius * math.sin(angle),
            scene_spec.camera_height,
            scene_spec.orbit_radius * math.cos(angle),
        ),
        target=scene_spec.target,
    )


def _capture(window, scene, args, mode, samples, output_path):
    restir_mode = mode in {"canonical", "pairwise", "generalized", "restir"}
    pairwise = mode == "pairwise" or mode == "generalized" or (
        mode == "restir" and args.pairwise_mis
    )
    generalized = mode == "generalized" or (
        mode == "restir" and args.generalized_mis
    )
    config = ol.RendererConfig(
        max_bounces=args.bounces,
        samples_per_pixel=samples,
        area_light_samples=args.area_light_samples,
        wavefront_secondary_area_light_samples=args.secondary_area_light_samples,
        wavefront_environment_samples=args.environment_samples,
        wavefront_unified_secondary_nee=args.unified_secondary_nee,
        wavefront_unified_primary_restir=args.unified_primary_restir,
        wavefront_stratified_primary_restir=args.stratified_primary_restir,
        wavefront_restir_di=restir_mode or mode == "conventional",
        wavefront_restir_candidates=args.restir_candidates,
        wavefront_restir_history_limit=args.restir_history_limit,
        wavefront_restir_spatial_reuse=(
            restir_mode and args.spatial_neighbors > 0
        ),
        wavefront_restir_spatial_neighbors=max(args.spatial_neighbors, 1),
        wavefront_restir_spatial_radius=args.spatial_radius,
        wavefront_restir_pairwise_mis=(
            pairwise and args.spatial_neighbors > 0
        ),
        wavefront_restir_generalized_mis=(
            generalized and args.spatial_neighbors > 0
        ),
        wavefront_restir_generalized_balance_cap=args.generalized_balance_cap,
        wavefront_execution_strategy=args.strategy,
        wavefront_material_bucketing=args.material_bucketing,
        wavefront_material_bucketing_start_bounce=(
            args.material_bucketing_start_bounce
        ),
        wavefront_persistent_coarse_tiles=args.persistent_coarse_tiles,
        wavefront_persistent_continuations=args.persistent_continuations,
        wavefront_scene_specialization=args.scene_specialization,
        wavefront_tile_capacity=args.tile_capacity,
        wavefront_profiling=True,
        wavefront_hdr_capture=True,
        direct_swapchain_storage=False,
    )
    captured = np.lib.format.open_memmap(
        output_path, mode="w+", dtype=np.float32,
        shape=(args.frames, args.height, args.width, 4),
    )
    timings = []
    with ol.VulkanGlfwPresenter(window, config=config) as presenter:
        if mode == "conventional":
            presenter.set_wavefront_restir_enabled(False)
        for index in range(args.frames):
            camera = _camera(
                index, args.frames, args.motion_radians, args.scene_spec
            )
            presenter.present_wavefront(scene, camera, args.width, args.height)
            captured[index] = presenter.capture_wavefront_hdr()
            timing = presenter.last_timings
            counters = timing.get("wavefront_work_counters", {})
            timings.append({
                "gpu_ms": timing.get("gpu_frame_ms", 0.0),
                "accepted": counters.get("restir_history_accepted", 0),
                "rejected": counters.get("restir_history_rejected", 0),
                "motion_pixels": timing.get(
                    "wavefront_temporal_motion_pixels", 0.0
                ),
                "motion_valid": timing.get(
                    "wavefront_temporal_motion_valid", True
                ),
                "restir_history_valid": timing.get(
                    "wavefront_restir_history_valid", False
                ),
            })
            print(f"{mode}: frame {index + 1}/{args.frames}")
    captured.flush()
    print(f"{mode}: capture complete -> {output_path}")
    return captured, timings


def _mean_gpu_ms(timings):
    values = [item["gpu_ms"] for item in timings if item["gpu_ms"] > 0.0]
    return float(np.mean(values)) if values else 0.0


def _evaluate_gate(summaries, gpu_means, args):
    failures = []
    conventional = summaries["conventional"]
    candidate_modes = tuple(
        mode for mode in ("canonical", "pairwise", "generalized")
        if mode in summaries
    )
    for mode in candidate_modes:
        summary = summaries[mode]
        if abs(summary["bias_mean"]) > args.gate_max_abs_bias:
            failures.append(
                f"{mode}: |bias_mean| {abs(summary['bias_mean']):.6g} > "
                f"{args.gate_max_abs_bias:.6g}"
            )
        error_ratio = summary["relative_rmse_mean"] / max(
            conventional["relative_rmse_mean"], 1e-12
        )
        if error_ratio > args.gate_max_error_ratio:
            failures.append(
                f"{mode}: relative-RMSE ratio {error_ratio:.4f} > "
                f"{args.gate_max_error_ratio:.4f}"
            )
    canonical = summaries.get("canonical")
    for mode in ("pairwise", "generalized"):
        if canonical is None or mode not in summaries:
            continue
        mae_ratio = summaries[mode]["mae_mean"] / max(
            canonical["mae_mean"], 1e-12
        )
        if mae_ratio > args.gate_max_mae_ratio:
            failures.append(
                f"{mode}: canonical MAE ratio {mae_ratio:.4f} > "
                f"{args.gate_max_mae_ratio:.4f}"
            )
    if (not args.gate_quality_only and canonical is not None
            and gpu_means["canonical"] > 0.0):
        for mode, limit in (
            ("pairwise", args.gate_max_pairwise_gpu_ratio),
            ("generalized", args.gate_max_generalized_gpu_ratio),
        ):
            if mode not in gpu_means:
                continue
            ratio = gpu_means[mode] / gpu_means["canonical"]
            if ratio > limit:
                failures.append(
                    f"{mode}: canonical GPU ratio {ratio:.4f} > {limit:.4f}"
                )
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--reference-samples", type=int, default=32)
    parser.add_argument("--bounces", type=int, default=8)
    parser.add_argument("--area-light-samples", type=int, default=1)
    parser.add_argument("--secondary-area-light-samples", type=int, default=1)
    parser.add_argument("--environment-samples", type=int, default=1)
    parser.add_argument(
        "--unified-secondary-nee", action="store_true",
        help="sample area or environment lighting through one mixture draw",
    )
    parser.add_argument(
        "--unified-primary-restir", action="store_true",
        help="include environment candidates in the primary ReSTIR reservoir",
    )
    parser.add_argument(
        "--stratified-primary-restir", action="store_true",
        help="reuse an independent primary environment-light reservoir",
    )
    parser.add_argument("--restir-candidates", type=int, default=1)
    parser.add_argument(
        "--restir-history-limit", type=int, default=6,
        help=("total represented-sample budget, including fresh candidates; "
              "the default fits one fresh, one temporal, and four spatial "
              "representatives"),
    )
    parser.add_argument("--spatial-neighbors", type=int, default=4,
                        choices=range(0, 9))
    parser.add_argument("--spatial-radius", type=int, default=4)
    parser.add_argument(
        "--pairwise-mis", action="store_true",
        help="use pairwise balance weights in the legacy 'restir' capture mode",
    )
    parser.add_argument(
        "--generalized-mis", action="store_true",
        help="balance each spatial proposal across compatible neighbors",
    )
    parser.add_argument(
        "--generalized-balance-cap", type=float, default=2.0,
        help="maximum generalized weight relative to canonical reuse",
    )
    parser.add_argument(
        "--compare-strategies", action="store_true",
        help="capture canonical, pairwise, and generalized spatial reuse",
    )
    parser.add_argument(
        "--gate", action="store_true",
        help="run all strategies and fail when quality/performance limits regress",
    )
    parser.add_argument(
        "--gate-quality-only", action="store_true",
        help="record GPU timings but enforce only HDR quality thresholds",
    )
    parser.add_argument(
        "--candidate-modes", nargs="+",
        choices=("canonical", "pairwise", "generalized"),
        help=("candidate estimators to capture; useful for focused gates "
              "(conventional is always captured)"),
    )
    parser.add_argument("--gate-max-abs-bias", type=float, default=0.01)
    parser.add_argument("--gate-max-error-ratio", type=float, default=1.20)
    parser.add_argument("--gate-max-mae-ratio", type=float, default=1.02)
    parser.add_argument("--gate-max-pairwise-gpu-ratio", type=float, default=1.25)
    parser.add_argument(
        "--gate-max-generalized-gpu-ratio", type=float, default=1.50
    )
    parser.add_argument("--tile-capacity", type=int, default=131072,
                        help="maximum paths per tiled GPU dispatch")
    parser.add_argument("--motion-radians", type=float, default=0.35)
    parser.add_argument(
        "--require-motion-rejection", action="store_true",
        help="fail unless camera motion invalidates temporal history",
    )
    parser.add_argument("--strategy", choices=(
        "wavefront", "hybrid", "megakernel", "persistent"
    ),
                        default="megakernel")
    parser.add_argument("--material-bucketing", action="store_true")
    parser.add_argument(
        "--material-bucketing-start-bounce", type=int, default=2,
    )
    parser.add_argument("--persistent-coarse-tiles", action="store_true")
    parser.add_argument("--persistent-continuations", action="store_true")
    specialization = parser.add_mutually_exclusive_group()
    specialization.add_argument(
        "--scene-specialization", action="store_true",
        help="enable the experimental opaque-only shader variant",
    )
    specialization.add_argument(
        "--no-scene-specialization", dest="scene_specialization",
        action="store_false", help=argparse.SUPPRESS,
    )
    parser.set_defaults(scene_specialization=False)
    parser.add_argument("--output", default="/tmp/ordinarylight_restir_quality")
    parser.add_argument(
        "--scene", choices=tuple(SCENES), default="area_lights",
        help="procedural regression scene to capture",
    )
    args = parser.parse_args()
    args.scene_spec = get_restir_scene(args.scene)
    args.compare_strategies = (
        args.compare_strategies or args.gate or args.candidate_modes is not None
    )
    if args.frames < 2:
        parser.error("--frames must be at least 2")
    if not 1 <= args.tile_capacity <= 4194304:
        parser.error("--tile-capacity must be between 1 and 4194304")
    if not 1 <= args.spatial_radius <= 32:
        parser.error("--spatial-radius must be between 1 and 32")
    if not 1.0 <= args.generalized_balance_cap <= 8.0:
        parser.error("--generalized-balance-cap must be between 1 and 8")
    gate_limits = (
        args.gate_max_abs_bias, args.gate_max_error_ratio,
        args.gate_max_mae_ratio, args.gate_max_pairwise_gpu_ratio,
        args.gate_max_generalized_gpu_ratio,
    )
    if any(value <= 0.0 for value in gate_limits):
        parser.error("all gate thresholds must be positive")
    if (args.spatial_neighbors > 0
            and args.restir_history_limit <= args.restir_candidates + 1):
        print(
            "warning: history budget admits no spatial representatives; "
            "use at least --restir-history-limit "
            f"{args.restir_candidates + 2}"
        )

    glfw = load_glfw()
    if not glfw.init():
        raise RuntimeError("GLFW initialization failed")
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.DECORATED, glfw.FALSE)
    window = glfw.create_window(args.width, args.height, "ReSTIR quality", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW Vulkan window creation failed")
    prefix = Path(args.output)
    if args.candidate_modes is not None:
        candidate_modes = ("conventional", *dict.fromkeys(args.candidate_modes))
    elif args.compare_strategies:
        candidate_modes = (
            "conventional", "canonical", "pairwise", "generalized"
        )
    else:
        candidate_modes = ("conventional", "restir")
    paths = {
        mode: prefix.with_name(prefix.name + f"_{mode}.npy")
        for mode in ("reference", *candidate_modes)
    }
    try:
        scene = args.scene_spec.build()
        reference, _ = _capture(
            window, scene, args, "reference", args.reference_samples,
            paths["reference"],
        )
        captures = {}
        timings_by_mode = {}
        for mode in candidate_modes:
            captures[mode], timings_by_mode[mode] = _capture(
                window, scene, args, mode, 1, paths[mode]
            )
    finally:
        glfw.destroy_window(window)
        glfw.terminate()

    metadata = {
        "width": args.width, "height": args.height, "frames": args.frames,
        "scene": args.scene,
        "motion_radians": args.motion_radians,
        "spatial_neighbors": args.spatial_neighbors,
        "spatial_radius": args.spatial_radius,
        "pairwise_mis": args.pairwise_mis,
        "generalized_mis": args.generalized_mis,
        "generalized_balance_cap": args.generalized_balance_cap,
        "compare_strategies": args.compare_strategies,
        "gate_quality_only": args.gate_quality_only,
        "candidate_modes": list(candidate_modes[1:]),
    }
    metadata["reference_samples"] = args.reference_samples
    metadata["format"] = "NumPy float32 RGBA memory-mapped sequence"
    metadata_path = prefix.with_name(prefix.name + "_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Captures saved incrementally; metadata -> {metadata_path}")
    comparisons = {
        mode: (reference, capture) for mode, capture in captures.items()
    }
    metrics_path = prefix.with_name(prefix.name + "_metrics.csv")
    performance_path = prefix.with_name(prefix.name + "_performance.csv")
    print("Computing per-frame HDR and temporal metrics...")
    write_temporal_quality_csv(metrics_path, comparisons)
    print(f"Quality metrics -> {metrics_path}")
    with performance_path.open(
        "w", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "mode", "frame", "gpu_ms", "history_accepted",
            "history_rejected", "history_acceptance", "motion_pixels",
            "motion_valid", "restir_history_valid",
        ))
        writer.writeheader()
        for mode, timings in timings_by_mode.items():
            for frame, timing in enumerate(timings):
                attempted = timing["accepted"] + timing["rejected"]
                writer.writerow({
                    "mode": mode, "frame": frame,
                    "gpu_ms": timing["gpu_ms"],
                    "history_accepted": timing["accepted"],
                    "history_rejected": timing["rejected"],
                    "history_acceptance": (
                        timing["accepted"] / attempted if attempted else 0.0
                    ),
                    "motion_pixels": timing["motion_pixels"],
                    "motion_valid": int(timing["motion_valid"]),
                    "restir_history_valid": int(
                        timing["restir_history_valid"]
                    ),
                })
    print(f"Performance and reuse metrics -> {performance_path}")
    print("Computing aggregate summaries...")
    summaries = {}
    for mode, pair in comparisons.items():
        print(mode)
        summaries[mode] = summarize_temporal_quality(*pair)
        for name, value in summaries[mode].items():
            print(f"  {name}={value:.8g}")
    gpu_means = {
        mode: _mean_gpu_ms(timings)
        for mode, timings in timings_by_mode.items()
    }
    failures = _evaluate_gate(summaries, gpu_means, args) if args.gate else []
    if args.require_motion_rejection:
        candidate_timings = [
            timing
            for mode, timings in timings_by_mode.items()
            if mode != "reference"
            for timing in timings
        ]
        rejected = [
            timing for timing in candidate_timings
            if not timing["motion_valid"]
        ]
        if not rejected:
            failures.append("camera motion never crossed the history limit")
        elif any(timing["restir_history_valid"] for timing in rejected):
            failures.append("ReSTIR history remained valid during fast motion")
    gate_summary = {
        "enabled": args.gate,
        "quality_only": args.gate_quality_only,
        "status": ("fail" if failures else "pass") if args.gate else "not_run",
        "failures": failures,
        "thresholds": {
            "max_abs_bias": args.gate_max_abs_bias,
            "max_error_ratio": args.gate_max_error_ratio,
            "max_mae_ratio": args.gate_max_mae_ratio,
            "max_pairwise_gpu_ratio": args.gate_max_pairwise_gpu_ratio,
            "max_generalized_gpu_ratio":
                args.gate_max_generalized_gpu_ratio,
        },
    }
    summary_path = prefix.with_name(prefix.name + "_summary.json")
    summary_path.write_text(json.dumps({
        "scene": args.scene, "quality": summaries, "gpu_ms_mean": gpu_means,
        "gate": gate_summary,
    }, indent=2) + "\n")
    print(f"Aggregate summary -> {summary_path}")
    if args.gate:
        if failures:
            print("FAIL: ReSTIR strategy gate")
            for failure in failures:
                print(f"  {failure}")
            raise SystemExit(1)
        print("PASS: ReSTIR strategies are within configured tolerances")


if __name__ == "__main__":
    main()
