"""Gate unbiased path termination against the full-path HDR baseline."""

import argparse
import json
from pathlib import Path

import numpy as np

import ordinarylight as ol
from ordinarylight.showcases.rooms import SCENES, get_restir_scene
from ordinarylight.integrations.temporal_quality import (
    save_hdr_sequence,
    summarize_temporal_quality,
    write_temporal_quality_csv,
)


def _capture(scene, scene_spec, args, mode, samples, *, bounces=None):
    roulette = mode == "roulette"
    bounces = args.bounces if bounces is None else bounces
    config = ol.RendererConfig(
        max_bounces=bounces,
        samples_per_pixel=samples,
        area_light_samples=args.area_light_samples,
        wavefront_secondary_area_light_samples=args.secondary_area_light_samples,
        wavefront_environment_samples=args.environment_samples,
        wavefront_execution_strategy=args.strategy,
        wavefront_russian_roulette=roulette,
        wavefront_russian_roulette_start=args.roulette_start,
        wavefront_russian_roulette_min_survival=args.minimum_survival,
        wavefront_tile_capacity=args.width * args.height,
    )
    frames = np.empty((args.frames, args.height, args.width, 4), np.float32)
    with ol.renderers.gi.VulkanGlobalIlluminationRenderer(config=config) as renderer:
        for index in range(args.frames):
            angle = np.pi + args.motion * index / max(args.frames - 1, 1)
            camera = ol.PerspectiveCamera(
                position=(
                    scene_spec.orbit_radius * np.sin(angle),
                    scene_spec.camera_height,
                    scene_spec.orbit_radius * np.cos(angle),
                ),
                target=scene_spec.target,
            )
            frames[index] = renderer.render_wavefront(
                scene, camera, args.width, args.height,
                samples=samples, frame_index=index,
            )
            print(f"{mode}: {index + 1}/{args.frames}")
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=tuple(SCENES), default="dense")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--reference-samples", type=int, default=16)
    parser.add_argument("--bounces", type=int, default=5)
    parser.add_argument(
        "--candidate-bounces", type=int,
        help=("compare this maximum path depth against --bounces; when omitted "
              "the candidate instead enables Russian roulette at the same depth"),
    )
    parser.add_argument("--motion", type=float, default=0.24)
    parser.add_argument("--strategy", choices=("wavefront", "megakernel"),
                        default="megakernel")
    parser.add_argument("--area-light-samples", type=int, default=1)
    parser.add_argument("--secondary-area-light-samples", type=int, default=0)
    parser.add_argument("--environment-samples", type=int, default=1)
    parser.add_argument("--roulette-start", type=int, default=3)
    parser.add_argument("--minimum-survival", type=float, default=0.1)
    parser.add_argument("--max-error-ratio", type=float, default=1.02)
    parser.add_argument("--max-temporal-ratio", type=float, default=1.02)
    parser.add_argument("--max-low-frequency-ratio", type=float, default=1.05)
    parser.add_argument("--max-abs-bias", type=float, default=0.01)
    parser.add_argument(
        "--output", type=Path,
        default=Path("/tmp/ordinarylight_path_termination_quality"),
    )
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    if args.frames < 3 or args.reference_samples < 2:
        parser.error("at least three frames and two reference samples are required")
    if not 1 <= args.bounces <= 16:
        parser.error("--bounces must be between 1 and 16")
    if args.candidate_bounces is not None \
            and not 1 <= args.candidate_bounces <= 16:
        parser.error("--candidate-bounces must be between 1 and 16")

    scene_spec = get_restir_scene(args.scene)
    scene = scene_spec.build()
    reference = _capture(scene, scene_spec, args, "reference",
                         args.reference_samples)
    baseline = _capture(scene, scene_spec, args, "baseline", 1)
    candidate_mode = (
        f"bounces_{args.candidate_bounces}"
        if args.candidate_bounces is not None else "roulette"
    )
    candidate = _capture(
        scene, scene_spec, args, candidate_mode, 1,
        bounces=args.candidate_bounces,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    for name, frames in (("reference", reference), ("baseline", baseline),
                         (candidate_mode, candidate)):
        save_hdr_sequence(args.output / f"{name}.npz", frames)
    comparisons = {
        "baseline": (reference, baseline),
        candidate_mode: (reference, candidate),
    }
    quality = {
        name: summarize_temporal_quality(*values)
        for name, values in comparisons.items()
    }
    result = {
        "scene": args.scene,
        "extent": [args.width, args.height],
        "frames": args.frames,
        "baseline_bounces": args.bounces,
        "candidate_bounces": args.candidate_bounces,
        "roulette_start": args.roulette_start,
        "quality": quality,
    }
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n")
    write_temporal_quality_csv(args.output / "metrics.csv", comparisons)
    print(json.dumps(result, indent=2))

    base = quality["baseline"]
    candidate = quality[candidate_mode]
    failures = []
    if candidate["relative_rmse_mean"] > (
            base["relative_rmse_mean"] * args.max_error_ratio):
        failures.append("relative RMSE regressed")
    if candidate["temporal_residual_rmse_mean"] > (
            base["temporal_residual_rmse_mean"] * args.max_temporal_ratio):
        failures.append("temporal residual regressed")
    if candidate["low_frequency_energy_ratio_mean"] > (
            base["low_frequency_energy_ratio_mean"]
            * args.max_low_frequency_ratio):
        failures.append("low-frequency noise structure regressed")
    if abs(candidate["bias_mean"]) > args.max_abs_bias:
        failures.append("absolute HDR bias exceeded limit")
    if args.gate and failures:
        raise SystemExit("FAIL: " + "; ".join(failures))
    if args.gate:
        print("PASS: path termination preserves configured HDR quality")


if __name__ == "__main__":
    main()
