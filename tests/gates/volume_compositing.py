"""Gate volume/surface compositing and execution parity."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

import ordinarylight as ol
from ordinarylight.showcases.volumes import build_volume_showcase


def _build_shadow_scene(density):
    """Create a deterministic point-light receiver with an intervening volume."""
    scene = ol.Scene()
    floor = np.asarray((
        (-4.0, 0.0, -4.0), (4.0, 0.0, -4.0),
        (4.0, 0.0, 4.0), (-4.0, 0.0, 4.0),
    ), np.float32)
    scene.add_mesh(
        floor, ((0, 1, 2), (0, 2, 3)),
        ol.Material(base_color=(0.72, 0.72, 0.72), roughness=0.8),
        name="volume-shadow-receiver",
    )
    transfer = ol.Texture1D(np.asarray((
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.38),
    ), np.float32))
    scene.add_volume(
        np.full((16, 16, 16), density, np.float32),
        ol.VolumeMaterial(
            transfer, density_scale=1.0, emission_scale=0.0,
            step_size=0.025,
        ),
        transform=(
            ol.Transform.translation((-0.75, 1.0, -0.75))
            @ ol.Transform.scale((1.5, 1.5, 1.5))
        ),
        value_range=(0.0, 1.0), name="shadow-medium",
    )
    scene.add_point_light(
        (0.0, 4.5, 0.0), color=(1.0, 0.96, 0.9), intensity=110.0,
    )
    return scene


def _build_overlap_scene(reverse=False):
    scene = ol.Scene()
    density = np.ones((12, 12, 12), np.float32)
    media = (
        (
            ol.Texture1D(((1.8, 0.08, 0.04, 0.12),) * 2),
            ol.Transform.translation((-1.15, -0.7, -0.3)),
            "red-medium",
        ),
        (
            ol.Texture1D(((0.04, 0.12, 2.0, 0.18),) * 2),
            ol.Transform.translation((-0.35, -0.7, -0.3)),
            "blue-medium",
        ),
    )
    for transfer, translation, name in reversed(media) if reverse else media:
        scene.add_volume(
            density,
            ol.VolumeMaterial(
                transfer, density_scale=1.0, emission_scale=1.0,
                step_size=0.03,
            ),
            transform=translation @ ol.Transform.scale((1.5, 1.4, 1.4)),
            value_range=(0.0, 1.0), name=name,
        )
    return scene


def _render(scene, camera, args, strategy):
    config = ol.RendererConfig(
        max_bounces=args.bounces,
        samples_per_pixel=args.samples,
        wavefront_tile_capacity=args.width * args.height,
        wavefront_execution_strategy=strategy,
        wavefront_fused_secondary=True,
        wavefront_subgroup_enqueue=True,
    )
    started = time.perf_counter()
    with ol.renderers.gi.VulkanGlobalIlluminationRenderer(config=config) as renderer:
        initialized = time.perf_counter()
        result = np.array(renderer.render_wavefront(
            scene, camera, args.width, args.height,
            samples=args.samples, frame_index=args.seed,
        ), copy=True)
        finished = time.perf_counter()
    print(
        f"  backend={initialized - started:.3f}s "
        f"render={finished - initialized:.3f}s"
    )
    return result


def _metrics(reference, candidate):
    difference = np.abs(reference[..., :3] - candidate[..., :3])
    scale = max(float(np.sqrt(np.mean(reference[..., :3] ** 2))), 1e-8)
    return {
        "max_abs": float(np.max(difference)),
        "relative_rmse": float(np.sqrt(np.mean(difference ** 2)) / scale),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--height", type=int, default=252)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--bounces", type=int, default=5)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--max-empty-volume-error", type=float, default=1e-6)
    parser.add_argument("--max-strategy-relative-rmse", type=float, default=0.01)
    parser.add_argument("--min-volume-effect", type=float, default=1e-4)
    parser.add_argument("--min-shadow-darkening", type=float, default=0.01)
    parser.add_argument("--max-overlap-order-error", type=float, default=2e-5)
    parser.add_argument(
        "--overlap-only", action="store_true",
        help="run only the insertion-order overlap check",
    )
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    if args.overlap_only:
        overlap_camera = ol.PerspectiveCamera(
            (0.0, 0.0, -4.0), (0.0, 0.0, 0.3), vertical_fov_degrees=42.0,
        )
        print("Rendering overlapping volumes in forward order...")
        forward = _render(
            _build_overlap_scene(), overlap_camera, args, "wavefront")
        print("Rendering overlapping volumes in reverse order...")
        reverse = _render(
            _build_overlap_scene(reverse=True), overlap_camera, args, "wavefront")
        print("Rendering overlapping volumes with the megakernel...")
        megakernel = _render(
            _build_overlap_scene(), overlap_camera, args, "megakernel")
        parity = _metrics(forward, reverse)
        strategy_parity = _metrics(forward, megakernel)
        result = {
            "overlap_order_parity": parity,
            "overlap_strategy_parity": strategy_parity,
        }
        print(json.dumps(result, indent=2))
        if args.summary is not None:
            args.summary.parent.mkdir(parents=True, exist_ok=True)
            args.summary.write_text(json.dumps(result, indent=2) + "\n")
        if parity["max_abs"] > args.max_overlap_order_error:
            raise SystemExit(
                "FAIL: overlapping media depend on scene insertion order "
                f"(max_abs={parity['max_abs']:.8g})"
            )
        if strategy_parity["relative_rmse"] > args.max_strategy_relative_rmse:
            raise SystemExit(
                "FAIL: overlapping-volume execution strategies diverge "
                f"(relative_rmse={strategy_parity['relative_rmse']:.8g})"
            )
        print("PASS: overlapping media are insertion-order independent")
        return

    angle = -1.01  # Places the orange sphere across the volume boundary.
    camera = ol.PerspectiveCamera(
        (7.5 * math.sin(angle), 3.2, 7.5 * math.cos(angle)),
        (0.0, 1.45, 0.0), vertical_fov_degrees=42.0,
    )
    full = build_volume_showcase(32)
    zero = build_volume_showcase(32)
    volume = zero.volumes[0]
    zero.update_volume(volume, data=np.zeros_like(volume.data))
    bare = build_volume_showcase(32)
    bare.remove_volume(bare.volumes[0])

    print("Rendering zero-density overlap control...")
    zero_wavefront = _render(zero, camera, args, "wavefront")
    print("Rendering no-volume overlap control...")
    bare_wavefront = _render(bare, camera, args, "wavefront")
    print("Rendering megakernel overlap control...")
    zero_megakernel = _render(zero, camera, args, "megakernel")
    print("Rendering active volume...")
    full_wavefront = _render(full, camera, args, "wavefront")

    shadow_camera = ol.PerspectiveCamera(
        (5.5, 3.0, 6.5), (0.0, 0.0, 0.0), vertical_fov_degrees=43.0,
    )
    print("Rendering volume-shadow control...")
    shadow_zero = _render(
        _build_shadow_scene(0.0), shadow_camera, args, "wavefront")
    print("Rendering absorbing volume shadow...")
    shadow_full = _render(
        _build_shadow_scene(1.0), shadow_camera, args, "wavefront")

    overlap_camera = ol.PerspectiveCamera(
        (0.0, 0.0, -4.0), (0.0, 0.0, 0.3), vertical_fov_degrees=42.0,
    )
    print("Rendering overlapping volumes in forward order...")
    overlap_forward = _render(
        _build_overlap_scene(), overlap_camera, args, "wavefront")
    print("Rendering overlapping volumes in reverse order...")
    overlap_reverse = _render(
        _build_overlap_scene(reverse=True), overlap_camera, args, "wavefront")
    print("Rendering overlapping volumes with the megakernel...")
    overlap_megakernel = _render(
        _build_overlap_scene(), overlap_camera, args, "megakernel")

    empty_parity = _metrics(bare_wavefront, zero_wavefront)
    strategy_parity = _metrics(zero_wavefront, zero_megakernel)
    volume_effect = _metrics(zero_wavefront, full_wavefront)
    # The lower half contains the receiver and excludes almost all directly
    # visible volume.  Measuring only negative luminance differences makes the
    # gate insensitive to any harmless emission or tone-map rearrangement.
    lower = slice(args.height // 2, args.height)
    luminance = np.asarray((0.2126, 0.7152, 0.0722), np.float32)
    zero_luma = shadow_zero[lower, :, :3] @ luminance
    full_luma = shadow_full[lower, :, :3] @ luminance
    shadow_darkening = float(np.mean(np.maximum(zero_luma - full_luma, 0.0)))
    shadow_darkening /= max(float(np.mean(zero_luma)), 1e-8)
    overlap_order_parity = _metrics(overlap_forward, overlap_reverse)
    overlap_strategy_parity = _metrics(overlap_forward, overlap_megakernel)
    result = {
        "extent": [args.width, args.height],
        "samples": args.samples,
        "empty_volume_parity": empty_parity,
        "strategy_parity": strategy_parity,
        "active_volume_effect": volume_effect,
        "volume_shadow_darkening": shadow_darkening,
        "overlap_order_parity": overlap_order_parity,
        "overlap_strategy_parity": overlap_strategy_parity,
    }
    print(json.dumps(result, indent=2))
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(result, indent=2) + "\n")

    if empty_parity["max_abs"] > args.max_empty_volume_error:
        raise SystemExit(
            "FAIL: a zero-density volume changes surface shading "
            f"(max_abs={empty_parity['max_abs']:.8g})"
        )
    if strategy_parity["relative_rmse"] > args.max_strategy_relative_rmse:
        raise SystemExit(
            "FAIL: volume execution strategies diverge "
            f"(relative_rmse={strategy_parity['relative_rmse']:.8g})"
        )
    if volume_effect["relative_rmse"] < args.min_volume_effect:
        raise SystemExit("FAIL: active-volume capture is vacuously unchanged")
    if shadow_darkening < args.min_shadow_darkening:
        raise SystemExit(
            "FAIL: absorbing volume does not attenuate direct lighting "
            f"(relative_darkening={shadow_darkening:.8g})"
        )
    if overlap_order_parity["max_abs"] > args.max_overlap_order_error:
        raise SystemExit(
            "FAIL: overlapping media depend on scene insertion order "
            f"(max_abs={overlap_order_parity['max_abs']:.8g})"
        )
    if overlap_strategy_parity["relative_rmse"] > args.max_strategy_relative_rmse:
        raise SystemExit(
            "FAIL: overlapping-volume execution strategies diverge "
            f"(relative_rmse={overlap_strategy_parity['relative_rmse']:.8g})"
        )
    print(
        "PASS: volume compositing, overlap, shadows, and execution parity "
        "are stable"
    )


if __name__ == "__main__":
    main()
