"""Gate HDR parity across all execution strategies."""

import argparse
import json
from pathlib import Path

import numpy as np

import ordinarylight as ol
from ordinarylight.showcases.rooms import SCENES, get_restir_scene


def _write_ppm(path, image):
    rgb = np.maximum(np.asarray(image)[..., :3], 0.0)
    mapped = rgb / (1.0 + rgb)
    encoded = np.clip(mapped ** (1.0 / 2.2) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    height, width, _ = encoded.shape
    Path(path).write_bytes(f"P6\n{width} {height}\n255\n".encode() + encoded.tobytes())


def _difference_image(a, b):
    difference = np.abs(np.asarray(a)[..., :3] - np.asarray(b)[..., :3])
    scale = np.percentile(difference, 99.0)
    return difference / max(float(scale), 1e-8)


def _render(
    strategy, scene, camera, args, *, persistent_coarse_tiles=False,
    persistent_continuations=False,
):
    config = ol.RendererConfig(
        max_bounces=args.bounces,
        samples_per_pixel=args.samples,
        area_light_samples=args.area_light_samples,
        wavefront_secondary_area_light_samples=(
            args.secondary_area_light_samples
        ),
        wavefront_environment_samples=args.environment_samples,
        wavefront_secondary_nee_probability=args.secondary_nee_probability,
        wavefront_tile_capacity=args.width * args.height,
        wavefront_execution_strategy=strategy,
        wavefront_ser=(strategy == "ser"),
        wavefront_persistent_coarse_tiles=persistent_coarse_tiles,
        wavefront_persistent_continuations=persistent_continuations,
        wavefront_fused_secondary=True,
        wavefront_subgroup_enqueue=True,
        wavefront_scene_specialization=True,
        wavefront_megakernel_single_warp=(
            args.megakernel_single_warp and strategy == "megakernel"
        ),
    )
    with ol.VulkanRayTracingBackend(config=config) as renderer:
        # The backend readback is a view over mapped Vulkan memory.  Keep the
        # pixels alive after the backend (and its allocation) is destroyed.
        return np.array(renderer.render_wavefront(
            scene, camera, args.width, args.height,
            samples=args.samples, frame_index=args.seed,
        ), copy=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--bounces", type=int, default=16)
    parser.add_argument("--area-light-samples", type=int, default=2)
    parser.add_argument("--secondary-area-light-samples", type=int, default=0)
    parser.add_argument("--environment-samples", type=int, default=1)
    parser.add_argument("--secondary-nee-probability", type=float, default=1.0)
    parser.add_argument("--megakernel-single-warp", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-relative-rmse", type=float, default=0.01)
    parser.add_argument(
        "--output-prefix", default="/tmp/ordinarylight_execution_parity"
    )
    parser.add_argument("--summary", type=Path)
    parser.add_argument(
        "--scene", choices=("feature_parity", *SCENES),
        default="feature_parity",
    )
    args = parser.parse_args()

    if args.scene == "feature_parity":
        scene = ol.build_feature_parity_scene()
        camera = ol.feature_parity_camera()
    else:
        scene_spec = get_restir_scene(args.scene)
        scene = scene_spec.build()
        camera = ol.PerspectiveCamera(
            position=(0.0, scene_spec.camera_height, -scene_spec.orbit_radius),
            target=scene_spec.target,
        )
    print("Rendering canonical wavefront implementation...")
    wavefront = _render("wavefront", scene, camera, args)
    finite_rgb = np.nan_to_num(wavefront[..., :3], copy=False)
    visible_fraction = float(np.mean(np.max(finite_rgb, axis=-1) > 1e-6))
    if visible_fraction < 0.01:
        raise SystemExit(
            "FAIL: canonical capture has less than 1% visible-pixel "
            "coverage; parity would be vacuous"
        )
    print("Rendering feature-equivalent megakernel implementation...")
    megakernel = _render("megakernel", scene, camera, args)
    print("Rendering SER ray-generation megakernel implementation...")
    ser = _render("ser", scene, camera, args)
    print("Rendering hybrid implementation...")
    hybrid = _render("hybrid", scene, camera, args)
    print("Rendering persistent-continuation hybrid implementation...")
    hybrid_persistent = _render(
        "hybrid", scene, camera, args, persistent_continuations=True
    )
    print("Rendering persistent implementation...")
    persistent = _render("persistent", scene, camera, args)
    print("Rendering coarse-tile persistent implementation...")
    persistent_coarse = _render(
        "persistent", scene, camera, args, persistent_coarse_tiles=True
    )
    comparisons = {
        "megakernel": ol.image_error_metrics(wavefront, megakernel),
        "ser": ol.image_error_metrics(wavefront, ser),
        "hybrid": ol.image_error_metrics(wavefront, hybrid),
        "hybrid_persistent": ol.image_error_metrics(
            wavefront, hybrid_persistent
        ),
        "persistent": ol.image_error_metrics(wavefront, persistent),
        "persistent_coarse": ol.image_error_metrics(
            wavefront, persistent_coarse
        ),
    }
    for strategy, metrics in comparisons.items():
        print(strategy)
        for name, value in metrics.items():
            print(f"  {name}={value:.8g}")
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps({
            "scene": args.scene,
            "extent": [args.width, args.height],
            "samples": args.samples,
            "visible_fraction": visible_fraction,
            "comparisons": comparisons,
        }, indent=2) + "\n")

    prefix = Path(args.output_prefix)
    _write_ppm(prefix.with_name(prefix.name + "_wavefront.ppm"), wavefront)
    _write_ppm(prefix.with_name(prefix.name + "_megakernel.ppm"), megakernel)
    _write_ppm(prefix.with_name(prefix.name + "_hybrid.ppm"), hybrid)
    _write_ppm(
        prefix.with_name(prefix.name + "_hybrid_persistent.ppm"),
        hybrid_persistent,
    )
    _write_ppm(prefix.with_name(prefix.name + "_persistent.ppm"), persistent)
    _write_ppm(
        prefix.with_name(prefix.name + "_persistent_coarse.ppm"),
        persistent_coarse,
    )
    _write_ppm(
        prefix.with_name(prefix.name + "_difference.ppm"),
        _difference_image(wavefront, megakernel),
    )
    for strategy, metrics in comparisons.items():
        if metrics["relative_rmse"] > args.max_relative_rmse:
            raise SystemExit(
                f"FAIL: {strategy} relative RMSE "
                f"{metrics['relative_rmse']:.6g} exceeds "
                f"{args.max_relative_rmse:.6g}"
            )
    print("PASS: execution strategies are within the configured HDR tolerance")


if __name__ == "__main__":
    main()
