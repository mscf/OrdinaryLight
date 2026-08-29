"""Compare the generated Ordinary Shade shade stage with production GLSL."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

import ordinarylight as ol
from ordinarylight.showcases.rooms import SCENES, get_restir_scene
from ordinarylight.showcases.multivolume import build_multivolume_showcase
from ordinarylight.showcases.volume_multiple_scattering import (
    build_volume_multiple_scattering_showcase,
)
from ordinarylight.showcases.volume_scattering import (
    build_volume_scattering_showcase,
)
from ordinarylight.showcases.volumes import build_volume_showcase


VOLUME_SCENES = {
    "volume": build_volume_showcase,
    "multivolume": build_multivolume_showcase,
    "volume_scattering": build_volume_scattering_showcase,
    "volume_multiple_scattering": build_volume_multiple_scattering_showcase,
}


@ol.material
def _parity_material(ctx):
    """Deterministic specialization used to gate generated material dispatch."""
    return ol.MaterialEvaluation(
        base_color=ol.mix(ctx.base_color, ol.vec3(0.12, 0.32, 0.78), 0.2),
        emission=ctx.emission,
        metallic=ctx.metallic,
        roughness=ctx.roughness * 0.8 + 0.05,
        transmission=ctx.transmission,
        ior=ctx.ior,
        attenuation_color=ctx.attenuation_color,
        attenuation_distance=ctx.attenuation_distance,
    )


@ol.material
def _attribute_material(ctx):
    tint = ctx.attribute("parity_tint", components=3)
    return ol.MaterialEvaluation(
        base_color=ctx.base_color * tint,
        emission=ctx.emission,
        metallic=ctx.metallic,
        roughness=ctx.roughness,
        transmission=ctx.transmission,
        ior=ctx.ior,
        attenuation_color=ctx.attenuation_color,
        attenuation_distance=ctx.attenuation_distance,
    )


def _build_attribute_scene():
    scene = ol.Scene()
    scene.set_environment(color=(0.025, 0.035, 0.06))
    scene.add_point_light((-2.0, 4.5, -2.0), intensity=85.0)
    material = ol.Material(
        base_color=(0.9, 0.9, 0.9), roughness=0.68,
        program=_attribute_material,
    )
    floor = (
        (-4.0, 0.0, -3.0), (4.0, 0.0, -3.0),
        (4.0, 0.0, 3.0), (-4.0, 0.0, 3.0),
    )
    scene.add_mesh(
        floor, ((0, 1, 2), (0, 2, 3)), material,
        attributes={"parity_tint": (
            (0.35, 0.65, 1.0), (1.0, 0.45, 0.3),
            (0.35, 1.0, 0.55), (0.75, 0.45, 1.0),
        )},
    )
    back = (
        (-4.0, 0.0, 3.0), (4.0, 0.0, 3.0),
        (4.0, 5.0, 3.0), (-4.0, 5.0, 3.0),
    )
    scene.add_mesh(
        back, ((0, 2, 1), (0, 3, 2)), material,
        attributes={"parity_tint": (
            (0.9, 0.45, 0.35), (0.35, 0.55, 1.0),
            (0.9, 0.8, 0.35), (0.4, 0.9, 0.65),
        )},
    )
    return scene, ol.PerspectiveCamera(
        position=(0.0, 2.2, -6.5), target=(0.0, 1.3, 0.5)
    )


def _write_ppm(path, image):
    rgb = np.maximum(np.asarray(image)[..., :3], 0.0)
    mapped = rgb / (1.0 + rgb)
    encoded = np.clip(mapped ** (1.0 / 2.2) * 255.0 + 0.5, 0, 255).astype(
        np.uint8
    )
    height, width, _ = encoded.shape
    Path(path).write_bytes(
        f"P6\n{width} {height}\n255\n".encode() + encoded.tobytes()
    )


def _render(scene, camera, args, *, generated):
    config = ol.RendererConfig(
        material_program=_parity_material if args.custom_material else None,
        max_bounces=args.bounces,
        samples_per_pixel=args.samples,
        area_light_samples=args.area_light_samples,
        wavefront_secondary_area_light_samples=args.secondary_area_light_samples,
        wavefront_environment_samples=args.environment_samples,
        wavefront_tile_capacity=min(
            args.width * args.height, args.tile_capacity
        ),
        wavefront_execution_strategy="wavefront",
        wavefront_fused_secondary=False,
        wavefront_scene_specialization=False,
        wavefront_ordinaryshade_shade=generated,
        wavefront_native_textures=args.native_textures,
        vulkan_pipeline_cache=not args.disable_pipeline_cache,
        vulkan_pipeline_cache_path=(
            str(args.pipeline_cache_path)
            if args.pipeline_cache_path is not None else None
        ),
    )
    construction_started = time.perf_counter()
    renderer = ol.renderers.gi.VulkanGlobalIlluminationRenderer(config=config)
    construction_ms = (time.perf_counter() - construction_started) * 1000.0
    try:
        # Exclude lazy driver pipeline compilation and first-use allocation
        # from the steady-state comparison while exercising the same stage.
        first_render_started = time.perf_counter()
        renderer.render_wavefront(
            scene, camera, args.width, args.height,
            samples=args.samples, frame_index=args.seed,
        )
        first_render_ms = (time.perf_counter() - first_render_started) * 1000.0
        timings = []
        image = None
        for _ in range(args.timing_runs):
            started = time.perf_counter()
            image = renderer.render_wavefront(
                scene, camera, args.width, args.height,
                samples=args.samples, frame_index=args.seed,
            )
            timings.append((time.perf_counter() - started) * 1000.0)
        validation_products = renderer.render_wavefront_outputs(
            scene, camera, args.width, args.height,
            outputs=("color", "depth"),
            samples=args.samples, frame_index=args.seed,
        )
        validation_image = np.array(validation_products["color"], copy=True)
        primary_hit_fraction = float(np.mean(np.isfinite(
            validation_products["depth"]
        )))
        return validation_image, {
            "construction_ms": construction_ms,
            "first_render_ms": first_render_ms,
            "blocking_render_median_ms": float(np.median(timings)),
            "blocking_render_samples_ms": timings,
            "primary_hit_fraction": primary_hit_fraction,
        }
    finally:
        renderer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--bounces", type=int, default=8)
    parser.add_argument("--tile-capacity", type=int, default=2_097_152)
    parser.add_argument("--timing-runs", type=int, default=3)
    parser.add_argument("--area-light-samples", type=int, default=2)
    parser.add_argument("--secondary-area-light-samples", type=int, default=1)
    parser.add_argument("--environment-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-relative-rmse", type=float, default=0.02)
    parser.add_argument("--output", type=Path, default=Path(
        "/tmp/ordinarylight_ordinaryshade_shade_parity"
    ))
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--native-textures", action="store_true")
    parser.add_argument("--custom-material", action="store_true")
    parser.add_argument("--custom-attributes", action="store_true")
    parser.add_argument("--disable-pipeline-cache", action="store_true")
    parser.add_argument("--pipeline-cache-path", type=Path)
    parser.add_argument(
        "--scene", choices=("feature_parity", *SCENES, *VOLUME_SCENES),
        default="feature_parity",
    )
    args = parser.parse_args()
    if (
        args.width <= 0 or args.height <= 0 or args.tile_capacity <= 0
        or args.timing_runs <= 0
    ):
        parser.error("capture dimensions must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    if args.custom_attributes:
        scene, camera = _build_attribute_scene()
    elif args.scene == "feature_parity":
        scene = ol.build_feature_parity_scene()
        camera = ol.feature_parity_camera()
    elif args.scene in SCENES:
        scene_spec = get_restir_scene(args.scene)
        scene = scene_spec.build()
        camera = ol.PerspectiveCamera(
            position=(0.0, scene_spec.camera_height, -scene_spec.orbit_radius),
            target=scene_spec.target,
        )
    else:
        scene = VOLUME_SCENES[args.scene](32)
        camera = ol.PerspectiveCamera(
            position=(0.0, 3.1, -8.2), target=(0.0, 1.25, -0.5)
        )
    production, production_stats = _render(
        scene, camera, args, generated=False
    )
    generated, generated_stats = _render(
        scene, camera, args, generated=True
    )
    metrics = ol.image_error_metrics(production, generated)
    generated_rgb = np.asarray(generated)[..., :3]
    output_validation = {
        "finite": bool(np.all(np.isfinite(generated_rgb))),
        "visible_fraction": float(np.mean(
            np.max(np.nan_to_num(generated_rgb), axis=-1) > 1e-6
        )),
        "maximum": float(np.max(np.nan_to_num(generated_rgb))),
        "dynamic_range": float(
            np.max(np.nan_to_num(generated_rgb))
            - np.min(np.nan_to_num(generated_rgb))
        ),
        "primary_hit_fraction": generated_stats["primary_hit_fraction"],
    }
    report = {
        "extent": [args.width, args.height],
        "samples": args.samples,
        "bounces": args.bounces,
        "tile_capacity": min(
            args.width * args.height, args.tile_capacity
        ),
        "seed": args.seed,
        "scene": args.scene,
        "native_textures": args.native_textures,
        "custom_material": args.custom_material,
        "custom_attributes": args.custom_attributes,
        "pipeline_cache": not args.disable_pipeline_cache,
        "pipeline_cache_path": (
            str(args.pipeline_cache_path)
            if args.pipeline_cache_path is not None else None
        ),
        "metrics": metrics,
        "output_validation": output_validation,
        "production_statistics": production_stats,
        "generated_statistics": generated_stats,
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )
    np.save(args.output / "production.npy", production)
    np.save(args.output / "generated.npy", generated)
    _write_ppm(args.output / "production.ppm", production)
    _write_ppm(args.output / "generated.ppm", generated)
    difference = np.abs(production[..., :3] - generated[..., :3])
    scale = max(float(np.percentile(difference, 99.0)), 1e-8)
    _write_ppm(args.output / "difference.ppm", difference / scale)
    print(json.dumps(report, indent=2, default=str))
    if args.gate:
        if metrics["relative_rmse"] > args.max_relative_rmse:
            raise SystemExit(
                "FAIL: generated shade stage exceeds the HDR parity tolerance"
            )
        if not output_validation["finite"]:
            raise SystemExit("FAIL: generated shade stage produced non-finite HDR")
        if output_validation["primary_hit_fraction"] < 0.01:
            raise SystemExit(
                "FAIL: generated shade stage has less than 1% primary-hit coverage"
            )
    print("PASS: generated shade-stage comparison completed")


if __name__ == "__main__":
    main()
