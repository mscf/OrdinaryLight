"""Gate shared visual semantics between Vulkan raster and GI renderers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import ordinarylight as ol
from ordinarylight.outputs import to_sdr


def _render_gi(
    scene, camera, extent, samples, material_modifier=None, *, max_bounces=8,
):
    implementation = ol.renderers.gi.VulkanGlobalIlluminationRenderer(
        config=ol.RendererConfig(
            samples_per_pixel=samples,
            max_bounces=max_bounces,
            wavefront_hdr_capture=True,
            material_modifier=material_modifier,
        )
    )
    with ol.Renderer(implementation=implementation) as renderer:
        return renderer.render(
            scene, camera, extent, outputs=("color", "object_id"),
        )


def _render_raster(
    scene, camera, extent, material_modifier=None,
    optical_quality="environment",
):
    program = ol.RasterProgram.scene(
        target="spirv",
        material_programs=scene.material_programs(ol.builtin_material),
        material_modifier=material_modifier,
    )
    implementation = ol.renderers.raster.VulkanRasterRenderer(
        program,
        config=ol.RasterConfig(
            state=ol.RasterState(cull_mode="none"),
            ambient_light=0.08,
            shading_model="pbr",
            tone_mapping="none",
            optical_quality=optical_quality,
        ),
    )
    with ol.Renderer(implementation=implementation) as renderer:
        return renderer.render(
            scene, camera, extent, outputs=("color", "object_id"),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--bounces", type=int, default=8)
    parser.add_argument(
        "--camera-pose",
        help="JSON camera pose, in the same format copied by the parity viewer",
    )
    parser.add_argument(
        "--scene", choices=(
            "feature", "materials", "modifier", "clearcoat", "sheen",
            "anisotropy", "thin-transmission", "subsurface",
            "environment-reflection", "refraction", "absorption",
            "nested-dielectric", "transparency",
        ), default="feature",
        help="shared scene semantics to compare",
    )
    parser.add_argument("--max-log-color-rmse", type=float, default=0.45)
    parser.add_argument(
        "--max-object-log-luminance-error", type=float,
        help="optional maximum absolute log mean-luminance error per object",
    )
    parser.add_argument(
        "--object-prefix",
        help="limit per-object luminance enforcement to matching mesh names",
    )
    parser.add_argument(
        "--min-object-edge-correlation", type=float,
        help="minimum edge correlation inside each matching object crop",
    )
    parser.add_argument("--min-edge-correlation", type=float, default=0.35)
    parser.add_argument("--min-coverage-iou", type=float, default=0.65)
    parser.add_argument(
        "--raster-optics", choices=("environment", "screen-space"),
        default="environment",
        help="raster optical quality tier (environment remains the baseline)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("/tmp/ordinarylight_renderer_visual_parity"),
    )
    parser.add_argument(
        "--capture-only", action="store_true",
        help="write evidence without enforcing thresholds",
    )
    args = parser.parse_args()
    if args.width < 1 or args.height < 1 or args.samples < 1 or args.bounces < 1:
        parser.error("dimensions and samples must be positive")
    extent = (args.width, args.height)
    advanced_baseline = Path(__file__).with_name("baselines").joinpath(
        "advanced_material_parity.json"
    )
    # Scene-wide accepted thresholds describe the default camera. Explicit
    # camera poses carry their own baselines because coverage and projected
    # edge correlation legitimately change with viewpoint.
    if advanced_baseline.is_file() and not args.camera_pose:
        accepted = json.loads(advanced_baseline.read_text())["scenes"].get(
            args.scene
        )
        if accepted is not None:
            args.max_log_color_rmse = accepted["max_log_color_rmse"]
            args.min_edge_correlation = accepted["min_edge_correlation"]
            args.min_coverage_iou = accepted["min_coverage_iou"]
    material_modifier = None
    advanced_builders = {}
    if args.scene in {
        "environment-reflection", "refraction", "absorption",
        "nested-dielectric", "transparency",
    }:
        from ordinarylight.showcases.optical_materials import (
            build_absorption_scene, build_environment_reflection_scene,
            build_nested_dielectric_scene, build_refraction_scene,
            build_transparency_scene,
        )
        optical_builders = {
            "environment-reflection": build_environment_reflection_scene,
            "refraction": build_refraction_scene,
            "absorption": build_absorption_scene,
            "nested-dielectric": build_nested_dielectric_scene,
            "transparency": build_transparency_scene,
        }
        scene = optical_builders[args.scene]()
        camera = ol.PerspectiveCamera((0,3.2,10),(0,1,0))
    elif args.scene in {
        "clearcoat", "sheen", "anisotropy", "thin-transmission", "subsurface",
    }:
        from ordinarylight.showcases.advanced_materials import (
            build_anisotropy_scene, build_clearcoat_scene, build_sheen_scene,
            build_subsurface_scene, build_thin_transmission_scene,
        )
        advanced_builders = {
            "clearcoat": build_clearcoat_scene,
            "sheen": build_sheen_scene,
            "anisotropy": build_anisotropy_scene,
            "thin-transmission": build_thin_transmission_scene,
            "subsurface": build_subsurface_scene,
        }
        scene = advanced_builders[args.scene]()
        camera = ol.PerspectiveCamera((0,3.2,10),(0,1,0))
    elif args.scene in {"materials", "modifier"}:
        from ordinarylight.showcases.raster_features import (
            build_material_program_parity_scene,
        )
        scene = build_material_program_parity_scene()
        camera = ol.PerspectiveCamera((0,3.2,10),(0,1,0))
        if args.scene == "modifier":
            from ordinarylight.showcases.raster_material_hooks import (
                advanced_surface_showcase_modifier,
            )
            material_modifier = advanced_surface_showcase_modifier
    else:
        scene = ol.build_feature_parity_scene()
        camera = ol.feature_parity_camera()
    if args.camera_pose:
        pose = json.loads(args.camera_pose)
        camera = ol.PerspectiveCamera(
            pose["position"], pose["target"], pose.get("up", (0.0, 1.0, 0.0)),
            vertical_fov_degrees=pose.get("vertical_fov_degrees", 45.0),
        )
    print("Rendering GI reference...")
    gi = _render_gi(
        scene, camera, extent, args.samples, material_modifier,
        max_bounces=args.bounces,
    )
    print("Rendering raster candidate...")
    raster = _render_raster(
        scene, camera, extent, material_modifier, args.raster_optics,
    )
    metrics = ol.renderer_visual_metrics(
        gi.color, raster.color,
        reference_mask=gi.object_id > 0,
        candidate_mask=raster.object_id > 0,
    )
    luminance_weights = np.asarray((0.2126, 0.7152, 0.0722), np.float32)
    object_metrics = {}
    object_names = {
        int(mesh.id): mesh.name for mesh in scene.visible_meshes
        if mesh.id is not None
    }
    object_ids = sorted(set(np.unique(gi.object_id).tolist()) &
                        set(np.unique(raster.object_id).tolist()))
    for object_id in object_ids:
        if int(object_id) <= 0:
            continue
        mask = (gi.object_id == object_id) & (raster.object_id == object_id)
        if not np.any(mask):
            continue
        reference_luminance = float(np.mean(
            gi.color[mask][..., :3] @ luminance_weights
        ))
        candidate_luminance = float(np.mean(
            raster.color[mask][..., :3] @ luminance_weights
        )) * float(metrics["exposure_scale"])
        log_error = float(abs(np.log(
            max(candidate_luminance, 1e-6) /
            max(reference_luminance, 1e-6)
        )))
        ys, xs = np.nonzero(mask)
        crop_metrics = ol.renderer_visual_metrics(
            gi.color[ys.min():ys.max() + 1, xs.min():xs.max() + 1],
            raster.color[ys.min():ys.max() + 1, xs.min():xs.max() + 1],
        )
        object_metrics[str(int(object_id))] = {
            "name": object_names.get(int(object_id), ""),
            "pixels": int(np.count_nonzero(mask)),
            "reference_mean_luminance": reference_luminance,
            "candidate_mean_luminance": candidate_luminance,
            "absolute_log_luminance_error": log_error,
            "crop_edge_correlation": crop_metrics["edge_correlation"],
            "crop_log_color_rmse": crop_metrics["log_color_rmse"],
        }
    report = {
        "extent": list(extent),
        "samples": args.samples,
        "bounces": args.bounces,
        "scene": args.scene,
        "raster_optics": args.raster_optics,
        "metrics": metrics,
        "object_metrics": object_metrics,
        "thresholds": {
            "max_log_color_rmse": args.max_log_color_rmse,
            "min_edge_correlation": args.min_edge_correlation,
            "min_coverage_iou": args.min_coverage_iou,
            "max_object_log_luminance_error": (
                args.max_object_log_luminance_error
            ),
            "min_object_edge_correlation": (
                args.min_object_edge_correlation
            ),
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    exposure = metrics["exposure_scale"]
    Image.fromarray(to_sdr(gi.color), "RGB").save(args.output / "gi.png")
    Image.fromarray(to_sdr(raster.color, exposure=exposure), "RGB").save(
        args.output / "raster.png"
    )
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    failures = []
    if metrics["log_color_rmse"] > args.max_log_color_rmse:
        failures.append("log color RMSE")
    if metrics["edge_correlation"] < args.min_edge_correlation:
        failures.append("edge correlation")
    if metrics["coverage_iou"] < args.min_coverage_iou:
        failures.append("coverage IoU")
    if args.max_object_log_luminance_error is not None and any(
        item["absolute_log_luminance_error"] >
        args.max_object_log_luminance_error
        for item in object_metrics.values()
        if (args.object_prefix is None
            or item["name"].startswith(args.object_prefix))
    ):
        failures.append("per-object luminance")
    if args.min_object_edge_correlation is not None and any(
        item["crop_edge_correlation"] < args.min_object_edge_correlation
        for item in object_metrics.values()
        if (args.object_prefix is None
            or item["name"].startswith(args.object_prefix))
    ):
        failures.append("per-object edge correlation")
    if failures and not args.capture_only:
        raise SystemExit("FAIL: raster/GI parity: " + ", ".join(failures))
    print(
        "CAPTURE: raster/GI visual evidence written"
        if failures else "PASS: raster/GI shared visual semantics"
    )


if __name__ == "__main__":
    main()
