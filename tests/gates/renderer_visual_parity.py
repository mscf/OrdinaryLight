"""Gate shared visual semantics between Vulkan raster and GI renderers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import ordinarylight as ol
from ordinarylight.outputs import to_sdr


def terminator_serration(image, mask):
    """Measure high-frequency horizontal steps near a shaded lower terminator."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 3:
        return 0.0
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    width, height = x1 - x0, y1 - y0
    left = x0 + round(width * 0.12)
    right = x0 + round(width * 0.88)
    top = y0 + round(height * 0.60)
    bottom = y0 + round(height * 0.92)
    if right - left < 3 or bottom <= top:
        return 0.0
    luminance = (
        np.asarray(image, np.float32)[..., :3] / 255.0
        @ np.asarray((0.2126, 0.7152, 0.0722), np.float32)
    )
    crop = luminance[top:bottom, left:right]
    valid = mask[top:bottom, left:right]
    second_difference = np.abs(
        crop[:, 2:] - 2.0 * crop[:, 1:-1] + crop[:, :-2]
    )
    valid_triplets = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
    if not np.any(valid_triplets):
        return 0.0
    # A one-percent SDR second difference separates real terminator teeth
    # from smooth tone-mapped gradients while remaining exposure invariant.
    return float(np.mean(second_difference[valid_triplets] > 0.01))


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
    optical_quality="environment", optical_debug_view="off", *, shadows=True,
    ambient_light=0.08, screen_space_ray_steps=24,
    screen_space_optical_layers=4,
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
            ambient_light=ambient_light,
            shading_model="pbr",
            tone_mapping="none",
            optical_quality=optical_quality,
            screen_space_ray_steps=screen_space_ray_steps,
            screen_space_optical_layers=screen_space_optical_layers,
            optical_debug_view=optical_debug_view,
            shadows=shadows,
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
            "point-shadows",
            "material-room",
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
        "--max-terminator-serration", type=float,
        help="maximum lower-terminator high-frequency fraction per object",
    )
    parser.add_argument(
        "--raster-optics", choices=("environment", "screen-space"),
        help=(
            "raster optical quality tier (defaults to screen-space for the "
            "thin-transmission showcase and environment otherwise)"
        ),
    )
    parser.add_argument(
        "--raster-optical-debug-view",
        choices=(
            "off", "hit", "uv", "depth-delta", "confidence", "object-id",
            "depth-trace", "refraction-hit", "refraction-uv",
            "refraction-source",
        ),
        default="off",
        help="capture a raster optical diagnostic instead of final shading",
    )
    parser.add_argument(
        "--disable-raster-shadows", action="store_true",
        help="capture the raster candidate without native shadow maps",
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
    if args.raster_optics is None:
        args.raster_optics = (
            "screen-space" if args.scene == "thin-transmission"
            else "environment"
        )
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
            args.max_object_log_luminance_error = accepted.get(
                "max_object_log_luminance_error",
                args.max_object_log_luminance_error,
            )
            args.min_object_edge_correlation = accepted.get(
                "min_object_edge_correlation",
                args.min_object_edge_correlation,
            )
            args.max_terminator_serration = accepted.get(
                "max_terminator_serration", args.max_terminator_serration,
            )
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
    elif args.scene == "point-shadows":
        from ordinarylight.showcases.raster_features import (
            build_point_shadow_scene,
        )
        scene = build_point_shadow_scene()
        camera = ol.PerspectiveCamera((0.0, 4.2, 8.5), (0.0, 0.9, 0.0))
    elif args.scene in {"materials", "modifier", "material-room"}:
        from ordinarylight.showcases.raster_features import (
            build_material_program_parity_scene, build_material_program_room_scene,
        )
        if args.scene == "material-room":
            scene = build_material_program_room_scene()
            camera = ol.PerspectiveCamera((0,3.6,10),(0,1.3,0))
            args.raster_optics = "screen-space"
        else:
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
        args.raster_optical_debug_view,
        shadows=not args.disable_raster_shadows,
        ambient_light=(0.015 if args.scene == "material-room" else 0.08),
        screen_space_ray_steps=(64 if args.scene == "material-room" else 24),
    )
    metrics = ol.renderer_visual_metrics(
        gi.color, raster.color,
        reference_mask=gi.object_id > 0,
        candidate_mask=raster.object_id > 0,
    )
    luminance_weights = np.asarray((0.2126, 0.7152, 0.0722), np.float32)
    candidate_sdr = to_sdr(raster.color, exposure=metrics["exposure_scale"])
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
            "terminator_serration": terminator_serration(
                candidate_sdr, raster.object_id == object_id,
            ),
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
            "max_terminator_serration": args.max_terminator_serration,
        },
    }
    args.output.mkdir(parents=True, exist_ok=True)
    exposure = metrics["exposure_scale"]
    Image.fromarray(to_sdr(gi.color), "RGB").save(args.output / "gi.png")
    Image.fromarray(candidate_sdr, "RGB").save(
        args.output / "raster.png"
    )
    np.save(args.output / "raster_object_id.npy", raster.object_id)
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
    def selected_object(item):
        return (
            args.object_prefix is None
            or (item["name"] is not None
                and item["name"].startswith(args.object_prefix))
        )
    if args.max_object_log_luminance_error is not None and any(
        item["absolute_log_luminance_error"] >
        args.max_object_log_luminance_error
        for item in object_metrics.values()
        if selected_object(item)
    ):
        failures.append("per-object luminance")
    if args.min_object_edge_correlation is not None and any(
        item["crop_edge_correlation"] < args.min_object_edge_correlation
        for item in object_metrics.values()
        if selected_object(item)
    ):
        failures.append("per-object edge correlation")
    if args.max_terminator_serration is not None and any(
        item["terminator_serration"] > args.max_terminator_serration
        for item in object_metrics.values()
        if item["name"] is not None and selected_object(item)
    ):
        failures.append("terminator serration")
    if failures and not args.capture_only:
        raise SystemExit("FAIL: raster/GI parity: " + ", ".join(failures))
    print(
        "CAPTURE: raster/GI visual evidence written"
        if failures else "PASS: raster/GI shared visual semantics"
    )


if __name__ == "__main__":
    main()
