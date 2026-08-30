"""Gate shared visual semantics between Vulkan raster and GI renderers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

import ordinarylight as ol
from ordinarylight.outputs import to_sdr


def _render_gi(scene, camera, extent, samples):
    implementation = ol.renderers.gi.VulkanGlobalIlluminationRenderer(
        config=ol.RendererConfig(
            samples_per_pixel=samples,
            max_bounces=6,
            wavefront_hdr_capture=True,
        )
    )
    with ol.Renderer(implementation=implementation) as renderer:
        return renderer.render(
            scene, camera, extent, outputs=("color", "object_id"),
        )


def _render_raster(scene, camera, extent):
    program = ol.RasterProgram.scene(
        target="spirv",
        material_programs=scene.material_programs(ol.builtin_material),
    )
    implementation = ol.renderers.raster.VulkanRasterRenderer(
        program,
        config=ol.RasterConfig(
            state=ol.RasterState(cull_mode="none"),
            ambient_light=0.08,
            shading_model="pbr",
            tone_mapping="none",
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
    parser.add_argument(
        "--scene", choices=("feature", "materials"), default="feature",
        help="shared scene semantics to compare",
    )
    parser.add_argument("--max-log-color-rmse", type=float, default=0.45)
    parser.add_argument("--min-edge-correlation", type=float, default=0.35)
    parser.add_argument("--min-coverage-iou", type=float, default=0.65)
    parser.add_argument(
        "--output", type=Path,
        default=Path("/tmp/ordinarylight_renderer_visual_parity"),
    )
    parser.add_argument(
        "--capture-only", action="store_true",
        help="write evidence without enforcing thresholds",
    )
    args = parser.parse_args()
    if args.width < 1 or args.height < 1 or args.samples < 1:
        parser.error("dimensions and samples must be positive")
    extent = (args.width, args.height)
    if args.scene == "materials":
        from ordinarylight.showcases.raster_features import (
            build_material_program_parity_scene,
        )
        scene = build_material_program_parity_scene()
        camera = ol.PerspectiveCamera((0,3.2,10),(0,1,0))
    else:
        scene = ol.build_feature_parity_scene()
        camera = ol.feature_parity_camera()
    print("Rendering GI reference...")
    gi = _render_gi(scene, camera, extent, args.samples)
    print("Rendering raster candidate...")
    raster = _render_raster(scene, camera, extent)
    metrics = ol.renderer_visual_metrics(
        gi.color, raster.color,
        reference_mask=gi.object_id > 0,
        candidate_mask=raster.object_id > 0,
    )
    report = {
        "extent": list(extent),
        "samples": args.samples,
        "scene": args.scene,
        "metrics": metrics,
        "thresholds": {
            "max_log_color_rmse": args.max_log_color_rmse,
            "min_edge_correlation": args.min_edge_correlation,
            "min_coverage_iou": args.min_coverage_iou,
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
    if failures and not args.capture_only:
        raise SystemExit("FAIL: raster/GI parity: " + ", ".join(failures))
    print(
        "CAPTURE: raster/GI visual evidence written"
        if failures else "PASS: raster/GI shared visual semantics"
    )


if __name__ == "__main__":
    main()
