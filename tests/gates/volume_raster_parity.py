"""Gate Vulkan/WebGPU HDR parity across the authored volume showcases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import ordinarylight as ol
from ordinarylight.outputs import to_sdr
from ordinarylight.showcases.catalog.volumes import SHOWCASES


def _render(showcase, backend_name, extent):
    scene = showcase.create_scene()
    camera = showcase.camera.camera(
        scene, showcase.camera.arc_radians or 0.0,
    )
    target = "spirv" if backend_name == "vulkan" else "wgsl"
    implementation = (
        ol.renderers.raster.VulkanRasterRenderer
        if backend_name == "vulkan" else
        ol.renderers.raster.WebGpuRasterRenderer
    )
    program = ol.RasterProgram.scene(
        target=target, validate=False,
        material_programs=scene.material_programs(ol.builtin_material),
    )
    config = ol.RasterConfig(
        **showcase.renderer,
        state=ol.RasterState(cull_mode="none"),
    )
    with ol.Renderer(
        implementation=implementation(program, config=config),
    ) as renderer:
        return renderer.render(scene, camera, extent)


def _metrics(reference, candidate):
    reference_rgb = np.asarray(reference[..., :3], np.float32)
    candidate_rgb = np.asarray(candidate[..., :3], np.float32)
    difference = candidate_rgb - reference_rgb
    reference_rms = max(
        float(np.sqrt(np.mean(reference_rgb * reference_rgb))), 1e-8,
    )
    return {
        "relative_rmse": float(
            np.sqrt(np.mean(difference * difference)) / reference_rms
        ),
        "mean_absolute_error": float(np.mean(np.abs(difference))),
        "maximum_absolute_error": float(np.max(np.abs(difference))),
    }


def _save_sdr(path, hdr):
    Image.fromarray(to_sdr(hdr, alpha=True)).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=90)
    parser.add_argument("--max-relative-rmse", type=float, default=0.01)
    parser.add_argument(
        "--output", type=Path,
        default=Path("/tmp/ordinarylight_volume_raster_parity"),
    )
    args = parser.parse_args()
    if args.width < 1 or args.height < 1:
        parser.error("width and height must be positive")
    if args.max_relative_rmse <= 0.0:
        parser.error("max-relative-rmse must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        "extent": [args.width, args.height],
        "max_relative_rmse": args.max_relative_rmse,
        "showcases": {},
    }
    failures = []
    for showcase in SHOWCASES:
        print(f"Rendering {showcase.id} through Vulkan raster...")
        vulkan = _render(showcase, "vulkan", (args.width, args.height))
        print(f"Rendering {showcase.id} through WebGPU raster...")
        webgpu = _render(showcase, "webgpu", (args.width, args.height))
        metrics = _metrics(vulkan, webgpu)
        report["showcases"][showcase.id] = metrics
        print(
            f"  relative_rmse={metrics['relative_rmse']:.6g} "
            f"mae={metrics['mean_absolute_error']:.6g} "
            f"max={metrics['maximum_absolute_error']:.6g}"
        )
        _save_sdr(args.output / f"{showcase.id}_vulkan.png", vulkan)
        _save_sdr(args.output / f"{showcase.id}_webgpu.png", webgpu)
        difference = np.abs(vulkan[..., :3] - webgpu[..., :3])
        difference_scale = max(float(np.max(difference)), 1e-8)
        difference_rgba = np.concatenate((
            difference / difference_scale,
            np.ones((*difference.shape[:2], 1), np.float32),
        ), axis=2)
        _save_sdr(
            args.output / f"{showcase.id}_difference.png", difference_rgba,
        )
        if metrics["relative_rmse"] > args.max_relative_rmse:
            failures.append(showcase.id)

    (args.output / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    if failures:
        raise SystemExit(
            "FAIL: raster volume parity exceeded tolerance for "
            + ", ".join(failures)
        )
    print("PASS: Vulkan and WebGPU volume HDR paths are within tolerance")


if __name__ == "__main__":
    main()
