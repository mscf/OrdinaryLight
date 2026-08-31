"""Native pixel gate for point shadows and mixed-light accumulation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import ordinarylight as ol
from ordinarylight.outputs import to_sdr
from ordinarylight.showcases.raster_features import (
    build_directional_shadow_scene,
    build_multi_light_shadow_scene,
    build_point_shadow_scene,
    build_spot_shadow_scene,
)


def _capture(scene, camera, extent, target, *, shadows):
    shader_target = "spirv" if target == "vulkan" else "wgsl"
    program = ol.RasterProgram.scene(
        target=shader_target,
        material_programs=scene.material_programs(ol.builtin_material),
    )
    renderer_type = (
        ol.renderers.raster.VulkanRasterRenderer
        if target == "vulkan" else
        ol.renderers.raster.WebGpuRasterRenderer
    )
    implementation = renderer_type(
        program,
        config=ol.RasterConfig(
            shadows=shadows, shadow_map_size=256, ambient_light=0.06,
            tone_mapping="none",
        ),
    )
    with ol.Renderer(implementation=implementation) as renderer:
        return renderer.render(scene, camera, extent, outputs=("color",))["color"]


def _shadow_metrics(lit, shadowed):
    difference = np.abs(lit[..., :3] - shadowed[..., :3])
    magnitude = np.max(difference, axis=2)
    return {
        "changed_fraction": float(np.mean(magnitude > 1e-3)),
        "mean_absolute_difference": float(np.mean(difference)),
        "maximum_absolute_difference": float(np.max(difference)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=("vulkan", "webgpu"), default="vulkan")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument(
        "--output", type=Path, default=Path("/tmp/ordinarylight_raster_lighting"),
    )
    args = parser.parse_args()
    if args.width < 1 or args.height < 1:
        parser.error("dimensions must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    baseline_path = Path(__file__).with_name("baselines") / "raster_lighting.json"
    baseline = json.loads(baseline_path.read_text())
    extent = (args.width, args.height)
    report = {
        "target": args.target,
        "extent": extent,
        "baseline": baseline,
        "scenes": {},
    }
    failures = []
    unshadowed_captures = {}
    for name, builder, camera in (
        (
            "point", build_point_shadow_scene,
            ol.PerspectiveCamera(
                (-3.3387779031968217, 6.215172679768388,
                 6.613735820046667),
                (0.0, 0.9, 0.0),
            ),
        ),
        (
            "directional", build_directional_shadow_scene,
            ol.PerspectiveCamera((0.0, 4.2, 8.5), (0.0, 0.9, 0.0)),
        ),
        (
            "spot", build_spot_shadow_scene,
            ol.PerspectiveCamera((0.0, 4.2, 8.5), (0.0, 0.9, 0.0)),
        ),
        (
            "mixed", build_multi_light_shadow_scene,
            ol.PerspectiveCamera((0.0, 4.2, 8.5), (0.0, 0.9, 0.0)),
        ),
    ):
        scene = builder()
        without = _capture(scene, camera, extent, args.target, shadows=False)
        unshadowed_captures[name] = without
        with_shadows = _capture(scene, camera, extent, args.target, shadows=True)
        metrics = _shadow_metrics(without, with_shadows)
        report["scenes"][name] = metrics
        Image.fromarray(to_sdr(without)).save(args.output / f"{name}-unshadowed.png")
        Image.fromarray(to_sdr(with_shadows)).save(args.output / f"{name}-shadowed.png")
        accepted = baseline["scenes"][name]
        if metrics["changed_fraction"] < accepted["min_changed_fraction"]:
            failures.append(f"{name}: shadowed-pixel coverage regressed")
        if (
            metrics["mean_absolute_difference"]
            < accepted["min_mean_absolute_difference"]
        ):
            failures.append(f"{name}: shadow response strength regressed")
    light_array_metrics = _shadow_metrics(
        unshadowed_captures["point"], unshadowed_captures["mixed"],
    )
    report["multi_light_accumulation"] = light_array_metrics
    accepted = baseline["multi_light_accumulation"]
    if light_array_metrics["changed_fraction"] < accepted["min_changed_fraction"]:
        failures.append("mixed-light pixel coverage regressed")
    if (
        light_array_metrics["mean_absolute_difference"]
        < accepted["min_mean_absolute_difference"]
    ):
        failures.append("mixed-light accumulation strength regressed")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        return 1
    print("PASS: native point and mixed-light shadow pixels are active")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
