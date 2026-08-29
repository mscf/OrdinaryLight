"""Validate lazy auto-strategy replacement on one persistent Vulkan device."""

import argparse
import json
from pathlib import Path

import numpy as np

import ordinarylight as ol
from ordinarylight.showcases.rooms import get_restir_scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    if args.width < 1 or args.height < 1:
        parser.error("dimensions must be positive")
    config = ol.RendererConfig(
        wavefront_execution_strategy="auto",
        samples_per_pixel=1,
        max_bounces=6,
        wavefront_tile_capacity=args.width * args.height,
    )
    sequence = ("diffuse", "glossy_glass", "diffuse")
    observations = []
    with ol.VulkanRayTracingBackend(config=config) as renderer:
        for frame_index, name in enumerate(sequence):
            spec = get_restir_scene(name)
            scene = spec.build()
            camera = ol.PerspectiveCamera(
                position=(0.0, spec.camera_height, -spec.orbit_radius),
                target=spec.target,
            )
            image = renderer.render_wavefront(
                scene, camera, args.width, args.height,
                samples=1, frame_index=frame_index,
            )
            observations.append({
                "scene": name,
                "strategy": renderer._core.wavefront_executor.strategy,
                "finite": bool(np.all(np.isfinite(image))),
                "maximum": float(np.max(image[..., :3])),
            })
    report = {
        "extent": [args.width, args.height],
        "observations": observations,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if args.gate:
        strategies = [item["strategy"] for item in observations]
        if strategies != ["wavefront", "megakernel", "wavefront"]:
            raise SystemExit(
                f"FAIL: unexpected auto-strategy transition {strategies}"
            )
        if not all(item["finite"] and item["maximum"] > 0.0
                   for item in observations):
            raise SystemExit("FAIL: strategy transition produced invalid HDR")
    print("PASS: lazy auto-strategy transition gate")


if __name__ == "__main__":
    main()
