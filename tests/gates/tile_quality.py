"""Require tiled wavefront execution to preserve deterministic HDR output."""

import argparse

import numpy as np

import ordinarylight as ol
from ordinarylight.integrations.temporal_quality import summarize_temporal_quality


def _render(scene, camera, args, capacity):
    config = ol.RendererConfig(
        max_bounces=args.bounces,
        samples_per_pixel=args.samples,
        wavefront_tile_capacity=capacity,
        wavefront_execution_strategy=args.strategy,
        wavefront_persistent_continuations=(args.strategy == "hybrid"),
        wavefront_fused_secondary=True,
        wavefront_subgroup_enqueue=True,
    )
    with ol.renderers.gi.VulkanGlobalIlluminationRenderer(config=config) as renderer:
        return renderer.render_wavefront(
            scene, camera, args.width, args.height,
            samples=args.samples, frame_index=args.seed,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--bounces", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--tile-capacity", type=int, default=32768)
    parser.add_argument("--strategy", default="hybrid")
    parser.add_argument("--max-absolute", type=float, default=0.0)
    args = parser.parse_args()
    pixels = args.width * args.height
    if not 1 <= args.tile_capacity < pixels:
        parser.error("tile-capacity must be positive and smaller than the image")

    scene = ol.build_feature_parity_scene()
    camera = ol.feature_parity_camera()
    print(f"Rendering one {pixels}-pixel tile...")
    reference = _render(scene, camera, args, pixels)
    print(f"Rendering tiles with capacity {args.tile_capacity}...")
    candidate = _render(scene, camera, args, args.tile_capacity)

    metrics = ol.image_error_metrics(reference, candidate)
    directional = summarize_temporal_quality(
        np.asarray([reference]), np.asarray([candidate])
    )
    for name, value in metrics.items():
        print(f"{name}={value:.8g}")
    print(
        "horizontal_band_rms="
        f"{directional['horizontal_band_rms_max']:.8g}"
    )
    print(f"band_anisotropy={directional['band_anisotropy_max']:.8g}")
    if metrics["max_absolute"] > args.max_absolute:
        raise SystemExit(
            "FAIL: tiled HDR max absolute error "
            f"{metrics['max_absolute']:.8g} exceeds {args.max_absolute:.8g}"
        )
    print("PASS: tiled and single-tile HDR output match")


if __name__ == "__main__":
    main()
