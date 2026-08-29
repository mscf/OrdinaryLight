"""Gate HDR and performance for volume empty-space skipping."""

import argparse
import json
import time

import numpy as np

import ordinarylight as ol


def build_sparse_volume_scene(resolution=129):
    coordinates = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    z, y, x = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    radius = np.sqrt(x * x + y * y + z * z)
    density = np.where(radius < 0.08, 1.0 - radius / 0.08, 0.0).astype(np.float32)
    scene = ol.Scene()
    scene.add_volume(
        density,
        ol.VolumeMaterial(
            ol.Texture1D(((0.0, 0.0, 0.0, 0.0),
                          (0.7, 0.25, 0.08, 0.16))),
            # Fine sampling makes this a meaningful sparse-traversal benchmark
            # instead of a benchmark dominated by fixed readback overhead.
            density_scale=1.0, emission_scale=1.0, step_size=0.0005,
        ),
        transform=(
            ol.Transform.translation((-1.0, -1.0, -1.0))
            @ ol.Transform.scale((2.0, 2.0, 2.0))
        ),
        value_range=(0.0, 1.0), name="sparse-emissive-sphere",
    )
    return scene


def _capture(scene, camera, args, enabled):
    config = ol.RendererConfig(
        max_bounces=1, samples_per_pixel=1,
        wavefront_tile_capacity=args.width * args.height,
        wavefront_execution_strategy="megakernel",
        volume_empty_space_skipping=enabled,
    )
    timings = []
    last = None
    with ol.renderers.gi.VulkanGlobalIlluminationRenderer(config=config) as renderer:
        for frame in range(args.warmup + args.frames):
            start = time.perf_counter()
            last = np.array(renderer.render_wavefront(
                scene, camera, args.width, args.height,
                samples=1, frame_index=args.seed,
            ), copy=True)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if frame >= args.warmup:
                timings.append(elapsed_ms)
    return last, np.asarray(timings, np.float64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--frames", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=23)
    # This is an end-to-end readback benchmark, so the threshold deliberately
    # guards a modest reproducible win rather than claiming the much larger
    # traversal-only gain hidden behind fixed synchronization/copy costs.
    parser.add_argument("--minimum-speedup", type=float, default=1.03)
    parser.add_argument("--maximum-relative-rmse", type=float, default=1e-6)
    args = parser.parse_args()
    scene = build_sparse_volume_scene()
    camera = ol.PerspectiveCamera((0.0, 0.0, -3.2), (0.0, 0.0, 0.0))
    statistics = ol.volume_empty_space_statistics(scene.visible_volumes)

    if args.rounds < 1:
        parser.error("--rounds must be positive")
    captures = {False: [], True: []}
    images = {}
    # Reverse the order every round to cancel clock warm-up, thermal drift,
    # and unrelated system load that otherwise bias two sequential captures.
    for round_index in range(args.rounds):
        order = (False, True) if round_index % 2 == 0 else (True, False)
        for enabled in order:
            image, timing = _capture(scene, camera, args, enabled)
            captures[enabled].append(timing)
            images.setdefault(enabled, image)
    baseline = images[False]
    accelerated = images[True]
    baseline_times = np.concatenate(captures[False])
    accelerated_times = np.concatenate(captures[True])
    difference = accelerated[..., :3] - baseline[..., :3]
    relative_rmse = float(
        np.sqrt(np.mean(difference * difference))
        / max(np.sqrt(np.mean(baseline[..., :3] ** 2)), 1e-8)
    )
    maximum_absolute_error = float(np.max(np.abs(difference)))
    baseline_median = float(np.median(baseline_times))
    accelerated_median = float(np.median(accelerated_times))
    speedup = baseline_median / max(accelerated_median, 1e-8)
    result = {
        "extent": [args.width, args.height],
        "rounds": args.rounds,
        "occupancy": statistics,
        "baseline_median_ms": baseline_median,
        "accelerated_median_ms": accelerated_median,
        "speedup": speedup,
        "maximum_absolute_error": maximum_absolute_error,
        "relative_rmse": relative_rmse,
    }
    print(json.dumps(result, indent=2))
    if relative_rmse > args.maximum_relative_rmse:
        raise SystemExit("FAIL: empty-space skipping changes HDR output")
    if speedup < args.minimum_speedup:
        raise SystemExit("FAIL: sparse-volume speedup is below threshold")
    print("PASS: empty-space skipping preserves HDR and accelerates sparse media")


if __name__ == "__main__":
    main()
