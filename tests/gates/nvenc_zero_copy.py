"""Opt-in 4K Vulkan/CUDA/NVENC correctness and performance gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import tempfile
import time

import ordinarylight as ol


def _scene():
    scene = ol.Scene()
    scene.add_mesh(
        ((-2, -1, 0), (2, -1, 0), (0, 2, 0)), ((0, 1, 2),),
        ol.Material(base_color=(0.8, 0.2, 0.08)),
    )
    scene.add_point_light((2, 3, -2), intensity=25.0)
    return scene, ol.PerspectiveCamera((0, 0, -4), (0, 0, 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=3840)
    parser.add_argument("--height", type=int, default=2160)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--maximum-median-ms", type=float, default=16.67)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.frames <= args.warmup:
        parser.error("frames must be greater than warmup")
    destination = args.output or Path(tempfile.gettempdir()) / (
        "ordinarylight-zero-copy-gate.h264"
    )
    scene, camera = _scene()
    samples = []
    gpu_samples = []
    with ol.Renderer(config=ol.RendererConfig(
        external_image_interop=True
    )) as renderer, ol.outputs.NvencVideoWriter(
        destination, (args.width, args.height), fps=60
    ) as video:
        for index in range(args.frames):
            started = time.perf_counter()
            frame = renderer.render_gpu(
                scene, camera, (args.width, args.height),
                frame_index=index, pixel_format="nv12",
            )
            video.write(frame)
            if index >= args.warmup:
                samples.append((time.perf_counter() - started) * 1000.0)
                gpu_samples.append(renderer.last_statistics.gpu_ms)
    report = {
        "extent": [args.width, args.height],
        "frames": args.frames,
        "encoded_bytes": destination.stat().st_size,
        "median_total_ms": statistics.median(samples),
        "maximum_total_ms": max(samples),
        "median_gpu_ms": statistics.median(gpu_samples),
        "effective_fps": 1000.0 / statistics.median(samples),
        "maximum_median_ms": args.maximum_median_ms,
    }
    print(json.dumps(report, indent=2))
    if report["encoded_bytes"] <= 0:
        raise SystemExit("FAIL: NVENC produced an empty stream")
    if report["median_total_ms"] > args.maximum_median_ms:
        raise SystemExit("FAIL: zero-copy median exceeds threshold")
    print("PASS: 4K Vulkan/CUDA/NVENC zero-copy gate")


if __name__ == "__main__":
    main()
