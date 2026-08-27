"""Opt-in 4K Vulkan/CUDA/NVENC correctness and performance gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
import time

import ordinarylight as ol


def _probe_stream(path):
    executable = shutil.which("ffprobe")
    if executable is None:
        return None
    completed = subprocess.run(
        [
            executable, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,pix_fmt,width,height",
            "-of", "json", str(path),
        ], check=True, capture_output=True, text=True,
    )
    streams = json.loads(completed.stdout).get("streams", [])
    return streams[0] if streams else None


def _probe_keyframe_count(path):
    executable = shutil.which("ffprobe")
    if executable is None:
        return None
    completed = subprocess.run(
        [
            executable, "-v", "error", "-select_streams", "v:0",
            "-show_entries", "frame=key_frame", "-of", "csv=p=0",
            str(path),
        ], check=True, capture_output=True, text=True,
    )
    return sum(line.strip().startswith("1") for line in completed.stdout.splitlines())


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
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--dynamic-motion-samples", action="store_true")
    parser.add_argument("--target-fps", type=float, default=60.0)
    parser.add_argument("--maximum-median-ms", type=float, default=16.67)
    parser.add_argument("--keyframe-interval-seconds", type=float)
    parser.add_argument(
        "--pixel-format", choices=("nv12", "p010"), default="nv12"
    )
    parser.add_argument("--codec", choices=("h264", "hevc", "av1"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.frames <= args.warmup:
        parser.error("frames must be greater than warmup")
    codec = args.codec or ("hevc" if args.pixel_format == "p010" else "h264")
    if args.pixel_format == "p010" and codec == "h264":
        parser.error("P010 requires --codec hevc or av1")
    suffix = {"h264": "h264", "hevc": "h265", "av1": "av1"}[codec]
    destination = args.output or Path(tempfile.gettempdir()) / (
        f"ordinarylight-zero-copy-{args.pixel_format}-gate.{suffix}"
    )
    scene, camera = _scene()
    samples = []
    gpu_samples = []
    with ol.Renderer(config=ol.RendererConfig(
        external_image_interop=True,
        samples_per_pixel=args.samples,
        stationary_delay_seconds=(60.0 if args.dynamic_motion_samples else 0.15),
        wavefront_interactive_target_fps=(
            args.target_fps if args.dynamic_motion_samples else None
        ),
        wavefront_interactive_sample_scaling=args.dynamic_motion_samples,
    )) as renderer, ol.outputs.NvencVideoWriter(
        destination, (args.width, args.height), fps=60,
        codec=codec, pixel_format=args.pixel_format,
        keyframe_interval_seconds=args.keyframe_interval_seconds,
    ) as video:
        effective_samples = []
        accumulation_states = []
        dynamic_sample_active = []
        for index in range(args.frames):
            started = time.perf_counter()
            frame = renderer.render_gpu(
                scene, camera, (args.width, args.height),
                frame_index=index, pixel_format=args.pixel_format,
            )
            video.write(frame)
            effective_samples.append(renderer.effective_samples_per_pixel)
            accumulation_states.append(renderer.accumulation_state.value)
            dynamic_sample_active.append(bool(
                renderer.last_statistics.timings.get(
                    "wavefront_interactive_sample_scaling", False
                )
            ))
            if index >= args.warmup:
                samples.append((time.perf_counter() - started) * 1000.0)
                gpu_samples.append(renderer.last_statistics.gpu_ms)
        forced_keyframes = video.forced_keyframe_count
    stream = _probe_stream(destination)
    decoded_keyframes = _probe_keyframe_count(destination)
    report = {
        "extent": [args.width, args.height],
        "frames": args.frames,
        "pixel_format": args.pixel_format,
        "codec": codec,
        "encoded_bytes": destination.stat().st_size,
        "forced_keyframes": forced_keyframes,
        "decoded_keyframes": decoded_keyframes,
        "effective_samples": effective_samples,
        "accumulation_states": accumulation_states,
        "dynamic_sample_active": dynamic_sample_active,
        "median_total_ms": statistics.median(samples),
        "maximum_total_ms": max(samples),
        "median_gpu_ms": statistics.median(gpu_samples),
        "effective_fps": 1000.0 / statistics.median(samples),
        "maximum_median_ms": args.maximum_median_ms,
        "bitstream": stream,
    }
    print(json.dumps(report, indent=2))
    if report["encoded_bytes"] <= 0:
        raise SystemExit("FAIL: NVENC produced an empty stream")
    expected_pixel_format = (
        "yuv420p10le" if args.pixel_format == "p010" else "yuv420p"
    )
    if stream is not None and stream.get("pix_fmt") != expected_pixel_format:
        raise SystemExit(
            "FAIL: encoded bitstream pixel format is "
            f"{stream.get('pix_fmt')!r}, expected {expected_pixel_format!r}"
        )
    if (
        args.keyframe_interval_seconds is not None
        and decoded_keyframes is not None
        and decoded_keyframes < forced_keyframes + 1
    ):
        raise SystemExit(
            "FAIL: bitstream does not contain the opening keyframe and all "
            "requested recovery keyframes"
        )
    if report["median_total_ms"] > args.maximum_median_ms:
        raise SystemExit("FAIL: zero-copy median exceeds threshold")
    print(
        f"PASS: Vulkan/CUDA/NVENC {args.pixel_format.upper()} "
        "zero-copy gate"
    )


if __name__ == "__main__":
    main()
