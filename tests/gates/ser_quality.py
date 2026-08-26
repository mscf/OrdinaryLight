"""Gate compute, raygen, and SER through the fused presentation path."""

import argparse
import json
import os
import statistics
from pathlib import Path

import numpy as np
import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw

from ordinarylight.showcases.rooms import SCENES, get_restir_scene


def _capture(window, scene, camera, args, mode):
    raygen = mode in {"raygen", "ser"}
    specialized = mode == "compute_specialized"
    config = ol.RendererConfig(
        max_bounces=args.bounces,
        samples_per_pixel=args.samples,
        wavefront_tile_capacity=min(args.width * args.height, 131_072),
        wavefront_execution_strategy="ser" if raygen else "megakernel",
        wavefront_ser=raygen,
        wavefront_ser_reorder=(mode == "ser"),
        wavefront_scene_specialization=True,
        wavefront_untextured_specialization=(
            args.untextured_specialization and specialized
        ),
        wavefront_untextured_specialization_part=args.untextured_part,
        wavefront_subgroup_enqueue=not args.no_subgroup_enqueue,
        wavefront_hdr_capture=True,
        direct_swapchain_storage=False,
    )
    with ol.VulkanGlfwPresenter(window, config=config) as presenter:
        gpu_times = []
        for frame in range(args.warmup_frames + args.benchmark_frames):
            presenter.present_wavefront(scene, camera, args.width, args.height)
            if frame >= args.warmup_frames:
                value = presenter.last_timings.get("gpu_frame_ms", 0.0)
                if value > 0.0:
                    gpu_times.append(value)
        image = presenter.capture_wavefront_hdr()
        median_gpu_ms = statistics.median(gpu_times) if gpu_times else 0.0
        return image, median_gpu_ms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--bounces", type=int, default=8)
    parser.add_argument("--warmup-frames", type=int, default=3)
    parser.add_argument("--benchmark-frames", type=int, default=10)
    parser.add_argument(
        "--window-scale", type=float,
        default=float(os.environ.get("WAVE_RENDER_DESKTOP_SCALE", "1.0")),
        help=(
            "framebuffer/logical scale; the project X11 backend is normally "
            "1.0 (override for a genuinely scaled backend)"
        ),
    )
    parser.add_argument("--min-framebuffer-ratio", type=float, default=0.98)
    parser.add_argument("--scene", choices=tuple(SCENES), default="diffuse")
    parser.add_argument("--max-relative-rmse", type=float, default=0.01)
    parser.add_argument("--metric-stride", type=int, default=0)
    parser.add_argument("--save-captures", action="store_true")
    parser.add_argument("--untextured-specialization", action="store_true")
    parser.add_argument(
        "--untextured-part", choices=("primary", "secondary", "full"),
        default="full",
    )
    parser.add_argument("--no-subgroup-enqueue", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=Path("/tmp/ordinarylight_ser_quality"),
    )
    args = parser.parse_args()

    glfw = load_glfw()
    if not glfw.init():
        raise RuntimeError("GLFW initialization failed")
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    if args.window_scale <= 0.0:
        raise ValueError("window scale must be positive")
    logical_width = max(1, round(args.width / args.window_scale))
    logical_height = max(1, round(args.height / args.window_scale))
    window = glfw.create_window(
        logical_width, logical_height, __doc__, None, None
    )
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW Vulkan window creation failed")
    framebuffer_width, framebuffer_height = glfw.get_framebuffer_size(window)
    width_ratio = framebuffer_width / args.width
    height_ratio = framebuffer_height / args.height
    print(
        f"target={args.width}x{args.height} "
        f"logical={logical_width}x{logical_height} "
        f"framebuffer={framebuffer_width}x{framebuffer_height} "
        f"scale={args.window_scale:.3f}"
    )
    if not (
        args.min_framebuffer_ratio <= width_ratio
        <= 1.0 / args.min_framebuffer_ratio
        and args.min_framebuffer_ratio <= height_ratio
        <= 1.0 / args.min_framebuffer_ratio
    ):
        glfw.destroy_window(window)
        glfw.terminate()
        raise RuntimeError(
            "framebuffer does not match the requested render target; "
            "adjust --window-scale"
        )
    spec = get_restir_scene(args.scene)
    scene = spec.build()
    camera = ol.PerspectiveCamera(
        position=(0.0, spec.camera_height, -spec.orbit_radius),
        target=spec.target,
    )
    try:
        modes = ["compute"]
        if args.untextured_specialization:
            modes.append("compute_specialized")
        modes.extend(("raygen", "ser"))
        results = {
            mode: _capture(window, scene, camera, args, mode)
            for mode in modes
        }
    finally:
        glfw.destroy_window(window)
        glfw.terminate()

    captures = {mode: result[0] for mode, result in results.items()}
    gpu_ms = {mode: result[1] for mode, result in results.items()}
    fps = {
        mode: (1000.0 / value if value > 0.0 else 0.0)
        for mode, value in gpu_ms.items()
    }
    args.output.mkdir(parents=True, exist_ok=True)
    if args.save_captures:
        np.savez_compressed(args.output / "captures.npz", **captures)
    metric_stride = args.metric_stride or max(
        1, int(np.ceil(np.sqrt(args.width * args.height / 1_000_000)))
    )
    scored = {
        mode: image[::metric_stride, ::metric_stride]
        for mode, image in captures.items()
    }
    visible = {
        mode: float(np.mean(np.max(image[..., :3], axis=-1) > 1e-6))
        for mode, image in scored.items()
    }
    compared_modes = [mode for mode in captures if mode != "compute"]
    metrics = {
        mode: ol.image_error_metrics(scored["compute"], scored[mode])
        for mode in compared_modes
    }
    summary = {
        "visible_fraction": visible,
        "metrics": metrics,
        "median_gpu_ms": gpu_ms,
        "gpu_fps": fps,
        "metric_stride": metric_stride,
        "target_extent": [args.width, args.height],
        "logical_extent": [logical_width, logical_height],
        "framebuffer_extent": [framebuffer_width, framebuffer_height],
        "window_scale": args.window_scale,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    for mode in captures:
        if visible[mode] < 0.01:
            raise SystemExit(f"FAIL: {mode} capture is effectively black")
    for mode in compared_modes:
        if metrics[mode]["relative_rmse"] > args.max_relative_rmse:
            raise SystemExit(f"FAIL: {mode} exceeds HDR tolerance")
    print("PASS: fused presentation outputs agree")


if __name__ == "__main__":
    main()
