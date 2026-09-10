"""Compare independent-path and shared-primary ReSTIR at fixed image settings."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.showcases.catalog.raster import SHOWCASES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--showcase", default="raster-material-program-room")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("/tmp/shared-restir"))
    parser.add_argument("--history-limit", type=int, default=4,
                        help="4 isolates fresh sampling; 20 exercises temporal reuse")
    parser.add_argument("--motion", type=float, default=0.0,
                        help="camera orbit in radians across captured frames")
    args = parser.parse_args()
    if min(args.width, args.height, args.frames) < 1 or args.warmup < 0:
        parser.error("dimensions and frames must be positive")
    item = next(item for item in SHOWCASES if item.id == args.showcase)
    scene = item.build()
    camera = item.camera.camera(scene, angle=-0.45)
    args.output.mkdir(parents=True, exist_ok=True)
    glfw = load_glfw()
    if not glfw.init():
        raise RuntimeError("GLFW initialization failed")
    window = None
    results = {}
    means = {}
    first_frames = {}
    try:
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        window = glfw.create_window(args.width, args.height, "ReSTIR comparison", None, None)
        if not window:
            raise RuntimeError("could not create comparison window")
        cases = [(False, n, 1) for n in (1, 2, 4)] + [
            (True, n, 1) for n in (1, 2, 4)
        ] + [(True, 2, 2), (True, 1, 8)]
        for shared, reservoirs, spp in cases:
            name = f"{'shared' if shared else 'legacy'}-r{reservoirs}-s{spp}"
            print("starting", name, flush=True)
            # Force both paths through the same current GLSL material compiler,
            # including for a built-in-only scene, to avoid stale SPIR-V comparisons.
            config = ol.RendererConfig(
                material_program=replace(ol.builtin_material),
                samples_per_pixel=spp, max_bounces=8,
                wavefront_restir_di=True,
                wavefront_restir_shared_primary=shared,
                wavefront_restir_reservoirs=reservoirs,
                wavefront_restir_candidates=4,
                wavefront_restir_history_limit=args.history_limit,
                wavefront_restir_spatial_reuse=False,
                wavefront_timestamps=True,
                wavefront_hdr_capture=True,
                direct_swapchain_storage=False,
                denoiser_enabled=False,
            )
            pixels = []
            gpu_ms = []
            tiles = []
            reuse_frames = 0
            with ol.VulkanGlfwPresenter(window, config=config) as renderer:
                if shared and spp == 8:
                    # Use disjoint random frames for the noisy reference, so
                    # shared samples do not favor one test case's RMSE.
                    renderer._core.wavefront_frame_sequence = 100000
                for frame in range(args.frames + args.warmup):
                    camera = item.camera.camera(
                        scene, angle=-0.45 + args.motion
                        * max(frame - args.warmup, 0) / max(args.frames - 1, 1),
                    )
                    if args.history_limit <= 4:
                        # Explicitly disable temporal eligibility for the fresh-
                        # sampling comparison. The legacy shader can otherwise
                        # reuse geometry-incompatible samples despite this cap.
                        renderer._core.wavefront_previous_present_camera = None
                    glfw.poll_events()
                    renderer.present_wavefront(scene, camera, args.width, args.height)
                    image = renderer.capture_wavefront_hdr()
                    history_valid = renderer.last_timings["wavefront_restir_history_valid"]
                    if shared and spp > 1:
                        assert not history_valid, "multiple paths reused single-hit guides"
                    if frame == 0:
                        first_frames[name] = image.copy()
                    if frame >= args.warmup:
                        pixels.append(image[..., :3].copy())
                        gpu_ms.append(renderer.last_timings["gpu_frame_ms"])
                        tiles.append(renderer.last_timings["wavefront_tiles"])
                        reuse_frames += int(history_valid)
            pixels = np.asarray(pixels)
            means[name] = pixels.mean(axis=0)
            np.save(args.output / f"{name}-mean.npy", means[name])
            results[name] = {
                "gpu_ms_median": float(np.median(gpu_ms)),
                "tiles": int(np.median(tiles)),
                "mean_radiance": float(pixels.mean()),
                "temporal_variance": float(pixels.var(axis=0).mean()),
                "finite": bool(np.isfinite(pixels).all()),
                "temporal_reuse_frames": reuse_frames,
                "frame_mean_standard_error": float(
                    pixels.mean(axis=(1, 2, 3)).std() / np.sqrt(args.frames)
                ),
            }
            print(name, json.dumps(results[name]), flush=True)
        reference = means["shared-r1-s8"]
        for name in results:
            results[name]["mean_image_rmse_to_s8"] = float(
                np.sqrt(np.mean((means[name] - reference) ** 2))
            )
        parity = float(np.max(np.abs(
            first_frames["legacy-r1-s1"] - first_frames["shared-r1-s1"]
        )))
        report = {"showcase": args.showcase, "extent": [args.width, args.height],
                  "frames": args.frames, "history_limit": args.history_limit,
                  "motion": args.motion,
                  "one_reservoir_first_frame_max_abs_difference": parity,
                  "results": results}
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print("report:", args.output / "report.json", "one-reservoir difference:", parity)
        assert all(r["finite"] for r in results.values()), "non-finite HDR output"
        assert parity < 0.002, "one-reservoir output regression"
        assert results["shared-r4-s1"]["tiles"] == results["shared-r1-s1"]["tiles"]
        assert results["shared-r2-s2"]["tiles"] == 2 * results["shared-r1-s1"]["tiles"]
        if args.width * args.height * args.frames >= 1000000:
            baseline = results["shared-r1-s1"]["mean_radiance"]
            for name in ("shared-r2-s1", "shared-r4-s1", "shared-r2-s2"):
                assert abs(results[name]["mean_radiance"] - baseline) < max(
                    0.05 * abs(baseline), 1e-6,
                ), f"radiance normalization regression: {name}"
    finally:
        if window:
            glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    main()
