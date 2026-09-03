"""Guard ReLAX against triangle-aligned noise on smooth matte meshes."""

import argparse
import json
from pathlib import Path
import statistics

import numpy as np

import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.showcases.raster_features import build_material_program_room_scene


DEFAULT_BASELINE = (
    Path(__file__).with_name("baselines")
    / "relax_tessellation_quality.json"
)


def _config(*, reference, width, height):
    return ol.RendererConfig(
        max_bounces=8,
        samples_per_pixel=16 if reference else 1,
        area_light_samples=2,
        wavefront_restir_di=not reference,
        progressive_accumulation=not reference,
        temporal_history=not reference,
        temporal_history_limit=32,
        denoiser_enabled=not reference,
        denoiser_iterations=3,
        wavefront_hdr_capture=True,
        wavefront_tile_capacity=width * height,
        direct_swapchain_storage=False,
    )


def _capture(window, scene, camera, frames, width, height, *, reference):
    result = np.empty((frames, height, width, 4), np.float32)
    timings = []
    with ol.VulkanGlfwPresenter(
        window, config=_config(reference=reference, width=width, height=height),
    ) as presenter:
        for index in range(frames):
            presenter.present_wavefront(scene, camera, width, height)
            result[index] = presenter.capture_wavefront_hdr()
            timings.append(float(presenter.last_timings.get("gpu_frame_ms", 0.0)))
    return result, timings


def _projected_edge_mask(scene, camera, width, height):
    subject = next(
        mesh for mesh in scene.render_meshes
        if mesh.name == "material-program-subject-0"
    )
    origin = np.asarray(camera.position, np.float64)
    forward = np.asarray(camera.target, np.float64) - origin
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(camera.up, np.float64))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = np.asarray(subject.world_vertices, np.float64) - origin
    distance = relative @ forward
    scale = np.tan(np.radians(camera.vertical_fov_degrees) * 0.5)
    pixels = np.column_stack((
        ((relative @ right) / (distance * scale * (width / height)) + 1.0)
        * 0.5 * width,
        (-(relative @ up) / (distance * scale) + 1.0) * 0.5 * height,
    ))
    mask = np.zeros((height, width), bool)
    edges = np.concatenate((
        subject.indices[:, (0, 1)], subject.indices[:, (1, 2)],
        subject.indices[:, (2, 0)],
    ))
    for start, end in edges:
        a, b = pixels[start], pixels[end]
        count = max(2, int(np.ceil(np.max(np.abs(b - a)))) + 1)
        points = np.rint(np.linspace(a, b, count)).astype(np.int32)
        valid = (
            (points[:, 0] >= 0) & (points[:, 0] < width)
            & (points[:, 1] >= 0) & (points[:, 1] < height)
        )
        mask[points[valid, 1], points[valid, 0]] = True
    return mask


def summarize(reference, candidate, projected_edges=None):
    weights = np.asarray((0.2126, 0.7152, 0.0722), np.float32)
    reference_rgb = np.maximum(reference[..., :3], 0.0)
    candidate_rgb = np.maximum(candidate[..., :3], 0.0)
    stable_reference = np.mean(reference_rgb, axis=0)
    # The close fixed pose makes the cyan diffuse sphere the dominant blue,
    # non-emissive subject. Erode the color mask to exclude its silhouette.
    mask = (
        (stable_reference[..., 2] > stable_reference[..., 0] * 1.35)
        & (stable_reference[..., 1] > stable_reference[..., 0] * 1.15)
        & (stable_reference[..., 2] < 4.0)
    )
    for _ in range(2):
        padded = np.pad(mask, 1, mode="constant")
        mask = np.logical_and.reduce([
            padded[y:y + mask.shape[0], x:x + mask.shape[1]]
            for y in range(3) for x in range(3)
        ])
    if np.count_nonzero(mask) < 256:
        raise RuntimeError("matte sphere mask is unexpectedly small")
    reference_luma = reference_rgb @ weights
    candidate_luma = candidate_rgb @ weights
    scale = max(float(np.sqrt(np.mean(reference_luma[:, mask] ** 2))), 1e-8)
    dark = np.maximum(reference_luma - candidate_luma, 0.0) / scale
    # Triangle seams are sparse; a tail statistic catches them without being
    # diluted by the otherwise smooth sphere interior.
    dark_tail = [float(np.percentile(frame[mask], 99.5)) for frame in dark]
    summary = {
        "dark_outlier_p995_p95": float(np.percentile(dark_tail, 95.0)),
        "masked_pixels": int(np.count_nonzero(mask)),
    }
    if projected_edges is not None:
        edge = mask & projected_edges
        interior = mask & ~projected_edges
        edge_dark = np.mean(dark[:, edge], axis=1)
        interior_dark = np.mean(dark[:, interior], axis=1)
        summary["triangle_edge_dark_excess_p95"] = float(np.percentile(
            np.maximum(edge_dark - interior_dark, 0.0), 95.0
        ))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=288)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--output", type=Path,
        default=Path("/tmp/ordinarylight_relax_tessellation"),
    )
    parser.add_argument("--accept-baseline", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    glfw = load_glfw()
    if not glfw.init():
        raise RuntimeError("GLFW initialization failed")
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(
        args.width, args.height, "ReLAX tessellation gate", None, None
    )
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW Vulkan window creation failed")
    camera = ol.PerspectiveCamera(
        (-7.704328335344836, 1.5426652500598559, 2.395024384194161),
        (0.0, 1.3, 0.0),
    )
    try:
        scene = build_material_program_room_scene()
        reference, _ = _capture(
            window, scene, camera, args.frames, args.width, args.height,
            reference=True,
        )
        candidate, timings = _capture(
            window, scene, camera, args.frames, args.width, args.height,
            reference=False,
        )
    finally:
        glfw.destroy_window(window)
        glfw.terminate()
    np.save(args.output / "reference.npy", reference, allow_pickle=False)
    np.save(args.output / "relax.npy", candidate, allow_pickle=False)
    projected_edges = _projected_edge_mask(
        scene, camera, args.width, args.height
    )
    summary = summarize(reference, candidate, projected_edges)
    summary["median_gpu_ms"] = float(statistics.median(timings))
    print(json.dumps(summary, indent=2))
    if args.accept_baseline:
        args.baseline.write_text(json.dumps(summary, indent=2) + "\n")
        return 0
    baseline = json.loads(args.baseline.read_text())
    limit = max(baseline["dark_outlier_p995_p95"] * 1.15,
                baseline["dark_outlier_p995_p95"] + 0.02)
    if summary["dark_outlier_p995_p95"] > limit:
        print(
            "FAIL: dark tessellation residual "
            f"{summary['dark_outlier_p995_p95']:.6g} > {limit:.6g}"
        )
        return 1
    edge_limit = max(baseline["triangle_edge_dark_excess_p95"] * 1.20,
                     baseline["triangle_edge_dark_excess_p95"] + 0.005)
    if summary["triangle_edge_dark_excess_p95"] > edge_limit:
        print(
            "FAIL: triangle-edge dark residual "
            f"{summary['triangle_edge_dark_excess_p95']:.6g} > {edge_limit:.6g}"
        )
        return 1
    print("PASS: ReLAX matte output has no triangle-seam regression")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
