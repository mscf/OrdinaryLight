"""Rendered same-mesh occlusion: compare depth gates with world-plane evidence."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path

import numpy as np

import ordinarylight as ol
from ordinarylight.renderers.gi.vulkan import VulkanGlobalIlluminationRenderer
from ordinarylight.showcases.materials import diffuse, quad, sphere
from tools.denoiser_motion.replay import ShaderReplay
from tools.denoiser_motion.run import previous_view_depth


def fixture(slope=0, curved=False, close=False, scale=1):
    scene = ol.Scene()
    vertices, indices = [], []
    # Two disconnected patches in ONE mesh/material: runtime identity cannot
    # distinguish the foreground edge from the newly exposed rear patch.
    for corners in (
        ((-4, -3, 0.08), (4, -3, 0.08), (4, 3, 0.08), (-4, 3, 0.08)),
        ((-4, -3, 0), (0, -3, 0), (0, 3, 0), (-4, 3, 0)),
    ):
        v, i = quad(*corners)
        v = np.asarray(v).copy()
        v[:, 2] += slope * v[:, 0]
        indices.extend(np.asarray(i) + len(vertices))
        vertices.extend(v)
    if curved:
        rear_v, rear_i = quad((-4, -3, 1.05), (4, -3, 1.05),
                             (4, 3, 1.05), (-4, 3, 1.05))
        front_v, front_i = sphere((0, 0, 0), 1, rings=48, segments=96)
        vertices = np.concatenate((rear_v, front_v))
        indices = np.concatenate((rear_i, front_i + len(rear_v)))
    if close:
        parts, triangles = [], []
        for xmin, xmax, offset in ((-4, 4, 0.08), (-4, 0, 0)):
            xx, yy = np.meshgrid(np.linspace(xmin, xmax, 33), np.linspace(-3, 3, 33))
            vv = np.stack((xx, yy, 0.2 * (xx**2 + yy**2) + offset), -1).reshape(-1, 3)
            ii = []
            for row in range(32):
                for col in range(32):
                    a = row * 33 + col
                    ii.extend(((a, a+1, a+34), (a, a+34, a+33)))
            triangles.extend(np.asarray(ii) + len(parts))
            parts.extend(vv)
        vertices, indices = np.asarray(parts), np.asarray(triangles)
    vertices = np.asarray(vertices) * scale
    scene.add_mesh(np.asarray(vertices), np.asarray(indices),
                   ol.Material(base_color=(0.6, 0.6, 0.6), program=diffuse))
    return scene


def run(output, slope=0, curved=False, close=False, scale=1):
    output.mkdir(parents=True, exist_ok=False)
    report = []
    for width in (64, 160, 320):
        height = width // 2
        scene = fixture(slope, curved, close, scale)
        config = ol.RendererConfig(max_bounces=1, denoiser_signal_capture=True,
                                   wavefront_tile_capacity=width * height)
        signals, positions, cameras, patches = [], [], [], []
        with VulkanGlobalIlluminationRenderer(config=config) as renderer:
            for index, x in enumerate((0, 0.3)):
                camera = ol.PerspectiveCamera(position=(x * scale, 0, -7 * scale), target=(x * scale, 0, 0))
                signal = renderer.capture_denoiser_signals(scene, camera, width, height,
                                                           frame_index=index)
                raw = renderer.capture_denoiser_raw(scene, camera, width, height,
                                                   frame_index=index)
                # Isolate geometry rejection from illumination changes.
                radiance = np.ones_like(signal.specular_radiance_hit_distance)
                signals.append(replace(signal, diffuse_radiance_hit_distance=radiance,
                                       specular_radiance_hit_distance=radiance.copy()))
                positions.append(raw["primary_position"][..., :3].copy())
                cameras.append(camera)
                patches.append(raw["primitive_id"] < (2048 if close else 2))
        old, current = signals
        y, x = np.indices((height, width))
        p = np.stack((x, y), axis=-1)
        homogeneous = np.concatenate((positions[1], np.ones((height, width, 1))), axis=-1)
        clip = homogeneous @ old.frame.world_to_clip.T
        ndc = clip[..., :2] / clip[..., 3:4]
        projected = (ndc * np.array([0.5, -0.5]) + 0.5) * [width, height] - 0.5
        foreground = current.view_z > 0
        motion_error = float(np.max(np.abs(
            current.motion[foreground] - (projected - p)[foreground]
        )))
        np.testing.assert_allclose(current.motion[foreground],
                                   (projected - p)[foreground], atol=2e-4)

        old.save(output / f"old-{width}.npz")
        current.save(output / f"current-{width}.npz")
        q = (p + current.motion + 0.5).astype(np.int32)
        inside = (q[..., 0] >= 0) & (q[..., 0] < width) & (q[..., 1] >= 0) & (q[..., 1] < height)
        qx, qy = np.clip(q[..., 0], 0, width-1), np.clip(q[..., 1], 0, height-1)
        finite = inside & (current.view_z > 0) & (old.view_z[qy, qx] > 0)
        old_patch = patches[0][qy, qx]
        current_patch = patches[1]
        invalid = finite & (old_patch != current_patch)
        valid = finite & ~invalid
        expected = previous_view_depth(positions[1], cameras[0], current.view_z > 0)
        # Independent geometric diagnostic, using unquantized captured positions.
        # No production binding or temporal shader implements this gate yet.
        delta = positions[1] - positions[0][qy, qx]
        distance = np.abs(np.sum(delta * old.normal_roughness[qy, qx, :3], axis=-1))
        plane_accept = distance <= 0.01
        normal_delta = np.linalg.norm(
            current.normal_roughness[..., :3] - old.normal_roughness[qy, qx, :3], axis=-1)
        footprint = 2 * old.view_z[qy, qx] * np.tan(
            np.radians(cameras[0].vertical_fov_degrees) / 2) / height
        tangent = np.sqrt(np.maximum(np.sum(delta * delta, axis=-1) - distance**2, 0))
        curvature_allowance = 0.5 * normal_delta * np.maximum(footprint, tangent)
        geometric_tolerance = old.view_z[qy, qx] * 1e-5 + curvature_allowance
        geometric_accept = distance <= geometric_tolerance
        for mode in ("original", "blanket", "adaptive"):
            replay = ShaderReplay(width, height, depth_footprint=mode == "adaptive")
            policy = dict(history_limit=8, normal_threshold=0.95,
                          depth_threshold=0.02 if mode == "blanket" else 0.005,
                          clamp_sigma=1, reactive_sigma=2.5)
            try:
                replay.denoise(old, old.view_z, policy, iterations=0)
                replay.denoise(current, expected, policy, iterations=0)
                accept = replay.last_acceptance[1] > 0
                report.append(dict(width=width, slope=slope, curved=curved, close=close, scale=scale, mode=mode, motion_max_error=motion_error,
                    invalid_pixels=int(invalid.sum()), valid_pixels=int(valid.sum()),
                    false_acceptance=float(accept[invalid].mean()) if invalid.any() else None,
                    valid_acceptance=float(accept[valid].mean()),
                    with_plane_false_acceptance=float((accept & plane_accept)[invalid].mean()) if invalid.any() else None,
                    with_plane_valid_acceptance=float((accept & plane_accept)[valid].mean()),
                    valid_plane_distance_percentiles=np.percentile(distance[valid], [50, 95, 99, 100]).tolist(),
                    geometric_extra_valid_rejections=int((accept & valid & ~geometric_accept).sum()),
                    geometric_remaining_false_accepts=int((accept & invalid & geometric_accept).sum()),
                    accepted_valid_pixels=int((accept & valid).sum()),
                    accepted_invalid_pixels=int((accept & invalid).sum()),
                    plane_threshold_sweep={
                        str(t): {
                            "extra_valid_rejections": int((accept & valid & (distance > t)).sum()),
                            "remaining_false_accepts": int((accept & invalid & (distance <= t)).sum()),
                        } for t in (0.001, 0.005, 0.01, 0.02, 0.05)
                    }))

            finally:
                replay.close()
                del replay
                gc.collect()
        np.savez_compressed(output / f"guides-{width}.npz", invalid=invalid, valid=valid,
                            plane_distance=distance, positions=positions)
        print(f"completed {width}", flush=True)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--slope", type=float, default=0)
    parser.add_argument("--curved", action="store_true")
    parser.add_argument("--close", action="store_true")
    parser.add_argument("--scale", type=float, default=1)
    args = parser.parse_args()
    run(args.output, args.slope, args.curved, args.close, args.scale)
