"""Capture camera-motion optics and replay identical signals through two denoisers.

Run from the checkout: python -m tools.denoiser_motion.run --output /tmp/optics-motion
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import ordinarylight as ol
from ordinarylight.denoising.reference import NrdRelaxReference
from ordinarylight.renderers.gi.vulkan import VulkanGlobalIlluminationRenderer
from ordinarylight.showcases.materials import (
    diffuse,
    fresnel_glass,
    mirror,
    quad,
    sphere,
)
from ordinarylight.showcases.rooms import _add_room, _add_emitter
from ordinarylight.targets.vulkan.core import (
    _camera_motion_pixels,
    _relax_temporal_policy,
)
from tests.gates.relax_motion_quality import evaluate_sequence


def fixture():
    scene = ol.Scene()
    _add_room(scene)
    _add_emitter(
        scene, ((-2, 6.7, -1), (2, 6.7, -1), (2, 6.7, 1), (-2, 6.7, 1)), (12, 10, 8)
    )
    materials = {
        "glossy": ol.Material(base_color=(0.85, 0.65, 0.25), metallic=1, roughness=0.2),
        "mirror": ol.Material(
            base_color=(0.96, 0.96, 0.96), metallic=1, roughness=0, program=mirror
        ),
        "glass": ol.Material(
            base_color=(0.94, 0.98, 1),
            transmission=1,
            roughness=0,
            ior=1.52,
            program=fresnel_glass,
        ),
    }
    for name, x in (("glossy", -2.5), ("glass", 2.5)):
        vertices, indices = sphere((x, 1.3, 0), 1.2, rings=16, segments=32)
        scene.add_mesh(vertices, indices, materials[name], name=name)
    vertices, indices = quad(
        (-1.05, 0.1, 0.5), (1.05, 0.1, 0.5), (1.05, 3, 0.5), (-1.05, 3, 0.5)
    )
    scene.add_mesh(vertices, indices, materials["mirror"], name="mirror")
    # Colored objects in front of and behind the optics reveal transport detail.
    for center, color in (
        ((-1.1, 1, -3), (0.9, 0.08, 0.04)),
        ((2.4, 1, 3), (0.05, 0.8, 0.2)),
    ):
        vertices, indices = sphere(center, 0.65, rings=12, segments=24)
        scene.add_mesh(
            vertices, indices, ol.Material(base_color=color, program=diffuse)
        )
    return scene, {name: scene.material_id(mat) for name, mat in materials.items()}


def camera(x):
    return ol.PerspectiveCamera(position=(x, 3.2, -9), target=(0, 1.5, 0.5))


def split_camera(view, width, height):
    from ordinarylight.raster import camera_matrix

    eye = np.asarray(view.position, np.float64)
    forward = np.asarray(view.target, np.float64) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, view.up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    world_to_view = np.eye(4)
    world_to_view[:3, :3] = (right, up, -forward)
    world_to_view[:3, 3] = -world_to_view[:3, :3] @ eye
    projection = camera_matrix(view, width, height) @ np.linalg.inv(world_to_view)
    return projection.astype(np.float32), world_to_view.astype(np.float32)


def camera_sequence(cameras, width, height):
    result = []
    for index, view in enumerate(cameras):
        projection, world_to_view = split_camera(view, width, height)
        old_projection, old_view = split_camera(
            cameras[max(index - 1, 0)], width, height
        )
        result.append(
            dict(
                view_to_clip=projection,
                world_to_view=world_to_view,
                previous_view_to_clip=old_projection,
                previous_world_to_view=old_view,
            )
        )
    return result


def previous_view_depth(positions, old_camera, foreground):
    direction = np.asarray(old_camera.target) - np.asarray(old_camera.position)
    direction = direction / np.linalg.norm(direction)
    value = (positions - np.asarray(old_camera.position)) @ direction
    return np.where(foreground, value, 0).astype(np.float32)


def region_error(truth, candidate, mask):
    if not np.any(mask):
        return None
    delta = np.log1p(np.maximum(candidate, 0)) - np.log1p(np.maximum(truth, 0))
    return float(np.sqrt(np.mean(delta[mask] ** 2)))


def write_gallery(output, arrays):
    from PIL import Image, ImageDraw

    frames, height, width, _ = arrays["reference"].shape
    names = list(arrays)
    images = []
    for index in range(frames):
        canvas = Image.new("RGB", (width * len(names), height + 28))
        draw = ImageDraw.Draw(canvas)
        for col, name in enumerate(names):
            rgb = np.maximum(arrays[name][index, ..., :3], 0)
            rgb = np.power(rgb / (1 + rgb), 1 / 2.2)
            panel = Image.fromarray(np.uint8(np.clip(rgb * 255, 0, 255)))
            canvas.paste(panel, (col * width, 28))
            draw.text((col * width + 4, 5), f"{name} / {index}", fill="white")
        canvas.save(output / f"comparison-{index:03d}.png")
        images.append(canvas)
    images[0].save(
        output / "comparison.gif",
        save_all=True,
        append_images=images[1:],
        duration=160,
        loop=0,
    )


def main(argv=None):
    from .replay import ShaderReplay

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--reference-samples", type=int, default=64)
    parser.add_argument(
        "--replay", type=Path, help="Reuse an existing capture without tracing again"
    )
    parser.add_argument("--history-floor", type=int, default=1, choices=range(1, 33))
    parser.add_argument("--travel", type=float, default=0.6)
    args = parser.parse_args(argv)
    if (
        args.width < 16
        or args.height < 16
        or args.frames < 7
        or args.reference_samples < 4
        or args.reference_samples > 128
        or args.reference_samples % 2
    ):
        parser.error(
            "need extent >=16, frames >=7 and even reference samples in [4,128]"
        )
    if args.replay:
        prior = json.loads((args.replay / "report.json").read_text())["configuration"]
        for key in ("width", "height", "frames", "reference_samples", "travel"):
            setattr(args, key, prior[key])
    args.output.mkdir(parents=True, exist_ok=False)
    scene, regions = fixture()
    config = ol.RendererConfig(
        max_bounces=8,
        samples_per_pixel=1,
        denoiser_signal_capture=True,
        denoiser_motion_history_floor=args.history_floor,
        wavefront_tile_capacity=args.width * args.height,
    )
    sequence, positions, references, noisy, cameras = [], [], [], [], []
    reference_halves = []
    if args.replay:
        prior_report = json.loads((args.replay / "report.json").read_text())
        adapter = prior_report["capture_adapter"]
        references = list(np.load(args.replay / "reference.npy"))
        reference_halves = list(
            np.load(args.replay / "diagnostics.npz")["reference_halves"]
        )
        noisy = list(np.load(args.replay / "raw.npy"))
        for index in range(args.frames):
            phase = np.clip((index - 2) / max(args.frames - 5, 1), 0, 1)
            cameras.append(camera(args.travel * (phase - 0.5)))
            sequence.append(
                ol.DenoiserSignals.load(args.replay / f"frame-{index:03d}.npz")
            )
            positions.append(
                np.load(args.replay / f"raw-{index:03d}.npz")["primary_position"]
            )
    else:
        with VulkanGlobalIlluminationRenderer(config=config) as renderer:
            adapter = renderer.device.name
            for index in range(args.frames):
                # Warm-up, camera movement, then recovery at the final viewpoint.
                phase = np.clip((index - 2) / max(args.frames - 5, 1), 0, 1)
                view = camera(args.travel * (phase - 0.5))
                signal = renderer.capture_denoiser_signals(
                    scene, view, args.width, args.height, frame_index=index
                )
                raw = renderer.capture_denoiser_raw(
                    scene, view, args.width, args.height, frame_index=index
                )
                signal.save(args.output / f"frame-{index:03d}.npz")
                np.savez_compressed(args.output / f"raw-{index:03d}.npz", **raw)
                # Independent seeds, same scene and camera; two references expose
                # remaining Monte Carlo error rather than assuming ground truth.
                halves = [
                    renderer.render_wavefront(
                        scene,
                        view,
                        args.width,
                        args.height,
                        samples=args.reference_samples // 2,
                        frame_index=10000 + index * 2 + half,
                    )[..., :3]
                    for half in range(2)
                ]
                references.append((halves[0] + halves[1]) * 0.5)
                reference_halves.append(halves)
                sequence.append(signal)
                positions.append(raw["primary_position"])
                noisy.append(raw["radiance"][..., :3])
                cameras.append(view)
                print(f"Captured {index + 1}/{args.frames}", flush=True)
    nrd = NrdRelaxReference()
    nrd_results = nrd.denoise_sequence(
        sequence, camera_matrices=camera_sequence(cameras, args.width, args.height)
    )
    gpu = ShaderReplay(args.width, args.height)
    uncapped_gpu = ShaderReplay(args.width, args.height)
    rendered, histories, policy_log, uncapped, lobes = [], [], [], [], []
    acceptance = []
    try:
        for index, signal in enumerate(sequence):
            old_camera = cameras[max(index - 1, 0)]
            motion_pixels = _camera_motion_pixels(
                old_camera, cameras[index], args.height
            )
            policy = _relax_temporal_policy(config, motion_pixels)
            depth = previous_view_depth(
                positions[index], old_camera, signal.view_z != 0
            )
            result, history = gpu.denoise(signal, depth, policy)
            alternative, _ = uncapped_gpu.denoise(
                signal,
                depth,
                {**policy, "history_limit": config.denoiser_history_limit},
            )
            uncapped.append(alternative[0] + alternative[1])
            lobes.append(result)
            rendered.append(result[0] + result[1])
            histories.append(history)
            acceptance.append(gpu.last_acceptance)
            policy_log.append({"motion_pixels": motion_pixels, **policy})
    finally:
        gpu.close()
        uncapped_gpu.close()
    arrays = {
        "reference": np.stack(references),
        "raw": np.stack(noisy),
        "ordinaryshade": np.stack(rendered),
        "uncapped": np.stack(uncapped),
        "nrd": np.stack([result.combined for result in nrd_results]),
    }
    masks = {
        name: np.stack([s.material_id == identifier for s in sequence])
        for name, identifier in regions.items()
    }
    halves = np.asarray(reference_halves)
    report = {
        "schema": 1,
        "configuration": {
            **vars(args),
            "output": str(args.output),
            "replay": str(args.replay) if args.replay else None,
        },
        "capture_adapter": adapter,
        "replay_adapter": gpu.adapter_info,
        "nrd_version": nrd_results[0].implementation_version,
        "nrd_frame_interval_ms": 1000 / 60,
        "scope": "Camera-only motion; production WGSL filter replay of canonical captures, not live Vulkan preparation parity. NRD glass inputs are outside its opaque-surface contract.",
        "policies": policy_log,
        "reference_split_log_rmse": region_error(
            halves[:, 0],
            halves[:, 1],
            np.ones(halves.shape[:1] + halves.shape[2:4], bool),
        ),
        "metrics": {
            name: evaluate_sequence(arrays["reference"], value)
            for name, value in arrays.items()
            if name != "reference"
        },
        "regions": {
            region: {
                "pixels": int(mask.sum()),
                **{
                    name: region_error(arrays["reference"], value, mask)
                    for name, value in arrays.items()
                    if name != "reference"
                },
                "reference_split_log_rmse": region_error(
                    halves[:, 0], halves[:, 1], mask
                ),
            }
            for region, mask in masks.items()
        },
        "signal_sum_max_error": float(
            max(
                np.max(
                    np.abs(
                        s.diffuse_radiance_hit_distance[..., :3]
                        + s.specular_radiance_hit_distance[..., :3]
                        - raw
                    )
                )
                for s, raw in zip(sequence, noisy)
            )
        ),
        "accepted_fraction": np.asarray(acceptance).mean(axis=(2, 3)).tolist(),
        "history_summary": [
            {
                "diffuse_mean": float(np.mean(h[0])),
                "specular_mean": float(np.mean(h[1])),
                "diffuse_over_one_fraction": float(np.mean(h[0] > 1)),
                "specular_over_one_fraction": float(np.mean(h[1] > 1)),
            }
            for h in histories
        ],
        "per_frame_regions": {
            region: {
                name: [
                    region_error(arrays["reference"][i], value[i], mask[i])
                    for i in range(args.frames)
                ]
                for name, value in arrays.items()
                if name != "reference"
            }
            for region, mask in masks.items()
        },
        "shader_hashes": {
            name: hashlib.sha256(
                (
                    Path("ordinarylight/shaders") / f"denoiser_relax_{name}.comp.wgsl"
                ).read_bytes()
            ).hexdigest()
            for name in ("temporal", "atrous")
        },
    }
    for name, value in arrays.items():
        np.save(args.output / f"{name}.npy", value)
    np.savez_compressed(
        args.output / "diagnostics.npz",
        histories=np.asarray(histories),
        acceptance=np.asarray(acceptance),
        ordinaryshade_lobes=np.asarray(lobes),
        nrd_lobes=np.asarray([[r.diffuse, r.specular] for r in nrd_results]),
        reference_halves=halves,
        **masks,
    )
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_gallery(args.output, {k: v for k, v in arrays.items() if k != "uncapped"})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
