"""Experimental static planar-mirror guide replacement using reflected-camera capture.

This is an offline guide experiment, not native mirror rendering integration.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

import ordinarylight as ol
from ordinarylight.denoising import DenoiserSignals
from ordinarylight.renderers.gi.vulkan import VulkanGlobalIlluminationRenderer
from ordinarylight.showcases.materials import diffuse, mirror, quad, sphere
from tools.denoiser_motion.replay import ShaderReplay
from tools.denoiser_motion.run import previous_view_depth, region_error, write_gallery


def reflect_z(value):
    """Reflect points or vectors across the fixed z=0 mirror plane."""
    result = np.asarray(value).copy()
    result[..., 2] *= -1
    return result


def reflected_camera(camera):
    return ol.PerspectiveCamera(
        position=tuple(reflect_z(camera.position)),
        target=tuple(reflect_z(camera.target)),
        up=tuple(reflect_z(camera.up)),
        vertical_fov_degrees=camera.vertical_fov_degrees,
    )


def fixture(include_mirror):
    scene = ol.Scene()
    vertices, indices = quad((-6, 0, -8), (6, 0, -8), (6, 0, 0), (-6, 0, 0))
    scene.add_mesh(vertices, indices, ol.Material(base_color=(0.55, 0.6, 0.65), program=diffuse))
    vertices, indices = quad((-2, 5, -3), (2, 5, -3), (2, 5, -1), (-2, 5, -1))
    scene.add_mesh(vertices, indices, ol.Material(
        emission=(12, 10, 8), emission_two_sided=True, program=diffuse,
    ))
    moving = None
    for name, center, color, radius in (
        ("moving", (-1, 1, -3), (0.9, 0.1, 0.05), 0.8),
        ("stationary", (1.4, 0.9, -1.8), (0.08, 0.35, 0.9), 0.7),
    ):
        vertices, indices = sphere(center, radius, rings=12, segments=24)
        mesh = scene.add_mesh(
            vertices, indices, ol.Material(base_color=color, program=diffuse), name=name,
        )
        if name == "moving":
            moving = mesh
    mirror_id = None
    if include_mirror:
        vertices, indices = quad((-2.5, 0, 0), (2.5, 0, 0), (2.5, 4, 0), (-2.5, 4, 0))
        material = ol.Material(base_color=(1, 1, 1), metallic=1, roughness=0, program=mirror)
        scene.add_mesh(vertices, indices, material, name="mirror")
        mirror_id = scene.material_id(material)
    return scene, moving, mirror_id


def move_object(scene, mesh, shift):
    """Notify scene revision tracking as well as updating the CPU transform."""
    scene.update_instance_transforms(
        (mesh,), (ol.Transform.translation((shift, 0, 0)),),
    )


def prepare_signal(source, radiance, mask, *, optical=False):
    """Use one radiance input with either real or reflected virtual guides."""
    normal = source.normal_roughness.copy()
    depth = source.view_z.copy()
    motion = source.motion.copy()
    material = source.material_id.copy()
    if optical:
        normal = normal[:, ::-1].copy()
        normal[..., :3] = reflect_z(normal[..., :3])
        depth = depth[:, ::-1].copy()
        motion = motion[:, ::-1].copy()
        motion[..., 0] *= -1
        material = material[:, ::-1].copy()
    depth[~mask] = 0
    normal[~mask] = 0
    motion[~mask] = 0
    material[~mask] = np.uint32(0xffffffff)
    rgba = np.zeros((*depth.shape, 4), np.float32)
    rgba[..., :3] = radiance
    rgba[~mask] = 0
    return DenoiserSignals(
        np.zeros_like(rgba), rgba, np.ascontiguousarray(normal),
        np.ascontiguousarray(depth), np.ascontiguousarray(motion),
        np.ascontiguousarray(material), source.frame,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--motion", choices=("camera", "object"), default="camera")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    width, height = 160, 120
    phases = [0, 0, 0.25, 0.5, 0.75, 1, 1, 1, 1, 1]
    actual, moving_real, mirror_id = fixture(True)
    virtual, moving_virtual, _ = fixture(False)
    moving_id = virtual.material_id(moving_virtual.material)
    config = ol.RendererConfig(
        max_bounces=3, denoiser_signal_capture=True,
        wavefront_tile_capacity=width * height,
    )
    arrays = {name: [] for name in ("reference", "primary_guides", "reflected_guides", "raw")}
    halves, masks, histories, surface_ids = [], [], [], []
    replays = [ShaderReplay(width, height), ShaderReplay(width, height)]
    policy = dict(history_limit=8, normal_threshold=0.95, depth_threshold=0.005,
                  clamp_sigma=1.0, reactive_sigma=2.5)
    old_camera, old_shift = None, 0.0
    try:
        with VulkanGlobalIlluminationRenderer(config=config) as real, \
                VulkanGlobalIlluminationRenderer(config=replace(config, max_bounces=2)) as reflected:
            for index, phase in enumerate(phases):
                shift = 1.5 * phase if args.motion == "object" else 0.0
                move_object(actual, moving_real, shift)
                move_object(virtual, moving_virtual, shift)
                x = 0.8 * (phase - 0.5) if args.motion == "camera" else 0.0
                camera = ol.PerspectiveCamera(position=(x, 2, -7), target=(0, 1.8, 0))
                optical_camera = reflected_camera(camera)
                primary_signal = real.capture_denoiser_signals(actual, camera, width, height, frame_index=index)
                primary_raw = real.capture_denoiser_raw(actual, camera, width, height, frame_index=index)
                optical_signal = reflected.capture_denoiser_signals(
                    virtual, optical_camera, width, height, frame_index=index,
                )
                optical_raw = reflected.capture_denoiser_raw(
                    virtual, optical_camera, width, height, frame_index=index,
                )
                mask = primary_signal.material_id == mirror_id
                # Reflection reverses handedness; reflected-camera raster X
                # runs opposite to the mirror's screen-space X.
                radiance = optical_raw["radiance"][:, ::-1, :3].copy()
                signal_primary = prepare_signal(primary_signal, radiance, mask)
                signal_optical = replace(
                    prepare_signal(optical_signal, radiance, mask, optical=True),
                    frame=primary_signal.frame,
                )
                old = old_camera or camera
                expected_primary = previous_view_depth(primary_raw["primary_position"], old, mask)
                previous_positions = optical_raw["primary_position"].copy()
                moving_mask = optical_signal.material_id == moving_id
                previous_positions[moving_mask, 0] -= shift - old_shift
                expected_optical = previous_view_depth(
                    reflect_z(previous_positions)[:, ::-1], old,
                    mask & (signal_optical.view_z > 0),
                )
                signal_primary.save(args.output / f"primary-{index:03d}.npz")
                signal_optical.save(args.output / f"reflected-{index:03d}.npz")
                np.savez_compressed(
                    args.output / f"depth-{index:03d}.npz",
                    primary=expected_primary, reflected=expected_optical,
                )
                lengths = []
                for name, replay, signal, expected in zip(
                    ("primary_guides", "reflected_guides"), replays,
                    (signal_primary, signal_optical), (expected_primary, expected_optical),
                ):
                    lobes, history = replay.denoise(signal, expected, policy)
                    arrays[name].append(lobes[0] + lobes[1])
                    lengths.append(history[1])
                reference_halves = []
                for batch in range(2):
                    image = reflected.render_wavefront(
                        virtual, optical_camera, width, height, samples=64,
                        frame_index=1000 + index * 2 + batch,
                    )
                    image = image[:, ::-1, :3].copy()
                    image[~mask] = 0
                    reference_halves.append(image)
                arrays["reference"].append(np.mean(reference_halves, axis=0))
                radiance[~mask] = 0
                arrays["raw"].append(radiance)
                halves.append(reference_halves)
                masks.append(mask)
                surface_ids.append(optical_signal.material_id[:, ::-1].copy())
                histories.append(lengths)
                old_camera, old_shift = camera, shift
                print(f"{args.motion}: frame {index + 1}/{len(phases)}", flush=True)
    finally:
        for replay in replays:
            replay.close()
    arrays = {name: np.asarray(values) for name, values in arrays.items()}
    mask = np.asarray(masks)
    identities = np.asarray(surface_ids)
    revealed = np.zeros_like(mask)
    if args.motion == "object":
        revealed[1:] = (identities[:-1] == moving_id) & (identities[1:] != moving_id) & mask[1:]
    report = {
        "scope": "offline static z=0 planar mirror, reflected camera, identical radiance",
        "motion": args.motion, "phases": phases, "policy": policy,
        "reference_samples": 128, "mirror_pixels": int(mask.sum()),
        "reference_split_error": region_error(np.asarray(halves)[:, 0], np.asarray(halves)[:, 1], mask),
        "revealed_pixels": int(revealed.sum()),
        "revealed_error": {
            name: region_error(arrays["reference"], arrays[name], revealed)
            for name in ("primary_guides", "reflected_guides")
        },
        "error": {name: region_error(arrays["reference"], value, mask)
                  for name, value in arrays.items() if name != "reference"},
    }
    np.savez_compressed(args.output / "arrays.npz", **arrays, mask=mask, histories=histories,
                        surface_ids=identities, revealed=revealed)
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    write_gallery(args.output, arrays)


if __name__ == "__main__":
    main()
