"""Paired geometric-gate comparison using identical rendered radiance."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import ordinarylight as ol
from ordinarylight.renderers.gi.vulkan import VulkanGlobalIlluminationRenderer
from ordinarylight.showcases.materials import quad, diffuse
from tools.denoiser_motion.folded_sequence import fixture
from tools.denoiser_motion.replay import ShaderReplay


def run(output, contrast=False):
    output.mkdir(parents=True, exist_ok=False)
    width, height = 160, 80
    phases = [0, 0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1, 0] + [0]*10
    scene = fixture(64)
    if contrast:
        from tools.denoiser_motion.rendered_occlusion import fixture as planes
        scene = planes(slope=1)

    v, i = quad((-2, 2, -4), (2, 2, -4), (2, 3, -4), (-2, 3, -4))
    if contrast:
        # Emitter lies inside the 0.08-wide gap. The front patch hides it
        # from the camera while the rear patch receives its illumination.
        v, i = quad((-1.5, -1, -1.46), (-0.1, -1, -0.06),
                    (-0.1, 1, -0.06), (-1.5, 1, -1.46))
    scene.add_mesh(v, i, ol.Material(emission=(12, 10, 8), emission_two_sided=True, program=diffuse))
    config = ol.RendererConfig(max_bounces=3, denoiser_signal_capture=True,
                               wavefront_tile_capacity=width*height)
    baseline = ShaderReplay(width, height, depth_footprint=True)
    gated = ShaderReplay(width, height, depth_footprint=True, plane_gate=True)
    policy = dict(history_limit=8, normal_threshold=0.95, depth_threshold=0.005,
                  clamp_sigma=1, reactive_sigma=2.5)
    arrays = {key: [] for key in ("reference", "raw", "baseline", "geometric")}
    reveals, splits = [], []
    previous_patch = None
    try:
        with VulkanGlobalIlluminationRenderer(config=config) as renderer, \
                VulkanGlobalIlluminationRenderer(config=config) as truth:
            for index, shift in enumerate(phases):
                camera = ol.PerspectiveCamera(position=(shift, 0, -7), target=(shift, 0, 0))
                signal = renderer.capture_denoiser_signals(scene, camera, width, height, frame_index=index)
                raw = renderer.capture_denoiser_raw(scene, camera, width, height, frame_index=index)
                patch = raw["primitive_id"] < (2 if contrast else 8192)
                reveal = np.zeros((height, width), bool)
                if previous_patch is not None:
                    yy, xx = np.indices((height, width))
                    motion = signal.motion.astype(np.float16).astype(np.float32)
                    q = (np.stack((xx, yy), -1)+motion+0.5).astype(int)
                    inside = (q[...,0]>=0)&(q[...,0]<width)&(q[...,1]>=0)&(q[...,1]<height)
                    qx, qy = np.clip(q[...,0],0,width-1), np.clip(q[...,1],0,height-1)
                    reveal = inside & patch & ~previous_patch[qy,qx] & (signal.view_z>0)
                reveals.append(reveal)
                previous_patch = patch
                arrays["raw"].append(signal.diffuse_radiance_hit_distance[..., :3] +
                                     signal.specular_radiance_hit_distance[..., :3])
                for key, replay in (("baseline", baseline), ("geometric", gated)):
                    lobes, _ = replay.denoise(signal, signal.view_z, policy, iterations=3,
                                              positions=raw["primary_position"][..., :3])
                    arrays[key].append(lobes[0]+lobes[1])
                batches = [truth.render_wavefront(scene, camera, width, height, samples=64,
                           frame_index=1000+index*4+j)[..., :3] for j in range(4 if contrast else 1)]
                arrays["reference"].append(np.mean(batches, axis=0))
                if contrast:
                    splits.append([np.mean(batches[::2], axis=0), np.mean(batches[1::2], axis=0)])
                print(f"frame {index+1}/{len(phases)}", flush=True)
    finally:
        baseline.close()
        gated.close()
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    np.savez_compressed(output/"arrays.npz", **arrays, revealed=np.asarray(reveals))
    frames = []
    for index in range(len(phases)):
        canvas = Image.new("RGB", (width*4, height+36))
        draw = ImageDraw.Draw(canvas)
        draw.text((3, 2), f"Frame {index}: {'moving' if 2 <= index <= 9 else 'holding'} - slowed playback", fill="white")
        for col, key in enumerate(arrays):
            rgb = np.maximum(arrays[key][index], 0)
            rgb = np.clip(rgb/(1+rgb), 0, 1)**(1/2.2)
            draw.text((col*width+3, 19), key, fill="white")
            canvas.paste(Image.fromarray((rgb*255).astype(np.uint8)), (col*width, 36))
        frames.append(canvas.resize((width*8, (height+36)*2), Image.Resampling.NEAREST))
    frames[0].save(output/"comparison.apng", save_all=True, append_images=frames[1:],
                   duration=[200]*19+[1200], loop=0)
    frames[7].save(output/"moving.png")
    frames[-1].save(output/"settled.png")
    report = {"contrast": contrast, "reference_samples": 256 if contrast else 64,
              "revealed_pixels": int(np.sum(reveals))}
    if contrast:
        split = np.asarray(splits)
        report["reference_split_log_rgb_rmse"] = float(np.sqrt(np.mean(
            (np.log1p(np.maximum(split[:,0],0))-np.log1p(np.maximum(split[:,1],0)))**2)))

    for key in ("baseline", "geometric"):
        error = np.log1p(np.maximum(arrays[key], 0))-np.log1p(np.maximum(arrays["reference"], 0))
        report[key] = {name: float(np.sqrt(np.mean(error[indices]**2))) for name, indices in
                       (("all_log_rgb_rmse", slice(None)), ("moving_log_rgb_rmse", slice(2, 10)),
                        ("settling_log_rgb_rmse", slice(10, None)))}
        mask = np.asarray(reveals)
        report[key]["revealed_log_rgb_rmse"] = float(np.sqrt(np.mean(error[mask]**2))) if mask.any() else None
    report["gate_baseline_max_rgb_difference"] = float(np.max(np.abs(arrays["baseline"]-arrays["geometric"])))
    (output/"report.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contrast", action="store_true")
    args = parser.parse_args()
    run(args.output, args.contrast)
