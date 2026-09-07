"""Stress the geometric allowance on moving views of tessellated folded patches."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path

import numpy as np

import ordinarylight as ol
from ordinarylight.renderers.gi.vulkan import VulkanGlobalIlluminationRenderer
from tools.denoiser_motion.replay import ShaderReplay


def fixture(segments):
    vertices, triangles = [], []
    for xmax, offset in ((4, 0.08), (0, 0)):
        x, y = np.meshgrid(np.linspace(-4, xmax, segments+1),
                           np.linspace(-3, 3, segments+1))
        v = np.stack((x, y, 0.4*np.sin(2*x) + 0.2*np.sin(2*y) + offset), -1).reshape(-1, 3)
        indices = []
        for row in range(segments):
            for col in range(segments):
                a = row*(segments+1)+col
                indices.extend(((a, a+1, a+segments+2), (a, a+segments+2, a+segments+1)))
        triangles.extend(np.asarray(indices)+len(vertices))
        vertices.extend(v)
    scene = ol.Scene()
    scene.add_mesh(np.asarray(vertices, np.float32), np.asarray(triangles, np.uint32),
                   ol.Material(base_color=(0.6, 0.6, 0.6)))
    return scene


def run(output, plane_gate=False):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    phases = [0, 0, 0.1, 0.2, 0.3, 0.4, 0.3, 0.2, 0.1, 0, 0, 0]
    for width in (160, 320):
        height = width // 2
        for segments in (16, 64):
            scene = fixture(segments)
            replay = ShaderReplay(width, height, depth_footprint=True, plane_gate=plane_gate)
            config = ol.RendererConfig(max_bounces=1, denoiser_signal_capture=True,
                                       wavefront_tile_capacity=width*height)
            previous = None
            policy = dict(history_limit=8, normal_threshold=0.95, depth_threshold=0.005,
                          clamp_sigma=1, reactive_sigma=2.5)
            try:
                with VulkanGlobalIlluminationRenderer(config=config) as renderer:
                    for index, shift in enumerate(phases):
                        camera = ol.PerspectiveCamera(position=(shift, 0, -7), target=(shift, 0, 0))
                        signal = renderer.capture_denoiser_signals(scene, camera, width, height, frame_index=index)
                        raw = renderer.capture_denoiser_raw(scene, camera, width, height, frame_index=index)
                        positions = raw["primary_position"][..., :3]
                        patches = raw["primitive_id"] < 2*segments*segments
                        rgba = np.ones_like(signal.specular_radiance_hit_distance)
                        signal = replace(signal, specular_radiance_hit_distance=rgba,
                                         diffuse_radiance_hit_distance=rgba.copy())
                        replay.denoise(signal, signal.view_z, policy, iterations=0, positions=positions)
                        if previous is not None:
                            old, old_pos, old_patch = previous
                            yy, xx = np.indices((height, width))
                            grid = np.stack((xx, yy), -1)
                            q = (grid+signal.motion.astype(np.float16).astype(np.float32)+0.5).astype(int)
                            inside = (q[..., 0]>=0)&(q[..., 0]<width)&(q[..., 1]>=0)&(q[..., 1]<height)
                            qx, qy = np.clip(q[..., 0], 0, width-1), np.clip(q[..., 1], 0, height-1)
                            finite = inside & (signal.view_z>0) & (old.view_z[qy,qx]>0)
                            invalid = finite & (patches != old_patch[qy,qx])
                            valid = finite & ~invalid
                            delta = positions-old_pos[qy,qx]
                            normal = old.normal_roughness[qy,qx,:3].astype(np.float16).astype(np.float32)
                            distance = np.abs(np.sum(delta*normal, -1))
                            change = np.linalg.norm(signal.normal_roughness[...,:3].astype(np.float16).astype(np.float32)-normal, axis=-1)
                            tangent = np.sqrt(np.maximum(np.sum(delta**2,-1)-distance**2,0))
                            footprint = 2*old.view_z[qy,qx]*np.tan(np.pi/8)/height
                            tolerance = old.view_z[qy,qx]*1e-5+change*np.maximum(footprint,tangent)
                            accepted = replay.last_acceptance[1]>0
                            rows.append(dict(width=width, segments=segments, frame=index, plane_gate=plane_gate,
                                phase=shift, valid=int(valid.sum()), invalid=int(invalid.sum()),
                                accepted_valid=int((accepted&valid).sum()),
                                accepted_invalid=int((accepted&invalid).sum()),
                                extra_valid_rejections=int((accepted&valid&(distance>tolerance)).sum()),
                                remaining_false_accepts=int((accepted&invalid&(distance<=tolerance)).sum())))
                        previous = signal, positions.copy(), patches.copy()
            finally:
                replay.close()
                del replay
                gc.collect()
            print(f"completed {width}, {segments}", flush=True)
    (output/"report.json").write_text(json.dumps(rows, indent=2)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plane-gate", action="store_true")
    args = parser.parse_args()
    run(args.output, args.plane_gate)
