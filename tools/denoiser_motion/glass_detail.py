"""Compare front-surface glass denoising under three independent motions."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import argparse
import json

import numpy as np
from PIL import Image, ImageDraw

import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.integrations.raster_workbench import _gi_config
from ordinarylight.showcases.glass_detail import build_glass_detail


def fixture():
    scene = build_glass_detail()
    glass = next(mesh for mesh in scene.meshes if mesh.name == "glass-sphere")
    bars = [mesh for mesh in scene.meshes if mesh.name.startswith("target-bar-")]
    return scene, glass, bars


def pose(scene, glass, bars, kind, x):
    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 3] = x
    if kind == 'glass':
        scene.update_instance(glass, transform=matrix)
    elif kind == 'target':
        scene.update_instance_transforms(bars, [matrix] * len(bars))
    offset = x if kind == 'camera' else 0.0
    return ol.PerspectiveCamera(position=(offset, 1, -5), target=(offset, 1, 0))


def display(a):
    a = np.maximum(a, 0)
    return (255 * (a / (1 + a)) ** (1 / 2.2)).clip(0, 255).astype('uint8')


def signal(a):
    return np.log1p(np.maximum(a @ np.array([.2126, .7152, .0722]), 0))


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    width, height = 320, 240
    cfg = _gi_config(SimpleNamespace(renderer={}, id='glass-detail'), capture=True)
    cfg = replace(cfg, direct_swapchain_storage=False,
                  wavefront_tile_capacity=width * height)
    refcfg = replace(cfg, denoiser_enabled=False, progressive_accumulation=False,
                     temporal_history=False, wavefront_restir_di=False,
                     samples_per_pixel=64)
    xs = list(np.linspace(-.3, .3, 16)) + [.3] * 32
    reference_indices = (7, 15)  # All stopped frames share pose 15.
    glfw = load_glfw()
    if not glfw.init():
        raise RuntimeError('GLFW unavailable')
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(width, height, 'Glass detail', None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError('Window creation failed')
    metrics = {}
    sheet = Image.new('RGB', (width * 4, (height + 24) * 3))
    draw = ImageDraw.Draw(sheet)
    try:
        for row, kind in enumerate(('camera', 'glass', 'target')):
            scene, glass, bars = fixture()
            frames = []
            with ol.VulkanGlfwPresenter(window, config=cfg) as presenter:
                for x in xs:
                    camera = pose(scene, glass, bars, kind, float(x))
                    presenter.present_wavefront(scene, camera, width, height)
                    frames.append(presenter.capture_wavefront_hdr()[..., :3])
            refs = []
            scene, glass, bars = fixture()
            with ol.VulkanGlfwPresenter(window, config=refcfg) as presenter:
                for index in reference_indices:
                    camera = pose(scene, glass, bars, kind, float(xs[index]))
                    batches = []
                    for batch in range(8):
                        presenter._core.wavefront_frame_sequence = 1000 + index * 8 + batch
                        presenter.present_wavefront(scene, camera, width, height)
                        batches.append(presenter.capture_wavefront_hdr()[..., :3])
                    refs.append(batches)
            frames, refs = np.asarray(frames), np.asarray(refs)
            assert np.isfinite(frames).all() and np.isfinite(refs).all()
            np.save(output / f'{kind}-frames.npy', frames)
            np.save(output / f'{kind}-reference-batches.npy', refs)
            # Conservative central patch stays inside the glass for all poses.
            patch = np.s_[85:155, 125:195]
            stats = {}
            for label, index, refindex in [('moving-middle', 7, 0),
                                           ('moving-end', 15, 1),
                                           ('stopped-1', 16, 1),
                                           ('stopped-32', 47, 1)]:
                candidate = signal(frames[index][patch])
                target = signal(refs[refindex].mean(0)[patch])
                stats[label] = {'rmse': float(np.sqrt(np.mean((candidate-target)**2))),
                                'horizontal_detail_ratio': float(np.mean(abs(np.diff(candidate, axis=1))) /
                                    max(np.mean(abs(np.diff(target, axis=1))), 1e-8))}
            stats['reference_split_rmse'] = float(np.sqrt(np.mean((
                signal(refs[1, ::2].mean(0)[patch]) - signal(refs[1, 1::2].mean(0)[patch]))**2)))
            metrics[kind] = stats
            for col, (label, frame) in enumerate([
                ('moving end', frames[15]), ('first stopped', frames[16]),
                ('32 stopped', frames[47]), ('512-sample reference', refs[1].mean(0)),
            ]):
                sheet.paste(Image.fromarray(display(frame)), (col*width, row*(height+24)+24))
                draw.text((col*width+4, row*(height+24)+4), f'{kind}: {label}', fill='white')
            print(kind, json.dumps(stats), flush=True)
    finally:
        glfw.destroy_window(window)
        glfw.terminate()
    sheet.save(output / 'comparison.png')
    (output / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/tmp/glass-detail'))
    run(parser.parse_args().output)
