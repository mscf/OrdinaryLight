"""Capture internal HDR for a preliminary render-scale detail comparison."""
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
from PIL import Image, ImageDraw
import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.integrations.raster_workbench import _gi_config
from tools.denoiser_motion.glass_detail import fixture, pose, display

out = Path('/tmp/render-scale-quality')
out.mkdir(exist_ok=True)
g = load_glfw()
assert g.init()
g.window_hint(g.CLIENT_API, g.NO_API)
g.window_hint(g.VISIBLE, g.FALSE)
w = g.create_window(640, 480, 'Scale quality', None, None)
metrics = {}
try:
    sheet = Image.new('RGB', (1280, 3 * 510))
    for row, subject in enumerate(('glass', 'target', 'camera')):
        for col, scale in enumerate((1., .5)):
            scene, glass, bars = fixture()
            config = _gi_config(SimpleNamespace(id='glass-detail', renderer={}),
                                capture=True, render_scale=scale)
            with ol.VulkanGlfwPresenter(w, config=config) as presenter:
                frames = []
                for x in list(np.linspace(-.3, .3, 12)) + [.3] * 12:
                    presenter.present_wavefront(scene, pose(scene, glass, bars, subject, x), 640, 480)
                    frames.append(presenter.capture_wavefront_hdr()[..., :3])
                frames = np.array(frames)
                assert np.isfinite(frames).all()
                np.save(out / f'{subject}-{scale}.npy', frames)
                metrics[f'{subject}-{scale}'] = {'shape': list(frames.shape),
                    'finite': True, 'mean_final_hdr': float(frames[-1].mean())}
                # Display-only enlargement of internal HDR, not swapchain readback.
                preview = Image.fromarray(display(frames[-1])).resize((640, 480), Image.Resampling.BILINEAR)
                sheet.paste(preview, (col * 640, row * 510 + 30))
                ImageDraw.Draw(sheet).text((col * 640 + 8, row * 510 + 8), f'{subject}: {scale:.0%} internal HDR', fill='white')
    sheet.save(out / 'comparison.png')
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
finally:
    g.destroy_window(w)
    g.terminate()
