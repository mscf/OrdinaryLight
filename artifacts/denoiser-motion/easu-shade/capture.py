from pathlib import Path
from types import SimpleNamespace
import json
import os
import numpy as np
from PIL import Image, ImageDraw
import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.integrations.raster_workbench import _gi_config
from tools.denoiser_motion.glass_detail import fixture, pose
from read_output import read_output

scale = float(os.environ.get("SCALE", ".5"))
out = Path(os.environ.get("OUT", "/tmp/easu-shade"))
out.mkdir(exist_ok=True)
width = int(os.environ.get("WIDTH", "640"))
height = int(os.environ.get("HEIGHT", "480"))
g = load_glfw()
assert g.init()
g.window_hint(g.CLIENT_API, g.NO_API)
g.window_hint(g.VISIBLE, g.FALSE)
w = g.create_window(width, height, "EASU comparison", None, None)
report = {}
sheet = Image.new("RGB", (2 * width, 3 * (height + 30)))
try:
    for row, subject in enumerate(("glass", "target", "camera")):
        captures = {}
        for col, mode in enumerate(("fsr1", "fsr1-shade")):
            scene, glass, bars = fixture()
            cfg = _gi_config(
                SimpleNamespace(id="glass-detail", renderer={}),
                capture=True,
                render_scale=scale,
                upscale_filter=mode,
            )
            frames = []
            hdr = []
            with ol.VulkanGlfwPresenter(w, config=cfg) as p:
                for x in list(np.linspace(-0.3, 0.3, 12)) + [0.3] * 12:
                    p.present_wavefront(
                        scene,
                        pose(scene, glass, bars, subject, float(x)),
                        width,
                        height,
                    )
                    frames.append(read_output(p._core)["output"])
                    hdr.append(p.capture_wavefront_hdr())
            captures[mode] = np.array(frames)
            np.save(out / f"{subject}-{mode}.npy", frames)
            if col == 0:
                baseline = np.array(hdr)
            else:
                assert np.array_equal(baseline, hdr), "upscale changed internal HDR"
            sheet.paste(
                Image.fromarray(frames[-1][..., :3]),
                (width * col, (height + 30) * row + 30),
            )
            ImageDraw.Draw(sheet).text(
                (width * col + 8, (height + 30) * row + 8),
                f"{subject}: {mode}",
                fill="white",
            )
        a = captures["fsr1"]
        b = captures["fsr1-shade"]
        if scale == 1:
            assert np.array_equal(a, b), "native rendering changed"
        else:
            assert np.max(np.abs(a.astype(int) - b.astype(int))) <= 1, (
                "port differs by more than one display level"
            )
        report[subject] = {
            "mean_absolute_display_difference": float(
                np.abs(a.astype(float) - b).mean()
            ),
            "internal_hdr_identical": True,
            "max_display_difference": int(
                np.max(np.abs(a.astype(int) - b.astype(int)))
            ),
            "changed_channels": int(np.count_nonzero(a != b)),
        }
        for mode, frames in captures.items():
            report[subject][mode + "_settled_display_std"] = float(
                (frames[-8:, ..., :3].astype(float) / 255).std(axis=0).mean()
            )
    sheet.save(out / "comparison.png")
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
finally:
    g.destroy_window(w)
    g.terminate()
