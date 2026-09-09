"""Compare native resolve recorders; use --force-recording to bypass command caching."""

import argparse
import ast, subprocess, textwrap, struct
from types import SimpleNamespace
import numpy as np
import vulkan as vk
import ordinarylight as ol
from ordinarylight.targets.vulkan.core import VulkanWavefrontExecutor
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.integrations.raster_workbench import _gi_config
from tools.denoiser_motion.glass_detail import fixture, pose

source = subprocess.check_output(
    [
        "git",
        "show",
        "966c745fc5ea6dbc879c285d174998ccb520619a:ordinarylight/targets/vulkan/core.py",
    ],
    text=True,
)
tree = ast.parse(source)
cls = next(
    n
    for n in tree.body
    if isinstance(n, ast.ClassDef) and n.name == "VulkanWavefrontExecutor"
)
method = next(
    n
    for n in cls.body
    if isinstance(n, ast.FunctionDef) and n.name == "record_path_to_hdr"
)
namespace = {"vk": vk, "struct": struct}
exec(textwrap.dedent(ast.get_source_segment(source, method)), namespace)
new = VulkanWavefrontExecutor.record_path_to_hdr
legacy = namespace["record_path_to_hdr"]
import json, time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--force-recording", action="store_true")
parser.add_argument("--output", default="/tmp/resolve-performance.json")
args = parser.parse_args()
WIDTH, HEIGHT = 1280, 720
WARMUP, SAMPLES = 12, 30
output = Path(args.output)
glfw = load_glfw()
assert glfw.init()
glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
window = glfw.create_window(WIDTH, HEIGHT, "resolve performance", None, None)
results = []
try:
    for motion in (False, True):
        for label in ("legacy", "shared", "shared", "legacy"):
            base = legacy if label == "legacy" else new
            timings = []

            def measured(self, *args, **kwargs):
                started = time.perf_counter()
                try:
                    return base(self, *args, **kwargs)
                finally:
                    timings.append((time.perf_counter() - started) * 1000)

            VulkanWavefrontExecutor.record_path_to_hdr = measured
            scene, glass, bars = fixture()
            cfg = _gi_config(
                SimpleNamespace(id="glass-detail", renderer={}),
                render_scale=0.5,
                upscale_filter="fsr1-shade",
                capture=False,
                present=True,
            )
            rows = []
            with ol.VulkanGlfwPresenter(window, config=cfg) as p:
                for i in range(WARMUP + SAMPLES):
                    camera = pose(
                        scene, glass, bars, "camera", i * 0.001 if motion else 0
                    )
                    if args.force_recording:
                        for frame in p._core.window_frames:
                            frame["wavefront_command_key"] = None
                    timings.clear()
                    start = time.perf_counter()
                    p.present_wavefront(scene, camera, WIDTH, HEIGHT)
                    vk.vkQueueWaitIdle(p._core.queue)
                    wall = (time.perf_counter() - start) * 1000
                    if i >= WARMUP:
                        row = {
                            key: float(p.last_timings[key])
                            for key in (
                                "gpu_frame_ms",
                                "wavefront_record_ms",
                                "wavefront_command_record_ms",
                                "wavefront_command_cache_hit",
                            )
                            if key in p.last_timings
                        }
                        row.update(
                            wall_ms=wall,
                            resolve_cpu_ms=sum(timings),
                            resolve_calls=len(timings),
                        )
                        rows.append(row)
            summary = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
            results.append(
                dict(path=label, motion=motion, summary=summary, frames=rows)
            )
            output.write_text(
                json.dumps(
                    dict(
                        width=WIDTH,
                        height=HEIGHT,
                        render_scale=0.5,
                        warmup=WARMUP,
                        samples=SAMPLES,
                        results=results,
                    ),
                    indent=2,
                )
            )
            print(label, "moving" if motion else "static", summary, flush=True)
finally:
    VulkanWavefrontExecutor.record_path_to_hdr = new
    glfw.destroy_window(window)
    glfw.terminate()
