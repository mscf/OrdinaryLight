"""Compare direct and composed fused-primary recorders; use --force-recording to bypass command caching."""

import argparse
from types import SimpleNamespace
import numpy as np
import vulkan as vk
import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.integrations.raster_workbench import _gi_config
from tools.denoiser_motion.glass_detail import fixture, pose

import ordinarylight.targets.vulkan.primary_graph as primary_graph
from primary_baseline import direct_primary

new = primary_graph.record_primary
legacy = direct_primary
import json, time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--force-recording", action="store_true")
parser.add_argument("--output", default="/tmp/primary-performance.json")
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--render-scale", type=float, default=0.5)
args = parser.parse_args()
WIDTH, HEIGHT = args.width, args.height
WARMUP, SAMPLES = 12, 30
output = Path(args.output)
glfw = load_glfw()
assert glfw.init()
glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
window = glfw.create_window(WIDTH, HEIGHT, "primary performance", None, None)
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

            primary_graph.record_primary = measured
            scene, glass, bars = fixture()
            cfg = _gi_config(
                SimpleNamespace(id="glass-detail", renderer={}),
                render_scale=args.render_scale,
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
                            primary_cpu_ms=sum(timings),
                            primary_calls=len(timings),
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
                        render_scale=args.render_scale,
                        warmup=WARMUP,
                        samples=SAMPLES,
                        results=results,
                    ),
                    indent=2,
                )
            )
            print(label, "moving" if motion else "static", summary, flush=True)
finally:
    primary_graph.record_primary = new
    glfw.destroy_window(window)
    glfw.terminate()
