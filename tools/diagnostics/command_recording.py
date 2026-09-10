"""A/B cache invalidation during camera motion; reports host cost and HDR parity."""
import argparse
from dataclasses import replace
import json
import time
from pathlib import Path

import numpy as np
import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.integrations.raster_workbench import _gi_config
from ordinarylight.showcases.catalog.raster import SHOWCASES
from ordinarylight.targets.vulkan import core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=96)
    parser.add_argument('--width', type=int, default=960)
    parser.add_argument('--height', type=int, default=540)
    parser.add_argument('--output', type=Path, default=Path('/tmp/command-recording.json'))
    parser.add_argument('--spp', type=int, default=2, choices=(1, 2))
    parser.add_argument('--fast', action='store_true', help='Use experimental four-bounce GI (requires --spp 1)')
    parser.add_argument('--uploaded-policy', action='store_true',
                        help='Compare recorded ReSTIR policy with per-frame uploaded policy')
    parser.add_argument('--history-resets', action='store_true',
                        help='Alternate camera jumps and stops; compare forced recording with reuse')
    args = parser.parse_args()
    if min(args.frames, args.width, args.height) <= 0:
        parser.error('frames and dimensions must be positive')
    if args.fast and args.spp != 1:
        parser.error('--fast requires --spp 1')
    item = next(s for s in SHOWCASES if s.id == 'optical-screen-rough-reflection')
    scene = item.build()
    config = replace(_gi_config(item, shared_primary=True, path_spp=args.spp,
                               restir_reservoirs=2, low_bounce=args.fast), wavefront_hdr_capture=True)
    glfw = load_glfw()
    if not glfw.init():
        raise RuntimeError('GLFW initialization failed')
    original = core._command_history_limits
    from ordinarylight.shaders import compiler
    original_compile = compiler.compile_wavefront_material_shader
    window = None
    results, images = {}, {}
    try:
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
        window = glfw.create_window(args.width, args.height, 'Command recording A/B', None, None)
        if not window:
            raise RuntimeError('Window creation failed')
        for name in ('baseline', 'optimized'):
            def compile_variant(*positional, **kwargs):
                if args.uploaded_policy and name == 'baseline':
                    kwargs['camera_restir_policy'] = False
                return original_compile(*positional, **kwargs)
            compiler.compile_wavefront_material_shader = compile_variant
            core._command_history_limits = (
                (lambda indirect, restir, **kw: (indirect, restir))
                if name == 'baseline' else original
            )
            rows = []
            with ol.VulkanGlfwPresenter(window, config=config) as renderer:
                angle = -0.45
                for frame in range(32 + args.frames):
                    # Vary speed deterministically to exercise motion-dependent
                    # history limits, independent of the achieved frame rate.
                    if frame >= 32:
                        if args.history_resets:
                            # Two moving frames followed by two stopped frames
                            # change history validity for both frame slots.
                            angle += 0.3 if frame % 4 < 2 else 0.0
                        else:
                            angle += 0.0003 + 0.002 * (0.5 + 0.5 * np.sin(frame * 0.7))
                    camera = item.camera.camera(scene, angle=float(angle))
                    glfw.poll_events()
                    if (args.history_resets or args.uploaded_policy) and name == 'baseline':
                        for slot in renderer._core.window_frames:
                            slot['wavefront_command_key'] = None
                    start = time.perf_counter()
                    renderer.present_wavefront(scene, camera, args.width, args.height)
                    elapsed = 1000 * (time.perf_counter() - start)
                    if frame >= 32:
                        t = renderer.last_timings
                        rows.append([t['wavefront_record_ms'], t['gpu_frame_ms'],
                                     elapsed, t['wavefront_command_cache_hit']])
                images[name] = renderer.capture_wavefront_hdr().copy()
            rows = np.array(rows)
            results[name] = {
                'record_median_ms': float(np.median(rows[:, 0])),
                'record_p95_ms': float(np.percentile(rows[:, 0], 95)),
                'gpu_median_ms': float(np.median(rows[:, 1])),
                'present_call_median_ms': float(np.median(rows[:, 2])),
                'present_call_mean_ms': float(rows[:, 2].mean()),
                'command_cache_hit_fraction': float(rows[:, 3].mean()),
            }
            print(name, json.dumps(results[name]), flush=True)
        assert all(np.isfinite(im).all() for im in images.values())
        difference = float(np.max(np.abs(images['baseline'] - images['optimized'])))
        results['final_hdr_max_absolute_difference'] = difference
        args.output.write_text(json.dumps(results, indent=2) + '\n')
        print('HDR difference:', difference, 'report:', args.output, flush=True)
        assert difference == 0, 'Cache optimization changed rendering'
        if args.history_resets or args.uploaded_policy:
            assert results['optimized']['command_cache_hit_fraction'] > 0.9, (
                'History resets caused repeated command recording'
            )
    finally:
        core._command_history_limits = original
        compiler.compile_wavefront_material_shader = original_compile
        if window:
            glfw.destroy_window(window)
        glfw.terminate()


if __name__ == '__main__':
    main()
