"""Measure record-time spikes in the actual animated raster_feature_viewer.

By default, start in raster and switch to experimental GI after 30 frames.
Unknown arguments (scene, dimensions, target) are passed to the viewer.
"""
import argparse
import gc
import json
import math
from pathlib import Path
import runpy
import sys
import time
from unittest.mock import patch

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=45)
    parser.add_argument('--report', type=Path, default=Path('/tmp/ordinarylight-record-stutter.json'))
    args, viewer_args = parser.parse_known_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds must be finite and positive')
    if '--readback' in viewer_args or '--diagnostic-frames' in viewer_args:
        parser.error('this diagnostic measures animated direct presentation')
    if '--target' not in viewer_args:
        viewer_args += ['--target', 'vulkan-raster', '--switch-target-after-frames', '30',
                        '--switch-target-to', 'wavefront-gi-fast']
    from ordinarylight.targets.vulkan.core import VulkanRayQueryCore
    frames, collections, starts = [], [], {}
    original = VulkanRayQueryCore.present_wavefront_window

    def observe_gc(phase, info):
        generation = info['generation']
        if phase == 'start':
            starts[generation] = time.perf_counter()
        else:
            end = time.perf_counter()
            start = starts.pop(generation, end)
            collections.append({'start': start, 'duration_ms': (end - start) * 1000,
                                'generation': generation, 'collected': info['collected']})

    def present(core, *positional, **kwargs):
        sequence = core.wavefront_frame_sequence
        start, cpu_start = time.perf_counter(), time.thread_time()
        result = original(core, *positional, **kwargs)
        # An acquisition timeout has no new timing sample.
        if sequence != core.wavefront_frame_sequence:
            frames.append({
                'start': start, 'call_ms': (time.perf_counter() - start) * 1000,
                'thread_cpu_ms': (time.thread_time() - cpu_start) * 1000,
                **{key: value for key, value in core.last_timings.items()
                   if key.endswith('_ms') or key in (
                       'wavefront_command_cache_hit', 'wavefront_output_extent',
                       'wavefront_render_extent', 'wavefront_restir_history_valid',
                       'wavefront_restir_effective_history_limit',
                       'wavefront_temporal_motion_pixels')},
            })
        return result

    original_argv = sys.argv
    gc.callbacks.append(observe_gc)
    try:
        with patch.object(VulkanRayQueryCore, 'present_wavefront_window', present):
            sys.argv = ['raster_feature_viewer', *viewer_args,
                        '--close-after-ms', str(round(args.seconds * 1000))]
            try:
                runpy.run_path(str(Path(__file__).resolve().parents[1] / 'raster_feature_viewer.py'),
                               run_name='__main__')
            except SystemExit as error:
                if error.code not in (None, 0):
                    raise
    finally:
        gc.callbacks.remove(observe_gc)
        sys.argv = original_argv
        measured = frames[50:]
        summary = {'frames': len(frames), 'warmup_discarded': min(50, len(frames))}
        if measured:
            values = [frame['wavefront_record_ms'] for frame in measured]
            summary.update(zip(('record_median_ms', 'record_p95_ms', 'record_p99_ms',
                                'record_max_ms'), map(float, np.percentile(values, (50, 95, 99, 100)))))
            summary['command_cache_hit_fraction'] = float(np.mean([
                frame['wavefront_command_cache_hit'] for frame in measured]))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({'viewer_args': viewer_args, 'seconds': args.seconds,
                                          'summary': summary, 'frames': frames,
                                          'collections': collections}, indent=2) + '\n')
        print(json.dumps(summary), flush=True)
        print('Report:', args.report, flush=True)
    return 0 if measured else 1


if __name__ == '__main__':
    raise SystemExit(main())
