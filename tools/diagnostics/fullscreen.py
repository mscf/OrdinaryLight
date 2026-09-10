"""Exercise the actual raster_feature_viewer F11 handler and save its timings.

Run from the component directory with python -m tools.diagnostics.fullscreen.
Unknown arguments are forwarded to raster_feature_viewer (scene, target, etc.).
"""
import argparse
import json
import logging
import os
from pathlib import Path
import runpy
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cold-cache', action='store_true',
                        help='use an isolated empty cache and disable NVIDIA disk caching')
    parser.add_argument('--switch-to', choices=('wavefront-gi', 'wavefront-gi-fast', 'vulkan-raster'),
                        help='select this mode after 30 frames, before testing F11')
    parser.add_argument('--first-toggle-seconds', type=float, default=5)
    parser.add_argument('--cycles', type=int, default=4,
                        help='number of fullscreen/windowed round trips')
    parser.add_argument('--interval-seconds', type=float, default=5)
    parser.add_argument('--timeout-seconds', type=float, default=180)
    parser.add_argument('--report', type=Path,
                        default=Path('/tmp/ordinarylight-fullscreen-report.json'))
    args, viewer_args = parser.parse_known_args()
    if args.switch_to:
        viewer_args += ['--switch-target-after-frames', '30',
                        '--switch-target-to', args.switch_to]
    if "--readback" in viewer_args:
        parser.error("this diagnostic exercises the direct viewer; omit --readback")
    import math
    if (args.cycles < 1 or any(not math.isfinite(v) or v <= 0 for v in (
            args.first_toggle_seconds, args.interval_seconds, args.timeout_seconds))):
        parser.error('cycles and timing values must be positive and finite')
    report = {'viewer_args': viewer_args, 'cold_cache': args.cold_cache,
              'events': [], 'timed_out': False, 'presentation_failed': False,
              'python_executable': sys.executable, 'python_version': sys.version,
              'debugger_attached': sys.gettrace() is not None,
              'monitoring_tools': [sys.monitoring.get_tool(i) for i in range(6)]
                  if hasattr(sys, 'monitoring') else [],
              'environment': {key: os.environ[key] for key in (
                  'TERM_PROGRAM', 'VIRTUAL_ENV', 'PYTHONPATH', 'PYTHONHOME',
                  'QT_QPA_PLATFORM', 'QT_SCALE_FACTOR', 'QT_SCREEN_SCALE_FACTORS',
                  'QT_AUTO_SCREEN_SCALE_FACTOR', 'QT_ENABLE_HIGHDPI_SCALING',
                  'XDG_SESSION_TYPE', 'DISPLAY', 'WAYLAND_DISPLAY',
                  'VK_ICD_FILENAMES', 'VK_DRIVER_FILES', '__NV_PRIME_RENDER_OFFLOAD',
                  '__GLX_VENDOR_LIBRARY_NAME', 'LD_LIBRARY_PATH',
              ) if key in os.environ}}
    if args.cold_cache:
        cache = tempfile.mkdtemp(prefix='ordinarylight-fullscreen-cache-')
        os.environ['XDG_CACHE_HOME'] = cache
        os.environ['__GL_SHADER_DISK_CACHE'] = '0'
        report['cache_directory'] = cache
    from PySide6 import QtCore, QtGui, QtTest
    from ordinarylight.integrations import raster_workbench as viewer

    class Capture(logging.Handler):
        def emit(self, record):
            try:
                event = json.loads(record.getMessage())
            except ValueError:
                return
            if event.get('event') in (
                'fullscreen_requested', 'fullscreen_frame_ready',
                'renderer_start', 'renderer_close', 'vulkan_runtime_close',
                'swapchain', 'swapchain_acquire_retry', 'compute_pipeline_create',
            ):
                report['events'].append(event)
                if event['event'] in ('fullscreen_requested', 'fullscreen_frame_ready',
                                      'renderer_start', 'renderer_close'):
                    print(json.dumps(event), flush=True)

    logger = logging.getLogger('ordinarylight.lifecycle')
    capture = Capture()
    logger.addHandler(capture)
    report["viewer_source"] = viewer.__file__
    original = viewer._direct_main

    def launch(*a, **kw):
        window = original(*a, **kw)
        report['qt_platform'] = QtGui.QGuiApplication.platformName()
        report['device_pixel_ratio'] = window.native_window.devicePixelRatio()
        started = time.monotonic()
        state = {'next': started + args.first_toggle_seconds, 'toggles': 0}
        timer = QtCore.QTimer(window)

        def poll():
            now = time.monotonic()
            failed = window.presentation_failed
            expired = now - started > args.timeout_seconds
            if failed or expired:
                report['presentation_failed'] = failed
                report['timed_out'] = expired
                report['status'] = window.status.text()
                timer.stop()
                window.close()
                return
            if getattr(window, '_fullscreen_transition', None) is not None:
                return
            if now < state['next']:
                return
            if state['toggles'] == args.cycles * 2:
                timer.stop()
                window.close()
                return
            state['toggles'] += 1
            state['next'] = now + args.interval_seconds
            QtTest.QTest.keyClick(window, QtCore.Qt.Key.Key_F11)

        timer.timeout.connect(poll)
        timer.start(20)
        return window

    viewer._direct_main = launch
    try:
        sys.argv = ['raster_feature_viewer', *viewer_args, '--profile-lifecycle']
        runpy.run_path(str(Path(__file__).resolve().parents[1] / 'raster_feature_viewer.py'),
                       run_name='__main__')
    except SystemExit as error:
        if error.code not in (None, 0):
            raise
    finally:
        viewer._direct_main = original
        logger.removeHandler(capture)
        report['completed_transitions'] = sum(
            event['event'] == 'fullscreen_frame_ready' for event in report['events']
        )
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')
        print(f'Fullscreen report: {args.report}', flush=True)
    return int(report['timed_out'] or report['presentation_failed']
               or report['completed_transitions'] != args.cycles * 2)


if __name__ == '__main__':
    raise SystemExit(main())
