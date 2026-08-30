"""Live Qt catalog for Ordinary Light raster features."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import ordinarylight as ol
from ordinarylight.integrations.workbench import discover_showcases
from ordinarylight.outputs import to_sdr


RESOLUTIONS = (
    ("Preview — 720p", (1280, 720)),
    ("Full HD — 1080p", (1920, 1080)),
    ("QHD — 1440p", (2560, 1440)),
    ("Ultra HD — 4K", (3840, 2160)),
)


def _direct_main(QtCore, QtGui, QtWidgets, showcases, args):
    """Run the Qt-owned XCB Vulkan-surface showcase."""
    from ordinarylight.integrations.qt_vulkan import QtVulkanSurface

    class NativeWindow(QtGui.QWindow):
        def __init__(self):
            super().__init__()
            self.owner = None
            self.last_position = None

        def mousePressEvent(self, event):
            self.last_position = event.position()

        def mouseMoveEvent(self, event):
            if self.last_position is None or self.owner.controller is None:
                return
            position = event.position()
            dx = position.x() - self.last_position.x()
            dy = position.y() - self.last_position.y()
            self.last_position = position
            if event.buttons() & QtCore.Qt.MouseButton.LeftButton:
                self.owner.controller.orbit(-dx * 0.007, -dy * 0.007)
            elif event.buttons() & (
                QtCore.Qt.MouseButton.RightButton |
                QtCore.Qt.MouseButton.MiddleButton
            ):
                self.owner.controller.pan(
                    dx / max(self.width(), 1), dy / max(self.height(), 1),
                )

        def mouseReleaseEvent(self, event):
            self.last_position = None

        def wheelEvent(self, event):
            if self.owner.controller is not None:
                self.owner.controller.dolly(event.angleDelta().y() / 120.0)

    class DirectWindow(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Ordinary Light — direct Vulkan raster showcase")
            self.resize(1500, 900)
            root = QtWidgets.QWidget(); self.setCentralWidget(root)
            layout = QtWidgets.QHBoxLayout(root)
            self.native_window = NativeWindow()
            self.native_window.owner = self
            self.surface = QtVulkanSurface(self.native_window)
            self.container = QtWidgets.QWidget.createWindowContainer(
                self.native_window,
            )
            self.container.setMinimumSize(640, 420)
            layout.addWidget(self.container, 1)
            panel = QtWidgets.QWidget(); panel.setMaximumWidth(430)
            form = QtWidgets.QFormLayout(panel); layout.addWidget(panel)
            self.feature = QtWidgets.QComboBox()
            for item in showcases:
                self.feature.addItem(item.title, item)
            self.resolution = QtWidgets.QComboBox()
            custom = (max(1, args.width), max(1, args.height))
            self.resolution.addItem(
                f"Custom — {custom[0]} × {custom[1]}", custom,
            )
            for title, extent in RESOLUTIONS:
                if extent != custom:
                    self.resolution.addItem(
                        f"{title} ({extent[0]} × {extent[1]})", extent,
                    )
            self.shadows = QtWidgets.QCheckBox(); self.shadows.setChecked(True)
            self.map_size = QtWidgets.QComboBox()
            for size in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
                self.map_size.addItem(str(size), size)
            self.animate = QtWidgets.QCheckBox(); self.animate.setChecked(True)
            self.description = QtWidgets.QLabel(wordWrap=True)
            self.help = QtWidgets.QLabel(
                "Direct Vulkan swapchain (no NumPy/QImage readback)\n"
                "Left drag: orbit · Right/middle drag: pan · Wheel: dolly",
                wordWrap=True,
            )
            button = QtWidgets.QPushButton("Apply and restart renderer")
            button.clicked.connect(self.restart)
            self.feature.currentIndexChanged.connect(self._selection_changed)
            form.addRow("Feature", self.feature)
            form.addRow("Render resolution", self.resolution)
            form.addRow("Enable shadows", self.shadows)
            form.addRow("Shadow map size", self.map_size)
            form.addRow("Animate camera", self.animate)
            form.addRow(self.description); form.addRow(self.help)
            form.addRow(button)
            self.status = QtWidgets.QLabel(wordWrap=True); form.addRow(self.status)
            self.renderer = None
            self.scene_value = None
            self.controller = None
            self.future = None
            self.executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="ordinarylight-qt-vulkan",
            )
            self.completed = deque(maxlen=240)
            self.last_tick = time.perf_counter()
            self.timer = QtCore.QTimer(self)
            self.timer.setInterval(1)
            self.timer.timeout.connect(self.tick)
            self.timer.start()
            self._selection_changed()
            QtCore.QTimer.singleShot(0, self.restart)

        @property
        def extent(self):
            return tuple(int(value) for value in self.resolution.currentData())

        def _selection_changed(self, _index=None):
            item = self.feature.currentData()
            self.description.setText(item.description)
            self.shadows.setChecked(bool(item.renderer.get("shadows", True)))
            value = int(item.renderer.get("shadow_map_size", 512))
            index = self.map_size.findData(value)
            self.map_size.setCurrentIndex(max(index, 0))

        def _wait(self):
            if self.future is not None:
                self.future.result()
                self.future = None

        def restart(self):
            self.status.setText("Initializing direct Vulkan renderer…")
            try:
                self._wait()
                if self.renderer is not None:
                    self.renderer.close()
                item = self.feature.currentData()
                self.scene_value = item.create_scene()
                self.controller = ol.ArcballCameraController.from_camera(
                    item.camera.camera(self.scene_value, angle=-0.45),
                )
                settings = dict(item.renderer)
                default_material = settings.get("material_program")
                if default_material is None:
                    default_material = ol.builtin_material
                target = ol.RasterProgram.scene(
                    target="spirv", validate=False,
                    material_programs=self.scene_value.material_programs(
                        default_material,
                    ),
                )
                settings.update(
                    shadows=self.shadows.isChecked(),
                    shadow_map_size=int(self.map_size.currentData()),
                )
                config = ol.RasterConfig(
                    state=ol.RasterState(cull_mode="none"),
                    ambient_light=float(settings.pop("ambient_light", 0.08)),
                    **settings,
                )
                self.renderer = ol.renderers.raster.VulkanRasterRenderer(
                    target, config=config,
                    instance=self.surface.instance,
                    surface=self.surface.surface,
                )
                self.completed.clear()
                self.status.setText("Direct Vulkan renderer ready")
            except Exception as error:
                self.status.setText(f"Renderer start failed: {error}")

        def tick(self):
            now = time.perf_counter()
            elapsed = min(now - self.last_tick, 0.1)
            self.last_tick = now
            if self.animate.isChecked() and self.controller is not None:
                self.controller.orbit(elapsed * 0.35, 0.0)
            if self.future is not None:
                if not self.future.done():
                    return
                try:
                    self.future.result()
                except Exception as error:
                    self.status.setText(f"Direct presentation failed: {error}")
                    self.future = None
                    return
                self.future = None
                complete = time.perf_counter()
                self.completed.append(complete)
                while self.completed and complete - self.completed[0] > 1.0:
                    self.completed.popleft()
                fps = 0.0
                if len(self.completed) > 1:
                    fps = (len(self.completed) - 1) / (
                        self.completed[-1] - self.completed[0]
                    )
                width, height = self.extent
                timings = self.renderer.last_timings
                ms = timings.get("total_ms", 0.0)
                pack_ms = timings.get("scene_pack_ms", 0.0)
                prepare_ms = timings.get("scene_prepare_ms", 0.0)
                submit_ms = timings.get("resident_submit_ms", 0.0)
                resident = "resident" if timings.get("resident_cache_hit") else "warming"
                present_mode = timings.get("present_mode", "unknown").upper()
                self.status.setText(
                    f"{fps:.1f} FPS | frame {ms:.2f} ms | pack {pack_ms:.2f} ms "
                    f"| prepare {prepare_ms:.2f} ms "
                    f"| submit {submit_ms:.2f} ms | {width} × {height} "
                    f"| {present_mode} · direct swapchain · {resident}",
                )
            if self.renderer is None or self.controller is None:
                return
            render_width, render_height = self.extent
            ratio = float(self.native_window.devicePixelRatio())
            surface_size = (
                max(1, round(self.native_window.width() * ratio)),
                max(1, round(self.native_window.height() * ratio)),
            )
            camera = self.controller.camera()
            self.future = self.executor.submit(
                self.renderer.present_frame,
                self.scene_value, camera, render_width, render_height,
                surface_size=surface_size,
            )

        def closeEvent(self, event):
            self.timer.stop()
            try:
                self._wait()
            finally:
                if self.renderer is not None:
                    self.renderer.close()
                self.executor.shutdown(wait=True, cancel_futures=True)
                self.surface.close()
            event.accept()

    window = DirectWindow(); window.show()
    return window


def _catalog():
    path = ROOT / "ordinarylight" / "showcases" / "catalog"
    return tuple(
        item for item in discover_showcases((path,))
        if "raster-feature" in item.tags
    )


def _renderer(showcase, backend_name, shadows, shadow_map_size):
    target = "spirv" if backend_name == "vulkan" else "wgsl"
    program = ol.RasterProgram.scene(target=target, validate=False)
    settings = dict(showcase.renderer)
    settings.update(shadows=shadows, shadow_map_size=shadow_map_size)
    config = ol.RasterConfig(
        state=ol.RasterState(cull_mode="none"),
        ambient_light=float(settings.pop("ambient_light", 0.08)),
        **settings,
    )
    implementation_type = (
        ol.renderers.raster.VulkanRasterRenderer
        if backend_name == "vulkan" else
        ol.renderers.raster.WebGpuRasterRenderer
    )
    return ol.Renderer(implementation=implementation_type(program, config=config))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--readback", action="store_true",
        help="use the NumPy/QImage comparison path instead of direct Vulkan",
    )
    args = parser.parse_args()
    if not args.readback:
        # PySide exposes Qt's XCB connection, allowing a Qt-owned native window
        # to become a Vulkan surface without embedding a GLFW child.
        os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
    except ImportError as error:
        raise RuntimeError("Install the Qt extra: pip install 'ordinarylight[qt]'") from error

    showcases = _catalog()
    if not showcases:
        raise RuntimeError("no raster feature showcases were discovered")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    if not args.readback:
        window = _direct_main(QtCore, QtGui, QtWidgets, showcases, args)
        return app.exec()

    class Viewport(QtWidgets.QLabel):
        def __init__(self, owner):
            super().__init__(alignment=QtCore.Qt.AlignmentFlag.AlignCenter)
            self.owner = owner
            self._last_position = None
            self.setMinimumSize(640, 420)
            self.setMouseTracking(True)

        def mousePressEvent(self, event):
            self._last_position = event.position()
            event.accept()

        def mouseMoveEvent(self, event):
            if self._last_position is None or self.owner.controller is None:
                return
            position = event.position()
            dx = float(position.x() - self._last_position.x())
            dy = float(position.y() - self._last_position.y())
            self._last_position = position
            buttons = event.buttons()
            if buttons & QtCore.Qt.MouseButton.LeftButton:
                self.owner.controller.orbit(-dx * 0.007, -dy * 0.007)
            elif buttons & (
                QtCore.Qt.MouseButton.RightButton |
                QtCore.Qt.MouseButton.MiddleButton
            ):
                self.owner.controller.pan(
                    dx / max(self.width(), 1), dy / max(self.height(), 1),
                )
            event.accept()

        def mouseReleaseEvent(self, event):
            self._last_position = None
            event.accept()

        def wheelEvent(self, event):
            if self.owner.controller is not None:
                self.owner.controller.dolly(event.angleDelta().y() / 120.0)
            event.accept()

        def resizeEvent(self, event):
            super().resizeEvent(event)
            self.owner._update_pixmap()

    class Window(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Ordinary Light — live raster feature showcase")
            self.resize(1500, 900)
            self.renderers = []
            self.jobs = []
            self.scene_value = None
            self.controller = None
            self._qimage = None
            self._pixels = None
            self._last_tick = time.perf_counter()
            self._completed_times = deque(maxlen=240)

            root = QtWidgets.QWidget(); self.setCentralWidget(root)
            layout = QtWidgets.QHBoxLayout(root)
            self.image = Viewport(self); layout.addWidget(self.image, 1)
            panel = QtWidgets.QWidget(); panel.setMaximumWidth(430)
            form = QtWidgets.QFormLayout(panel); layout.addWidget(panel)
            self.feature = QtWidgets.QComboBox()
            for item in showcases:
                self.feature.addItem(item.title, item)
            self.backend = QtWidgets.QComboBox()
            self.backend.addItems(("vulkan", "webgpu", "both"))
            self.resolution = QtWidgets.QComboBox()
            custom = (max(1, args.width), max(1, args.height))
            self.resolution.addItem(f"Custom — {custom[0]} × {custom[1]}", custom)
            for title, extent in RESOLUTIONS:
                if extent != custom:
                    self.resolution.addItem(f"{title} ({extent[0]} × {extent[1]})", extent)
            self.shadows = QtWidgets.QCheckBox(); self.shadows.setChecked(True)
            self.map_size = QtWidgets.QComboBox()
            for size in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
                self.map_size.addItem(str(size), size)
            self.animate = QtWidgets.QCheckBox(); self.animate.setChecked(True)
            self.live = QtWidgets.QCheckBox(); self.live.setChecked(True)
            self.description = QtWidgets.QLabel(wordWrap=True)
            self.help = QtWidgets.QLabel(
                "Left drag: orbit\nRight/middle drag: pan\nWheel: dolly",
                wordWrap=True,
            )
            button = QtWidgets.QPushButton("Apply and restart live renderer")
            button.clicked.connect(self.restart)
            self.feature.currentIndexChanged.connect(self._selection_changed)
            form.addRow("Feature", self.feature)
            form.addRow("Backend", self.backend)
            form.addRow("Resolution", self.resolution)
            form.addRow("Enable shadows", self.shadows)
            form.addRow("Shadow map size", self.map_size)
            form.addRow("Animate camera", self.animate)
            form.addRow("Live rendering", self.live)
            form.addRow(self.description); form.addRow(self.help)
            form.addRow(button)
            self.status = QtWidgets.QLabel(wordWrap=True); form.addRow(self.status)

            self.timer = QtCore.QTimer(self)
            self.timer.setInterval(5)
            self.timer.timeout.connect(self._tick)
            self.timer.start()
            self._selection_changed()
            QtCore.QTimer.singleShot(0, self.restart)

        @property
        def extent(self):
            width, height = self.resolution.currentData()
            return int(width), int(height)

        def _selection_changed(self, _index=None):
            item = self.feature.currentData()
            self.description.setText(item.description)
            value = int(item.renderer.get("shadow_map_size", 512))
            index = self.map_size.findData(value)
            self.map_size.setCurrentIndex(max(index, 0))

        def _close_renderers(self):
            for _name, job in self.jobs:
                job.cancel()
            self.jobs.clear()
            for _name, renderer in self.renderers:
                renderer.close()
            self.renderers.clear()

        def restart(self):
            self.status.setText("Initializing live renderer…")
            try:
                self._close_renderers()
                item = self.feature.currentData()
                selected = self.backend.currentText()
                names = ("vulkan", "webgpu") if selected == "both" else (selected,)
                self.scene_value = item.create_scene()
                authored = item.camera.camera(self.scene_value, angle=-0.45)
                self.controller = ol.ArcballCameraController.from_camera(authored)
                self.renderers = [
                    (name, _renderer(
                        item, name, self.shadows.isChecked(), int(self.map_size.currentData()),
                    ))
                    for name in names
                ]
                self._completed_times.clear()
                self._last_tick = time.perf_counter()
                self.status.setText("Live renderer ready")
            except Exception as error:
                self._close_renderers()
                self.status.setText(f"Renderer start failed: {error}")

        def _submit(self):
            if not self.renderers or self.scene_value is None or self.controller is None:
                return
            camera = self.controller.camera()
            self.jobs = [
                (name, renderer.render_async(self.scene_value, camera, self.extent))
                for name, renderer in self.renderers
            ]

        def _tick(self):
            now = time.perf_counter()
            elapsed = min(now - self._last_tick, 0.1)
            self._last_tick = now
            if self.animate.isChecked() and self.controller is not None:
                self.controller.orbit(elapsed * 0.35, 0.0)
            if self.jobs:
                if not all(job.done() for _name, job in self.jobs):
                    return
                try:
                    completed = [(name, to_sdr(job.result())) for name, job in self.jobs]
                    statistics = [job.statistics for _name, job in self.jobs]
                    self.jobs.clear()
                    self._display(completed)
                    completed_at = time.perf_counter()
                    self._completed_times.append(completed_at)
                    while self._completed_times and completed_at - self._completed_times[0] > 1.0:
                        self._completed_times.popleft()
                    fps = 0.0
                    if len(self._completed_times) > 1:
                        fps = (len(self._completed_times) - 1) / (
                            self._completed_times[-1] - self._completed_times[0]
                        )
                    gpu = [stat.gpu_ms for stat in statistics if stat and stat.gpu_ms is not None]
                    gpu_text = f" | GPU {max(gpu):.2f} ms" if gpu else ""
                    width, height = self.extent
                    self.status.setText(
                        f"{fps:.1f} FPS{gpu_text} | {width} × {height} | "
                        + ", ".join(name for name, _image in completed)
                    )
                except Exception as error:
                    self.jobs.clear()
                    self.status.setText(f"Render failed: {error}")
                    return
            if self.live.isChecked() or self._qimage is None:
                self._submit()

        def _display(self, images):
            width, height = self.extent
            gap = 8
            canvas = np.full(
                (height, len(images) * width + (len(images) - 1) * gap, 4),
                255, np.uint8,
            )
            for index, (_name, pixels) in enumerate(images):
                x = index * (width + gap)
                canvas[:, x:x + width, :3] = pixels
            self._pixels = np.ascontiguousarray(canvas)
            self._qimage = QtGui.QImage(
                self._pixels.data, self._pixels.shape[1], self._pixels.shape[0],
                self._pixels.strides[0], QtGui.QImage.Format.Format_RGBA8888,
            ).copy()
            self._update_pixmap()

        def _update_pixmap(self):
            if self._qimage is None:
                return
            self.image.setPixmap(QtGui.QPixmap.fromImage(self._qimage).scaled(
                self.image.size(), QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.SmoothTransformation,
            ))

        def closeEvent(self, event):
            self.timer.stop()
            self._close_renderers()
            event.accept()

    window = Window(); window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
