"""Live Qt catalog for comparing Ordinary Light rendering targets."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The source-checkout showcase uses Python-authored Ordinary Shade hooks.  Make
# the sibling checkout available without requiring developers to install an
# editable package first.  Installed distributions continue to use their
# normally installed dependency.
SHADE_ROOT = ROOT.parent / "ordinaryshade"
if (
    (SHADE_ROOT / "ordinaryshade").is_dir()
    and str(SHADE_ROOT) not in sys.path
):
    sys.path.insert(0, str(SHADE_ROOT))

import ordinarylight as ol
from ordinarylight.integrations.workbench import discover_showcases
from ordinarylight.outputs import to_sdr


RESOLUTIONS = (
    ("Preview — 720p", (1280, 720)),
    ("Full HD — 1080p", (1920, 1080)),
    ("QHD — 1440p", (2560, 1440)),
    ("Ultra HD — 4K", (3840, 2160)),
)

TARGETS = (
    ("Vulkan raster", "vulkan-raster"),
    ("Wavefront GI", "wavefront-gi"),
    ("WebGPU raster", "webgpu-raster"),
)


def _gi_config(showcase, *, present=False):
    """Build the interactive GI configuration corresponding to a showcase."""
    settings = dict(showcase.renderer)
    return ol.RendererConfig(
        samples_per_pixel=1,
        max_bounces=int(settings.get("max_bounces", 8)),
        present_mode="mailbox",
        progressive_accumulation=False,
        material_program=settings.get("material_program"),
        material_modifier=settings.get(
            "material_modifier", settings.get("material_hook")
        ),
        direct_swapchain_storage=bool(present),
    )


def _preserved_view(showcase, scene, controller, active_showcase_id):
    """Retain scene/camera state unless the selected showcase changed."""
    if scene is not None and active_showcase_id == showcase.id:
        return scene, controller, active_showcase_id
    scene = showcase.create_scene()
    controller = ol.ArcballCameraController.from_camera(
        showcase.camera.camera(scene, angle=-0.45),
    )
    return scene, controller, showcase.id


def _direct_render_extent(target, selected_extent, surface_extent):
    """Resolve the extent consumed by a direct-presentation renderer."""
    if target == "wavefront-gi":
        return tuple(surface_extent)
    return tuple(selected_extent)


def _surface_aspect_extent(selected_extent, surface_extent):
    """Fit a resolution budget to the native viewport aspect ratio."""
    selected_width, selected_height = (int(value) for value in selected_extent)
    surface_width, surface_height = (int(value) for value in surface_extent)
    scale = min(
        selected_width / max(surface_width, 1),
        selected_height / max(surface_height, 1),
    )
    return (
        max(1, int(round(surface_width * scale))),
        max(1, int(round(surface_height * scale))),
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
            self.setWindowTitle("Ordinary Light — renderer parity showcase")
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
            self.readback_image = QtWidgets.QLabel(
                alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
            )
            self.readback_image.setMinimumSize(640, 420)
            self.viewport_stack = QtWidgets.QStackedWidget()
            self.viewport_stack.addWidget(self.container)
            self.viewport_stack.addWidget(self.readback_image)
            layout.addWidget(self.viewport_stack, 1)
            panel = QtWidgets.QWidget(); panel.setMaximumWidth(430)
            form = QtWidgets.QFormLayout(panel); layout.addWidget(panel)
            self.feature = QtWidgets.QComboBox()
            for item in showcases:
                self.feature.addItem(item.title, item)
            self.target = QtWidgets.QComboBox()
            for title, key in TARGETS:
                self.target.addItem(title, key)
            selected_target = self.target.findData(args.target)
            self.target.setCurrentIndex(max(selected_target, 0))
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
            self._selected_extent = custom
            self.resolution.currentIndexChanged.connect(
                self._resolution_changed,
            )
            self.shadows = QtWidgets.QCheckBox(); self.shadows.setChecked(True)
            self.map_size = QtWidgets.QComboBox()
            for size in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
                self.map_size.addItem(str(size), size)
            self.animate = QtWidgets.QCheckBox(); self.animate.setChecked(True)
            self.description = QtWidgets.QLabel(wordWrap=True)
            self.help = QtWidgets.QLabel(
                "Vulkan targets use the direct swapchain; WebGPU currently "
                "uses offscreen QImage readback.\n"
                "Left drag: orbit · Right/middle drag: pan · Wheel: dolly",
                wordWrap=True,
            )
            button = QtWidgets.QPushButton("Apply and restart renderer")
            button.clicked.connect(self.restart)
            self.feature.currentIndexChanged.connect(self._selection_changed)
            self.target.currentIndexChanged.connect(self._target_changed)
            form.addRow("Feature", self.feature)
            form.addRow("Rendering target", self.target)
            form.addRow("Render resolution", self.resolution)
            form.addRow("Enable shadows", self.shadows)
            form.addRow("Shadow map size", self.map_size)
            form.addRow("Animate camera", self.animate)
            form.addRow(self.description); form.addRow(self.help)
            form.addRow(button)
            self.status = QtWidgets.QLabel(wordWrap=True); form.addRow(self.status)
            self.renderer = None
            self.renderer_target = None
            self.scene_value = None
            self.active_showcase_id = None
            self.controller = None
            self._readback_pixels = None
            self._readback_qimage = None
            self.future = None
            self.presentation_failed = False
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
            return self._selected_extent

        def _resolution_changed(self, _index=None):
            value = self.resolution.currentData()
            if value is not None:
                self._selected_extent = tuple(int(item) for item in value)

        def _selection_changed(self, _index=None):
            item = self.feature.currentData()
            self.description.setText(item.description)
            self.shadows.setChecked(bool(item.renderer.get("shadows", True)))
            value = int(item.renderer.get("shadow_map_size", 512))
            index = self.map_size.findData(value)
            self.map_size.setCurrentIndex(max(index, 0))

        def _target_changed(self, _index=None):
            if self.scene_value is not None:
                self.restart()

        def _wait(self):
            if self.future is not None:
                future, self.future = self.future, None
                return future.result()

        def _close_renderer(self):
            """Drain and close the active renderer without stranding Qt."""
            errors = []
            try:
                self._wait()
            except Exception as error:
                errors.append(error)
            renderer, self.renderer = self.renderer, None
            if renderer is not None:
                try:
                    renderer.close()
                except Exception as error:
                    errors.append(error)
            return errors

        def restart(self):
            target_key = self.target.currentData()
            self.status.setText(f"Initializing {self.target.currentText()}…")
            try:
                previous_target = self.renderer_target
                close_errors = self._close_renderer()
                if close_errors:
                    self.status.setText(
                        f"Recovering from presentation failure: {close_errors[0]}"
                    )
                if (
                    previous_target in {"vulkan-raster", "wavefront-gi"}
                    and target_key in {"vulkan-raster", "wavefront-gi"}
                ):
                    self.surface.recreate_surface()
                item = self.feature.currentData()
                (
                    self.scene_value, self.controller, self.active_showcase_id,
                ) = _preserved_view(
                    item, self.scene_value, self.controller,
                    self.active_showcase_id,
                )
                settings = dict(item.renderer)
                if target_key == "wavefront-gi":
                    self.renderer = ol.VulkanSurfacePresenter(
                        self.surface.instance, self.surface.surface,
                        config=_gi_config(item, present=True),
                    )
                    self.viewport_stack.setCurrentWidget(self.container)
                elif target_key == "vulkan-raster":
                    default_material = (
                        settings.get("material_program") or ol.builtin_material
                    )
                    program = ol.RasterProgram.scene(
                        target="spirv", validate=False,
                        material_programs=self.scene_value.material_programs(
                            default_material,
                        ),
                        material_modifier=settings.get(
                            "material_modifier", settings.get("material_hook")
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
                        program, config=config,
                        instance=self.surface.instance,
                        surface=self.surface.surface,
                    )
                    self.viewport_stack.setCurrentWidget(self.container)
                else:
                    self.renderer = _renderer(
                        item, self.scene_value, target_key,
                        self.shadows.isChecked(),
                        int(self.map_size.currentData()),
                    )
                    self.viewport_stack.setCurrentWidget(self.readback_image)
                self.renderer_target = target_key
                self.presentation_failed = False
                self.completed.clear()
                self.status.setText(
                    f"{self.target.currentText()} ready; scene and camera retained"
                )
            except Exception as error:
                self.renderer = None
                self.renderer_target = None
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
                    result = self.future.result()
                except Exception as error:
                    detail = str(error) or repr(error)
                    self.status.setText(
                        f"Presentation failed ({type(error).__name__}): {detail}"
                    )
                    traceback.print_exception(error)
                    self.future = None
                    self.presentation_failed = True
                    return
                self.future = None
                if self.renderer_target == "webgpu-raster":
                    self._display_readback(result)
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
                path = (
                    "QImage readback" if self.renderer_target == "webgpu-raster"
                    else "direct swapchain"
                )
                self.status.setText(
                    f"{fps:.1f} FPS | frame {ms:.2f} ms | pack {pack_ms:.2f} ms "
                    f"| prepare {prepare_ms:.2f} ms "
                    f"| submit {submit_ms:.2f} ms | {width} × {height} "
                    f"| {self.target.currentText()} · {present_mode} · {path} · {resident}",
                )
            if (
                self.renderer is None or self.controller is None
                or self.presentation_failed
            ):
                return
            render_width, render_height = self.extent
            ratio = float(self.native_window.devicePixelRatio())
            surface_size = (
                max(1, round(self.native_window.width() * ratio)),
                max(1, round(self.native_window.height() * ratio)),
            )
            camera = self.controller.camera()
            if self.renderer_target == "vulkan-raster":
                raster_width, raster_height = _surface_aspect_extent(
                    (render_width, render_height), surface_size,
                )
                self.future = self.executor.submit(
                    self.renderer.present_frame,
                    self.scene_value, camera, raster_width, raster_height,
                    surface_size=surface_size,
                )
            elif self.renderer_target == "wavefront-gi":
                # A direct Vulkan swapchain must follow the native Qt client
                # extent.  The selected output extent can differ by title-bar
                # and fractional-scaling pixels when the window is maximized;
                # repeatedly passing that logical target would force GI to
                # rebuild all swapchain/history resources every frame.
                gi_width, gi_height = _direct_render_extent(
                    self.renderer_target,
                    (render_width, render_height),
                    surface_size,
                )
                self.future = self.executor.submit(
                    self.renderer.present_wavefront,
                    self.scene_value, camera, gi_width, gi_height,
                    render_extent=(render_width, render_height),
                )
            else:
                self.future = self.executor.submit(
                    self.renderer.render,
                    self.scene_value, camera, (render_width, render_height),
                )

        def _display_readback(self, image):
            pixels = to_sdr(image)
            if pixels.shape[2] == 3:
                rgba = np.full((*pixels.shape[:2], 4), 255, np.uint8)
                rgba[..., :3] = pixels
                pixels = rgba
            self._readback_pixels = np.ascontiguousarray(pixels)
            self._readback_qimage = QtGui.QImage(
                self._readback_pixels.data,
                self._readback_pixels.shape[1],
                self._readback_pixels.shape[0],
                self._readback_pixels.strides[0],
                QtGui.QImage.Format.Format_RGBA8888,
            ).copy()
            self.readback_image.setPixmap(
                QtGui.QPixmap.fromImage(self._readback_qimage).scaled(
                    self.readback_image.size(),
                    QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                    QtCore.Qt.TransformationMode.SmoothTransformation,
                )
            )

        def closeEvent(self, event):
            self.timer.stop()
            try:
                errors = self._close_renderer()
                if errors:
                    print(f"Renderer shutdown after failure: {errors[0]}")
            finally:
                self.executor.shutdown(wait=True, cancel_futures=True)
                try:
                    self.surface.close()
                except Exception as error:
                    print(f"Vulkan surface shutdown warning: {error}")
            event.accept()

    window = DirectWindow(); window.show()
    return window


def _catalog():
    path = ROOT / "ordinarylight" / "showcases" / "catalog"
    return tuple(
        item for item in discover_showcases((path,))
        if "raster-feature" in item.tags
    )


def _renderer(showcase, scene, backend_name, shadows, shadow_map_size):
    if backend_name == "wavefront-gi":
        return ol.Renderer(
            config=_gi_config(showcase), renderer_preference="gi",
        )
    target = "spirv" if backend_name == "vulkan-raster" else "wgsl"
    settings = dict(showcase.renderer)
    default_material = settings.get("material_program") or ol.builtin_material
    program = ol.RasterProgram.scene(
        target=target, validate=False,
        material_programs=scene.material_programs(default_material),
        material_modifier=settings.get(
            "material_modifier", settings.get("material_hook")
        ),
    )
    settings.update(shadows=shadows, shadow_map_size=shadow_map_size)
    config = ol.RasterConfig(
        state=ol.RasterState(cull_mode="none"),
        ambient_light=float(settings.pop("ambient_light", 0.08)),
        **settings,
    )
    implementation_type = (
        ol.renderers.raster.VulkanRasterRenderer
        if backend_name == "vulkan-raster" else
        ol.renderers.raster.WebGpuRasterRenderer
    )
    return ol.Renderer(implementation=implementation_type(program, config=config))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--target", choices=tuple(key for _title, key in TARGETS),
        default="vulkan-raster",
        help="initial rendering target",
    )
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
            for title, key in TARGETS:
                self.backend.addItem(title, (key,))
            self.backend.addItem(
                "Vulkan + WebGPU raster", ("vulkan-raster", "webgpu-raster"),
            )
            self.backend.addItem(
                "All three targets",
                ("wavefront-gi", "vulkan-raster", "webgpu-raster"),
            )
            selected_target = next(
                (
                    index for index in range(self.backend.count())
                    if self.backend.itemData(index) == (args.target,)
                ),
                0,
            )
            self.backend.setCurrentIndex(selected_target)
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
            self.backend.currentIndexChanged.connect(self._target_changed)
            form.addRow("Feature", self.feature)
            form.addRow("Rendering target", self.backend)
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
            self.active_showcase_id = None

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

        def _target_changed(self, _index=None):
            if self.scene_value is not None:
                self.restart()

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
                names = tuple(self.backend.currentData())
                (
                    self.scene_value, self.controller, self.active_showcase_id,
                ) = _preserved_view(
                    item, self.scene_value, self.controller,
                    self.active_showcase_id,
                )
                self.renderers = [
                    (name, _renderer(
                        item, self.scene_value, name, self.shadows.isChecked(),
                        int(self.map_size.currentData()),
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
