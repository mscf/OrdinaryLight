"""Canonical Qt feature workbench for Ordinary Light showcase scripts."""

from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
import math
import os
from pathlib import Path
from queue import Queue
import sys
from threading import Thread
import time

import numpy as np

import ordinarylight as ol
from ordinarylight.integrations.workbench import Showcase, discover_showcases


def _default_showcase_paths():
    """Return packaged catalog scripts; user paths are added by discovery."""
    return (Path(__file__).resolve().parents[1] / "showcases" / "catalog",)


class _SceneLoader:
    """One daemon worker for lazy scene construction.

    A running Python callable cannot be forcefully cancelled safely. Making
    this isolated loader a daemon ensures a slow third-party showcase cannot
    keep the process alive after Qt and Vulkan have shut down.
    """

    def __init__(self):
        self._commands = Queue()
        self._closed = False
        self._thread = Thread(
            target=self._run, name="ordinarylight-load", daemon=True
        )
        self._thread.start()

    def submit(self, function, *args, **kwargs):
        if self._closed:
            raise RuntimeError("scene loader is closed")
        future = Future()
        self._commands.put((future, function, args, kwargs))
        return future

    def close(self):
        if not self._closed:
            self._closed = True
            self._commands.put(None)

    def _run(self):
        while True:
            command = self._commands.get()
            if command is None:
                return
            future, function, args, kwargs = command
            if not future.set_running_or_notify_cancel():
                continue
            try:
                future.set_result(function(*args, **kwargs))
            except BaseException as error:
                future.set_exception(error)


@dataclass
class SceneEntry:
    name: str
    scene: ol.Scene | None
    source: str
    showcase: Showcase | None = None
    camera_target: tuple[float, float, float] | None = None
    orbit_radius: float | None = None
    camera_height: float | None = None
    presentation_arc_radians: float | None = None


QUALITY_PRESETS = {
    "raw": {
        "samples": 1, "scale": 1.0, "area_light_samples": 1,
        "progressive": False, "temporal": False,
        "denoiser": False, "restir": False, "restir_spatial": False,
        "advanced_temporal": False,
        "diffuse_filter": False, "diffuse_filter_strength": 0.0,
    },
    "balanced": {
        # Preserve one sample at native spatial resolution.  Two samples at
        # half scale only perform half as much primary-ray work overall and
        # magnify each noisy internal pixel over four output pixels.
        "samples": 1, "scale": 1.0, "area_light_samples": 2,
        "progressive": False, "temporal": True,
        "denoiser": False, "restir": False, "restir_spatial": False,
        "advanced_temporal": False,
        "diffuse_filter": True, "diffuse_filter_strength": 0.35,
    },
    "fast": {
        "samples": 2, "scale": 0.5, "area_light_samples": 2,
        "progressive": False, "temporal": True,
        "denoiser": False, "restir": False, "restir_spatial": False,
        "advanced_temporal": False,
        "diffuse_filter": True, "diffuse_filter_strength": 0.35,
    },
    "clean": {
        # "Clean" must not silently exchange spatial quality for throughput.
        "samples": 2, "scale": 1.0, "area_light_samples": 2,
        "progressive": False, "temporal": True,
        "denoiser": False, "restir": False, "restir_spatial": False,
        "advanced_temporal": False,
        "diffuse_filter": True, "diffuse_filter_strength": 0.35,
    },
}


class WorkbenchState:
    """Scene collection and camera fitting independent of any GUI toolkit."""

    def __init__(self):
        self.scenes: list[SceneEntry] = []
        self.active_index: int | None = None

    @property
    def active(self):
        return None if self.active_index is None else self.scenes[self.active_index]

    def add(
        self, name, scene, *, source="procedural", activate=True,
        camera_target=None, orbit_radius=None, camera_height=None,
        presentation_arc_radians=None,
    ):
        if not isinstance(scene, ol.Scene):
            raise TypeError("scene must be a ordinarylight.Scene")
        entry = SceneEntry(
            str(name), scene, str(source), None,
            None if camera_target is None else tuple(float(v) for v in camera_target),
            None if orbit_radius is None else float(orbit_radius),
            None if camera_height is None else float(camera_height),
            (
                None if presentation_arc_radians is None
                else float(presentation_arc_radians)
            ),
        )
        self.scenes.append(entry)
        if activate:
            self.active_index = len(self.scenes) - 1
        return entry

    def add_showcase(self, showcase, *, activate=False, source="showcase"):
        if not isinstance(showcase, Showcase):
            raise TypeError("showcase must be a Showcase")
        camera = showcase.camera
        entry = SceneEntry(
            showcase.title, None, str(source), showcase,
            camera.target, camera.radius, camera.height, camera.arc_radians,
        )
        self.scenes.append(entry)
        if activate:
            self.active_index = len(self.scenes) - 1
        return entry

    def remove(self, index):
        index = int(index)
        entry = self.scenes.pop(index)
        if not self.scenes:
            self.active_index = None
        elif self.active_index is None:
            pass
        elif self.active_index == index:
            self.active_index = min(index, len(self.scenes) - 1)
        elif self.active_index > index:
            self.active_index -= 1
        return entry

    def activate(self, index):
        index = int(index)
        if not 0 <= index < len(self.scenes):
            raise IndexError("scene index out of range")
        self.active_index = index
        return self.scenes[index]

    @staticmethod
    def camera_parameters(entry):
        if not isinstance(entry, SceneEntry):
            raise TypeError("entry must be a SceneEntry")
        if entry.scene is None:
            raise RuntimeError("scene has not been constructed yet")
        if entry.orbit_radius is not None:
            return (
                np.asarray(entry.camera_target, dtype=np.float64),
                entry.orbit_radius,
                entry.camera_height,
            )
        bounds_min, bounds_max = entry.scene.bounds()
        center = (np.asarray(bounds_min) + np.asarray(bounds_max)) * 0.5
        radius = max(float(np.linalg.norm(bounds_max - bounds_min)) * 0.5, 0.5)
        return center, radius * 2.2, float(center[1] + radius * 0.25)


def _qt():
    try:
        from PySide6 import QtCore, QtGui, QtWidgets
    except ImportError as error:
        raise RuntimeError(
            "The Qt workbench requires PySide6. Install it with "
            "`pip install -e '.[qt]'` or `pip install PySide6`."
        ) from error
    return QtCore, QtGui, QtWidgets


def main():
    # PySide wheels built without Vulkan headers omit QVulkanInstance. Use a
    # shared XWayland backend so Qt can embed GLFW's Vulkan-capable native child.
    os.environ["QT_QPA_PLATFORM"] = os.environ.get(
        "ORDINARYLIGHT_QT_PLATFORM",
        os.environ.get("WAVE_RENDER_QT_PLATFORM", "xcb"),
    )
    os.environ.setdefault("QT_IM_MODULE", "compose")
    os.environ["WAVE_RENDER_GLFW_PLATFORM"] = "x11"
    os.environ["PYGLFW_LIBRARY_VARIANT"] = "x11"
    QtCore, QtGui, QtWidgets = _qt()
    from ordinarylight.integrations.presentation import AsyncPresenter
    from ordinarylight.integrations.resize import ResizeRecreationGate
    from ordinarylight.integrations.glfw_platform import load_glfw

    class EmbeddedViewport:
        """Nonblocking facade over a worker-owned Vulkan presenter."""

        def __init__(self, glfw, glfw_window, qt_window, scene, camera, config):
            self.glfw = glfw
            self.glfw_window = glfw_window
            self.qt_window = qt_window
            self.scene = scene
            self.camera = camera
            self.config = config
            self.frame_index = 0
            self.last_statistics = None
            self.last_timings = {}
            self._resources_allocated = False
            self._resize_gate = ResizeRecreationGate(0.15)
            self._worker = AsyncPresenter(
                lambda worker_config: ol.VulkanGlfwPresenter(
                    glfw_window, config=worker_config
                )
            )
            self._generation = self._worker.restart(config)

        @property
        def ready(self):
            return self._worker.ready

        def reconfigure(self, config):
            self.config = config
            self.frame_index = 0
            self.last_statistics = None
            self.last_timings = {}
            self._resources_allocated = False
            self._generation = self._worker.restart(config)

        def set_scene(self, scene):
            self.scene = scene
            self.reset_sequence()

        def set_camera(self, camera):
            self.camera = camera
            self.reset_sequence()

        def reset_sequence(self):
            self.frame_index = 0
            self._worker.reset()

        def step(self):
            self.glfw.poll_events()
            completed = None
            for event in self._worker.poll():
                if event.generation != self._generation:
                    continue
                if event.kind == "error":
                    raise RuntimeError(
                        f"Vulkan presentation worker failed: {event.error}"
                    ) from event.error
                if event.kind == "frame":
                    completed = event.statistics
                    self.last_statistics = completed
                    self.last_timings = dict(event.timings or {})
                    self.frame_index = completed.frame_index + 1
                    self._resources_allocated = True
            width, height = self.glfw.get_framebuffer_size(self.glfw_window)
            # QWindow wrappers around foreign X11 children do not reliably
            # update isExposed(). The native framebuffer extent is the useful
            # source of truth for an embedded GLFW surface.
            if width < 1 or height < 1:
                return None
            if not self._resize_gate.should_render(
                (width, height), time.perf_counter(),
                resources_allocated=self._resources_allocated,
            ):
                return completed
            self._worker.request_frame(
                self.scene, self.camera, (width, height),
                frame_index=self.frame_index,
            )
            return completed

        def close(self):
            self._worker.close()

    class Workbench(QtWidgets.QMainWindow):
        def __init__(self, glfw, glfw_window, foreign_window):
            super().__init__()
            self.glfw = glfw
            self.glfw_window = glfw_window
            self.foreign_window = foreign_window
            self.setWindowTitle("Ordinary Light workbench")
            self.resize(1500, 900)
            self.state = WorkbenchState()
            self.viewport = None
            self._surface_ready = False
            self.started = time.perf_counter()
            self.orbit_phase = 0.0
            self._frame_times = deque(maxlen=60)
            self._loader = _SceneLoader()
            self._load_future = None
            self._smoke_frames = max(0, int(os.environ.get(
                "ORDINARYLIGHT_WORKBENCH_SMOKE_FRAMES", "0"
            )))
            self._showcase_paths = _default_showcase_paths()
            self._build_ui()
            self.statusBar().showMessage("Renderer stopped")
            self._populate_builtins()
            self.timer = QtCore.QTimer(self)
            self.timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
            self.timer.timeout.connect(self._frame)
            # A small nonzero interval prevents a continuously-ready timer from
            # starving ordinary Qt input and paint events between GPU frames.
            self.timer.start(1)
            QtCore.QTimer.singleShot(0, self._start_after_show)

        def _start_after_show(self):
            self._surface_ready = True
            if self.state.active is not None and self.state.active.scene is None:
                self._queue_showcase(self.state.active_index)
            else:
                self._restart_viewport()

        def _build_ui(self):
            root = QtWidgets.QWidget()
            self.setCentralWidget(root)
            root_layout = QtWidgets.QHBoxLayout(root)
            self.render_window = self.foreign_window
            self.render_container = QtWidgets.QWidget.createWindowContainer(
                self.render_window, root
            )
            self.render_container.setMinimumSize(480, 270)
            root_layout.addWidget(self.render_container, 1)

            sidebar = QtWidgets.QWidget()
            sidebar.setMinimumWidth(380)
            sidebar.setMaximumWidth(520)
            layout = QtWidgets.QVBoxLayout(sidebar)
            root_layout.addWidget(sidebar)

            scenes = QtWidgets.QGroupBox("Scenes")
            scene_layout = QtWidgets.QVBoxLayout(scenes)
            self.scene_list = QtWidgets.QListWidget()
            self.scene_list.currentRowChanged.connect(self._activate_scene)
            scene_layout.addWidget(self.scene_list)
            row = QtWidgets.QHBoxLayout()
            self.load_button = QtWidgets.QPushButton("Load glTF / GLB…")
            self.load_button.clicked.connect(self._load_scene)
            self.unload_button = QtWidgets.QPushButton("Unload")
            self.unload_button.clicked.connect(self._unload_scene)
            self.reload_button = QtWidgets.QPushButton("Reload scripts")
            self.reload_button.clicked.connect(self._reload_showcases)
            row.addWidget(self.load_button)
            row.addWidget(self.unload_button)
            row.addWidget(self.reload_button)
            scene_layout.addLayout(row)
            layout.addWidget(scenes)

            settings = QtWidgets.QGroupBox("Renderer configuration")
            form = QtWidgets.QFormLayout(settings)
            self.quality_preset = QtWidgets.QComboBox()
            self.quality_preset.addItem("Raw 1 spp", "raw")
            self.quality_preset.addItem("Native + temporal (recommended)", "balanced")
            self.quality_preset.addItem("Fast 2 spp / 0.5×", "fast")
            self.quality_preset.addItem("Clean native 2 spp", "clean")
            self.quality_preset.setCurrentIndex(1)
            self.samples = QtWidgets.QSpinBox()
            self.samples.setRange(1, 64)
            self.samples.setValue(1)
            self.bounces = QtWidgets.QSpinBox()
            self.bounces.setRange(1, 16)
            self.bounces.setValue(6)
            self.scale = QtWidgets.QDoubleSpinBox()
            self.scale.setRange(0.25, 1.0)
            self.scale.setSingleStep(0.05)
            self.scale.setValue(1.0)
            self.exposure = QtWidgets.QDoubleSpinBox()
            self.exposure.setRange(0.05, 16.0)
            self.exposure.setValue(1.0)
            self.present_mode = QtWidgets.QComboBox()
            self.present_mode.addItems(("mailbox", "fifo", "immediate"))
            form.addRow("Quality preset", self.quality_preset)
            form.addRow("Samples per pixel", self.samples)
            form.addRow("Maximum bounces", self.bounces)
            form.addRow("Internal render scale", self.scale)
            form.addRow("Exposure", self.exposure)
            form.addRow("Present mode", self.present_mode)

            self.progressive = QtWidgets.QCheckBox("Progressive accumulation")
            self.progressive.setEnabled(False)
            self.progressive.setToolTip(
                "Not currently used by the wavefront swapchain path; temporal "
                "reconstruction supplies its moving-camera history."
            )
            self.temporal = QtWidgets.QCheckBox("Temporal reconstruction")
            self.denoiser = QtWidgets.QCheckBox("Denoiser")
            self.denoiser.setEnabled(False)
            self.denoiser.setToolTip(
                "The à-trous pass is not yet connected to wavefront presentation."
            )
            self.restir = QtWidgets.QCheckBox("ReSTIR DI")
            self.restir.toggled.connect(self._restir_dependency_changed)
            self.restir_spatial = QtWidgets.QCheckBox("Spatial ReSTIR reuse")
            self.dynamic = QtWidgets.QCheckBox("Dynamic resolution")
            self.animate = QtWidgets.QCheckBox("Animate camera")
            self.animate.setChecked(True)
            for control in (
                self.progressive, self.temporal, self.denoiser, self.restir,
                self.restir_spatial, self.dynamic, self.animate,
            ):
                form.addRow(control)
            self.quality_preset.currentIndexChanged.connect(
                self._apply_quality_preset
            )
            self._apply_quality_preset()
            self.apply_button = QtWidgets.QPushButton("Apply and restart renderer")
            self.apply_button.clicked.connect(self._restart_viewport)
            form.addRow(self.apply_button)
            layout.addWidget(settings)

            controls = QtWidgets.QHBoxLayout()
            self.pause_button = QtWidgets.QPushButton("Pause")
            self.pause_button.setCheckable(True)
            self.pause_button.toggled.connect(
                lambda paused: self.pause_button.setText("Resume" if paused else "Pause")
            )
            reset = QtWidgets.QPushButton("Reset accumulation")
            reset.clicked.connect(self._reset)
            controls.addWidget(self.pause_button)
            controls.addWidget(reset)
            layout.addLayout(controls)

            self.stats = QtWidgets.QLabel("Renderer stopped")
            self.stats.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
            self.stats.setWordWrap(True)
            layout.addWidget(self.stats)
            self.log = QtWidgets.QPlainTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumBlockCount(300)
            layout.addWidget(self.log, 1)

        def _apply_quality_preset(self, _index=None):
            preset = QUALITY_PRESETS[self.quality_preset.currentData()]
            self.samples.setValue(preset["samples"])
            self.scale.setValue(preset["scale"])
            self.progressive.setChecked(preset["progressive"])
            self.temporal.setChecked(preset["temporal"])
            self.denoiser.setChecked(preset["denoiser"])
            self.restir.setChecked(preset["restir"])
            self.restir_spatial.setChecked(preset["restir_spatial"])

        def _restir_dependency_changed(self, enabled):
            if enabled:
                self.samples.setValue(1)

        def _populate_builtins(self, *, select=True):
            try:
                catalog = discover_showcases(self._showcase_paths)
            except Exception as error:
                self._message(f"Could not discover showcases: {error}")
                catalog = ()
            for showcase in catalog:
                entry = self.state.add_showcase(
                    showcase, source="showcase script", activate=False,
                )
                self.scene_list.addItem(entry.name)
            if select and self.state.scenes:
                # Start on the established quality/performance showcase so the
                # workbench opens with a known visual baseline.
                baseline_row = next(
                    (
                        index for index, entry in enumerate(self.state.scenes)
                        if entry.showcase is not None
                        and entry.showcase.id == "area-lights"
                    ),
                    0,
                )
                self.scene_list.setCurrentRow(baseline_row)

        def _reload_showcases(self):
            if self._load_future is not None:
                self._message("Wait for the current scene load to finish")
                return
            active_id = (
                self.state.active.showcase.id
                if self.state.active is not None
                and self.state.active.showcase is not None else None
            )
            retained = [
                entry for entry in self.state.scenes
                if entry.showcase is None
            ]
            self.state.scenes = retained
            self.state.active_index = None
            self.scene_list.blockSignals(True)
            self.scene_list.clear()
            for entry in retained:
                self.scene_list.addItem(entry.name)
            self.scene_list.blockSignals(False)
            self._populate_builtins(select=False)
            if active_id is not None:
                for index, entry in enumerate(self.state.scenes):
                    if entry.showcase is not None and entry.showcase.id == active_id:
                        self.scene_list.setCurrentRow(index)
                        break
            if self.scene_list.currentRow() < 0 and self.state.scenes:
                self.scene_list.setCurrentRow(0)
            self._message("Reloaded showcase scripts")

        def _config(self):
            restir = self.restir.isChecked()
            dynamic = self.dynamic.isChecked()
            preset = QUALITY_PRESETS[self.quality_preset.currentData()]
            temporal = self.temporal.isChecked() or dynamic
            advanced_temporal = preset["advanced_temporal"] and temporal
            return ol.RendererConfig(
                samples_per_pixel=self.samples.value(),
                area_light_samples=preset["area_light_samples"],
                max_bounces=self.bounces.value(),
                present_mode=self.present_mode.currentText(),
                progressive_accumulation=False,
                denoiser_enabled=False,
                wavefront_render_scale=self.scale.value(),
                wavefront_exposure=self.exposure.value(),
                wavefront_dynamic_resolution=dynamic,
                wavefront_temporal_reconstruction=temporal,
                wavefront_temporal_weight=0.93,
                wavefront_temporal_variance_confidence=advanced_temporal,
                wavefront_temporal_variance_strength=0.5,
                wavefront_temporal_material_confidence=advanced_temporal,
                wavefront_temporal_transmission_history_scale=0.5,
                wavefront_temporal_reprojection_search=advanced_temporal,
                wavefront_temporal_outlier_confidence=advanced_temporal,
                wavefront_temporal_outlier_strength=0.75,
                wavefront_restir_di=restir,
                wavefront_restir_candidates=1,
                wavefront_restir_history_limit=6,
                wavefront_restir_spatial_reuse=restir and self.restir_spatial.isChecked(),
                wavefront_restir_spatial_neighbors=4,
                wavefront_restir_spatial_radius=4,
                wavefront_diffuse_filter=preset["diffuse_filter"] and temporal,
                wavefront_diffuse_filter_strength=max(
                    preset["diffuse_filter_strength"], 0.01
                ),
                wavefront_execution_strategy="auto",
                wavefront_profiling=True,
            )

        def _camera(self, angle=0.0):
            entry = self.state.active
            center, radius, height = self.state.camera_parameters(entry)
            return ol.PerspectiveCamera(
                (float(center[0] + radius * math.sin(angle)), height,
                 float(center[2] + radius * math.cos(angle))),
                tuple(float(value) for value in center),
            )

        def _restart_viewport(self):
            self._frame_times.clear()
            if self.state.active is None or self.state.active.scene is None:
                self._close_viewport()
                return
            try:
                self.started = time.perf_counter()
                if self.viewport is None:
                    self.viewport = EmbeddedViewport(
                        self.glfw, self.glfw_window, self.render_window,
                        self.state.active.scene, self._camera(), self._config(),
                    )
                else:
                    self.viewport.scene = self.state.active.scene
                    self.viewport.camera = self._camera()
                    self.viewport.reconfigure(self._config())
                self._message(f"Started: {self.state.active.name}")
            except Exception as error:
                self.viewport = None
                self._message(f"Renderer start failed: {error}")

        def _activate_scene(self, index):
            if index < 0 or index >= len(self.state.scenes):
                return
            self.state.activate(index)
            if not self._surface_ready:
                return
            if self.state.active.scene is None:
                self._queue_showcase(index)
                return
            if self.viewport is None:
                self._restart_viewport()
            else:
                self.viewport.set_scene(self.state.active.scene)
                self.viewport.set_camera(self._camera())
                self.started = time.perf_counter()
                self._message(f"Activated: {self.state.active.name}")

        def _load_scene(self):
            if self._load_future is not None:
                return
            # Rendering runs on its own serialized worker, so this nested file
            # dialog no longer reenters Vulkan presentation on the GUI thread.
            self.render_container.hide()
            try:
                path, _ = QtWidgets.QFileDialog.getOpenFileName(
                    self, "Load scene", "", "glTF scenes (*.glb *.gltf)",
                    options=QtWidgets.QFileDialog.Option.DontUseNativeDialog,
                )
            finally:
                self.render_container.show()
            if not path:
                return
            self.load_button.setEnabled(False)
            self._message(f"Loading {path}…")
            self._load_future = (
                "gltf", path, self._loader.submit(ol.loaders.load, path)
            )

        def _queue_showcase(self, index):
            if self._load_future is not None:
                self._message("Another scene is already loading")
                return
            if index is None or not 0 <= index < len(self.state.scenes):
                return
            entry = self.state.scenes[index]
            if entry.scene is not None:
                self._restart_viewport()
                return
            self.scene_list.setEnabled(False)
            self.load_button.setEnabled(False)
            self._message(f"Building showcase: {entry.name}…")
            self._load_future = (
                "showcase", index,
                self._loader.submit(entry.showcase.create_scene),
            )

        def _poll_scene_load(self):
            if self._load_future is None:
                return
            kind, subject, future = self._load_future
            if not future.done():
                return
            self._load_future = None
            self.load_button.setEnabled(True)
            self.scene_list.setEnabled(True)
            try:
                scene = future.result()
                if kind == "gltf":
                    path = subject
                    entry = self.state.add(Path(path).name, scene, source=path)
                    self.scene_list.addItem(entry.name)
                    self.scene_list.setCurrentRow(len(self.state.scenes) - 1)
                else:
                    index = subject
                    if not 0 <= index < len(self.state.scenes):
                        return
                    entry = self.state.scenes[index]
                    entry.scene = scene
                    if self.state.active_index == index:
                        self._restart_viewport()
                triangles = sum(len(mesh.indices) for mesh in scene.meshes)
                self._message(
                    f"Loaded {entry.name}: {len(scene.meshes)} meshes, "
                    f"{triangles:,} triangles"
                )
            except Exception as error:
                self._message(f"Scene load failed: {error}")

        def _unload_scene(self):
            row = self.scene_list.currentRow()
            if row < 0:
                return
            self.state.remove(row)
            self.scene_list.takeItem(row)
            if self.state.active_index is None:
                self._close_viewport()
                self.stats.setText("No scene loaded")
            else:
                self.scene_list.setCurrentRow(self.state.active_index)

        def _reset(self):
            if self.viewport is not None:
                self.viewport.reset_sequence()

        def _frame(self):
            self._poll_scene_load()
            if self.viewport is None or self.pause_button.isChecked():
                return
            try:
                if self.animate.isChecked():
                    phase = (
                        (time.perf_counter() - self.started) * 0.22
                        + self.orbit_phase
                    )
                    arc = self.state.active.presentation_arc_radians
                    angle = arc * math.sin(phase) if arc is not None else phase
                    self.viewport.camera = self._camera(angle)
                statistics = self.viewport.step()
                if statistics is None:
                    return
                now = time.perf_counter()
                self._frame_times.append(now)
                cadence_fps = None
                if len(self._frame_times) >= 2:
                    elapsed = self._frame_times[-1] - self._frame_times[0]
                    if elapsed > 0.0:
                        cadence_fps = (len(self._frame_times) - 1) / elapsed
                gpu = statistics.gpu_ms
                self.stats.setText(
                    f"{statistics.width} × {statistics.height} | frame {statistics.frame_index:,} | "
                    f"{cadence_fps:.1f} FPS | GPU {gpu:.2f} ms"
                    if cadence_fps is not None and gpu is not None else
                    f"{statistics.width} × {statistics.height} | frame {statistics.frame_index:,}"
                )
                fps_text = (
                    f"{cadence_fps:.1f} FPS" if cadence_fps is not None
                    else "Measuring FPS…"
                )
                gpu_text = f" | GPU {gpu:.2f} ms" if gpu is not None else ""
                self.statusBar().showMessage(
                    f"{fps_text}{gpu_text} | "
                    f"{statistics.width} × {statistics.height} | internal "
                    f"{self.viewport.last_timings.get('wavefront_render_extent', '?')}"
                )
                if (
                    self._smoke_frames
                    and statistics.frame_index + 1 >= self._smoke_frames
                ):
                    QtCore.QTimer.singleShot(0, self.close)
            except Exception as error:
                self._message(f"Render failed: {error}")
                self._close_viewport()

        def _message(self, text):
            text = str(text)
            self.log.appendPlainText(text)
            print(text, flush=True)

        def _close_viewport(self):
            if self.viewport is not None:
                self.viewport.close()
                self.viewport = None

        def closeEvent(self, event):
            self.timer.stop()
            self._close_viewport()
            if self._load_future is not None:
                self._load_future[2].cancel()
            self._loader.close()
            event.accept()

    glfw = load_glfw(default_linux="x11")
    if not glfw.init():
        raise RuntimeError("GLFW X11 initialization failed")
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.DECORATED, glfw.FALSE)
    glfw_window = glfw.create_window(1280, 720, "Ordinary Light viewport", None, None)
    if not glfw_window:
        glfw.terminate()
        raise RuntimeError("GLFW Vulkan child-window creation failed")

    app = QtWidgets.QApplication(sys.argv)
    foreign_window = QtGui.QWindow.fromWinId(glfw.get_x11_window(glfw_window))
    window = Workbench(glfw, glfw_window, foreign_window)
    window.show()
    try:
        return app.exec()
    finally:
        window._close_viewport()
        foreign_window.setParent(None)
        glfw.destroy_window(glfw_window)
        glfw.terminate()


if __name__ == "__main__":
    raise SystemExit(main())
