"""Live Qt catalog for comparing Ordinary Light rendering targets."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
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
from ordinarylight.denoising.reference import NrdRelaxReference
from ordinarylight.integrations.workbench import discover_showcases
from ordinarylight.outputs import to_sdr
from ordinarylight.renderers.raster._diagnostics import frame_difference


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

DENOISERS = (
    ("Ordinary Shade ReLAX (live)", "ordinaryshade"),
    ("NVIDIA NRD ReLAX (reference capture)", "nrd-reference"),
)


class _NrdReferencePreview:
    """Cold, readback-oriented NRD preview for honest visual A/B testing.

    NRD currently owns a separate Vulkan device, so this path deliberately
    captures canonical signals and runs the optional reference bridge instead
    of pretending to be the production zero-copy denoiser.  Results are cached
    for a stable scene/camera/extent; moving the camera produces a new capture.
    """

    def __init__(self, config, *, renderer=None, sequence_frames=4):
        self._renderer = renderer
        if self._renderer is None:
            self._renderer = ol.renderers.gi.VulkanGlobalIlluminationRenderer(
                config=config,
            )
        self._reference = NrdRelaxReference()
        if not self._reference.available:
            self._renderer.close()
            raise RuntimeError(
                "NRD reference bridge is unavailable; run "
                "`python tools/nrd_reference/bootstrap.py`"
            )
        self._sequence_frames = int(sequence_frames)
        self._frame_index = 0
        self._cache_key = None
        self._cached_image = None
        self.last_timings = {}

    @staticmethod
    def _key(scene, camera, width, height):
        values = np.concatenate((
            np.asarray(camera.position, np.float64),
            np.asarray(camera.target, np.float64),
            np.asarray(camera.up, np.float64),
            np.asarray((camera.vertical_fov_degrees, width, height), np.float64),
        ))
        return id(scene), values.tobytes()

    def render(self, scene, camera, size):
        width, height = map(int, size)
        key = self._key(scene, camera, width, height)
        if key == self._cache_key and self._cached_image is not None:
            self.last_timings = dict(self.last_timings, cache_hit=True)
            return self._cached_image
        started = time.perf_counter()
        signals = []
        # A canonical signal frame retains roughly 64 bytes/pixel before the
        # bridge's packed upload. Avoid turning a 4K visual comparison into a
        # multi-gigabyte host allocation; lower resolutions still exercise a
        # useful temporal sequence.
        memory_budget = 384 * 1024 * 1024
        sequence_frames = min(
            self._sequence_frames,
            max(1, memory_budget // max(1, width * height * 64)),
        )
        for _ in range(sequence_frames):
            signals.append(self._renderer.capture_denoiser_signals(
                scene, camera, width, height, frame_index=self._frame_index,
            ))
            self._frame_index += 1
        captured = time.perf_counter()
        result = self._reference.denoise_sequence(signals)[-1]
        finished = time.perf_counter()
        self._cached_image = np.ascontiguousarray(result.combined, np.float32)
        self._cache_key = key
        self.last_timings = {
            "total_ms": (finished - started) * 1000.0,
            "signal_capture_ms": (captured - started) * 1000.0,
            "reference_ms": (finished - captured) * 1000.0,
            "sequence_frames": sequence_frames,
            "cache_hit": False,
            "implementation": result.implementation_version,
        }
        return self._cached_image

    def close(self):
        self._renderer.close()


def _camera_pose_from_json(text):
    """Parse a viewer camera-pose payload into validated values."""
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid camera-pose JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError("camera-pose JSON must contain an object")
    showcase_id = payload.get("showcase")
    if not isinstance(showcase_id, str) or not showcase_id:
        raise ValueError("camera pose requires a non-empty 'showcase' string")
    try:
        camera = ol.PerspectiveCamera(
            position=tuple(payload["position"]),
            target=tuple(payload["target"]),
            up=tuple(payload.get("up", (0.0, 1.0, 0.0))),
            vertical_fov_degrees=float(
                payload.get("vertical_fov_degrees", 45.0)
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid camera pose: {error}") from error
    if not np.allclose(camera.up, (0.0, 1.0, 0.0), atol=1.0e-7):
        raise ValueError(
            "the arcball viewer currently supports only an up vector of [0,1,0]"
        )
    return showcase_id, camera


def _camera_pose_argument(value):
    """Load a camera pose from inline JSON or a JSON text file."""
    value = str(value).strip()
    if value.startswith("{"):
        text = value
    else:
        path = Path(value).expanduser()
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as error:
            raise ValueError(
                f"could not read camera-pose file {path}: {error}"
            ) from error
    return _camera_pose_from_json(text)


def _gi_config(
    showcase, *, present=False, capture=False, restir_reservoirs=4,
    denoiser_enabled=True, denoiser_iterations=3,
):
    """Build the interactive GI configuration corresponding to a showcase."""
    settings = dict(showcase.renderer)
    return ol.RendererConfig(
        samples_per_pixel=1,

        wavefront_restir_di=True,
        wavefront_restir_reservoirs=int(restir_reservoirs),
        wavefront_restir_candidates=4,
        wavefront_restir_history_limit=4,
        wavefront_restir_spatial_reuse=False,
        # # Temporal reuse
        # wavefront_restir_history_limit=4,
        # wavefront_restir_history_motion_pixels=16.0,

        # # Spatial reuse
        # wavefront_restir_spatial_reuse=True,
        # wavefront_restir_spatial_neighbors=4,
        # wavefront_restir_spatial_radius=4,

        # # Higher-quality neighbor weighting
        # wavefront_restir_pairwise_mis=True,
        # wavefront_restir_generalized_mis=True,
        # wavefront_restir_generalized_balance_cap=8.0,

        # # Keep area and environment lighting in separate reservoirs
        # wavefront_stratified_primary_restir=True,
        # wavefront_unified_primary_restir=False,

        # wavefront_restir_di=True,
        # wavefront_restir_candidates=4,
        # wavefront_restir_history_limit=4,
        # wavefront_restir_spatial_reuse=False,

        max_bounces=int(settings.get("max_bounces", 8)),
        present_mode="mailbox",
        # The Ordinary Shade ReLAX stage owns a recurrent temporal history.
        # Progressive history supplies the previous-frame guides while
        # temporal_history permits camera reprojection instead of resetting on
        # every arcball update.
        progressive_accumulation=bool(denoiser_enabled),
        temporal_history=bool(denoiser_enabled),
        temporal_history_limit=32,
        denoiser_enabled=bool(denoiser_enabled),
        denoiser_iterations=int(denoiser_iterations),
        material_program=settings.get("material_program"),
        material_modifier=settings.get(
            "material_modifier", settings.get("material_hook")
        ),
        direct_swapchain_storage=bool(present),
        wavefront_hdr_capture=bool(capture),
    )


def _gi_temporal_variance_report(images):
    """Summarize stationary GI variation at lags one and two.

    A renderer alternating between two independent history chains has much
    larger n-vs-(n-1) error than n-vs-(n-2) error.  Keeping this calculation
    independent of Vulkan also makes the diagnostic itself unit-testable.
    """
    frames = [np.asarray(image, dtype=np.float32)[..., :3] for image in images]
    if len(frames) < 3:
        raise ValueError("GI temporal analysis requires at least three frames")
    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        raise ValueError("all GI diagnostic frames must have the same shape")

    def differences(lag):
        result = []
        for index in range(lag, len(frames)):
            delta = frames[index] - frames[index - lag]
            result.append({
                "frame": index,
                "previous_frame": index - lag,
                "rmse": float(np.sqrt(np.mean(delta * delta, dtype=np.float64))),
                "mean_absolute_error": float(
                    np.mean(np.abs(delta), dtype=np.float64)
                ),
                "maximum_absolute_error": float(np.max(np.abs(delta))),
            })
        return result

    adjacent = differences(1)
    lag_two = differences(2)

    def mean_metric(rows, key):
        return float(np.mean([row[key] for row in rows], dtype=np.float64))

    adjacent_rmse = mean_metric(adjacent, "rmse")
    lag_two_rmse = mean_metric(lag_two, "rmse")
    ratio = lag_two_rmse / adjacent_rmse if adjacent_rmse > 0.0 else 0.0
    even_mean = np.mean(frames[0::2], axis=0, dtype=np.float64)
    odd_mean = np.mean(frames[1::2], axis=0, dtype=np.float64)
    parity_delta = even_mean - odd_mean
    luminance = [
        frame[..., 0] * 0.2126
        + frame[..., 1] * 0.7152
        + frame[..., 2] * 0.0722
        for frame in frames
    ]
    return {
        "frame_count": len(frames),
        "shape": list(shape),
        "frame_hashes": [
            hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()[:16]
            for frame in frames
        ],
        "adjacent": adjacent,
        "lag_two": lag_two,
        "mean_adjacent_rmse": adjacent_rmse,
        "mean_lag_two_rmse": lag_two_rmse,
        "lag_two_to_adjacent_rmse_ratio": ratio,
        "even_odd_mean_rmse": float(
            np.sqrt(np.mean(parity_delta * parity_delta, dtype=np.float64))
        ),
        "mean_luminance": float(np.mean(luminance, dtype=np.float64)),
        "nonzero_luminance_fraction": float(np.mean(
            np.stack(luminance) > 1.0e-6, dtype=np.float64,
        )),
        "alternating_history_signature": bool(
            adjacent_rmse > 0.0 and ratio < 0.5
        ),
    }


def _preserved_view(showcase, scene, controller, active_showcase_id):
    """Retain scene/camera state unless the selected showcase changed."""
    if scene is not None and active_showcase_id == showcase.id:
        return scene, controller, active_showcase_id
    scene = showcase.create_scene()
    controller = ol.ArcballCameraController.from_camera(
        showcase.camera.camera(scene, angle=-0.45),
    )
    return scene, controller, showcase.id


def _set_optional_scene_lights(scene, enabled):
    """Enable or mute analytic lights explicitly marked optional by a scene."""
    for entry in scene.metadata.get("optional_scene_lights", ()):
        light = scene.get_light(int(entry["id"]))
        intensity = float(entry["intensity"]) if enabled else 0.0
        if isinstance(light, ol.PointLight):
            scene.update_point_light(light, intensity=intensity)
        elif isinstance(light, ol.DirectionalLight):
            scene.update_directional_light(light, intensity=intensity)
        elif isinstance(light, ol.SpotLight):
            scene.update_spot_light(light, intensity=intensity)


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
            if args.diagnostic_camera_pose is not None:
                self.container.setFixedSize(args.width, args.height)
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
            startup_showcase = (
                args.diagnostic_camera_pose[0]
                if args.diagnostic_camera_pose is not None else args.showcase
            )
            if startup_showcase is not None:
                startup_index = next((
                    index for index in range(self.feature.count())
                    if self.feature.itemData(index).id == startup_showcase
                ), -1)
                if startup_index < 0:
                    raise ValueError(f"unknown showcase {startup_showcase!r}")
                self.feature.setCurrentIndex(startup_index)
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
            self.scene_lights = QtWidgets.QCheckBox()
            self.scene_lights.setChecked(True)
            self.scene_lights.toggled.connect(self._scene_light_changed)
            self.map_size = QtWidgets.QComboBox()
            for size in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
                self.map_size.addItem(str(size), size)
            self.restir_reservoirs = QtWidgets.QComboBox()
            for count in (1, 2, 4, 8):
                self.restir_reservoirs.addItem(str(count), count)
            self.restir_reservoirs.setCurrentIndex(
                self.restir_reservoirs.findData(4)
            )
            self.restir_reservoirs.setEnabled(
                self.target.currentData() == "wavefront-gi"
            )
            self.denoiser = QtWidgets.QCheckBox()
            self.denoiser.setChecked(True)
            self.denoiser_backend = QtWidgets.QComboBox()
            for title, key in DENOISERS:
                self.denoiser_backend.addItem(title, key)
            self.denoiser_iterations = QtWidgets.QComboBox()
            for count in range(1, 6):
                self.denoiser_iterations.addItem(str(count), count)
            self.denoiser_iterations.setCurrentIndex(
                self.denoiser_iterations.findData(3)
            )
            gi_selected = self.target.currentData() == "wavefront-gi"
            self.denoiser.setEnabled(gi_selected)
            self.denoiser_backend.setEnabled(gi_selected)
            self.denoiser_iterations.setEnabled(gi_selected)
            self.animate = QtWidgets.QCheckBox()
            self.animate.setChecked(args.diagnostic_camera_pose is None)
            self.slow_diagnostic = QtWidgets.QCheckBox()
            self.slow_diagnostic.setChecked(
                args.diagnostic_camera_pose is not None
            )
            self.description = QtWidgets.QLabel(wordWrap=True)
            self.help = QtWidgets.QLabel(
                "Vulkan targets use the direct swapchain; WebGPU currently "
                "uses offscreen QImage readback.\n"
                "Left drag: orbit · Right/middle drag: pan · Wheel: dolly",
                wordWrap=True,
            )
            button = QtWidgets.QPushButton("Apply and restart renderer")
            button.clicked.connect(self.restart)
            copy_pose = QtWidgets.QPushButton("Copy camera pose")
            copy_pose.clicked.connect(self._copy_camera_pose)
            paste_pose = QtWidgets.QPushButton("Paste camera pose")
            paste_pose.clicked.connect(self._paste_camera_pose)
            copy_diagnostics = QtWidgets.QPushButton("Copy live diagnostics")
            copy_diagnostics.clicked.connect(self._copy_live_diagnostics)
            self.feature.currentIndexChanged.connect(self._selection_changed)
            self.target.currentIndexChanged.connect(self._target_changed)
            form.addRow("Feature", self.feature)
            form.addRow("Rendering target", self.target)
            form.addRow("Render resolution", self.resolution)
            form.addRow("Enable shadows", self.shadows)
            form.addRow("Enable optional scene light", self.scene_lights)
            form.addRow("Shadow map size", self.map_size)
            form.addRow("ReSTIR reservoirs", self.restir_reservoirs)
            form.addRow("Enable GI denoising", self.denoiser)
            form.addRow("GI denoiser implementation", self.denoiser_backend)
            form.addRow("ReLAX A-trous iterations", self.denoiser_iterations)
            form.addRow("Animate scene / camera", self.animate)
            form.addRow("Slow swapchain diagnostic (2 FPS)", self.slow_diagnostic)
            form.addRow(self.description); form.addRow(self.help)
            form.addRow(button)
            form.addRow(copy_pose)
            form.addRow(paste_pose)
            form.addRow(copy_diagnostics)
            self.status = QtWidgets.QLabel(wordWrap=True); form.addRow(self.status)
            self.renderer = None
            self.renderer_target = None
            # Track the mode installed on ``renderer`` independently of the
            # controls.  A combo-box change can precede the deferred renderer
            # restart by one or more frames.
            self.renderer_denoiser_backend = None
            self.scene_value = None
            self.active_showcase_id = None
            self.controller = None
            self._startup_camera = (
                args.diagnostic_camera_pose[1]
                if args.diagnostic_camera_pose is not None else None
            )
            self._readback_pixels = None
            self._readback_qimage = None
            self.future = None
            self.restart_pending = False
            self.presentation_failed = False
            self.executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="ordinarylight-qt-vulkan",
            )
            self.completed = deque(maxlen=240)
            self.completed_frame_count = 0
            self.automatic_switch_requested = False
            self.diagnostic_frames = deque(maxlen=240)
            self.gi_diagnostic_images = []
            self.gi_diagnostic_metadata = []
            self.gi_diagnostic_warmup = 0
            self._diagnostic_exit_requested = False
            self.last_tick = time.perf_counter()
            self.last_submission = 0.0
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

        def _copy_camera_pose(self):
            if self.controller is None:
                return
            camera = self.controller.camera()
            showcase = self.feature.currentData()
            payload = {
                "showcase": showcase.id,
                "position": list(camera.position),
                "target": list(camera.target),
                "up": list(camera.up),
                "vertical_fov_degrees": camera.vertical_fov_degrees,
            }
            text = json.dumps(payload, separators=(",", ":"))
            QtWidgets.QApplication.clipboard().setText(text)
            self.status.setText(f"Camera pose copied: {text}")

        def _paste_camera_pose(self):
            clipboard_text = QtWidgets.QApplication.clipboard().text().strip()
            text, accepted = QtWidgets.QInputDialog.getMultiLineText(
                self,
                "Paste camera pose",
                "Camera-pose JSON:",
                clipboard_text,
            )
            if not accepted:
                return
            try:
                showcase_id, camera = _camera_pose_from_json(text.strip())
                showcase_index = next(
                    (
                        index for index in range(self.feature.count())
                        if self.feature.itemData(index).id == showcase_id
                    ),
                    -1,
                )
                if showcase_index < 0:
                    raise ValueError(f"unknown showcase {showcase_id!r}")
                showcase_changed = showcase_index != self.feature.currentIndex()
                if showcase_changed:
                    self.feature.blockSignals(True)
                    self.feature.setCurrentIndex(showcase_index)
                    self.feature.blockSignals(False)
                    self._selection_changed()
                    self.scene_value = None
                    self.active_showcase_id = None
                item = self.feature.currentData()
                self.scene_value, _, self.active_showcase_id = _preserved_view(
                    item, self.scene_value, self.controller,
                    self.active_showcase_id,
                )
                self.controller = ol.ArcballCameraController.from_camera(camera)
                # Resident raster commands contain camera-dependent transparent
                # draw ordering. Restarting is required only when the showcase
                # changes; an in-place pose update is otherwise sufficient.
                if showcase_changed:
                    self.restart()
                self.status.setText(
                    f"Camera pose applied: {showcase_id} at {camera.position}"
                )
            except ValueError as error:
                self.status.setText(f"Camera pose rejected: {error}")
                QtWidgets.QMessageBox.warning(
                    self, "Invalid camera pose", str(error),
                )

        def _copy_live_diagnostics(self):
            payload = {
                "showcase": self.feature.currentData().id,
                "target": self.renderer_target,
                "frames": list(self.diagnostic_frames),
            }
            text = json.dumps(payload, separators=(",", ":"))
            QtWidgets.QApplication.clipboard().setText(text)
            self.status.setText(
                f"Copied {len(self.diagnostic_frames)} live diagnostic frames"
            )

        def _selection_changed(self, _index=None):
            item = self.feature.currentData()
            self.description.setText(item.description)
            self.shadows.setChecked(bool(item.renderer.get("shadows", True)))
            supports_light_toggle = bool(
                item.renderer.get("scene_light_toggle", False)
            )
            self.scene_lights.setEnabled(supports_light_toggle)
            self.scene_lights.setChecked(True)
            value = int(item.renderer.get("shadow_map_size", 512))
            index = self.map_size.findData(value)
            self.map_size.setCurrentIndex(max(index, 0))

        def _target_changed(self, _index=None):
            gi_selected = self.target.currentData() == "wavefront-gi"
            self.restir_reservoirs.setEnabled(gi_selected)
            self.denoiser.setEnabled(gi_selected)
            self.denoiser_backend.setEnabled(gi_selected)
            self.denoiser_iterations.setEnabled(gi_selected)
            if self.scene_value is not None:
                self.restart_pending = True
                self.status.setText(
                    "Finishing the active frame before switching targets…"
                )
                if self.future is None:
                    QtCore.QTimer.singleShot(0, self._finish_pending_restart)

        def _scene_light_changed(self, _enabled=None):
            if self.scene_value is None:
                return
            self.restart_pending = True
            self.status.setText(
                "Finishing the active frame before updating scene lighting…"
            )
            if self.future is None:
                QtCore.QTimer.singleShot(0, self._finish_pending_restart)

        def _finish_pending_restart(self):
            if not self.restart_pending:
                return
            if self.future is not None:
                return
            self.restart_pending = False
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
            self.renderer_denoiser_backend = None
            if renderer is not None:
                try:
                    renderer.close()
                except Exception as error:
                    errors.append(error)
            return errors

        def restart(self):
            self.restart_pending = False
            target_key = self.target.currentData()
            requested_denoiser_backend = (
                self.denoiser_backend.currentData()
                if target_key == "wavefront-gi" and self.denoiser.isChecked()
                else None
            )
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
                _set_optional_scene_lights(
                    self.scene_value, self.scene_lights.isChecked(),
                )
                if self._startup_camera is not None:
                    self.controller = ol.ArcballCameraController.from_camera(
                        self._startup_camera,
                    )
                    self._startup_camera = None
                settings = dict(item.renderer)
                if target_key == "wavefront-gi":
                    nrd_reference = (
                        requested_denoiser_backend == "nrd-reference"
                    )
                    gi_config = _gi_config(
                        item, present=not nrd_reference,
                        capture=args.diagnostic_frames > 0,
                        restir_reservoirs=int(
                            self.restir_reservoirs.currentData()
                        ),
                        denoiser_enabled=(
                            self.denoiser.isChecked() and not nrd_reference
                        ),
                        denoiser_iterations=int(
                            self.denoiser_iterations.currentData()
                        ),
                    )
                    if nrd_reference:
                        gi_config = replace(
                            gi_config,
                            progressive_accumulation=False,
                            temporal_history=False,
                            denoiser_enabled=False,
                            denoiser_signal_capture=True,
                            direct_swapchain_storage=False,
                        )
                        capture_renderer = ol.VulkanSurfacePresenter(
                            self.surface.instance, self.surface.surface,
                            config=gi_config,
                        )
                        self.renderer = _NrdReferencePreview(
                            gi_config, renderer=capture_renderer,
                        )
                        self.viewport_stack.setCurrentWidget(
                            self.readback_image
                        )
                    else:
                        self.renderer = ol.VulkanSurfacePresenter(
                            self.surface.instance, self.surface.surface,
                            config=gi_config,
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
                    settings.pop("scene_light_toggle", None)
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
                self.renderer_denoiser_backend = requested_denoiser_backend
                self.presentation_failed = False
                self.completed.clear()
                self.diagnostic_frames.clear()
                self.gi_diagnostic_images.clear()
                self.gi_diagnostic_metadata.clear()
                self.gi_diagnostic_warmup = 0
                self.status.setText(
                    f"{self.target.currentText()} ready; scene and camera retained"
                )
            except Exception as error:
                self.renderer = None
                self.renderer_target = None
                self.renderer_denoiser_backend = None
                self.status.setText(f"Renderer start failed: {error}")

        def tick(self):
            now = time.perf_counter()
            elapsed = min(now - self.last_tick, 0.1)
            self.last_tick = now
            item = self.feature.currentData()
            if (
                self.animate.isChecked() and self.controller is not None
                and item.animate is None
            ):
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
                nrd_reference = (
                    self.renderer_target == "wavefront-gi"
                    and self.renderer_denoiser_backend == "nrd-reference"
                )
                if self.renderer_target == "webgpu-raster" or nrd_reference:
                    self._display_readback(result)
                complete = time.perf_counter()
                self.completed.append(complete)
                self.completed_frame_count += 1
                while self.completed and complete - self.completed[0] > 1.0:
                    self.completed.popleft()
                fps = 0.0
                if len(self.completed) > 1:
                    fps = (len(self.completed) - 1) / (
                        self.completed[-1] - self.completed[0]
                    )
                width, height = self.extent
                timings = self.renderer.last_timings
                if (
                    self.renderer_target == "wavefront-gi"
                    and not nrd_reference
                    and args.diagnostic_frames > 0
                    and not self._diagnostic_exit_requested
                ):
                    if self.gi_diagnostic_warmup < args.diagnostic_warmup_frames:
                        self.gi_diagnostic_warmup += 1
                    else:
                        image = self.renderer.capture_wavefront_hdr()
                        self.gi_diagnostic_images.append(image)
                        self.gi_diagnostic_metadata.append({
                            key: timings.get(key) for key in (
                                "wavefront_frame_slot",
                                "wavefront_history_source_slot",
                                "wavefront_history_dependency_waited",
                                "wavefront_history_chain_enabled",
                                "swapchain_image_index",
                            ) if timings.get(key) is not None
                        })
                        if len(self.gi_diagnostic_images) >= args.diagnostic_frames:
                            report = {
                                "showcase": self.feature.currentData().id,
                                "target": self.renderer_target,
                                "extent": list(self.extent),
                                "warmup_frames": self.gi_diagnostic_warmup,
                                "frame_metadata": self.gi_diagnostic_metadata,
                                "summary": _gi_temporal_variance_report(
                                    self.gi_diagnostic_images,
                                ),
                            }
                            report_path = Path(args.diagnostic_report)
                            report_path.parent.mkdir(parents=True, exist_ok=True)
                            report_path.write_text(
                                json.dumps(report, indent=2) + "\n",
                                encoding="utf-8",
                            )
                            self._diagnostic_exit_requested = True
                            print(
                                f"wrote GI temporal diagnostic report: {report_path}",
                                flush=True,
                            )
                            QtCore.QTimer.singleShot(0, self.close)
                ms = timings.get("total_ms", 0.0)
                pack_ms = timings.get("scene_pack_ms", 0.0)
                prepare_ms = timings.get("scene_prepare_ms", 0.0)
                submit_ms = timings.get("resident_submit_ms", 0.0)
                swapchain_index = timings.get("swapchain_image_index")
                resident = "resident" if timings.get("resident_cache_hit") else "warming"
                present_mode = timings.get("present_mode", "unknown").upper()
                diagnostic = {
                    key: timings.get(key) for key in (
                        "present_submission", "present_frame_slot",
                        "swapchain_image_index", "camera_uniform_hash",
                        "resident_cache_key_hash", "resident_generation_hash",
                        "resident_cache_slot", "render_finished_slot",
                        "resident_cache_hit", "present_hdr_hash",
                        "present_captured_submission",
                        "present_hdr_max_difference", "present_hdr_rmse",
                        "present_hdr_changed_pixels", "present_hdr_changed_bounds",
                        "present_opaque_hdr_hash",
                        "present_opaque_hdr_max_difference",
                        "present_opaque_hdr_rmse",
                        "present_opaque_hdr_changed_pixels",
                        "present_opaque_hdr_changed_bounds",
                        "present_depth_hash", "present_depth_max_difference",
                        "present_depth_rmse", "present_depth_changed_pixels",
                        "present_depth_changed_bounds",
                    ) if timings.get(key) is not None
                }
                if diagnostic and self.renderer_target == "vulkan-raster":
                    self.diagnostic_frames.append(diagnostic)
                    if self.slow_diagnostic.isChecked():
                        print(
                            "raster_present_diagnostic "
                            + json.dumps(diagnostic, sort_keys=True),
                            flush=True,
                        )
                    captured = [
                        frame for frame in self.diagnostic_frames
                        if frame.get("present_hdr_hash") is not None
                    ]
                    if (
                        args.diagnostic_frames > 0
                        and len(captured) >= args.diagnostic_frames
                        and not self._diagnostic_exit_requested
                    ):
                        selected = captured[-args.diagnostic_frames:]
                        report = {
                            "showcase": self.feature.currentData().id,
                            "target": self.renderer_target,
                            "extent": list(self.extent),
                            "frames": selected,
                            "summary": {
                                "unique_hdr_hashes": len({
                                    frame["present_hdr_hash"]
                                    for frame in selected
                                }),
                                "maximum_absolute_difference": max(
                                    frame.get(
                                        "present_hdr_max_difference", 0.0
                                    ) for frame in selected
                                ),
                                "maximum_rmse": max(
                                    frame.get("present_hdr_rmse", 0.0)
                                    for frame in selected
                                ),
                                "maximum_changed_pixels": max(
                                    frame.get(
                                        "present_hdr_changed_pixels", 0
                                    ) for frame in selected
                                ),
                            },
                        }
                        report_path = Path(args.diagnostic_report)
                        report_path.parent.mkdir(parents=True, exist_ok=True)
                        report_path.write_text(
                            json.dumps(report, indent=2) + "\n",
                            encoding="utf-8",
                        )
                        self._diagnostic_exit_requested = True
                        print(
                            f"wrote raster diagnostic report: {report_path}",
                            flush=True,
                        )
                        QtCore.QTimer.singleShot(0, self.close)
                path = (
                    "NRD reference readback"
                    if nrd_reference else
                    "QImage readback"
                    if self.renderer_target == "webgpu-raster" else
                    "direct swapchain"
                )
                self.status.setText(
                    f"{fps:.1f} FPS | frame {ms:.2f} ms | pack {pack_ms:.2f} ms "
                    f"| prepare {prepare_ms:.2f} ms "
                    f"| submit {submit_ms:.2f} ms | {width} × {height} "
                    f"| {self.target.currentText()} · {present_mode} · {path} · {resident}"
                    + (
                        f" · image {swapchain_index}"
                        if swapchain_index is not None else ""
                    )
                    + (
                        "\ninput "
                        f"{diagnostic.get('camera_uniform_hash')} · cache "
                        f"{diagnostic.get('resident_cache_key_hash')} · slot "
                        f"{diagnostic.get('resident_cache_slot')} · frame-slot "
                        f"{diagnostic.get('present_frame_slot')}"
                        if diagnostic else ""
                    )
                    + (
                        "\npresented HDR "
                        f"{diagnostic.get('present_hdr_hash')} · max "
                        f"{diagnostic.get('present_hdr_max_difference', 0.0):.6g}"
                        " · RMSE "
                        f"{diagnostic.get('present_hdr_rmse', 0.0):.6g}"
                        " · pixels "
                        f"{diagnostic.get('present_hdr_changed_pixels', 0)}"
                        if diagnostic.get("present_hdr_hash") else ""
                    )
                    + (
                        "\nGI denoiser: "
                        + (
                            "NVIDIA NRD ReLAX reference capture"
                            if nrd_reference else
                            f"Ordinary Shade ReLAX · "
                            f"{self.denoiser_iterations.currentData()} "
                            "A-trous iteration(s)"
                            if self.denoiser.isChecked()
                            else "disabled (raw GI)"
                        )
                        if self.renderer_target == "wavefront-gi" else ""
                    ),
                )
                if nrd_reference:
                    self.status.setText(
                        self.status.text()
                        + "\nNRD reference: "
                        f"{timings.get('sequence_frames', 0)} signal frame(s) · "
                        f"capture {timings.get('signal_capture_ms', 0.0):.1f} ms · "
                        f"NRD bridge {timings.get('reference_ms', 0.0):.1f} ms"
                        + (" · cached" if timings.get("cache_hit") else "")
                    )
                if (
                    args.switch_target_after_frames > 0
                    and not self.automatic_switch_requested
                    and self.completed_frame_count
                    >= args.switch_target_after_frames
                ):
                    self.automatic_switch_requested = True
                    target_index = self.target.findData(args.switch_target_to)
                    QtCore.QTimer.singleShot(
                        0, lambda: self.target.setCurrentIndex(target_index),
                    )
                if self.restart_pending:
                    QtCore.QTimer.singleShot(0, self._finish_pending_restart)
                    return
            if (
                self.renderer is None or self.controller is None
                or self.presentation_failed
            ):
                return
            if (
                self.slow_diagnostic.isChecked()
                and now - self.last_submission < 0.5
            ):
                return
            render_width, render_height = self.extent
            ratio = float(self.native_window.devicePixelRatio())
            surface_size = (
                max(1, round(self.native_window.width() * ratio)),
                max(1, round(self.native_window.height() * ratio)),
            )
            camera = self.controller.camera()
            if self.animate.isChecked() and item.animate is not None:
                item.animate(self.scene_value, now)
            self.last_submission = now
            if self.renderer_target == "vulkan-raster":
                raster_width, raster_height = _surface_aspect_extent(
                    (render_width, render_height), surface_size,
                )
                self.future = self.executor.submit(
                    self.renderer.present_frame,
                    self.scene_value, camera, raster_width, raster_height,
                    surface_size=surface_size,
                    diagnostic_readback=(
                        args.diagnostic_camera_pose is not None
                    ),
                )
            elif (
                self.renderer_target == "wavefront-gi"
                and self.renderer_denoiser_backend == "nrd-reference"
            ):
                if not isinstance(self.renderer, _NrdReferencePreview):
                    self.status.setText(
                        "Waiting for the NRD reference renderer restart…"
                    )
                    return
                self.future = self.executor.submit(
                    self.renderer.render,
                    self.scene_value, camera,
                    (render_width, render_height),
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
        if {"raster-feature", "volumes"}.intersection(item.tags)
    )


def _renderer(
    showcase, scene, backend_name, shadows, shadow_map_size,
    restir_reservoirs=4,
):
    if backend_name == "wavefront-gi":
        return ol.Renderer(
            config=_gi_config(
                showcase, restir_reservoirs=restir_reservoirs,
            ),
            renderer_preference="gi",
        )
    target = "spirv" if backend_name == "vulkan-raster" else "wgsl"
    settings = dict(showcase.renderer)
    settings.pop("scene_light_toggle", None)
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
        "--showcase",
        help="initial showcase identifier",
    )
    parser.add_argument(
        "--diagnostic-pose", metavar="JSON_OR_FILE",
        help=(
            "start at an inline/file camera pose with animation disabled, "
            "a fixed-size viewport, diagnostic logging, and a 2 FPS cap"
        ),
    )
    parser.add_argument(
        "--diagnostic-frames", type=int, default=0,
        help="capture this many direct HDR frames, write a report, and exit",
    )
    parser.add_argument(
        "--diagnostic-warmup-frames", type=int, default=8,
        help="completed frames to discard before direct HDR capture",
    )
    parser.add_argument(
        "--diagnostic-report",
        default="/tmp/ordinarylight-raster-diagnostic.json",
        help="output path used with --diagnostic-frames",
    )
    parser.add_argument(
        "--readback", action="store_true",
        help="use the NumPy/QImage comparison path instead of direct Vulkan",
    )
    parser.add_argument(
        "--switch-target-after-frames", type=int, default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--switch-target-to", choices=tuple(key for _title, key in TARGETS),
        default="wavefront-gi", help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    args.diagnostic_camera_pose = None
    if args.diagnostic_pose:
        try:
            args.diagnostic_camera_pose = _camera_pose_argument(
                args.diagnostic_pose,
            )
        except ValueError as error:
            parser.error(str(error))
        if args.showcase and args.showcase != args.diagnostic_camera_pose[0]:
            parser.error(
                "--showcase must match the showcase in --diagnostic-pose"
            )
    if args.diagnostic_frames < 0:
        parser.error("--diagnostic-frames must not be negative")
    if args.diagnostic_warmup_frames < 0:
        parser.error("--diagnostic-warmup-frames must not be negative")
    if args.switch_target_after_frames < 0:
        parser.error("--switch-target-after-frames must not be negative")
    if args.diagnostic_frames and args.diagnostic_camera_pose is None:
        parser.error("--diagnostic-frames requires --diagnostic-pose")
    if args.diagnostic_frames and args.readback:
        parser.error("--diagnostic-frames currently measures direct presentation")
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
            self._last_submission = 0.0
            self._completed_times = deque(maxlen=240)
            self._diagnostic_reference = None
            self._startup_camera = (
                args.diagnostic_camera_pose[1]
                if args.diagnostic_camera_pose is not None else None
            )

            root = QtWidgets.QWidget(); self.setCentralWidget(root)
            layout = QtWidgets.QHBoxLayout(root)
            self.image = Viewport(self); layout.addWidget(self.image, 1)
            if args.diagnostic_camera_pose is not None:
                self.image.setFixedSize(args.width, args.height)
            panel = QtWidgets.QWidget(); panel.setMaximumWidth(430)
            form = QtWidgets.QFormLayout(panel); layout.addWidget(panel)
            self.feature = QtWidgets.QComboBox()
            for item in showcases:
                self.feature.addItem(item.title, item)
            startup_showcase = (
                args.diagnostic_camera_pose[0]
                if args.diagnostic_camera_pose is not None else args.showcase
            )
            if startup_showcase is not None:
                startup_index = next((
                    index for index in range(self.feature.count())
                    if self.feature.itemData(index).id == startup_showcase
                ), -1)
                if startup_index < 0:
                    raise ValueError(f"unknown showcase {startup_showcase!r}")
                self.feature.setCurrentIndex(startup_index)
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
            self.scene_lights = QtWidgets.QCheckBox()
            self.scene_lights.setChecked(True)
            self.scene_lights.toggled.connect(self._scene_light_changed)
            self.map_size = QtWidgets.QComboBox()
            for size in (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192):
                self.map_size.addItem(str(size), size)
            self.restir_reservoirs = QtWidgets.QComboBox()
            for count in (1, 2, 4, 8):
                self.restir_reservoirs.addItem(str(count), count)
            self.restir_reservoirs.setCurrentIndex(
                self.restir_reservoirs.findData(4)
            )
            self.restir_reservoirs.setEnabled(
                "wavefront-gi" in tuple(self.backend.currentData())
            )
            self.animate = QtWidgets.QCheckBox()
            self.animate.setChecked(args.diagnostic_camera_pose is None)
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
            form.addRow("Enable optional scene light", self.scene_lights)
            form.addRow("Shadow map size", self.map_size)
            form.addRow("ReSTIR reservoirs", self.restir_reservoirs)
            form.addRow("Animate scene / camera", self.animate)
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
            supports_light_toggle = bool(
                item.renderer.get("scene_light_toggle", False)
            )
            self.scene_lights.setEnabled(supports_light_toggle)
            self.scene_lights.setChecked(True)
            value = int(item.renderer.get("shadow_map_size", 512))
            index = self.map_size.findData(value)
            self.map_size.setCurrentIndex(max(index, 0))

        def _target_changed(self, _index=None):
            self.restir_reservoirs.setEnabled(
                "wavefront-gi" in tuple(self.backend.currentData())
            )
            if self.scene_value is not None:
                self.restart()

        def _scene_light_changed(self, _enabled=None):
            if getattr(self, "scene_value", None) is not None:
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
                _set_optional_scene_lights(
                    self.scene_value, self.scene_lights.isChecked(),
                )
                if self._startup_camera is not None:
                    self.controller = ol.ArcballCameraController.from_camera(
                        self._startup_camera,
                    )
                    self._startup_camera = None
                self.renderers = [
                    (name, _renderer(
                        item, self.scene_value, name, self.shadows.isChecked(),
                        int(self.map_size.currentData()),
                        int(self.restir_reservoirs.currentData()),
                    ))
                    for name in names
                ]
                self._completed_times.clear()
                self._diagnostic_reference = None
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
                    diagnostic_pixels = np.concatenate(
                        [image for _name, image in completed], axis=1,
                    )
                    diagnostic_hash = hashlib.sha256(
                        diagnostic_pixels.tobytes()
                    ).hexdigest()[:16]
                    difference_text = ""
                    if args.diagnostic_camera_pose is not None:
                        if self._diagnostic_reference is None:
                            self._diagnostic_reference = diagnostic_pixels.copy()
                            difference_text = " | reference"
                        else:
                            difference = frame_difference(
                                self._diagnostic_reference,
                                diagnostic_pixels,
                            )
                            difference_text = (
                                " | diff max "
                                f"{difference['maximum_absolute_difference']:.6g}"
                                f" · RMSE {difference['rmse']:.6f}"
                                f" · pixels {difference['changed_pixels']}"
                            )
                    self.status.setText(
                        f"{fps:.1f} FPS{gpu_text} | {width} × {height} | "
                        + ", ".join(name for name, _image in completed)
                        + f" | SDR {diagnostic_hash}"
                        + difference_text
                    )
                except Exception as error:
                    self.jobs.clear()
                    self.status.setText(f"Render failed: {error}")
                    return
            if (
                args.diagnostic_camera_pose is not None
                and now - self._last_submission < 0.5
            ):
                return
            if self.live.isChecked() or self._qimage is None:
                self._last_submission = now
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
