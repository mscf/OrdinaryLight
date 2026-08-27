"""Optional high-level GLFW viewport for native Vulkan presentation."""

from __future__ import annotations

import time

from ..renderer import RenderStatistics
from ..scene import PerspectiveCamera, Scene
from ..backends.vulkan import RendererConfig, VulkanGlfwPresenter
from .glfw_platform import load_glfw
from .resize import ResizeRecreationGate


class NativeViewport:
    """Manage a GLFW window and present a scene without host pixel readback.

    Applications may call :meth:`step` from their own event loop or use
    :meth:`run`. A controller is any callable accepting ``(viewport, dt)``;
    it may replace ``viewport.camera`` or update resources in ``viewport.scene``.
    """

    def __init__(
        self,
        scene,
        camera,
        *,
        size=(1280, 720),
        title="Ordinary Light",
        config=None,
        controller=None,
        resize_settle_seconds=0.15,
        _glfw=None,
        _window=None,
        _presenter=None,
        _clock=time.perf_counter,
    ):
        self._validate_scene_camera(scene, camera)
        width, height = self._validate_size(size)
        self.scene = scene
        self.camera = camera
        self.controller = controller
        self.config = config or RendererConfig()
        self._clock = _clock
        self._last_step_time = None
        self._frame_index = 0
        self._last_statistics = None
        self._closed = False
        self._owns_glfw = _glfw is None
        self._glfw = _glfw or load_glfw()
        if self._owns_glfw and not self._glfw.init():
            raise RuntimeError("GLFW initialization failed")
        try:
            if _window is None:
                self._glfw.window_hint(self._glfw.CLIENT_API, self._glfw.NO_API)
                self._window = self._glfw.create_window(
                    width, height, str(title), None, None
                )
                if not self._window:
                    raise RuntimeError("GLFW Vulkan window creation failed")
            else:
                self._window = _window
            self._presenter = _presenter or VulkanGlfwPresenter(
                self._window, config=self.config
            )
        except Exception:
            if self._owns_glfw:
                self._glfw.terminate()
            raise
        self._resize_gate = ResizeRecreationGate(resize_settle_seconds)

    @staticmethod
    def _validate_scene_camera(scene, camera):
        if not isinstance(scene, Scene):
            raise TypeError("scene must be a Scene")
        if not isinstance(camera, PerspectiveCamera):
            raise TypeError("camera must be a PerspectiveCamera")

    @staticmethod
    def _validate_size(size):
        try:
            width, height = (int(value) for value in size)
        except (TypeError, ValueError) as error:
            raise TypeError("size must be a (width, height) pair") from error
        if width < 1 or height < 1:
            raise ValueError("viewport dimensions must be positive")
        return width, height

    @property
    def window(self):
        """Native GLFW window handle for optional application integration."""
        return self._window

    @property
    def presenter(self):
        """Low-level presenter for advanced renderer controls."""
        return self._presenter

    @property
    def frame_index(self):
        return self._frame_index

    @property
    def last_statistics(self):
        return self._last_statistics

    @property
    def should_close(self):
        return self._closed or bool(
            self._glfw.window_should_close(self._window)
        )

    def request_close(self):
        if not self._closed:
            self._glfw.set_window_should_close(self._window, True)

    def set_scene(self, scene):
        self._validate_scene_camera(scene, self.camera)
        self.scene = scene
        self.reset_sequence()

    def set_camera(self, camera):
        self._validate_scene_camera(self.scene, camera)
        self.camera = camera
        self.reset_sequence()

    def reset_sequence(self, frame_index=0):
        frame_index = int(frame_index)
        if frame_index < 0:
            raise ValueError("frame_index cannot be negative")
        self._frame_index = frame_index
        self._presenter.reset_accumulation()

    def step(self):
        """Process events and present at most one frame.

        Returns the new :class:`RenderStatistics`, or ``None`` while minimized,
        during resize settling, or after close was requested.
        """
        if self._closed:
            raise RuntimeError("viewport is closed")
        self._glfw.poll_events()
        if self.should_close:
            return None
        now = self._clock()
        dt = 0.0 if self._last_step_time is None else now - self._last_step_time
        self._last_step_time = now
        if self.controller is not None:
            self.controller(self, dt)
            self._validate_scene_camera(self.scene, self.camera)
        width, height = self._glfw.get_framebuffer_size(self._window)
        if width < 1 or height < 1:
            return None
        if not self._resize_gate.should_render(
            (width, height), now,
            resources_allocated=self._presenter.swapchain_image_count > 0,
        ):
            return None
        self._presenter.present_wavefront(
            self.scene, self.camera, width, height
        )
        statistics = RenderStatistics(
            frame_index=self._frame_index,
            size=(width, height),
            samples=self._presenter.effective_samples_per_pixel,
            timings=self._presenter.last_timings,
        )
        self._last_statistics = statistics
        self._frame_index += 1
        return statistics

    def run(self, *, max_frames=None, on_frame=None):
        """Present until close, optionally stopping after rendered frames."""
        if max_frames is not None:
            max_frames = int(max_frames)
            if max_frames < 0:
                raise ValueError("max_frames cannot be negative")
        rendered = 0
        while not self.should_close and (
            max_frames is None or rendered < max_frames
        ):
            statistics = self.step()
            if statistics is None:
                self._glfw.wait_events_timeout(0.01)
                continue
            rendered += 1
            if on_frame is not None:
                on_frame(self, statistics)
        return rendered

    def close(self):
        if self._closed:
            return
        self._presenter.close()
        self._glfw.destroy_window(self._window)
        if self._owns_glfw:
            self._glfw.terminate()
        self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("viewport is closed")
        return self

    def __exit__(self, *_args):
        self.close()
