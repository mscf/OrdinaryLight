"""High-level, backend-neutral rendering interface."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import asyncio
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from threading import Lock
from types import MappingProxyType
from typing import Any

import numpy as np

from .capabilities import capabilities_from_backend
from .backends.base import RenderBackend
from .cameras import CAMERA_TYPES, Camera
from .effects import ObjectEffect
from .scene import Scene
from .state import AccumulationState


def _default_backend(config, config_options):
    """Construct the optional Vulkan backend without importing it at module load."""
    try:
        from .backends.vulkan import RendererConfig, VulkanRayTracingBackend
    except ImportError as error:
        if error.name == "vulkan":
            raise RuntimeError(
                "The default renderer requires the Vulkan extra; install "
                "ordinarylight[vulkan], or pass an explicit backend"
            ) from error
        raise
    return VulkanRayTracingBackend(
        config=config or RendererConfig(**config_options)
    )


class RenderStatistics(Mapping):
    """Immutable, backend-neutral measurements for one completed render."""

    def __init__(self, *, frame_index, size, samples, timings=None):
        timings = dict(timings or {})
        self._timings = MappingProxyType(timings)
        self._values = MappingProxyType({
            "frame_index": int(frame_index),
            "width": int(size[0]),
            "height": int(size[1]),
            "samples": samples,
            "total_ms": self._duration(
                timings, "total_ms", "frame_ms", "cpu_frame_ms"
            ),
            "gpu_ms": self._duration(
                timings, "gpu_ms", "gpu_frame_ms", "dispatch_gpu_ms"
            ),
        })

    @staticmethod
    def _duration(timings, *names):
        for name in names:
            value = timings.get(name)
            if value is not None:
                return float(value)
        return None

    @property
    def timings(self):
        """Complete backend-specific timing measurements."""
        return self._timings

    def as_dict(self, *, include_timings=True):
        """Return a flat record suitable for tables and serialization."""
        record = dict(self._values)
        if include_timings:
            for name, value in self._timings.items():
                record.setdefault(name, value)
        return record

    def __getitem__(self, name):
        return self._values[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __getattr__(self, name):
        try:
            return self._values[name]
        except KeyError as error:
            raise AttributeError(name) from error


class RenderFrame(Mapping):
    """Immutable named collection of renderer products and frame metadata."""

    def __init__(self, products, *, metadata=None):
        if not products:
            raise ValueError("a render frame requires at least one product")
        normalized = {}
        for name, value in products.items():
            if not isinstance(name, str) or not name:
                raise TypeError("render product names must be non-empty strings")
            if not isinstance(value, np.ndarray):
                raise TypeError(f"render product {name!r} must be a NumPy array")
            normalized[name] = value
        self._products = MappingProxyType(normalized)
        self._metadata = MappingProxyType(dict(metadata or {}))

    @property
    def metadata(self):
        return self._metadata

    def __getitem__(self, name):
        return self._products[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._products)

    def __len__(self):
        return len(self._products)

    def __getattr__(self, name):
        try:
            return self._products[name]
        except KeyError as error:
            raise AttributeError(name) from error


@dataclass(frozen=True)
class _CompletedRender:
    value: Any
    statistics: RenderStatistics


class RenderJob:
    """A nonblocking render submission owned by one :class:`Renderer`.

    Jobs retain their scene, camera, and output arguments until completion.
    Calls submitted through one renderer execute in order so a backend never
    receives concurrent mutation or Vulkan calls from multiple Python threads.
    """

    def __init__(self, future, *, frame_index, scene, camera):
        self._future = future
        self._frame_index = int(frame_index)
        self._scene = scene
        self._camera = camera

    @property
    def frame_index(self):
        return self._frame_index

    def done(self):
        return self._future.done()

    def running(self):
        return self._future.running()

    def cancelled(self):
        return self._future.cancelled()

    @property
    def statistics(self):
        if not self.done() or self.cancelled():
            return None
        try:
            return self._future.result().statistics
        except Exception:
            return None

    def cancel(self):
        """Cancel a submission that has not started executing."""
        return self._future.cancel()

    def wait(self, timeout=None):
        """Wait up to ``timeout`` seconds and return whether the job finished."""
        done, _pending = wait((self._future,), timeout=timeout)
        return bool(done)

    def result(self, timeout=None):
        """Return the render result, propagating cancellation or render errors."""
        return self._future.result(timeout=timeout).value

    def exception(self, timeout=None):
        """Return the render error, or ``None`` after successful completion."""
        return self._future.exception(timeout=timeout)

    def add_done_callback(self, callback):
        """Invoke ``callback(job)`` once this submission reaches a terminal state."""
        if not callable(callback):
            raise TypeError("callback must be callable")
        self._future.add_done_callback(lambda _future: callback(self))
        return self

    async def _async_result(self):
        completed = await asyncio.wrap_future(self._future)
        return completed.value

    def __await__(self):
        """Allow ``frame = await renderer.render_async(...)``."""
        return self._async_result().__await__()


def _size(value):
    try:
        width, height = value
    except (TypeError, ValueError) as error:
        raise TypeError("size must be a (width, height) pair") from error
    width, height = int(width), int(height)
    if width < 1 or height < 1:
        raise ValueError("render dimensions must be positive")
    return width, height


class Renderer:
    """Render scenes to HDR NumPy arrays without exposing backend mechanics.

    By default this class owns the optional Vulkan backend. A compatible
    :class:`ordinarylight.backends.RenderBackend` may be supplied explicitly.

    ``render()`` returns a ``float32`` array with shape ``(height, width, 4)``.
    The first three channels contain linear HDR radiance.  The fourth channel
    is reserved for renderer metadata and should not be treated as alpha.
    """

    def __init__(
        self,
        *,
        config: Any | None = None,
        backend: RenderBackend | None = None,
        **config_options,
    ):
        if config is not None and config_options:
            raise TypeError("pass config or renderer options, not both")
        if backend is not None and (config is not None or config_options):
            raise TypeError("a supplied backend owns its configuration")
        self._backend = (
            backend if backend is not None
            else _default_backend(config, config_options)
        )
        if not callable(getattr(self._backend, "render_frame", None)):
            raise TypeError("backend must implement RenderBackend.render_frame()")
        if not callable(getattr(self._backend, "close", None)):
            raise TypeError("backend must provide close()")
        self._frame_index = 0
        self._last_statistics = None
        self._capabilities = capabilities_from_backend(self._backend)
        self._state_lock = Lock()
        self._submission_lock = Lock()
        self._backend_lock = Lock()
        self._accepting_jobs = True
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="ordinarylight-render"
        )

    @property
    def config(self):
        """The immutable configuration owned by the active backend."""
        return getattr(self._backend, "config", None)

    @property
    def device(self):
        """Selected device information, when exposed by the backend."""
        return getattr(self._backend, "device", None)

    @property
    def last_timings(self):
        """Timing measurements reported by the most recent render."""
        return dict(getattr(self._backend, "last_timings", {}))

    @property
    def last_statistics(self):
        """Structured measurements for the most recent successful render."""
        return self._last_statistics

    @property
    def frame_index(self):
        """Index that will seed the next render in the current sequence."""
        with self._state_lock:
            return self._frame_index

    @property
    def available_outputs(self):
        """Named products supported by the active backend."""
        if self._backend is None:
            return ()
        return self._capabilities.outputs

    @property
    def capabilities(self):
        """Immutable semantic features, products, limits, and device details."""
        return self._capabilities

    @property
    def accumulation_state(self):
        """Current stationary-accumulation state reported by the backend."""
        return getattr(
            self._backend, "accumulation_state", AccumulationState.DISABLED
        )

    @property
    def accumulated_frames(self):
        """Frames represented by the current progressive history."""
        return int(getattr(self._backend, "accumulated_frames", 0))

    @property
    def effective_samples_per_pixel(self):
        """Sample budget selected for the most recently completed frame."""
        return int(getattr(
            self._backend, "effective_samples_per_pixel",
            getattr(self.config, "samples_per_pixel", 1),
        ))

    def reset_sequence(self, frame_index=0):
        """Reset deterministic sampling for a new camera or data sequence."""
        frame_index = int(frame_index)
        if frame_index < 0:
            raise ValueError("frame_index cannot be negative")
        with self._submission_lock:
            with self._state_lock:
                if not self._accepting_jobs:
                    raise RuntimeError("renderer is closed")
                self._frame_index = frame_index
            # Sequence resets are ordered behind already-submitted rendering
            # and ahead of later submissions.
            self._executor.submit(self._reset_backend_history).result()

    def replace_scene(self, scene: Scene, *, reset_sequence=True):
        """Make ``scene`` resident without recreating this renderer.

        Vulkan backends retain their device, compiled pipelines, presentation
        resources, and exported video-frame pool.  The operation is ordered
        after already submitted asynchronous renders.  Backends without
        resident-scene support raise ``RuntimeError`` rather than silently
        rebuilding themselves.
        """
        if not isinstance(scene, Scene):
            raise TypeError("scene must be a Scene")
        with self._submission_lock:
            with self._state_lock:
                if not self._accepting_jobs or self._backend is None:
                    raise RuntimeError("renderer is closed")
            future = self._executor.submit(self._replace_scene_serialized, scene)
            future.result()
            if reset_sequence:
                with self._state_lock:
                    self._frame_index = 0
        return scene

    def _replace_scene_serialized(self, scene):
        with self._backend_lock:
            replace_scene = getattr(self._backend, "replace_scene", None)
            if not callable(replace_scene):
                raise RuntimeError(
                    "the active backend does not support resident scene replacement"
                )
            replace_scene(scene)
            self._last_statistics = None

    def reconfigure(self, **changes):
        """Apply settings supported by the resident backend in place.

        Common per-frame Vulkan settings currently include
        ``samples_per_pixel``, ``max_bounces``, ``wavefront_exposure``, and
        ``wavefront_render_scale``. Structural changes raise ``RuntimeError``
        with an explicit recreation requirement.
        """
        if not changes:
            return self.config
        with self._submission_lock:
            with self._state_lock:
                if not self._accepting_jobs or self._backend is None:
                    raise RuntimeError("renderer is closed")
            return self._executor.submit(
                self._reconfigure_serialized, dict(changes)
            ).result()

    def _reconfigure_serialized(self, changes):
        with self._backend_lock:
            reconfigure = getattr(self._backend, "reconfigure", None)
            if not callable(reconfigure):
                raise RuntimeError(
                    "the active backend requires renderer recreation for settings"
                )
            config = reconfigure(**changes)
            self._last_statistics = None
            return config

    def _reset_backend_history(self):
        with self._backend_lock:
            reset_history = getattr(self._backend, "reset_output_history", None)
            if callable(reset_history):
                reset_history()

    def apply_object_effect(self, scene: Scene, reference, effect: ObjectEffect):
        """Apply a transient visual effect to one scene object.

        The operation is ordered after prior render submissions and before
        later ones. Picking and application selection state remain independent.
        """
        if not isinstance(scene, Scene):
            raise TypeError("scene must be a Scene")
        if not isinstance(effect, ObjectEffect):
            raise TypeError("effect must be an ordinarylight.effects object")
        scene.object_triangle_range(reference)
        return self._submit_object_effect(scene, reference, effect)

    def clear_object_effect(self):
        """Remove the active transient object effect, if supported."""
        return self._submit_object_effect(None, None, None)

    def _submit_object_effect(self, scene, reference, effect):
        with self._submission_lock:
            with self._state_lock:
                if not self._accepting_jobs or self._backend is None:
                    raise RuntimeError("renderer is closed")
            return self._executor.submit(
                self._object_effect_serialized, scene, reference, effect
            ).result()

    def _object_effect_serialized(self, scene, reference, effect):
        with self._backend_lock:
            name = "clear_object_effect" if effect is None else "apply_object_effect"
            operation = getattr(self._backend, name, None)
            if not callable(operation):
                raise RuntimeError(
                    "the active backend does not support renderer-side object effects"
                )
            if effect is None:
                return operation()
            return operation(scene, reference, effect)

    def render(
        self,
        scene: Scene,
        camera: Camera,
        size,
        *,
        samples: int | None = None,
        frame_index: int | None = None,
        out: np.ndarray | None = None,
        outputs=None,
    ):
        """Render one frame as linear HDR radiance, blocking until complete.

        ``size`` is ``(width, height)``. If ``frame_index`` is omitted, the
        renderer advances its deterministic sequence after each successful
        call. Supplying ``out`` reuses a caller-owned float32 array.
        """
        return self.render_async(
            scene, camera, size, samples=samples, frame_index=frame_index,
            out=out, outputs=outputs,
        ).result()

    def render_async(
        self,
        scene: Scene,
        camera: Camera,
        size,
        *,
        samples: int | None = None,
        frame_index: int | None = None,
        out: np.ndarray | None = None,
        outputs=None,
    ):
        """Submit a frame and immediately return a :class:`RenderJob`.

        Submissions execute in order on a renderer-owned worker. A queued job
        can be cancelled; a running Vulkan submission completes normally.
        """
        if not isinstance(scene, Scene):
            raise TypeError("scene must be a Scene")
        if not isinstance(camera, CAMERA_TYPES):
            raise TypeError("camera must be a supported ordinarylight camera")
        width, height = _size(size)
        with self._submission_lock:
            with self._state_lock:
                if not self._accepting_jobs or self._backend is None:
                    raise RuntimeError("renderer is closed")
                current_frame = (
                    self._frame_index if frame_index is None else int(frame_index)
                )
                if current_frame < 0:
                    raise ValueError("frame_index cannot be negative")
                # Reserve deterministic sequence indices at submission time. A
                # cancelled queued job may intentionally leave a gap.
                self._frame_index = current_frame + 1
            future = self._executor.submit(
                self._render_serialized, scene, camera, width, height,
                samples, current_frame, out, outputs,
            )
        return RenderJob(
            future, frame_index=current_frame, scene=scene, camera=camera
        )

    def render_gpu(
        self, scene: Scene, camera: Camera, size, *, samples=None,
        frame_index=None, pixel_format="rgba8",
    ):
        """Submit a frame whose color product remains GPU-resident.

        This optional backend capability returns a
        :class:`ordinarylight.GpuFrame`. The call performs CPU command
        recording and submission but does not wait for pixel readback; use the
        frame's exported synchronization object from the consuming GPU API.
        ``pixel_format`` is ``"rgba8"`` for general Vulkan image interop,
        ``"nv12"`` for 8-bit video, or ``"p010"`` for 10-bit video input.
        """
        if not isinstance(scene, Scene):
            raise TypeError("scene must be a Scene")
        if not isinstance(camera, CAMERA_TYPES):
            raise TypeError("camera must be a supported ordinarylight camera")
        width, height = _size(size)
        with self._submission_lock:
            with self._state_lock:
                if not self._accepting_jobs or self._backend is None:
                    raise RuntimeError("renderer is closed")
                current_frame = (
                    self._frame_index if frame_index is None else int(frame_index)
                )
                if current_frame < 0:
                    raise ValueError("frame_index cannot be negative")
                self._frame_index = current_frame + 1
            render_gpu_frame = getattr(self._backend, "render_gpu_frame", None)
            if not callable(render_gpu_frame):
                raise RuntimeError(
                    "the active backend does not support GPU-resident output"
                )
            with self._backend_lock:
                frame = render_gpu_frame(
                    scene, camera, width, height, samples=samples,
                    frame_index=current_frame, pixel_format=pixel_format,
                )
                effective_samples = int(
                    frame.attributes.get(
                        "samples_per_pixel",
                        getattr(
                            self._backend, "effective_samples_per_pixel",
                            samples if samples is not None
                            else getattr(self.config, "samples_per_pixel", 1),
                        ),
                    )
                )
                self._last_statistics = RenderStatistics(
                    frame_index=current_frame, size=(width, height),
                    samples=effective_samples, timings=self.last_timings,
                )
                return frame

    def _render_serialized(
        self, scene, camera, width, height, samples, current_frame, out, outputs,
    ):
        with self._backend_lock:
            value = self._render_now(
                scene, camera, width, height, samples=samples,
                current_frame=current_frame, out=out, outputs=outputs,
            )
            return _CompletedRender(value, self._last_statistics)

    def _render_now(
        self, scene, camera, width, height, *, samples, current_frame, out, outputs,
    ):
        if current_frame < 0:
            raise ValueError("frame_index cannot be negative")
        named = outputs is not None
        if outputs is None:
            requested = ("color",)
        elif isinstance(outputs, str):
            requested = (outputs,)
        else:
            requested = tuple(outputs)
        if not requested or any(
            not isinstance(name, str) or not name for name in requested
        ):
            raise ValueError("outputs must contain one or more product names")
        if len(set(requested)) != len(requested):
            raise ValueError("outputs cannot contain duplicate names")
        unavailable = tuple(
            name for name in requested if name not in self.available_outputs
        )
        if unavailable:
            raise ValueError(
                f"unsupported render outputs {unavailable}; available outputs "
                f"are {self.available_outputs}"
            )
        if named and out is not None and not isinstance(out, Mapping):
            raise TypeError("named outputs require out to be a mapping of arrays")
        if not named and out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError("out must be a NumPy array")
            if out.shape != (height, width, 4):
                raise ValueError(
                    f"out must have shape {(height, width, 4)}, got {out.shape}"
                )
            if out.dtype != np.float32:
                raise TypeError("out must use float32 components")

        if named and callable(getattr(self._backend, "render_products", None)):
            products = self._backend.render_products(
                scene, camera, width, height, outputs=requested,
                samples=samples, frame_index=current_frame,
            )
        else:
            products = {"color": self._backend.render_frame(
                scene, camera, width, height,
                samples=samples, frame_index=current_frame,
            )}
        products = {name: np.asarray(products[name]) for name in requested}
        expected = {
            "color": ((height, width, 4), np.dtype(np.float32)),
            "variance": ((height, width), np.dtype(np.float32)),
            "depth": ((height, width), np.dtype(np.float32)),
            "normal": ((height, width, 3), np.dtype(np.float32)),
            "instance_id": ((height, width), np.dtype(np.uint32)),
            "object_id": ((height, width), np.dtype(np.uint32)),
            "material_id": ((height, width), np.dtype(np.uint32)),
            "motion": ((height, width, 2), np.dtype(np.float32)),
        }
        for name, result in products.items():
            shape, dtype = expected[name]
            if result.shape != shape or result.dtype != dtype:
                raise RuntimeError(
                    f"backend returned invalid {name!r}; expected {dtype} "
                    f"{shape}, got {result.dtype} {result.shape}"
                )
        statistics = RenderStatistics(
            frame_index=current_frame,
            size=(width, height),
            samples=samples,
            timings=self.last_timings,
        )
        self._last_statistics = statistics
        if named:
            if out is not None:
                unknown = set(out) - set(requested)
                if unknown:
                    raise ValueError(f"out contains unrequested products: {unknown}")
                for name, destination in out.items():
                    if not isinstance(destination, np.ndarray):
                        raise TypeError(f"out[{name!r}] must be a NumPy array")
                    if (destination.shape != products[name].shape
                            or destination.dtype != products[name].dtype):
                        raise ValueError(
                            f"out[{name!r}] must match {products[name].dtype} "
                            f"{products[name].shape}"
                        )
                    np.copyto(destination, products[name])
                    products[name] = destination
            return RenderFrame(products, metadata={
                "frame_index": current_frame,
                "size": (width, height),
                "samples": samples,
                "timings": self.last_timings,
                "statistics": statistics,
            })
        result = products["color"]
        if out is not None:
            np.copyto(out, result)
            return out
        return result

    def close(self):
        """Release renderer resources. Safe to call more than once."""
        with self._submission_lock:
            with self._state_lock:
                if not self._accepting_jobs:
                    return
                self._accepting_jobs = False
        self._executor.shutdown(wait=True, cancel_futures=True)
        with self._backend_lock:
            if self._backend is not None:
                self._backend.close()
                self._backend = None

    def __enter__(self):
        if self._backend is None:
            raise RuntimeError("renderer is closed")
        return self

    def __exit__(self, *_args):
        self.close()
