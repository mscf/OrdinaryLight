"""Threaded presentation scheduling for responsive GUI integrations."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Queue
from threading import Lock, Thread

from ..renderer import RenderStatistics


@dataclass(frozen=True)
class PresentationEvent:
    """One state, frame, or error event produced by :class:`AsyncPresenter`."""

    kind: str
    generation: int
    statistics: RenderStatistics | None = None
    timings: dict | None = None
    error: BaseException | None = None


class AsyncPresenter:
    """Own a presenter on one worker and admit at most one queued frame.

    The presenter factory and every method on the resulting presenter execute
    on the same worker thread. GUI threads only enqueue immutable frame
    requests and poll completion events, preventing reentrant Vulkan teardown
    or rendering from nested toolkit event loops.
    """

    def __init__(self, presenter_factory):
        if not callable(presenter_factory):
            raise TypeError("presenter_factory must be callable")
        self._factory = presenter_factory
        self._commands = Queue()
        self._events = Queue()
        self._lock = Lock()
        self._generation = 0
        self._ready = False
        self._frame_pending = False
        self._closed = False
        self._thread = Thread(
            target=self._run, name="ordinarylight-present", daemon=True
        )
        self._thread.start()

    @property
    def generation(self):
        with self._lock:
            return self._generation

    @property
    def ready(self):
        with self._lock:
            return self._ready and not self._closed

    @property
    def busy(self):
        with self._lock:
            return self._frame_pending

    def restart(self, config):
        """Asynchronously replace the presenter and return its generation."""
        with self._lock:
            if self._closed:
                raise RuntimeError("async presenter is closed")
            self._generation += 1
            generation = self._generation
            self._ready = False
        self._commands.put(("restart", generation, config))
        return generation

    def request_frame(self, scene, camera, size, *, frame_index=0):
        """Queue one frame if ready and idle; return whether it was accepted."""
        width, height = (int(value) for value in size)
        if width < 1 or height < 1:
            return False
        with self._lock:
            if self._closed or not self._ready or self._frame_pending:
                return False
            self._frame_pending = True
            generation = self._generation
        self._commands.put((
            "frame", generation, scene, camera, (width, height), int(frame_index),
        ))
        return True

    def reset(self):
        """Queue an accumulation reset after any running frame."""
        with self._lock:
            if self._closed:
                return False
            generation = self._generation
        self._commands.put(("reset", generation))
        return True

    def set_object_effect(self, triangle_range=None, effect=None):
        """Queue an object effect for a packed-triangle range, or clear it."""
        bindings = () if triangle_range is None else ((triangle_range, effect),)
        return self.set_object_effects(bindings)

    def set_object_effects(self, bindings=()):
        """Queue replacement of the ordered packed-range effect collection."""
        with self._lock:
            if self._closed:
                return False
            generation = self._generation
        value = tuple(
            (tuple(int(item) for item in triangle_range), effect)
            for triangle_range, effect in bindings
        )
        self._commands.put(("object_effects", generation, value))
        return True

    def poll(self):
        """Return all currently available worker events without blocking."""
        events = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except Empty:
                return tuple(events)

    def close(self, timeout=None):
        """Stop after current work and wait until Vulkan resources are released."""
        with self._lock:
            if self._closed:
                return True
            self._closed = True
            self._ready = False
        self._commands.put(("stop",))
        self._thread.join(timeout)
        return not self._thread.is_alive()

    def _event(self, kind, generation, **values):
        self._events.put(PresentationEvent(kind, generation, **values))

    def _run(self):
        presenter = None
        try:
            while True:
                command = self._commands.get()
                kind = command[0]
                if kind == "stop":
                    break
                if kind == "restart":
                    _kind, generation, config = command
                    try:
                        if presenter is not None:
                            presenter.close()
                        presenter = self._factory(config)
                    except BaseException as error:
                        presenter = None
                        with self._lock:
                            if generation == self._generation:
                                self._ready = False
                        self._event("error", generation, error=error)
                    else:
                        with self._lock:
                            if generation == self._generation:
                                self._ready = True
                        self._event("ready", generation)
                    continue
                if kind == "reset":
                    _kind, generation = command
                    if presenter is not None and generation == self.generation:
                        try:
                            presenter.reset_accumulation()
                        except BaseException as error:
                            self._event("error", generation, error=error)
                    continue
                if kind == "object_effects":
                    _kind, generation, bindings = command
                    if presenter is not None and generation == self.generation:
                        try:
                            presenter.set_object_effects(bindings)
                        except BaseException as error:
                            self._event("error", generation, error=error)
                    continue
                if kind == "frame":
                    (
                        _kind, generation, scene, camera, size, frame_index,
                    ) = command
                    try:
                        if presenter is None or generation != self.generation:
                            continue
                        presenter.present_wavefront(scene, camera, *size)
                        timings = dict(presenter.last_timings)
                        statistics = RenderStatistics(
                            frame_index=frame_index, size=size,
                            samples=presenter.effective_samples_per_pixel,
                            timings=timings,
                        )
                        self._event(
                            "frame", generation, statistics=statistics,
                            timings=timings,
                        )
                    except BaseException as error:
                        self._event("error", generation, error=error)
                    finally:
                        with self._lock:
                            self._frame_pending = False
                    continue
        finally:
            if presenter is not None:
                try:
                    presenter.close()
                except BaseException as error:
                    self._event("error", self.generation, error=error)


__all__ = ["AsyncPresenter", "PresentationEvent"]
