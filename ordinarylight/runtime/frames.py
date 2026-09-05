"""Bounded CPU frame admission; GPU stages remain queued without host waits."""

from dataclasses import dataclass
from operator import index


@dataclass
class VulkanFrameSlot:
    index: int
    completion: object = None


class VulkanFrameRing:
    """Wait only when reusing a slot; applications own each slot's resources."""

    def __init__(self, runtime, frames=2):
        self.runtime = runtime
        count = index(frames)
        if not 1 <= count <= 16:
            raise ValueError("Frame ring size must be between one and sixteen")
        self.slots = tuple(VulkanFrameSlot(i) for i in range(count))
        self._next = 0
        self._acquired = None
        self.closed = False
        runtime.require_open()
        runtime.retain(self)

    def acquire(self):
        with self.runtime.lock:
            if self.closed or self._acquired is not None:
                raise RuntimeError("Frame ring is closed or a slot awaits submission")
            slot = self.slots[self._next]
            if slot.completion is not None:
                slot.completion.wait()
            self._acquired = slot
            return slot

    def submit(self, graph, *, after=()):
        with self.runtime.lock:
            if self.closed or self._acquired is None:
                raise RuntimeError("Acquire a frame slot before submitting")
            completion = graph.execute(self.runtime, after=after)
            self._acquired.completion = completion
            self._acquired = None
            self._next = (self._next + 1) % len(self.slots)
            return completion

    def cancel(self):
        with self.runtime.lock:
            self._acquired = None

    def close(self):
        with self.runtime.lock:
            if self.closed:
                return
            for slot in self.slots:
                if slot.completion is not None:
                    slot.completion.wait()
            self.closed = True
            self.runtime.release(self)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
