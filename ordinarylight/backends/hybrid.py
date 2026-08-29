"""Composable hybrid raster/global-illumination backend."""

from __future__ import annotations

import numpy as np

from ..capabilities import RendererCapabilities


class HybridBackend:
    """Combine a fast raster frame with a secondary lighting backend.

    The secondary backend may return indirect-only radiance (``add``) or a
    complete image blended over raster output (``mix``). Both children retain
    their own device/resource lifetime and can therefore be replaced by native
    Vulkan, WebGPU, or reference implementations.
    """

    available_outputs = ("color",)

    def __init__(self, raster_backend, lighting_backend, *, mode="add", weight=1.0):
        if mode not in {"add", "mix"}:
            raise ValueError("hybrid mode must be add or mix")
        if not 0.0 <= float(weight) <= 1.0:
            raise ValueError("hybrid weight must be in [0, 1]")
        self.raster_backend = raster_backend
        self.lighting_backend = lighting_backend
        self.mode, self.weight = mode, float(weight)
        self.config = {"mode": mode, "weight": self.weight}
        self.last_timings = {}
        self.capabilities = RendererCapabilities(
            backend="hybrid",
            features=frozenset({"raster", "global_illumination", "hybrid"}),
        )

    def render_frame(self, scene, camera, width, height, *, samples=None, frame_index=0):
        raster = self.raster_backend.render_frame(
            scene, camera, width, height, samples=samples, frame_index=frame_index,
        )
        lighting = self.lighting_backend.render_frame(
            scene, camera, width, height, samples=samples, frame_index=frame_index,
        )
        if self.mode == "add":
            result = raster.copy()
            result[..., :3] += lighting[..., :3] * self.weight
        else:
            result = raster * (1.0 - self.weight) + lighting * self.weight
        self.last_timings = {
            "raster_ms": float(getattr(self.raster_backend, "last_timings", {}).get("total_ms", 0.0)),
            "lighting_ms": float(getattr(self.lighting_backend, "last_timings", {}).get("total_ms", 0.0)),
        }
        return np.asarray(result, np.float32)

    def reset_output_history(self):
        for backend in (self.raster_backend, self.lighting_backend):
            reset = getattr(backend, "reset_output_history", None)
            if callable(reset): reset()

    def close(self):
        closed = set()
        for backend in (self.raster_backend, self.lighting_backend):
            if id(backend) not in closed:
                backend.close(); closed.add(id(backend))


__all__ = ["HybridBackend"]
