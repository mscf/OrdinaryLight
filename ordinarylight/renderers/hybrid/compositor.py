"""Composable hybrid raster/global-illumination renderer."""

from __future__ import annotations

import numpy as np

from ...capabilities import RendererCapabilities
from ..base import RendererImplementation, RendererImplementationInfo


class HybridRenderer(RendererImplementation):
    """Combine a fast raster frame with a secondary lighting renderer.

    The secondary renderer may return indirect-only radiance (``add``) or a
    complete image blended over raster output (``mix``). Both children retain
    their own device/resource lifetime and can therefore be replaced by native
    Vulkan, WebGPU, or reference implementations.
    """

    available_outputs = ("color",)
    implementation = RendererImplementationInfo(
        name="hybrid-compositor", family="hybrid", graphics_api="composite",
    )

    def __init__(self, raster_renderer, lighting_renderer, *, mode="add", weight=1.0):
        if mode not in {"add", "mix"}:
            raise ValueError("hybrid mode must be add or mix")
        if not 0.0 <= float(weight) <= 1.0:
            raise ValueError("hybrid weight must be in [0, 1]")
        self.raster_renderer = raster_renderer
        self.lighting_renderer = lighting_renderer
        self.mode, self.weight = mode, float(weight)
        self.config = {"mode": mode, "weight": self.weight}
        self.last_timings = {}
        self.capabilities = RendererCapabilities(
            renderer="hybrid",
            features=frozenset({"raster", "global_illumination", "hybrid"}),
        )

    def render_frame(self, scene, camera, width, height, *, samples=None, frame_index=0):
        raster = self.raster_renderer.render_frame(
            scene, camera, width, height, samples=samples, frame_index=frame_index,
        )
        lighting = self.lighting_renderer.render_frame(
            scene, camera, width, height, samples=samples, frame_index=frame_index,
        )
        if self.mode == "add":
            result = raster.copy()
            result[..., :3] += lighting[..., :3] * self.weight
        else:
            result = raster * (1.0 - self.weight) + lighting * self.weight
        self.last_timings = {
            "raster_ms": float(
                getattr(self.raster_renderer, "last_timings", {}).get(
                    "total_ms", 0.0,
                )
            ),
            "lighting_ms": float(
                getattr(self.lighting_renderer, "last_timings", {}).get(
                    "total_ms", 0.0,
                )
            ),
        }
        return np.asarray(result, np.float32)

    def reset_output_history(self):
        for renderer in (self.raster_renderer, self.lighting_renderer):
            reset = getattr(renderer, "reset_output_history", None)
            if callable(reset): reset()

    def close(self):
        closed = set()
        for renderer in (self.raster_renderer, self.lighting_renderer):
            if id(renderer) not in closed:
                renderer.close(); closed.add(id(renderer))


__all__ = ["HybridRenderer"]
