"""Toolkit-neutral scientific inspection helpers."""

from __future__ import annotations

from dataclasses import dataclass

from .clipping import ClipRegion
from .scalar_field import RayProbeResult, ScalarField3D
from .transfer import TransferFunction


@dataclass(slots=True)
class ScientificInspector:
    """Probe one scalar field from explicit or live viewport coordinates."""

    field: ScalarField3D
    transfer_function: TransferFunction
    clipping: ClipRegion | None = None
    step_size: float | None = None
    opacity_threshold: float = 0.01
    last_result: RayProbeResult | None = None

    def __post_init__(self):
        if not isinstance(self.field, ScalarField3D):
            raise TypeError("field must be a ScalarField3D")
        if not isinstance(self.transfer_function, TransferFunction):
            raise TypeError("transfer_function must be a TransferFunction")
        if self.clipping is not None and not isinstance(self.clipping, ClipRegion):
            raise TypeError("clipping must be a ClipRegion or None")
        if self.step_size is not None and self.step_size <= 0:
            raise ValueError("step_size must be positive")
        if not 0.0 <= self.opacity_threshold <= 1.0:
            raise ValueError("opacity_threshold must be in [0, 1]")

    def probe(self, camera, viewport_size, pixel, *, mapping=None):
        """Probe presentation coordinates and retain the complete result."""
        self.last_result = self.field.probe_pixel(
            camera, viewport_size, pixel, self.transfer_function,
            mapping=mapping, step_size=self.step_size,
            opacity_threshold=self.opacity_threshold, clipping=self.clipping,
        )
        return self.last_result

    def probe_viewport(self, viewport, *, mapping=None):
        """Probe the current cursor of a compatible live viewport."""
        return self.probe(
            viewport.camera, viewport.framebuffer_size, viewport.cursor_pixel(),
            mapping=mapping,
        )

    def clear(self):
        self.last_result = None
