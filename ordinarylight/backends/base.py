"""Backend-neutral renderer implementation contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class RenderBackend(Protocol):
    """Structural contract implemented by Ordinary Light render backends.

    Backends own their configuration and execution resources. ``render_frame``
    returns linear HDR color, while ``render_products`` optionally returns
    additional named products. Implementations serialize their own mutable
    device state; :class:`ordinarylight.Renderer` serializes calls per backend.
    """

    available_outputs: tuple[str, ...]
    config: Any
    device: Any
    last_timings: Mapping[str, Any]

    def render_frame(
        self,
        scene,
        camera,
        width: int,
        height: int,
        *,
        samples: int | None = None,
        frame_index: int = 0,
    ) -> np.ndarray:
        """Render one linear-HDR color product."""
        ...

    def close(self) -> None:
        """Release backend resources. Implementations should be idempotent."""
        ...


@runtime_checkable
class ProductRenderBackend(RenderBackend, Protocol):
    """Optional extension for backends supporting multiple named products."""

    def render_products(
        self,
        scene,
        camera,
        width: int,
        height: int,
        *,
        outputs: tuple[str, ...],
        samples: int | None = None,
        frame_index: int = 0,
    ) -> Mapping[str, np.ndarray]:
        """Render the requested named products."""
        ...


@runtime_checkable
class GpuRenderBackend(RenderBackend, Protocol):
    """Optional extension for zero-copy GPU-resident color output."""

    def render_gpu_frame(
        self, scene, camera, width: int, height: int, *,
        samples: int | None = None, frame_index: int = 0,
        pixel_format: str = "rgba8",
    ):
        """Submit a frame and return a managed GPU-resident product."""
        ...


__all__ = ["GpuRenderBackend", "ProductRenderBackend", "RenderBackend"]
