"""Renderer implementation contracts and classification metadata."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

import numpy as np


RendererFamily = Literal["global_illumination", "raster", "hybrid", "reference"]
GraphicsApi = Literal["vulkan", "webgpu", "cpu", "composite"]


@dataclass(frozen=True)
class RendererImplementationInfo:
    """Stable identity for a concrete renderer implementation."""

    name: str
    family: RendererFamily
    graphics_api: GraphicsApi


class RendererImplementation:
    """Nominal base for concrete renderers exposed by this package.

    Runtime interoperability remains structural through
    :class:`RendererProtocol`;
    this base adds discoverable identity without constraining third-party
    implementations to inherit from Ordinary Light classes.
    """

    implementation: ClassVar[RendererImplementationInfo]

    @classmethod
    def implementation_info(cls) -> RendererImplementationInfo:
        return cls.implementation


@runtime_checkable
class RendererProtocol(Protocol):
    """Structural contract implemented by Ordinary Light renderers.

    Implementations own their configuration and execution resources. ``render_frame``
    returns linear HDR color, while ``render_products`` optionally returns
    additional named products. Implementations serialize their own mutable
    device state; :class:`ordinarylight.Renderer` serializes calls per implementation.
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
        """Release renderer resources. Implementations should be idempotent."""
        ...


@runtime_checkable
class ProductRendererProtocol(RendererProtocol, Protocol):
    """Optional extension for renderers supporting multiple named products."""

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
class GpuRendererProtocol(RendererProtocol, Protocol):
    """Optional extension for zero-copy GPU-resident color output."""

    def render_gpu_frame(
        self, scene, camera, width: int, height: int, *,
        samples: int | None = None, frame_index: int = 0,
        pixel_format: str = "rgba8",
    ):
        """Submit a frame and return a managed GPU-resident product."""
        ...


@runtime_checkable
class ResidentSceneRendererProtocol(RendererProtocol, Protocol):
    """Optional extension for replacing GPU scene data in place."""

    def replace_scene(self, scene) -> None:
        """Make ``scene`` resident while retaining renderer initialization."""
        ...


@runtime_checkable
class ObjectEffectRendererProtocol(RendererProtocol, Protocol):
    """Optional extension for transient renderer-side object effects."""

    def apply_object_effect(self, scene, reference, effect) -> None:
        """Apply ``effect`` to one object identified within ``scene``."""
        ...

    def clear_object_effect(self) -> None:
        """Remove the active transient object effect."""
        ...


@runtime_checkable
class MultiObjectEffectRendererProtocol(ObjectEffectRendererProtocol, Protocol):
    """Optional extension for multiple simultaneous renderer-side effects."""

    def set_object_effects(self, scene, bindings) -> None:
        """Replace the ordered collection of active object effects."""
        ...


@runtime_checkable
class PickRendererProtocol(RendererProtocol, Protocol):
    """Optional extension for accelerated asynchronous scene picking."""

    def pick(self, scene, camera, viewport_size, pixel, *, options, mapping=None):
        """Return the closest policy-compatible pick result, if any."""
        ...


__all__ = [
    "GraphicsApi", "RendererFamily", "RendererImplementation",
    "RendererImplementationInfo",
    "GpuRendererProtocol", "MultiObjectEffectRendererProtocol",
    "ObjectEffectRendererProtocol", "PickRendererProtocol",
    "ProductRendererProtocol", "RendererProtocol",
    "ResidentSceneRendererProtocol",
]
