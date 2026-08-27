"""Renderer backend contracts and implementations."""

from .base import (
    GpuRenderBackend, ProductRenderBackend, RenderBackend,
    ResidentSceneBackend,
)
from .reference import ReferenceBackend, ReferenceConfig


def __getattr__(name):
    if name == "vulkan":
        from importlib import import_module

        module = import_module(f"{__name__}.vulkan")
        globals()[name] = module
        return module
    raise AttributeError(name)

__all__ = [
    "ProductRenderBackend",
    "GpuRenderBackend",
    "ReferenceBackend",
    "ReferenceConfig",
    "RenderBackend",
    "ResidentSceneBackend",
    "vulkan",
]
