"""Canonical namespace for concrete Ordinary Light renderers."""

from . import hybrid, raster, reference
from .base import (
    GpuRendererProtocol, GraphicsApi, MultiObjectEffectRendererProtocol,
    ObjectEffectRendererProtocol, PickRendererProtocol,
    ProductRendererProtocol, RendererProtocol,
    RendererFamily, RendererImplementation, RendererImplementationInfo,
    ResidentSceneRendererProtocol,
)


def __getattr__(name):
    if name == "gi":
        from importlib import import_module

        module = import_module(f"{__name__}.gi")
        globals()[name] = module
        return module
    raise AttributeError(name)


__all__ = [
    "GpuRendererProtocol", "GraphicsApi", "MultiObjectEffectRendererProtocol",
    "ObjectEffectRendererProtocol", "PickRendererProtocol",
    "ProductRendererProtocol", "RendererProtocol", "RendererFamily",
    "RendererImplementation", "RendererImplementationInfo",
    "ResidentSceneRendererProtocol",
    "gi", "hybrid", "raster", "reference",
]
