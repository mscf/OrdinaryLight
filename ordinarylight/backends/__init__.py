"""Renderer backend contracts and implementations."""

from .base import (
    GpuRenderBackend, MultiObjectEffectBackend, ObjectEffectBackend,
    PickBackend, ProductRenderBackend,
    RenderBackend, ResidentSceneBackend,
)
from .reference import ReferenceBackend, ReferenceConfig


def __getattr__(name):
    if name == "vulkan":
        from importlib import import_module

        module = import_module(f"{__name__}.vulkan")
        globals()[name] = module
        return module
    if name in {"WebGpuRasterBackend"}:
        from .webgpu_raster import WebGpuRasterBackend
        globals()[name] = WebGpuRasterBackend
        return WebGpuRasterBackend
    if name == "VulkanRasterBackend":
        from .vulkan_raster import VulkanRasterBackend
        globals()[name] = VulkanRasterBackend
        return VulkanRasterBackend
    raise AttributeError(name)

__all__ = [
    "ProductRenderBackend",
    "GpuRenderBackend",
    "MultiObjectEffectBackend", "ObjectEffectBackend", "PickBackend",
    "ReferenceBackend",
    "ReferenceConfig",
    "RenderBackend",
    "ResidentSceneBackend",
    "vulkan", "VulkanRasterBackend", "WebGpuRasterBackend",
]
