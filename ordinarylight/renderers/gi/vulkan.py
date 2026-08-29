"""Vulkan global-illumination renderer and presentation support."""

from ...targets.vulkan.api import (
    RendererConfig, VulkanDeviceInfo, VulkanGlfwPresenter,
    _VulkanGlobalIlluminationEngine,
    VulkanSurfacePresenter, probe_vulkan_devices,
)
from ..base import RendererImplementation, RendererImplementationInfo


class VulkanGlobalIlluminationRenderer(
    _VulkanGlobalIlluminationEngine, RendererImplementation,
):
    """Hardware ray-traced Vulkan global-illumination renderer."""

    implementation = RendererImplementationInfo(
        name="vulkan-global-illumination",
        family="global_illumination",
        graphics_api="vulkan",
    )


__all__ = [
    "RendererConfig", "VulkanDeviceInfo", "VulkanGlfwPresenter",
    "VulkanGlobalIlluminationRenderer", "VulkanSurfacePresenter",
    "probe_vulkan_devices",
]
