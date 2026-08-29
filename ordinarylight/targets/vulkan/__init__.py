"""Vulkan execution target, device discovery, and presentation."""

from ..base import ExecutionTargetInfo
from .api import (
    RendererConfig, VulkanDeviceInfo, VulkanGlfwPresenter,
    VulkanSurfacePresenter, probe_vulkan_devices,
)

info = ExecutionTargetInfo(name="vulkan", shader_format="spirv", gpu=True)

__all__ = [
    "RendererConfig", "VulkanDeviceInfo", "VulkanGlfwPresenter",
    "VulkanSurfacePresenter", "info", "probe_vulkan_devices",
]
