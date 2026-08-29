"""Raster renderer implementations grouped by graphics API."""


def __getattr__(name):
    if name == "VulkanRasterRenderer":
        from .vulkan import VulkanRasterRenderer
        globals()[name] = VulkanRasterRenderer
        return VulkanRasterRenderer
    if name == "WebGpuRasterRenderer":
        from .webgpu import WebGpuRasterRenderer
        globals()[name] = WebGpuRasterRenderer
        return WebGpuRasterRenderer
    raise AttributeError(name)


__all__ = ["VulkanRasterRenderer", "WebGpuRasterRenderer"]
