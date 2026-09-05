"""Algorithm-independent GPU services (Vulkan loaded on demand)."""

_MODULES = {
    "VulkanRuntime": "vulkan",
    "VulkanCapabilities": "vulkan",
    "VulkanBuffer": "resources",
    "VulkanImage": "resources",
    "VulkanCompletion": "resources",
    "VulkanKernel": "kernel",
    "compile_compute": "kernel",
    "VulkanOutput": "output",
    "VulkanOutputFrame": "output",
}
__all__ = list(_MODULES)


def __getattr__(name):
    if name in _MODULES:
        from importlib import import_module

        return getattr(import_module(f"{__name__}.{_MODULES[name]}"), name)
    raise AttributeError(name)
