"""Shared Vulkan compatibility contract."""

VULKAN_API_VERSION = (1, 2, 0)


def vulkan_api_version(vk):
    """Return the encoded minimum Vulkan API version for an imported binding."""
    return vk.VK_MAKE_VERSION(*VULKAN_API_VERSION)


__all__ = ["VULKAN_API_VERSION", "vulkan_api_version"]
