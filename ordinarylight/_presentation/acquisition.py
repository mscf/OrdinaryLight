"""Shared bounded swapchain acquisition policy (timeouts are nanoseconds)."""

from operator import index

DEFAULT_ACQUIRE_TIMEOUT_NS = 16_000_000


def validate_acquire_timeout(value):
    try:
        timeout = index(value)
    except TypeError as exc:
        raise ValueError("acquire_timeout_ns must be a finite uint64 integer") from exc
    if isinstance(value, bool) or not 0 <= timeout < (1 << 64) - 1:
        raise ValueError("acquire_timeout_ns must be between 0 and UINT64_MAX - 1")
    return timeout


def acquire_image(acquire, device, swapchain, semaphore, timeout_ns):
    """Return an image index, or None when no image was acquired.

    Timeout/not-ready do not signal the acquisition semaphore. Callers must skip
    submission and leave their frame fence signaled so the slot can be retried.
    Other WSI results retain the caller's existing recovery/error handling.
    """
    import vulkan as vk

    timeout_ns = validate_acquire_timeout(timeout_ns)
    try:
        return acquire(device, swapchain, timeout_ns, semaphore, vk.VK_NULL_HANDLE)
    except (vk.VkTimeout, vk.VkNotReady):
        return None
