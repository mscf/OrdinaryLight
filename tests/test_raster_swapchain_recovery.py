"""Resize WSI results must preserve submission and semaphore lifetimes."""
from types import SimpleNamespace

import pytest
import vulkan as vk

from ordinarylight.renderers.raster.vulkan import VulkanRasterRenderer


def renderer():
    r = VulkanRasterRenderer.__new__(VulkanRasterRenderer)
    r.vk = vk
    r.device = r.queue = r._swapchain = vk.VK_NULL_HANDLE
    r._swapchain_extent = (1280, 720)
    r.config = SimpleNamespace(acquire_timeout_ns=16_000_000)
    return r


@pytest.mark.parametrize("result", [vk.VkErrorOutOfDateKhr, vk.VkSuboptimalKhr])
def test_present_resize_defers_recreation_until_next_frame(result):
    r = renderer()
    def present(*args):
        raise result()
    r._queue_present = present
    r._present_image(0, vk.VK_NULL_HANDLE)
    assert r._swapchain_recreate_pending
    assert r._swapchain_extent == (1280, 720)


def test_suboptimal_acquire_preserves_acquired_index_and_extent():
    r = renderer()
    def acquire(*args):
        args[-1][0] = 2
        raise vk.VkSuboptimalKhr()
    r._acquire_next_image = acquire
    assert r._acquire_present_image(vk.VK_NULL_HANDLE) == 2
    assert r._swapchain_recreate_pending
    assert r._swapchain_extent == (1280, 720)


@pytest.mark.parametrize("result", [vk.VkErrorOutOfDateKhr, vk.VkTimeout, vk.VkNotReady])
def test_unsuccessful_acquisition_skips_submission(result):
    r = renderer()
    def acquire(*args):
        raise result()
    r._acquire_next_image = acquire
    assert r._acquire_present_image(vk.VK_NULL_HANDLE) is None
    assert bool(getattr(r, '_swapchain_recreate_pending', False)) == (
        result is vk.VkErrorOutOfDateKhr
    )


def test_real_device_errors_are_not_hidden():
    r = renderer()
    def present(*args):
        raise vk.VkErrorDeviceLost()
    r._queue_present = present
    with pytest.raises(vk.VkErrorDeviceLost):
        r._present_image(0, vk.VK_NULL_HANDLE)
