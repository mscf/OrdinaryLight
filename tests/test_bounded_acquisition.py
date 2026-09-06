"""Deterministic WSI timeout tests without a live compositor."""

from threading import RLock
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import vulkan as vk

from ordinarylight._presentation import acquire_image, DEFAULT_ACQUIRE_TIMEOUT_NS
from ordinarylight.targets.vulkan.api import RendererConfig
from ordinarylight.raster import RasterConfig
from ordinarylight.runtime.output import VulkanOutput
from ordinarylight.renderers.raster.vulkan import VulkanRasterRenderer


@pytest.mark.parametrize("factory", [RendererConfig, RasterConfig])
def test_timeout_configuration(factory):
    assert factory().acquire_timeout_ns == DEFAULT_ACQUIRE_TIMEOUT_NS
    assert factory(acquire_timeout_ns=0).acquire_timeout_ns == 0
    for value in [-1, (1 << 64) - 1, 1 << 64, 1.5, float("inf"), None, True]:
        with pytest.raises(ValueError, match="acquire_timeout_ns"):
            factory(acquire_timeout_ns=value)


@pytest.mark.parametrize("timeout", [0, 123456])
def test_acquisition_preserves_index_zero_and_error_recovery(timeout):
    acquire = Mock(return_value=0)
    assert acquire_image(acquire, "device", "swapchain", "semaphore", timeout) == 0
    acquire.assert_called_once_with(
        "device", "swapchain", timeout, "semaphore", vk.VK_NULL_HANDLE
    )
    for error in [vk.VkSuboptimalKhr, vk.VkErrorOutOfDateKhr, vk.VkErrorDeviceLost]:
        acquire.side_effect = error
        with pytest.raises(error):
            acquire_image(acquire, "device", "swapchain", "semaphore", timeout)


@pytest.mark.parametrize("error", [vk.VkTimeout, vk.VkNotReady])
def test_raster_timeout_does_not_submit_reset_or_advance_slot(error):
    renderer = object.__new__(VulkanRasterRenderer)
    renderer.vk = SimpleNamespace(VK_TRUE=vk.VK_TRUE, vkWaitForFences=Mock())
    renderer.device = "device"
    renderer.config = RasterConfig(acquire_timeout_ns=42)
    renderer._ensure_swapchain = Mock()
    renderer._swapchain = "swapchain"
    renderer._present_frames = [{"fence": "fence", "image_available": "semaphore"}]
    renderer._present_frame_index = 0
    renderer._acquire_next_image = Mock(side_effect=error)
    renderer._render_finished_for_image = Mock(
        side_effect=AssertionError("used unacquired image")
    )
    # No recording, submission, or fence-reset entry points are provided.
    for _ in range(2):
        assert renderer.render(None, 32, 32, present=True) is None
        assert renderer._present_frame_index == 0
    assert renderer._acquire_next_image.call_args.args[2] == 42
    renderer._render_finished_for_image.assert_not_called()


@pytest.mark.parametrize("error", [vk.VkTimeout, vk.VkNotReady])
def test_output_timeout_keeps_acquisition_slot_reusable(error):
    runtime = SimpleNamespace(
        lock=RLock(),
        require_open=Mock(),
        retain=Mock(),
        device="device",
        acquire_next_image=Mock(side_effect=error),
    )
    output = VulkanOutput(runtime, acquire_timeout_ns=0)
    output._ensure_swapchain = Mock(return_value=True)
    output.swapchain = "swapchain"
    semaphore = SimpleNamespace(handle="semaphore")
    output._acquire_slots = [[semaphore, None]]
    output._drop_swapchain = Mock(side_effect=AssertionError("unexpected recreation"))
    frame = SimpleNamespace(
        image=SimpleNamespace(
            runtime=runtime,
            require_open=Mock(),
            format=vk.VK_FORMAT_R8G8B8A8_UNORM,
            width=32,
            height=32,
        )
    )
    for _ in range(2):
        assert output.present_operation(frame) is None
        assert output._pending_acquire is None
        assert output._present_next == 0
    assert runtime.acquire_next_image.call_args.args[2] == 0
    frame.hdr = frame.image
    output._presentation_target = frame
    assert output.present(frame.image, after=None) is False
    output._drop_swapchain.assert_not_called()


def test_output_rejects_infinite_timeout_before_retaining_runtime():
    runtime = SimpleNamespace(retain=Mock())
    with pytest.raises(ValueError, match="acquire_timeout_ns"):
        VulkanOutput(runtime, acquire_timeout_ns=(1 << 64) - 1)
    runtime.retain.assert_not_called()
