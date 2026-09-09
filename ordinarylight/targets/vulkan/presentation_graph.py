"""Borrowed native display bindings for the common presentation blit."""

import vulkan as vk
from ...pipeline.graph import VulkanGraph
from ...runtime.blit import blit_operation
from .denoiser_graph import _Binding


def record_display_blit(core, command, frame, image_index, extent):
    """The parent owns acquisition, submission, image lifetime and slot fences."""
    width, height = extent
    source = _Binding(
        runtime=core.runtime,
        image=frame["image"],
        width=width,
        height=height,
        format=vk.VK_FORMAT_R8G8B8A8_UNORM,
        usage=vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
        layout=vk.VK_IMAGE_LAYOUT_GENERAL,
        require_open=core.runtime.require_open,
    )
    target = _Binding(
        runtime=core.runtime,
        image=core.swapchain_images[image_index],
        width=width,
        height=height,
        format=core.swapchain_format,
        usage=vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        layout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
        require_open=core.runtime.require_open,
    )
    recording = (
        VulkanGraph()
        .add("present", blit_operation(source, target, present=True))
        .compile()
        .prepare_recording(core.runtime)
    )
    recording.record(command)
    return recording
