"""Diagnostic readback of the non-direct RGBA reconstruction output."""

import numpy as np
import vulkan as vk


def read_output(core):
    frame = core.window_frames[(core.window_frame_index - 1) % 2]
    width, height = core.swapchain_extent
    specs = {"output": (np.uint8, 4)}
    images = {"output": frame["image"]}
    vk.vkWaitForFences(core.device, 1, [frame["fence"]], vk.VK_TRUE, (1 << 64) - 1)
    result = {}
    for name, (dtype, channels) in specs.items():
        image = images[name]
        size = width * height * channels * np.dtype(dtype).itemsize
        buf = core._create_buffer(
            size,
            vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )

        def record(command):
            barrier = core._image_barrier(
                image,
                vk.VK_IMAGE_LAYOUT_GENERAL,
                vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                vk.VK_ACCESS_SHADER_WRITE_BIT,
                vk.VK_ACCESS_TRANSFER_READ_BIT,
            )
            vk.vkCmdPipelineBarrier(
                command,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                None,
                0,
                None,
                1,
                [barrier],
            )
            region = vk.VkBufferImageCopy(
                bufferOffset=0,
                bufferRowLength=0,
                bufferImageHeight=0,
                imageSubresource=vk.VkImageSubresourceLayers(
                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    mipLevel=0,
                    baseArrayLayer=0,
                    layerCount=1,
                ),
                imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
                imageExtent=vk.VkExtent3D(width=width, height=height, depth=1),
            )
            vk.vkCmdCopyImageToBuffer(
                command,
                image,
                vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                buf.buffer,
                1,
                [region],
            )
            barrier = core._image_barrier(
                image,
                vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                vk.VK_IMAGE_LAYOUT_GENERAL,
                vk.VK_ACCESS_TRANSFER_READ_BIT,
                vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
            )
            vk.vkCmdPipelineBarrier(
                command,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0,
                None,
                0,
                None,
                1,
                [barrier],
            )

        try:
            core._single_use(record)
            mapped = vk.vkMapMemory(core.device, buf.memory, 0, size, 0)
            result[name] = (
                np.frombuffer(mapped, dtype=dtype)
                .copy()
                .reshape(height, width, channels)
            )
            vk.vkUnmapMemory(core.device, buf.memory)
        finally:
            vk.vkDestroyBuffer(core.device, buf.buffer, None)
            vk.vkFreeMemory(core.device, buf.memory, None)
            core._buffers.remove(buf)
    return result
