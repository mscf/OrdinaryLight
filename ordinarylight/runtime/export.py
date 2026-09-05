"""Opaque-FD handoff for the independent output service."""

import vulkan as vk
from ..gpu import GpuFrame, VulkanImageMetadata


def export_frame(frame):
    frame.wait()
    image = frame.image
    r = image.runtime
    semaphore = vk.vkCreateSemaphore(
        r.device,
        vk.VkSemaphoreCreateInfo(
            pNext=vk.VkExportSemaphoreCreateInfo(
                handleTypes=vk.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT
            )
        ),
        None,
    )
    fence = None
    command = None
    try:
        fence = vk.vkCreateFence(r.device, vk.VkFenceCreateInfo(), None)
        command = vk.vkAllocateCommandBuffers(
            r.device,
            vk.VkCommandBufferAllocateInfo(
                commandPool=r.command_pool,
                level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1,
            ),
        )[0]
        vk.vkBeginCommandBuffer(
            command,
            vk.VkCommandBufferBeginInfo(
                flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
            ),
        )
        vk.vkCmdPipelineBarrier(
            command,
            vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0,
            0,
            None,
            0,
            None,
            1,
            [
                vk.VkImageMemoryBarrier(
                    srcAccessMask=vk.VK_ACCESS_MEMORY_WRITE_BIT,
                    dstAccessMask=0,
                    oldLayout=image.layout,
                    newLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
                    srcQueueFamilyIndex=r.queue_family,
                    dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_EXTERNAL,
                    image=image.image,
                    subresourceRange=vk.VkImageSubresourceRange(
                        aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                        levelCount=1,
                        layerCount=1,
                    ),
                )
            ],
        )
        vk.vkEndCommandBuffer(command)
        vk.vkQueueSubmit(
            r.queue,
            1,
            [
                vk.VkSubmitInfo(
                    commandBufferCount=1,
                    pCommandBuffers=[command],
                    signalSemaphoreCount=1,
                    pSignalSemaphores=[semaphore],
                )
            ],
            fence,
        )
    except Exception:
        if command is not None:
            vk.vkFreeCommandBuffers(r.device, r.command_pool, 1, [command])
        if fence is not None:
            vk.vkDestroyFence(r.device, fence, None)
        vk.vkDestroySemaphore(r.device, semaphore, None)
        raise

    def export_fd(memory):
        info = (
            vk.VkMemoryGetFdInfoKHR(
                memory=image.memory,
                handleType=vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
            )
            if memory
            else vk.VkSemaphoreGetFdInfoKHR(
                semaphore=semaphore,
                handleType=vk.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
            )
        )
        fd = vk.ffi.new("int *")
        function = r.get_memory_fd if memory else r.get_semaphore_fd
        result = function(r.device, vk.ffi.addressof(info), fd)
        if result != vk.VK_SUCCESS:
            raise RuntimeError(f"External FD export failed: {result}")
        return fd[0]

    def wait(timeout=None):
        try:
            vk.vkWaitForFences(
                r.device,
                1,
                [fence],
                vk.VK_TRUE,
                (1 << 64) - 1 if timeout is None else max(0, int(timeout * 1e9)),
            )
        except vk.VkTimeout:
            return False
        return True

    def close(*_args):
        with r.lock:
            wait()
            vk.vkFreeCommandBuffers(r.device, r.command_pool, 1, [command])
            vk.vkDestroyFence(r.device, fence, None)
            vk.vkDestroySemaphore(r.device, semaphore, None)
            frame.close()

    def handle(value):
        return int(vk.ffi.cast("uintptr_t", value))

    metadata = VulkanImageMetadata(
        width=image.width,
        height=image.height,
        format="VK_FORMAT_R8G8B8A8_UNORM",
        format_value=image.format,
        layout="VK_IMAGE_LAYOUT_GENERAL",
        memory_size=image.memory_size,
        memory_offset=0,
        dedicated_allocation=True,
        device_uuid=r.device_uuid,
        image_handle=handle(image.image),
        memory_handle=handle(image.memory),
        device_handle=handle(r.device),
        physical_device_handle=handle(r.physical_device),
        completion_fence_handle=handle(fence),
        queue_family_index=r.queue_family,
    )
    return GpuFrame(
        api="vulkan",
        metadata=metadata,
        export_memory_fd=lambda: export_fd(True),
        export_ready_semaphore_fd=lambda: export_fd(False),
        wait=wait,
        close=close,
        attributes={"source": "external_hdr", "color_space": "srgb"},
    )
