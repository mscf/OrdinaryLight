"""Resource operations for the compact six-word indirect reservoir ABI."""

from operator import index
import vulkan as vk
from ..pipeline.graph import VulkanOperation
from ..pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse


def clear_indirect_reservoirs(buffer, *, count):
    """Zero count 24-byte reservoirs and make them ready for compute access.

    The caller owns the transfer-destination buffer and keeps it alive through
    submission completion. Any trailing allocation bytes remain untouched.
    """
    count = index(count)
    buffer.require_open()
    if not 0 <= count <= min(0xFFFFFFFF, buffer.byte_size // 24):
        raise ValueError("Reservoir count exceeds buffer storage")
    if not buffer.usage & vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT:
        raise ValueError("Reservoir clear requires transfer destination usage")
    resource = VulkanResource.buffer(buffer)

    def record(command):
        if count == 0:
            return
        size = count * 24
        vk.vkCmdFillBuffer(command, resource.handle, 0, size, 0)
        # Native consumers can be recorded outside this graph; publish the fill
        # to compute explicitly as well as declaring its graph resource access.
        barrier = vk.VkBufferMemoryBarrier(
            srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            buffer=resource.handle,
            offset=0,
            size=size,
        )
        vk.vkCmdPipelineBarrier(
            command,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            0,
            None,
            1,
            [barrier],
            0,
            None,
        )

    return VulkanOperation(
        [
            VulkanPass(
                "clear_reservoirs",
                (
                    VulkanResourceUse(
                        resource,
                        vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                        vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    ),
                ),
                record,
            )
        ],
        validate=buffer.require_open,
    )
