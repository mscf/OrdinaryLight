"""GPU display-image transfer shared by native and application presentation."""

import vulkan as vk
from ..pipeline.graph import VulkanOperation
from ..pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse


def blit_operation(source, target, *, present=False, after=()):
    """Nearest blit of full RGBA8/BGRA8 UNORM images, with explicit hazards.

    Images belong to one runtime and remain caller-owned. Source returns to
    GENERAL; target ends in GENERAL, or PRESENT_SRC_KHR when present=True.
    The caller must acquire a presentation target and supply its acquire/signal
    semaphores to the enclosing graph submission. No acquisition or submission
    happens here. No tone mapping or transfer-function conversion is performed.
    """
    formats = (vk.VK_FORMAT_R8G8B8A8_UNORM, vk.VK_FORMAT_B8G8R8A8_UNORM)

    def validate():
        source.require_open()
        target.require_open()
        if source.runtime is not target.runtime:
            raise ValueError("Blit images must belong to the same runtime")
        if source.image == target.image:
            raise ValueError("Blit images must not alias")
        for image, usage in (
            (source, vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT),
            (target, vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT),
        ):
            if image.format not in formats or not image.usage & usage:
                raise ValueError(
                    "Blit requires RGBA8/BGRA8 UNORM images with transfer usage"
                )
            if min(image.width, image.height) <= 0:
                raise ValueError("Blit image dimensions must be positive")

    validate()
    for image, feature in (
        (source, vk.VK_FORMAT_FEATURE_BLIT_SRC_BIT),
        (target, vk.VK_FORMAT_FEATURE_BLIT_DST_BIT),
    ):
        properties = vk.vkGetPhysicalDeviceFormatProperties(
            source.runtime.physical_device, image.format
        )
        if not properties.optimalTilingFeatures & feature:
            raise ValueError("Image format does not support GPU blitting")

    def use(image, stage, access, layout):
        return VulkanResourceUse(VulkanResource.image(image), stage, access, layout)

    def record(command):
        layers = vk.VkImageSubresourceLayers(
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, layerCount=1
        )
        vk.vkCmdBlitImage(
            command,
            source.image,
            vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            target.image,
            vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            [
                vk.VkImageBlit(
                    srcSubresource=layers,
                    srcOffsets=[
                        vk.VkOffset3D(0, 0, 0),
                        vk.VkOffset3D(source.width, source.height, 1),
                    ],
                    dstSubresource=layers,
                    dstOffsets=[
                        vk.VkOffset3D(0, 0, 0),
                        vk.VkOffset3D(target.width, target.height, 1),
                    ],
                )
            ],
            vk.VK_FILTER_NEAREST,
        )

    after = tuple(after)
    return VulkanOperation(
        [
            VulkanPass(
                "blit",
                (
                    use(
                        source,
                        vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                        vk.VK_ACCESS_TRANSFER_READ_BIT,
                        vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    ),
                    use(
                        target,
                        vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                        vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                        vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    ),
                ),
                record,
            ),
            VulkanPass(
                "blit.finish",
                (
                    use(
                        source,
                        vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        vk.VK_ACCESS_SHADER_READ_BIT,
                        vk.VK_IMAGE_LAYOUT_GENERAL,
                    ),
                    use(
                        target,
                        vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT
                        if present
                        else vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0 if present else vk.VK_ACCESS_SHADER_READ_BIT,
                        vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
                        if present
                        else vk.VK_IMAGE_LAYOUT_GENERAL,
                    ),
                ),
                lambda command: None,
            ),
        ],
        validate=validate,
        dependencies=lambda: after,
    )
