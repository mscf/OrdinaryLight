"""Tone mapping and presentation of application-produced HDR, without GI."""

from dataclasses import dataclass
from importlib.resources import files
import math
import struct

import vulkan as vk

from .kernel import VulkanKernel
from .resources import VulkanImage
from ..pipeline.vulkan import (
    VulkanPass,
    VulkanPassPipeline,
    VulkanResource,
    VulkanResourceUse,
)


def _use(image, stage, access, layout):
    return VulkanResourceUse(VulkanResource.image(image), stage, access, layout)


@dataclass
class VulkanOutputFrame:
    """Owned RGBA8 image with completion; close when consumers finish."""

    image: VulkanImage
    completion: object
    kernel: VulkanKernel

    def wait(self):
        self.completion.wait()
        return self

    def close(self):
        self.wait()
        self.kernel.close()
        self.image.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


class VulkanOutput:
    """Reusable output stage accepting linear RGBA32F storage images.

    ``after`` is mandatory: pass the producer's completion. No tracing,
    accumulation, scene upload, or GI initialization occurs on this path.
    """

    def __init__(self, runtime):
        runtime.require_open()
        self.runtime = runtime
        self.swapchain = None
        self.extent = None
        self.closed = False
        runtime.retain(self)

    def _require_open(self):
        self.runtime.require_open()
        if self.closed:
            raise RuntimeError("Vulkan output is closed")

    def tone_map(self, hdr, *, after, exposure=1.0, exportable=False):
        self._require_open()
        hdr.require_open()
        if hdr.runtime is not self.runtime or after.runtime is not self.runtime:
            raise ValueError("HDR and completion must belong to the output runtime")
        if (
            hdr.format != vk.VK_FORMAT_R32G32B32A32_SFLOAT
            or not hdr.usage & vk.VK_IMAGE_USAGE_STORAGE_BIT
        ):
            raise ValueError("HDR input must be an RGBA32F storage image")
        if not math.isfinite(exposure) or exposure < 0:
            raise ValueError("exposure must be finite and nonnegative")
        output = self.runtime.image(
            hdr.width,
            hdr.height,
            format=vk.VK_FORMAT_R8G8B8A8_UNORM,
            exportable=exportable,
        )
        kernel = None
        try:
            kernel = VulkanKernel(
                self.runtime,
                files("ordinarylight.shaders")
                .joinpath("external_hdr_tone_map.comp.spv")
                .read_bytes(),
                {0: VulkanResource.image(hdr), 1: VulkanResource.image(output)},
                push_constant_size=4,
            )
            uses = (
                _use(
                    hdr,
                    vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    vk.VK_ACCESS_SHADER_READ_BIT,
                    vk.VK_IMAGE_LAYOUT_GENERAL,
                ),
                _use(
                    output,
                    vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    vk.VK_ACCESS_SHADER_WRITE_BIT,
                    vk.VK_IMAGE_LAYOUT_GENERAL,
                ),
            )
            completion = VulkanPassPipeline(
                [
                    VulkanPass(
                        "tone_map",
                        uses,
                        lambda command: kernel.bind(
                            command, struct.pack("f", exposure)
                        ),
                        ((hdr.width + 7) // 8, (hdr.height + 7) // 8, 1),
                    )
                ]
            ).execute(self.runtime, after=(after,))
            return VulkanOutputFrame(output, completion, kernel)
        except Exception:
            if kernel is not None:
                kernel.close()
            output.close()
            raise

    def export(self, hdr, *, after, exposure=1.0):
        """Return an RGBA8 GpuFrame with opaque memory/ready-semaphore FDs.

        External consumers must finish before frame.close(). This initial output
        service does not offer NV12/P010 conversion or release semaphores.
        """
        from .export import export_frame

        frame = self.tone_map(hdr, after=after, exposure=exposure, exportable=True)
        try:
            return export_frame(frame)
        except Exception:
            frame.close()
            raise

    def read(self, frame):
        """Explicit diagnostic readback of an output frame; never used by present."""
        self._require_open()
        image = frame.image
        with self.runtime.buffer(image.width * image.height * 4) as buffer:
            uses = (
                _use(
                    image,
                    vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                    vk.VK_ACCESS_TRANSFER_READ_BIT,
                    vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                ),
                VulkanResourceUse(
                    VulkanResource.buffer(buffer),
                    vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                    vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                ),
            )

            def copy(command):
                vk.vkCmdCopyImageToBuffer(
                    command,
                    image.image,
                    vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    buffer.buffer,
                    1,
                    [
                        vk.VkBufferImageCopy(
                            imageSubresource=vk.VkImageSubresourceLayers(
                                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, layerCount=1
                            ),
                            imageExtent=vk.VkExtent3D(image.width, image.height, 1),
                        )
                    ],
                )

            VulkanPassPipeline([VulkanPass("readback", uses, copy)]).execute(
                self.runtime, after=(frame.completion,)
            ).wait()
            return buffer.read()

    def _ensure_swapchain(self, width, height):
        r = self.runtime
        if r.surface is None:
            raise RuntimeError("Presentation requires a runtime created with a surface")
        caps = r.get_surface_capabilities(r.physical_device, r.surface)
        if not caps.supportedUsageFlags & vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT:
            raise RuntimeError("Surface does not support transfer destination images")
        extent = (int(caps.currentExtent.width), int(caps.currentExtent.height))
        if extent[0] == 0xFFFFFFFF:
            extent = (
                max(caps.minImageExtent.width, min(width, caps.maxImageExtent.width)),
                max(
                    caps.minImageExtent.height, min(height, caps.maxImageExtent.height)
                ),
            )
        if min(extent) <= 0:
            return False
        if self.swapchain is not None and extent == self.extent:
            return True
        formats = list(r.get_surface_formats(r.physical_device, r.surface))
        chosen = next(
            (
                f
                for f in formats
                if f.format
                in (vk.VK_FORMAT_B8G8R8A8_UNORM, vk.VK_FORMAT_R8G8B8A8_UNORM)
                and f.colorSpace == vk.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
            ),
            None,
        )
        if chosen is None:
            raise RuntimeError(
                "Output presentation requires an RGBA/BGRA UNORM sRGB-nonlinear surface"
            )
        for fmt, required in (
            (vk.VK_FORMAT_R8G8B8A8_UNORM, vk.VK_FORMAT_FEATURE_BLIT_SRC_BIT),
            (chosen.format, vk.VK_FORMAT_FEATURE_BLIT_DST_BIT),
        ):
            if (
                not vk.vkGetPhysicalDeviceFormatProperties(
                    r.physical_device, fmt
                ).optimalTilingFeatures
                & required
            ):
                raise RuntimeError("Surface format does not support GPU blitting")
        count = max(2, caps.minImageCount)
        if caps.maxImageCount:
            count = min(count, caps.maxImageCount)
        alpha = next(
            bit
            for bit in (
                vk.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                vk.VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
                vk.VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
                vk.VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
            )
            if caps.supportedCompositeAlpha & bit
        )
        vk.vkDeviceWaitIdle(r.device)
        replacement = r.create_swapchain(
            r.device,
            vk.VkSwapchainCreateInfoKHR(
                surface=r.surface,
                minImageCount=count,
                imageFormat=chosen.format,
                imageColorSpace=chosen.colorSpace,
                imageExtent=vk.VkExtent2D(*extent),
                imageArrayLayers=1,
                imageUsage=vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                imageSharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
                preTransform=caps.currentTransform,
                compositeAlpha=alpha,
                presentMode=vk.VK_PRESENT_MODE_FIFO_KHR,
                clipped=vk.VK_TRUE,
                oldSwapchain=self.swapchain or vk.VK_NULL_HANDLE,
            ),
            None,
        )
        if self.swapchain is not None:
            r.destroy_swapchain(r.device, self.swapchain, None)
        self.swapchain = replacement
        self.extent = extent
        self.images = list(r.get_swapchain_images(r.device, replacement))
        return True

    def present(self, hdr, *, after, exposure=1.0, surface_size=None):
        """Tone-map and GPU-blit to the runtime surface; synchronous first version."""
        self._require_open()
        with self.runtime.lock:
            if not self._ensure_swapchain(*(surface_size or (hdr.width, hdr.height))):
                return False
            with self.tone_map(hdr, after=after, exposure=exposure) as frame:
                r = self.runtime
                acquired = vk.vkCreateFence(r.device, vk.VkFenceCreateInfo(), None)
                try:
                    try:
                        index = r.acquire_next_image(
                            r.device,
                            self.swapchain,
                            (1 << 64) - 1,
                            vk.VK_NULL_HANDLE,
                            acquired,
                        )
                    except (vk.VkErrorOutOfDateKhr, vk.VkSuboptimalKhr):
                        vk.vkDeviceWaitIdle(r.device)
                        r.destroy_swapchain(r.device, self.swapchain, None)
                        self.swapchain = None
                        return False
                    vk.vkWaitForFences(
                        r.device, 1, [acquired], vk.VK_TRUE, (1 << 64) - 1
                    )
                    target = self.images[index]

                    def copy(command):
                        sub = vk.VkImageSubresourceRange(
                            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                            levelCount=1,
                            layerCount=1,
                        )

                        def barrier(old, new, src, dst):
                            vk.vkCmdPipelineBarrier(
                                command,
                                vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                0,
                                0,
                                None,
                                0,
                                None,
                                1,
                                [
                                    vk.VkImageMemoryBarrier(
                                        srcAccessMask=src,
                                        dstAccessMask=dst,
                                        oldLayout=old,
                                        newLayout=new,
                                        srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                                        dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                                        image=target,
                                        subresourceRange=sub,
                                    )
                                ],
                            )

                        barrier(
                            vk.VK_IMAGE_LAYOUT_UNDEFINED,
                            vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            0,
                            vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                        )
                        layers = vk.VkImageSubresourceLayers(
                            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, layerCount=1
                        )
                        vk.vkCmdBlitImage(
                            command,
                            frame.image.image,
                            vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            target,
                            vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            1,
                            [
                                vk.VkImageBlit(
                                    srcSubresource=layers,
                                    srcOffsets=[
                                        vk.VkOffset3D(0, 0, 0),
                                        vk.VkOffset3D(hdr.width, hdr.height, 1),
                                    ],
                                    dstSubresource=layers,
                                    dstOffsets=[
                                        vk.VkOffset3D(0, 0, 0),
                                        vk.VkOffset3D(*self.extent, 1),
                                    ],
                                )
                            ],
                            vk.VK_FILTER_NEAREST,
                        )
                        barrier(
                            vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                            vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                            0,
                        )

                    VulkanPassPipeline(
                        [
                            VulkanPass(
                                "present_copy",
                                (
                                    _use(
                                        frame.image,
                                        vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                                        vk.VK_ACCESS_TRANSFER_READ_BIT,
                                        vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    ),
                                ),
                                copy,
                            )
                        ]
                    ).execute(r, after=(frame.completion,)).wait()
                    try:
                        r.queue_present(
                            r.queue,
                            vk.VkPresentInfoKHR(
                                swapchainCount=1,
                                pSwapchains=[self.swapchain],
                                pImageIndices=[index],
                            ),
                        )
                    except (vk.VkErrorOutOfDateKhr, vk.VkSuboptimalKhr):
                        vk.vkDeviceWaitIdle(r.device)
                        r.destroy_swapchain(r.device, self.swapchain, None)
                        self.swapchain = None
                    vk.vkQueueWaitIdle(r.queue)
                    return True
                finally:
                    vk.vkDestroyFence(r.device, acquired, None)

    def close(self):
        with self.runtime.lock:
            if self.closed:
                return
            vk.vkDeviceWaitIdle(self.runtime.device)
            if self.swapchain is not None:
                self.runtime.destroy_swapchain(
                    self.runtime.device, self.swapchain, None
                )
            self.closed = True
            self.runtime.release(self)

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_exc):
        self.close()
