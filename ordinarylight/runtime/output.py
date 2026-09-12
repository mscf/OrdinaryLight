"""Tone mapping and presentation of application-produced HDR, without GI."""

from dataclasses import dataclass
from importlib.resources import files
from .._presentation import (
    acquire_image, DEFAULT_ACQUIRE_TIMEOUT_NS, validate_acquire_timeout,
)

import math
import struct
from operator import index

import vulkan as vk

from .blit import blit_operation
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


class VulkanToneMapTarget(VulkanOutputFrame):
    """Persistent tone-map image/pipeline; operations can join an application graph."""

    def __init__(self, runtime, hdr, *, exportable=False, extent=None):
        hdr.require_open()
        if (
            hdr.runtime is not runtime
            or hdr.format not in (vk.VK_FORMAT_R32G32B32A32_SFLOAT,
                                  vk.VK_FORMAT_R16G16B16A16_SFLOAT)
            or not hdr.usage & vk.VK_IMAGE_USAGE_STORAGE_BIT
        ):
            raise ValueError("Tone mapping requires same-runtime RGBA16F or RGBA32F storage HDR")
        extent = (hdr.width, hdr.height) if extent is None else tuple(map(index, extent))
        if (len(extent) != 2 or min(extent) < 1
                or extent[0] > hdr.width or extent[1] > hdr.height):
            raise ValueError("Tone-map extent must fit inside the HDR image")
        self.runtime, self.hdr = runtime, hdr
        self.closed = False
        self.completion = None
        self.kernel = None
        self.image = runtime.image(
            *extent,
            format=vk.VK_FORMAT_R8G8B8A8_UNORM,
            exportable=exportable,
        )
        try:
            self.kernel = VulkanKernel(
                runtime,
                files("ordinarylight.shaders")
                .joinpath("external_hdr_tone_map_16f.comp.spv"
                          if hdr.format == vk.VK_FORMAT_R16G16B16A16_SFLOAT
                          else "external_hdr_tone_map.comp.spv")
                .read_bytes(),
                {0: VulkanResource.image(hdr), 1: VulkanResource.image(self.image)},
                push_constant_size=4,
            )
        except Exception:
            self.image.close()
            raise

    def require_open(self):
        if self.closed:
            raise RuntimeError("Tone-map target is closed")
        self.kernel.require_open()

    def operation(self, *, exposure=1.0, after=()):
        from ..pipeline.graph import VulkanOperation

        self.require_open()
        if not math.isfinite(exposure) or exposure < 0:
            raise ValueError("Exposure must be finite and nonnegative")
        uses = (
            _use(
                self.hdr,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_ACCESS_SHADER_READ_BIT,
                vk.VK_IMAGE_LAYOUT_GENERAL,
            ),
            _use(
                self.image,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_ACCESS_SHADER_WRITE_BIT,
                vk.VK_IMAGE_LAYOUT_GENERAL,
            ),
        )
        return VulkanOperation(
            [
                VulkanPass(
                    "tone_map",
                    uses,
                    lambda command: self.kernel.bind(
                        command, struct.pack("f", exposure)
                    ),
                    ((self.image.width + 7) // 8, (self.image.height + 7) // 8, 1),
                )
            ],
            validate=self.require_open,
            dependencies=lambda: tuple(after)
            + ((self.completion,) if self.completion is not None else ()),
            submitted=lambda completion: setattr(self, "completion", completion),
        )

    def wait(self):
        if self.completion is not None:
            self.completion.wait()
        return self

    def close(self):
        with self.runtime.lock:
            if self.closed:
                return
            self.wait()
            self.kernel.close()
            self.image.close()
            self.closed = True


class _SwapchainImage:
    def __init__(self, output, image):
        self.output = output
        self.runtime = output.runtime
        self.image = image
        self.layout = vk.VK_IMAGE_LAYOUT_UNDEFINED
        self.width, self.height = output.extent
        self.format = output._swap_format
        self.usage = vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT
        self.generation = output._swap_generation

    def require_open(self):
        self.output._require_open()
        if self.generation != self.output._swap_generation:
            raise ValueError("Swapchain changed; acquire a new presentation operation")


class VulkanOutput:
    """Reusable output stage accepting linear RGBA16F/RGBA32F storage images.

    ``after`` is mandatory: pass the producer's completion. No tracing,
    accumulation, scene upload, or GI initialization occurs on this path.
    """

    def __init__(self, runtime, *, acquire_timeout_ns=DEFAULT_ACQUIRE_TIMEOUT_NS):
        self.acquire_timeout_ns = validate_acquire_timeout(acquire_timeout_ns)
        runtime.require_open()
        self.runtime = runtime
        self.swapchain = None
        self.extent = None
        self.closed = False
        self._presentation_target = None
        self._swap_generation = 0
        self._present_signals = []
        self._acquire_slots = []
        self._present_next = 0
        self._pending_acquire = None
        self._present_success = False
        runtime.retain(self)

    def _require_open(self):
        self.runtime.require_open()
        if self.closed:
            raise RuntimeError("Vulkan output is closed")

    def prepare(self, hdr, *, exportable=False, extent=None):
        self._require_open()
        return VulkanToneMapTarget(self.runtime, hdr, exportable=exportable, extent=extent)

    def tone_map(self, hdr, *, after, exposure=1.0, exportable=False):
        target = self.prepare(hdr, exportable=exportable)
        try:
            target.operation(exposure=exposure, after=(after,)).execute(self.runtime)
            return target
        except Exception:
            target.close()
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
        format_storage = r.get_surface_formats(r.physical_device, r.surface)
        formats = list(format_storage)
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
        from .resources import VulkanSemaphore

        for semaphore in self._present_signals:
            semaphore.close()
        self.swapchain = replacement
        self.extent = extent
        self._swap_format = chosen.format
        self.images = list(r.get_swapchain_images(r.device, replacement))
        self._swap_generation += 1
        self._swap_images = [_SwapchainImage(self, image) for image in self.images]
        self._present_signals = [VulkanSemaphore(r) for _ in self.images]
        if not self._acquire_slots:
            self._acquire_slots = [[VulkanSemaphore(r), None] for _ in range(2)]
        return True

    def _drop_swapchain(self):
        vk.vkDeviceWaitIdle(self.runtime.device)
        for semaphore in self._present_signals:
            semaphore.close()
        self._present_signals = []
        if self.swapchain is not None:
            self.runtime.destroy_swapchain(self.runtime.device, self.swapchain, None)
            self.swapchain = None
        self._swap_generation += 1

    def cancel_presentation(self):
        """Retire an acquired-but-unsubmitted operation after graph preparation fails."""
        with self.runtime.lock:
            if self._pending_acquire is not None:
                semaphore = self._pending_acquire
                self.runtime.submit(
                    lambda command: None,
                    wait_semaphores=[
                        (semaphore, vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT)
                    ],
                ).wait()
                self._pending_acquire = None
                self._drop_swapchain()

    def present_operation(self, frame, *, surface_size=None):
        """Acquire a swapchain image and return a single-use graph operation.

        Acquisition uses a binary semaphore and a finite timeout. Timeout/not-ready
        returns None without submission. Reusing an acquisition slot can also wait
        for its previous submission. The operation signals presentation on the GPU.
        Submit it or call cancel_presentation before requesting another.
        """
        from ..pipeline.graph import VulkanOperation

        with self.runtime.lock:
            self._require_open()
            if self._pending_acquire is not None:
                raise RuntimeError(
                    "Submit or cancel the acquired presentation operation"
                )
            source = frame.image
            source.require_open()
            if (
                source.runtime is not self.runtime
                or source.format != vk.VK_FORMAT_R8G8B8A8_UNORM
            ):
                raise ValueError(
                    "Presentation requires a same-runtime RGBA8 output image"
                )
            if not self._ensure_swapchain(
                *(surface_size or (source.width, source.height))
            ):
                return None
            slot = self._acquire_slots[self._present_next]
            if slot[1] is not None:
                slot[1].wait()
            acquired = slot[0]
            try:
                image_index = acquire_image(
                    self.runtime.acquire_next_image, self.runtime.device,
                    self.swapchain, acquired.handle, self.acquire_timeout_ns,
                )
                if image_index is None:
                    return None
            except vk.VkSuboptimalKhr:
                # Suboptimal acquisition still signals its semaphore. Consume
                # that signal before retiring this swapchain.
                self._pending_acquire = acquired
                self.cancel_presentation()
                return None
            except vk.VkErrorOutOfDateKhr:
                self._drop_swapchain()
                return None
            self._pending_acquire = acquired
            target = self._swap_images[image_index]
            finished = self._present_signals[image_index]
            used = False

            def validate():
                if used or self._pending_acquire is not acquired:
                    raise RuntimeError("Presentation operation is single-use")
                target.require_open()
                source.require_open()
                transfer.validate()

            def submitted(completion):
                nonlocal used
                used = True
                slot[1] = completion
                self._pending_acquire = None
                self._present_next = (self._present_next + 1) % len(self._acquire_slots)
                self._present_success = True
                try:
                    self.runtime.queue_present(
                        self.runtime.queue,
                        vk.VkPresentInfoKHR(
                            waitSemaphoreCount=1,
                            pWaitSemaphores=[finished.handle],
                            swapchainCount=1,
                            pSwapchains=[self.swapchain],
                            pImageIndices=[image_index],
                        ),
                    )
                except (vk.VkErrorOutOfDateKhr, vk.VkSuboptimalKhr):
                    self._present_success = False
                    self._drop_swapchain()

            transfer = blit_operation(source, target, present=True)
            return VulkanOperation(
                transfer.passes,
                validate=validate,
                submitted=submitted,
                dependencies=lambda: (
                    (frame.completion,) if frame.completion is not None else ()
                ),
                wait_semaphores=[(acquired, vk.VK_PIPELINE_STAGE_TRANSFER_BIT)],
                signal_semaphores=[finished],
            )

    def present(self, hdr, *, after, exposure=1.0, surface_size=None):
        """Reuse tone-map allocations and present through a single graph submission."""
        from ..pipeline.graph import VulkanGraph

        with self.runtime.lock:
            self._require_open()
            if (
                self._presentation_target is None
                or self._presentation_target.hdr is not hdr
            ):
                if self._presentation_target is not None:
                    self._presentation_target.close()
                self._presentation_target = self.prepare(hdr)
            frame = self._presentation_target
            operation = self.present_operation(frame, surface_size=surface_size)
            if operation is None:
                return False
            try:
                graph = VulkanGraph().add(
                    "tone_map", frame.operation(exposure=exposure)
                )
                graph.add("present", operation)
                graph.compile().execute(self.runtime, after=(after,))
                return self._present_success
            except Exception:
                self.cancel_presentation()
                raise

    def close(self):
        with self.runtime.lock:
            if self.closed:
                return
            self.cancel_presentation()
            vk.vkDeviceWaitIdle(self.runtime.device)
            if self._presentation_target is not None:
                self._presentation_target.close()
                self._presentation_target = None
            self._drop_swapchain()
            for semaphore, completion in self._acquire_slots:
                if completion is not None:
                    completion.wait()
                semaphore.close()
            self._acquire_slots = []
            self.closed = True
            self.runtime.release(self)

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_exc):
        self.close()
