"""Native GI preparation and history on an application-owned Vulkan runtime."""

from types import MappingProxyType
from operator import index
from dataclasses import replace

import vulkan as vk

from ..pipeline.graph import VulkanGraph, VulkanOperation
from ..pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse
from ..targets.vulkan.api import RendererConfig, VulkanGlfwPresenter
from ..targets.vulkan.gi_images import native_gi_images, native_gi_buffers


class PreparedGiFrame:
    """One frame advance, recorded once and committed after graph submission.

    Add ``operation`` to an application VulkanGraph. ``images`` are resident GPU
    outputs, valid until this slot is reused or the pipeline is resized/closed.
    Submit all consumers on the same runtime queue before preparing another use
    of this slot. Cancel unsubmitted frames explicitly. Never execute an old
    operation again: prepare a new frame; native commands are reused internally.
    """

    def __init__(self, owner, generator, packet):
        self.owner, self.runtime = owner, owner.runtime
        self._generator, self._packet = generator, packet
        self.slot = packet["slot"]
        self.render_extent, self.output_extent = packet["render_extent"], packet["output_extent"]
        self.images = MappingProxyType(dict(native_gi_images(owner._core, self.slot)))
        self.buffers = native_gi_buffers(owner._core, self.slot)
        self.sample_count = packet["sample_count"]
        self.completion = None
        self._recorded = self._finished = False
        uses = []
        uses.extend(VulkanResourceUse(
            VulkanResource.buffer(buffer), vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            vk.VK_ACCESS_SHADER_WRITE_BIT,
        ) for buffer in self.buffers.values())
        for images, access in (
            (self.images, vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT
             | vk.VK_ACCESS_TRANSFER_READ_BIT | vk.VK_ACCESS_TRANSFER_WRITE_BIT),
            (native_gi_images(owner._core, 1 - self.slot), vk.VK_ACCESS_SHADER_READ_BIT),
        ):
            uses.extend(VulkanResourceUse(
                VulkanResource.image(image), vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                access,
                vk.VK_IMAGE_LAYOUT_GENERAL,
            ) for image in images.values())
        resident = owner.resident
        if owner.config.geometry_resources is not None:
            uses.extend(owner.config.geometry_resources.uses)
        if owner.config.material_resources is not None:
            uses.extend(owner.config.material_resources.uses)
        for name in ("tlas", *(n for n, b in resident.bindings.items() if b is not None)):
            uses.append(VulkanResourceUse(
                resident.resource(name), vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                (vk.VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR if name == "tlas"
                 else vk.VK_ACCESS_SHADER_READ_BIT),
            ))
        for kind in ("textures", "volumes"):
            for image, sampler in resident.sampled_resources(kind):
                uses.append(VulkanResourceUse(image, vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                               vk.VK_ACCESS_SHADER_READ_BIT,
                                               vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL))
                uses.append(VulkanResourceUse(sampler, vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                                               vk.VK_ACCESS_SHADER_READ_BIT))
        # Textures and application bindings may alias the same resource. Vulkan
        # hazards describe allocations, not descriptor slots.
        merged = {}
        for use in uses:
            key = use.resource.kind, use.resource.handle
            old = merged.get(key)
            if old is not None:
                if old.layout != use.layout:
                    raise ValueError("Aliased GI images require identical layouts")
                resource = use.resource
                if resource.kind == "buffer":
                    lo = min(resource.offset, old.resource.offset)
                    hi = max(resource.offset + resource.size,
                             old.resource.offset + old.resource.size)
                    resource = replace(resource, offset=lo, size=hi - lo)
                use = replace(use, resource=resource, stage=use.stage | old.stage,
                              access=use.access | old.access)
            merged[key] = use
        self.operation = VulkanOperation(
            [VulkanPass("native_gi", tuple(merged.values()), self._record)],
            validate=self.require_open, submitted=self._submitted,
        )

    def require_open(self):
        self.owner.require_open()
        if self._finished or self.owner._pending is not self:
            raise RuntimeError("GI frame has already been submitted or cancelled")
        self.owner.resident.require_open()
        for image in self.images.values():
            image.require_open()
        for buffer in self.buffers.values():
            buffer.require_open()

    def _record(self, command):
        self.require_open()
        if self._recorded:
            raise RuntimeError("GI frame commands were already recorded")
        self._packet["record"](command)
        self._recorded = True

    def _submitted(self, completion):
        if not self._recorded or self._finished:
            raise RuntimeError("Record a fresh GI frame before publishing submission")
        if completion.runtime is not self.runtime:
            raise ValueError("GI completion must belong to its runtime")
        self.completion = completion
        self._finished = True
        self.owner._pending = None
        try:
            self._generator.send(completion)
        except StopIteration:
            pass

    def cancel(self):
        """Discard an unsubmitted frame; its next preparation rebuilds state."""
        if self._finished:
            return
        self._generator.close()
        self._finished = True
        self.owner._pending = None
        # Recording can initialize policy flags; they must not masquerade as
        # executed GPU initialization after cancellation.
        self.owner._core.swapchain_extent = None
        self.owner.invalidate_gi_history()


class VulkanWavefrontPipeline(VulkanGlfwPresenter):
    """Persistent native GI without a window, swapchain or display processing.

    The application owns runtime and resident scene. Preparation may allocate or
    wait at a frame-slot/resize boundary. Graph recording performs no allocation,
    readback, submission or CPU wait between GPU stages. OrdinaryLight's native
    transport, denoiser and history implementation is shared with presentation.
    """

    def __init__(self, runtime, resident, *, config=None):
        from ..targets.vulkan.core import VulkanRayQueryCore
        self._pending, self._core = None, None
        self.config = config or runtime.config
        if not isinstance(self.config, RendererConfig):
            raise TypeError("config must be RendererConfig")
        if self.config.external_image_interop or runtime._headless_surface:
            raise ValueError("Native graph GI uses same-device images, not external video export")
        if self.config.wavefront_upscale_filter == "fsr2":
            raise ValueError("Apply temporal upscaling after native GI as an application graph stage")
        if resident.runtime is not runtime:
            raise ValueError("Resident scene must belong to the supplied runtime")
        resident.require_open()
        try:
            self._core = VulkanRayQueryCore(runtime=runtime, config=self.config,
                                          offscreen_wavefront=True)
            self._core.use_scene_resources(resident)
            self.resident = resident
            self.device_name = self._core.device_name
        except BaseException:
            if self._core is not None:
                self._core.close()
            self._core = None
            raise

    def require_open(self):
        if self._core is None:
            raise RuntimeError("Native GI pipeline is closed")
        self.runtime.require_open()

    def _between_frames(self):
        self.require_open()
        if self._pending is not None:
            raise RuntimeError("Submit or cancel the prepared GI frame before changing pipeline state")

    def reconfigure(self, **changes):
        self._between_frames()
        return super().reconfigure(**changes)

    def invalidate_gi_commands(self):
        self._between_frames()
        return super().invalidate_gi_commands()

    def set_wavefront_restir_enabled(self, enabled):
        self._between_frames()
        return super().set_wavefront_restir_enabled(enabled)

    def prepare(self, camera, extent, *, render_extent=None):
        """Prepare one frame operation; output extent does not imply presentation."""
        self.require_open()
        extent = tuple(index(n) for n in extent)
        if len(extent) != 2 or min(extent) <= 0:
            raise ValueError("extent must contain two positive integers")
        with self.runtime.lock:
            if self._pending is not None:
                raise RuntimeError("Submit or cancel the prepared GI frame first")
            generator = self._core.prepare_wavefront_window(
                self.resident.scene, camera, *extent, render_extent=render_extent,
            )
            try:
                packet = next(generator)
                self._pending = PreparedGiFrame(self, generator, packet)
            except BaseException:
                generator.close()
                self._core.swapchain_extent = None
                raise
            return self._pending

    def render(self, camera, extent, *, render_extent=None):
        """Submit native lighting without display processing; returns GPU outputs."""
        with self.runtime.lock:
            frame = self.prepare(camera, extent, render_extent=render_extent)
            try:
                VulkanGraph().add("gi", frame.operation).compile().execute(self.runtime)
            except BaseException:
                frame.cancel()
                raise
            return frame

    def use_scene_resources(self, resident):
        self.require_open()
        if self._pending is not None:
            raise RuntimeError("Submit or cancel the prepared frame before rebinding")
        super().use_scene_resources(resident)
        self.resident = resident

    def close(self):
        if self._core is None:
            return
        with self.runtime.lock:
            if self._pending is not None:
                self._pending.cancel()
            for frame in self._core.window_frames:
                completion = frame.get("gi_external_completion")
                if completion is not None:
                    completion.wait()
            super().close()
