"""Native presenter bindings for the reusable reconstruction component."""

import vulkan as vk

from ...pipeline.graph import VulkanGraph
from ...runtime.reconstruction import VulkanReconstruction
from .denoiser_graph import _Binding, _QueueCompletion


class NativeReconstructionGraph:
    def __init__(self, core):
        self.core, self.runtime = core, core.runtime
        self.executor = core.wavefront_executor
        self.fsr2 = (
            getattr(core, "fsr2", None)
            if core.config.wavefront_upscale_filter == "fsr2"
            else None
        )
        self.closed = self.idle = False
        self.stages = []
        self.recordings = [None, None]
        self.key = (self.executor, self.fsr2)
        with self.runtime.lock:
            before = set(self.runtime._consumers)
            try:
                self._initialize()
            except Exception:
                self.close()
                raise
            self.consumers = set(self.runtime._consumers) - before

    def require_open(self):
        self.runtime.require_open()
        if self.closed:
            raise RuntimeError("Native reconstruction bindings have been retired")

    def _image(self, image, view, extent, format):
        return _Binding(
            runtime=self.runtime,
            image=image,
            view=view,
            width=extent[0],
            height=extent[1],
            format=format,
            layout=vk.VK_IMAGE_LAYOUT_GENERAL,
            usage=vk.VK_IMAGE_USAGE_STORAGE_BIT,
            require_open=self.require_open,
        )

    def _initialize(self):
        for slot, frame in enumerate(self.core.window_frames):
            extent = frame["wavefront_allocation_extent"]
            output_extent = self.core.swapchain_extent

            def resource(frame, name, fmt, size=extent):
                return self._image(
                    frame[name + "_image"], frame[name + "_view"], size, fmt
                )

            previous = self.core.window_frames[1 - slot]
            hdr = resource(frame, "wavefront_hdr", vk.VK_FORMAT_R16G16B16A16_SFLOAT)
            if self.fsr2 is not None:
                hdr = self.fsr2.stage.outputs[slot]
            if self.core.swapchain_direct_storage:
                fmt = (
                    vk.VK_FORMAT_B8G8R8A8_UNORM
                    if self.core.swapchain_bgra_storage
                    else vk.VK_FORMAT_R8G8B8A8_UNORM
                )
                outputs = [
                    self._image(image, view, output_extent, fmt)
                    for image, view in zip(
                        self.core.swapchain_images, self.core.swapchain_image_views
                    )
                ]
            else:
                outputs = [
                    self._image(
                        frame["image"],
                        frame["image_view"],
                        output_extent,
                        vk.VK_FORMAT_R8G8B8A8_UNORM,
                    )
                ]
            cameras = []
            for i in (1 - slot, slot):
                buffer = self.executor.camera_buffers[i]
                cameras.append(
                    _Binding(
                        runtime=self.runtime,
                        buffer=buffer.buffer,
                        byte_size=buffer.size,
                        usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        require_open=self.require_open,
                    )
                )
            history_extent = (
                output_extent
                if (
                    self.core.config.wavefront_temporal_reconstruction
                    or self.core.config.stationary_accumulation
                )
                else (1, 1)
            )
            self.stages.append(
                VulkanReconstruction(
                    self.runtime,
                    hdr=hdr,
                    position=resource(
                        frame, "wavefront_position", vk.VK_FORMAT_R32_SFLOAT
                    ),
                    normal=resource(frame, "wavefront_normal", vk.VK_FORMAT_R32_UINT),
                    previous_color=resource(
                        previous,
                        "wavefront_history_color",
                        vk.VK_FORMAT_B10G11R11_UFLOAT_PACK32,
                        history_extent,
                    ),
                    previous_position=resource(
                        previous, "wavefront_position", vk.VK_FORMAT_R32_SFLOAT
                    ),
                    previous_normal=resource(
                        previous, "wavefront_normal", vk.VK_FORMAT_R32_UINT
                    ),
                    history_color=resource(
                        frame,
                        "wavefront_history_color",
                        vk.VK_FORMAT_B10G11R11_UFLOAT_PACK32,
                        history_extent,
                    ),
                    outputs=outputs,
                    previous_camera=cameras[0],
                    current_camera=cameras[1],
                    material=resource(
                        frame, "wavefront_material", vk.VK_FORMAT_R32_UINT
                    ),
                )
            )

    def record(self, command, slot, extent, settings, history_valid, effects):
        with self.runtime.lock:
            self.require_open()
            graph = VulkanGraph().add(
                "reconstruct",
                self.stages[slot].operation(
                    extent=extent,
                    settings=settings,
                    history_valid=history_valid,
                    effects=effects,
                    external_output=self.core.swapchain_direct_storage,
                ),
            )
            recording = graph.compile().prepare_recording(self.runtime)
            recording.record(command)
            self.recordings[slot] = recording

    def submitted(self, slot):
        if self.recordings[slot] is not None:
            self.recordings[slot].submitted(_QueueCompletion(self))

    def close(self):
        if self.closed:
            return
        with self.runtime.lock:
            try:
                vk.vkDeviceWaitIdle(self.runtime.device)
            except vk.VkErrorDeviceLost:
                pass
            self.idle = True
            for stage in reversed(self.stages):
                stage.close()
            self.recordings = [None, None]
            self.closed = True
