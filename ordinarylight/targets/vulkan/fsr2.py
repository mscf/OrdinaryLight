"""Native frame adapter for the reusable FSR2 graph component."""

import math
import time
import vulkan as vk

from ...runtime._fsr2 import library as library
from ...runtime.fsr2 import VulkanFsr2
from ...pipeline.graph import VulkanGraph
from .denoiser_graph import _Binding, _QueueCompletion


class Fsr2:
    def __init__(self, core, extent, output):
        self.core, self.runtime = core, core.runtime
        self.extent, self.output = extent, output
        self.closed = self.idle = False
        self.stage = None
        self.pending = None
        self.last_time = time.monotonic()
        with self.runtime.lock:
            before = set(self.runtime._consumers)
            inputs = []
            for frame in core.window_frames:
                width, height = frame["wavefront_allocation_extent"]
                bindings = {}
                for name in ("hdr", "view_z", "motion", "normal_roughness"):
                    prefix = (
                        "wavefront_hdr" if name == "hdr" else "wavefront_relax_" + name
                    )
                    bindings[name] = _Binding(
                        runtime=self.runtime,
                        image=frame[prefix + "_image"],
                        view=frame[prefix + "_view"],
                        width=width,
                        height=height,
                        format=(
                            vk.VK_FORMAT_R32_SFLOAT
                            if name == "view_z"
                            else vk.VK_FORMAT_R16G16B16A16_SFLOAT
                        ),
                        usage=(
                            vk.VK_IMAGE_USAGE_SAMPLED_BIT
                            if name == "hdr"
                            else vk.VK_IMAGE_USAGE_STORAGE_BIT
                        ),
                        layout=vk.VK_IMAGE_LAYOUT_GENERAL,
                        require_open=self.require_open,
                    )
                inputs.append(bindings)
            self.stage = VulkanFsr2(
                self.runtime, inputs=inputs, extent=extent, output_extent=output
            )

            # Reconstruction is recorded separately before graph layout commits.
            # Establish its imported output layout once, before either recording.
            def initialize(command):
                barriers = [
                    core._image_barrier(
                        image.image,
                        vk.VK_IMAGE_LAYOUT_UNDEFINED,
                        vk.VK_IMAGE_LAYOUT_GENERAL,
                        0,
                        vk.VK_ACCESS_SHADER_WRITE_BIT,
                    )
                    for image in self.stage.outputs
                ]
                vk.vkCmdPipelineBarrier(
                    command,
                    vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    0,
                    None,
                    0,
                    None,
                    len(barriers),
                    barriers,
                )

            try:
                core._single_use(initialize)
                for image in self.stage.outputs:
                    image.layout = vk.VK_IMAGE_LAYOUT_GENERAL
            except Exception:
                self.close()
                raise
            self.consumers = set(self.runtime._consumers) - before

    def require_open(self):
        self.runtime.require_open()
        if self.closed:
            raise RuntimeError("Native FSR2 bindings have been retired")

    def jitter(self, sequence):
        return self.stage.jitter(sequence)

    def record(self, command, slot, jitter, camera, reset=False):
        with self.runtime.lock:
            self.require_open()
            now = time.monotonic()
            operation = self.stage.operation(
                slot=slot,
                jitter=jitter,
                fov_y=math.radians(camera.vertical_fov_degrees),
                dt_ms=min(1000.0, max(0.001, (now - self.last_time) * 1000)),
                reset=reset,
            )
            recording = (
                VulkanGraph()
                .add("fsr2", operation)
                .compile()
                .prepare_recording(self.runtime)
            )
            recording.record(command)
            self.pending = recording, now

    def submitted(self):
        recording, now = self.pending
        recording.submitted(_QueueCompletion(self))
        self.last_time = now
        self.pending = None

    def close(self):
        if self.closed:
            return
        with self.runtime.lock:
            try:
                vk.vkDeviceWaitIdle(self.runtime.device)
            except vk.VkErrorDeviceLost:
                pass
            self.idle = True
            if self.stage is not None:
                self.stage.close()
            self.pending = None
            self.closed = True
