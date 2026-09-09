"""Native queue bindings for scene-independent primary ray generation."""

import struct
import vulkan as vk
from ...runtime.ray_generation import VulkanRayGeneration
from ...pipeline.graph import VulkanGraph
from .denoiser_graph import _Binding


class NativeRayGeneration:
    def __init__(self, executor):
        self.executor = executor
        self.runtime = executor.core.runtime
        self.stages = []
        with self.runtime.lock:
            before = set(self.runtime._consumers)

            def buffer(b):
                return _Binding(
                    runtime=self.runtime,
                    buffer=b.buffer,
                    byte_size=b.size,
                    usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                    | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    require_open=self.runtime.require_open,
                )

            try:
                for camera in executor.camera_buffers:
                    self.stages.append(
                        VulkanRayGeneration(
                            self.runtime,
                            camera=buffer(camera),
                            rays=buffer(executor.ray_buffer),
                            paths=buffer(executor.path_buffer),
                            media=buffer(executor.medium_buffer),
                            capacity=executor.capacity,
                        )
                    )
            except Exception:
                self.close()
                raise
            self.consumers = set(self.runtime._consumers) - before

    def record(self, command, slot, constants):
        width, height, x, y, tw, th, count, sample = struct.unpack("8I", constants)
        operation = self.stages[slot].operation(
            extent=(width, height),
            tile_origin=(x, y),
            tile_extent=(tw, th),
            sample_index=sample,
            sample_count=count & 0x7FFFFFFF,
            capture_secondary=bool(count & 0x80000000),
        )
        # Buffer-only operation: no image layout/history publication. The native
        # executor owns submission, queue serialization and completion waits.
        VulkanGraph().add("generate", operation).compile().prepare_recording(
            self.runtime
        ).record(command)

    def close(self):
        for stage in reversed(self.stages):
            stage.close()
        self.stages = []
