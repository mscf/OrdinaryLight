"""Borrowed native bindings for ordinary (non-bucketed) intersection."""

import vulkan as vk
from ...runtime.intersection import VulkanIntersection
from ...pipeline.graph import VulkanGraph
from ...pipeline.vulkan import VulkanResource
from .denoiser_graph import _Binding


class NativeIntersection:
    def __init__(self, executor):
        self.runtime = executor.core.runtime
        self.stages = []
        self.key = (
            executor.core.scene_tlas.handle,
            executor.core.scene_vertex_buffer.buffer,
        )
        with self.runtime.lock:
            before = set(self.runtime._consumers)

            def buffer(b):
                return _Binding(
                    runtime=self.runtime,
                    buffer=b.buffer,
                    byte_size=b.size,
                    usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                    | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
                    | vk.VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                    require_open=self.runtime.require_open,
                )

            owner = _Binding(
                runtime=self.runtime, require_open=self.runtime.require_open
            )
            tlas = VulkanResource(owner, "acceleration_structure", self.key[0])
            vertices = VulkanResource.buffer(buffer(executor.core.scene_vertex_buffer))
            self.indirect = buffer(executor.indirect_buffer)
            try:
                for rays in (executor.ray_buffer, executor.next_ray_buffer):
                    self.stages.append(
                        VulkanIntersection(
                            self.runtime,
                            tlas=tlas,
                            vertices=vertices,
                            rays=buffer(rays),
                            hits=buffer(executor.hit_buffer),
                            capacity=executor.capacity,
                        )
                    )
            except Exception:
                self.close()
                raise
            self.consumers = set(self.runtime._consumers) - before

    def record(self, command, slot):
        # Parent owns command submission and all borrowed allocations. No image
        # layouts or temporal state require a submitted callback here.
        VulkanGraph().add(
            "intersection", self.stages[slot].operation(indirect=self.indirect)
        ).compile().prepare_recording(self.runtime).record(command)

    def close(self):
        for stage in reversed(self.stages):
            stage.close()
        self.stages = []
