"""GPU preparation of indirect dispatch for 64-thread wavefront consumers."""

from importlib.resources import files
import vulkan as vk
from .kernel import VulkanKernel
from ..pipeline.graph import VulkanOperation
from ..pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse


class VulkanQueueDispatch:
    """Borrow a queue header and storage/indirect argument buffer.

    The producer must supply a valid count/capacity header. Capacity must fit the
    queue allocation and the device's dispatch limit. This stage clamps count to
    that capacity; it does not validate GPU-written headers or reset the queue.
    """

    def __init__(self, runtime, *, queue, arguments):
        self.runtime, self.closed, self.completion = runtime, False, None
        with runtime.lock:
            runtime.require_open()
            for buffer in (queue, arguments):
                buffer.require_open()
                if buffer.runtime is not runtime:
                    raise ValueError("Dispatch buffers must share a runtime")
            if queue.byte_size < 16 or arguments.byte_size < 12:
                raise ValueError(
                    "Dispatch requires a 16-byte header and 12-byte arguments"
                )
            if queue.buffer == arguments.buffer:
                raise ValueError("Dispatch buffers must not alias")
            if not arguments.usage & vk.VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT:
                raise ValueError("Dispatch arguments require indirect buffer usage")
            self.kernel = VulkanKernel(
                runtime,
                files("ordinarylight.shaders")
                .joinpath("wavefront_prepare_indirect.comp.spv")
                .read_bytes(),
                {0: VulkanResource.buffer(queue), 1: VulkanResource.buffer(arguments)},
            )

    def require_open(self):
        if self.closed:
            raise RuntimeError("Queue dispatch stage is closed")
        self.kernel.require_open()

    def operation(self, *, after=()):
        """Write (ceil(min(count, capacity)/64), 1, 1), including empty queues."""
        self.require_open()

        def record(command):
            self.kernel.bind(command)
            vk.vkCmdDispatch(command, 1, 1, 1)

        after = tuple(after)
        return VulkanOperation(
            [
                VulkanPass(
                    "prepare_dispatch",
                    tuple(
                        VulkanResourceUse(
                            resource,
                            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            vk.VK_ACCESS_SHADER_READ_BIT
                            if binding == 0
                            else vk.VK_ACCESS_SHADER_WRITE_BIT,
                        )
                        for binding, resource in self.kernel.bindings.items()
                    ),
                    record,
                )
            ],
            validate=self.require_open,
            dependencies=lambda: after
            + ((self.completion,) if self.completion else ()),
            submitted=lambda completion: setattr(self, "completion", completion),
        )

    def close(self):
        with self.runtime.lock:
            if self.closed:
                return
            if self.completion is not None:
                self.completion.wait()
            self.kernel.close()
            self.closed = True

    def __enter__(self):
        self.require_open()
        return self

    def __exit__(self, *_exc):
        self.close()
