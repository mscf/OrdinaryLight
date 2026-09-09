"""Closest-hit triangle intersection for the split wavefront queue ABI."""

from importlib.resources import files
from operator import index
import vulkan as vk
from .kernel import VulkanKernel
from ..pipeline.graph import VulkanOperation
from ..pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse


class VulkanIntersection:
    """Borrow a TLAS, world-space vertex buffer and ray/hit queues.

    TLAS instance custom indices are triangle offsets into the vertex buffer
    (three vec4s per triangle), matching resident Scene resources. This is the
    existing opaque closest-hit query, not material/alpha-aware transport.
    """

    def __init__(self, runtime, *, tlas, vertices, rays, hits, capacity):
        self.runtime, self.closed, self.completion = runtime, False, None
        self.capacity = index(capacity)
        if not 0 < self.capacity <= 0xFFFFFFFF:
            raise ValueError("Intersection capacity must fit a positive uint32")
        self.bindings = {
            0: tlas,
            1: VulkanResource.buffer(rays),
            2: VulkanResource.buffer(hits),
            3: vertices,
        }
        self.leases = []
        with runtime.lock:
            runtime.require_open()
            if tlas.kind != "acceleration_structure" or vertices.kind != "buffer":
                raise ValueError(
                    "Intersection requires TLAS and vertex resource bindings"
                )
            if vertices.size < 48 or vertices.size % 16:
                raise ValueError("Vertices require vec4 triangle storage")
            for resource in self.bindings.values():
                resource.owner.require_open()
                if resource.owner.runtime is not runtime:
                    raise ValueError("Intersection resources must share a runtime")
            if (
                len({r.handle for r in (self.bindings[1], self.bindings[2], vertices)})
                != 3
            ):
                raise ValueError("Intersection buffers must not alias")
            for buffer in (rays, hits):
                if buffer.byte_size < 16 + self.capacity * 48:
                    raise ValueError("Intersection queue is smaller than capacity")
            if not hits.usage & vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT:
                raise ValueError("Hit queue requires transfer destination usage")
            self.kernel = VulkanKernel(
                runtime,
                files("ordinarylight.shaders")
                .joinpath("wavefront_intersect.comp.spv")
                .read_bytes(),
                self.bindings,
            )
            # Resident scenes use borrower leases; VulkanKernel already retains
            # ordinary buffer allocations. Native adapters hold their own lease.
            for owner in {tlas.owner, vertices.owner}:
                if hasattr(owner, "_borrowers"):
                    owner._borrowers.add(self)
                    self.leases.append(owner)

    def require_open(self):
        if self.closed:
            raise RuntimeError("Intersection stage is closed")
        self.kernel.require_open()

    def operation(self, *, indirect=None, after=()):
        """Intersect at most capacity rays; optionally use a dispatch-args buffer.

        Indirect dispatch is caller-prepared (three uint32s at offset zero) and
        must cover the active queue with 64-thread workgroups. Keep it alive until
        completion. Hit header reset handles zero-ray dispatches without stale hits.
        """
        self.require_open()
        extra = []
        if indirect is not None:
            indirect.require_open()
            if (
                indirect.runtime is not self.runtime
                or indirect.byte_size < 12
                or not indirect.usage & vk.VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
            ):
                raise ValueError("Invalid intersection indirect buffer")
            extra.append(
                VulkanResourceUse(
                    VulkanResource.buffer(indirect),
                    vk.VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                    vk.VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
                )
            )
        hit = self.bindings[2]

        def reset(command):
            for offset, value in ((0, 0), (4, self.capacity), (8, 0)):
                vk.vkCmdFillBuffer(command, hit.handle, offset, 4, value)

        def record(command):
            self.kernel.bind(command)
            if indirect is None:
                vk.vkCmdDispatch(command, (self.capacity + 63) // 64, 1, 1)
            else:
                vk.vkCmdDispatchIndirect(command, indirect.buffer, 0)

        uses = [
            VulkanResourceUse(
                resource,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
                if i == 0
                else vk.VK_ACCESS_SHADER_READ_BIT
                | (vk.VK_ACCESS_SHADER_WRITE_BIT if i == 2 else 0),
            )
            for i, resource in self.bindings.items()
        ]
        after = tuple(after)
        return VulkanOperation(
            [
                VulkanPass(
                    "hits.reset",
                    (
                        VulkanResourceUse(
                            hit,
                            vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                            vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                        ),
                    ),
                    reset,
                ),
                VulkanPass("intersect", tuple(uses + extra), record),
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
            for owner in self.leases:
                owner._borrowers.discard(self)
            self.closed = True

    def __enter__(self):
        self.require_open()
        return self

    def __exit__(self, *_exc):
        self.close()
