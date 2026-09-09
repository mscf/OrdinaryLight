"""Graph operation for the packaged compact indirect-candidate shader."""

from operator import index
import struct
import vulkan as vk
from ..pipeline.graph import VulkanOperation
from ..pipeline.vulkan import VulkanPass, VulkanResourceUse


def indirect_candidates_operation(
    kernel,
    *,
    source_extent,
    reservoir_extent,
    history_valid=False,
    frame_index=0,
    spatial=False,
    profiling=False,
    history_limit=4,
):
    """Use explicit shader bindings 0..12, including previous-frame resources.

    Caller owns the kernel/resources and publishes history validity after a
    successful submission. Current and previous reservoir storage must differ.
    """
    kernel.require_open()
    source_extent, reservoir_extent = (
        tuple(map(index, source_extent)),
        tuple(map(index, reservoir_extent)),
    )
    if (
        len(source_extent) != 2
        or len(reservoir_extent) != 2
        or any(
            not (0 < r <= 0x7FFFFFFF and 0 < s <= 0x7FFFFFFF)
            for s, r in zip(source_extent, reservoir_extent)
        )
    ):
        raise ValueError("Invalid source/reservoir extent")
    frame_index, history_limit = index(frame_index), index(history_limit)
    if not 0 <= frame_index <= 0xFFFFFFFF or not 0 < history_limit <= 0xFFFFFFFF:
        raise ValueError("Invalid frame index/history limit")
    bindings = kernel.bindings
    if set(bindings) != set(range(13)):
        raise ValueError("Candidate kernel requires bindings 0 through 12")
    images = {
        1: vk.VK_FORMAT_R16G16B16A16_SFLOAT,
        2: vk.VK_FORMAT_R32_SFLOAT,
        3: vk.VK_FORMAT_R32_UINT,
        6: vk.VK_FORMAT_R32_SFLOAT,
        7: vk.VK_FORMAT_R32_UINT,
        9: vk.VK_FORMAT_R32_UINT,
        10: vk.VK_FORMAT_R32_UINT,
    }
    for b, r in bindings.items():
        expected = (
            "image"
            if b in images
            else "acceleration_structure"
            if b == 12
            else "buffer"
        )
        if r.kind != expected:
            raise ValueError("Candidate binding kind mismatch")
        if b in images and (
            r.owner.format != images[b]
            or r.owner.width < source_extent[0]
            or r.owner.height < source_extent[1]
        ):
            raise ValueError("Candidate image format/extent mismatch")
    count = reservoir_extent[0] * reservoir_extent[1]
    for b, size in (
        (0, count * 24),
        (5, count * 24),
        (4, 64),
        (8, 64),
        (11, 72 if profiling else 4),
    ):
        if bindings[b].size < size:
            raise ValueError("Candidate buffer is too small")
    if bindings[0].handle == bindings[5].handle:
        raise ValueError("Current and previous reservoirs must not alias")
    for writable in (0, 11):
        if any(
            bindings[writable].handle == bindings[b].handle
            for b in (0, 4, 5, 8, 11)
            if b != writable
        ):
            raise ValueError("Writable candidate buffers must not alias other bindings")
    constants = struct.pack(
        "9I",
        *source_extent,
        *reservoir_extent,
        int(bool(history_valid)),
        frame_index,
        int(bool(spatial)),
        int(bool(profiling)),
        history_limit,
    )
    uses = {}
    for b, r in bindings.items():
        access = (
            vk.VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
            if b == 12
            else vk.VK_ACCESS_SHADER_READ_BIT
        )
        if b in (0, 11):
            access |= vk.VK_ACCESS_SHADER_WRITE_BIT
        key = (r.kind, r.handle)
        if key in uses:
            access |= uses[key].access
        uses[key] = VulkanResourceUse(
            r,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            access,
            vk.VK_IMAGE_LAYOUT_GENERAL if b in images else None,
        )
    passes = []
    counter = bindings[11]
    if profiling:
        if not counter.owner.usage & vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT:
            raise ValueError("Profiling counters require transfer destination usage")

        def clear(command):
            vk.vkCmdFillBuffer(command, counter.handle, counter.offset, counter.size, 0)

        passes.append(
            VulkanPass(
                "reset_counters",
                (
                    VulkanResourceUse(
                        counter,
                        vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                        vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    ),
                ),
                clear,
            )
        )

    def record(command):
        kernel.bind(command, constants)
        vk.vkCmdDispatch(
            command, (reservoir_extent[0] + 7) // 8, (reservoir_extent[1] + 7) // 8, 1
        )

    passes.append(VulkanPass("candidates", tuple(uses.values()), record))
    if profiling:
        passes.append(
            VulkanPass(
                "counters_host_ready",
                (
                    VulkanResourceUse(
                        counter,
                        vk.VK_PIPELINE_STAGE_HOST_BIT,
                        vk.VK_ACCESS_HOST_READ_BIT,
                    ),
                ),
                lambda command: None,
            )
        )
    return VulkanOperation(passes, validate=kernel.require_open)
