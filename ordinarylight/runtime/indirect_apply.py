"""Compact indirect reservoir visualization and HDR correction operation."""

import math
from operator import index
import struct
import vulkan as vk
from ..pipeline.graph import VulkanOperation
from ..pipeline.vulkan import VulkanResourceUse, VulkanPass


def indirect_apply_operation(
    kernel,
    *,
    output_extent,
    reservoir_extent,
    mode="apply",
    history_limit=4,
    strength=1.0,
):
    """Record the packaged indirect-debug shader against explicit bindings 0..3.

    Modes are radiance, history, validity, acceptance and apply. Apply reads and
    corrects existing HDR; diagnostic modes overwrite it. Caller owns resources.
    """
    kernel.require_open()
    modes = {"radiance": 1, "history": 2, "validity": 3, "acceptance": 4, "apply": 5}
    if mode not in modes:
        raise ValueError("Unknown indirect output mode")
    output_extent, reservoir_extent = (
        tuple(map(index, output_extent)),
        tuple(map(index, reservoir_extent)),
    )
    if any(
        len(e) != 2 or any(not 0 < n <= 0x7FFFFFFF for n in e)
        for e in (output_extent, reservoir_extent)
    ):
        raise ValueError("Invalid indirect output extent")
    history_limit = index(history_limit)
    strength = float(strength)
    if (
        not 0 < history_limit <= 0xFFFFFFFF
        or not math.isfinite(strength)
        or not 0 <= strength <= 1
    ):
        raise ValueError("Invalid history limit/apply strength")
    bindings = kernel.bindings
    if set(bindings) != set(range(4)):
        raise ValueError("Indirect output requires bindings 0..3")
    count = reservoir_extent[0] * reservoir_extent[1]
    for b, size in ((0, count * 24), (2, count * 4)):
        if bindings[b].kind != "buffer" or bindings[b].size < size:
            raise ValueError("Indirect reservoir/seed buffer too small")
    for b, fmt in ((1, vk.VK_FORMAT_R16G16B16A16_SFLOAT), (3, vk.VK_FORMAT_R32_UINT)):
        r = bindings[b]
        if (
            r.kind != "image"
            or r.owner.format != fmt
            or r.owner.width < output_extent[0]
            or r.owner.height < output_extent[1]
        ):
            raise ValueError("Indirect image format/extent mismatch")
    constants = struct.pack(
        "6If", *output_extent, *reservoir_extent, modes[mode], history_limit, strength
    )
    uses = {}
    for b, r in bindings.items():
        access = vk.VK_ACCESS_SHADER_READ_BIT
        if b == 1:
            access = vk.VK_ACCESS_SHADER_WRITE_BIT | (access if mode == "apply" else 0)
        key = (r.kind, r.handle)
        if key in uses:
            access |= uses[key].access
        uses[key] = VulkanResourceUse(
            r,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            access,
            vk.VK_IMAGE_LAYOUT_GENERAL if b in (1, 3) else None,
        )

    def record(command):
        kernel.bind(command, constants)
        vk.vkCmdDispatch(
            command, (output_extent[0] + 7) // 8, (output_extent[1] + 7) // 8, 1
        )

    return VulkanOperation(
        [VulkanPass("indirect_output", tuple(uses.values()), record)],
        validate=kernel.require_open,
    )
