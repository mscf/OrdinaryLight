"""Public image views over native GI allocations; no allocation or submission."""

from types import MappingProxyType
import weakref

import vulkan as vk

from ...pipeline.gi import GiImage, GiBuffer


def native_gi_buffers(core, slot):
    frame = core.window_frames[slot]
    buffer = frame.get("wavefront_primary_hit_buffer")
    if buffer is None:
        return MappingProxyType({})
    old = frame.get("gi_buffers")
    if old is not None and old["primary_hits"].buffer == buffer.buffer:
        return old
    owner = weakref.ref(core)
    generation = core.swapchain_generation

    def validate():
        current = owner()
        if (current is None or current.device is None or current.swapchain_generation != generation
                or frame.get("wavefront_primary_hit_buffer") is not buffer):
            raise RuntimeError("GI buffer allocation has been retired")
        current.runtime.require_open()

    frame["gi_buffers"] = MappingProxyType({"primary_hits": GiBuffer(
        core.runtime, buffer.buffer, buffer.size,
        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, validate,
    )})
    return frame["gi_buffers"]


def native_gi_images(core, slot):
    frame = core.window_frames[slot]
    if "gi_images" in frame:
        return frame["gi_images"]
    owner = weakref.ref(core)
    generation = core.swapchain_generation

    def validate():
        current = owner()
        if (current is None or current.device is None
                or current.swapchain_generation != generation
                or slot >= len(current.window_frames)
                or current.window_frames[slot] is not frame
                or frame.get("wavefront_hdr_image") is None):
            raise RuntimeError("GI image allocation has been retired")
        current.runtime.require_open()

    rgba = vk.VK_FORMAT_R16G16B16A16_SFLOAT
    scalar = vk.VK_FORMAT_R32_SFLOAT
    uint = vk.VK_FORMAT_R32_UINT
    names = {
        "hdr": ("wavefront_hdr", rgba),
        "raw_hdr": ("wavefront_raw_hdr", rgba),
        "ray_distance": ("wavefront_position", scalar),
        "packed_normal_class": ("wavefront_normal", uint),
        "material_signature": ("wavefront_material", uint),
    }
    names.update({
        name: ("wavefront_relax_" + name, fmt)
        for name, fmt in (
            ("diffuse", rgba), ("specular", rgba),
            ("normal_roughness", rgba), ("motion", rgba),
            ("view_z", scalar), ("identity", uint),
            ("temporal_diffuse", rgba), ("temporal_specular", rgba),
            ("diffuse_history", scalar), ("specular_history", scalar),
            ("atrous_diffuse", rgba), ("atrous_specular", rgba),
        )
    })
    width, height = frame["wavefront_allocation_extent"]
    extents = frame.get("gi_image_extents", {})
    images = {}
    for name, (prefix, fmt) in names.items():
        if frame.get(prefix + "_image") is not None:
            image_width, image_height = extents.get(
                prefix, extents.get("relax", (width, height))
                if prefix.startswith("wavefront_relax_") else (width, height),
            )
            usage = vk.VK_IMAGE_USAGE_STORAGE_BIT
            if name == "raw_hdr":
                usage |= vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT
            images[name] = GiImage(
                core.runtime, frame[prefix + "_image"], frame[prefix + "_view"],
                image_width, image_height, fmt, vk.VK_IMAGE_LAYOUT_GENERAL,
                usage, validate,
            )
    frame["gi_images"] = MappingProxyType(images)
    return frame["gi_images"]


def record_raw_hdr(frame):
    """Preserve raw transport before in-place denoising; GPU copy only."""
    def barrier():
        vk.vkCmdPipelineBarrier(
            frame.command, vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0,
            1, [vk.VkMemoryBarrier(
                srcAccessMask=vk.VK_ACCESS_MEMORY_READ_BIT | vk.VK_ACCESS_MEMORY_WRITE_BIT,
                dstAccessMask=vk.VK_ACCESS_MEMORY_READ_BIT | vk.VK_ACCESS_MEMORY_WRITE_BIT,
            )], 0, None, 0, None,
        )
    barrier()
    layers = vk.VkImageSubresourceLayers(
        aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, mipLevel=0,
        baseArrayLayer=0, layerCount=1,
    )
    width, height = frame.render_extent
    vk.vkCmdCopyImage(
        frame.command, frame.images["hdr"].image, vk.VK_IMAGE_LAYOUT_GENERAL,
        frame.images["raw_hdr"].image, vk.VK_IMAGE_LAYOUT_GENERAL,
        1, [vk.VkImageCopy(
            srcSubresource=layers, srcOffset=vk.VkOffset3D(x=0, y=0, z=0),
            dstSubresource=layers, dstOffset=vk.VkOffset3D(x=0, y=0, z=0),
            extent=vk.VkExtent3D(width=width, height=height, depth=1),
        )],
    )
    barrier()
