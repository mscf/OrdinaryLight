"""Diagnostic lazy continuation fields for rejected selected-diffuse paths.

Private scratch convention: position_valid.w == -1 marks absent continuation.
Only used with matched sampled-indirect preparation; not a public record ABI.
Packed mode additionally requires a matching resolve kernel. Native placeholders
are safe only when unused reservoir writes are disabled by the resolve policy.
Algorithms remain typed OrdinaryShade; no generated shader bodies are edited.
"""
from pathlib import Path
from primary_hit_layout_experiment import load_module


def variant(directory, *, packed=False):
    from primary_capture_store_experiment import variant as deferred
    module=deferred(directory)
    source=Path(module.__file__).read_text()
    begin=source.index("        secondary_paths[path_index] = SecondaryPathState(",source.index("    if capture_secondary:",source.index("def processPrimaryPixel")))
    end=source.index("    elif indirect_capture_pixel:",begin)
    complete=source[begin:end]
    replacement='''        local_only = False
        if osh.specialization('WAVE_SELECTED_DIFFUSE'):
            local_only = (selected_diffuse_active and sampled_specular == 0.0
                and bsdf_pdf == 0.0 and not osh.any_value(path.throughput.rgb > osh.vec3(0.0)))
        if local_only:
            secondary_paths[path_index].position_valid = osh.vec4(0.0, 0.0, 0.0, -1.0)
            secondary_paths[path_index].primary_radiance = osh.vec4(path.radiance.rgb, pbrSpecularProbability(material))
            secondary_paths[path_index].specular_radiance_hit_distance = osh.vec4(primary_specular, -1.0)
            secondary_paths[path_index].primary_position = osh.vec4(position, 1.0 + osh.clamp(material.base_roughness.a, 0.0, 1.0))
            secondary_paths[path_index].primary_geometry = geometry
        else:
'''+''.join('    '+line if line.strip() else line for line in complete.splitlines(True))
    if packed:
        # Contiguous first five vectors; last three vectors are dormant.
        start=replacement.index("        if local_only:")
        stop=replacement.index("        else:",start)
        local=replacement[start:stop]
        for old,new in (("primary_radiance","normal_pdf"),("specular_radiance_hit_distance","primary_throughput"),("primary_position","primary_radiance"),("primary_geometry","diffuse_radiance_hit_distance")):
            local=local.replace("]."+old+" =", "].__"+new+" =")
        local=local.replace("].__", "].")
        replacement=replacement[:start]+local+replacement[stop:]
    return load_module('ordinarylight.shaders.selected_local_state_diagnostic',source[:begin]+replacement+source[end:],directory)


def compile_prepare(directory, *, packed=False):
    import ordinaryshade as osh
    from ordinarylight.denoising import kernels
    from ordinarylight.runtime import compile_compute
    source=Path(kernels.__file__).read_text()
    needle='    secondary = secondary_paths[path_index]\n'
    assert source.count(needle)==1
    source=source.replace(needle,needle+decode(packed))
    module=load_module('selected_local_prepare_diagnostic',source,directory)
    return compile_compute(osh.compile(module.prepare_relax_signals,helpers=(
        module.prepare_decode_normal,module.prepare_unpack_normal,
        module.prepare_previous_pixel,module.prepare_custom_surface_history,
    )).source)


def decode(packed):
    if packed:
        return '''    if secondary.position_valid.w < 0.0:
        secondary = SecondaryPathState(
            osh.vec4(0.0), osh.vec4(0.0), osh.vec4(0.0, 0.0, 0.0, 1.0),
            secondary.normal_pdf, osh.vec4(0.0, 0.0, 0.0, -1.0),
            secondary.primary_throughput, secondary.primary_radiance,
            secondary.diffuse_radiance_hit_distance)
'''
    return '''    if secondary.position_valid.w < 0.0:
        secondary.position_valid = osh.vec4(0.0)
        secondary.normal_pdf = osh.vec4(0.0)
        secondary.primary_throughput = osh.vec4(0.0, 0.0, 0.0, 1.0)
        secondary.diffuse_radiance_hit_distance = osh.vec4(0.0, 0.0, 0.0, -1.0)
'''


def compile_resolve(directory):
    import ordinaryshade as osh
    import generate_core_shaders as core
    from ordinarylight.runtime import compile_compute
    source=Path(core.__file__).read_text()
    begin=source.index('def wavefront_path_to_hdr(')
    end=source.index('\n\n@osh.compute',begin)
    body=source[begin:end]
    needle='    secondary = secondary_paths[path_index]\n'
    assert body.count(needle)==1
    body=body.replace(needle,needle+decode(True))
    module=load_module('selected_local_resolve_diagnostic',source[:begin]+body+source[end:],directory)
    return compile_compute(osh.compile(module.wavefront_path_to_hdr,helpers=(
        module.indirectEncodeNormal,module.indirectPackRgb9e5,
        module.emptyIndirectLightReservoir,module.storeIndirectLightReservoir,
    )).source)


def read_guide(renderer,name):
    """Explicit diagnostic readback of every guide channel, outside timing."""
    import numpy as np
    import vulkan as vk
    from ordinarylight.pipeline.vulkan import VulkanResource,VulkanResourceUse,VulkanPass,VulkanPassPipeline
    image=renderer.frame.images[name]
    width,height=renderer.frame.render_extent
    if image.format==vk.VK_FORMAT_R16G16B16A16_SFLOAT:
        dtype,channels=np.float16,4
    elif image.format==vk.VK_FORMAT_R32_SFLOAT:
        dtype,channels=np.float32,1
    else:raise ValueError('Unsupported diagnostic guide format')
    with renderer.runtime.buffer(width*height*channels*np.dtype(dtype).itemsize) as buffer:
        resource=VulkanResource.image(image)
        def copy(command):
            vk.vkCmdCopyImageToBuffer(command,image.image,vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,buffer.buffer,1,[vk.VkBufferImageCopy(
                imageSubresource=vk.VkImageSubresourceLayers(aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,layerCount=1),
                imageExtent=vk.VkExtent3D(width,height,1))])
        passes=(VulkanPass('read_guide',(
            VulkanResourceUse(resource,vk.VK_PIPELINE_STAGE_TRANSFER_BIT,vk.VK_ACCESS_TRANSFER_READ_BIT,vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL),
            VulkanResourceUse(VulkanResource.buffer(buffer),vk.VK_PIPELINE_STAGE_TRANSFER_BIT,vk.VK_ACCESS_TRANSFER_WRITE_BIT)),copy),
            VulkanPass('restore_guide',(VulkanResourceUse(resource,vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                vk.VK_ACCESS_SHADER_READ_BIT|vk.VK_ACCESS_SHADER_WRITE_BIT,vk.VK_IMAGE_LAYOUT_GENERAL),),lambda command:None))
        VulkanPassPipeline(passes).execute(renderer.runtime,after=(renderer.completion,)).wait()
        return np.frombuffer(buffer.read(),dtype).reshape(height,width,channels).astype(np.float32)
