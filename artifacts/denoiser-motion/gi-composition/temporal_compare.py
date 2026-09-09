"""Compare temporal+spatial graph history against native camera/glass sequences."""

from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import json
import numpy as np
import vulkan as vk
import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.integrations.raster_workbench import _gi_config
from ordinarylight.pipeline.graph import VulkanGraph
from ordinarylight.pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse
from ordinarylight.runtime import (
    VulkanRelaxHistory,
    VulkanRelaxTemporal,
    VulkanRelaxSpatial,
    VulkanKernel,
    compile_compute,
)
from tools.denoiser_motion.glass_detail import fixture, pose


def copy_images(core, pairs, width, height):
    def record(command):
        for source, target in pairs:
            barriers = [
                core._image_barrier(
                    source,
                    vk.VK_IMAGE_LAYOUT_GENERAL,
                    vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    vk.VK_ACCESS_SHADER_WRITE_BIT,
                    vk.VK_ACCESS_TRANSFER_READ_BIT,
                ),
                core._image_barrier(
                    target.image,
                    target.layout,
                    vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    vk.VK_ACCESS_MEMORY_READ_BIT | vk.VK_ACCESS_MEMORY_WRITE_BIT
                    if target.layout
                    else 0,
                    vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                ),
            ]
            vk.vkCmdPipelineBarrier(
                command,
                vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                None,
                0,
                None,
                2,
                barriers,
            )
            layers = vk.VkImageSubresourceLayers(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, layerCount=1
            )
            region = vk.VkImageCopy(
                srcSubresource=layers,
                dstSubresource=layers,
                extent=vk.VkExtent3D(width, height, 1),
            )
            vk.vkCmdCopyImage(
                command,
                source,
                vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                target.image,
                vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                [region],
            )
            barriers = [
                core._image_barrier(
                    source,
                    vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    vk.VK_IMAGE_LAYOUT_GENERAL,
                    vk.VK_ACCESS_TRANSFER_READ_BIT,
                    vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
                ),
                core._image_barrier(
                    target.image,
                    vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    vk.VK_IMAGE_LAYOUT_GENERAL,
                    vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
                ),
            ]
            vk.vkCmdPipelineBarrier(
                command,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0,
                0,
                None,
                0,
                None,
                2,
                barriers,
            )

    core._single_use(record)
    for source, target in pairs:
        target.layout = vk.VK_IMAGE_LAYOUT_GENERAL


def sequence(window, subject):
    width, height = 161, 121
    config = replace(
        _gi_config(
            SimpleNamespace(id="glass-detail", renderer={}),
            capture=True,
            denoiser_iterations=1,
        ),
        denoiser_signal_capture=True,
    )
    scene, glass, bars = fixture()
    with (
        ol.VulkanGlfwPresenter(window, config=config) as presenter,
        ExitStack() as stack,
    ):
        runtime = presenter._core.runtime

        def image(fmt):
            return stack.enter_context(runtime.image(width, height, format=fmt))

        rgba, scalar, uint = (
            vk.VK_FORMAT_R16G16B16A16_SFLOAT,
            vk.VK_FORMAT_R32_SFLOAT,
            vk.VK_FORMAT_R32_UINT,
        )
        histories = []
        for _ in range(2):
            histories.append(
                stack.enter_context(
                    VulkanRelaxHistory(
                        runtime,
                        normal_roughness=image(rgba),
                        view_z=image(scalar),
                        material=image(uint),
                        identity=image(uint),
                    )
                )
            )
        diffuse, specular, motion, hdr = [image(rgba) for _ in range(4)]
        expected_images = [image(fmt) for fmt in (rgba, rgba, scalar, scalar)]
        result = stack.enter_context(runtime.buffer(width * height * 32))
        errors = []
        for index in range(8):
            presenter.present_wavefront(
                scene, pose(scene, glass, bars, subject, index * 0.005), width, height
            )
            expected_hdr = presenter.capture_wavefront_hdr()
            core = presenter._core
            slot = (core.window_frame_index - 1) % 2
            native = core.window_frames[slot]
            current, previous = histories[slot], histories[1 - slot]
            pairs = list(
                zip(
                    (
                        native["wavefront_relax_" + name + "_image"]
                        for name in ("normal_roughness", "view_z")
                    ),
                    current.guides[:2],
                )
            )
            pairs += [
                (native["wavefront_material_image"], current.guides[2]),
                (native["wavefront_relax_identity_image"], current.guides[3]),
                (native["wavefront_relax_diffuse_image"], diffuse),
                (native["wavefront_relax_specular_image"], specular),
                (native["wavefront_relax_motion_image"], motion),
                (native["wavefront_hdr_image"], hdr),
            ]
            pairs += list(
                zip(
                    (
                        native["wavefront_relax_" + name + "_image"]
                        for name in (
                            "temporal_diffuse",
                            "temporal_specular",
                            "diffuse_history",
                            "specular_history",
                        )
                    ),
                    expected_images,
                )
            )
            copy_images(core, pairs, width, height)
            buffer = core.wavefront_executor.relax_temporal_constant_buffers[slot]
            mapped = vk.vkMapMemory(core.device, buffer.memory, 0, 32, 0)
            policy = np.frombuffer(mapped, np.float32).copy()
            vk.vkUnmapMemory(core.device, buffer.memory)
            with (
                VulkanRelaxTemporal(
                    runtime,
                    diffuse=diffuse,
                    specular=specular,
                    motion=motion,
                    previous=previous,
                    output=current,
                    history_limit=policy[2],
                    normal_threshold=policy[4],
                    depth_threshold=policy[5],
                    clamp_sigma=policy[6],
                    reactive_sigma=policy[7],
                ) as temporal,
                VulkanRelaxSpatial(
                    runtime,
                    diffuse=current.diffuse,
                    specular=current.specular,
                    normal_roughness=current.guides[0],
                    view_z=current.guides[1],
                    material=current.guides[2],
                    output=hdr,
                    iterations=1,
                ) as spatial,
            ):
                bindings = {
                    i: VulkanResource.image(im)
                    for i, im in enumerate([hdr, *current.images, *expected_images])
                }
                bindings[9] = VulkanResource.buffer(result)
                source = """#version 460
layout(local_size_x=8,local_size_y=8) in;
"""
                for i, fmt in enumerate(
                    (
                        "rgba16f",
                        "rgba16f",
                        "rgba16f",
                        "r32f",
                        "r32f",
                        "rgba16f",
                        "rgba16f",
                        "r32f",
                        "r32f",
                    )
                ):
                    source += (
                        f"layout(binding={i},{fmt}) readonly uniform image2D im{i};\n"
                    )
                source += """layout(binding=9,std430) buffer Result {vec4 values[];};
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy); ivec2 size=imageSize(im0);
if(any(greaterThanEqual(p,size)))return; int k=p.y*size.x+p.x;
values[k]=imageLoad(im0,p);
vec4 d=abs(imageLoad(im1,p)-imageLoad(im5,p));
vec4 s=abs(imageLoad(im2,p)-imageLoad(im6,p));
values[size.x*size.y+k]=vec4(max(max(d.x,d.y),max(d.z,d.w)),max(max(s.x,s.y),max(s.z,s.w)),
abs(imageLoad(im3,p).r-imageLoad(im7,p).r),abs(imageLoad(im4,p).r-imageLoad(im8,p).r));}
"""
                with VulkanKernel(runtime, compile_compute(source), bindings) as reader:
                    uses = tuple(
                        VulkanResourceUse(
                            r,
                            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            vk.VK_ACCESS_SHADER_WRITE_BIT
                            if i == 9
                            else vk.VK_ACCESS_SHADER_READ_BIT,
                            None if i == 9 else vk.VK_IMAGE_LAYOUT_GENERAL,
                        )
                        for i, r in bindings.items()
                    )
                    graph = VulkanGraph().add(
                        "temporal", temporal.operation(reset=not bool(policy[3]))
                    )
                    graph.add("spatial", spatial.operation())
                    graph.add(
                        "read",
                        VulkanPass(
                            "read",
                            uses,
                            reader.bind,
                            ((width + 7) // 8, (height + 7) // 8, 1),
                        ),
                    )
                    graph.compile().execute(runtime).wait()
                    actual = np.frombuffer(result.read(), np.float32).reshape(
                        2, height, width, 4
                    )
                    error = max(
                        float(np.abs(actual[0] - expected_hdr).max()),
                        float(actual[1].max()),
                    )
                    assert error == 0, (subject, index, error)
                    errors.append(error)
        return {
            "frames": len(errors),
            "max_hdr_temporal_and_history_error": max(errors),
        }


def main():
    glfw = load_glfw()
    assert glfw.init()
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(161, 121, "Temporal comparison", None, None)
    report = {}
    try:
        for subject in ("camera", "glass"):
            report[subject] = sequence(window, subject)
            print(subject, report[subject], flush=True)
    finally:
        glfw.destroy_window(window)
        glfw.terminate()
    Path("/tmp/relax-temporal-comparison.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
