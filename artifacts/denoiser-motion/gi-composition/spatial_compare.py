"""Compare the standalone graph stage with native viewer spatial denoising."""

from contextlib import ExitStack
from dataclasses import replace
from types import SimpleNamespace
import json
from pathlib import Path

import numpy as np
import vulkan as vk
import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.integrations.raster_workbench import _gi_config
from ordinarylight.pipeline.graph import VulkanGraph
from ordinarylight.pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse
from ordinarylight.runtime import VulkanRelaxSpatial, VulkanKernel, compile_compute
from tools.denoiser_motion.glass_detail import fixture, pose


def compare(presenter, width, height, iterations):
    core = presenter._core
    expected = presenter.capture_wavefront_hdr()
    frame = core.window_frames[(core.window_frame_index - 1) % 2]
    runtime = core.runtime
    names = (
        "wavefront_relax_temporal_diffuse_image",
        "wavefront_relax_temporal_specular_image",
        "wavefront_relax_normal_roughness_image",
        "wavefront_relax_view_z_image",
        "wavefront_material_image",
        "wavefront_hdr_image",
    )
    formats = (vk.VK_FORMAT_R16G16B16A16_SFLOAT,) * 3 + (
        vk.VK_FORMAT_R32_SFLOAT,
        vk.VK_FORMAT_R32_UINT,
        vk.VK_FORMAT_R16G16B16A16_SFLOAT,
    )
    with ExitStack() as stack:
        images = [
            stack.enter_context(runtime.image(width, height, format=f)) for f in formats
        ]

        def copy(command):
            for name, destination in zip(names, images):
                source = frame[name]
                barriers = [
                    core._image_barrier(
                        source,
                        vk.VK_IMAGE_LAYOUT_GENERAL,
                        vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        vk.VK_ACCESS_SHADER_WRITE_BIT,
                        vk.VK_ACCESS_TRANSFER_READ_BIT,
                    ),
                    core._image_barrier(
                        destination.image,
                        vk.VK_IMAGE_LAYOUT_UNDEFINED,
                        vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        0,
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
                    len(barriers),
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
                    destination.image,
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
                        destination.image,
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
                    len(barriers),
                    barriers,
                )

        core._single_use(copy)
        for image in images:
            image.layout = vk.VK_IMAGE_LAYOUT_GENERAL
        stage = stack.enter_context(
            VulkanRelaxSpatial(
                runtime,
                **dict(
                    zip(
                        (
                            "diffuse",
                            "specular",
                            "normal_roughness",
                            "view_z",
                            "material",
                            "output",
                        ),
                        images,
                    )
                ),
                iterations=iterations,
            )
        )
        result = stack.enter_context(runtime.buffer(width * height * 16))
        source, buffer = VulkanResource.image(images[-1]), VulkanResource.buffer(result)
        reader = stack.enter_context(
            VulkanKernel(
                runtime,
                compile_compute("""#version 460
layout(local_size_x=8,local_size_y=8) in;
layout(binding=0,rgba16f) readonly uniform image2D source_image;
layout(binding=1,std430) buffer Result { vec4 values[]; };
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy); ivec2 size=imageSize(source_image);
if(any(greaterThanEqual(p,size)))return; values[p.y*size.x+p.x]=imageLoad(source_image,p);}
"""),
                {0: source, 1: buffer},
            )
        )
        graph = VulkanGraph().add("spatial", stage.operation())
        graph.add(
            "readback",
            VulkanPass(
                "readback",
                (
                    VulkanResourceUse(
                        source,
                        vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        vk.VK_ACCESS_SHADER_READ_BIT,
                        vk.VK_IMAGE_LAYOUT_GENERAL,
                    ),
                    VulkanResourceUse(
                        buffer,
                        vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        vk.VK_ACCESS_SHADER_WRITE_BIT,
                    ),
                ),
                reader.bind,
                ((width + 7) // 8, (height + 7) // 8, 1),
            ),
            after=("spatial",),
        )
        graph.compile().execute(runtime).wait()
        actual = np.frombuffer(result.read(), np.float32).reshape(height, width, 4)
        assert np.isfinite(actual).all()
        error = float(np.abs(expected - actual).max())
        assert np.array_equal(actual, expected), error
        return error


def main():
    glfw = load_glfw()
    assert glfw.init()
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    width, height = 161, 121
    window = glfw.create_window(width, height, "Spatial graph comparison", None, None)
    report = {}
    try:
        for iterations in range(1, 6):
            cfg = replace(
                _gi_config(
                    SimpleNamespace(id="glass-detail", renderer={}),
                    capture=True,
                    denoiser_iterations=iterations,
                ),
                denoiser_signal_capture=True,
            )
            scene, glass, bars = fixture()
            with ol.VulkanGlfwPresenter(window, config=cfg) as presenter:
                errors = []
                for frame_index in range(6):
                    presenter.present_wavefront(
                        scene,
                        pose(scene, glass, bars, "camera", frame_index * 0.005),
                        width,
                        height,
                    )
                    errors.append(compare(presenter, width, height, iterations))
            report[str(iterations)] = {
                "frames": len(errors),
                "max_absolute_hdr_error": max(errors),
            }
            print(iterations, report[str(iterations)], flush=True)
    finally:
        glfw.destroy_window(window)
        glfw.terminate()
    Path("/tmp/relax-spatial-comparison.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
