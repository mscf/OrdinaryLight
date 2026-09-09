"""Set-1 material reads consume graph producers before layout publication."""

import os
import numpy as np
import pytest
import vulkan as vk
import ordinarylight as ol
from ordinarylight.materials import MaterialGraph, MaterialNode, MaterialResource
from ordinarylight.runtime import VulkanKernel, VulkanSampler, compile_compute
from ordinarylight.pipeline.graph import VulkanGraph
from ordinarylight.pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse


@pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_GRAPH") != "1", reason="opt-in GPU"
)
def test_material_set_reads_graph_image_producer():
    program = MaterialGraph(
        {
            "uv": MaterialNode("constant", value=(0.5, 0.5), type="vec2"),
            "pixel": MaterialNode("texture", ("uv",), value="texture", type="vec4"),
            "rgb": MaterialNode("components", ("pixel",), value="rgb", type="vec3"),
        },
        {"emission": "rgb"},
        resources=(MaterialResource("texture", "texture"),),
    ).compile()
    with (
        ol.VulkanRuntime() as runtime,
        runtime.image(1, 1) as image,
        VulkanSampler(runtime) as sampler,
        runtime.buffer(16) as output,
    ):
        with ol.VulkanMaterialResources(
            runtime, [program], {"texture": (image, sampler)}
        ) as material:
            source = (
                "#version 460\nlayout(local_size_x=1) in;\n"
                + material.source
                + """
layout(set=0,binding=0,std430) buffer Output {vec4 value;};
void main(){value=ol_graph_texture(vec2(.5));}
"""
            )
            with VulkanKernel(
                runtime,
                compile_compute(source),
                {0: VulkanResource.buffer(output)},
                material_resources=material,
            ) as kernel:

                def producer(color):
                    return VulkanPass(
                        "fill",
                        (
                            VulkanResourceUse(
                                VulkanResource.image(image),
                                vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                                vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                                vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            ),
                        ),
                        lambda command: vk.vkCmdClearColorImage(
                            command,
                            image.image,
                            vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            vk.VkClearColorValue(float32=color),
                            1,
                            [
                                vk.VkImageSubresourceRange(
                                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                                    levelCount=1,
                                    layerCount=1,
                                )
                            ],
                        ),
                    )

                VulkanGraph().add("init", producer([0, 0, 0, 0])).compile().execute(
                    runtime
                ).wait()
                assert image.layout == vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL
                for color in ([0.25, 0.5, 0.75, 1], [1, 0.5, 0.25, 1]):
                    graph = VulkanGraph().add(
                        "sample",
                        VulkanPass(
                            "sample",
                            material.uses
                            + (
                                VulkanResourceUse(
                                    VulkanResource.buffer(output),
                                    vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    vk.VK_ACCESS_SHADER_WRITE_BIT,
                                ),
                            ),
                            kernel.bind,
                            (1, 1, 1),
                        ),
                    )
                    graph.add("fill", producer(color))
                    assert graph.compile().order == ("fill", "sample")
                    graph.compile().execute(runtime).wait()
                    np.testing.assert_allclose(
                        np.frombuffer(output.read(), np.float32), color
                    )
                with pytest.raises(RuntimeError, match="attached"):
                    material.close()
            assert not material._borrowers
