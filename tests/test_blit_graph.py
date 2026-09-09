"""Display transfer scaling, channel conversion and imported-resource hazards."""

from contextlib import ExitStack
from types import SimpleNamespace
import os
import numpy as np
import pytest
import vulkan as vk
from ordinarylight.runtime import (
    VulkanRuntime,
    VulkanKernel,
    VulkanOutput,
    compile_compute,
    blit_operation,
)
from ordinarylight.pipeline.graph import VulkanGraph
from ordinarylight.pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse


@pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_GRAPH") != "1", reason="opt-in GPU"
)
@pytest.mark.parametrize(
    "format", [vk.VK_FORMAT_R8G8B8A8_UNORM, vk.VK_FORMAT_B8G8R8A8_UNORM]
)
def test_blit_scale_and_channels(format):
    with VulkanRuntime() as runtime, ExitStack() as stack:
        source = stack.enter_context(
            runtime.image(2, 1, format=vk.VK_FORMAT_R8G8B8A8_UNORM)
        )
        target = stack.enter_context(runtime.image(4, 2, format=format))
        resource = VulkanResource.image(source)
        writer = stack.enter_context(
            VulkanKernel(
                runtime,
                compile_compute("""#version 460
layout(local_size_x=2) in;
layout(binding=0,rgba8) writeonly uniform image2D output_image;
void main(){int x=int(gl_GlobalInvocationID.x);imageStore(output_image,ivec2(x,0),x==0?vec4(1,0,0,1):vec4(0,0,1,1));}
"""),
                {0: resource},
            )
        )
        graph = VulkanGraph().add("blit", blit_operation(source, target))
        graph.add(
            "write",
            VulkanPass(
                "write",
                (
                    VulkanResourceUse(
                        resource,
                        vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        vk.VK_ACCESS_SHADER_WRITE_BIT,
                        vk.VK_IMAGE_LAYOUT_GENERAL,
                    ),
                ),
                writer.bind,
                (1, 1, 1),
            ),
        )
        completion = graph.compile().execute(runtime)
        completion.wait()
        assert source.layout == target.layout == vk.VK_IMAGE_LAYOUT_GENERAL
        with VulkanOutput(runtime) as output:
            data = np.frombuffer(
                output.read(SimpleNamespace(image=target, completion=completion)),
                np.uint8,
            ).reshape(2, 4, 4)
        first, second = [255, 0, 0, 255], [0, 0, 255, 255]
        if format == vk.VK_FORMAT_B8G8R8A8_UNORM:
            first, second = second, first
        np.testing.assert_array_equal(data, [[first, first, second, second]] * 2)
        with pytest.raises(ValueError, match="alias"):
            blit_operation(source, source)
        target.usage = vk.VK_IMAGE_USAGE_STORAGE_BIT
        with pytest.raises(ValueError, match="transfer usage"):
            blit_operation(source, target)
