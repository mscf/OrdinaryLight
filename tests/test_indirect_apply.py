from contextlib import ExitStack
from importlib.resources import files
import os
import numpy as np
import pytest
import vulkan as vk
from ordinarylight.runtime import (
    VulkanRuntime,
    VulkanKernel,
    compile_compute,
    indirect_apply_operation,
)
from ordinarylight.pipeline.graph import VulkanGraph
from ordinarylight.pipeline.vulkan import VulkanResource, VulkanResourceUse, VulkanPass


@pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_GRAPH") != "1", reason="opt-in GPU"
)
@pytest.mark.parametrize("mode", ["apply", "radiance", "history"])
def test_empty_reservoir_output(mode):
    with VulkanRuntime() as runtime, ExitStack() as stack:

        def buffer(size):
            return stack.enter_context(runtime.buffer(size, data=bytes(size)))

        hdr = stack.enter_context(
            runtime.image(1, 1, format=vk.VK_FORMAT_R16G16B16A16_SFLOAT)
        )
        material = stack.enter_context(
            runtime.image(1, 1, format=vk.VK_FORMAT_R32_UINT)
        )
        bindings = {
            0: VulkanResource.buffer(buffer(24)),
            1: VulkanResource.image(hdr),
            2: VulkanResource.buffer(buffer(4)),
            3: VulkanResource.image(material),
        }
        kernel = stack.enter_context(
            VulkanKernel(
                runtime,
                files("ordinarylight.shaders")
                .joinpath("wavefront_indirect_debug.comp.spv")
                .read_bytes(),
                bindings,
                push_constant_size=28,
            )
        )
        init = stack.enter_context(
            VulkanKernel(
                runtime,
                compile_compute("""#version 460
layout(local_size_x=1) in;
layout(binding=1,rgba16f) writeonly uniform image2D hdr;
layout(binding=3,r32ui) writeonly uniform uimage2D material;
void main(){imageStore(hdr,ivec2(0),vec4(2,3,4,1));imageStore(material,ivec2(0),uvec4(0));}
"""),
                {b: bindings[b] for b in (1, 3)},
            )
        )
        result = buffer(16)
        reader = stack.enter_context(
            VulkanKernel(
                runtime,
                compile_compute("""#version 460
layout(local_size_x=1) in;
layout(binding=0,rgba16f) readonly uniform image2D hdr;
layout(binding=1,std430) buffer Result {vec4 value;};
void main(){value=imageLoad(hdr,ivec2(0));}
"""),
                {0: bindings[1], 1: VulkanResource.buffer(result)},
            )
        )

        def use(r, a):
            return VulkanResourceUse(
                r,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                a,
                vk.VK_IMAGE_LAYOUT_GENERAL if r.kind == "image" else None,
            )

        graph = VulkanGraph().add(
            "initialize",
            VulkanPass(
                "init",
                tuple(use(bindings[b], vk.VK_ACCESS_SHADER_WRITE_BIT) for b in (1, 3)),
                init.bind,
                (1, 1, 1),
            ),
        )
        graph.add(
            "apply",
            indirect_apply_operation(
                kernel, output_extent=(1, 1), reservoir_extent=(1, 1), mode=mode
            ),
            after=("initialize",),
        )
        graph.add(
            "read",
            VulkanPass(
                "read",
                (
                    use(bindings[1], vk.VK_ACCESS_SHADER_READ_BIT),
                    use(VulkanResource.buffer(result), vk.VK_ACCESS_SHADER_WRITE_BIT),
                ),
                reader.bind,
                (1, 1, 1),
            ),
            after=("apply",),
        )
        graph.compile().execute(runtime).wait()
        np.testing.assert_allclose(
            np.frombuffer(result.read(), np.float32),
            [2, 3, 4, 1] if mode == "apply" else [0.04, 0.04, 0.04, 1],
            atol=0.0001,
        )
        with pytest.raises(ValueError, match="mode"):
            indirect_apply_operation(
                kernel, output_extent=(1, 1), reservoir_extent=(1, 1), mode="unknown"
            )
