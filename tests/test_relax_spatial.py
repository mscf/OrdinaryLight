"""Independent spatial-stage contracts and opt-in graph GPU validation."""

import os
import struct
from contextlib import ExitStack

import numpy as np
import pytest
import vulkan as vk

from ordinarylight.denoising.spatial import spatial_settings, atrous_constants
from ordinarylight.pipeline.graph import VulkanGraph
from ordinarylight.pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse
from ordinarylight.runtime import (
    VulkanRuntime,
    VulkanKernel,
    VulkanRelaxSpatial,
    compile_compute,
)


def test_spatial_constants_and_validation():
    assert struct.unpack("8f", atrous_constants(19, 13, 0, 1, 4))[-1] == 1
    assert struct.unpack("8f", atrous_constants(19, 13, 1, 1, 4))[2] == 2
    assert struct.unpack("8f", atrous_constants(19, 13, 1, 1, 4))[-1] == 0
    for settings in (
        (0, 13, 3, 4),
        (19, 13, 0, 4),
        (19, 13, 6, 4),
        (19, 13, 3, float("nan")),
        (19, 13, 3, 0),
    ):
        with pytest.raises(ValueError):
            spatial_settings(*settings)


def use(resource, access):
    return VulkanResourceUse(
        resource,
        vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        access,
        vk.VK_IMAGE_LAYOUT_GENERAL if resource.kind == "image" else None,
    )


@pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_GRAPH") != "1",
    reason="opt-in Vulkan GPU test",
)
@pytest.mark.parametrize("iterations", (1, 2, 3, 4, 5))
def test_independent_graph_filter_and_lifetime(iterations):
    width, height = 19, 13
    with VulkanRuntime() as runtime, ExitStack() as stack:
        formats = (vk.VK_FORMAT_R16G16B16A16_SFLOAT,) * 3 + (
            vk.VK_FORMAT_R32_SFLOAT,
            vk.VK_FORMAT_R32_UINT,
            vk.VK_FORMAT_R16G16B16A16_SFLOAT,
        )
        images = [
            stack.enter_context(runtime.image(width, height, format=f)) for f in formats
        ]
        resources = {i: VulkanResource.image(image) for i, image in enumerate(images)}
        writer_source = """#version 460
layout(local_size_x=8,local_size_y=8) in;
layout(binding=0,rgba16f) writeonly uniform image2D d;
layout(binding=1,rgba16f) writeonly uniform image2D s;
layout(binding=2,rgba16f) writeonly uniform image2D n;
layout(binding=3,r32f) writeonly uniform image2D z;
layout(binding=4,r32ui) writeonly uniform uimage2D m;
layout(binding=5,rgba16f) writeonly uniform image2D o;
void main() {
 ivec2 p=ivec2(gl_GlobalInvocationID.xy); if(any(greaterThanEqual(p,imageSize(d)))) return;
 imageStore(d,p,vec4(.25,.5,.75,1)); imageStore(s,p,vec4(.125,.25,.5,1));
 imageStore(n,p,vec4(0,0,1,.5)); imageStore(z,p,vec4(p.x==0 ? 0 : 1));
 imageStore(m,p,uvec4(1)); imageStore(o,p,vec4(2,3,4,1));
}"""
        writer = stack.enter_context(
            VulkanKernel(runtime, compile_compute(writer_source), resources)
        )
        bindings = dict(
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
        )
        with pytest.raises(ValueError, match="alias"):
            VulkanRelaxSpatial(runtime, **{**bindings, "specular": images[0]})
        with pytest.raises(ValueError, match="extent"):
            VulkanRelaxSpatial(runtime, **bindings, extent=(width + 1, height))
        stage = stack.enter_context(
            VulkanRelaxSpatial(runtime, **bindings, iterations=iterations)
        )
        result = stack.enter_context(runtime.buffer(width * height * 16))
        result_resource = VulkanResource.buffer(result)
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
                {0: resources[5], 1: result_resource},
            )
        )
        groups = ((width + 7) // 8, (height + 7) // 8, 1)
        graph = VulkanGraph().add(
            "initialize",
            VulkanPass(
                "init",
                tuple(
                    use(r, vk.VK_ACCESS_SHADER_WRITE_BIT) for r in resources.values()
                ),
                writer.bind,
                groups,
            ),
        )
        if iterations % 2:
            graph.add("filter", stage.filter_operation(), after=("initialize",))
            # Resource hazards infer filter -> compose, without an explicit edge.
            graph.add("spatial", stage.compose_operation(), after=("initialize",))
        else:
            graph.add("spatial", stage.operation(), after=("initialize",))
        graph.add(
            "read",
            VulkanPass(
                "read",
                (
                    use(resources[5], vk.VK_ACCESS_SHADER_READ_BIT),
                    use(result_resource, vk.VK_ACCESS_SHADER_WRITE_BIT),
                ),
                reader.bind,
                groups,
            ),
            after=("spatial",),
        )
        compiled = graph.compile()
        for _ in range(2):
            completion = compiled.execute(runtime)
            completion.wait()
            data = np.frombuffer(result.read(), np.float32).reshape(height, width, 4)
            np.testing.assert_array_equal(
                data[:, 0], np.tile([2, 3, 4, 1], (height, 1))
            )
            np.testing.assert_array_equal(
                data[:, 1:], np.tile([0.375, 0.75, 1.25, 1], (height, width - 1, 1))
            )
        # Descriptor ownership prevents closing an input before the prepared stage.
        with pytest.raises(RuntimeError):
            images[0].close()
        stage.close()
        with pytest.raises(RuntimeError, match="closed"):
            compiled.execute(runtime)
