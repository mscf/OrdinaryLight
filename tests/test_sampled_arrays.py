"""Combined sampler array descriptors, values and shared allocation leases."""

from contextlib import ExitStack
import os
import numpy as np
import pytest
import vulkan as vk
from ordinarylight.runtime import (
    VulkanRuntime,
    VulkanKernel,
    VulkanSampler,
    compile_compute,
)
from ordinarylight.pipeline.graph import VulkanGraph
from ordinarylight.pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse


@pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_GRAPH") != "1", reason="opt-in GPU"
)
def test_sampled_array_elements_and_shared_lifetimes():
    with VulkanRuntime() as runtime, ExitStack() as stack:
        images = [stack.enter_context(runtime.image(1, 1)) for _ in range(2)]
        sampler = stack.enter_context(VulkanSampler(runtime))
        output = stack.enter_context(runtime.buffer(48))
        textures = [VulkanResource.sampled_image(im) for im in images]
        sam = VulkanResource.sampler(sampler)
        producer = stack.enter_context(
            VulkanKernel(
                runtime,
                compile_compute("""#version 460
layout(local_size_x=1) in;
layout(binding=0,rgba32f) writeonly uniform image2D first;
layout(binding=1,rgba32f) writeonly uniform image2D second;
void main(){imageStore(first,ivec2(0),vec4(1,2,3,4));imageStore(second,ivec2(0),vec4(5,6,7,8));}
"""),
                {i: VulkanResource.image(im) for i, im in enumerate(images)},
            )
        )
        source = compile_compute("""#version 460
layout(local_size_x=1) in;
layout(binding=0) uniform sampler2D textures[3];
layout(binding=1,std430) buffer Output {vec4 values[];};
void main(){values[0]=textureLod(textures[0],vec2(.5),0);values[1]=textureLod(textures[1],vec2(.5),0);values[2]=textureLod(textures[2],vec2(.5),0);}
""")
        consumer = stack.enter_context(
            VulkanKernel(
                runtime,
                source,
                {1: VulkanResource.buffer(output)},
                sampled_image_arrays={
                    0: [(textures[0], sam), (textures[1], sam), (textures[0], sam)]
                },
            )
        )

        def use(r, access):
            return VulkanResourceUse(
                r,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                access,
                vk.VK_IMAGE_LAYOUT_GENERAL if r.kind == "image" else None,
            )

        graph = VulkanGraph().add(
            "sample",
            VulkanPass(
                "sample",
                tuple(use(r, vk.VK_ACCESS_SHADER_READ_BIT) for r in textures)
                + (use(VulkanResource.buffer(output), vk.VK_ACCESS_SHADER_WRITE_BIT),),
                consumer.bind,
                (1, 1, 1),
            ),
        )
        graph.add(
            "fill",
            VulkanPass(
                "fill",
                tuple(
                    use(r, vk.VK_ACCESS_SHADER_WRITE_BIT)
                    for r in producer.bindings.values()
                ),
                producer.bind,
                (1, 1, 1),
            ),
        )
        graph.compile().execute(runtime).wait()
        np.testing.assert_array_equal(
            np.frombuffer(output.read(), np.float32),
            [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4],
        )
        with pytest.raises(RuntimeError, match="borrowers"):
            sampler.close()
        with pytest.raises(RuntimeError, match="borrowers"):
            images[0].close()
        for arrays in ({0: []}, {0: [(VulkanResource.image(images[0]), sam)]}):
            with pytest.raises(ValueError):
                VulkanKernel(runtime, source, {}, sampled_image_arrays=arrays)
        with pytest.raises(ValueError, match="overlap"):
            VulkanKernel(
                runtime,
                source,
                {0: VulkanResource.buffer(output)},
                sampled_image_arrays={0: [(textures[0], sam)]},
            )
        consumer.close()
        assert not sampler._borrowers


@pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_GRAPH") != "1", reason="opt-in GPU"
)
def test_resident_volume_array_layout_and_lease():
    import ordinarylight as ol

    scene = ol.Scene()
    scene.add_volume(np.full((2, 2, 2), 0.75, np.float32))
    with (
        VulkanRuntime() as runtime,
        runtime.upload_scene(scene) as resident,
        runtime.buffer(16) as output,
    ):
        pairs = resident.sampled_resources("volumes", count=2)
        assert pairs[0] is pairs[1]
        source = compile_compute("""#version 460
layout(local_size_x=1) in;
layout(binding=0) uniform sampler3D volumes[2];
layout(binding=1,std430) buffer Output {vec4 value;};
void main(){value=vec4(textureLod(volumes[0],vec3(.5),0).r,textureLod(volumes[1],vec3(.5),0).r,0,1);}
""")
        with VulkanKernel(
            runtime,
            source,
            {1: VulkanResource.buffer(output)},
            sampled_image_arrays={0: pairs},
            sampled_image_layouts={0: vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
        ) as kernel:
            with pytest.raises(RuntimeError, match="consumers"):
                resident.close()
            graph = VulkanGraph().add(
                "sample",
                VulkanPass(
                    "sample",
                    (
                        VulkanResourceUse(
                            pairs[0][0],
                            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            vk.VK_ACCESS_SHADER_READ_BIT,
                            vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        ),
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
            graph.compile().execute(runtime).wait()
            np.testing.assert_allclose(
                np.frombuffer(output.read(), np.float32), [0.75, 0.75, 0, 1], atol=1e-6
            )
            assert (
                pairs[0][0].owner.layout == vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            )
        assert not resident._borrowers


@pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_GRAPH") != "1", reason="opt-in GPU"
)
def test_resident_texture_color_space_order():
    import ordinarylight as ol
    from ordinarylight.targets.vulkan import RendererConfig

    scene = ol.Scene()
    texture = ol.Texture(np.full((1, 1, 4), 128, np.uint8))
    scene.add_mesh(
        [[-1, -1, 0], [1, -1, 0], [0, 1, 0]],
        [[0, 1, 2]],
        ol.Material(base_color_texture=texture),
    )
    with VulkanRuntime(
        config=RendererConfig(wavefront_native_textures=True)
    ) as runtime:
        if not runtime.native_textures_supported:
            pytest.skip("Native sampled textures unavailable")
        with (
            runtime.upload_scene(
                scene, config=RendererConfig(wavefront_native_textures=True)
            ) as resident,
            runtime.buffer(16) as result,
        ):
            pairs = resident.sampled_resources("textures", count=3)
            assert pairs[0] is pairs[2]
            with VulkanKernel(
                runtime,
                compile_compute("""#version 460
layout(local_size_x=1) in;
layout(binding=0) uniform sampler2D textures[3];
layout(binding=1,std430) buffer Output {vec4 value;};
void main(){value=vec4(textureLod(textures[0],vec2(.5),0).r,textureLod(textures[1],vec2(.5),0).r,textureLod(textures[2],vec2(.5),0).r,1);}
"""),
                {1: VulkanResource.buffer(result)},
                sampled_image_arrays={0: pairs},
                sampled_image_layouts={0: vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            ) as kernel:
                uses = tuple(
                    VulkanResourceUse(
                        pair[0],
                        vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        vk.VK_ACCESS_SHADER_READ_BIT,
                        vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    )
                    for pair in pairs[:2]
                )
                uses += (
                    VulkanResourceUse(
                        VulkanResource.buffer(result),
                        vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        vk.VK_ACCESS_SHADER_WRITE_BIT,
                    ),
                )
                VulkanGraph().add(
                    "sample", VulkanPass("sample", uses, kernel.bind, (1, 1, 1))
                ).compile().execute(runtime).wait()
                linear = ((128 / 255 + 0.055) / 1.055) ** 2.4
                np.testing.assert_allclose(
                    np.frombuffer(result.read(), np.float32),
                    [linear, 128 / 255, linear, 1],
                    atol=0.001,
                )
