"""Signal scatter and camera reprojection from known native path records."""

from contextlib import ExitStack
import os
import struct
import numpy as np
import pytest
import vulkan as vk
from ordinarylight.runtime import (
    VulkanRuntime,
    VulkanRelaxPrepare,
    VulkanRelaxSpatial,
    VulkanKernel,
    compile_compute,
)
from ordinarylight.pipeline.graph import VulkanGraph
from ordinarylight.pipeline.vulkan import VulkanResource, VulkanResourceUse, VulkanPass
from ordinarylight.wavefront import HOT_PATH_STATE_DTYPE, SECONDARY_PATH_STATE_DTYPE


@pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_GRAPH") != "1", reason="opt-in GPU"
)
@pytest.mark.parametrize("sampled", [False, True])
@pytest.mark.parametrize("custom_history", [False, True])
def test_signal_scatter_and_static_motion(sampled, custom_history):
    with VulkanRuntime() as runtime, ExitStack() as stack:

        def buffer(data):
            return stack.enter_context(runtime.buffer(len(data), data=data))

        paths_data = np.zeros(1, HOT_PATH_STATE_DTYPE)
        paths_data["radiance"][0, :3] = [3, 6, 9]
        secondary_data = np.zeros(1, SECONDARY_PATH_STATE_DTYPE)
        secondary_data["primary_position"][0] = [0, 0, 0, 1.5]
        secondary_data["primary_radiance"][0] = [3, 6, 9, 1 / 3]
        secondary_data["diffuse_radiance_hit_distance"][0] = [2, 4, 6, 0]
        secondary_data["specular_radiance_hit_distance"][0] = [1, 2, 3, 0]
        secondary_data["primary_geometry"].view(np.uint32)[0, 3] = 7
        paths, secondary = (
            buffer(paths_data.tobytes()),
            buffer(secondary_data.tobytes()),
        )
        camera = buffer(
            struct.pack("16f", 0, 0, 2, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 1, 0, 0)
        )
        identity = (17, 42, 4, 71)
        expected_identity = 7
        if custom_history:
            expected_identity = 2166136261
            for word in identity:
                expected_identity = ((expected_identity ^ word) * 16777619) & 0xffffffff
            # The triangle address is deliberately unusable in this variant.
            secondary_data['primary_geometry'].view(np.uint32)[0, 0] = 0xfffffffd
            secondary.upload(secondary_data)
        vertices = buffer(struct.pack('4I4f', *identity, 0, 0, 0, 1) if custom_history else bytes(48))
        names = (
            "packed_normal",
            "packed_material",
            "diffuse",
            "specular",
            "normal_roughness",
            "view_z",
            "motion",
            "identity",
        )
        formats = (
            (vk.VK_FORMAT_R32_UINT,) * 2
            + (vk.VK_FORMAT_R16G16B16A16_SFLOAT,) * 3
            + (
                vk.VK_FORMAT_R32_SFLOAT,
                vk.VK_FORMAT_R16G16B16A16_SFLOAT,
                vk.VK_FORMAT_R32_UINT,
            )
        )
        images = {
            name: stack.enter_context(runtime.image(1, 1, format=format))
            for name, format in zip(names, formats)
        }
        resources = {
            i: VulkanResource.image(images[name]) for i, name in enumerate(names)
        }
        stage = stack.enter_context(
            VulkanRelaxPrepare(
                runtime,
                paths=paths,
                secondary_paths=secondary,
                current_camera=camera,
                previous_camera=camera,
                previous_vertices=vertices,
                custom_history=custom_history,
                capacity=1,
                **images,
            )
        )
        hdr = stack.enter_context(
            runtime.image(1, 1, format=vk.VK_FORMAT_R16G16B16A16_SFLOAT)
        )
        hdr_resource = VulkanResource.image(hdr)
        spatial = stack.enter_context(
            VulkanRelaxSpatial(
                runtime,
                diffuse=images["diffuse"],
                specular=images["specular"],
                normal_roughness=images["normal_roughness"],
                view_z=images["view_z"],
                material=images["identity"],
                output=hdr,
                iterations=1,
            )
        )
        writer = stack.enter_context(
            VulkanKernel(
                runtime,
                compile_compute("""#version 460
layout(local_size_x=1) in;
layout(binding=0,r32ui) writeonly uniform uimage2D normal_image;
layout(binding=1,r32ui) writeonly uniform uimage2D material_image;
layout(binding=8,rgba16f) writeonly uniform image2D hdr_image;
void main(){imageStore(hdr_image,ivec2(0),vec4(0));imageStore(normal_image,ivec2(0),uvec4(16384u|(16384u<<15)));
imageStore(material_image,ivec2(0),uvec4(0));}
"""),
                {0: resources[0], 1: resources[1], 8: hdr_resource},
            )
        )
        result = buffer(bytes(7 * 16))
        declarations = "\n".join(
            f"layout(binding={i},{fmt}) readonly uniform {typ}image2D image_{i};"
            for i, fmt, typ in [
                (2, "rgba16f", ""),
                (3, "rgba16f", ""),
                (4, "rgba16f", ""),
                (5, "r32f", ""),
                (6, "rgba16f", ""),
                (7, "r32ui", "u"),
            ]
        )
        stores = "\n".join(
            f"values[{i - 2}]=vec4(imageLoad(image_{i},ivec2(0)));" for i in range(2, 8)
        )
        reader = stack.enter_context(
            VulkanKernel(
                runtime,
                compile_compute(
                    "#version 460\nlayout(local_size_x=1) in;\n"
                    + declarations
                    + "\nlayout(binding=9,rgba16f) readonly uniform image2D hdr_image;"
                    + "\nlayout(binding=8,std430) buffer Result {vec4 values[];};\nvoid main(){"
                    + stores
                    + "values[6]=imageLoad(hdr_image,ivec2(0));"
                    + "}"
                ),
                {
                    **{i: resources[i] for i in range(2, 8)},
                    8: VulkanResource.buffer(result),
                    9: hdr_resource,
                },
            )
        )

        def use(resource, access):
            return VulkanResourceUse(
                resource,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                access,
                vk.VK_IMAGE_LAYOUT_GENERAL if resource.kind == "image" else None,
            )

        graph = VulkanGraph().add(
            "initialize",
            VulkanPass(
                "init",
                tuple(use(resources[i], vk.VK_ACCESS_SHADER_WRITE_BIT) for i in (0, 1))
                + (use(hdr_resource, vk.VK_ACCESS_SHADER_WRITE_BIT),),
                writer.bind,
                (1, 1, 1),
            ),
        )
        graph.add("prepare", stage.operation(path_count=1, sampled_indirect=sampled))
        graph.add("spatial", spatial.operation(), after=("initialize", "prepare"))
        graph.add(
            "read",
            VulkanPass(
                "read",
                tuple(
                    use(resources[i], vk.VK_ACCESS_SHADER_READ_BIT) for i in range(2, 8)
                )
                + (
                    use(VulkanResource.buffer(result), vk.VK_ACCESS_SHADER_WRITE_BIT),
                    use(hdr_resource, vk.VK_ACCESS_SHADER_READ_BIT),
                ),
                reader.bind,
                (1, 1, 1),
            ),
        )
        graph.compile().execute(runtime).wait()
        values = np.frombuffer(result.read(), np.float32).reshape(7, 4)
        np.testing.assert_allclose(values[0], [2, 4, 6, 0], atol=0.004)
        np.testing.assert_allclose(values[1], [1, 2, 3, 0], atol=0.004)
        np.testing.assert_allclose(values[2], [0, 0, 1, 0.5], atol=0.001)
        assert values[3, 0] == 2
        np.testing.assert_allclose(values[4], [0, 0, 2, 0], atol=0.001)
        assert values[5, 0] == np.float32(expected_identity)
        np.testing.assert_allclose(values[6, :3], [3, 6, 9], atol=0.02)
        if custom_history:
            # A previous point shifted by 0.25 projects by 0.0625 pixels at
            # depth 2 with this camera's unit vertical scale and unit extent.
            vertices.upload(struct.pack('4I4f', *identity, .25, 0, 0, 1))
            graph.compile().execute(runtime).wait()
            moved = np.frombuffer(result.read(), np.float32).reshape(7, 4)
            np.testing.assert_allclose(moved[4], [.0625, 0, 2, 0], atol=.001)
            vertices.upload(struct.pack('4I4f', *identity, .25, 0, 0, 0))
            graph.compile().execute(runtime).wait()
            rejected = np.frombuffer(result.read(), np.float32).reshape(7, 4)
            np.testing.assert_allclose(rejected[4], 0, atol=.001)
        with pytest.raises(ValueError, match="capacity"):
            stage.operation(path_count=2)
        with pytest.raises(RuntimeError, match="borrowers"):
            images["diffuse"].close()
