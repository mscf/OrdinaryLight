"""Generation, intersection and actual split shading without a GI renderer."""

from contextlib import ExitStack
import os
import struct
import numpy as np
import pytest
import vulkan as vk
import ordinarylight as ol
from ordinarylight.runtime import (
    VulkanRayGeneration,
    VulkanPathResolve,
    VulkanKernel,
    compile_compute,
    VulkanIntersection,
    VulkanQueueDispatch,
    shade_operation,
    split_trace_graph,
    reset_ray_queue,
)
from ordinarylight.pipeline.graph import VulkanGraph, VulkanOperation
from ordinarylight.pipeline.vulkan import VulkanResource, VulkanPass, VulkanResourceUse
from ordinarylight.wavefront import (
    HOT_PATH_STATE_DTYPE,
    HIT_DTYPE,
    SECONDARY_PATH_STATE_DTYPE,
    prepare_shading,
)
from ordinarylight.wavefront.shading import (
    shade_buffer_bindings,
    WavefrontShadeSettings,
)


@pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_GRAPH") != "1", reason="opt-in GPU"
)
@pytest.mark.parametrize("hit", [False, True])
@pytest.mark.parametrize("indirect_dispatch", [False, True])
@pytest.mark.parametrize("bounces", [1, 3])
def test_primary_shading_graph(hit, indirect_dispatch, bounces):
    scene = ol.Scene()
    scene.add_mesh(
        [[-1, -1, 0], [1, -1, 0], [0, 1, 0]],
        [[0, 1, 2]],
        ol.Material(
            base_color=(1, 1, 1), metallic=1, roughness=0, emission=(2, 1, 0.5)
        ),
    )
    with ol.VulkanRuntime() as runtime, ExitStack() as stack:
        resident = stack.enter_context(runtime.upload_scene(scene))

        def buffer(size, data=None):
            return stack.enter_context(runtime.buffer(size, data=data))

        rays, hits, paths, media = buffer(64), buffer(64), buffer(48), buffer(64)
        next_rays = buffer(64, struct.pack("4I", 0, 1, 0, 0) + bytes(48))
        secondary = buffer(
            SECONDARY_PATH_STATE_DTYPE.itemsize,
            bytes(SECONDARY_PATH_STATE_DTYPE.itemsize),
        )
        jitter = struct.unpack("f", struct.pack("I", 0x38003800))[0]
        camera = buffer(
            64,
            struct.pack(
                "16f",
                0 if hit else 4,
                0,
                2,
                0,
                0,
                0,
                -1,
                0,
                1,
                0,
                0,
                jitter,
                0,
                1,
                0,
                0,
            ),
        )
        generator = stack.enter_context(
            VulkanRayGeneration(
                runtime, camera=camera, rays=rays, paths=paths, media=media, capacity=1
            )
        )
        intersection = stack.enter_context(
            VulkanIntersection(
                runtime,
                tlas=resident.resource("tlas"),
                vertices=resident.resource("vertex"),
                rays=rays,
                hits=hits,
                capacity=1,
            )
        )
        named = {
            name: resident.resource(name)
            for name in (
                "material",
                "vertex",
                "attribute",
                "light",
                "area_light",
                "texture",
                "texture_binding",
                "volume_header",
                "volume_scalar",
                "volume_transfer",
                "triangle_volume",
            )
        }
        named.update(
            {
                name: VulkanResource.buffer(value)
                for name, value in dict(
                    hits=hits,
                    rays=rays,
                    paths=paths,
                    next_rays=next_rays,
                    media=media,
                    secondary_paths=secondary,
                ).items()
            }
        )
        bindings = dict(shade_buffer_bindings(named))
        bindings[8] = resident.resource("tlas")
        pairs = resident.sampled_resources("volumes", count=16)
        kernel = stack.enter_context(
            prepare_shading().create_kernel(
                runtime,
                bindings,
                sampled_image_arrays={21: pairs},
                sampled_image_layouts={21: vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL},
            )
        )
        indirect = stack.enter_context(
            runtime.buffer(
                12,
                usage=vk.VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
                | vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            )
        )
        dispatch = {"indirect": indirect} if indirect_dispatch else {"ray_capacity": 1}
        graph = VulkanGraph().add(
            "shade",
            shade_operation(kernel, WavefrontShadeSettings(max_bounces=1), **dispatch),
            after=("generate", "intersect"),
        )
        if indirect_dispatch:
            queue_dispatch = stack.enter_context(
                VulkanQueueDispatch(runtime, queue=rays, arguments=indirect)
            )
            graph.add("dispatch", queue_dispatch.operation())
        graph.add(
            "intersect",
            intersection.operation(indirect=indirect if indirect_dispatch else None),
        )
        graph.add("generate", generator.operation(extent=(1, 1)))
        if bounces > 1:
            reverse = dict(bindings)
            reverse[1], reverse[6] = bindings[6], bindings[1]
            reverse_kernel = stack.enter_context(
                prepare_shading().create_kernel(
                    runtime,
                    reverse,
                    sampled_image_arrays={21: pairs},
                    sampled_image_layouts={
                        21: vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                    },
                )
            )
            dispatches = [
                stack.enter_context(
                    VulkanQueueDispatch(runtime, queue=q, arguments=indirect)
                )
                for q in (rays, next_rays)
            ]
            reverse_intersection = stack.enter_context(
                VulkanIntersection(
                    runtime,
                    tlas=resident.resource("tlas"),
                    vertices=resident.resource("vertex"),
                    rays=next_rays,
                    hits=hits,
                    capacity=1,
                )
            )
            primary_hits = buffer(64)

            def capture(command):
                vk.vkCmdCopyBuffer(
                    command,
                    hits.buffer,
                    primary_hits.buffer,
                    1,
                    [vk.VkBufferCopy(srcOffset=0, dstOffset=0, size=64)],
                )

            primary_guides = VulkanOperation(
                [
                    VulkanPass(
                        "capture_primary",
                        (
                            VulkanResourceUse(
                                VulkanResource.buffer(hits),
                                vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                                vk.VK_ACCESS_TRANSFER_READ_BIT,
                            ),
                            VulkanResourceUse(
                                VulkanResource.buffer(primary_hits),
                                vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                                vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                            ),
                        ),
                        capture,
                    )
                ]
            )
            graph = split_trace_graph(
                generate=generator.operation(extent=(1, 1)),
                queues=(rays, next_rays),
                dispatches=dispatches,
                intersections=(intersection, reverse_intersection),
                kernels=(kernel, reverse_kernel),
                arguments=indirect,
                settings=WavefrontShadeSettings(max_bounces=bounces),
                capacity=1,
                primary_guides=primary_guides,
            )
        compiled = graph.compile()
        assert compiled.order == (
            ("generate",)
            + tuple(
                name
                for b in range(bounces)
                for stage in ("reset", "dispatch", "intersect", "shade")
                for name in (
                    (f"bounce.{b}.{stage}", "primary_guides")
                    if b == 0 and stage == "intersect"
                    else (f"bounce.{b}.{stage}",)
                )
            )
            if bounces > 1
            else (
                ("generate", "dispatch", "intersect", "shade")
                if indirect_dispatch
                else ("generate", "intersect", "shade")
            )
        )
        hdr = stack.enter_context(
            runtime.image(1, 1, format=vk.VK_FORMAT_R16G16B16A16_SFLOAT)
        )
        resolve = stack.enter_context(
            VulkanPathResolve(
                runtime,
                paths=paths,
                hdr=hdr,
                secondary_paths=secondary,
                reservoirs=buffer(24),
                camera=camera,
                seeds=buffer(4),
                capacity=1,
            )
        )
        result = buffer(16)
        image_resource, result_resource = (
            VulkanResource.image(hdr),
            VulkanResource.buffer(result),
        )
        reader = stack.enter_context(
            VulkanKernel(
                runtime,
                compile_compute("""#version 460
layout(local_size_x=1) in;
layout(binding=0,rgba16f) readonly uniform image2D source_image;
layout(binding=1,std430) buffer Result { vec4 value; };
void main(){value=imageLoad(source_image,ivec2(0));}
"""),
                {0: image_resource, 1: result_resource},
            )
        )
        graph.add("resolve", resolve.operation(path_count=1))
        graph.add(
            "read_hdr",
            VulkanPass(
                "read",
                (
                    VulkanResourceUse(
                        image_resource,
                        vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        vk.VK_ACCESS_SHADER_READ_BIT,
                        vk.VK_IMAGE_LAYOUT_GENERAL,
                    ),
                    VulkanResourceUse(
                        result_resource,
                        vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        vk.VK_ACCESS_SHADER_WRITE_BIT,
                    ),
                ),
                reader.bind,
                (1, 1, 1),
            ),
        )
        compiled = graph.compile()
        assert compiled.order[-2:] == ("resolve", "read_hdr")
        compiled.execute(runtime).wait()
        state = np.frombuffer(paths.read(), HOT_PATH_STATE_DTYPE)
        if hit and bounces > 1:
            # A reflected environment contribution proves that the continuation
            # queue was consumed; empty later bounces must preserve it.
            assert np.all(state["radiance"][0, :3] > [2, 1, 0.5])
        else:
            np.testing.assert_allclose(
                state["radiance"][0, :3],
                [2, 1, 0.5] if hit else [0.018, 0.022, 0.032],
                atol=1e-6,
            )
        np.testing.assert_allclose(
            np.frombuffer(result.read(), np.float32)[:3],
            state["radiance"][0, :3],
            rtol=0.001,
            atol=1e-6,
        )
        assert np.frombuffer(result.read(), np.float32)[3] == 1
        if bounces > 1:
            captured = np.frombuffer(primary_hits.read(), HIT_DTYPE, count=1, offset=16)
            assert captured["position_t"][0, 3] == (2 if hit else -1)
            # Later empty bounces reset the shared hit header; capture survives.
            assert struct.unpack_from("I", hits.read())[0] == 0
        first = paths.read()
        if bounces > 1:
            # Independent direct-dispatch submissions provide a scheduling
            # reference without recycled indirect arguments or graph versions.
            generator.operation(extent=(1, 1)).execute(runtime).wait()
            for bounce in range(bounces):
                i = bounce % 2
                reset_ray_queue((next_rays, rays)[i], capacity=1).execute(
                    runtime
                ).wait()
                (intersection, reverse_intersection)[i].operation().execute(
                    runtime
                ).wait()
                shade_operation(
                    (kernel, reverse_kernel)[i],
                    WavefrontShadeSettings(max_bounces=bounces),
                    ray_capacity=1,
                ).execute(runtime).wait()
            assert paths.read() == first
        compiled.execute(runtime).wait()
        assert paths.read() == first
        assert int(state["metadata"][0, 3]) & 1 == 0
        assert struct.unpack_from("I", next_rays.read())[0] == 0
        with pytest.raises(ValueError, match="Select"):
            shade_operation(kernel, WavefrontShadeSettings())
        with pytest.raises(ValueError, match="capacity"):
            shade_operation(kernel, WavefrontShadeSettings(), ray_capacity=2)
