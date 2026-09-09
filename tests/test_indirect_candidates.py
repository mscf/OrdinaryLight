"""Standalone candidate shader with empty screen-space geometry and profiling."""

from contextlib import ExitStack
from importlib.resources import files
import os
import numpy as np
import pytest
import vulkan as vk
import ordinarylight as ol
from ordinarylight.runtime import (
    VulkanKernel,
    indirect_candidates_operation,
    compile_compute,
)
from ordinarylight.pipeline.graph import VulkanGraph
from ordinarylight.pipeline.vulkan import VulkanResource, VulkanResourceUse, VulkanPass


@pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_GRAPH") != "1", reason="opt-in GPU"
)
@pytest.mark.parametrize("profiling", [False, True])
def test_empty_candidate_geometry(profiling):
    scene = ol.Scene()
    scene.add_mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    with ol.VulkanRuntime() as runtime, ExitStack() as stack:
        resident = stack.enter_context(runtime.upload_scene(scene))

        def buffer(size):
            return stack.enter_context(runtime.buffer(size, data=bytes(size)))

        current, previous, camera, counters = (
            buffer(24),
            buffer(24),
            buffer(64),
            buffer(72),
        )
        bindings = {
            0: VulkanResource.buffer(current),
            5: VulkanResource.buffer(previous),
            4: VulkanResource.buffer(camera),
            8: VulkanResource.buffer(camera),
            11: VulkanResource.buffer(counters),
            12: resident.resource("tlas"),
        }
        formats = {
            1: ("rgba16f", vk.VK_FORMAT_R16G16B16A16_SFLOAT),
            2: ("r32f", vk.VK_FORMAT_R32_SFLOAT),
            3: ("r32ui", vk.VK_FORMAT_R32_UINT),
            6: ("r32f", vk.VK_FORMAT_R32_SFLOAT),
            7: ("r32ui", vk.VK_FORMAT_R32_UINT),
            9: ("r32ui", vk.VK_FORMAT_R32_UINT),
            10: ("r32ui", vk.VK_FORMAT_R32_UINT),
        }
        declarations, stores = [], []
        for b, (fmt, value) in formats.items():
            image = stack.enter_context(runtime.image(1, 1, format=value))
            bindings[b] = VulkanResource.image(image)
            integer = fmt == "r32ui"
            declarations.append(
                f"layout(binding={b},{fmt}) writeonly uniform {'u' if integer else ''}image2D i{b};"
            )
            stores.append(
                f"imageStore(i{b},ivec2(0),{'uvec4' if integer else 'vec4'}(0));"
            )
        initializer = stack.enter_context(
            VulkanKernel(
                runtime,
                compile_compute(
                    "#version 460\nlayout(local_size_x=1) in;\n"
                    + "\n".join(declarations)
                    + "\nvoid main(){"
                    + "".join(stores)
                    + "}"
                ),
                {b: bindings[b] for b in formats},
            )
        )
        kernel = stack.enter_context(
            VulkanKernel(
                runtime,
                files("ordinarylight.shaders")
                .joinpath("wavefront_indirect_candidates.comp.spv")
                .read_bytes(),
                bindings,
                push_constant_size=36,
            )
        )
        graph = VulkanGraph().add(
            "initialize",
            VulkanPass(
                "clear",
                tuple(
                    VulkanResourceUse(
                        bindings[b],
                        vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        vk.VK_ACCESS_SHADER_WRITE_BIT,
                        vk.VK_IMAGE_LAYOUT_GENERAL,
                    )
                    for b in formats
                ),
                initializer.bind,
                (1, 1, 1),
            ),
        )
        graph.add(
            "candidates",
            indirect_candidates_operation(
                kernel,
                source_extent=(1, 1),
                reservoir_extent=(1, 1),
                profiling=profiling,
            ),
        )
        graph.compile().execute(runtime).wait()
        assert current.read() == bytes(24)
        if profiling:
            assert np.frombuffer(counters.read(), np.uint32).sum() > 0
        with pytest.raises(ValueError, match="extent"):
            indirect_candidates_operation(
                kernel, source_extent=(0, 1), reservoir_extent=(1, 1)
            )
