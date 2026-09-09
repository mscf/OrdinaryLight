"""Packaged path resolution: pixel mapping, accumulation and signal capture."""

from contextlib import ExitStack
import os
import numpy as np
import pytest
import vulkan as vk
from ordinarylight.runtime import (
    VulkanRuntime,
    VulkanPathResolve,
    VulkanKernel,
    compile_compute,
)
from ordinarylight.pipeline.graph import VulkanGraph
from ordinarylight.pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse
from ordinarylight.wavefront import HOT_PATH_STATE_DTYPE, SECONDARY_PATH_STATE_DTYPE


@pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_GRAPH") != "1", reason="opt-in GPU"
)
@pytest.mark.parametrize(
    "capture,sampled", [(False, False), (True, False), (True, True)]
)
def test_resolve_accumulation_and_capture(capture, sampled):
    with VulkanRuntime() as runtime, ExitStack() as stack:

        def buffer(size, data=None):
            return stack.enter_context(runtime.buffer(size, data=data))

        records = np.zeros(2, HOT_PATH_STATE_DTYPE)
        records["metadata"][:, 0] = [1, 0]
        records["radiance"][:, :3] = [[2, 4, 6], [4, 8, 12]]
        paths = buffer(records.nbytes, records.tobytes())
        secondary_data = np.zeros(2, SECONDARY_PATH_STATE_DTYPE)
        secondary = buffer(secondary_data.nbytes, secondary_data.tobytes())
        reservoirs = buffer(48, bytes([255]) * 48)
        seeds = buffer(8, bytes(8))
        camera = buffer(64, bytes(64))
        hdr = stack.enter_context(
            runtime.image(2, 1, format=vk.VK_FORMAT_R16G16B16A16_SFLOAT)
        )
        stage = stack.enter_context(
            VulkanPathResolve(
                runtime,
                paths=paths,
                hdr=hdr,
                secondary_paths=secondary,
                reservoirs=reservoirs,
                camera=camera,
                seeds=seeds,
                capacity=2,
                reservoir_extent=(2, 1),
            )
        )
        result = buffer(32)
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
layout(binding=1,std430) buffer Result { vec4 values[]; };
void main(){uint x=gl_GlobalInvocationID.x; values[x]=imageLoad(source_image,ivec2(x,0));}
"""),
                {0: image_resource, 1: result_resource},
            )
        )

        def read_operation():
            return VulkanPass(
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
                (2, 1, 1),
            )

        for sample in range(2):
            graph = VulkanGraph().add(
                "resolve",
                stage.operation(
                    path_count=2,
                    sample_index=sample,
                    sample_count=2,
                    capture_secondary=capture,
                    sampled_indirect=sampled,
                ),
            )
            graph.add("read", read_operation())
            graph.compile().execute(runtime).wait()
            expected = np.array([[4, 8, 12], [2, 4, 6]]) * ((sample + 1) / 2)
            pixels = np.frombuffer(result.read(), np.float32).reshape(2, 4)
            np.testing.assert_array_equal(pixels[:, :3], expected)
            np.testing.assert_array_equal(pixels[:, 3], 1)
            if sample == 0:
                assert reservoirs.read() == bytes([255]) * 48
        if capture:
            assert reservoirs.read() == bytes(48)
            signals = np.frombuffer(secondary.read(), SECONDARY_PATH_STATE_DTYPE)
            np.testing.assert_array_equal(
                signals["diffuse_radiance_hit_distance"][:, :3],
                np.zeros((2, 3)) if sampled else records["radiance"][:, :3],
            )
            assert seeds.read() != bytes(8)
        else:
            assert secondary.read() == secondary_data.tobytes()
            assert reservoirs.read() == bytes([255]) * 48
        with pytest.raises(ValueError, match="capacity"):
            stage.operation(path_count=3)
        with pytest.raises(ValueError, match="sample"):
            stage.operation(path_count=2, sample_count=0)
        with pytest.raises(RuntimeError, match="borrowers"):
            paths.close()
