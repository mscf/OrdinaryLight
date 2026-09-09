"""Temporal history submission and graph integration contracts."""

from contextlib import ExitStack
import os
import struct

import numpy as np
import pytest
import vulkan as vk

from ordinarylight.denoising.temporal import temporal_constants
from ordinarylight.pipeline.graph import VulkanGraph
from ordinarylight.pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse
from ordinarylight.runtime import (
    VulkanRuntime,
    VulkanKernel,
    compile_compute,
    VulkanRelaxHistory,
    VulkanRelaxTemporal,
)


def test_policy_validation():
    assert struct.unpack("8f", temporal_constants(19, 13, False))[3] == 0
    assert struct.unpack("8f", temporal_constants(19, 13, True))[3] == 1
    for policy in (
        {"history_limit": 0},
        {"normal_threshold": 2},
        {"depth_threshold": -1},
        {"clamp_sigma": float("nan")},
        {"reactive_sigma": -1},
    ):
        with pytest.raises(ValueError):
            temporal_constants(19, 13, True, **policy)


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
def test_temporal_ring_reset_and_failed_submission(monkeypatch):
    width, height = 19, 13
    with VulkanRuntime() as runtime, ExitStack() as stack:

        def image(fmt):
            return stack.enter_context(runtime.image(width, height, format=fmt))

        rgba, scalar, uint = (
            vk.VK_FORMAT_R16G16B16A16_SFLOAT,
            vk.VK_FORMAT_R32_SFLOAT,
            vk.VK_FORMAT_R32_UINT,
        )
        diffuse, specular, motion = image(rgba), image(rgba), image(rgba)
        # Static guides may be shared between histories; radiance and length may not.
        normal, depth, material, identity = (
            image(rgba),
            image(scalar),
            image(uint),
            image(uint),
        )
        history = [
            stack.enter_context(
                VulkanRelaxHistory(
                    runtime,
                    normal_roughness=normal,
                    view_z=depth,
                    material=material,
                    identity=identity,
                )
            )
            for _ in range(2)
        ]
        resources = {
            i: VulkanResource.image(im)
            for i, im in enumerate(
                (diffuse, specular, motion, normal, depth, material, identity)
            )
        }
        writer = stack.enter_context(
            VulkanKernel(
                runtime,
                compile_compute("""#version 460
layout(local_size_x=8,local_size_y=8) in;
layout(binding=0,rgba16f) writeonly uniform image2D d;
layout(binding=1,rgba16f) writeonly uniform image2D s;
layout(binding=2,rgba16f) writeonly uniform image2D motion;
layout(binding=3,rgba16f) writeonly uniform image2D n;
layout(binding=4,r32f) writeonly uniform image2D z;
layout(binding=5,r32ui) writeonly uniform uimage2D m;
layout(binding=6,r32ui) writeonly uniform uimage2D id;
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy); if(any(greaterThanEqual(p,imageSize(d))))return;
imageStore(d,p,vec4(.25,.5,.75,1)); imageStore(s,p,vec4(.125,.25,.5,1));
imageStore(motion,p,vec4(0,0,1,0)); imageStore(n,p,vec4(0,0,1,.5));
imageStore(z,p,vec4(1)); imageStore(m,p,uvec4(1)); imageStore(id,p,uvec4(7));}
"""),
                resources,
            )
        )
        groups = ((width + 7) // 8, (height + 7) // 8, 1)
        VulkanGraph().add(
            "initialize",
            VulkanPass(
                "init",
                tuple(
                    use(r, vk.VK_ACCESS_SHADER_WRITE_BIT) for r in resources.values()
                ),
                writer.bind,
                groups,
            ),
        ).compile().execute(runtime).wait()
        stages = [
            stack.enter_context(
                VulkanRelaxTemporal(
                    runtime,
                    diffuse=diffuse,
                    specular=specular,
                    motion=motion,
                    previous=history[1 - i],
                    output=history[i],
                )
            )
            for i in range(2)
        ]
        result = stack.enter_context(runtime.buffer(width * height * 16))
        buffer = VulkanResource.buffer(result)
        graphs = []
        for index, stage in enumerate(stages):
            radiance = VulkanResource.image(history[index].diffuse)
            length = VulkanResource.image(history[index].diffuse_length)
            reader = stack.enter_context(
                VulkanKernel(
                    runtime,
                    compile_compute("""#version 460
layout(local_size_x=8,local_size_y=8) in;
layout(binding=0,rgba16f) readonly uniform image2D radiance;
layout(binding=1,r32f) readonly uniform image2D history_length;
layout(binding=2,std430) buffer Result {vec4 values[];};
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy); ivec2 size=imageSize(radiance);
if(any(greaterThanEqual(p,size)))return;
values[p.y*size.x+p.x]=vec4(imageLoad(radiance,p).rgb,imageLoad(history_length,p).r);}
"""),
                    {0: radiance, 1: length, 2: buffer},
                )
            )
            graph = VulkanGraph().add("temporal", stage.operation())
            graph.add(
                "read",
                VulkanPass(
                    "read",
                    (
                        use(radiance, vk.VK_ACCESS_SHADER_READ_BIT),
                        use(length, vk.VK_ACCESS_SHADER_READ_BIT),
                        use(buffer, vk.VK_ACCESS_SHADER_WRITE_BIT),
                    ),
                    reader.bind,
                    groups,
                ),
            )
            graphs.append(graph.compile())
        # Preparing a graph or failing its submission must not publish history.
        assert not any(h.valid for h in history)
        submit = runtime.submit

        def fail(*args, **kwargs):
            raise RuntimeError("injected submit failure")

        monkeypatch.setattr(runtime, "submit", fail)
        with pytest.raises(RuntimeError, match="injected"):
            graphs[0].execute(runtime)
        assert not history[0].valid and history[0].completion is None
        monkeypatch.setattr(runtime, "submit", submit)
        for frame_index in range(4):
            index = frame_index % 2
            graphs[index].execute(runtime).wait()
            assert history[index].valid
            data = np.frombuffer(result.read(), np.float32).reshape(-1, 4)
            np.testing.assert_array_equal(
                data[:, :3], np.tile([0.25, 0.5, 0.75], (width * height, 1))
            )
            np.testing.assert_array_equal(data[:, 3], frame_index + 1)
        previous_completion = history[1].completion
        history[1].reset()
        assert history[1].completion is previous_completion
        graphs[0].execute(runtime).wait()
        np.testing.assert_array_equal(
            np.frombuffer(result.read(), np.float32).reshape(-1, 4)[:, 3], 1
        )
        with pytest.raises(RuntimeError, match="consumers"):
            history[0].close()
        stages[0].close()
        with pytest.raises(RuntimeError, match="closed"):
            graphs[0].execute(runtime)
