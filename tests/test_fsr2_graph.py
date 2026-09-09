"""Independent FSR2 graph execution, preparation ABI and resource leases."""

from contextlib import ExitStack
import os
import math
import numpy as np
import pytest
import vulkan as vk
from ordinarylight.runtime import (
    VulkanRuntime,
    VulkanKernel,
    VulkanFsr2,
    compile_compute,
)
from ordinarylight.pipeline.graph import VulkanGraph
from ordinarylight.pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse

pytestmark = pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_GRAPH") != "1",
    reason="opt-in GPU and built FSR2 bridge",
)


def test_independent_fsr2_preparation_history_and_lifetime():
    with VulkanRuntime() as runtime, ExitStack() as stack:
        inputs = []
        for _ in range(2):
            inputs.append(
                {
                    name: stack.enter_context(runtime.image(17, 13, format=fmt))
                    for name, fmt in dict(
                        hdr=vk.VK_FORMAT_R16G16B16A16_SFLOAT,
                        view_z=vk.VK_FORMAT_R32_SFLOAT,
                        motion=vk.VK_FORMAT_R16G16B16A16_SFLOAT,
                        normal_roughness=vk.VK_FORMAT_R16G16B16A16_SFLOAT,
                    ).items()
                }
            )
        writers = []
        source = """#version 460
layout(local_size_x=8,local_size_y=8) in;
layout(binding=0,rgba16f) writeonly uniform image2D hdr;
layout(binding=1,r32f) writeonly uniform image2D z;
layout(binding=2,rgba16f) writeonly uniform image2D motion;
layout(binding=3,rgba16f) writeonly uniform image2D normal;
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy);if(any(greaterThanEqual(p,imageSize(hdr))))return;
imageStore(hdr,p,vec4(.25,.5,.75,1));imageStore(z,p,vec4(2));
imageStore(motion,p,vec4(.25,-.25,2,0));imageStore(normal,p,vec4(0,0,1,.5));}
"""
        for frame in inputs:
            writers.append(
                stack.enter_context(
                    VulkanKernel(
                        runtime,
                        compile_compute(source),
                        {
                            i: VulkanResource.image(v)
                            for i, v in enumerate(frame.values())
                        },
                    )
                )
            )
        stage = stack.enter_context(
            VulkanFsr2(runtime, inputs=inputs, extent=(17, 13), output_extent=(33, 25))
        )
        result = stack.enter_context(runtime.buffer(33 * 25 * 32))
        readers = []
        for slot in range(2):
            readers.append(
                stack.enter_context(
                    VulkanKernel(
                        runtime,
                        compile_compute("""#version 460
layout(local_size_x=8,local_size_y=8) in;
layout(binding=0,rgba16f) readonly uniform image2D hdr;
layout(binding=1,r32f) readonly uniform image2D depth;
layout(binding=2,rg16f) readonly uniform image2D motion;
layout(binding=3,r8) readonly uniform image2D reactive;
layout(binding=4,std430) buffer Result {vec4 values[];};
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy);if(any(greaterThanEqual(p,imageSize(hdr))))return;
int i=p.y*33+p.x;values[i]=imageLoad(hdr,p);
if(all(lessThan(p,imageSize(depth))))values[33*25+p.y*17+p.x]=vec4(imageLoad(depth,p).x,imageLoad(motion,p).xy,imageLoad(reactive,p).x);}
"""),
                        {
                            0: VulkanResource.image(stage.outputs[slot]),
                            **{
                                i + 1: VulkanResource.image(v)
                                for i, v in enumerate(stage.frames[slot][:3])
                            },
                            4: VulkanResource.buffer(result),
                        },
                    )
                )
            )

        def use(resource, access):
            return VulkanResourceUse(
                resource,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                access,
                vk.VK_IMAGE_LAYOUT_GENERAL if resource.kind == "image" else None,
            )

        for sequence in range(4):
            slot = sequence % 2
            op = stage.operation(
                slot=slot,
                jitter=(0.25, -0.25),
                fov_y=math.pi / 3,
                dt_ms=16,
                reset=sequence == 3,
            )
            graph = VulkanGraph().add(
                "write",
                VulkanPass(
                    "write",
                    tuple(
                        use(r, vk.VK_ACCESS_SHADER_WRITE_BIT)
                        for r in writers[slot].bindings.values()
                    ),
                    writers[slot].bind,
                    (3, 2, 1),
                ),
            )
            graph.add("upscale", op)
            graph.add(
                "read",
                VulkanPass(
                    "read",
                    tuple(
                        use(
                            r,
                            vk.VK_ACCESS_SHADER_WRITE_BIT
                            if i == 4
                            else vk.VK_ACCESS_SHADER_READ_BIT,
                        )
                        for i, r in readers[slot].bindings.items()
                    ),
                    readers[slot].bind,
                    (5, 4, 1),
                ),
            )
            graph.compile().execute(runtime).wait()
            assert stage.context.history_valid
            data = np.frombuffer(result.read(), np.float32)
            rgb = data[: 33 * 25 * 4].reshape(25, 33, 4)[..., :3]
            assert np.isfinite(rgb).all()
            np.testing.assert_allclose(
                rgb[4:-4, 4:-4],
                np.broadcast_to([0.25, 0.5, 0.75], rgb[4:-4, 4:-4].shape),
                atol=0.02,
            )
            prep = data[33 * 25 * 4 : 33 * 25 * 4 + 17 * 13 * 4].reshape(-1, 4)
            np.testing.assert_allclose(
                prep, np.tile([(500 - 0.1) / 9999.9, 0, 0, 0], (17 * 13, 1)), atol=1e-6
            )
        double = VulkanGraph()
        for slot in range(2):
            double.add(
                str(slot), stage.operation(slot=slot, jitter=(0, 0), fov_y=1, dt_ms=16)
            )
        with pytest.raises(ValueError, match="Only one frame"):
            double.compile().prepare_recording(runtime)
        with pytest.raises(ValueError, match="Invalid FSR2"):
            stage.operation(jitter=(0, 0), fov_y=0, dt_ms=16)
        with pytest.raises(RuntimeError, match="borrowers"):
            stage.close()
        with pytest.raises(RuntimeError, match="borrowers"):
            inputs[0]["hdr"].close()
        with pytest.raises(RuntimeError, match="replayed"):
            op.execute(runtime)
