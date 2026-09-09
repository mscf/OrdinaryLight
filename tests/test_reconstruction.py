"""Standalone reconstruction settings, array selection and GPU graph behavior."""

from contextlib import ExitStack
import os
import struct
import numpy as np
import pytest
import vulkan as vk

from ordinarylight.pipeline.reconstruction import (
    ReconstructionSettings,
    ReconstructionEffect,
)
from ordinarylight.pipeline.graph import VulkanGraph
from ordinarylight.pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse
from ordinarylight.runtime import (
    VulkanRuntime,
    VulkanKernel,
    VulkanReconstruction,
    compile_compute,
)


def test_reconstruction_packing_and_validation():
    effect = ReconstructionEffect(kind=2, strength=0.5, color=(1, 0, 0, 1))
    constants = ReconstructionSettings(upscale_filter="fsr1-shade").pack(
        (19, 13), effects=(effect,)
    )
    assert len(constants) == 256
    assert struct.unpack_from("2I", constants, 4) == (19, 13)
    assert struct.unpack_from("I", constants, 60) == (4,)
    assert struct.unpack_from("IIfI", constants, 64) == (2, 1, 0.5, 0)
    for kwargs in (
        {"exposure": float("nan")},
        {"history_weight": 2},
        {"upscale_filter": "unknown"},
        {"upscale_filter": "fsr1", "temporal_enabled": True},
    ):
        with pytest.raises(ValueError):
            ReconstructionSettings(**kwargs).pack((19, 13))
    with pytest.raises(ValueError):
        ReconstructionSettings().pack((0, 13))


@pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_GRAPH") != "1", reason="opt-in GPU"
)
@pytest.mark.parametrize(
    "mode", ("bilinear", "clamped-cubic", "fsr1", "fsr1-shade", "temporal", "tint")
)
def test_reconstruction_graph_output_array(mode):
    with VulkanRuntime() as runtime, ExitStack() as stack:

        def image(width, height, fmt):
            return stack.enter_context(runtime.image(width, height, format=fmt))

        hdr = image(7, 5, vk.VK_FORMAT_R16G16B16A16_SFLOAT)
        position = image(7, 5, vk.VK_FORMAT_R32_SFLOAT)
        normal = image(7, 5, vk.VK_FORMAT_R32_UINT)
        material = image(7, 5, vk.VK_FORMAT_R32_UINT)
        old = image(13, 9, vk.VK_FORMAT_B10G11R11_UFLOAT_PACK32)
        history = image(13, 9, vk.VK_FORMAT_B10G11R11_UFLOAT_PACK32)
        outputs = [image(13, 9, vk.VK_FORMAT_R8G8B8A8_UNORM) for _ in range(2)]
        camera = stack.enter_context(
            runtime.buffer(
                64,
                data=struct.pack(
                    "16f", 0, 0, 0, 0, 0, 0, -1, 1, 1, 0, 0, 0, 0, 1, 0, 0
                ),
            )
        )
        resources = {
            i: VulkanResource.image(im)
            for i, im in enumerate(
                (hdr, position, normal, material, old, history, *outputs)
            )
        }
        source = "#version 460\nlayout(local_size_x=8,local_size_y=8) in;\n"
        for i, fmt in enumerate(
            (
                "rgba16f",
                "r32f",
                "r32ui",
                "r32ui",
                "r11f_g11f_b10f",
                "r11f_g11f_b10f",
                "rgba8",
                "rgba8",
            )
        ):
            source += f"layout(binding={i},{fmt}) writeonly uniform {'uimage2D' if 'ui' in fmt else 'image2D'} im{i};\n"
        source += """void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy);
if(all(lessThan(p,imageSize(im0)))){imageStore(im0,p,vec4(.25,.5,.75,1)); imageStore(im1,p,vec4(0));imageStore(im2,p,uvec4(0));imageStore(im3,p,uvec4(0));}
if(all(lessThan(p,imageSize(im4)))){imageStore(im4,p,vec4(0));imageStore(im5,p,vec4(0));imageStore(im6,p,vec4(1,0,0,1));imageStore(im7,p,vec4(0));}}
"""
        if mode == "tint":
            source = source.replace(
                "imageStore(im3,p,uvec4(0))", "imageStore(im3,p,uvec4(536870912))"
            )
        writer = stack.enter_context(
            VulkanKernel(runtime, compile_compute(source), resources)
        )
        stage = stack.enter_context(
            VulkanReconstruction(
                runtime,
                hdr=hdr,
                position=position,
                normal=normal,
                previous_color=old,
                previous_position=position,
                previous_normal=normal,
                history_color=history,
                outputs=outputs,
                previous_camera=camera,
                current_camera=camera,
                material=material,
            )
        )
        result = stack.enter_context(runtime.buffer(13 * 9 * 48))
        result_resource = VulkanResource.buffer(result)
        reader = stack.enter_context(
            VulkanKernel(
                runtime,
                compile_compute("""#version 460
layout(local_size_x=8,local_size_y=8) in;
layout(binding=0,rgba8) readonly uniform image2D first;
layout(binding=1,rgba8) readonly uniform image2D second;
layout(binding=2,std430) buffer Result {vec4 values[];};
layout(binding=3,r11f_g11f_b10f) readonly uniform image2D history;
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy);ivec2 size=imageSize(first);
if(any(greaterThanEqual(p,size)))return;int i=p.y*size.x+p.x;
values[i]=imageLoad(first,p);values[size.x*size.y+i]=imageLoad(second,p);
values[2*size.x*size.y+i]=imageLoad(history,p);}
"""),
                {0: resources[6], 1: resources[7], 2: result_resource, 3: resources[5]},
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
            "initialize",
            VulkanPass(
                "init",
                tuple(
                    use(r, vk.VK_ACCESS_SHADER_WRITE_BIT) for r in resources.values()
                ),
                writer.bind,
                (2, 2, 1),
            ),
        )
        graph.add(
            "reconstruct",
            stage.operation(
                settings=ReconstructionSettings(
                    upscale_filter="bilinear" if mode in ("temporal", "tint") else mode,
                    temporal_enabled=mode == "temporal",
                ),
                effects=(
                    ReconstructionEffect(kind=2, strength=0.5, color=(1, 0, 0, 1)),
                )
                if mode == "tint"
                else (),
                output_index=1,
            ),
            after=("initialize",),
        )
        graph.add(
            "read",
            VulkanPass(
                "read",
                (
                    use(resources[6], vk.VK_ACCESS_SHADER_READ_BIT),
                    use(resources[7], vk.VK_ACCESS_SHADER_READ_BIT),
                    use(resources[5], vk.VK_ACCESS_SHADER_READ_BIT),
                    use(result_resource, vk.VK_ACCESS_SHADER_WRITE_BIT),
                ),
                reader.bind,
                (2, 2, 1),
            ),
            after=("reconstruct",),
        )
        graph.compile().execute(runtime).wait()
        data = np.frombuffer(result.read(), np.float32).reshape(3, 9, 13, 4)
        np.testing.assert_array_equal(data[0], np.tile([1, 0, 0, 1], (9, 13, 1)))
        x = np.array([0.25, 0.5, 0.75])
        linear = np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)
        encoded = np.where(
            linear <= 0.0031308, 12.92 * linear, 1.055 * linear ** (1 / 2.4) - 0.055
        )
        if mode == "tint":
            encoded = encoded * 0.5 + np.array([1, 0, 0]) * 0.5
        np.testing.assert_allclose(
            data[1, :, :, :3], np.tile(encoded, (9, 13, 1)), atol=1 / 255
        )
        np.testing.assert_array_equal(data[1, :, :, 3], 1)
        expected_history = [0.25, 0.5, 0.75] if mode == "temporal" else [0, 0, 0]
        np.testing.assert_array_equal(
            data[2, :, :, :3], np.tile(expected_history, (9, 13, 1))
        )
        with pytest.raises(RuntimeError):
            outputs[0].close()
