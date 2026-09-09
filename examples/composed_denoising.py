"""Independent signal preparation → temporal history → spatial HDR example.

Run: python -m examples.composed_denoising --output /tmp/composed-denoising.npz
Uses synthetic static-surface path records, not the native GI renderer.
"""

from contextlib import ExitStack
import argparse
from pathlib import Path
import struct
import numpy as np
import vulkan as vk
from ordinarylight.runtime import (
    VulkanRuntime,
    VulkanRelaxPrepare,
    VulkanRelaxHistory,
    VulkanRelaxTemporal,
    VulkanRelaxSpatial,
    VulkanKernel,
    compile_compute,
)
from ordinarylight.pipeline.graph import VulkanGraph
from ordinarylight.pipeline.vulkan import VulkanResource, VulkanResourceUse, VulkanPass
from ordinarylight.wavefront import HOT_PATH_STATE_DTYPE, SECONDARY_PATH_STATE_DTYPE


def run(*, frames=6, reset_frame=4):
    """Return HDR and history-length measurements; reset both histories mid-run."""
    if frames < 1 or not 0 <= reset_frame < frames:
        raise ValueError("Require positive frames and a reset frame within the run")
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
        vertices = buffer(bytes(48))
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
                capacity=1,
                **images,
            )
        )
        hdr = stack.enter_context(
            runtime.image(1, 1, format=vk.VK_FORMAT_R16G16B16A16_SFLOAT)
        )
        hdr_resource = VulkanResource.image(hdr)
        # The surface and camera are static: guides can safely be shared.
        # Moving scenes must retain separate guides for each history slot.
        histories = [
            stack.enter_context(
                VulkanRelaxHistory(
                    runtime,
                    normal_roughness=images["normal_roughness"],
                    view_z=images["view_z"],
                    material=images["packed_material"],
                    identity=images["identity"],
                )
            )
            for _ in range(2)
        ]
        temporal = [
            stack.enter_context(
                VulkanRelaxTemporal(
                    runtime,
                    diffuse=images["diffuse"],
                    specular=images["specular"],
                    motion=images["motion"],
                    previous=histories[1 - i],
                    output=histories[i],
                )
            )
            for i in range(2)
        ]
        spatial = [
            stack.enter_context(
                VulkanRelaxSpatial(
                    runtime,
                    diffuse=h.diffuse,
                    specular=h.specular,
                    normal_roughness=images["normal_roughness"],
                    view_z=images["view_z"],
                    material=images["packed_material"],
                    output=hdr,
                    iterations=1,
                )
            )
            for h in histories
        ]
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
        result = buffer(bytes(2 * 16))
        reader_code = compile_compute("""#version 460
layout(local_size_x=1) in;
layout(binding=0,rgba16f) readonly uniform image2D hdr_image;
layout(binding=1,r32f) readonly uniform image2D history_length;
layout(binding=2,std430) buffer Result {vec4 values[];};
void main(){values[0]=imageLoad(hdr_image,ivec2(0));
values[1]=imageLoad(history_length,ivec2(0));}
""")
        readers = [
            stack.enter_context(
                VulkanKernel(
                    runtime,
                    reader_code,
                    {
                        0: hdr_resource,
                        1: VulkanResource.image(h.diffuse_length),
                        2: VulkanResource.buffer(result),
                    },
                )
            )
            for h in histories
        ]

        def use(resource, access):
            return VulkanResourceUse(
                resource,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                access,
                vk.VK_IMAGE_LAYOUT_GENERAL if resource.kind == "image" else None,
            )

        measurements = []
        for frame in range(frames):
            slot = frame % 2
            if frame == reset_frame:
                for history in histories:
                    history.reset()
            graph = VulkanGraph().add(
                "initialize",
                VulkanPass(
                    "init",
                    tuple(
                        use(resources[i], vk.VK_ACCESS_SHADER_WRITE_BIT) for i in (0, 1)
                    )
                    + (use(hdr_resource, vk.VK_ACCESS_SHADER_WRITE_BIT),),
                    writer.bind,
                    (1, 1, 1),
                ),
            )
            graph.add("prepare", stage.operation(path_count=1))
            graph.add("temporal", temporal[slot].operation(), after=("prepare",))
            graph.add(
                "spatial", spatial[slot].operation(), after=("initialize", "temporal")
            )
            graph.add(
                "read",
                VulkanPass(
                    "read",
                    (
                        use(
                            VulkanResource.buffer(result), vk.VK_ACCESS_SHADER_WRITE_BIT
                        ),
                        use(hdr_resource, vk.VK_ACCESS_SHADER_READ_BIT),
                        use(
                            VulkanResource.image(histories[slot].diffuse_length),
                            vk.VK_ACCESS_SHADER_READ_BIT,
                        ),
                    ),
                    readers[slot].bind,
                    (1, 1, 1),
                ),
            )
            graph.compile().execute(runtime).wait()
            values = np.frombuffer(result.read(), np.float32).reshape(2, 4)
            measurements.append(np.concatenate((values[0, :3], values[1, :1])))
        return np.array(measurements)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("composed-denoising.npz"))
    args = parser.parse_args()
    measurements = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, hdr=measurements[:, :3], history_length=measurements[:, 3])
    for frame, (r, g, b, length) in enumerate(measurements):
        print(f"frame {frame}: HDR=({r:.3f}, {g:.3f}, {b:.3f}), history={length:.0f}")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
