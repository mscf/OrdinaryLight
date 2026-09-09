"""Trace and denoise a static triangle while the camera moves.

Run with python -m examples.composed_motion --output /tmp/composed-motion.npz.
This uses independent runtime stages, without a native renderer or window.
"""

import argparse
from contextlib import ExitStack
from pathlib import Path
import struct
import numpy as np
import vulkan as vk
import ordinarylight as ol
from ordinarylight.runtime import (
    VulkanRuntime,
    VulkanRayGeneration,
    VulkanIntersection,
    VulkanQueueDispatch,
    VulkanPathResolve,
    VulkanPrimaryMetadata,
    prepare_primary_metadata_shader,
    VulkanRelaxPrepare,
    VulkanRelaxHistory,
    VulkanRelaxTemporal,
    VulkanRelaxSpatial,
    VulkanKernel,
    compile_compute,
    split_trace_graph,
)
from ordinarylight.wavefront import prepare_shading, prepare_primary_metadata
from ordinarylight.wavefront.shading import (
    shade_buffer_bindings,
    WavefrontShadeSettings,
)
from ordinarylight.pipeline.graph import VulkanGraph, VulkanOperation
from ordinarylight.pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse


def run(frames=6, width=16, height=16):
    if frames < 2 or width < 4 or height < 4:
        raise ValueError("Require at least two frames and a 4x4 extent")
    capacity = width * height
    scene = ol.Scene()
    vertices = np.array([[-20, -20, 0], [20, -20, 0], [0, 20, 0]], np.float32)
    scene.add_mesh(
        vertices, [[0, 1, 2]], ol.Material(roughness=0.35, emission=(2, 1, 0.5))
    )
    prepared_metadata = prepare_primary_metadata(scene)
    metadata_spirv = prepare_primary_metadata_shader()
    with VulkanRuntime() as runtime, ExitStack() as stack:
        resident = stack.enter_context(runtime.upload_scene(scene))

        def buffer(size, data=None, **kwargs):
            return stack.enter_context(runtime.buffer(size, data=data, **kwargs))

        def image(format):
            return stack.enter_context(runtime.image(width, height, format=format))

        def use(resource, access):
            return VulkanResourceUse(
                resource,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                access,
                vk.VK_IMAGE_LAYOUT_GENERAL if resource.kind == "image" else None,
            )

        rays = [buffer(16 + capacity * 48) for _ in range(2)]
        hits, paths, media = (
            buffer(16 + capacity * 48),
            buffer(capacity * 48),
            buffer(capacity * 64),
        )
        secondary = buffer(capacity * 128, bytes(capacity * 128))
        camera, previous_camera = buffer(64), buffer(64)
        previous_vertices = buffer(
            48, np.column_stack((vertices, np.zeros(3))).astype(np.float32).tobytes()
        )
        arguments = buffer(
            12,
            usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | vk.VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
        )
        generator = stack.enter_context(
            VulkanRayGeneration(
                runtime,
                camera=camera,
                rays=rays[0],
                paths=paths,
                media=media,
                capacity=capacity,
            )
        )
        dispatches = [
            stack.enter_context(
                VulkanQueueDispatch(runtime, queue=q, arguments=arguments)
            )
            for q in rays
        ]
        intersections = [
            stack.enter_context(
                VulkanIntersection(
                    runtime,
                    tlas=resident.resource("tlas"),
                    vertices=resident.resource("vertex"),
                    rays=q,
                    hits=hits,
                    capacity=capacity,
                )
            )
            for q in rays
        ]
        common = {
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
        common.update(
            {
                name: VulkanResource.buffer(value)
                for name, value in dict(
                    hits=hits, paths=paths, media=media, secondary_paths=secondary
                ).items()
            }
        )
        kernels = []
        prepared = prepare_shading()
        for i in range(2):
            bindings = dict(
                shade_buffer_bindings(
                    dict(
                        common,
                        rays=VulkanResource.buffer(rays[i]),
                        next_rays=VulkanResource.buffer(rays[1 - i]),
                    )
                )
            )
            bindings[8] = resident.resource("tlas")
            kernels.append(
                stack.enter_context(
                    prepared.create_kernel(
                        runtime,
                        bindings,
                        sampled_image_arrays={
                            21: resident.sampled_resources("volumes", count=16)
                        },
                        sampled_image_layouts={
                            21: vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                        },
                    )
                )
            )
        rgba, uint, scalar = (
            vk.VK_FORMAT_R16G16B16A16_SFLOAT,
            vk.VK_FORMAT_R32_UINT,
            vk.VK_FORMAT_R32_SFLOAT,
        )
        hdr = image(rgba)
        resolve = stack.enter_context(
            VulkanPathResolve(
                runtime,
                paths=paths,
                hdr=hdr,
                secondary_paths=secondary,
                reservoirs=buffer(capacity * 24),
                seeds=buffer(capacity * 4),
                camera=camera,
                capacity=capacity,
                reservoir_extent=(width, height),
            )
        )
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
        formats = (uint, uint, rgba, rgba, rgba, scalar, rgba, uint)
        slots = [dict(zip(names, (image(fmt) for fmt in formats))) for _ in range(2)]
        metadata_buffer = buffer(len(prepared_metadata.data), prepared_metadata.data)
        metadata_stages = [
            stack.enter_context(
                VulkanPrimaryMetadata(
                    runtime,
                    paths=paths,
                    secondary_paths=secondary,
                    metadata=metadata_buffer,
                    material=s["packed_material"],
                    capacity=capacity,
                    triangle_count=prepared_metadata.triangle_count,
                    spirv=metadata_spirv,
                )
            )
            for s in slots
        ]
        preparations = [
            stack.enter_context(
                VulkanRelaxPrepare(
                    runtime,
                    paths=paths,
                    secondary_paths=secondary,
                    current_camera=camera,
                    previous_camera=previous_camera,
                    previous_vertices=previous_vertices,
                    capacity=capacity,
                    **images,
                )
            )
            for images in slots
        ]
        histories = [
            stack.enter_context(
                VulkanRelaxHistory(
                    runtime,
                    normal_roughness=s["normal_roughness"],
                    view_z=s["view_z"],
                    material=s["packed_material"],
                    identity=s["identity"],
                )
            )
            for s in slots
        ]
        temporals = [
            stack.enter_context(
                VulkanRelaxTemporal(
                    runtime,
                    diffuse=s["diffuse"],
                    specular=s["specular"],
                    motion=s["motion"],
                    previous=histories[1 - i],
                    output=histories[i],
                )
            )
            for i, s in enumerate(slots)
        ]
        spatials = [
            stack.enter_context(
                VulkanRelaxSpatial(
                    runtime,
                    diffuse=h.diffuse,
                    specular=h.specular,
                    normal_roughness=s["normal_roughness"],
                    view_z=s["view_z"],
                    material=s["packed_material"],
                    output=hdr,
                    iterations=1,
                )
            )
            for h, s in zip(histories, slots)
        ]
        # Produce geometric normal guides while the primary hit queue is live.
        # Material IDs and base roughness are applied from the prepared scene
        # table after tracing; this pass initializes the guide image.
        init_code = compile_compute("""#version 460
layout(local_size_x=64) in;
layout(binding=0,r32ui) writeonly uniform uimage2D n;
layout(binding=1,r32ui) writeonly uniform uimage2D m;
layout(binding=2,std430) readonly buffer Hits {uint hit_words[];};
layout(binding=3,std430) readonly buffer Paths {uint path_words[];};
void main(){uint i=gl_GlobalInvocationID.x;if(i>=min(hit_words[0],hit_words[1]))return;
uint base=4u+i*12u; uint path=hit_words[base+11u]; uint pixel=path_words[path*12u+8u];
ivec2 size=imageSize(n); if(pixel>=uint(size.x*size.y))return;
ivec2 p=ivec2(pixel%uint(size.x),pixel/uint(size.x));
vec3 normal=vec3(uintBitsToFloat(hit_words[base+4u]),uintBitsToFloat(hit_words[base+5u]),uintBitsToFloat(hit_words[base+6u]));
uint packed=0u;
if(uintBitsToFloat(hit_words[base+3u])>=0.0){
normal/=abs(normal.x)+abs(normal.y)+abs(normal.z);
vec2 oct=normal.xy;if(normal.z<0.0)oct=(1.0-abs(oct.yx))*mix(vec2(-1),vec2(1),greaterThanEqual(oct,vec2(0)));
uvec2 unit=uvec2(round(clamp(oct*.5+.5,0.0,1.0)*32767.0));packed=unit.x|(unit.y<<15u);}
imageStore(n,p,uvec4(packed));imageStore(m,p,uvec4(0));}
""")
        initializers = [
            stack.enter_context(
                VulkanKernel(
                    runtime,
                    init_code,
                    {
                        0: VulkanResource.image(s["packed_normal"]),
                        1: VulkanResource.image(s["packed_material"]),
                        2: VulkanResource.buffer(hits),
                        3: VulkanResource.buffer(paths),
                    },
                )
            )
            for s in slots
        ]
        result = buffer(capacity * 48)
        read_code = compile_compute("""#version 460
layout(local_size_x=8,local_size_y=8) in;
layout(binding=0,rgba16f) readonly uniform image2D hdr;
layout(binding=1,r32f) readonly uniform image2D history;
layout(binding=2,rgba16f) readonly uniform image2D motion;
layout(binding=3,std430) buffer Result {vec4 values[];};
void main(){ivec2 p=ivec2(gl_GlobalInvocationID.xy);ivec2 size=imageSize(hdr);
if(any(greaterThanEqual(p,size)))return;int i=(p.y*size.x+p.x)*3;
values[i]=imageLoad(hdr,p);values[i+1]=imageLoad(history,p);values[i+2]=imageLoad(motion,p);}
""")
        readers = [
            stack.enter_context(
                VulkanKernel(
                    runtime,
                    read_code,
                    {
                        0: VulkanResource.image(hdr),
                        1: VulkanResource.image(h.diffuse_length),
                        2: VulkanResource.image(s["motion"]),
                        3: VulkanResource.buffer(result),
                    },
                )
            )
            for h, s in zip(histories, slots)
        ]
        groups = ((width + 7) // 8, (height + 7) // 8, 1)

        def camera_bytes(x):
            jitter = struct.unpack("f", struct.pack("I", 0x38003800))[0]
            return struct.pack(
                "16f", x, 0, 2, 0, 0, 0, -1, 0, 1, 0, 0, jitter, 0, 1, 0, 0
            )

        captures = []
        for frame in range(frames):
            i = frame % 2
            # Quarter-pixel horizontal displacement per frame at depth two.
            step = 1 / height
            camera.upload(camera_bytes(frame * step))
            previous_camera.upload(camera_bytes(max(0, frame - 1) * step))
            secondary.upload(bytes(capacity * 128))
            trace = split_trace_graph(
                generate=generator.operation(
                    extent=(width, height), capture_secondary=True
                ),
                queues=rays,
                dispatches=dispatches,
                intersections=intersections,
                kernels=kernels,
                arguments=arguments,
                settings=WavefrontShadeSettings(max_bounces=2, capture_secondary=True),
                capacity=capacity,
                primary_guides=VulkanOperation(
                    [
                        VulkanPass(
                            "primary_guides",
                            tuple(
                                use(
                                    r,
                                    vk.VK_ACCESS_SHADER_WRITE_BIT
                                    if b < 2
                                    else vk.VK_ACCESS_SHADER_READ_BIT,
                                )
                                for b, r in initializers[i].bindings.items()
                            ),
                            initializers[i].bind,
                            ((capacity + 63) // 64, 1, 1),
                        )
                    ]
                ),
            )
            trace.compile().execute(runtime).wait()
            # Separate submissions keep explicit trace resource versions local;
            # path and signal data remain resident on the same GPU.
            graph = VulkanGraph().add(
                "resolve",
                resolve.operation(path_count=capacity, capture_secondary=True),
            )
            graph.add(
                "metadata",
                metadata_stages[i].operation(path_count=capacity),
                after=("resolve",),
            )
            graph.add(
                "prepare",
                preparations[i].operation(path_count=capacity),
                after=("metadata",),
            )
            graph.add("temporal", temporals[i].operation(), after=("prepare",))
            graph.add("spatial", spatials[i].operation(), after=("resolve", "temporal"))
            graph.add(
                "read",
                VulkanPass(
                    "read",
                    tuple(
                        use(
                            r,
                            vk.VK_ACCESS_SHADER_WRITE_BIT
                            if b == 3
                            else vk.VK_ACCESS_SHADER_READ_BIT,
                        )
                        for b, r in readers[i].bindings.items()
                    ),
                    readers[i].bind,
                    groups,
                ),
            )
            graph.compile().execute(runtime).wait()
            captures.append(
                np.frombuffer(result.read(), np.float32)
                .reshape(height, width, 3, 4)
                .copy()
            )
        data = np.array(captures)
        return dict(
            hdr=data[:, :, :, 0, :3],
            history_length=data[:, :, :, 1, 0],
            motion=data[:, :, :, 2, :],
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("composed-motion.npz"))
    args = parser.parse_args()
    data = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **data)
    rgb = data["hdr"][-1]
    rgb = np.clip(rgb / (1 + rgb), 0, 1) ** (1 / 2.2)
    preview = args.output.with_suffix(".ppm")
    preview.write_bytes(
        f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode()
        + np.uint8(rgb * 255).tobytes()
    )
    for frame in range(len(data["hdr"])):
        print(
            f"frame {frame}: mean history={data['history_length'][frame].mean():.2f}, "
            f"center motion={data['motion'][frame, 8, 8, :2]}"
        )
    print(f"Saved {args.output} and {preview}")


if __name__ == "__main__":
    main()
