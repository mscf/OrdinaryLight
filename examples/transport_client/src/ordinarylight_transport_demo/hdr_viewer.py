"""GPU palette -> fused visibility/HDR -> tone mapping -> optional presentation.

Independent boxes and stored application colors demonstrate the public pipeline;
there is no voxel reconstruction or lighting policy in this client.
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import struct
from time import perf_counter

import numpy as np
from PIL import Image
import vulkan as vk
from ordinarylight.geometry import BoxBatch
from ordinarylight.runtime import (
    VulkanRuntime,
    VulkanOutput,
    VulkanKernel,
    compile_compute,
)
from ordinarylight.pipeline.graph import VulkanGraph, VulkanOperation
from ordinarylight.pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse
from ordinarylight.transport import (
    VulkanTransportScene,
    VulkanRayQuery,
    TransportMaterial,
)
from .visibility_probe import TimedSchedule

PALETTE = """#version 460
layout(local_size_x=64) in;
layout(set=0,binding=0,std430) writeonly buffer Palette { vec4 colors[]; };
layout(push_constant) uniform Frame { uint frame; } pc;
void main() {
    uint i=gl_GlobalInvocationID.x; if(i>=uint(colors.length())) return;
    vec3 rgb=fract(float(i)*vec3(.618,.371,.173)+float(pc.frame)*.01);
    colors[i]=vec4(.1+rgb*3.0,1);
}
"""


def run(
    output="/tmp/hdr-viewer.png",
    width=1280,
    height=720,
    boxes=8192,
    frames=12,
    present=False,
    layout="fixed_size",
):
    if min(width, height, boxes, frames) < 1 or width * height > 2097152:
        raise ValueError("Use positive dimensions/counts, at most 2097152 pixels")
    if layout not in ("fixed_size", "fixed_coverage", "overlap"):
        raise ValueError("Unknown box distribution")
    glfw = window = None
    if present:
        import glfw

        if not glfw.init():
            raise RuntimeError("GLFW initialization failed")
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        window = glfw.create_window(
            width, height, "OrdinaryLight stored-color HDR", None, None
        )
        if window is None:
            glfw.terminate()
            raise RuntimeError("Window creation failed")
    try:
        with ExitStack() as stack:
            runtime = stack.enter_context(VulkanRuntime(glfw_window=window))
            setup_seconds = {}
            started = perf_counter()
            rng = np.random.default_rng(251)
            centers = rng.uniform(-10, 10, (boxes, 3))
            half = rng.uniform(0.05, 0.25, (boxes, 3))
            if layout == "fixed_coverage":
                half[:, :2] *= np.sqrt(8192 / boxes)
            elif layout == "overlap":
                centers[:, :2] *= 0.1
                half[:, :2] = rng.uniform(6, 10, (boxes, 2))
            batch = BoxBatch(np.stack((centers - half, centers + half), axis=1))
            setup_seconds["box_generation"] = perf_counter() - started
            started = perf_counter()
            records = stack.enter_context(
                runtime.buffer(
                    batch.records.nbytes, data=batch.records, memory="device"
                )
            )
            palette = stack.enter_context(runtime.buffer(boxes * 16, memory="device"))
            hdr = stack.enter_context(runtime.image(width, height))
            setup_seconds["record_upload_and_targets"] = perf_counter() - started
            started = perf_counter()
            geometry = batch.geometries()
            setup_seconds["geometry_declarations"] = perf_counter() - started
            started = perf_counter()
            scene = stack.enter_context(
                VulkanTransportScene(
                    runtime,
                    custom_geometry=geometry,
                    custom_resources={batch.resource_name: records},
                    custom_materials=[TransportMaterial()],
                )
            )
            setup_seconds["scene_construction"] = perf_counter() - started
            started = perf_counter()
            y, x = np.mgrid[:height, :width]
            origins = np.column_stack(
                (
                    (x.ravel() + 0.5) / width * 22 - 11,
                    (y.ravel() + 0.5) / height * 22 - 11,
                    np.full(width * height, 15),
                )
            )
            query = stack.enter_context(
                VulkanRayQuery(
                    scene,
                    origins,
                    np.tile([0, 0, -1], (width * height, 1)),
                    colors=palette,
                    memory="device",
                    hdr=hdr,
                )
            )
            display = stack.enter_context(VulkanOutput(runtime))
            tone = stack.enter_context(display.prepare(hdr))
            resource = VulkanResource.buffer(palette)
            producer = stack.enter_context(
                VulkanKernel(
                    runtime,
                    compile_compute(PALETTE),
                    {0: resource},
                    push_constant_size=4,
                )
            )
            frame = [0]
            palette_op = VulkanOperation(
                [
                    VulkanPass(
                        "palette",
                        (
                            VulkanResourceUse(
                                resource,
                                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                vk.VK_ACCESS_SHADER_WRITE_BIT,
                            ),
                        ),
                        lambda command: producer.bind(
                            command, struct.pack("I", frame[0])
                        ),
                        ((boxes + 63) // 64, 1, 1),
                    )
                ]
            )
            operations = [palette_op, query.operation(), tone.operation()]
            combined = VulkanOperation(
                [p for op in operations for p in op.passes],
                validate=lambda: [op.validate() for op in operations],
                submitted=lambda completion: [
                    op.submitted(completion) for op in operations
                ],
            )
            timer = (
                stack.enter_context(TimedSchedule(runtime, combined))
                if not present
                else None
            )
            setup_seconds["query_and_output_setup"] = perf_counter() - started
            measurements = []
            submitted_frames = 0
            for frame[0] in range(frames):
                if window is None:
                    sample = timer.execute()
                    if frame[0] >= 2:
                        measurements.append(sample)
                else:
                    glfw.poll_events()
                    if glfw.window_should_close(window):
                        break
                    graph = (
                        VulkanGraph()
                        .add("tone", operations[2])
                        .add("visibility", operations[1])
                        .add("palette", palette_op)
                    )
                    try:
                        presentation = display.present_operation(
                            tone, surface_size=glfw.get_framebuffer_size(window)
                        )
                        if presentation is not None:
                            graph.add("present", presentation)
                        graph.compile().execute(runtime).wait()
                    except Exception:
                        display.cancel_presentation()
                        raise
                submitted_frames += 1
            report = dict(
                width=width,
                height=height,
                boxes=boxes,
                frames=submitted_frames,
                present=present,
                layout=layout,
                device=vk.vkGetPhysicalDeviceProperties(
                    runtime.physical_device
                ).deviceName,
                gpu_medians={
                    key: float(np.median([m[key] for m in measurements]))
                    for key in measurements[0]
                }
                if measurements
                else {},
                timing_samples=measurements,
                setup_seconds=setup_seconds,
            )
            destination = Path(output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if submitted_frames:
                result = query.read()
                report["hit_count"] = int(np.count_nonzero(result["identity"][:, 0]))
                report["invalid_paths"] = int(
                    np.count_nonzero(result["identity"][:, 3])
                )
                if report["invalid_paths"]:
                    raise RuntimeError("Invalid visibility results")
                rgba = np.frombuffer(display.read(tone), np.uint8).reshape(
                    height, width, 4
                )
                Image.fromarray(rgba).save(destination)
            destination.with_suffix(".json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            return {k: v for k, v in report.items() if k != "timing_samples"}
    finally:
        if window is not None:
            glfw.destroy_window(window)
            glfw.terminate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/hdr-viewer.png")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--boxes", type=int, default=8192)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--present", action="store_true")
    parser.add_argument(
        "--layout",
        choices=("fixed_size", "fixed_coverage", "overlap"),
        default="fixed_size",
    )
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
