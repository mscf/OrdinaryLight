"""Image-resolution stored-color visibility: full-hit lookup versus fused lookup."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
import vulkan as vk
from PIL import Image
from ordinarylight.geometry import BoxBatch
from ordinarylight.runtime import VulkanRuntime, VulkanKernel, compile_compute
from ordinarylight.pipeline.graph import VulkanOperation
from ordinarylight.pipeline.vulkan import VulkanResource, VulkanPass, VulkanResourceUse
from ordinarylight.transport import (
    VulkanTransportScene,
    TransportMaterial,
    VulkanRayQuery,
    COLOR_HIT_DTYPE,
)
from .visibility_probe import TimedSchedule

LOOKUP = """#version 460
layout(local_size_x=64) in;
struct Hit { vec4 pd; vec4 gn; vec4 sn; uvec4 identity; uvec4 boundary; };
struct ColorHit { vec4 color; uvec4 identity; };
layout(set=0,binding=0,std430) readonly buffer Hits { Hit hits[]; };
layout(set=0,binding=1,std430) readonly buffer Colors { vec4 colors[]; };
layout(set=0,binding=2,std430) writeonly buffer Output { ColorHit result[]; };
void main() {
    uint i=gl_GlobalInvocationID.x; if(i>=uint(hits.length())) return;
    Hit hit=hits[i]; uint status=hit.boundary.w;
    vec4 color=vec4(0,0,0,1);
    if(status==0u && hit.identity.x!=0u) {
        if(hit.identity.z>=uint(colors.length())) status=32u;
        else {
            color=colors[hit.identity.z];
            if(any(isnan(color))||any(isinf(color))) { status=16u; color=vec4(0,0,0,1); }
        }
    }
    result[i].color=color;
    result[i].identity=uvec4(hit.identity.x,hit.identity.z,hit.identity.w,status);
}
"""


def run(
    output="/tmp/color-probe.json",
    width=1280,
    height=720,
    boxes=8192,
    repeats=9,
    memory="device",
):
    if min(width, height, boxes) < 1 or width * height > 2_097_152 or repeats < 3:
        raise ValueError(
            "Use positive dimensions up to 2097152 pixels and at least three rounds"
        )
    rng = np.random.default_rng(251)
    centers = rng.uniform(-10, 10, (boxes, 3))
    half = rng.uniform(0.05, 0.25, (boxes, 3))
    batch = BoxBatch(np.stack((centers - half, centers + half), axis=1))
    partition = batch.partition(8)
    y, x = np.mgrid[:height, :width]
    origins = np.column_stack(
        (
            (x.ravel() + 0.5) / width * 22 - 11,
            (y.ravel() + 0.5) / height * 22 - 11,
            np.full(width * height, 15),
        )
    )
    directions = np.tile([0, 0, -1], (width * height, 1))
    color_values = np.column_stack(
        (rng.uniform(0.05, 1, (boxes, 3)), np.ones(boxes))
    ).astype(np.float32)
    schedules, results, measurements = {}, {}, {}
    with ExitStack() as stack:
        runtime = stack.enter_context(VulkanRuntime())
        storage = stack.enter_context(
            runtime.buffer(batch.records.nbytes, data=batch.records, memory=memory)
        )
        indices = stack.enter_context(
            runtime.buffer(
                partition.indices.nbytes, data=partition.indices, memory=memory
            )
        )
        palette = stack.enter_context(
            runtime.buffer(color_values.nbytes, data=color_values, memory=memory)
        )
        for layout in ("partitioned", "per_box"):
            resources = {batch.resource_name: storage}
            geometry = [batch.geometry(i, 1) for i in range(boxes)]
            if layout == "partitioned":
                geometry = partition.geometries
                resources[partition.index_resource_name] = indices
            scene = stack.enter_context(
                VulkanTransportScene(
                    runtime,
                    custom_geometry=geometry,
                    custom_resources=resources,
                    custom_materials=[TransportMaterial()],
                )
            )
            for mode in ("full_hit_lookup", "fused"):
                query = stack.enter_context(
                    VulkanRayQuery(
                        scene,
                        origins,
                        directions,
                        colors=palette if mode == "fused" else None,
                        memory=memory,
                    )
                )
                operation = query.operation()
                result = query.hits
                if mode == "full_hit_lookup":
                    result = stack.enter_context(
                        runtime.buffer(
                            width * height * COLOR_HIT_DTYPE.itemsize, memory=memory
                        )
                    )
                    bindings = {
                        i: VulkanResource.buffer(buffer)
                        for i, buffer in enumerate((query.hits, palette, result))
                    }
                    kernel = stack.enter_context(
                        VulkanKernel(runtime, compile_compute(LOOKUP), bindings)
                    )
                    uses = tuple(
                        VulkanResourceUse(
                            resource,
                            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            vk.VK_ACCESS_SHADER_WRITE_BIT
                            if i == 2
                            else vk.VK_ACCESS_SHADER_READ_BIT,
                        )
                        for i, resource in bindings.items()
                    )
                    lookup = VulkanPass(
                        "color_lookup",
                        uses,
                        lambda command, kernel=kernel: kernel.bind(command),
                        ((width * height + 63) // 64, 1, 1),
                    )
                    operation = VulkanOperation(
                        [*operation.passes, lookup],
                        validate=operation.validate,
                        dependencies=operation.dependencies,
                        submitted=operation.submitted,
                    )
                name = f"{layout}/{mode}"
                schedules[name] = stack.enter_context(TimedSchedule(runtime, operation))
                results[name] = result
                measurements[name] = []
        for iteration in range(repeats + 2):
            for name in rng.permutation(list(schedules)):
                record = schedules[name].execute()
                if iteration >= 2:
                    measurements[name].append(record)
        reference = None
        for result in results.values():
            actual = np.frombuffer(result.read(), COLOR_HIT_DTYPE)
            assert not actual["identity"][:, 3].any()
            if reference is None:
                reference = actual.copy()
            else:
                np.testing.assert_array_equal(actual, reference)
        # Allocation placement is part of the measurement, not a hidden assumption.
        buffer = next(iter(results.values()))
        report = dict(
            device=vk.vkGetPhysicalDeviceProperties(runtime.physical_device).deviceName,
            width=width,
            height=height,
            boxes=boxes,
            repetitions=repeats,
            warmups=2,
            buffer_memory_flags=buffer.memory_flags,
            memory_policy=memory,
            hit_count=int((reference["identity"][:, 0] != 0).sum()),
            results={
                name: dict(
                    median={
                        key: float(np.median([v[key] for v in values]))
                        for key in values[0]
                    },
                    samples=values,
                )
                for name, values in measurements.items()
            },
        )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    rgb = reference["color"][:, :3].reshape(height, width, 3)
    srgb = np.where(
        rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.maximum(rgb, 0) ** (1 / 2.4) - 0.055
    )
    Image.fromarray(np.uint8(np.clip(srgb, 0, 1) * 255)).save(
        destination.with_suffix(".png")
    )
    return {
        **{k: v for k, v in report.items() if k != "results"},
        "medians": {k: v["median"] for k, v in report["results"].items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory", choices=("host", "device"), default="device")
    parser.add_argument("--output", default="/tmp/color-probe.json")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--boxes", type=int, default=8192)
    parser.add_argument("--repeats", type=int, default=9)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
