"""GPU-generated samples, changing active counts, and weighted output groups."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import struct

import numpy as np
import vulkan as vk
import ordinarylight as ol
from ordinarylight.geometry import SdfSphere
from ordinarylight.runtime import VulkanKernel, compile_compute
from ordinarylight.pipeline.vulkan import VulkanResource, VulkanResourceUse, VulkanPass
from ordinarylight.pipeline.graph import VulkanGraph
from ordinarylight.transport import (
    GpuSampleReduction,
    GpuTransportSamples,
    GpuSampleAccumulator,
    VulkanTransportScene,
    VulkanTransportIntegrator,
    TransportMaterial,
)

PRODUCER = """
#version 460
layout(local_size_x=1) in;
struct Input { vec4 position; vec4 gn; vec4 sn; vec4 incoming; uvec4 identity; uvec4 media; };
layout(set=0,binding=0,std430) writeonly buffer Samples { Input samples[]; };
layout(set=0,binding=1,std430) writeonly buffer Counts { uvec4 counts; };
layout(set=0,binding=2,std430) writeonly buffer Groups { uvec4 groups[]; };
layout(set=0,binding=3,std430) writeonly buffer Indices { uint indices[]; };
layout(set=0,binding=4,std430) writeonly buffer Weights { vec2 weights[]; };
layout(push_constant) uniform Constants { uint frame; } pc;
void main() {
    uint group_count=pc.frame%3u+1u;
    counts=uvec4(group_count*2u,group_count,0,0);
    for(uint group=0u;group<group_count;++group) {
        groups[group]=uvec4(group,group*2u,2,0);
        for(uint face=0u;face<2u;++face) {
            uint slot=group*2u+face;
            samples[slot]=Input(vec4(0),vec4(0,0,1,0),vec4(0,0,1,0),
                vec4(0,0,-1,0),uvec4(group,face,face,1),uvec4(0,0,0xffffffffu,0));
            indices[slot]=slot;
            weights[slot]=vec2(face==0u?1.0:3.0);
        }
    }
}
"""


def run(frames=6, output="/tmp/gpu-reduction.json"):
    if frames < 3:
        raise ValueError("Use at least three frames to exercise every output group")
    with ExitStack() as stack:
        runtime = stack.enter_context(ol.VulkanRuntime())
        inputs = stack.enter_context(GpuTransportSamples(runtime, 2))
        mapping = stack.enter_context(GpuSampleReduction(runtime, 2, group_capacity=1))
        accumulation = stack.enter_context(GpuSampleAccumulator(runtime, 3))
        scene = stack.enter_context(
            VulkanTransportScene(
                runtime,
                custom_geometry=[SdfSphere(center=(100, 0, 0)).geometry()],
                custom_materials=[
                    TransportMaterial("emission", emission=(1, 0, 0)),
                    TransportMaterial("emission", emission=(0, 0, 1)),
                ],
            )
        )
        transport = stack.enter_context(
            VulkanTransportIntegrator(scene, inputs, accumulation, reduction=mapping)
        )
        frame = [0]
        capacities = []
        for frame[0] in range(frames):
            required_groups = frame[0] % 3 + 1
            if required_groups * 2 > transport.capacity:
                transport = stack.enter_context(
                    transport.grow_capacity(
                        required_groups * 2, group_capacity=required_groups
                    )
                )
            capacities.append(transport.capacity)
            mapping = transport.gpu_reduction
            buffers = (
                transport.samples.buffer,
                mapping.counts,
                mapping.groups,
                mapping.indices,
                mapping.weights,
            )
            bindings = {
                i: VulkanResource.buffer(buffer) for i, buffer in enumerate(buffers)
            }
            # Bindings must be rebuilt after migration and released before the
            # next growth or before closing the integrator-owned allocations.
            with VulkanKernel(
                runtime, compile_compute(PRODUCER), bindings, push_constant_size=4
            ) as producer:
                graph = VulkanGraph().add("transport", transport.accumulate_operation())
                graph.add(
                    "produce",
                    VulkanPass(
                        "produce",
                        tuple(
                            VulkanResourceUse(
                                resource,
                                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                vk.VK_ACCESS_SHADER_WRITE_BIT,
                            )
                            for resource in bindings.values()
                        ),
                        lambda command: producer.bind(
                            command, struct.pack("<I", frame[0])
                        ),
                        (1, 1, 1),
                    ),
                )
                schedule = graph.compile()
                schedule.execute(runtime).wait()
        means = accumulation.means()
        np.testing.assert_allclose(means, np.tile([0.25, 0, 0.75], (3, 1)))
        records = accumulation.read()
        report = dict(
            frames=frames,
            capacities=capacities,
            order=schedule.order,
            means=means.tolist(),
            valid_samples=records["counts"][:, 1].tolist(),
            invalid_paths=int(records["counts"][:, 2].sum()),
        )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--output", default="/tmp/gpu-reduction.json")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
