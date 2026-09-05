"""Graph materials and GPU-discovered fixed slots using public APIs only."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
from PIL import Image
import ordinarylight as ol
from ordinarylight.geometry import SdfSphere
from ordinarylight.materials import MaterialGraph, MaterialNode, MaterialResource
from ordinarylight.pipeline.vulkan import VulkanResource, VulkanResourceUse, VulkanPass
from ordinarylight.pipeline.graph import VulkanGraph
from ordinarylight.runtime import (
    VulkanRuntime,
    VulkanKernel,
    VulkanOutput,
    compile_compute,
)
from ordinarylight.transport import (
    GpuCustomGeometry,
    VulkanTransportScene,
    VulkanTransportIntegrator,
    TransportMaterial,
    OpticalMedium,
    MediumBoundary,
    GpuSampleAccumulator,
    SampleReduction,
    ray_samples,
)
import vulkan as vk

_DISCOVER = """#version 460
layout(local_size_x=1) in;
struct Record { vec4 lower; vec4 upper; vec4 parameters; uvec4 metadata; };
layout(set=0,binding=0,std430) writeonly buffer Output { Record records[]; };
void main() {
    for(uint i=0u;i<3u;++i) {
        vec3 center=vec3(float(i)*2.2-2.2,0,0);
        records[i]=Record(vec4(center-vec3(0.85),0),vec4(center+vec3(0.85),0),
            vec4(center,0.85),uvec4(0,i,i==1u?7u:0xffffffffu,i));
    }
}
"""


def run(output="/tmp/material-graph.png", samples=64):
    width, height = 120, 64
    emission = MaterialGraph(
        {
            "gain": MaterialNode("uniform", value="emission", type="vec4"),
            "rgb": MaterialNode("components", ("gain",), value="rgb", type="vec3"),
        },
        {"emission": "rgb"},
        resources=(MaterialResource("emission", "uniform"),),
    )
    metal = MaterialGraph(
        {
            "rough": MaterialNode("constant", value=0.35),
            "metal": MaterialNode("constant", value=1),
        },
        {"roughness": "rough", "metallic": "metal"},
    )
    y, x = np.mgrid[:height, :width]
    origins = np.column_stack(
        (
            (x.ravel() + 0.5 - width / 2) * 7.8 / width,
            (height / 2 - y.ravel() - 0.5) * 4 / height,
            np.full(width * height, 5),
        )
    )
    # Two coverage samples per output, deliberately unequal weights.
    origins = np.repeat(origins, 2, axis=0)
    origins[:, 0] += np.tile([-0.01, 0.01], width * height)
    with ExitStack() as stack:
        runtime = stack.enter_context(VulkanRuntime())
        gain = stack.enter_context(
            runtime.buffer(16, data=np.array([2, 0.35, 0.08, 1], np.float32))
        )
        slots = stack.enter_context(runtime.buffer(3 * 64))
        scene = stack.enter_context(
            VulkanTransportScene(
                runtime,
                custom_capacity=3,
                intersection_programs=[SdfSphere().geometry().program],
                custom_materials=[
                    TransportMaterial("pbr", albedo=(0.8, 0.6, 0.2), program=metal),
                    TransportMaterial("dielectric", roughness=0.2),
                    TransportMaterial("emission", program=emission),
                ],
                media=[OpticalMedium(), OpticalMedium(1.5)],
                boundaries=[MediumBoundary(7, 0, 1)],
                lights=[ol.PointLight((-3, 4, 4), intensity=30)],
                material_resources={"emission": gain},
            )
        )
        accumulation = stack.enter_context(
            GpuSampleAccumulator(runtime, width * height, extent=(width, height))
        )
        transport = stack.enter_context(
            VulkanTransportIntegrator(
                scene,
                ray_samples(origins, np.tile([0, 0, -1], (len(origins), 1))),
                accumulation,
                reduction=SampleReduction(
                    np.repeat(np.arange(width * height), 2),
                    weights=np.tile([1, 3], width * height),
                ),
            )
        )
        updater = stack.enter_context(GpuCustomGeometry(scene, slots))
        producer = stack.enter_context(
            VulkanKernel(
                runtime, compile_compute(_DISCOVER), {0: VulkanResource.buffer(slots)}
            )
        )
        output_service = stack.enter_context(VulkanOutput(runtime))
        tone = stack.enter_context(output_service.prepare(accumulation.hdr))
        schedule = (
            VulkanGraph()
            .add("tone", tone.operation())
            .add("resolve", accumulation.resolve_operation())
        )
        schedule.add(
            "transport",
            transport.accumulate_operation(
                samples_per_element=samples,
                max_bounces=16,
                max_steps=8192,
                environment=(0.3, 0.4, 0.6),
                environment_nee=True,
            ),
        )
        schedule.add("geometry", updater.operation())
        schedule.add(
            "discover",
            VulkanPass(
                "discover",
                (
                    VulkanResourceUse(
                        VulkanResource.buffer(slots),
                        vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        vk.VK_ACCESS_SHADER_WRITE_BIT,
                    ),
                ),
                lambda command: producer.bind(command),
                (1, 1, 1),
            ),
        )
        compiled = schedule.compile()
        compiled.execute(runtime).wait()
        diagnostics = updater.read_diagnostics()
        records = accumulation.read()
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(
            np.frombuffer(output_service.read(tone), np.uint8).reshape(height, width, 4)
        ).save(destination)
        report = dict(
            output=str(destination),
            order=compiled.order,
            geometry_errors=int(diagnostics.sum()),
            invalid_paths=int(records["counts"][:, 2].sum()),
            samples=samples,
        )
        destination.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/material-graph.png")
    parser.add_argument("--samples", type=int, default=64)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
