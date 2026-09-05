"""GPU occupancy animation in fixed grid slots, using only public APIs."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image
import ordinaryshade as osh
from ordinarylight.geometry import (
    CustomGeometry,
    IntersectionProgram,
    IntersectionResource,
)
from ordinarylight.pipeline.vulkan import VulkanResource
from ordinarylight.pipeline.graph import VulkanGraph, reflected_operation
from ordinarylight.runtime import (
    VulkanRuntime,
    VulkanKernel,
    VulkanOutput,
    VulkanFrameRing,
    compile_compute,
)
from ordinarylight.transport import (
    VulkanTransportScene,
    TransportMaterial,
    VulkanTransportIntegrator,
    GpuSampleAccumulator,
    ray_samples,
)


@osh.structure
class FrameParameters:
    phase: osh.u32


@osh.compute(workgroup_size=(4, 1, 1))
def populate(
    occupancy: osh.storage_buffer(osh.u32, access="write", binding=0),
    parameters: osh.push_constants(FrameParameters),
):
    cell = osh.global_invocation_id.x
    occupancy[cell] = osh.u32(0)
    if cell == parameters.phase % osh.u32(4):
        occupancy[cell] = osh.u32(1)


_INTERSECTION = """
uint gridIntersection(vec3 origin,vec3 direction,float t_min,float t_max,vec4 parameters,
    float tolerance,uint max_steps,out float distance,out vec3 normal) {
    bool found=false;
    distance=t_max; normal=vec3(0,0,1);
    uint begin=uint(parameters.x), count=uint(parameters.y);
    if(begin+count>uint(occupancy.length())) return 2u;
    for(uint cell=begin;cell<begin+count;++cell) {
        if(occupancy[cell]==0u) continue;
        vec3 center=vec3(float(cell)-1.5,0,0);
        vec3 lower=center-vec3(0.45),upper=center+vec3(0.45);
        float near_t=t_min,far_t=distance;
        vec3 near_n=vec3(0),far_n=vec3(0);
        bool intersects=true;
        for(int axis=0;axis<3;++axis) {
            if(abs(direction[axis])<1e-20) {
                if(origin[axis]<lower[axis]||origin[axis]>upper[axis]) intersects=false;
            } else {
                float a=(lower[axis]-origin[axis])/direction[axis];
                float b=(upper[axis]-origin[axis])/direction[axis];
                vec3 n=vec3(0); n[axis]=direction[axis]>0?-1.0:1.0;
                if(min(a,b)>near_t) { near_t=min(a,b); near_n=n; }
                if(max(a,b)<far_t) { far_t=max(a,b); far_n=-n; }
            }
        }
        if(!intersects||near_t>far_t) continue;
        vec3 start=origin+t_min*direction;
        bool inside=all(greaterThanEqual(start,lower))&&all(lessThanEqual(start,upper));
        float candidate=inside?far_t:near_t;
        vec3 candidate_normal=inside?far_n:near_n;
        if(candidate>=t_min&&candidate<=distance&&dot(candidate_normal,candidate_normal)>0.5) {
            found=true; distance=candidate; normal=candidate_normal;
        }
    }
    return found?1u:0u;
}
"""


def run(*, frames=8, output="/tmp/animated-grid.png", present=False):
    if frames < 1:
        raise ValueError("Frame count must be positive")
    width, height = 64, 32
    glfw = window = None
    if present:
        import glfw

        if not glfw.init():
            raise RuntimeError("Cannot initialize GLFW")
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        window = glfw.create_window(
            768, 384, "Fixed grid / GPU occupancy / dynamic chunks", None, None
        )
        if window is None:
            glfw.terminate()
            raise RuntimeError("Cannot create window")
    try:
        with ExitStack() as stack:
            runtime = stack.enter_context(VulkanRuntime(glfw_window=window))
            occupancy = stack.enter_context(runtime.buffer(16))
            shader = osh.compile(populate)
            producer = stack.enter_context(
                VulkanKernel(
                    runtime,
                    compile_compute(shader.source),
                    {0: VulkanResource.buffer(occupancy)},
                    push_constant_size=4,
                )
            )
            program = IntersectionProgram(
                "gridIntersection",
                _INTERSECTION,
                resources=[IntersectionResource("occupancy", element_type="uint")],
            )

            def chunk(number):
                return CustomGeometry(
                    ((-2 + number * 2, -1, -1), (number * 2, 1, 1)),
                    program,
                    (number * 2, 2, 0, 0),
                    identity=number,
                )

            scene = stack.enter_context(
                VulkanTransportScene(
                    runtime,
                    custom_geometry=[chunk(0)],
                    custom_capacity=1,
                    custom_materials=[
                        TransportMaterial("diffuse", (0.5, 0.5, 0.5), (1, 0.4, 0.1))
                    ],
                    custom_resources={"occupancy": occupancy},
                )
            )
            accumulator = stack.enter_context(
                GpuSampleAccumulator(runtime, width * height, extent=(width, height))
            )
            xx, yy = np.meshgrid(
                np.linspace(-2.2, 2.2, width), np.linspace(-1.1, 1.1, height)
            )
            origins = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, 3)))
            transport = stack.enter_context(
                VulkanTransportIntegrator(
                    scene,
                    ray_samples(origins, np.tile([0, 0, -1], (xx.size, 1))),
                    accumulator,
                )
            )
            display = stack.enter_context(VulkanOutput(runtime))
            tone = stack.enter_context(display.prepare(accumulator.hdr))
            ring = stack.enter_context(VulkanFrameRing(runtime, 2))
            hdr, rgba, materials, inputs = (
                accumulator.hdr,
                tone.image,
                scene.resource("materials").handle,
                transport.samples.buffer,
            )
            orders, revisions = [], []
            environment = np.array([0.02, 0.03, 0.05])
            active = {0}
            for frame in range(frames):
                ring.acquire()
                if frame == 2:
                    scene.reserve_custom_geometry(2)
                    active.add(1)
                updates = {}
                if frame == 2:
                    updates[1] = chunk(1)
                if frame == 4:
                    updates[0] = None
                    active.discard(0)
                if frame == 6:
                    updates[0] = chunk(0)
                    active.add(0)
                graph = VulkanGraph()
                # Deliberately add consumers before producers to exercise graph ordering.
                graph.add("tone", tone.operation())
                graph.add("resolve", accumulator.resolve_operation())
                graph.add(
                    "transport",
                    transport.accumulate_operation(
                        samples_per_element=4, max_bounces=4, environment=environment
                    ),
                    after=["invalidate"],
                )
                graph.add("invalidate", accumulator.reset_operation())
                graph.add(
                    "populate_grid",
                    reflected_operation(
                        producer,
                        shader.reflection,
                        workgroups=(1, 1, 1),
                        push_constants=struct.pack("<I", frame),
                    ),
                )
                if updates:
                    graph.add(
                        "update_chunks", scene.update_custom_geometry_operation(updates)
                    )
                if window is not None:
                    glfw.poll_events()
                    operation = display.present_operation(
                        tone, surface_size=glfw.get_framebuffer_size(window)
                    )
                    if operation is not None:
                        graph.add("present", operation)
                try:
                    compiled = graph.compile()
                    ready = ring.submit(compiled)
                except Exception:
                    display.cancel_presentation()
                    ring.cancel()
                    raise
                orders.append(compiled.order)
                revisions.append(scene.geometry_revision)
                if window is not None and glfw.window_should_close(window):
                    break
            ready.wait()
            # Only final verification/export reads back GPU data.
            means = accumulator.means()
            expected = np.tile(environment, (xx.size, 1))
            cell = frame % 4
            visible = (np.abs(xx.ravel() - (cell - 1.5)) <= 0.45) & (
                np.abs(yy.ravel()) <= 0.45
            )
            if cell // 2 in active:
                expected[visible] = np.array([1, 0.4, 0.1]) + 0.5 * environment
            np.testing.assert_allclose(means, expected, rtol=2e-5, atol=1e-6)
            assert accumulator.hdr is hdr and tone.image is rgba
            assert (
                scene.resource("materials").handle == materials
                and transport.samples.buffer is inputs
            )
            pixels = display.read(tone)
            destination = Path(output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            Image.frombytes("RGBA", (width, height), pixels).resize(
                (768, 384), Image.Resampling.NEAREST
            ).save(destination)
            report = dict(
                frames=len(orders),
                custom_capacity=scene.custom_capacity,
                geometry_revisions=revisions,
                graph_orders=orders,
                fixed_grid_cells=4,
                active_chunks=sorted(active),
                persistent_hdr=True,
                persistent_tone_map=True,
                preserved_materials_and_samples=True,
                exact_reference=True,
                invalid_paths=int(accumulator.read()["counts"][:, 2].sum()),
            )
            destination.with_suffix(".json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            return report
    finally:
        if window is not None:
            glfw.destroy_window(window)
            glfw.terminate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--output", default="/tmp/animated-grid.png")
    parser.add_argument("--present", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
