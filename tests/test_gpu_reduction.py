"""GPU-authored maps are validated before indirect transport and reduction."""

import os
import struct
import numpy as np
import pytest
import ordinarylight as ol
from ordinarylight.geometry import SdfSphere
from ordinarylight.transport import (
    GpuSampleReduction,
    GpuTransportSamples,
    GpuSampleAccumulator,
    VulkanTransportScene,
    VulkanTransportIntegrator,
    TransportMaterial,
    SampleReduction,
)
from ordinarylight.runtime import VulkanKernel, compile_compute
from ordinarylight.pipeline.vulkan import VulkanResource, VulkanResourceUse, VulkanPass
from ordinarylight.pipeline.graph import VulkanGraph
import vulkan as vk

pytestmark = pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_TRANSPORT") != "1",
    reason="opt-in GPU reduction validation",
)

PRODUCER = """
#version 460
layout(local_size_x=1) in;
struct Input { vec4 position; vec4 gn; vec4 sn; vec4 incoming; uvec4 identity; uvec4 media; };
layout(set=0,binding=0,std430) writeonly buffer Samples { Input samples[]; };
layout(set=0,binding=1,std430) buffer Counts { uvec4 counts; };
layout(set=0,binding=2,std430) buffer Groups { uvec4 groups[]; };
layout(set=0,binding=3,std430) buffer Indices { uint indices[]; };
layout(set=0,binding=4,std430) buffer Weights { vec2 weights[]; };
layout(push_constant) uniform Constants { uint sample_count; uint bad; } pc;
void main() {
    counts=uvec4(pc.sample_count,pc.sample_count/2u,0,0);
    for(uint i=0u;i<4u;++i) {
        samples[i]=Input(vec4(0),vec4(0,0,1,0),vec4(0,0,1,0),vec4(0,0,-1,0),
                        uvec4(i/2u,i,i%2u,1),uvec4(0,0,0xffffffffu,0));
        indices[i]=i;
        weights[i]=vec2(i==1u||i==2u?3:1);
    }
    groups[0]=uvec4(0,0,2,0); groups[1]=uvec4(1,2,2,0);
    if(pc.bad==1u) counts.x=5u;
    if(pc.bad==2u) groups[1].y=3u;
    if(pc.bad==3u) indices[1]=0u;
    if(pc.bad==4u) samples[1].identity.x=1u;
    if(pc.bad==5u) weights[0]=vec2(uintBitsToFloat(0x7fc00000u));
    if(pc.bad==6u) groups[1].x=99u;
    if(pc.bad==7u) groups[1].x=0u;
    if(pc.bad==8u) indices[0]=0xffffffffu;
    if(pc.bad==9u) groups[0].z=0xffffffffu;
}
"""


@pytest.fixture
def workload():
    from contextlib import ExitStack

    with ExitStack() as stack:
        runtime = stack.enter_context(ol.VulkanRuntime())
        samples = stack.enter_context(GpuTransportSamples(runtime, 4))
        mapping = stack.enter_context(GpuSampleReduction(runtime, 4, group_capacity=2))
        output = stack.enter_context(GpuSampleAccumulator(runtime, 2))
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
        integrator = stack.enter_context(
            VulkanTransportIntegrator(scene, samples, output, reduction=mapping)
        )
        buffers = (
            samples.buffer,
            mapping.counts,
            mapping.groups,
            mapping.indices,
            mapping.weights,
        )
        bindings = {
            i: VulkanResource.buffer(buffer) for i, buffer in enumerate(buffers)
        }
        producer = stack.enter_context(
            VulkanKernel(
                runtime, compile_compute(PRODUCER), bindings, push_constant_size=8
            )
        )
        yield runtime, samples, mapping, output, integrator, producer, bindings


def schedule(workload, active=4, bad=0):
    runtime, samples, mapping, output, integrator, producer, bindings = workload

    def record(command):
        producer.bind(command, struct.pack("<2I", active, bad))

    graph = VulkanGraph().add(
        "transport", integrator.accumulate_operation(samples_per_element=3)
    )
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
            record,
            (1, 1, 1),
        ),
    )
    return graph.compile()


def test_gpu_counts_maps_weights_and_zero_work(workload, monkeypatch):
    runtime, samples, mapping, output, integrator, producer, bindings = workload
    for buffer in (
        samples.buffer,
        mapping.counts,
        mapping.groups,
        mapping.indices,
        mapping.weights,
    ):
        monkeypatch.setattr(buffer, "read", lambda: pytest.fail("GPU input readback"))
    monkeypatch.setattr(
        SampleReduction, "pack", lambda *args: pytest.fail("host grouping")
    )
    for active in [4, 2, 0, 4]:
        output.reset().wait()
        graph = schedule(workload, active)
        assert graph.order.index("produce") < graph.order.index("transport")
        graph.execute(runtime).wait()
        expected = [[0.25, 0, 0.75], [0.75, 0, 0.25]]
        if active < 4:
            expected[1] = [0, 0, 0]
        if active == 0:
            expected[0] = [0, 0, 0]
        np.testing.assert_allclose(output.means(), expected)
        np.testing.assert_array_equal(
            output.read()["counts"][:, 1], [6 if active else 0, 6 if active == 4 else 0]
        )
    with pytest.raises(RuntimeError, match="integrators"):
        mapping.close()


@pytest.mark.parametrize("bad", range(1, 10))
def test_invalid_gpu_map_is_rejected_before_accumulation(workload, bad):
    runtime, _, _, output, _, _, _ = workload
    schedule(workload, bad=bad).execute(runtime).wait()
    records = output.read(strict=False)
    assert np.all(records["counts"][:, 2] == 32)
    assert not records["radiance"].any()
    assert not records["counts"][:, 1].any()
    with pytest.raises(RuntimeError, match="invalid"):
        output.read()
