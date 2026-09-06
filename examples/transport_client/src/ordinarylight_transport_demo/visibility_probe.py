"""GPU timestamp and host timing breakdown for persistent first-hit workloads."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time

import numpy as np
import vulkan as vk
from ordinarylight.geometry import BoxBatch
from ordinarylight.runtime import VulkanRuntime
from ordinarylight.pipeline.graph import VulkanGraph, VulkanOperation
from ordinarylight.pipeline.vulkan import VulkanPass
from ordinarylight.transport import (
    VulkanRayQuery,
    VulkanTransportScene,
    VulkanTransportIntegrator,
    GpuSampleAccumulator,
    TransportMaterial,
    ray_samples,
)


class TimedSchedule:
    """Serial benchmark helper; each query pool is reused only after completion."""

    def __init__(self, runtime, operation):
        self.runtime = runtime
        properties = vk.vkGetPhysicalDeviceProperties(runtime.physical_device)
        family = vk.vkGetPhysicalDeviceQueueFamilyProperties(runtime.physical_device)[
            runtime.queue_family
        ]
        if not family.timestampValidBits:
            raise RuntimeError("This queue does not support GPU timestamps")
        self.period = properties.limits.timestampPeriod
        self.mask = (1 << family.timestampValidBits) - 1
        self.names = [p.name for p in operation.passes]
        self.pool = vk.vkCreateQueryPool(
            runtime.device,
            vk.VkQueryPoolCreateInfo(
                queryType=vk.VK_QUERY_TYPE_TIMESTAMP, queryCount=2 * len(self.names)
            ),
            None,
        )
        self.last = None
        try:
            passes = []
            for i, stage in enumerate(operation.passes):

                def record(command, stage=stage, i=i):
                    if i == 0:
                        vk.vkCmdResetQueryPool(
                            command, self.pool, 0, 2 * len(self.names)
                        )
                    vk.vkCmdWriteTimestamp(
                        command, vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, self.pool, i * 2
                    )
                    stage.record(command)
                    if stage.workgroups is not None:
                        vk.vkCmdDispatch(command, *stage.workgroups)
                    vk.vkCmdWriteTimestamp(
                        command,
                        vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                        self.pool,
                        i * 2 + 1,
                    )

                passes.append(VulkanPass(stage.name, stage.uses, record))
            wrapped = VulkanOperation(
                passes,
                validate=operation.validate,
                prepare=operation.prepare,
                dependencies=operation.dependencies,
                submitted=operation.submitted,
                wait_semaphores=operation.wait_semaphores,
                signal_semaphores=operation.signal_semaphores,
            )
            self.graph = VulkanGraph().add("measured", wrapped)
            self.compiled = self.graph.compile()
        except Exception:
            self.close()
            raise

    def execute(self, *, recompile=False):
        start = time.perf_counter()
        compiled = self.graph.compile() if recompile else self.compiled
        self.last = compiled.execute(self.runtime)
        submitted = time.perf_counter()
        self.last.wait()
        finished = time.perf_counter()
        values = vk.ffi.new("uint64_t[]", len(self.names) * 2)
        vk.vkGetQueryPoolResults(
            self.runtime.device,
            self.pool,
            0,
            len(self.names) * 2,
            vk.ffi.sizeof(values),
            values,
            8,
            vk.VK_QUERY_RESULT_64_BIT | vk.VK_QUERY_RESULT_WAIT_BIT,
        )
        delta = lambda a, b: ((int(b) - int(a)) & self.mask) * self.period / 1e6
        return dict(
            host_submit_ms=(submitted - start) * 1000,
            host_wait_ms=(finished - submitted) * 1000,
            wall_ms=(finished - start) * 1000,
            gpu_total_ms=delta(values[0], values[len(self.names) * 2 - 1]),
            **{
                f"gpu_{name}_ms": delta(values[i * 2], values[i * 2 + 1])
                for i, name in enumerate(self.names)
            },
        )

    def close(self):
        if self.last is not None:
            self.last.wait()
        vk.vkDestroyQueryPool(self.runtime.device, self.pool, None)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def run(
    output="/tmp/visibility-probe.json", boxes=8192, rays=32768, max_boxes=8, repeats=15
):
    if min(boxes, rays, max_boxes) < 1 or boxes > 65536 or repeats < 3:
        raise ValueError(
            "Use positive sizes, boxes <= 65536 and at least three repetitions"
        )
    rng = np.random.default_rng(251)
    centers = rng.uniform(-10, 10, (boxes, 3))
    half = rng.uniform(0.05, 0.25, (boxes, 3))
    batch = BoxBatch(np.stack((centers - half, centers + half), axis=1))
    partition = batch.partition(max_boxes)
    origins = np.column_stack(
        (rng.uniform(-11, 11, rays), rng.uniform(-11, 11, rays), np.full(rays, 15))
    )
    directions = np.tile([0, 0, -1], (rays, 1))
    measurements, schedules, queries, outputs = {}, {}, {}, {}
    with ExitStack() as stack:
        runtime = stack.enter_context(VulkanRuntime())
        device = vk.vkGetPhysicalDeviceProperties(runtime.physical_device)
        storage = stack.enter_context(
            runtime.buffer(batch.records.nbytes, data=batch.records)
        )
        indices = stack.enter_context(
            runtime.buffer(partition.indices.nbytes, data=partition.indices)
        )
        for layout in ("linear", "partitioned", "per_box"):
            resources = {batch.resource_name: storage}
            geometry = [batch.geometry()]
            if layout == "partitioned":
                geometry = partition.geometries
                resources[partition.index_resource_name] = indices
            elif layout == "per_box":
                geometry = [batch.geometry(i, 1) for i in range(boxes)]
            scene = stack.enter_context(
                VulkanTransportScene(
                    runtime,
                    custom_geometry=geometry,
                    custom_resources=resources,
                    custom_materials=[
                        TransportMaterial("emission", emission=(1, 0.5, 0.25))
                    ],
                )
            )
            query = stack.enter_context(VulkanRayQuery(scene, origins, directions))
            queries[layout] = query
            output_buffer = stack.enter_context(GpuSampleAccumulator(runtime, rays))
            outputs[layout] = output_buffer
            transport = stack.enter_context(
                VulkanTransportIntegrator(
                    scene, ray_samples(origins, directions), output_buffer
                )
            )
            for mode, operation in [
                ("visibility", query.operation(max_steps=max(256, boxes))),
                (
                    "transport",
                    transport.accumulate_operation(
                        max_bounces=0, max_steps=max(8192, boxes)
                    ),
                ),
            ]:
                schedules[(layout, mode)] = stack.enter_context(
                    TimedSchedule(runtime, operation)
                )
                measurements[f"{layout}/{mode}"] = []
            measurements[f"{layout}/transport_recompile"] = []
        for iteration in range(repeats + 2):
            for name in rng.permutation(list(measurements)):
                layout, mode = name.split("/")
                schedule = schedules[
                    (layout, "transport" if mode == "transport_recompile" else mode)
                ]
                sample = schedule.execute(recompile=mode == "transport_recompile")
                if iteration >= 2:
                    measurements[name].append(sample)
        reference = queries["linear"].read()
        for layout, query in queries.items():
            actual = query.read()
            assert not actual["boundary"][:, 3].any()
            np.testing.assert_array_equal(
                actual["identity"][:, 2:], reference["identity"][:, 2:]
            )
            np.testing.assert_allclose(
                actual["position_distance"], reference["position_distance"], atol=3e-5
            )
            visible = (actual["identity"][:, 0] != 0).astype(np.float32)
            np.testing.assert_allclose(
                outputs[layout].means(), visible[:, None] * [1, 0.5, 0.25], atol=1e-6
            )
        report = dict(
            device=device.deviceName,
            driver_version=int(device.driverVersion),
            boxes=boxes,
            rays=rays,
            groups=len(partition.geometries),
            max_boxes=max_boxes,
            repetitions=repeats,
            warmups=2,
            hit_count=int((reference["identity"][:, 0] != 0).sum()),
            timing="GPU TOP/BOTTOM timestamps; host submit includes recording; readback excluded",
            results={
                name: dict(
                    median={
                        key: float(np.median([s[key] for s in samples]))
                        for key in samples[0]
                    },
                    samples=samples,
                )
                for name, samples in measurements.items()
            },
        )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return {
        **{k: v for k, v in report.items() if k != "results"},
        "medians": {k: v["median"] for k, v in report["results"].items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/visibility-probe.json")
    parser.add_argument("--boxes", type=int, default=8192)
    parser.add_argument("--rays", type=int, default=32768)
    parser.add_argument("--max-boxes", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=15)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
