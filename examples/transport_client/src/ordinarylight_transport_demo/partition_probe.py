"""Compare linear box scans with hardware-accelerated spatial groups.

Timings cover host submission and completion, not isolated GPU shader execution.
Pipeline/scene construction and output readback are outside the timed interval.
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time

import numpy as np
from ordinarylight.geometry import BoxBatch
from ordinarylight.runtime import VulkanRuntime
from ordinarylight.transport import (
    VulkanTransportScene,
    VulkanTransportIntegrator,
    GpuSampleAccumulator,
    TransportMaterial,
    ray_samples,
)


def run(
    output="/tmp/partition-probe.json", boxes=2048, rays=8192, max_boxes=8, repeats=9
):
    if min(boxes, rays, max_boxes) < 1 or repeats < 3:
        raise ValueError("Use positive sizes and at least three timing repetitions")
    rng = np.random.default_rng(251)
    centers = rng.uniform(-10, 10, (boxes, 3))
    half = rng.uniform(0.05, 0.25, (boxes, 3))
    batch = BoxBatch(np.stack((centers - half, centers + half), axis=1))
    start = time.perf_counter()
    partition = batch.partition(max_boxes)
    partition_build_ms = (time.perf_counter() - start) * 1000
    origins = np.column_stack(
        (rng.uniform(-11, 11, rays), rng.uniform(-11, 11, rays), np.full(rays, 15))
    )
    inputs = ray_samples(origins, np.tile([0, 0, -1], (rays, 1)))
    timings = {name: [] for name in ("linear", "partitioned", "per_box")}
    with ExitStack() as stack:
        runtime = stack.enter_context(VulkanRuntime())
        storage = stack.enter_context(
            runtime.buffer(batch.records.nbytes, data=batch.records)
        )
        indices = stack.enter_context(
            runtime.buffer(partition.indices.nbytes, data=partition.indices)
        )
        transports, outputs = {}, {}
        for name in timings:
            resources = {batch.resource_name: storage}
            geometry = [batch.geometry()]
            if name == "partitioned":
                geometry = partition.geometries
                resources[partition.index_resource_name] = indices
            elif name == "per_box":
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
            accumulation = stack.enter_context(GpuSampleAccumulator(runtime, rays))
            transports[name] = stack.enter_context(
                VulkanTransportIntegrator(scene, inputs, accumulation)
            )
            outputs[name] = accumulation
        for iteration in range(repeats + 2):
            for name in rng.permutation(list(timings)):
                start = time.perf_counter()
                transports[name].accumulate(
                    max_bounces=0, max_steps=max(8192, boxes)
                ).wait()
                elapsed = (time.perf_counter() - start) * 1000
                if iteration >= 2:
                    timings[name].append(elapsed)
        reference = outputs["linear"].means()
        for accumulation in outputs.values():
            np.testing.assert_allclose(accumulation.means(), reference, atol=1e-6)
        report = dict(
            boxes=boxes,
            rays=rays,
            groups=len(partition.geometries),
            max_boxes=max_boxes,
            repetitions=repeats,
            warmup_repetitions=2,
            partition_build_ms=partition_build_ms,
            hit_count=int(np.count_nonzero(reference[:, 0])),
            invalid_paths={
                name: int(out.read()["counts"][:, 2].sum())
                for name, out in outputs.items()
            },
            submission_completion_ms={
                name: dict(
                    median=float(np.median(values)),
                    minimum=min(values),
                    maximum=max(values),
                    samples=values,
                )
                for name, values in timings.items()
            },
        )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/partition-probe.json")
    parser.add_argument("--boxes", type=int, default=2048)
    parser.add_argument("--rays", type=int, default=8192)
    parser.add_argument("--max-boxes", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=9)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
