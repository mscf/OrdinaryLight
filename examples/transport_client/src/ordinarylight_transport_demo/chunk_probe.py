"""Grouped procedural-box reference versus one custom primitive per box.

This is a correctness probe, not a production sparse traversal algorithm: each
group scans its active records. Boxes must be disjoint, including boundaries.
"""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import time

import numpy as np
from ordinarylight.geometry import (
    BoxBatch,
)
from ordinarylight.materials import MaterialGraph, MaterialNode, MaterialResource
from ordinarylight.runtime import VulkanRuntime
from ordinarylight.transport import (
    VulkanTransportScene,
    VulkanTransportIntegrator,
    GpuSampleAccumulator,
    TransportMaterial,
    OpticalMedium,
    MediumBoundary,
    intersect_rays,
    ray_samples,
)


def fixture():
    records = np.zeros((8, 3, 4), np.float32)
    for i in range(8):
        center = np.array([(i % 4) * 1.5 - 2.25, (i // 4) * 1.5 - 0.75, 0])
        records[i, 0] = [*(center - 0.45), 1]
        records[i, 1, :3] = center + 0.45
        records[i, 2] = [i % 4, 71 if i % 4 == 2 else -1, 100 + i, 0]
    # Separate glass cells must have distinct boundary identities.
    records[6, 2, 1] = 72
    return BoxBatch(
        records[:, :2, :3],
        materials=records[:, 2, 0].astype(np.uint32),
        boundaries=np.where(
            records[:, 2, 1] < 0, 0xFFFFFFFF, records[:, 2, 1].astype(np.float64)
        ).astype(np.uint32),
        identities=records[:, 2, 2].astype(np.uint32),
    )


def run(output="/tmp/chunk-probe.json", samples=64):
    if samples < 1:
        raise ValueError("samples must be positive")
    batch = fixture()
    records = batch.records.copy()
    emission = MaterialGraph(
        {
            "gain": MaterialNode("uniform", value="gain", type="vec4"),
            "rgb": MaterialNode("components", ("gain",), value="rgb", type="vec3"),
        },
        {"emission": "rgb"},
        resources=(MaterialResource("gain", "uniform"),),
    )
    metal = MaterialGraph(
        {
            "m": MaterialNode("constant", value=1),
            "r": MaterialNode("constant", value=0.3),
        },
        {"metallic": "m", "roughness": "r"},
    )
    rng = np.random.default_rng(1729)
    origins = np.column_stack(
        (rng.uniform(-3.5, 3.5, 2048), rng.uniform(-2, 2, 2048), np.full(2048, 3))
    )
    directions = np.tile([0, 0, -1], (len(origins), 1))
    report = {
        "algorithm": "bounded independent-box scan; not an acceleration benchmark",
        "frames": [],
    }
    with ExitStack() as stack:
        runtime = stack.enter_context(VulkanRuntime())
        storage = stack.enter_context(runtime.buffer(records.nbytes, data=records))
        gain = stack.enter_context(
            runtime.buffer(16, data=np.array([2, 0.5, 0.1, 0], np.float32))
        )
        scenes = []
        for per_box in [True, False]:
            geometry = []
            for first, count in (
                [(i, 1) for i in range(8)] if per_box else [(0, 4), (4, 4)]
            ):
                geometry.append(batch.geometry(first, count))
            scenes.append(
                stack.enter_context(
                    VulkanTransportScene(
                        runtime,
                        custom_geometry=geometry,
                        custom_resources={batch.resource_name: storage},
                        custom_materials=[
                            TransportMaterial("diffuse", albedo=(0.4, 0.6, 0.2)),
                            TransportMaterial(
                                "pbr", albedo=(0.8, 0.5, 0.2), program=metal
                            ),
                            TransportMaterial("dielectric"),
                            TransportMaterial("emission", program=emission),
                        ],
                        material_resources={"gain": gain},
                        media=[OpticalMedium(), OpticalMedium(1.5)],
                        boundaries=[MediumBoundary(71, 0, 1), MediumBoundary(72, 0, 1)],
                    )
                )
            )
        outputs = [
            stack.enter_context(GpuSampleAccumulator(runtime, len(origins)))
            for _ in scenes
        ]
        integrators = [
            stack.enter_context(
                VulkanTransportIntegrator(
                    scene, ray_samples(origins, directions), accumulator
                )
            )
            for scene, accumulator in zip(scenes, outputs)
        ]
        for frame in range(3):
            if frame == 1:
                records[0, 0, 3] = 0  # activation edit without rebuilding chunk bounds
            elif frame == 2:
                records[0, 0, 3] = 1
                gain.upload(np.array([0.1, 0.5, 2, 0], np.float32))
            storage.upload(records)
            hits = [intersect_rays(scene, origins, directions) for scene in scenes]
            for result in hits:
                assert not result["boundary"][:, 3].any()
            for name in (
                "position_distance",
                "geometric_normal",
                "shading_normal",
                "boundary",
            ):
                np.testing.assert_allclose(hits[0][name], hits[1][name], atol=1e-5)
            np.testing.assert_array_equal(
                hits[0]["identity"][:, 2:], hits[1]["identity"][:, 2:]
            )
            timings = []
            for transport, accumulation in zip(integrators, outputs):
                accumulation.reset().wait()
                start = time.perf_counter()
                transport.accumulate(
                    samples_per_element=samples,
                    max_bounces=12,
                    environment=(0.3, 0.4, 0.6),
                    environment_nee=True,
                ).wait()
                timings.append((time.perf_counter() - start) * 1000)
            means = [accumulation.means() for accumulation in outputs]
            np.testing.assert_allclose(means[0], means[1], rtol=1e-5, atol=1e-6)
            report["frames"].append(
                dict(
                    frame=frame,
                    hit_count=int((hits[0]["identity"][:, 0] != 0).sum()),
                    max_radiance_error=float(np.max(np.abs(means[0] - means[1]))),
                    per_box_ms=timings[0],
                    chunk_ms=timings[1],
                )
            )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="/tmp/chunk-probe.json")
    parser.add_argument("--samples", type=int, default=64)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
