"""Public bulk sphere edits, refits and removals; opt-in offscreen GPU client."""

import argparse
import json

import numpy as np

from ordinarylight.geometry import CustomGeometryBatch, IntersectionProgram
from ordinarylight.runtime import VulkanRuntime
from ordinarylight.pipeline.graph import VulkanGraph
from ordinarylight.transport import (
    TransportMaterial,
    VulkanTransportScene,
    VulkanRayQuery,
)


def spheres(count, z=0, identity_offset=0):
    centers = np.column_stack(
        (np.arange(count) * 2, np.zeros(count), np.full(count, z))
    )
    return CustomGeometryBatch(
        np.stack((centers - 0.5, centers + 0.5), axis=1),
        IntersectionProgram.sdf_sphere(),
        np.column_stack((centers, np.full(count, 0.5))),
        identities=np.arange(count, dtype=np.uint32) + identity_offset,
    )


def run(count=128):
    if not 1 <= count <= 65536:
        raise ValueError("Use 1 to 65536 spheres")
    origins = np.column_stack(
        (np.arange(count) * 2, np.zeros(count), np.full(count, 3))
    )
    with (
        VulkanRuntime() as runtime,
        VulkanTransportScene(
            runtime,
            custom_geometry=spheres(count),
            custom_materials=[TransportMaterial()],
        ) as scene,
        VulkanRayQuery(scene, origins, np.tile([0, 0, -1], (count, 1))) as query,
    ):
        query.operation().execute(runtime).wait()
        initial = query.read()
        np.testing.assert_allclose(initial["position_distance"][:, 3], 2.5, atol=1e-4)
        with scene.prepare_custom_geometry_update(
            np.arange(count), spheres(count, 1, count)
        ) as edit:
            graph = (
                VulkanGraph()
                .add("edit", edit.operation())
                .add("query", query.operation(), after=("edit",))
            )
            graph.compile().execute(runtime).wait()
            moved = query.read()
            np.testing.assert_allclose(moved["position_distance"][:, 3], 1.5, atol=1e-4)
            np.testing.assert_array_equal(
                moved["identity"][:, 2], np.arange(count) + count
            )
        with scene.prepare_custom_geometry_update(
            np.arange(0, count, 2), None
        ) as removal:
            graph = (
                VulkanGraph()
                .add("remove", removal.operation())
                .add("query", query.operation(), after=("remove",))
            )
            graph.compile().execute(runtime).wait()
            removed = query.read()
            np.testing.assert_array_equal(removed["identity"][::2, 0], 0)
            np.testing.assert_array_equal(removed["identity"][1::2, 0], 2)
        # Structural growth must preserve both moved and disabled slots.
        query.close()
        scene.reserve_custom_geometry(count + 2)
        with VulkanRayQuery(
            scene, origins, np.tile([0, 0, -1], (count, 1))
        ) as grown_query:
            grown_query.operation().execute(runtime).wait()
            grown = grown_query.read()
            np.testing.assert_array_equal(grown, removed)
        for result in (initial, moved, removed, grown):
            if result["boundary"][:, 3].any():
                raise RuntimeError("Invalid intersection result")
    return dict(spheres=count, moved=count, removed=(count + 1) // 2, invalid_paths=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=128)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
