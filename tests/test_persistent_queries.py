"""Persistent GPU visibility preserves hits and resource lifetime contracts."""

import os
import numpy as np
import pytest
import ordinarylight as ol
from ordinarylight.geometry import BoxBatch
from ordinarylight.pipeline.graph import VulkanGraph
from ordinarylight.transport import (
    VulkanTransportScene,
    VulkanRayQuery,
    TransportMaterial,
    intersect_rays,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_TRANSPORT") != "1",
    reason="opt-in persistent query GPU validation",
)


def test_persistent_query_replay_input_updates_and_lifetimes(monkeypatch):
    batch = BoxBatch([[[-1, -1, -1], [1, 1, 1]]], identities=0x80000001)
    with (
        ol.VulkanRuntime() as runtime,
        runtime.buffer(batch.records.nbytes, data=batch.records) as records,
    ):
        with VulkanTransportScene(
            runtime,
            custom_geometry=[batch.geometry()],
            custom_resources={batch.resource_name: records},
            custom_materials=[TransportMaterial()],
        ) as scene:
            origins, directions = [[0, 0, 3], [3, 0, 3]], [[0, 0, -1]] * 2
            expected = intersect_rays(scene, origins, directions)
            with VulkanRayQuery(scene, origins, directions) as query:
                with pytest.raises(RuntimeError, match="before reading"):
                    query.read()
                with pytest.raises(RuntimeError):
                    scene.close()
                inputs, hits, kernel = query.inputs, query.hits, query.kernel
                graph = VulkanGraph().add("visibility", query.operation()).compile()
                with monkeypatch.context() as patch:
                    patch.setattr(
                        hits, "read", lambda: pytest.fail("implicit query readback")
                    )
                    for _ in range(3):
                        graph.execute(runtime).wait()
                np.testing.assert_array_equal(query.read(), expected)
                packed = np.zeros((2, 2, 4), np.float32)
                packed[:, 0, :3] = [3, 0, 3]
                packed[:, 1, :3] = [0, 0, -1]
                inputs.upload(packed)
                graph.execute(runtime).wait()
                assert not query.read()["identity"][:, 0].any()
                assert (
                    query.inputs is inputs
                    and query.hits is hits
                    and query.kernel is kernel
                )
                for options in [
                    dict(t_min=-1),
                    dict(t_max=0),
                    dict(tolerance=0),
                    dict(max_steps=0),
                ]:
                    with pytest.raises(ValueError):
                        query.operation(**options)
            with pytest.raises(RuntimeError, match="closed"):
                graph.execute(runtime)
            assert inputs.closed and hits.closed
