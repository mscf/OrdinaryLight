"""Fused HDR visibility composes with persistent tone mapping without readback."""

import os
from contextlib import ExitStack
import numpy as np
import pytest
import vulkan as vk
import ordinarylight as ol
from ordinarylight.geometry import BoxBatch
from ordinarylight.runtime import VulkanOutput
from ordinarylight.pipeline.graph import VulkanGraph
from ordinarylight.transport import (
    VulkanTransportScene,
    VulkanRayQuery,
    TransportMaterial,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_TRANSPORT") != "1",
    reason="opt-in GPU output validation",
)


def test_hdr_visibility_graph_matches_compact_colors(monkeypatch):
    batch = BoxBatch([[[-1, -1, -1], [1, 1, 1]]])
    with ExitStack() as stack:
        runtime = stack.enter_context(ol.VulkanRuntime())
        records = stack.enter_context(
            runtime.buffer(batch.records.nbytes, data=batch.records, memory="device")
        )
        palette = stack.enter_context(
            runtime.buffer(
                16, data=np.array([2, 0.5, 0.1, 1], np.float32), memory="device"
            )
        )
        hdr = stack.enter_context(runtime.image(3, 2))
        scene = stack.enter_context(
            VulkanTransportScene(
                runtime,
                custom_geometry=[batch.geometry()],
                custom_resources={batch.resource_name: records},
                custom_materials=[TransportMaterial()],
            )
        )
        origins = [[0, 0, 3], [3, 0, 3], [0, 0, 3], [3, 0, 3], [0, 0, 3], [3, 0, 3]]
        query = stack.enter_context(
            VulkanRayQuery(
                scene,
                origins,
                [[0, 0, -1]] * 6,
                colors=palette,
                memory="device",
                hdr=hdr,
            )
        )
        output = stack.enter_context(VulkanOutput(runtime))
        tone = stack.enter_context(output.prepare(hdr))
        graph = (
            VulkanGraph()
            .add("tone", tone.operation())
            .add("visibility", query.operation())
            .compile()
        )
        assert graph.order == ("visibility", "tone") or graph.order == [
            "visibility",
            "tone",
        ]
        for color in ([2, 0.5, 0.1, 1], [0.05, 1, 3, 1]):
            palette.upload(np.array(color, np.float32))
            with monkeypatch.context() as patch:
                patch.setattr(
                    query.hits, "read", lambda: pytest.fail("implicit hit readback")
                )
                patch.setattr(
                    output, "read", lambda *a: pytest.fail("implicit image readback")
                )
                graph.execute(runtime).wait()
            pixels = np.frombuffer(output.read(tone), np.uint8).reshape(6, 4)
            linear = np.maximum(query.read()["color"][:, :3], 0)
            aces = np.clip(
                linear
                * (2.51 * linear + 0.03)
                / (linear * (2.43 * linear + 0.59) + 0.14),
                0,
                1,
            )
            srgb = np.where(
                aces <= 0.0031308, 12.92 * aces, 1.055 * aces ** (1 / 2.4) - 0.055
            )
            np.testing.assert_allclose(pixels[:, :3], srgb * 255, atol=1)
            assert (pixels[:, 3] == 255).all()
        with pytest.raises(RuntimeError, match="borrowers"):
            hdr.close()
        with pytest.raises(ValueError, match="palette"):
            VulkanRayQuery(scene, origins, [[0, 0, -1]] * 6, hdr=hdr)
        with pytest.raises(ValueError, match="ray count"):
            VulkanRayQuery(scene, origins[:1], [[0, 0, -1]], colors=palette, hdr=hdr)
