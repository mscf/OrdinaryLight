"""Standalone closest-hit queue records and zero-ray dispatch behavior."""

from contextlib import ExitStack
import os
import struct
import numpy as np
import pytest
import vulkan as vk
import ordinarylight as ol
from ordinarylight.runtime import VulkanIntersection
from ordinarylight.wavefront import RAY_DTYPE, HIT_DTYPE


@pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_GRAPH") != "1", reason="opt-in GPU"
)
@pytest.mark.parametrize("indirect_dispatch", [False, True])
def test_intersection_records_and_empty_queue(indirect_dispatch):
    scene = ol.Scene()
    scene.add_mesh([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], [[0, 1, 2]])
    with ol.VulkanRuntime() as runtime, ExitStack() as stack:
        resident = stack.enter_context(runtime.upload_scene(scene))
        records = np.zeros(2, dtype=RAY_DTYPE)
        records["origin_tmin"] = [[0, 0, 2, 0.001], [4, 0, 2, 0.001]]
        records["direction_tmax"] = [[0, 0, -1, 100]] * 2
        records["path_index"] = [7, 9]
        rays = stack.enter_context(
            runtime.buffer(112, data=struct.pack("4I", 2, 2, 0, 0) + records.tobytes())
        )
        hits = stack.enter_context(runtime.buffer(112))
        indirect = stack.enter_context(
            runtime.buffer(
                12,
                data=struct.pack("3I", 1, 1, 1),
                usage=vk.VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
                | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            )
        )
        stage = stack.enter_context(
            VulkanIntersection(
                runtime,
                tlas=resident.resource("tlas"),
                vertices=resident.resource("vertex"),
                rays=rays,
                hits=hits,
                capacity=2,
            )
        )
        for count in (2, 0):
            rays.upload(struct.pack("4I", count, 2, 0, 0) + records.tobytes())
            indirect.upload(struct.pack("3I", 1 if count else 0, 1, 1))
            stage.operation(indirect=indirect if indirect_dispatch else None).execute(
                runtime
            ).wait()
            data = hits.read()
            np.testing.assert_array_equal(
                np.frombuffer(data, np.uint32, count=3), [count, 2, 0]
            )
            if count:
                values = np.frombuffer(data, HIT_DTYPE, count=2, offset=16)
                np.testing.assert_allclose(
                    values["position_t"][0], [0, 0, 0, 2], atol=1e-6
                )
                np.testing.assert_allclose(
                    values["geometric_normal"][0], [0, 0, 1], atol=1e-6
                )
                assert values["position_t"][1, 3] == -1
                np.testing.assert_array_equal(
                    values["primitive_index"], [0, 0xFFFFFFFF]
                )
                np.testing.assert_array_equal(values["path_index"], [7, 9])
                np.testing.assert_array_equal(values["ray_index"], [0, 1])
                np.testing.assert_allclose(
                    values["barycentrics"][0], [0.25, 0.5], atol=1e-6
                )
        with pytest.raises(RuntimeError, match="consumers"):
            resident.close()
        with pytest.raises(RuntimeError, match="borrowers"):
            hits.close()


@pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_GRAPH") != "1", reason="opt-in GPU"
)
def test_generation_and_intersection_compose_in_dependency_order():
    from ordinarylight.runtime import VulkanRayGeneration, VulkanQueueDispatch
    from ordinarylight.pipeline.graph import VulkanGraph

    scene = ol.Scene()
    scene.add_mesh([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], [[0, 1, 2]])
    with ol.VulkanRuntime() as runtime, ExitStack() as stack:
        resident = stack.enter_context(runtime.upload_scene(scene))
        camera = stack.enter_context(
            runtime.buffer(
                64,
                data=struct.pack(
                    "16f",
                    0,
                    0,
                    2,
                    0,
                    0,
                    0,
                    -1,
                    0,
                    1,
                    0,
                    0,
                    struct.unpack("f", struct.pack("I", 0x38003800))[0],
                    0,
                    1,
                    0,
                    0,
                ),
            )
        )
        rays = stack.enter_context(runtime.buffer(64))
        hits = stack.enter_context(runtime.buffer(64))
        paths = stack.enter_context(runtime.buffer(48))
        media = stack.enter_context(runtime.buffer(64))
        generator = stack.enter_context(
            VulkanRayGeneration(
                runtime, camera=camera, rays=rays, paths=paths, media=media, capacity=1
            )
        )
        intersection = stack.enter_context(
            VulkanIntersection(
                runtime,
                tlas=resident.resource("tlas"),
                vertices=resident.resource("vertex"),
                rays=rays,
                hits=hits,
                capacity=1,
            )
        )
        arguments = stack.enter_context(
            runtime.buffer(
                12,
                usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                | vk.VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
                | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            )
        )
        dispatch = stack.enter_context(
            VulkanQueueDispatch(runtime, queue=rays, arguments=arguments)
        )
        graph = (
            VulkanGraph()
            .add("intersect", intersection.operation(indirect=arguments))
            .add("dispatch", dispatch.operation())
            .add("generate", generator.operation(extent=(1, 1)))
        )
        compiled = graph.compile()
        assert compiled.order == ("generate", "dispatch", "intersect")
        compiled.execute(runtime).wait()
        value = np.frombuffer(hits.read(), HIT_DTYPE, count=1, offset=16)
        np.testing.assert_allclose(value["position_t"][0], [0, 0, 0, 2], atol=1e-6)
