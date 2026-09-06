"""Packing and independent geometric references for grouped custom boxes."""

import os
import numpy as np
import pytest
from ordinarylight.geometry import BoxBatch


def test_box_batch_retains_full_width_ids_and_inactive_bounds():
    batch = BoxBatch(
        [[[-1, -1, -1], [1, 1, 1]], [[3, 0, 0], [4, 1, 1]]],
        identities=[0xFFFFFFFF, 0x80000001],
        active=[True, False],
    )
    np.testing.assert_array_equal(
        batch.records[:, 2].view(np.uint32)[:, 2], [0xFFFFFFFF, 0x80000001]
    )
    assert not batch.records.flags.writeable
    assert batch.geometry().bounds == ((-1.0, -1.0, -1.0), (4.0, 1.0, 1.0))
    assert batch.geometry(1).parameters[:2] == (1, 1)
    for first, count in [(-1, 1), (0, 0), (1, 2), (2, 1)]:
        with pytest.raises(ValueError, match="interval"):
            batch.geometry(first, count)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(materials=-1),
        dict(identities=2**32),
        dict(boundaries=1.5),
        dict(materials=[1, 2]),
        dict(active=1),
        dict(resource_name="bad name"),
    ],
)
def test_box_batch_rejects_invalid_records(kwargs):
    with pytest.raises(ValueError):
        BoxBatch([[[0, 0, 0], [1, 1, 1]]], **kwargs)


@pytest.mark.parametrize(
    "bounds", [[], [[[0, 0, 0], [0, 1, 1]]], [[[0, 0, 0], [float("inf"), 1, 1]]]]
)
def test_box_batch_rejects_invalid_bounds(bounds):
    with pytest.raises(ValueError):
        BoxBatch(bounds)


@pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_TRANSPORT") != "1",
    reason="opt-in grouped box GPU validation",
)
def test_grouped_box_hits_against_analytic_intervals_and_resource_edits():
    import ordinarylight as ol
    from ordinarylight.transport import (
        VulkanTransportScene,
        TransportMaterial,
        intersect_rays,
    )

    batch = BoxBatch(
        [[[-1, -1, -1], [1, 1, 1]], [[2, -1, -1], [3, 1, 1]]],
        identities=[0x80000001, 0xFFFFFFFE],
        materials=[0, 1],
    )
    origins = [[0, 0, 3], [0, 0, 0], [2.5, 0, 3], [-3, 0, 0], [1.5, 0, 3], [0, 2, 0]]
    directions = [[0, 0, -1], [1, 0, 0], [0, 0, -1], [1, 0, 0], [0, 0, -1], [1, 0, 0]]
    with (
        ol.VulkanRuntime() as runtime,
        runtime.buffer(batch.records.nbytes, data=batch.records) as buffer,
    ):
        with VulkanTransportScene(
            runtime,
            custom_geometry=[batch.geometry()],
            custom_materials=[TransportMaterial(), TransportMaterial()],
            custom_resources={batch.resource_name: buffer},
        ) as scene:
            hits = intersect_rays(scene, origins, directions)
            assert not hits["boundary"][:, 3].any()
            np.testing.assert_array_equal(hits["identity"][:, 0], [2, 2, 2, 2, 0, 0])
            np.testing.assert_allclose(hits["position_distance"][:4, 3], [2, 1, 2, 2])
            np.testing.assert_allclose(
                hits["geometric_normal"][:4, :3],
                [[0, 0, 1], [1, 0, 0], [0, 0, 1], [-1, 0, 0]],
            )
            np.testing.assert_array_equal(
                hits["identity"][:4, 2],
                [0x80000001, 0x80000001, 0xFFFFFFFE, 0x80000001],
            )
            np.testing.assert_array_equal(hits["identity"][:4, 3], [0, 0, 1, 0])
            records = batch.records.copy()
            records[0, 0, 3] = 0
            buffer.upload(records)
            updated = intersect_rays(scene, origins, directions)
            np.testing.assert_array_equal(updated["identity"][:, 0], [0, 2, 2, 2, 0, 0])
            np.testing.assert_allclose(
                updated["position_distance"][[1, 2, 3], 3], [2, 2, 5]
            )


@pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_TRANSPORT") != "1",
    reason="opt-in grouped box GPU validation",
)
def test_box_batch_reports_invalid_device_records():
    import ordinarylight as ol
    from ordinarylight.transport import (
        VulkanTransportScene,
        TransportMaterial,
        intersect_rays,
    )

    batch = BoxBatch([[[-1, -1, -1], [1, 1, 1]]])
    with (
        ol.VulkanRuntime() as runtime,
        runtime.buffer(batch.records.nbytes, data=batch.records) as buffer,
    ):
        with VulkanTransportScene(
            runtime,
            custom_geometry=[batch.geometry()],
            custom_materials=[TransportMaterial()],
            custom_resources={batch.resource_name: buffer},
        ) as scene:
            for value in [float("nan"), 2.0]:
                records = batch.records.copy()
                records[0, 0, 3] = value
                buffer.upload(records)
                hits = intersect_rays(scene, [[0, 0, 3]], [[0, 0, -1]])
                assert hits["boundary"][0, 3] == 1
