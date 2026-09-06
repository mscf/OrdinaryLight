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
@pytest.mark.parametrize("bulk", [False, True])
def test_grouped_box_hits_against_analytic_intervals_and_resource_edits(bulk):
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
            custom_geometry=batch.geometries() if bulk else [batch.geometry()],
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


def test_partition_preserves_slots_and_covers_inactive_boxes():
    rng = np.random.default_rng(104)
    centers = rng.uniform(-20, 20, (37, 3))
    batch = BoxBatch(np.stack((centers - 0.2, centers + 0.2), axis=1), active=False)
    partition = batch.partition(4)
    np.testing.assert_array_equal(np.sort(partition.indices), np.arange(37))
    np.testing.assert_array_equal(partition.indices, batch.partition(4).indices)
    assert not partition.indices.flags.writeable
    for geometry in partition.geometries:
        first, count = map(int, geometry.parameters[:2])
        assert 1 <= count <= 4
        records = batch.records[partition.indices[first : first + count]]
        np.testing.assert_allclose(geometry.bounds[0], records[:, 0, :3].min(axis=0))
        np.testing.assert_allclose(geometry.bounds[1], records[:, 1, :3].max(axis=0))
    shifted = batch.records[:, :2, :3] + [30, 0, 0]
    for original, refitted in zip(partition.geometries, partition.refit(shifted)):
        np.testing.assert_allclose(
            np.array(original.bounds) + [30, 0, 0], refitted.bounds, atol=3e-6
        )
        assert refitted.parameters == original.parameters
    with pytest.raises(ValueError, match="slots"):
        partition.refit(shifted[:1])
    with pytest.raises(ValueError, match="positive"):
        batch.partition(0)


@pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_TRANSPORT") != "1",
    reason="opt-in spatial partition GPU validation",
)
def test_partition_matches_scan_for_oblique_inside_and_edited_rays():
    from contextlib import ExitStack
    import ordinarylight as ol
    from ordinarylight.transport import (
        VulkanTransportScene,
        TransportMaterial,
        intersect_rays,
    )

    rng = np.random.default_rng(103)
    centers = rng.uniform(-10, 10, (257, 3))
    half = rng.uniform(0.02, 0.1, (257, 3))
    batch = BoxBatch(
        np.stack((centers - half, centers + half), axis=1),
        identities=np.arange(257, dtype=np.uint32) + 0x80000000,
    )
    partition = batch.partition(8)
    direction = rng.normal(size=(257, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    origins = np.concatenate((centers, centers - direction * 25))
    directions = np.concatenate((direction, direction))
    with ExitStack() as stack:
        runtime = stack.enter_context(ol.VulkanRuntime())
        storage = stack.enter_context(
            runtime.buffer(batch.records.nbytes, data=batch.records)
        )
        indices = stack.enter_context(
            runtime.buffer(partition.indices.nbytes, data=partition.indices)
        )
        common = dict(custom_materials=[TransportMaterial()])
        reference = stack.enter_context(
            VulkanTransportScene(
                runtime,
                custom_geometry=[batch.geometry()],
                custom_resources={batch.resource_name: storage},
                **common,
            )
        )
        optimized = stack.enter_context(
            VulkanTransportScene(
                runtime,
                custom_geometry=partition.geometries,
                custom_resources={
                    batch.resource_name: storage,
                    partition.index_resource_name: indices,
                },
                **common,
            )
        )
        for edited in [False, True]:
            if edited:
                records = batch.records.copy()
                records[::3, 0, 3] = 0
                records[:, 2].view(np.uint32)[:, 2] += 1000
                storage.upload(records)
            expected = intersect_rays(reference, origins, directions, max_steps=512)
            actual = intersect_rays(optimized, origins, directions, max_steps=512)
            for hits in [expected, actual]:
                assert not hits["boundary"][:, 3].any()
            np.testing.assert_array_equal(
                actual["identity"][:, 2:], expected["identity"][:, 2:]
            )
            np.testing.assert_allclose(
                actual["position_distance"], expected["position_distance"], atol=3e-5
            )
            np.testing.assert_allclose(
                actual["geometric_normal"], expected["geometric_normal"], atol=1e-6
            )
        records[:, :2, :3] += [40, 0, 0]
        storage.upload(records)
        optimized.update_custom_geometry(
            dict(enumerate(partition.refit(records[:, :2, :3]))), mode="refit"
        ).wait()
        shifted_origins = origins + [40, 0, 0]
        shifted = intersect_rays(optimized, shifted_origins, directions)
        assert not shifted["boundary"][:, 3].any()
        np.testing.assert_array_equal(
            shifted["identity"][:, 2:], actual["identity"][:, 2:]
        )
        np.testing.assert_allclose(
            shifted["position_distance"][:, 3],
            actual["position_distance"][:, 3],
            atol=3e-5,
        )
        corrupt = np.full_like(partition.indices, 0xFFFFFFFF)
        indices.upload(corrupt)
        errors = intersect_rays(optimized, shifted_origins[:1], directions[:1])
        assert errors["boundary"][0, 3] == 1


@pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_TRANSPORT") != "1",
    reason="opt-in glass-only box palette validation",
)
def test_glass_only_palette_has_valid_enclosing_boundary():
    from contextlib import ExitStack
    import ordinarylight as ol
    from ordinarylight.transport import (
        VulkanTransportScene,
        TransportMaterial,
        OpticalMedium,
        MediumBoundary,
        intersect_rays,
    )

    batch = BoxBatch([[[-1, -1, -1], [1, 1, 1]]], boundaries=71, identities=42)
    partition = batch.partition()
    assert batch.geometry().boundary == 71
    assert partition.refit(batch.records[:, :2, :3])[0].boundary == 71
    with ExitStack() as stack:
        runtime = stack.enter_context(ol.VulkanRuntime())
        storage = stack.enter_context(
            runtime.buffer(batch.records.nbytes, data=batch.records)
        )
        indices = stack.enter_context(
            runtime.buffer(partition.indices.nbytes, data=partition.indices)
        )
        for indexed in [False, True]:
            resources = {batch.resource_name: storage}
            if indexed:
                resources[partition.index_resource_name] = indices
            with VulkanTransportScene(
                runtime,
                custom_geometry=partition.geometries if indexed else [batch.geometry()],
                custom_resources=resources,
                custom_materials=[TransportMaterial("dielectric")],
                media=[OpticalMedium(), OpticalMedium(1.5)],
                boundaries=[MediumBoundary(71, 0, 1)],
            ) as scene:
                hits = intersect_rays(scene, [[0, 0, 3]], [[0, 0, -1]])
                np.testing.assert_array_equal(hits["boundary"][0], [0, 0, 1, 0])
                assert hits["identity"][0, 2] == 42
