"""CPU-only bulk geometry validation and ABI compatibility; no Vulkan device."""

from types import SimpleNamespace
import struct

import numpy as np
import pytest

from ordinarylight.geometry import (
    BoxBatch,
    CustomGeometry,
    CustomGeometryBatch,
    IntersectionProgram,
)
from ordinarylight.transport import TransportMaterial, VulkanTransportScene
from ordinarylight.transport._custom_batch import CustomSlots, prepare_custom_geometry
from ordinarylight.transport._dynamic_scene import _pack

PROGRAM = IntersectionProgram("testIntersection", "// test callback")
BOUNDS = np.array([[[0, 0, 0], [1, 2, 3]], [[4, 5, 6], [7, 8, 9]]], np.float32)


def scene():
    result = SimpleNamespace(
        programs={"other": IntersectionProgram("other", "// another callback")},
        _custom_materials=(TransportMaterial(), TransportMaterial("dielectric")),
        triangle_count=7,
        boundary_indices={900: 0, 42: 1},
    )
    result._boundary_index = (
        lambda identity, material: VulkanTransportScene._boundary_index(
            result, identity, material
        )
    )
    return result


def test_bulk_matches_existing_wire_contract_and_objects():
    batch = CustomGeometryBatch(
        BOUNDS,
        PROGRAM,
        [[1, 2, 3, 4], [5, 6, 7, 8]],
        materials=[0, 1],
        boundaries=[0xFFFFFFFF, 42],
        identities=[0xFFFFFFFF, 0x80000001],
    )
    target = scene()
    packed, bounds, slots = prepare_custom_geometry(target, batch, 4)
    assert packed.dtype.itemsize == 64
    for i in range(2):
        assert packed[i].tobytes() == _pack(target, batch[i])
    assert packed[1].tobytes() == struct.pack(
        "<12f4I", 4, 5, 6, 0, 7, 8, 9, 0, 5, 6, 7, 8, 1, 8, 1, 0x80000001
    )
    np.testing.assert_array_equal(bounds[:2], BOUNDS.reshape(2, 6))
    assert len(slots) == 4 and slots[-1] is None
    assert slots[1].boundary == 42
    assert np.all(packed["metadata"][2:, 0] == 0xFFFFFFFF)
    legacy, legacy_bounds, _ = prepare_custom_geometry(scene(), list(batch), 4)
    assert packed.tobytes() == legacy.tobytes()
    np.testing.assert_array_equal(bounds, legacy_bounds)


def test_box_adapter_matches_per_box_geometry_including_inactive():
    boxes = BoxBatch(
        BOUNDS,
        active=[False, True],
        materials=[0, 1],
        boundaries=[0xFFFFFFFF, 900],
        identities=[99, 0xFFFFFFFF],
    )
    bulk = boxes.geometries()
    for i in range(2):
        assert bulk[i] == boxes.geometry(i, 1)
    packed, _, _ = prepare_custom_geometry(scene(), bulk, 2)
    legacy, _, _ = prepare_custom_geometry(
        scene(), [boxes.geometry(i, 1) for i in range(2)], 2
    )
    assert packed.tobytes() == legacy.tobytes()


def test_snapshot_owns_immutable_arrays():
    bounds = BOUNDS.copy()
    parameters = np.ones((2, 4))
    batch = CustomGeometryBatch(bounds, PROGRAM, parameters)
    bounds[:] = 0
    parameters[:] = 0
    np.testing.assert_array_equal(batch.bounds, BOUNDS)
    assert batch.parameters.all()
    for value in (
        batch.bounds,
        batch.parameters,
        batch.materials,
        batch.boundaries,
        batch.identities,
    ):
        with pytest.raises(ValueError):
            value.setflags(write=True)
    assert batch[-1] == batch[1]
    assert batch[:] == (batch[0], batch[1])
    with pytest.raises(IndexError):
        batch[2]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(bounds=[]),
        dict(bounds=[[[0, 0, 0], [0, 1, 1]]]),
        dict(bounds=[[[1e20, 0, 0], [1e20 + 1, 1, 1]]]),
        dict(bounds=[[[0, 0, 0], [1e100, 1, 1]]]),
        dict(parameters=[0, 1, 2]),
        dict(parameters=[0, 0, 0, float("nan")]),
        dict(parameters=[0, 0, 0, 1e100]),
        dict(materials=-1),
        dict(materials=1.5),
        dict(identities=2**32),
        dict(boundaries=True),
        dict(identities=[0, 1, 2]),
    ],
)
def test_invalid_arrays_rejected(kwargs):
    with pytest.raises(ValueError):
        CustomGeometryBatch(**(dict(bounds=BOUNDS, program=PROGRAM) | kwargs))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(materials=2), "custom material"),
        (dict(materials=1), "explicit medium boundary"),
        (dict(boundaries=42), "dielectric material"),
        (dict(materials=1, boundaries=43), "unknown boundary"),
    ],
)
def test_scene_rejects_invalid_material_boundary_relationships(kwargs, match):
    with pytest.raises(ValueError, match=match):
        prepare_custom_geometry(
            scene(), CustomGeometryBatch(BOUNDS, PROGRAM, **kwargs), 2
        )


def test_program_conflicts_are_rejected():
    target = scene()
    target.programs[PROGRAM.name] = IntersectionProgram(PROGRAM.name, "// conflicting")
    with pytest.raises(ValueError, match="Conflicting"):
        prepare_custom_geometry(target, CustomGeometryBatch(BOUNDS, PROGRAM), 2)


def test_packing_sparse_edits_and_growth_does_not_expand_batch(monkeypatch):
    batch = CustomGeometryBatch(BOUNDS, PROGRAM)
    replacement = CustomGeometry(BOUNDS[1], PROGRAM, (4, 3, 2, 1), identity=99)

    def no_expansion(*args):
        raise AssertionError("Bulk packing must not instantiate individual geometry")

    monkeypatch.setattr(CustomGeometryBatch, "__getitem__", no_expansion)
    target = scene()
    _, _, slots = prepare_custom_geometry(target, batch, 3)
    edited = slots.updated({0: None, 2: replacement})
    packed, bounds, grown = prepare_custom_geometry(target, edited, 5)
    assert grown.base is batch and len(grown) == 5
    assert grown[0] is None and grown[2] is replacement and grown[4] is None
    assert slots.updates == {}  # submitted snapshots do not mutate older views
    assert packed[2].tobytes() == _pack(target, replacement)
    assert packed[0].tobytes() == _pack(target, None)
    np.testing.assert_array_equal(bounds[2], BOUNDS[1].reshape(6))
    with pytest.raises(IndexError):
        grown[5]
    with pytest.raises(ValueError):
        CustomSlots(batch, 1)


def test_scene_constructor_uploads_bulk_without_expansion(monkeypatch):
    """Exercise real constructor wiring with device allocation/build boundaries stubbed."""
    from threading import RLock
    from unittest.mock import MagicMock
    from ordinarylight.transport import _dynamic_scene
    from ordinarylight.targets.vulkan import scene as scene_module

    allocations = []

    def buffer(size, *, data):
        allocation = SimpleNamespace(
            size=size,
            buffer=len(allocations) + 1,
            data=(
                np.frombuffer(data, np.uint8).copy()
                if isinstance(data, bytes)
                else np.asarray(data).copy()
            ),
            close=lambda: None,
        )
        allocations.append(allocation)
        return allocation

    runtime = SimpleNamespace(
        lock=RLock(),
        require_open=lambda: None,
        retain=lambda owner: None,
        buffer=buffer,
        device=object(),
        release=lambda owner: None,
    )
    monkeypatch.setattr(
        scene_module, "VulkanSceneUploader", lambda runtime: MagicMock()
    )
    builds = []

    def build(owner):
        builds.append(owner._custom_bounds.copy())
        owner._custom_blas = object()
        owner._bounds_buffer = SimpleNamespace(
            buffer=900, size=owner._custom_bounds.nbytes
        )
        owner._instance_buffer = object()
        owner._instance_count = 1
        owner._custom_shape = object()
        owner.tlas = SimpleNamespace(handle=901)

    monkeypatch.setattr(_dynamic_scene, "build_acceleration", build)
    import vulkan as vk

    monkeypatch.setattr(vk, "vkDeviceWaitIdle", lambda device: None)
    batch = CustomGeometryBatch(BOUNDS, PROGRAM, identities=[17, 31])
    legacy = VulkanTransportScene(
        runtime,
        custom_geometry=list(batch),
        custom_materials=[TransportMaterial()],
        custom_capacity=4,
    )

    def no_expansion(*args):
        raise AssertionError("Scene construction expanded the batch")

    monkeypatch.setattr(CustomGeometryBatch, "__getitem__", no_expansion)
    bulk = VulkanTransportScene(
        runtime,
        custom_geometry=batch,
        custom_materials=[TransportMaterial()],
        custom_capacity=4,
    )
    assert bulk.custom_geometry.base is batch
    for name in legacy._buffers:
        assert (
            bulk._buffers[name].data.tobytes() == legacy._buffers[name].data.tobytes()
        )
    np.testing.assert_array_equal(builds[0], builds[1])
    # A submitted deletion must preserve the array-backed shadow too.
    operation = bulk.update_custom_geometry_operation({0: None})
    operation.validate()
    operation.submitted(SimpleNamespace())
    assert bulk.custom_geometry.base is batch
    assert bulk.custom_geometry[0] is None
    assert bulk.geometry_revision == 1
    assert bulk.reserve_custom_geometry(6)
    assert bulk.custom_geometry.base is batch
    assert len(bulk.custom_geometry) == 6
    from ordinarylight.transport._custom_batch import CUSTOM_DTYPE

    grown = np.frombuffer(bulk._buffers["custom"].data, CUSTOM_DTYPE)
    assert grown["metadata"][0, 0] == 0xFFFFFFFF
    assert grown["metadata"][1, 3] == 31
    assert np.all(grown["metadata"][2:, 0] == 0xFFFFFFFF)
    assert bulk.binding_revision == 1
    bulk.close()
    legacy.close()
