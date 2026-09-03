import tempfile
from pathlib import Path

import numpy as np

from ordinarylight.showcases.scientific import (
    ScientificWorkbenchController,
    build_scientific_scalar_field_scene,
    build_scientific_scalar_field_showcase,
)


def test_scientific_showcase_shares_field_transfer_and_clipping():
    showcase = build_scientific_scalar_field_showcase(10)
    assert showcase.scene.metadata["showcase"] == "scientific-scalar-field"
    assert len(showcase.scene.volumes) == 1
    assert len(showcase.scene.meshes) == 4
    assert showcase.transfer_function.rgba[0, 3] == 0.0
    assert 0.0 < showcase.transfer_function.rgba[-1, 3] < 0.2
    assert all(item.field is showcase.field for item in showcase.slices.values())
    assert all(
        item.transfer_function is showcase.transfer_function
        and item.clipping is showcase.clipping
        for item in (*showcase.slices.values(), showcase.isosurface)
    )
    assert build_scientific_scalar_field_scene(8).metadata["showcase"] \
        == "scientific-scalar-field"


def test_scientific_showcase_partial_update_preserves_scene_handles():
    showcase = build_scientific_scalar_field_showcase(10)
    volume_id = showcase.volume.id
    mesh_ids = tuple(mesh.id for mesh in showcase.scene.meshes)
    prior_volume_revision = showcase.volume.data_revision
    revision = showcase.update_region(
        (4, 4, 4), np.full((2, 2, 2), 0.9, np.float32),
    )
    assert revision == 1
    assert showcase.volume.id == volume_id
    assert showcase.volume.data_revision == prior_volume_revision + 1
    assert tuple(mesh.id for mesh in showcase.scene.meshes) == mesh_ids
    assert showcase.field.updates_since(0) == ((1, (4, 4, 4), (2, 2, 2)),)


def test_scientific_showcase_can_refresh_isosurface_in_place():
    showcase = build_scientific_scalar_field_showcase(8)
    mesh_id = showcase.isosurface_mesh.id
    showcase.update_region(
        (3, 3, 3), np.full((2, 2, 2), 1.0, np.float32),
        refresh_isosurface=True,
    )
    assert showcase.isosurface_mesh.id == mesh_id
    assert len(showcase.isosurface.indices)


def _series_data():
    coordinates = np.linspace(-1.0, 1.0, 9, dtype=np.float32)
    z, y, x = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    first = np.exp(-4.0 * (x * x + y * y + z * z))
    second = np.exp(-4.0 * ((x - 0.15) ** 2 + y * y + z * z))
    return np.stack((first, second)).astype(np.float32)


def test_scientific_controller_loads_memory_mapped_numpy_series():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "simulation.npy"
        np.save(path, _series_data(), allow_pickle=False)
        controller = ScientificWorkbenchController.from_numpy(path)
        owner = controller.series.data
        memory_mapped = isinstance(owner, np.memmap)
        while getattr(owner, "base", None) is not None:
            owner = owner.base
            memory_mapped |= isinstance(owner, np.memmap)
        assert memory_mapped
        assert controller.scene.scientific_controller is controller
        assert controller.series.data.shape == (2, 1, 9, 9, 9)
        assert controller.field.metadata["time_index"] == 0


def test_scientific_controller_updates_views_without_replacing_handles():
    controller = ScientificWorkbenchController.from_array(
        _series_data(), axis_order="tzyx", times=(0.0, 0.5),
        channels=("density",), channel_units=("kg/m3",),
    )
    handles = (
        controller.showcase.volume.id,
        tuple(mesh.id for mesh in controller.scene.meshes),
    )
    controller.select_frame(1)
    assert controller.field.metadata["time"] == 0.5
    assert handles == (
        controller.showcase.volume.id,
        tuple(mesh.id for mesh in controller.scene.meshes),
    )
    transfer = controller.configure_transfer(
        colormap="magma", mode="percentile", opacity=(0.0, 0.3),
    )
    assert controller.showcase.transfer_function is transfer
    assert all(
        item.transfer_function is transfer
        for item in (*controller.showcase.slices.values(),
                     controller.showcase.isosurface)
    )


def test_scientific_controller_representation_roi_and_playback():
    controller = ScientificWorkbenchController.from_array(
        _series_data(), axis_order="tzyx",
    )
    controller.showcase.set_representation("slices")
    assert controller.showcase.volume.visible
    assert controller.showcase.volume.render_mode == "slice"
    assert not any(mesh.visible for mesh in controller.showcase.slice_meshes.values())
    assert not controller.showcase.isosurface_mesh.visible
    data_revision = controller.showcase.volume.data_revision
    controller.showcase.set_representation("isosurface")
    assert controller.showcase.volume.render_mode == "isosurface"
    assert not controller.showcase.isosurface_mesh.visible
    geometry_revision = controller.scene.geometry_revision
    controller.showcase.set_isovalue(
        controller.showcase.isosurface.value + 0.01
    )
    assert controller.scene.geometry_revision == geometry_revision
    assert controller.showcase.volume.data_revision == data_revision
    clipping = controller.set_roi((1, 1, 1), (7, 7, 7))
    assert clipping.roi.minimum == (1.0, 1.0, 1.0)
    controller.playing = True
    assert not controller.advance(10.0)
    assert not controller.advance(10.01)
    assert controller.advance(11.0)
    assert controller.time_index == 1


def test_transfer_and_unchanged_controls_do_not_rebuild_geometry():
    controller = ScientificWorkbenchController.from_array(
        _series_data(), axis_order="tzyx",
    )
    geometry_revision = controller.scene.geometry_revision
    revision = controller.scene.revision
    controller.select_frame(0, 0)
    controller.showcase.set_slice_index("x", 4)
    controller.showcase.set_isovalue(controller.showcase.isosurface.value)
    controller.showcase.set_representation("combined")
    assert controller.scene.revision == revision
    data_revision = controller.showcase.volume.data_revision
    value_range = controller.showcase.transfer_function.mapping.value_range
    controller.configure_transfer(
        colormap="magma", value_range=value_range, opacity=(0.0, 0.25),
    )
    assert controller.scene.geometry_revision == geometry_revision
    assert controller.scene.shading_revision > 0
    assert controller.showcase.volume.data_revision == data_revision


def test_log_transfer_switch_retains_physical_gpu_volume_data():
    controller = ScientificWorkbenchController.from_array(
        _series_data() + 1.0, axis_order="tzyx",
    )
    volume = controller.showcase.volume
    expected = controller.showcase.field.data.copy()
    data_revision = volume.data_revision

    controller.configure_transfer(
        mode="log", value_range=(1.0, float(expected.max())),
    )

    assert volume.value_mapping == "log"
    assert volume.data_revision == data_revision
    np.testing.assert_array_equal(volume.data, expected)


def test_volume_only_playback_defers_hidden_topology_updates():
    controller = ScientificWorkbenchController.from_array(
        _series_data(), axis_order="tzyx",
    )
    controller.showcase.set_representation("volume")
    geometry_revision = controller.scene.geometry_revision
    controller.select_frame(1)
    assert controller.showcase.slices_dirty
    assert controller.showcase.isosurface_dirty
    # The resident volume changes as shading data; hidden triangle topology
    # waits until that representation is requested again.
    assert controller.scene.geometry_revision == geometry_revision
    controller.showcase.set_representation("slices")
    # GPU-native slice mode does not need to rebuild stale CPU slice meshes.
    assert controller.showcase.slices_dirty
    switched_geometry_revision = controller.scene.geometry_revision
    controller.showcase.set_slice_index("z", 3)
    assert controller.scene.geometry_revision == switched_geometry_revision
    assert controller.showcase.volume.slice_axis == "z"
    assert controller.showcase.volume.slice_position == 3 / 8
    assert controller.showcase.volume.slice_positions[2] == 3 / 8
    # Changing axes at the same index is still a real GPU-header update.
    assert controller.showcase.set_slice_index("x", 3)
    assert controller.showcase.volume.slice_axis == "x"
    assert controller.showcase.volume.slice_position == 3 / 8
    assert controller.showcase.volume.slice_positions[0] == 3 / 8
    assert controller.scene.geometry_revision == switched_geometry_revision
    controller.select_frame(0)
    assert controller.scene.geometry_revision == switched_geometry_revision
