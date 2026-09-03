import numpy as np

from ordinarylight.showcases.scientific import (
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
