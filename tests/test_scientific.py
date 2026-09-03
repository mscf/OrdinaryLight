import json
from pathlib import Path

import numpy as np

import ordinarylight as ol


def test_scalar_field_coordinates_probe_and_volume_adapter():
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    field = ol.ScalarField3D(
        data, spacing=(0.5, 2.0, 3.0), origin=(10.0, 20.0, 30.0),
        unit="K", name="temperature",
    )
    np.testing.assert_allclose(field.index_to_world((2, 1, 1)), (11, 22, 33))
    result = field.probe((11, 22, 33))
    assert result.nearest_index == (2, 1, 1)
    assert result.value == data[1, 1, 2]
    assert result.unit == "K" and result.valid

    transfer = ol.TransferFunction(
        ((0, 0, 1, 0), (1, 0, 0, 1)),
        ol.ScalarMapping("linear", (0, 23)),
    )
    scene = ol.Scene()
    volume = field.add_volume(scene, transfer, step_size=0.25)
    np.testing.assert_allclose(volume.transform.matrix[:3, 3], (10, 20, 30))
    np.testing.assert_allclose(np.diag(volume.transform.matrix)[:3], (1.5, 4, 3))
    assert volume.metadata["scientific"]["unit"] == "K"
    json.dumps(scene.snapshot())


def test_mapping_modes_and_nonfinite_values():
    data = np.asarray(([-10.0, -1.0], [1.0, np.nan]), np.float32)
    mapping = ol.ScalarMapping("symlog", (-10, 10), linear_threshold=1)
    normalized, valid = mapping.normalize(data)
    assert normalized[0, 0] == 0 and normalized[1, 0] > 0.5
    assert not valid[1, 1] and normalized[1, 1] == 0

    logarithmic = ol.ScalarMapping("log", (1, 100))
    normalized, valid = logarithmic.normalize((1, 10, 100, -1))
    np.testing.assert_allclose(normalized[:3], (0, 0.5, 1), atol=1e-6)
    assert not valid[3]

    transfer = ol.TransferFunction(((0, 0, 1, 1), (1, 0, 0, 1)), mapping)
    encoded, valid = transfer.encode_volume((-10, np.nan, 10))
    assert encoded[1] == 0 and not valid[1]
    sampled = transfer.volume_material().transfer_function.sample(encoded)
    np.testing.assert_allclose(sampled[1], (0, 0, 0, 0))
    np.testing.assert_allclose(sampled[[0, 2]], ((0, 0, 1, 1), (1, 0, 0, 1)))


def test_percentile_range_is_recorded_reproducibly():
    data = np.arange(101, dtype=np.float32).reshape(1, 1, 101)
    # Scalar fields require true 3-D renderable extents; mapping itself works
    # for arbitrary array shapes and resolves its data-dependent range.
    mapping = ol.ScalarMapping("percentile", percentiles=(10, 90))
    assert mapping.resolved_range(data) == (10.0, 90.0)
    transfer = ol.TransferFunction(((0, 0, 0, 0), (1, 1, 1, 1)), mapping)
    assert transfer.snapshot(data)["value_range"] == [10.0, 90.0]


def test_orthogonal_slice_uses_field_coordinates_transfer_and_exact_probe():
    data = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    field = ol.ScalarField3D(
        data, spacing=(2, 3, 4), origin=(10, 20, 30), unit="Pa", name="p",
    )
    transfer = ol.TransferFunction(
        ((0, 0, 1, 1), (1, 0, 0, 1)), ol.ScalarMapping("linear", (0, 59)),
    )
    scalar_slice = field.slice("y", 2, transfer)
    np.testing.assert_array_equal(scalar_slice.values, data[:, 2, :])
    np.testing.assert_allclose(
        scalar_slice.world_corners,
        ((10, 26, 30), (18, 26, 30), (18, 26, 38), (10, 26, 38)),
    )
    probe = scalar_slice.probe(field.index_to_world((3, 2, 1)))
    assert probe.valid and probe.value == data[1, 2, 3] and probe.unit == "Pa"
    assert not scalar_slice.probe(field.index_to_world((3, 1, 1))).valid

    scene = ol.Scene()
    mesh = scalar_slice.add_to_scene(scene)
    assert mesh.material.program is ol.scientific_slice_material
    np.testing.assert_allclose(
        mesh.attributes["scientific_rgba"], scalar_slice.rgba.reshape((-1, 4)),
    )
    assert mesh.metadata["scientific"]["kind"] == "scalar_slice"
    picked = ol.pick_ray(scene, (16, 0, 34), (0, 1, 0))
    assert picked is not None
    assert scalar_slice.probe(picked.position).value == data[1, 2, 3]


def test_isosurface_is_deterministic_coordinate_aware_and_pickable():
    z, y, x = np.mgrid[:4, :4, :4]
    data = x.astype(np.float32)
    field = ol.ScalarField3D(
        data, spacing=(2, 3, 4), origin=(10, 20, 30), unit="mol/m3", name="c",
    )
    transfer = ol.TransferFunction(
        ((0, 0, 1, 1), (1, 0, 0, 1)), ol.ScalarMapping("linear", (0, 3)),
    )
    surface = field.isosurface(1.5, transfer)
    repeated = field.isosurface(1.5, transfer)
    assert len(surface.indices) > 0
    np.testing.assert_array_equal(surface.index_vertices, repeated.index_vertices)
    np.testing.assert_array_equal(surface.indices, repeated.indices)
    np.testing.assert_allclose(surface.index_vertices[:, 0], 1.5)
    np.testing.assert_allclose(surface.world_vertices[:, 0], 13.0)

    scene = ol.Scene()
    mesh = surface.add_to_scene(scene)
    assert mesh.material.program is ol.unlit_material
    picked = ol.pick_ray(scene, (0, 24.5, 34), (1, 0, 0))
    assert picked is not None
    probe = surface.probe(picked.position)
    assert probe.valid and probe.value == 1.5 and probe.unit == "mol/m3"
    snapshot = mesh.metadata["scientific"]
    assert snapshot["algorithm"] == "marching_tetrahedra"


def test_isosurface_skips_cells_with_missing_data():
    data = np.zeros((3, 3, 3), np.float32)
    data[:, :, 1:] = 1
    complete = ol.ScalarField3D(data).isosurface(
        0.5, ol.TransferFunction(((0, 0, 0, 1), (1, 1, 1, 1))),
    )
    data[0, 0, 0] = np.nan
    missing = ol.ScalarField3D(data).isosurface(0.5, complete.transfer_function)
    assert 0 < len(missing.indices) < len(complete.indices)


def test_dynamic_field_updates_preserve_scene_identity_and_geometry():
    field = ol.ScalarField3D(np.zeros((4, 4, 4), np.float32), unit="K")
    transfer = ol.TransferFunction(
        ((0, 0, 0, 0), (1, 0, 0, 1)), ol.ScalarMapping("linear", (0, 10)),
    )
    scene = ol.Scene()
    volume = field.add_volume(scene, transfer)
    volume_id = volume.id
    geometry_revision = scene.geometry_revision
    shading_revision = scene.shading_revision

    previous = field.revision
    field.update((1, 1, 1), np.full((2, 1, 2), 5, np.float32))
    assert field.updates_since(previous) == ((1, (1, 1, 1), (2, 1, 2)),)
    current = field.sync_volume(
        scene, volume, transfer, since_revision=previous,
    )
    assert current == 1 and volume.id == volume_id
    assert scene.geometry_revision == geometry_revision
    assert scene.shading_revision > shading_revision
    assert volume.data_revision == 1
    assert volume.dirty_regions == (((1, 1, 1), (2, 1, 2)),)
    np.testing.assert_allclose(volume.data[1:3, 1:2, 1:3], 0.75)


def test_data_derived_mapping_falls_back_to_full_remap():
    field = ol.ScalarField3D(np.arange(8, dtype=np.float32).reshape(2, 2, 2))
    transfer = ol.TransferFunction(
        ((0, 0, 0, 0), (1, 1, 1, 1)), ol.ScalarMapping("linear"),
    )
    scene = ol.Scene()
    volume = field.add_volume(scene, transfer)
    field.update((0, 0, 0), np.asarray([[[100]]], np.float32))
    field.sync_volume(scene, volume, transfer, since_revision=0)
    assert volume.dirty_regions == (((0, 0, 0), volume.shape),)
    assert volume.data[0, 0, 0] == 1.0


def test_trilinear_sample_reports_physical_value_and_invalid_cells():
    z, y, x = np.mgrid[:3, :3, :3]
    field = ol.ScalarField3D(
        (x + 10 * y + 100 * z).astype(np.float32),
        spacing=(2, 3, 4), origin=(10, 20, 30), unit="K",
    )
    sample = field.sample(field.index_to_world((0.5, 1.5, 0.25)))
    assert sample.valid and sample.unit == "K"
    np.testing.assert_allclose(sample.value, 40.5)
    assert not field.sample((0, 0, 0)).valid

    field.data[0, 1, 0] = np.nan
    assert not field.sample(field.index_to_world((0.5, 1.5, 0.25))).valid


def test_ray_probe_passes_transparent_samples_and_returns_full_value_context():
    data = np.zeros((6, 2, 2), np.float32)
    data[3:] = 1.0
    field = ol.ScalarField3D(data, spacing=(1, 1, 1), unit="kg/m3")
    transfer = ol.TransferFunction(
        ((0, 0, 1, 0), (1, 0, 0, 1)), ol.ScalarMapping("linear", (0, 1)),
    )
    result = field.probe_ray(
        (0.5, 0.5, -2), (0, 0, 1), transfer,
        step_size=0.25, opacity_threshold=0.1,
    )
    assert result is not None and result.unit == "kg/m3"
    assert result.index_position[2] > 2.0
    assert result.value > 0.0 and result.normalized_value == result.value
    assert result.rgba[0] > 0.0 and result.accumulated_opacity >= 0.1

    assert field.probe_ray(
        (5, 5, -2), (0, 0, 1), transfer,
    ) is None

    camera = ol.PerspectiveCamera((0.5, 0.5, -2), (0.5, 0.5, 3))
    cursor = field.probe_pixel(
        camera, (31, 31), (15, 15), transfer,
        step_size=0.25, opacity_threshold=0.1,
    )
    assert cursor is not None and cursor.value > 0.0


def test_shared_clipping_applies_to_slice_isosurface_and_ray_probe():
    z, y, x = np.mgrid[:5, :5, :5]
    field = ol.ScalarField3D(x.astype(np.float32), name="x")
    transfer = ol.TransferFunction(
        ((0, 0, 1, 0), (1, 0, 0, 1)), ol.ScalarMapping("linear", (0, 4)),
    )
    clipping = ol.ClipRegion(
        planes=(ol.ClipPlane((0, 1, 0), 1.5),),
        roi=ol.RegionOfInterest((1, 0, 0), (3, 4, 4), "index"),
    )

    scalar_slice = field.slice("z", 2, transfer, clipping=clipping)
    scene = ol.Scene()
    slice_mesh = scalar_slice.add_texture_to_scene(scene)
    assert np.min(slice_mesh.vertices[:, 0]) >= 1 - 1e-6
    assert np.max(slice_mesh.vertices[:, 0]) <= 3 + 1e-6
    assert np.min(slice_mesh.vertices[:, 1]) >= 1.5 - 1e-6
    assert scalar_slice.probe((2, 1, 2)).valid is False
    assert scalar_slice.probe((2, 2, 2)).valid is True
    assert slice_mesh.metadata["scientific"]["clipping"]["roi"]["space"] == "index"

    surface = field.isosurface(2, transfer, clipping=clipping)
    assert len(surface.indices) > 0
    assert np.min(surface.world_vertices[:, 1]) >= 1.5 - 1e-6
    assert np.min(surface.world_vertices[:, 0]) >= 1 - 1e-6
    assert np.max(surface.world_vertices[:, 0]) <= 3 + 1e-6

    unclipped = field.probe_ray(
        (-2, 1, 2), (1, 0, 0), transfer,
        step_size=0.25, opacity_threshold=0.01,
    )
    clipped = field.probe_ray(
        (-2, 1, 2), (1, 0, 0), transfer,
        step_size=0.25, opacity_threshold=0.01, clipping=clipping,
    )
    assert unclipped is not None
    assert clipped is None  # y=1 is rejected by the shared plane.


def test_rotated_field_index_roi_becomes_world_clip_planes():
    direction = np.asarray(((0, -1, 0), (1, 0, 0), (0, 0, 1)), np.float64)
    field = ol.ScalarField3D(
        np.zeros((3, 3, 3), np.float32), origin=(10, 20, 30),
        direction=direction,
    )
    clipping = ol.ClipRegion(roi=ol.RegionOfInterest((1, 0, 0), (2, 2, 2)))
    inside = field.index_to_world((1.5, 1, 1))
    outside = field.index_to_world((0.5, 1, 1))
    assert clipping.contains(field, inside)
    assert not clipping.contains(field, outside)
    assert all(plane.signed_distance(inside) >= -1e-8
               for plane in clipping.world_planes(field))


def test_scientific_volume_carries_shared_clipping_and_metadata():
    field = ol.ScalarField3D(np.ones((3, 3, 3), np.float32), name="density")
    transfer = ol.TransferFunction(((0, 0, 0, 0), (1, 1, 1, 1)))
    clipping = ol.ClipRegion(
        roi=ol.RegionOfInterest((1, 0, 0), (2, 2, 2), "index"),
    )
    volume = field.add_volume(ol.Scene(), transfer, clipping=clipping)
    assert len(volume.clip_planes) == 6
    assert volume.metadata["scientific"]["clipping"] == clipping.snapshot()


def test_reproducible_image_export_records_and_verifies_full_state(tmp_path):
    field = ol.ScalarField3D(
        np.arange(8, dtype=np.float32).reshape(2, 2, 2),
        spacing=(1, 2, 3), unit="K", name="temperature",
    )
    transfer = ol.TransferFunction(
        ((0, 0, 1, 0), (1, 0, 0, 1)), ol.ScalarMapping("linear", (0, 7)),
    )
    clipping = ol.ClipRegion(planes=(ol.ClipPlane((1, 0, 0), 0.25),))
    scene = ol.Scene(metadata={"experiment": "export-test"})
    field.add_volume(scene, transfer, clipping=clipping)
    camera = ol.PerspectiveCamera((0, 0, -2), (0, 0, 0))
    image = np.linspace(0, 1, 48, dtype=np.float32).reshape(4, 4, 3)
    path = tmp_path / "result.png"
    document = ol.export_scientific_image(
        path, image, field=field, scene=scene, camera=camera,
        transfer_function=transfer, clipping=clipping,
        renderer={"backend": "reference", "samples": 4}, seed=17,
        tone_mapping="clip",
    )
    assert path.exists() and Path(f"{path}.json").exists()
    assert document["schema"] == ol.SCIENTIFIC_EXPORT_SCHEMA
    assert document["random_seed"] == 17
    assert document["field"]["sha256"] == ol.scalar_field_sha256(field)
    verified = ol.verify_scientific_export(path, field=field)
    assert verified == document
    with np.testing.assert_raises(FileExistsError):
        ol.export_scientific_image(
            path, image, field=field, scene=scene, camera=camera,
            transfer_function=transfer,
        )


def test_export_verifier_detects_image_or_field_changes(tmp_path):
    field = ol.ScalarField3D(np.zeros((2, 2, 2), np.float32))
    transfer = ol.TransferFunction(((0, 0, 0, 0), (1, 1, 1, 1)))
    path = tmp_path / "audit.png"
    ol.export_scientific_image(
        path, np.zeros((2, 2, 3), np.uint8), field=field,
        scene=ol.Scene(), camera=ol.PerspectiveCamera((0, 0, -2), (0, 0, 0)),
        transfer_function=transfer,
    )
    changed = ol.ScalarField3D(np.ones((2, 2, 2), np.float32))
    with np.testing.assert_raises_regex(ValueError, "source field checksum"):
        ol.verify_scientific_export(path, field=changed)
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with np.testing.assert_raises_regex(ValueError, "encoded image checksum"):
        ol.verify_scientific_export(path)


def test_builtin_colormaps_are_deterministic_and_transfer_compatible():
    assert ol.available_colormaps()[:2] == ("viridis", "cividis")
    forward = ol.colormap("cividis", 17)
    reverse = ol.colormap("cividis", 17, reverse=True)
    assert forward.shape == (17, 3) and forward.dtype == np.float32
    np.testing.assert_array_equal(forward, reverse[::-1])
    # Cividis is designed with increasing display luminance.
    luminance = forward @ np.asarray((0.2126, 0.7152, 0.0722), np.float32)
    assert np.all(np.diff(luminance) > 0)

    transfer = ol.TransferFunction.from_colormap(
        "viridis", samples=9, opacity=(0, 0.25, 1),
        mapping=ol.ScalarMapping("log", (1, 100)),
    )
    assert transfer.rgba.shape == (9, 4)
    assert transfer.rgba[0, 3] == 0 and transfer.rgba[-1, 3] == 1
    snapshot = transfer.snapshot()
    assert snapshot["mode"] == "log" and len(snapshot["rgba"]) == 9


def test_colormap_and_opacity_validation():
    with np.testing.assert_raises_regex(ValueError, "unknown color map"):
        ol.colormap("rainbow")
    with np.testing.assert_raises(ValueError):
        ol.colormap("viridis", 1)
    with np.testing.assert_raises(ValueError):
        ol.opacity_curve((0, 2), 8)
    alpha = ol.opacity_curve(lambda x: x * x, 5)
    np.testing.assert_allclose(alpha, (0, 0.0625, 0.25, 0.5625, 1))


def test_float_slice_preserves_colors_and_rgba8_path_handles_variable_alpha():
    data = np.arange(12, dtype=np.float32).reshape(3, 2, 2)
    field = ol.ScalarField3D(data)
    opaque = ol.TransferFunction.from_colormap(
        "viridis", samples=16, mapping=ol.ScalarMapping("linear", (0, 11)),
    )
    scalar_slice = field.slice("z", 1, opaque)
    mesh = scalar_slice.add_to_scene(ol.Scene())
    assert mesh.attributes["scientific_rgba"].dtype == np.float32
    np.testing.assert_array_equal(
        mesh.attributes["scientific_rgba"], scalar_slice.rgba.reshape((-1, 4)),
    )
    assert mesh.metadata["scientific"]["representation"] == "float32_vertex_rgba"

    translucent = ol.TransferFunction.from_colormap(
        "viridis", samples=16, opacity=(0, 1),
        mapping=ol.ScalarMapping("linear", (0, 11)),
    )
    translucent_slice = field.slice("z", 1, translucent)
    with np.testing.assert_raises_regex(ValueError, "uniform opacity"):
        translucent_slice.add_to_scene(ol.Scene())
    compatibility = translucent_slice.add_texture_to_scene(ol.Scene())
    assert compatibility.material.base_color_texture.pixels.dtype == np.uint8
    assert compatibility.metadata["scientific"]["representation"] == "rgba8_texture"


def test_time_channel_series_preserves_views_coordinates_and_units():
    source = np.arange(2 * 3 * 4 * 5 * 6, dtype=np.float32).reshape(2, 3, 4, 5, 6)
    series = ol.ScalarFieldSeries.from_array(
        source, axis_order="tczyx", times=(0.0, 0.5),
        channels=("temperature", "pressure", "density"),
        channel_units=("K", "Pa", "kg/m3"), time_unit="s",
        spacing=(2, 3, 4), origin=(10, 20, 30), name="simulation",
    )
    frame = series.frame(1, "pressure")
    assert np.shares_memory(frame.data, source)
    np.testing.assert_array_equal(frame.data, source[1, 1])
    assert frame.unit == "Pa" and frame.metadata["time"] == 0.5
    assert frame.metadata["channel"] == "pressure"
    np.testing.assert_allclose(frame.index_to_world((1, 1, 1)), (12, 23, 34))
    assert series.snapshot()["axis_order"] == "tczyx"


def test_series_axis_adaptation_temporal_interpolation_and_updates():
    # Input deliberately orders space x/z/y; labels make the conversion explicit.
    source = np.zeros((2, 3, 4, 5), np.float32)
    source[1] = 10
    series = ol.ScalarFieldSeries.from_array(
        source, axis_order="tzyx", times=(2, 4), channels=("signal",),
    )
    nearest = series.at_time(2.9, "signal", interpolation="nearest")
    assert np.shares_memory(nearest.data, source) and np.max(nearest.data) == 0
    interpolated = series.at_time(3, "signal", interpolation="linear")
    assert not np.shares_memory(interpolated.data, source)
    np.testing.assert_allclose(interpolated.data, 5)
    assert interpolated.metadata["source_frames"] == [0, 1]

    revision = series.update(
        0, "signal", (1, 1, 1), np.asarray([[[7]]], np.float32),
    )
    assert revision == 1 and source[0, 1, 1, 1] == 7
    assert series.updates_since(0) == ((1, 0, 0, (1, 1, 1), (1, 1, 1)),)


def test_series_rejects_ambiguous_axes_and_coordinates():
    data = np.zeros((2, 2, 2, 2), np.float32)
    with np.testing.assert_raises_regex(ValueError, "axis_order"):
        ol.ScalarFieldSeries.from_array(data, axis_order="abcd")
    with np.testing.assert_raises_regex(ValueError, "strictly increasing"):
        ol.ScalarFieldSeries.from_array(data, axis_order="tzyx", times=(1, 1))
    with np.testing.assert_raises_regex(ValueError, "unique"):
        ol.ScalarFieldSeries.from_array(
            np.zeros((2, 2, 2, 2), np.float32), axis_order="czyx",
            channels=("same", "same"),
        )


def test_scientific_inspector_probes_a_viewport_cursor_and_retains_context():
    field = ol.ScalarField3D(
        np.ones((3, 3, 3), np.float32), origin=(-1, -1, -1), unit="K",
    )
    transfer = ol.TransferFunction.from_colormap(
        "viridis", mapping=ol.ScalarMapping("linear", (0, 1)),
    )
    camera = ol.PerspectiveCamera((0, 0, -4), (0, 0, 0))

    class Viewport:
        framebuffer_size = (101, 101)
        cursor_pixel = staticmethod(lambda: (50, 50))

    viewport = Viewport()
    viewport.camera = camera
    inspector = ol.ScientificInspector(field, transfer, step_size=0.1)
    result = inspector.probe_viewport(viewport)
    assert result is inspector.last_result
    assert result is not None and result.unit == "K"
    np.testing.assert_allclose(result.world_position[:2], (0, 0), atol=1e-6)
    inspector.clear()
    assert inspector.last_result is None
