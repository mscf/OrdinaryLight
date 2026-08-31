import numpy as np

import ordinarylight as ol


def test_raster_gpu_abi_has_aligned_portable_records():
    for dtype in (
        ol.CAMERA_DTYPE, ol.MATERIAL_DTYPE, ol.LIGHT_DTYPE, ol.DRAW_DTYPE,
        ol.SHADOW_DTYPE,
    ):
        assert dtype.isalignedstruct
        assert dtype.itemsize % 16 == 0


def test_raster_gpu_scene_packs_materials_lights_draws_and_textures():
    texture = ol.Texture(np.full((1, 1, 4), 255, np.uint8))
    mesh = ol.Mesh(
        [[-1, -1, 0], [1, -1, 0], [0, 1, 0]], [[0, 1, 2]],
        ol.Material(
            base_color=(0.2, 0.4, 0.8), metallic=0.75, roughness=0.25,
            transmission=0.5, base_color_texture=texture,
            clearcoat=0.8, clearcoat_roughness=0.12,
            sheen_color=(0.2, 0.4, 0.8), anisotropy=0.6,
            subsurface=0.4, subsurface_color=(1.0, 0.2, 0.1),
            thin_walled=True, clearcoat_texture=texture,
        ),
        transform=ol.Transform.translation((1, 2, 3)),
    )
    scene = ol.Scene([mesh], [ol.PointLight((0, 2, 1), intensity=4)])
    packed = ol.pack_raster_gpu_scene(
        scene, ol.PerspectiveCamera((0, 0, 4), (0, 0, 0)), 640, 480,
    )
    assert packed.materials.shape == (1,)
    assert packed.lights.shape == (1,)
    assert packed.draws.shape == (1,)
    assert packed.textures == (texture,)
    np.testing.assert_allclose(
        packed.materials["base_color_roughness"][0], (0.2, 0.4, 0.8, 0.25),
    )
    assert packed.materials["emission_metallic"][0, 3] == 0.75
    assert packed.materials["texture_indices"][0, 0] == 0
    np.testing.assert_allclose(
        packed.materials["advanced0"][0], (0.8, 0.12, 0.5, 0.6),
    )
    assert packed.materials["advanced1"][0, 0] == 0.4
    assert packed.materials["advanced1"][0, 2] == 1.0
    assert packed.materials["advanced_texture_indices"][0, 0] == 0
    np.testing.assert_allclose(packed.draws["model"][0, :3, 3], (1, 2, 3))
    assert len(packed.shadow_maps) == 6
    assert tuple(item.kind for item in packed.shadow_maps) == ("point",) * 6
    assert tuple(item.face_index for item in packed.shadow_maps) == tuple(range(6))


def test_scene_mesh_exposes_the_packed_multi_light_gpu_array():
    mesh = ol.Mesh(
        [[-1, -1, 0], [1, -1, 0], [0, 1, 0]], [[0, 1, 2]],
        ol.Material(),
    )
    scene = ol.Scene([mesh], [
        ol.DirectionalLight((0, -1, -1), color=(1, 0, 0), intensity=2),
        ol.PointLight((1, 2, 3), color=(0, 1, 0), intensity=4, range=7),
        ol.SpotLight(
            (-1, 2, 3), (0, -1, -1), color=(0, 0, 1), intensity=6,
            inner_cone_angle=0.2, outer_cone_angle=0.5, range=9,
        ),
    ])
    raster = ol.scene_mesh(
        scene, ol.PerspectiveCamera((0, 0, 4), (0, 0, 0)), 64, 64,
    )
    lights = np.frombuffer(raster.resources["light_buffer"], ol.LIGHT_DTYPE)
    assert raster.resources["light_count"] == 3
    assert lights.shape == (3,)
    assert tuple(lights["position_type"][:, 3]) == (1.0, 0.0, 2.0)
    np.testing.assert_allclose(lights["color_intensity"][:, 3], (2, 4, 6))
    np.testing.assert_allclose(lights["direction_range"][1, 3], 7)
    np.testing.assert_allclose(lights["spot"][2, :2], (0.2, 0.5))


def test_scene_without_environment_keeps_environment_sampling_disabled():
    from ordinarylight.raster._core import prepare_scene_mesh_resources

    mesh = ol.Mesh(
        [[-1, -1, 0], [1, -1, 0], [0, 1, 0]], [[0, 1, 2]],
        ol.Material(transmission=1.0, thin_walled=True),
    )
    scene = ol.Scene([mesh])
    prepared = prepare_scene_mesh_resources(scene, ol.RasterConfig())
    assert prepared["environment_parameters"] == (None,)

    packed = ol.pack_raster_gpu_scene(
        scene, ol.PerspectiveCamera((0, 0, 4), (0, 0, 0)), 640, 480,
        environment_parameters=prepared["environment_parameters"],
    )
    assert packed.materials["environment_rotation_log_range"][0, 2] == 0.0
    np.testing.assert_allclose(
        packed.materials["environment_color_intensity"][0], 0.0,
    )


def test_shadow_map_plan_supports_point_directional_and_spot_lights():
    scene = ol.Scene(lights=[
        ol.PointLight((0, 1, 0)),
        ol.DirectionalLight((0, -1, 0)),
        ol.SpotLight((0, 2, 0), (0, -1, 0)),
    ])
    requests = ol.plan_shadow_maps(scene, extent=(512, 256))
    assert tuple(item.kind for item in requests) == (
        *("point",) * 6, "directional", "spot",
    )
    assert tuple(item.face_index for item in requests[:6]) == tuple(range(6))
    assert requests[0].extent == (512, 256)
    assert requests[0].view_projection.shape == (4, 4)
    assert np.all(np.isfinite(requests[6].view_projection))
    assert not requests[0].view_projection.flags.writeable


def test_shadow_map_limit_counts_lights_not_point_faces():
    scene = ol.Scene(lights=[
        ol.PointLight((0, 1, 0)),
        ol.DirectionalLight((0, -1, 0)),
    ])
    requests = ol.plan_shadow_maps(scene, max_maps=1)
    assert len(requests) == 6
    assert {item.light_index for item in requests} == {0}


def test_native_point_shadow_atlas_packs_six_records_and_faces():
    from ordinarylight.raster._core import prepare_scene_mesh_resources
    from ordinarylight.showcases.raster_features import build_point_shadow_scene

    scene = build_point_shadow_scene()
    prepared = prepare_scene_mesh_resources(
        scene, ol.RasterConfig(shadow_map_size=64), native_shadow_maps=True,
    )
    assert len(prepared["shadow_requests"]) == 6
    assert len(prepared["shadow_records"]) == 6
    assert prepared["shadow_rectangle"] == (0, 0, 192, 128, 192, 128)
    assert prepared["shadow_vertices"].shape[1] == 4
    assert prepared["shadow_indices"].size > 0
    assert prepared["shadow_indices"].size % 3 == 0


def test_native_point_shadow_atlas_caps_total_attachment_extent():
    from ordinarylight.raster._core import prepare_scene_mesh_resources
    from ordinarylight.showcases.raster_features import build_point_shadow_scene

    prepared = prepare_scene_mesh_resources(
        build_point_shadow_scene(),
        ol.RasterConfig(shadow_map_size=8192),
        native_shadow_maps=True,
    )
    _x, _y, width, height, atlas_width, atlas_height = prepared[
        "shadow_rectangle"
    ]
    assert max(width, height, atlas_width, atlas_height) <= 8192
    assert prepared["shadow_map_size"] == 8192 // 3
    assert all(
        request.extent == (prepared["shadow_map_size"],) * 2
        for request in prepared["shadow_requests"]
    )


def test_mixed_light_shadow_scene_packs_ten_independent_records():
    from ordinarylight.raster._core import prepare_scene_mesh_resources
    from ordinarylight.showcases.raster_features import (
        build_multi_light_shadow_scene,
    )

    scene = build_multi_light_shadow_scene()
    prepared = prepare_scene_mesh_resources(
        scene, ol.RasterConfig(shadow_map_size=32), native_shadow_maps=True,
    )
    assert len(prepared["shadow_records"]) == 8
    assert tuple(item.kind for item in prepared["shadow_requests"]) == (
        *("point",) * 6, "spot", "directional",
    )
    assert {
        int(record["parameters"][0]) for record in prepared["shadow_records"]
    } == {
        0, 1, 2,
    }
