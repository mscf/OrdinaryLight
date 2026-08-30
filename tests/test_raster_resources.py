import numpy as np

import ordinarylight as ol


def test_raster_gpu_abi_has_aligned_portable_records():
    for dtype in (
        ol.CAMERA_DTYPE, ol.MATERIAL_DTYPE, ol.LIGHT_DTYPE, ol.DRAW_DTYPE,
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
    assert packed.shadow_maps == ()


def test_shadow_map_plan_supports_directional_and_spot_first():
    scene = ol.Scene(lights=[
        ol.PointLight((0, 1, 0)),
        ol.DirectionalLight((0, -1, 0)),
        ol.SpotLight((0, 2, 0), (0, -1, 0)),
    ])
    requests = ol.plan_shadow_maps(scene, extent=(512, 256))
    assert tuple(item.kind for item in requests) == ("directional", "spot")
    assert requests[0].extent == (512, 256)
    assert requests[0].view_projection.shape == (4, 4)
    assert np.all(np.isfinite(requests[1].view_projection))
    assert not requests[0].view_projection.flags.writeable
