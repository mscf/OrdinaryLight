from pathlib import Path

import numpy as np
import pytest

import ordinarylight as ol
from ordinarylight.showcases.optical_materials import (
    build_absorption_scene, build_environment_reflection_scene,
    build_nested_dielectric_scene, build_reflection_probe_scene,
    build_refraction_scene, build_transparency_scene,
)


def _triangle(z=0.0):
    return (
        np.asarray(((-1, 0, z), (1, 0, z), (0, 1, z)), np.float32),
        np.asarray(((0, 1, 2),), np.uint32),
    )


def test_optical_material_parameters_validate_and_pack():
    texture = ol.Texture(np.full((1, 1, 4), 255, np.uint8))
    material = ol.Material(
        transmission=1.0, thickness=2.5, opacity=0.4,
        alpha_mode="mask", alpha_cutoff=0.25,
        thickness_texture=texture,
    )
    scene = ol.Scene()
    scene.add_mesh(*_triangle(), material, texcoords=((0, 0), (1, 0), (0, 1)))
    packed = scene.triangle_material_data()[0]
    assert packed[7, 3] == pytest.approx(2.5)
    assert packed[11, 0] >= 0.0
    np.testing.assert_allclose(packed[11, 1:], (0.4, 0.25, 1.0))
    for kwargs in (
        {"thickness": -1}, {"opacity": 1.1}, {"alpha_cutoff": -0.1},
        {"alpha_mode": "unknown"},
    ):
        with pytest.raises(ValueError):
            ol.Material(**kwargs)


def test_reflection_probe_is_scene_owned_and_validated():
    image = np.ones((4, 8, 3), np.float32)
    probe = ol.ReflectionProbe(image, position=(1, 2, 3), radius=4)
    scene = ol.Scene()
    assert scene.add_reflection_probe(probe) is probe
    assert scene.reflection_probes == [probe]
    assert scene.remove_reflection_probe(probe) is probe
    with pytest.raises(ValueError):
        ol.ReflectionProbe(image, radius=0)
    with pytest.raises(ValueError):
        ol.ReflectionProbe(np.ones((4, 8), np.float32))


def test_raster_scene_packs_environment_and_transparency():
    scene = build_environment_reflection_scene()
    camera = ol.PerspectiveCamera((0, 3, 8), (0, 1, 0))
    mesh = ol.scene_mesh(scene, camera, 320, 180)
    assert mesh.resources["base_color_atlas"].size > 0
    materials = np.frombuffer(mesh.resources["material_buffer"], ol.MATERIAL_DTYPE)
    assert materials[0]["environment_rect"][2] > 0
    assert materials[0]["environment_color_intensity"][3] > 0

    transparent = build_transparency_scene()
    mesh = ol.scene_mesh(transparent, camera, 320, 180)
    assert mesh.resources["transparent"] is True
    assert np.any(mesh.vertices[:, 16] < 1.0)
    expected_opaque_indices = sum(
        item.indices.size for item in transparent.visible_meshes
        if item.material.alpha_mode != "blend"
    )
    assert mesh.resources["opaque_index_count"] == expected_opaque_indices
    assert mesh.resources["shadow_indices"].size == expected_opaque_indices


def test_transparency_sort_uses_signed_camera_space_depth():
    scene = build_transparency_scene()

    def transparent_material_order(camera):
        mesh = ol.scene_mesh(scene, camera, 320, 180)
        offset = mesh.resources["opaque_index_count"]
        # Each showcase layer is one indexed quad (six indices). Location 19
        # is the packed scene material index at float offset 53.
        return tuple(
            int(mesh.vertices[mesh.indices[offset + index * 6], 53])
            for index in range(3)
        )

    front = ol.PerspectiveCamera((0, 2, 8), (0, 1.5, 0))
    back = ol.PerspectiveCamera((0, 2, -8), (0, 1.5, 0))
    assert transparent_material_order(front) == (5, 6, 7)
    assert transparent_material_order(back) == (7, 6, 5)


def test_gi_sources_keep_nested_media_tir_and_beer_absorption():
    shader_root = Path(ol.__file__).parent / "shaders"
    primary = (shader_root / "wavefront_primary_impl.glsl").read_text()
    shade = (shader_root / "wavefront_shade.comp").read_text()
    combined = primary + shade
    assert "medium_depth" in combined
    assert "ordinarylight_secondary_refracted_direction" in combined
    assert "ordinarylight_secondary_medium_depth" in combined
    assert "material.ior_distance.y" in combined
    assert "vec3 absorption = pow(" in combined


def test_optical_showcases_construct_distinct_valid_scenes():
    builders = (
        build_environment_reflection_scene, build_reflection_probe_scene,
        build_refraction_scene, build_absorption_scene,
        build_nested_dielectric_scene, build_transparency_scene,
    )
    scenes = [builder() for builder in builders]
    assert all(scene.visible_meshes for scene in scenes)
    assert all(
        any(mesh.name == "checker-floor" for mesh in scene.visible_meshes)
        and any(mesh.name == "pattern-wall" for mesh in scene.visible_meshes)
        for scene in scenes
    )
    assert all(isinstance(scene.lights[0], ol.SpotLight) for scene in scenes)
    assert all(
        any(mesh.name == "warm-light-strip" for mesh in scene.visible_meshes)
        for scene in scenes
    )
    assert scenes[0].environment is not None
    assert scenes[1].reflection_probes
    assert any(mesh.material.transmission for mesh in scenes[2].visible_meshes)
    assert any(mesh.material.attenuation_distance < 1 for mesh in scenes[3].visible_meshes)
    assert sum(mesh.material.transmission > 0 for mesh in scenes[4].visible_meshes) >= 2
    assert any(mesh.material.alpha_mode == "blend" for mesh in scenes[5].visible_meshes)


def test_optical_diorama_receivers_face_the_room_and_key_light():
    scene = build_environment_reflection_scene()
    floor = next(mesh for mesh in scene.visible_meshes if mesh.name == "checker-floor")
    wall = next(mesh for mesh in scene.visible_meshes if mesh.name == "pattern-wall")
    np.testing.assert_allclose(
        floor.world_normals, np.tile((0.0, 1.0, 0.0), (4, 1)),
    )
    np.testing.assert_allclose(
        wall.world_normals, np.tile((0.0, 0.0, 1.0), (4, 1)),
    )

    light = scene.lights[0]
    floor_center = floor.world_vertices.mean(axis=0)
    incoming = np.asarray(light.position) - floor_center
    incoming /= np.linalg.norm(incoming)
    # Reversed receiver winding zeroes N.L, leaving no direct contribution
    # for the shadow map to attenuate.
    assert float(np.dot(floor.world_normals[0], incoming)) > 0.5


def test_native_targets_split_transparent_depth_write_pass():
    root = Path(ol.__file__).parent / "renderers" / "raster"
    vulkan = (root / "vulkan.py").read_text()
    webgpu = (root / "webgpu.py").read_text()
    for source in (vulkan, webgpu):
        assert '"opaque_index_count"' in source
        assert '"optical_opaque_index_count"' in source
        assert '"transparent_index_count"' in source
        assert '"optical_transmissive_index_count"' in source
        assert "transparent_count" in source
        assert '"optical-opaque"' in source
        assert '"transmissive"' in source
    assert 'pass_kind == "opaque"' in vulkan
    assert "self.state.depth_write and not optical" in webgpu
    assert "pass_kind == \"transmissive\"" in vulkan
    assert '"back" if pass_kind == "transmissive"' in webgpu
    # Optical redraws must remain occluded by the opaque prepass.  An ALWAYS
    # comparison makes objects behind walls visibly overwrite those walls.
    assert 'pass_kind == "optical-opaque" else' not in vulkan
    assert '"always" if pass_kind == "optical-opaque"' not in webgpu
    assert "vk.VK_COMPARE_OP_LESS_OR_EQUAL" in vulkan
    assert '"less-equal" if optical' in webgpu


def test_screen_space_optics_is_explicit_and_preserves_environment_default():
    assert ol.RasterConfig().optical_quality == "environment"
    config = ol.RasterConfig(
        optical_quality="screen-space", screen_space_ray_steps=32,
    )
    assert config.optical_quality == "screen-space"
    assert config.screen_space_ray_steps == 32
    assert tuple(stage.name for stage in ol.create_raster_pipeline(config).stages)[:3] == (
        "shadow_maps", "opaque_prepass", "screen_space_optics",
    )
    with pytest.raises(ValueError):
        ol.RasterConfig(optical_quality="path-traced")
    with pytest.raises(ValueError):
        ol.RasterConfig(screen_space_ray_steps=2)


def test_screen_space_optics_partitions_prepass_and_optical_draws():
    scene = build_refraction_scene()
    camera = ol.PerspectiveCamera((0, 3, 8), (0, 1, 0))
    packed = ol.scene_mesh(
        scene, camera, 320, 180,
        ol.RasterConfig(optical_quality="screen-space"),
    )
    assert packed.resources["opaque_prepass_index_count"] > 0
    assert packed.resources["optical_index_count"] > 0
    assert (
        packed.resources["opaque_prepass_index_count"]
        + packed.resources["optical_index_count"]
        == packed.indices.size
    )
    # Transmission is composited back-to-front even when the authored glTF
    # alpha mode is opaque. It must not enter the overwrite-only reflector
    # range, where front and rear faces can replace one another.
    assert packed.resources["optical_opaque_index_count"] == 0
    assert packed.resources["transparent_index_count"] > 0


def test_screen_space_optical_draws_sort_far_to_near_from_either_side():
    material = lambda color: ol.Material(
        base_color=color, transmission=1.0, opacity=1.0,
    )
    scene = ol.Scene()
    scene.add_mesh(*_triangle(1.0), material((1, 0, 0)), name="near-front")
    scene.add_mesh(*_triangle(-1.0), material((0, 0, 1)), name="near-back")
    config = ol.RasterConfig(optical_quality="screen-space")

    def order(camera):
        packed = ol.scene_mesh(scene, camera, 64, 64, config)
        offset = next(
            item.offset // 4 for item in packed.layout.attributes
            if item.semantic == "material_index"
        )
        decoded = tuple(
            int(packed.vertices[packed.indices[index * 3], offset])
            for index in range(2)
        )
        assert packed.resources["camera_order_token"] == decoded
        return decoded

    assert order(ol.PerspectiveCamera((0, 0, 5), (0, 0, 0))) == (1, 0)
    assert order(ol.PerspectiveCamera((0, 0, -5), (0, 0, 0))) == (0, 1)


def test_nested_dielectric_shell_order_is_stable_under_camera_motion():
    scene = build_nested_dielectric_scene()
    visible = tuple(scene.visible_meshes)
    outer = next(index for index, mesh in enumerate(visible)
                 if mesh.name == "outer-glass")
    inner = next(index for index, mesh in enumerate(visible)
                 if mesh.name == "inner-liquid")
    config = ol.RasterConfig(optical_quality="screen-space")

    orders = []
    for offset in (-0.02, 0.0, 0.02):
        camera = ol.PerspectiveCamera(
            (offset, 3.0, 9.0), (0.0, 1.45, 0.0),
        )
        packed = ol.scene_mesh(scene, camera, 320, 180, config)
        orders.append(packed.resources["camera_order_token"])

    assert all(order.index(outer) < order.index(inner) for order in orders)
    assert orders[0] == orders[1] == orders[2]


def test_opaque_reflectors_are_available_to_neighboring_screen_rays():
    scene = ol.Scene()
    reflector = ol.Material(metallic=1.0, roughness=0.1)
    scene.add_mesh(*_triangle(0.5), reflector, name="reflector-a")
    scene.add_mesh(*_triangle(-0.5), reflector, name="reflector-b")
    packed = ol.scene_mesh(
        scene, ol.PerspectiveCamera((0, 0, 5), (0, 0, 0)), 64, 64,
        ol.RasterConfig(optical_quality="screen-space"),
    )
    # Both reflectors appear once in opaque scene color and once again in the
    # optical composite. Their second draw uses equal-depth testing.
    assert packed.resources["opaque_prepass_index_count"] == 6
    assert packed.resources["optical_index_count"] == 6
    assert packed.indices.size == 12


def test_screen_space_resources_are_portable_shader_bindings():
    program = ol.RasterProgram.scene(target="wgsl", validate=False)
    bindings = {
        resource["name"]: resource["binding"]
        for resource in program.fragment.reflection.resources
    }
    assert bindings["scene_color"] == 6
    assert bindings["scene_depth"] == 7
    assert bindings["scene_sampler"] == 8
    assert bindings["scene_depth_sampler"] == 9
    assert "textureSampleLevel(scene_depth, scene_depth_sampler, ray_uv, 0" in program.fragment.source
    assert "for (var ray_step: i32 = 1; ray_step < 25" in program.fragment.source
    assert "for (var refine_step" in program.fragment.source
    assert "edge_confidence" in program.fragment.source
    assert "different_object" in program.fragment.source
    assert "object_tag" in program.fragment.source
    assert "depth_confidence" in program.fragment.source
    assert "reflection_radius" in program.fragment.source
