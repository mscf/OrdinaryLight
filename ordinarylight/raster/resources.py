"""Target-neutral GPU resource ABI for raster scene evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..lights import DirectionalLight, PointLight, SpotLight


CAMERA_DTYPE = np.dtype([
    ("view_projection", np.float32, (4, 4)),
    ("position_exposure", np.float32, (4,)),
], align=True)

MATERIAL_DTYPE = np.dtype([
    ("base_color_roughness", np.float32, (4,)),
    ("emission_metallic", np.float32, (4,)),
    ("attenuation_transmission", np.float32, (4,)),
    ("ior_distance_program_flags", np.float32, (4,)),
    ("texture_indices", np.float32, (4,)),
    ("normal_occlusion_transmission", np.float32, (4,)),
    ("advanced0", np.float32, (4,)),
    ("advanced1", np.float32, (4,)),
    ("sheen_color", np.float32, (4,)),
    ("subsurface_color", np.float32, (4,)),
    ("advanced_texture_indices", np.float32, (4,)),
    ("optical", np.float32, (4,)),
    ("environment_rect", np.float32, (4,)),
    ("environment_color_intensity", np.float32, (4,)),
    ("environment_rotation_log_range", np.float32, (4,)),
], align=True)

LIGHT_DTYPE = np.dtype([
    ("position_type", np.float32, (4,)),
    ("direction_range", np.float32, (4,)),
    ("color_intensity", np.float32, (4,)),
    ("spot", np.float32, (4,)),
], align=True)

DRAW_DTYPE = np.dtype([
    ("model", np.float32, (4, 4)),
    ("normal", np.float32, (4, 4)),
    ("indices", np.uint32, (4,)),
], align=True)


@dataclass(frozen=True, slots=True)
class RasterGpuScene:
    """Packed renderer-neutral resources consumed by native raster targets."""

    camera: np.ndarray
    materials: np.ndarray
    lights: np.ndarray
    draws: np.ndarray
    textures: tuple
    shadow_maps: tuple = ()
    programs: tuple = ()

    def __post_init__(self):
        expected = (
            (self.camera, CAMERA_DTYPE, (1,)),
            (self.materials, MATERIAL_DTYPE, None),
            (self.lights, LIGHT_DTYPE, None),
            (self.draws, DRAW_DTYPE, None),
        )
        for value, dtype, shape in expected:
            if value.dtype != dtype or not value.flags.c_contiguous:
                raise TypeError("raster GPU records must use their canonical dtype")
            if shape is not None and value.shape != shape:
                raise ValueError(f"raster GPU record must have shape {shape}")


def _texture_table(scene):
    textures = []
    lookup = {}
    for mesh in scene.visible_meshes:
        for texture in (
            mesh.material.base_color_texture,
            mesh.material.metallic_roughness_texture,
            mesh.material.emissive_texture,
            mesh.material.normal_texture,
            mesh.material.occlusion_texture,
            mesh.material.transmission_texture,
            mesh.material.thickness_texture,
            mesh.material.clearcoat_texture,
            mesh.material.sheen_texture,
            mesh.material.anisotropy_texture,
            mesh.material.subsurface_texture,
        ):
            if texture is not None and id(texture) not in lookup:
                lookup[id(texture)] = len(textures)
                textures.append(texture)
    return tuple(textures), lookup


def pack_raster_gpu_scene(
    scene, camera, width, height, *, exposure=1.0, default_program=None,
    environment_rectangle=None, environment_log_range=0.0,
    environment_parameters=None,
):
    """Pack one scene revision into the shared Vulkan/WebGPU raster ABI."""
    from ._core import camera_matrix

    camera_data = np.zeros(1, CAMERA_DTYPE)
    camera_data["view_projection"][0] = camera_matrix(camera, width, height)
    camera_data["position_exposure"][0] = (*camera.position, float(exposure))
    textures, texture_lookup = _texture_table(scene)
    if default_program is None:
        from ..materials import builtin_material
        default_program = builtin_material
    programs = scene.material_programs(default_program)
    materials = np.zeros(len(scene.visible_meshes), MATERIAL_DTYPE)
    draws = np.zeros(len(scene.visible_meshes), DRAW_DTYPE)
    for index, mesh in enumerate(scene.visible_meshes):
        material = mesh.material
        materials["base_color_roughness"][index] = (
            *material.base_color, material.roughness,
        )
        materials["emission_metallic"][index] = (
            *material.emission, material.metallic,
        )
        materials["attenuation_transmission"][index] = (
            *material.attenuation_color, material.transmission,
        )
        program = material.program or default_program
        program_id = next(
            item for item, candidate in enumerate(programs)
            if candidate is program
        )
        raster_kinds = {
            "pbr": 0.0, "diffuse": 1.0, "mirror": 2.0,
            "glass": 3.0, "unlit": 4.0,
        }
        materials["ior_distance_program_flags"][index] = (
            material.ior, material.attenuation_distance, float(program_id),
            raster_kinds[program.raster_kind]
            + (0.25 if material.emission_two_sided else 0.0),
        )
        texture_values = tuple(
            -1 if texture is None else texture_lookup[id(texture)]
            for texture in (
                material.base_color_texture,
                material.metallic_roughness_texture,
                material.emissive_texture,
                material.normal_texture,
                material.occlusion_texture,
                material.transmission_texture,
            )
        )
        materials["texture_indices"][index] = (*texture_values[:3], texture_values[3])
        materials["normal_occlusion_transmission"][index] = (
            material.normal_scale, texture_values[4],
            material.occlusion_strength, texture_values[5],
        )
        materials["advanced0"][index] = (
            material.clearcoat, material.clearcoat_roughness,
            material.sheen_roughness, material.anisotropy,
        )
        materials["advanced1"][index] = (
            material.subsurface, material.subsurface_radius,
            float(material.thin_walled), material.thickness,
        )
        materials["sheen_color"][index] = (*material.sheen_color, 0.0)
        materials["subsurface_color"][index] = (
            *material.subsurface_color, 0.0,
        )
        materials["advanced_texture_indices"][index] = tuple(
            -1 if texture is None else texture_lookup[id(texture)]
            for texture in (
                material.clearcoat_texture, material.sheen_texture,
                material.anisotropy_texture, material.subsurface_texture,
            )
        )
        alpha_modes = {"opaque": 0.0, "mask": 1.0, "blend": 2.0}
        materials["optical"][index] = (
            -1 if material.thickness_texture is None else
            texture_lookup[id(material.thickness_texture)],
            material.opacity, material.alpha_cutoff,
            alpha_modes[material.alpha_mode],
        )
        environment = scene.environment
        if environment_rectangle is not None:
            x, y, rect_width, rect_height, atlas_width, atlas_height = environment_rectangle
            materials["environment_rect"][index] = (
                x / atlas_width, y / atlas_height,
                rect_width / atlas_width, rect_height / atlas_height,
            )
        if environment_parameters is not None:
            color_intensity, rotation = environment_parameters
            materials["environment_color_intensity"][index] = color_intensity
            materials["environment_rotation_log_range"][index] = (
                rotation, environment_log_range, 1.0, 0.0,
            )
        elif environment is not None:
            materials["environment_color_intensity"][index] = (
                *environment.color, environment.intensity,
            )
            materials["environment_rotation_log_range"][index] = (
                environment.rotation, environment_log_range, 1.0, 0.0,
            )
        draws["model"][index] = mesh.transform.matrix
        normal = np.eye(4, dtype=np.float32)
        normal[:3, :3] = np.linalg.inv(mesh.transform.matrix[:3, :3]).T
        draws["normal"][index] = normal
        draws["indices"][index] = (
            index, int(mesh.id or 0), 0, 0,
        )

    lights = np.zeros(len(scene.lights), LIGHT_DTYPE)
    for index, light in enumerate(scene.lights):
        if isinstance(light, DirectionalLight):
            lights["position_type"][index, 3] = 1.0
            lights["direction_range"][index] = (*light.direction, 0.0)
        else:
            lights["position_type"][index] = (*light.position, 0.0 if isinstance(light, PointLight) else 2.0)
            lights["direction_range"][index, 3] = (
                float(light.range) if light.range is not None else -1.0
            )
            if isinstance(light, SpotLight):
                lights["direction_range"][index, :3] = light.direction
                lights["spot"][index, :2] = (
                    light.inner_cone_angle, light.outer_cone_angle,
                )
        lights["color_intensity"][index] = (*light.color, light.intensity)
    from .shadows import plan_shadow_maps
    return RasterGpuScene(
        camera_data, materials, lights, draws, textures,
        plan_shadow_maps(scene), programs,
    )


__all__ = [
    "CAMERA_DTYPE", "DRAW_DTYPE", "LIGHT_DTYPE", "MATERIAL_DTYPE",
    "RasterGpuScene", "pack_raster_gpu_scene",
]
