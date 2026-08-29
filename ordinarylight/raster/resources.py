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
    ("texture_indices", np.int32, (4,)),
    ("texture_indices_extra", np.int32, (4,)),
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
        ):
            if texture is not None and id(texture) not in lookup:
                lookup[id(texture)] = len(textures)
                textures.append(texture)
    return tuple(textures), lookup


def pack_raster_gpu_scene(scene, camera, width, height, *, exposure=1.0):
    """Pack one scene revision into the shared Vulkan/WebGPU raster ABI."""
    from ._core import camera_matrix

    camera_data = np.zeros(1, CAMERA_DTYPE)
    camera_data["view_projection"][0] = camera_matrix(camera, width, height)
    camera_data["position_exposure"][0] = (*camera.position, float(exposure))
    textures, texture_lookup = _texture_table(scene)
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
        materials["texture_indices_extra"][index] = (
            texture_values[4], texture_values[5], 0, 0,
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
        plan_shadow_maps(scene),
    )


__all__ = [
    "CAMERA_DTYPE", "DRAW_DTYPE", "LIGHT_DTYPE", "MATERIAL_DTYPE",
    "RasterGpuScene", "pack_raster_gpu_scene",
]
