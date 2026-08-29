"""Portable material and direct-light evaluation for raster renderers."""

from __future__ import annotations

import numpy as np

from ..lights import DirectionalLight, SpotLight


def _sample_texture(mesh, texture, transform):
    """Sample one material texture at mesh vertices using glTF UV semantics."""
    uv = mesh.texcoords if transform.texcoord_set == 0 else mesh.texcoords1
    cosine, sine = np.cos(transform.rotation), np.sin(transform.rotation)
    addressed = uv * np.asarray(transform.scale, np.float32)
    addressed = addressed @ np.array(
        ((cosine, sine), (-sine, cosine)), np.float32,
    )
    addressed += np.asarray(transform.offset, np.float32)
    for axis, mode in enumerate((texture.wrap_s, texture.wrap_t)):
        if mode == "repeat":
            addressed[:, axis] %= 1.0
        elif mode == "mirror":
            addressed[:, axis] = 1.0 - np.abs(
                (addressed[:, axis] % 2.0) - 1.0
            )
        else:
            addressed[:, axis] = np.clip(addressed[:, axis], 0.0, 1.0)
    height, width = texture.pixels.shape[:2]
    x = np.clip(
        np.rint(addressed[:, 0] * (width - 1)).astype(int), 0, width - 1,
    )
    y = np.clip(
        np.rint((1.0 - addressed[:, 1]) * (height - 1)).astype(int),
        0, height - 1,
    )
    return texture.pixels[y, x].astype(np.float32) / 255.0


def material_channels(mesh, textures=True):
    """Resolve glTF metallic/roughness material channels at mesh vertices."""
    material = mesh.material
    count = len(mesh.vertices)
    base_color = np.broadcast_to(
        np.asarray(material.base_color, np.float32), (count, 3),
    ).copy()
    metallic = np.full(count, material.metallic, np.float32)
    roughness = np.full(count, material.roughness, np.float32)
    emission = np.broadcast_to(
        np.asarray(material.emission, np.float32), (count, 3),
    ).copy()
    transmission = np.full(count, material.transmission, np.float32)
    occlusion = np.ones(count, np.float32)
    if not textures:
        return base_color, metallic, roughness, emission, transmission, occlusion
    if material.base_color_texture is not None:
        base_color *= _sample_texture(
            mesh, material.base_color_texture, material.base_color_transform,
        )[:, :3]
    if material.metallic_roughness_texture is not None:
        sample = _sample_texture(
            mesh, material.metallic_roughness_texture,
            material.metallic_roughness_transform,
        )
        roughness *= sample[:, 1]
        metallic *= sample[:, 2]
    if material.emissive_texture is not None:
        emission *= _sample_texture(
            mesh, material.emissive_texture, material.emissive_transform,
        )[:, :3]
    if material.transmission_texture is not None:
        transmission *= _sample_texture(
            mesh, material.transmission_texture, material.transmission_transform,
        )[:, 0]
    if material.occlusion_texture is not None:
        sampled = _sample_texture(
            mesh, material.occlusion_texture, material.occlusion_transform,
        )[:, 0]
        occlusion = 1.0 + material.occlusion_strength * (sampled - 1.0)
    return (
        base_color, np.clip(metallic, 0.0, 1.0),
        np.clip(roughness, 0.04, 1.0), emission,
        np.clip(transmission, 0.0, 1.0), np.clip(occlusion, 0.0, 1.0),
    )


def _fresnel_schlick(cosine, f0):
    return f0 + (1.0 - f0) * (1.0 - cosine[:, None]) ** 5


def evaluate_vertex_lighting(scene, mesh, camera, config, shadow_visibility):
    """Evaluate the shared Cook-Torrance direct-light approximation."""
    (
        base_color, metallic, roughness, emission, transmission, occlusion,
    ) = material_channels(mesh, config.textures)
    if not config.direct_lighting:
        return base_color + emission
    positions = mesh.world_vertices
    normals = mesh.world_normals
    view = np.asarray(camera.position, np.float32) - positions
    view /= np.maximum(np.linalg.norm(view, axis=1, keepdims=True), 1e-6)
    ndotv = np.clip(np.sum(normals * view, axis=1), 0.0, 1.0)
    f0 = 0.04 * (1.0 - metallic[:, None]) + base_color * metallic[:, None]

    environment = np.full_like(base_color, config.ambient_light)
    if scene.environment is not None:
        environment *= (
            np.asarray(scene.environment.color, np.float32)
            * float(scene.environment.intensity)
        )
        if scene.environment.image is not None:
            environment *= np.mean(scene.environment.image, axis=(0, 1))
    diffuse_weight = (1.0 - metallic) * (1.0 - transmission)
    color = base_color * environment * diffuse_weight[:, None] * occlusion[:, None]
    color += (
        environment * f0 * (1.0 - 0.5 * roughness[:, None])
        * occlusion[:, None]
    )
    color += (
        environment * np.asarray(mesh.material.attenuation_color, np.float32)
        * transmission[:, None] * (1.0 - f0)
    )

    def accumulate_light(
        incoming, attenuation, light_color, maximum_distance, *, casts_shadow=True,
    ):
        nonlocal color
        attenuation = np.asarray(attenuation, np.float32).copy()
        ndotl = np.clip(np.sum(normals * incoming, axis=1), 0.0, 1.0)
        if config.shadows and casts_shadow:
            attenuation *= shadow_visibility(
                scene, mesh, positions, incoming, maximum_distance,
            )
        half_vector = incoming + view
        half_vector /= np.maximum(
            np.linalg.norm(half_vector, axis=1, keepdims=True), 1e-6,
        )
        ndoth = np.clip(np.sum(normals * half_vector, axis=1), 0.0, 1.0)
        vdoth = np.clip(np.sum(view * half_vector, axis=1), 0.0, 1.0)
        alpha = roughness * roughness
        alpha2 = alpha * alpha
        denominator = ndoth * ndoth * (alpha2 - 1.0) + 1.0
        distribution = alpha2 / np.maximum(
            np.pi * denominator * denominator, 1e-6,
        )
        k = (roughness + 1.0) ** 2 / 8.0
        geometry_v = ndotv / np.maximum(ndotv * (1.0 - k) + k, 1e-6)
        geometry_l = ndotl / np.maximum(ndotl * (1.0 - k) + k, 1e-6)
        fresnel = _fresnel_schlick(vdoth, f0)
        specular = (
            distribution[:, None] * geometry_v[:, None] * geometry_l[:, None]
            * fresnel
            / np.maximum(4.0 * ndotv[:, None] * ndotl[:, None], 1e-6)
        )
        diffuse = (
            (1.0 - fresnel) * diffuse_weight[:, None] * base_color / np.pi
        )
        color += (
            (diffuse + specular) * np.asarray(light_color, np.float32)
            * (ndotl * attenuation)[:, None]
        )

    for light in scene.lights:
        light_color = np.asarray(light.color, np.float32) * float(light.intensity)
        if isinstance(light, DirectionalLight):
            direction = -np.asarray(light.direction, np.float32)
            direction /= np.linalg.norm(direction)
            incoming = np.broadcast_to(direction, normals.shape)
            attenuation = np.ones(len(normals), np.float32)
            maximum_distance = np.full(len(positions), np.inf, np.float32)
        else:
            delta = np.asarray(light.position, np.float32) - positions
            distance = np.linalg.norm(delta, axis=1)
            incoming = delta / np.maximum(distance[:, None], 1e-6)
            attenuation = 1.0 / np.maximum(distance * distance, 1e-4)
            maximum_distance = distance
            if light.range is not None:
                attenuation *= np.clip(1.0 - distance / light.range, 0.0, 1.0) ** 2
            if isinstance(light, SpotLight):
                axis = np.asarray(light.direction, np.float32)
                axis /= np.linalg.norm(axis)
                cosine = np.sum(-incoming * axis, axis=1)
                outer, inner = (
                    np.cos(light.outer_cone_angle),
                    np.cos(light.inner_cone_angle),
                )
                attenuation *= np.clip(
                    (cosine - outer) / max(inner - outer, 1e-5), 0.0, 1.0,
                )
        accumulate_light(
            incoming, attenuation, light_color, maximum_distance,
        )

    # Treat emissive meshes as bounded area lights. This is a deterministic
    # single-sample approximation of the same scene resource used by GI.
    for emitter in scene.visible_meshes:
        if emitter is mesh or not np.any(emitter.material.emission):
            continue
        triangles = emitter.world_vertices[emitter.indices]
        cross = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        triangle_areas = np.linalg.norm(cross, axis=1) * 0.5
        area = float(np.sum(triangle_areas))
        if area <= 1e-8:
            continue
        center = np.average(
            triangles.mean(axis=1), axis=0, weights=triangle_areas,
        )
        emitter_normal = np.sum(cross, axis=0)
        emitter_normal /= max(float(np.linalg.norm(emitter_normal)), 1e-8)
        delta = center - positions
        distance = np.linalg.norm(delta, axis=1)
        incoming = delta / np.maximum(distance[:, None], 1e-6)
        facing = np.sum(-incoming * emitter_normal, axis=1)
        facing = np.abs(facing) if emitter.material.emission_two_sided else np.maximum(facing, 0.0)
        attenuation = area * facing / np.maximum(distance * distance, 1e-4)
        emitter_color = np.asarray(emitter.material.emission, np.float32)
        if config.textures and emitter.material.emissive_texture is not None:
            emitter_color *= np.mean(
                emitter.material.emissive_texture.pixels[..., :3], axis=(0, 1),
            ) / 255.0
        # Vertex visibility would interpolate one binary result across a whole
        # triangle and create large triangular artifacts. Emissive-mesh
        # shadows remain disabled until the native per-fragment shadow pass.
        accumulate_light(
            incoming, attenuation, emitter_color, distance, casts_shadow=False,
        )
    return color + emission


__all__ = ["evaluate_vertex_lighting", "material_channels"]
