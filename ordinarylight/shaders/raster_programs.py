"""Built-in per-fragment raster shaders authored with Ordinary Shade."""

import ordinaryshade as osh

from ..materials.gpu import (
    SurfaceContext, SurfaceParameters, blend_surface_parameters,
    default_material_modifier,
)

# Compatibility names for the first raster-only material-hook baseline.
RasterMaterialContext = SurfaceContext
RasterSurface = SurfaceParameters
blend_raster_surfaces = blend_surface_parameters
default_raster_material_hook = default_material_modifier


@osh.structure
class SceneVertexOutput:
    position: osh.builtin(osh.vec4, "position")
    base_color: osh.location(osh.vec3, 0)
    world_normal: osh.location(osh.vec3, 1)
    world_position: osh.location(osh.vec3, 2)
    material: osh.location(osh.vec4, 3)
    emission: osh.location(osh.vec3, 4)
    camera_position: osh.location(osh.vec3, 5)
    light_position_type: osh.location(osh.vec4, 6)
    light_color_ambient: osh.location(osh.vec4, 7)
    base_color_uv: osh.location(osh.vec2, 8)
    shadow_coordinate: osh.location(osh.vec4, 9)
    shadow_visibility: osh.location(osh.f32, 10)
    world_tangent: osh.location(osh.vec4, 11)
    metallic_roughness_uv: osh.location(osh.vec2, 12)
    emissive_uv: osh.location(osh.vec2, 13)
    normal_uv: osh.location(osh.vec2, 14)
    occlusion_uv: osh.location(osh.vec2, 15)
    transmission_uv: osh.location(osh.vec2, 16)
    material_index: osh.location(osh.f32, 17)
    thickness_uv: osh.location(osh.vec2, 18)
    clearcoat_uv: osh.location(osh.vec2, 19)
    sheen_uv: osh.location(osh.vec2, 20)
    anisotropy_uv: osh.location(osh.vec2, 21)
    subsurface_uv: osh.location(osh.vec2, 22)


@osh.structure
class ShadowVertexOutput:
    position: osh.builtin(osh.vec4, "position")
    clip_depth: osh.location(osh.vec2, 0)


@osh.structure
class RasterCamera:
    view_projection: osh.mat4
    position_exposure: osh.vec4


@osh.structure
class RasterMaterial:
    base_color_roughness: osh.vec4
    emission_metallic: osh.vec4
    attenuation_transmission: osh.vec4
    ior_distance_program_flags: osh.vec4
    texture_indices: osh.vec4
    normal_occlusion_transmission: osh.vec4
    advanced0: osh.vec4
    advanced1: osh.vec4
    sheen_color: osh.vec4
    subsurface_color: osh.vec4
    advanced_texture_indices: osh.vec4
    optical: osh.vec4
    environment_rect: osh.vec4
    environment_color_intensity: osh.vec4
    environment_rotation_log_range: osh.vec4


@osh.vertex
def shadow_vertex(
    position: osh.location(osh.vec4, 0),
) -> ShadowVertexOutput:
    return ShadowVertexOutput(position, osh.vec2(position.z, position.w))


@osh.fragment
def shadow_fragment(
    clip_depth: osh.location(osh.vec2, 0),
) -> osh.location(osh.vec4, 0):
    depth = clip_depth.x / osh.maximum(osh.absolute(clip_depth.y), 0.000001)
    normalized_depth = depth * 0.5 + 0.5
    encoded_depth = osh.power(
        osh.maximum(1.0 - normalized_depth, 0.0), 0.25,
    )
    return osh.vec4(1.0, 1.0, 1.0, encoded_depth)


@osh.vertex
def scene_vertex(
    position: osh.location(osh.vec4, 0),
    base_color: osh.location(osh.vec3, 1),
    world_normal: osh.location(osh.vec3, 2),
    world_position: osh.location(osh.vec3, 3),
    material: osh.location(osh.vec4, 4),
    emission: osh.location(osh.vec3, 5),
    camera_position: osh.location(osh.vec3, 6),
    light_position_type: osh.location(osh.vec4, 7),
    light_color_ambient: osh.location(osh.vec4, 8),
    base_color_uv: osh.location(osh.vec2, 9),
    shadow_coordinate: osh.location(osh.vec4, 10),
    shadow_visibility: osh.location(osh.f32, 11),
    world_tangent: osh.location(osh.vec4, 13),
    metallic_roughness_uv: osh.location(osh.vec2, 14),
    emissive_uv: osh.location(osh.vec2, 15),
    normal_uv: osh.location(osh.vec2, 16),
    occlusion_uv: osh.location(osh.vec2, 17),
    transmission_uv: osh.location(osh.vec2, 18),
    material_index: osh.location(osh.f32, 19),
    thickness_uv: osh.location(osh.vec2, 20),
    clearcoat_uv: osh.location(osh.vec2, 21),
    sheen_uv: osh.location(osh.vec2, 22),
    anisotropy_uv: osh.location(osh.vec2, 23),
    subsurface_uv: osh.location(osh.vec2, 24),
    camera: osh.uniform_buffer(RasterCamera, binding=3),
) -> SceneVertexOutput:
    return SceneVertexOutput(
        camera.view_projection * position,
        base_color, world_normal, world_position, material,
        emission, camera.position_exposure.xyz,
        light_position_type, light_color_ambient,
        base_color_uv, shadow_coordinate, shadow_visibility,
        world_tangent, metallic_roughness_uv, emissive_uv, normal_uv,
        occlusion_uv, transmission_uv, material_index, thickness_uv,
        clearcoat_uv, sheen_uv, anisotropy_uv, subsurface_uv,
    )


@osh.fragment
def scene_fragment(
    base_color: osh.location(osh.vec3, 0),
    world_normal: osh.location(osh.vec3, 1),
    world_position: osh.location(osh.vec3, 2),
    material: osh.location(osh.vec4, 3),
    emission: osh.location(osh.vec3, 4),
    camera_position: osh.location(osh.vec3, 5),
    light_position_type: osh.location(osh.vec4, 6),
    light_color_ambient: osh.location(osh.vec4, 7),
    base_color_uv: osh.location(osh.vec2, 8),
    shadow_coordinate: osh.location(osh.vec4, 9),
    shadow_visibility: osh.location(osh.f32, 10),
    world_tangent: osh.location(osh.vec4, 11),
    metallic_roughness_uv: osh.location(osh.vec2, 12),
    emissive_uv: osh.location(osh.vec2, 13),
    normal_uv: osh.location(osh.vec2, 14),
    occlusion_uv: osh.location(osh.vec2, 15),
    transmission_uv: osh.location(osh.vec2, 16),
    material_index: osh.location(osh.f32, 17),
    thickness_uv: osh.location(osh.vec2, 18),
    clearcoat_uv: osh.location(osh.vec2, 19),
    sheen_uv: osh.location(osh.vec2, 20),
    anisotropy_uv: osh.location(osh.vec2, 21),
    subsurface_uv: osh.location(osh.vec2, 22),
    base_color_atlas: osh.sampled_texture_2d(binding=0),
    base_color_sampler: osh.sampler(binding=1),
    shadow_map: osh.sampled_depth_texture_2d(binding=2),
    shadow_sampler: osh.comparison_sampler(binding=4),
    materials: osh.storage_buffer(RasterMaterial, access="read", binding=5),
) -> osh.location(osh.vec4, 0):
    if material_index < -0.5:
        return osh.vec4(base_color + emission, material.w)
    # Material identity is constant across a draw, but portable graphics
    # interfaces carry it through a floating-point varying.  Perspective
    # interpolation can reconstruct an authored integer as the next smaller
    # representable float (for example 0.99999994).  Truncating that value
    # produces sporadic reads from the preceding material record.  Decode to
    # the nearest integer so those harmless interpolation errors cannot change
    # material identity.
    material_id = osh.u32(material_index + 0.5)
    base_color_roughness = materials[material_id].base_color_roughness
    emission_metallic = materials[material_id].emission_metallic
    attenuation_transmission = materials[material_id].attenuation_transmission
    ior_distance_program_flags = materials[material_id].ior_distance_program_flags
    normal_occlusion_transmission = materials[material_id].normal_occlusion_transmission
    advanced0 = materials[material_id].advanced0
    advanced1 = materials[material_id].advanced1
    advanced_sheen_color = materials[material_id].sheen_color.xyz
    advanced_subsurface_color = materials[material_id].subsurface_color.xyz
    advanced_texture_indices = materials[material_id].advanced_texture_indices
    optical = materials[material_id].optical
    environment_rect = materials[material_id].environment_rect
    environment_color_intensity = materials[material_id].environment_color_intensity
    environment_rotation_log_range = materials[material_id].environment_rotation_log_range
    base_color_sample = base_color_atlas.sample_with(
        base_color_sampler, base_color_uv,
    )
    sampled_base_color = base_color_roughness.xyz * base_color_sample.xyz
    metallic_roughness_sample = base_color_atlas.sample_with(
        base_color_sampler, metallic_roughness_uv,
    )
    sampled_emission = emission_metallic.xyz * base_color_atlas.sample_with(
        base_color_sampler, emissive_uv,
    ).xyz
    normal_sample = base_color_atlas.sample_with(
        base_color_sampler, normal_uv,
    ).xyz * 2.0 - osh.vec3(1.0)
    occlusion_sample = base_color_atlas.sample_with(
        base_color_sampler, occlusion_uv,
    ).x
    transmission_sample = base_color_atlas.sample_with(
        base_color_sampler, transmission_uv,
    ).x
    thickness_sample = base_color_atlas.sample_with(
        base_color_sampler, thickness_uv,
    ).x if optical.x >= 0.0 else 1.0
    clearcoat_sample = base_color_atlas.sample_with(
        base_color_sampler, clearcoat_uv,
    ).x if advanced_texture_indices.x >= 0.0 else 1.0
    sheen_sample = base_color_atlas.sample_with(
        base_color_sampler, sheen_uv,
    ).xyz if advanced_texture_indices.y >= 0.0 else osh.vec3(1.0)
    anisotropy_sample = base_color_atlas.sample_with(
        base_color_sampler, anisotropy_uv,
    ).x if advanced_texture_indices.z >= 0.0 else 1.0
    subsurface_sample = base_color_atlas.sample_with(
        base_color_sampler, subsurface_uv,
    ).x if advanced_texture_indices.w >= 0.0 else 1.0
    shadow_w = osh.maximum(osh.absolute(shadow_coordinate.w), 0.000001)
    projected_shadow = shadow_coordinate.xyz / shadow_w
    geometric_normal = world_normal / osh.maximum(osh.length(world_normal), 0.000001)
    tangent = world_tangent.xyz / osh.maximum(
        osh.length(world_tangent.xyz), 0.000001,
    )
    bitangent = osh.cross(geometric_normal, tangent) * world_tangent.w
    scaled_normal_sample = osh.vec3(
        normal_sample.x * normal_occlusion_transmission.x,
        normal_sample.y * normal_occlusion_transmission.x,
        normal_sample.z,
    )
    mapped_normal = (
        tangent * scaled_normal_sample.x
        + bitangent * scaled_normal_sample.y
        + geometric_normal * scaled_normal_sample.z
    )
    normal = mapped_normal / osh.maximum(osh.length(mapped_normal), 0.000001)
    view_delta = camera_position - world_position
    view = view_delta / osh.maximum(osh.length(view_delta), 0.000001)
    program_kind = osh.floor(ior_distance_program_flags.w)
    mirror_program = program_kind > 1.5 and program_kind < 2.5
    glass_program = program_kind > 2.5 and program_kind < 3.5
    unlit_program = program_kind > 3.5
    metallic = (
        1.0 if mirror_program else
        emission_metallic.w * metallic_roughness_sample.z
    )
    base_roughness = osh.maximum(
        base_color_roughness.w * metallic_roughness_sample.y,
        0.04,
    )
    roughness = 0.04 if mirror_program or glass_program else base_roughness
    base_transmission = attenuation_transmission.w * transmission_sample
    transmission = 1.0 if glass_program else base_transmission
    occlusion = osh.mix(
        1.0, occlusion_sample,
        normal_occlusion_transmission.z,
    )
    hooked = ordinarylight_material_modifier(
        SurfaceParameters(
            sampled_base_color, sampled_emission, normal, metallic,
            roughness, transmission, occlusion,
            advanced0.x * clearcoat_sample, advanced0.y,
            advanced_sheen_color * sheen_sample, advanced0.z,
            advanced0.w * anisotropy_sample, advanced1.z,
            advanced1.x * subsurface_sample, advanced_subsurface_color,
            advanced1.y,
        ),
        SurfaceContext(
            base_color_uv, normal, view,
            ior_distance_program_flags.z,
        ),
    )
    surface_base_color = hooked.base_color
    surface_emission = hooked.emission
    surface_normal = hooked.normal / osh.maximum(
        osh.length(hooked.normal), 0.000001,
    )
    surface_metallic = osh.maximum(
        0.0, osh.minimum(1.0, hooked.metallic),
    )
    surface_roughness = osh.maximum(
        0.04, osh.minimum(1.0, hooked.roughness),
    )
    surface_transmission = osh.maximum(
        0.0, osh.minimum(1.0, hooked.transmission),
    )
    surface_occlusion = osh.maximum(
        0.0, osh.minimum(1.0, hooked.occlusion),
    )
    surface_clearcoat = osh.maximum(
        0.0, osh.minimum(1.0, hooked.clearcoat),
    )
    surface_clearcoat_roughness = osh.maximum(
        0.04, osh.minimum(1.0, hooked.clearcoat_roughness),
    )
    surface_sheen = hooked.sheen_color * (
        1.0 - osh.maximum(0.0, osh.minimum(1.0, hooked.sheen_roughness))
    )
    surface_anisotropy = osh.maximum(
        -1.0, osh.minimum(1.0, hooked.anisotropy),
    )
    surface_thin_walled = osh.maximum(
        0.0, osh.minimum(1.0, hooked.thin_walled),
    )
    surface_subsurface = osh.maximum(
        0.0, osh.minimum(1.0, hooked.subsurface),
    )
    surface_subsurface_radius = osh.maximum(
        0.0, osh.minimum(1.0, hooked.subsurface_radius),
    )
    light_delta = light_position_type.xyz - world_position
    point_distance = osh.maximum(osh.length(light_delta), 0.000001)
    point_incoming = light_delta / point_distance
    direction = -light_position_type.xyz
    directional_incoming = direction / osh.maximum(osh.length(direction), 0.000001)
    incoming = (
        directional_incoming if light_position_type.w > 0.5 else point_incoming
    )
    distance = 1.0 if light_position_type.w > 0.5 else point_distance
    half_delta = incoming + view
    half_vector = half_delta / osh.maximum(osh.length(half_delta), 0.000001)
    ndotl = osh.maximum(
        surface_normal.x * incoming.x + surface_normal.y * incoming.y
        + surface_normal.z * incoming.z,
        0.0,
    )
    receiver_bias = osh.maximum(0.00002, (1.0 - ndotl) * 0.0001)
    pcf_visibility = shadow_map.sample_compare_with(
        shadow_sampler, projected_shadow.xy,
        projected_shadow.z - receiver_bias,
    )
    shadow_map_visibility = (
        1.0 if osh.absolute(shadow_coordinate.w) < 0.000001 else
        pcf_visibility
    )
    ndotv = osh.maximum(
        surface_normal.x * view.x + surface_normal.y * view.y
        + surface_normal.z * view.z, 0.0,
    )
    ndoth = osh.maximum(
        surface_normal.x * half_vector.x + surface_normal.y * half_vector.y
        + surface_normal.z * half_vector.z,
        0.0,
    )
    vdoth = osh.maximum(
        view.x * half_vector.x + view.y * half_vector.y
        + view.z * half_vector.z,
        0.0,
    )
    raw_surface_alpha = material.w * base_color_sample.w
    masked_surface_alpha = 1.0 if raw_surface_alpha >= optical.z else 0.0
    surface_alpha = (
        masked_surface_alpha
        if optical.w > 0.5 and optical.w < 1.5 else raw_surface_alpha
    )
    f0 = osh.mix(
        osh.vec3(0.04), surface_base_color, osh.vec3(surface_metallic),
    )
    fresnel = f0 + (osh.vec3(1.0) - f0) * osh.power(
        osh.vec3(1.0 - vdoth), osh.vec3(5.0),
    )
    alpha = surface_roughness * surface_roughness
    alpha_x = osh.maximum(0.02, alpha * (1.0 - 0.7 * surface_anisotropy))
    alpha_y = osh.maximum(0.02, alpha * (1.0 + 0.7 * surface_anisotropy))
    tangent_dot_half = (
        tangent.x * half_vector.x + tangent.y * half_vector.y
        + tangent.z * half_vector.z
    )
    bitangent_dot_half = (
        bitangent.x * half_vector.x + bitangent.y * half_vector.y
        + bitangent.z * half_vector.z
    )
    anisotropic_denominator = (
        tangent_dot_half * tangent_dot_half / (alpha_x * alpha_x)
        + bitangent_dot_half * bitangent_dot_half / (alpha_y * alpha_y)
        + ndoth * ndoth
    )
    distribution = 1.0 / osh.maximum(
        3.14159265 * alpha_x * alpha_y
        * anisotropic_denominator * anisotropic_denominator,
        0.000001,
    )
    k = (surface_roughness + 1.0) * (surface_roughness + 1.0) / 8.0
    geometry_v = ndotv / osh.maximum(ndotv * (1.0 - k) + k, 0.000001)
    geometry_l = ndotl / osh.maximum(ndotl * (1.0 - k) + k, 0.000001)
    specular = distribution * geometry_v * geometry_l * fresnel / osh.maximum(
        4.0 * ndotv * ndotl, 0.000001,
    )
    coat_alpha = surface_clearcoat_roughness * surface_clearcoat_roughness
    coat_alpha2 = coat_alpha * coat_alpha
    coat_denominator = ndoth * ndoth * (coat_alpha2 - 1.0) + 1.0
    coat_distribution = coat_alpha2 / osh.maximum(
        3.14159265 * coat_denominator * coat_denominator, 0.000001,
    )
    coat_fresnel = 0.04 + 0.96 * osh.power(1.0 - vdoth, 5.0)
    coat_k = (
        (surface_clearcoat_roughness + 1.0)
        * (surface_clearcoat_roughness + 1.0) / 8.0
    )
    coat_geometry_v = ndotv / osh.maximum(
        ndotv * (1.0 - coat_k) + coat_k, 0.000001,
    )
    coat_geometry_l = ndotl / osh.maximum(
        ndotl * (1.0 - coat_k) + coat_k, 0.000001,
    )
    clearcoat_specular = osh.vec3(
        surface_clearcoat * coat_distribution * coat_geometry_v
        * coat_geometry_l * coat_fresnel
        / osh.maximum(4.0 * ndotv * ndotl, 0.000001)
    )
    base_energy = 1.0 - surface_clearcoat * coat_fresnel
    diffuse_weight = (1.0 - surface_metallic) * (1.0 - surface_transmission)
    diffuse_base = (
        (osh.vec3(1.0) - fresnel) * surface_base_color
        * diffuse_weight / 3.14159265
    )
    diffuse = osh.mix(
        diffuse_base, diffuse_base * hooked.subsurface_color,
        surface_subsurface,
    )
    wrapped_ndotl = osh.maximum(
        0.0, (ndotl + surface_subsurface_radius)
        / (1.0 + surface_subsurface_radius),
    )
    diffuse_ndotl = osh.mix(ndotl, wrapped_ndotl, surface_subsurface)
    attenuation = 1.0 / (distance * distance)
    visibility = attenuation * shadow_visibility * shadow_map_visibility
    direct = (
        (diffuse + surface_sheen) * base_energy * light_color_ambient.xyz
        * (diffuse_ndotl * visibility)
        + (specular * base_energy + clearcoat_specular) * light_color_ambient.xyz
        * (ndotl * visibility)
    )
    optical_thickness = advanced1.w * thickness_sample
    absorption_exponent = optical_thickness / osh.maximum(
        ior_distance_program_flags.y, 0.000001,
    )
    transmission_tint = osh.mix(
        osh.power(
            osh.maximum(attenuation_transmission.xyz, osh.vec3(0.000001)),
            osh.vec3(absorption_exponent),
        ), surface_base_color, surface_thin_walled,
    )
    incident = -view
    reflected = incident - surface_normal * (
        2.0 * (incident.x * surface_normal.x
               + incident.y * surface_normal.y
               + incident.z * surface_normal.z)
    )
    raw_refracted = osh.refract(
        incident, surface_normal,
        1.0 / osh.maximum(ior_distance_program_flags.x, 1.0001),
    )
    refracted = reflected if (
        raw_refracted.x * raw_refracted.x
        + raw_refracted.y * raw_refracted.y
        + raw_refracted.z * raw_refracted.z < 0.0001
    ) else raw_refracted
    reflection_uv = osh.vec2(
        osh.fraction(
            osh.arctangent2(reflected.z, reflected.x) / 6.28318531
            + 0.5 + environment_rotation_log_range.x / 6.28318531
        ),
        osh.arccosine(osh.maximum(-1.0, osh.minimum(1.0, reflected.y)))
        / 3.14159265,
    )
    refraction_uv = osh.vec2(
        osh.fraction(
            osh.arctangent2(refracted.z, refracted.x) / 6.28318531
            + 0.5 + environment_rotation_log_range.x / 6.28318531
        ),
        osh.arccosine(osh.maximum(-1.0, osh.minimum(1.0, refracted.y)))
        / 3.14159265,
    )
    reflected_encoded = base_color_atlas.sample_with(
        base_color_sampler,
        environment_rect.xy + reflection_uv * environment_rect.zw,
    ).xyz
    refracted_encoded = base_color_atlas.sample_with(
        base_color_sampler,
        environment_rect.xy + refraction_uv * environment_rect.zw,
    ).xyz
    reflected_environment = (
        osh.power(
            osh.vec3(2.0),
            reflected_encoded * environment_rotation_log_range.y,
        ) - osh.vec3(1.0)
    ) * environment_color_intensity.xyz * environment_color_intensity.w
    refracted_environment = (
        osh.power(
            osh.vec3(2.0),
            refracted_encoded * environment_rotation_log_range.y,
        ) - osh.vec3(1.0)
    ) * environment_color_intensity.xyz * environment_color_intensity.w
    environment_enabled = environment_rotation_log_range.z
    ambient = (
        surface_base_color * diffuse_weight
        + f0 * (1.0 - 0.5 * surface_roughness)
        + transmission_tint * surface_transmission * (osh.vec3(1.0) - f0)
    ) * light_color_ambient.w * surface_occlusion
    base_shaded = ambient + direct + surface_emission
    transmitted_shaded = osh.mix(
        base_shaded,
        refracted_environment * transmission_tint,
        surface_transmission * environment_enabled,
    )
    shaded = (
        transmitted_shaded
        + reflected_environment * fresnel * environment_enabled
    )
    return osh.vec4(surface_base_color if unlit_program else shaded, surface_alpha)


__all__ = [
    "RasterCamera", "RasterMaterial", "RasterMaterialContext", "RasterSurface",
    "SceneVertexOutput", "ShadowVertexOutput", "blend_raster_surfaces",
    "default_raster_material_hook", "scene_fragment",
    "scene_vertex", "shadow_fragment", "shadow_vertex",
]
