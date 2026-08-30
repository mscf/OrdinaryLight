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
    position: osh.invariant(osh.builtin(osh.vec4, "position"))
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
    object_id: osh.location(osh.f32, 23)


@osh.structure
class ShadowVertexOutput:
    position: osh.invariant(osh.builtin(osh.vec4, "position"))
    clip_depth: osh.location(osh.vec2, 0)


@osh.structure
class RasterCamera:
    view_projection: osh.mat4
    position_exposure: osh.vec4
    viewport_optics: osh.vec4
    optical_diagnostic: osh.vec4


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
    object_id: osh.location(osh.f32, 12),
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
        clearcoat_uv, sheen_uv, anisotropy_uv, subsurface_uv, object_id,
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
    object_id: osh.location(osh.f32, 23),
    base_color_atlas: osh.sampled_texture_2d(binding=0),
    base_color_sampler: osh.sampler(binding=1),
    shadow_map: osh.sampled_depth_texture_2d(binding=2),
    shadow_sampler: osh.comparison_sampler(binding=4),
    materials: osh.storage_buffer(RasterMaterial, access="read", binding=5),
    scene_color: osh.sampled_texture_2d(binding=6),
    scene_depth: osh.sampled_depth_texture_2d(binding=7),
    scene_sampler: osh.sampler(binding=8),
    scene_depth_sampler: osh.sampler(binding=9),
    camera: osh.uniform_buffer(RasterCamera, binding=3),
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
    object_tag = osh.floor(object_id + 0.5) + 1.0
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
    # Four raster-only prefiltered environment levels are packed side by side.
    # Interpolate adjacent levels using perceptual roughness, analogous to a
    # cubemap mip/LOD lookup without requiring backend-specific texture LODs.
    # The stored blur radii grow nonlinearly (0, 2, 8, 24 texels), so map the
    # perceptual roughness somewhat faster than a linear four-level index.
    environment_level = osh.minimum(surface_roughness * 5.0, 3.0)
    environment_level_low = osh.floor(environment_level)
    environment_level_high = osh.minimum(environment_level_low + 1.0, 3.0)
    environment_level_mix = osh.fraction(environment_level)
    reflection_uv_low = osh.vec2(
        (reflection_uv.x + environment_level_low) * 0.25,
        reflection_uv.y,
    )
    reflection_uv_high = osh.vec2(
        (reflection_uv.x + environment_level_high) * 0.25,
        reflection_uv.y,
    )
    refraction_uv_low = osh.vec2(
        (refraction_uv.x + environment_level_low) * 0.25,
        refraction_uv.y,
    )
    refraction_uv_high = osh.vec2(
        (refraction_uv.x + environment_level_high) * 0.25,
        refraction_uv.y,
    )
    reflected_encoded_low = base_color_atlas.sample_with(
        base_color_sampler,
        environment_rect.xy + reflection_uv_low * environment_rect.zw,
    ).xyz
    reflected_encoded_high = base_color_atlas.sample_with(
        base_color_sampler,
        environment_rect.xy + reflection_uv_high * environment_rect.zw,
    ).xyz
    refracted_encoded_low = base_color_atlas.sample_with(
        base_color_sampler,
        environment_rect.xy + refraction_uv_low * environment_rect.zw,
    ).xyz
    refracted_encoded_high = base_color_atlas.sample_with(
        base_color_sampler,
        environment_rect.xy + refraction_uv_high * environment_rect.zw,
    ).xyz
    reflected_encoded = osh.mix(
        reflected_encoded_low, reflected_encoded_high,
        environment_level_mix,
    )
    refracted_encoded = osh.mix(
        refracted_encoded_low, refracted_encoded_high,
        environment_level_mix,
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
    screen_clip = camera.view_projection * osh.vec4(world_position, 1.0)
    screen_ndc = screen_clip.xyz / osh.maximum(
        osh.absolute(screen_clip.w), 0.000001,
    )
    screen_uv = osh.vec2(
        screen_ndc.x * 0.5 + 0.5,
        (screen_ndc.y * 0.5 + 0.5) if camera.viewport_optics.z < 0.0
        else (0.5 - screen_ndc.y * 0.5),
    )
    screen_pass_enabled = (
        1.0 if (
            osh.absolute(camera.viewport_optics.z) > 0.5
            and osh.absolute(camera.viewport_optics.z) < 1.5
        ) else 0.0
    )
    screen_enabled = screen_pass_enabled * osh.maximum(
        surface_transmission, surface_metallic,
    )
    # March the world-space reflection ray through projected screen space and
    # accept only crossings of the completed opaque depth buffer.  This makes
    # neighboring on-screen geometry participate in reflections while misses,
    # off-screen rays, and geometry hidden from the camera retain the portable
    # environment/probe result.
    quality_distance = osh.minimum(camera.viewport_optics.w / 24.0, 2.0)
    refraction_travel = (
        0.025 + optical_thickness * 0.018
    ) * quality_distance
    reflection_screen_uv = screen_uv
    reflection_hit = 0.0
    reflection_origin = world_position + surface_normal * 0.06
    reflection_extent = osh.maximum(8.0, osh.length(view_delta) * 3.0)
    previous_ray_fraction = 0.0
    previous_depth_delta = -1.0
    diagnostic_depth_delta = -1.0
    diagnostic_ray_step = 0.0
    diagnostic_confidence = 0.0
    diagnostic_depth_trace = 0.0
    # A fixed-bound native loop is preferable here. Explicit expansion is
    # available through Ordinary Shade's ``unroll_range``, but duplicating the
    # texture-heavy refinement body increases driver code-generation variance
    # on some Vulkan implementations without improving the march.
    for ray_step in range(1, 25):
        ray_fraction = osh.f32(ray_step) / 24.0
        ray_distance = 0.12 + ray_fraction * ray_fraction * reflection_extent
        ray_world = reflection_origin + reflected * ray_distance
        ray_clip = camera.view_projection * osh.vec4(ray_world, 1.0)
        if ray_clip.w <= 0.000001:
            break
        ray_ndc = ray_clip.xyz / ray_clip.w
        ray_uv = osh.vec2(
            ray_ndc.x * 0.5 + 0.5,
            (ray_ndc.y * 0.5 + 0.5) if camera.viewport_optics.z < 0.0
            else (0.5 - ray_ndc.y * 0.5),
        )
        if (
            ray_uv.x <= 0.001 or ray_uv.x >= 0.999
            or ray_uv.y <= 0.001 or ray_uv.y >= 0.999
        ):
            break
        sampled_depth = scene_depth.sample_depth_level_with(
            scene_depth_sampler, ray_uv, 0,
        )
        diagnostic_depth_trace = (
            diagnostic_depth_trace + sampled_depth * ray_fraction
        )
        depth_delta = ray_ndc.z - sampled_depth
        diagnostic_depth_delta = depth_delta
        diagnostic_ray_step = ray_fraction
        # A valid hit crosses from in front of opaque depth to just behind it.
        # Requiring the crossing (rather than accepting any nearby sample)
        # rejects the reflector's own grazing silhouette.  NDC depth is highly
        # compressed here, so the former milliscale band admitted geometry
        # several world units away and visibly shimmered during motion.
        depth_thickness = 0.00008 + ray_fraction * 0.00042
        if (
            previous_depth_delta <= 0.00001
            and depth_delta > 0.00001
        ):
            # The coarse march is deliberately cheap, but choosing its first
            # sample directly makes the reflected texel jump by a whole step
            # as the camera moves. Refine the depth crossing before sampling
            # scene color so adjacent frames converge on the same surface.
            lower_fraction = previous_ray_fraction
            upper_fraction = ray_fraction
            refined_uv = ray_uv
            refined_delta = depth_delta
            for refine_step in range(4):
                middle_fraction = (lower_fraction + upper_fraction) * 0.5
                middle_distance = (
                    0.12
                    + middle_fraction * middle_fraction * reflection_extent
                )
                middle_world = reflection_origin + reflected * middle_distance
                middle_clip = (
                    camera.view_projection * osh.vec4(middle_world, 1.0)
                )
                middle_ndc = middle_clip.xyz / osh.maximum(
                    middle_clip.w, 0.000001,
                )
                middle_uv = osh.vec2(
                    middle_ndc.x * 0.5 + 0.5,
                    (middle_ndc.y * 0.5 + 0.5)
                    if camera.viewport_optics.z < 0.0
                    else (0.5 - middle_ndc.y * 0.5),
                )
                middle_depth = scene_depth.sample_depth_level_with(
                    scene_depth_sampler, middle_uv, 0,
                )
                middle_delta = middle_ndc.z - middle_depth
                if middle_delta > 0.00001:
                    upper_fraction = middle_fraction
                    refined_uv = middle_uv
                    refined_delta = middle_delta
                else:
                    lower_fraction = middle_fraction
            refined_thickness = 0.00006 + upper_fraction * 0.00030
            edge_distance = osh.minimum(
                osh.minimum(refined_uv.x, 1.0 - refined_uv.x),
                osh.minimum(refined_uv.y, 1.0 - refined_uv.y),
            )
            edge_confidence = osh.clamp(
                (edge_distance - 0.002) / 0.018, 0.0, 1.0,
            )
            if refined_delta < refined_thickness:
                candidate = scene_color.sample_level_with(
                    scene_sampler, refined_uv, 0.0,
                )
                different_object = (
                    1.0 if osh.absolute(candidate.w - object_tag) > 0.25
                    else 0.0
                )
                # A hit adjacent to a depth discontinuity is not stable under
                # sub-pixel motion: at one resolution it sees foreground and
                # at another it sees background. Measure a one-pixel cross in
                # the opaque depth buffer and smoothly prefer the environment
                # fallback instead of letting that binary choice shimmer.
                hit_texel = osh.vec2(
                    1.0 / osh.maximum(camera.viewport_optics.x, 1.0),
                    1.0 / osh.maximum(camera.viewport_optics.y, 1.0),
                )
                hit_depth = scene_depth.sample_depth_level_with(
                    scene_depth_sampler, refined_uv, 0,
                )
                depth_left = scene_depth.sample_depth_level_with(
                    scene_depth_sampler,
                    refined_uv - osh.vec2(hit_texel.x, 0.0),
                    0,
                )
                depth_right = scene_depth.sample_depth_level_with(
                    scene_depth_sampler,
                    refined_uv + osh.vec2(hit_texel.x, 0.0),
                    0,
                )
                depth_down = scene_depth.sample_depth_level_with(
                    scene_depth_sampler,
                    refined_uv - osh.vec2(0.0, hit_texel.y),
                    0,
                )
                depth_up = scene_depth.sample_depth_level_with(
                    scene_depth_sampler,
                    refined_uv + osh.vec2(0.0, hit_texel.y),
                    0,
                )
                depth_spread = osh.maximum(
                    osh.maximum(
                        osh.absolute(depth_left - hit_depth),
                        osh.absolute(depth_right - hit_depth),
                    ),
                    osh.maximum(
                        osh.absolute(depth_down - hit_depth),
                        osh.absolute(depth_up - hit_depth),
                    ),
                )
                depth_confidence = osh.clamp(
                    1.0 - depth_spread / 0.00075, 0.0, 1.0,
                )
                reflection_screen_uv = refined_uv
                reflection_hit = (
                    edge_confidence * different_object * depth_confidence
                )
                diagnostic_confidence = edge_confidence * depth_confidence
            if camera.optical_diagnostic.x < 5.5:
                break
        previous_ray_fraction = ray_fraction
        previous_depth_delta = depth_delta
    refraction_screen_uv = screen_uv + refracted.xy * refraction_travel
    refraction_valid = (
        refraction_screen_uv.x > 0.001 and refraction_screen_uv.x < 0.999
        and refraction_screen_uv.y > 0.001 and refraction_screen_uv.y < 0.999
    )
    refraction_hit = 1.0 if refraction_valid else 0.0
    # Rough SSR must filter in screen space. A single hit texel aliases scene
    # detail and produces resolution-dependent blocks that flicker under tiny
    # camera changes. This compact tent kernel has a radius measured in output
    # pixels, so 1080p and 4K converge toward the same appearance.
    reflection_texel = osh.vec2(
        1.0 / osh.maximum(camera.viewport_optics.x, 1.0),
        1.0 / osh.maximum(camera.viewport_optics.y, 1.0),
    )
    reflection_radius = 1.0 + surface_roughness * surface_roughness * 10.0
    reflection_offset = reflection_texel * reflection_radius
    screen_reflected = (
        scene_color.sample_level_with(
            scene_sampler, reflection_screen_uv, 0.0,
        ).xyz * 4.0
        + scene_color.sample_level_with(
            scene_sampler,
            reflection_screen_uv + osh.vec2(reflection_offset.x, 0.0),
            0.0,
        ).xyz
        + scene_color.sample_level_with(
            scene_sampler,
            reflection_screen_uv - osh.vec2(reflection_offset.x, 0.0),
            0.0,
        ).xyz
        + scene_color.sample_level_with(
            scene_sampler,
            reflection_screen_uv + osh.vec2(0.0, reflection_offset.y),
            0.0,
        ).xyz
        + scene_color.sample_level_with(
            scene_sampler,
            reflection_screen_uv - osh.vec2(0.0, reflection_offset.y),
            0.0,
        ).xyz
    ) * 0.125
    screen_refracted = scene_color.sample_level_with(
        scene_sampler, refraction_screen_uv, 0.0,
    ).xyz
    reflected_source = osh.mix(
        reflected_environment, screen_reflected,
        reflection_hit * screen_enabled * (1.0 - surface_roughness),
    )
    refracted_source = osh.mix(
        refracted_environment, screen_refracted,
        refraction_hit * screen_enabled * (1.0 - surface_roughness),
    )
    environment_enabled = environment_rotation_log_range.z
    ambient = (
        surface_base_color * diffuse_weight
        + f0 * (1.0 - 0.5 * surface_roughness)
        + transmission_tint * surface_transmission * (osh.vec3(1.0) - f0)
    ) * light_color_ambient.w * surface_occlusion
    base_shaded = ambient + direct + surface_emission
    transmitted_shaded = osh.mix(
        base_shaded,
        refracted_source * transmission_tint,
        surface_transmission * environment_enabled,
    )
    shaded = (
        transmitted_shaded
        + reflected_source * fresnel * environment_enabled
    )
    prepass_alpha = (
        object_tag if osh.absolute(camera.viewport_optics.z) > 1.5
        else surface_alpha
    )
    result = osh.vec4(
        surface_base_color if unlit_program else shaded, prepass_alpha,
    )
    diagnostic_mode = camera.optical_diagnostic.x
    if diagnostic_mode > 0.5 and diagnostic_mode < 1.5:
        return osh.vec4(osh.vec3(reflection_hit), 1.0)
    if diagnostic_mode > 1.5 and diagnostic_mode < 2.5:
        return osh.vec4(reflection_screen_uv, diagnostic_ray_step, 1.0)
    if diagnostic_mode > 2.5 and diagnostic_mode < 3.5:
        return osh.vec4(
            diagnostic_depth_delta,
            osh.absolute(diagnostic_depth_delta),
            diagnostic_ray_step,
            1.0,
        )
    if diagnostic_mode > 3.5 and diagnostic_mode < 4.5:
        return osh.vec4(
            diagnostic_confidence, reflection_hit, diagnostic_ray_step, 1.0,
        )
    if diagnostic_mode > 4.5 and diagnostic_mode < 5.5:
        return osh.vec4(object_tag, material_index, 0.0, 1.0)
    if diagnostic_mode > 5.5:
        return osh.vec4(
            diagnostic_depth_trace,
            osh.fraction(diagnostic_depth_trace),
            diagnostic_ray_step,
            1.0,
        )
    return result


__all__ = [
    "RasterCamera", "RasterMaterial", "RasterMaterialContext", "RasterSurface",
    "SceneVertexOutput", "ShadowVertexOutput", "blend_raster_surfaces",
    "default_raster_material_hook", "scene_fragment",
    "scene_vertex", "shadow_fragment", "shadow_vertex",
]
