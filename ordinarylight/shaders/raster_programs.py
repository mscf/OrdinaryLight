"""Built-in per-fragment raster shaders authored with Ordinary Shade."""

import ordinaryshade as osh


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


@osh.structure
class RasterMaterialContext:
    uv: osh.vec2
    world_position: osh.vec3
    view_direction: osh.vec3
    program_id: osh.f32


@osh.structure
class RasterSurface:
    base_color: osh.vec3
    emission: osh.vec3
    normal: osh.vec3
    metallic: osh.f32
    roughness: osh.f32
    transmission: osh.f32
    occlusion: osh.f32


@osh.function
def blend_raster_surfaces(
    base: RasterSurface, layer: RasterSurface, weight: osh.f32,
) -> RasterSurface:
    amount = osh.maximum(0.0, osh.minimum(1.0, weight))
    mixed_normal = osh.normalize(osh.mix(base.normal, layer.normal, amount))
    return RasterSurface(
        osh.mix(base.base_color, layer.base_color, amount),
        osh.mix(base.emission, layer.emission, amount),
        mixed_normal,
        osh.mix(base.metallic, layer.metallic, amount),
        osh.mix(base.roughness, layer.roughness, amount),
        osh.mix(base.transmission, layer.transmission, amount),
        osh.mix(base.occlusion, layer.occlusion, amount),
    )


@osh.function(name="ordinarylight_raster_material_hook")
def default_raster_material_hook(
    surface: RasterSurface, context: RasterMaterialContext,
) -> RasterSurface:
    return surface


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
    camera: osh.uniform_buffer(RasterCamera, binding=3),
) -> SceneVertexOutput:
    return SceneVertexOutput(
        camera.view_projection * position,
        base_color, world_normal, world_position, material,
        emission, camera.position_exposure.xyz,
        light_position_type, light_color_ambient,
        base_color_uv, shadow_coordinate, shadow_visibility,
        world_tangent, metallic_roughness_uv, emissive_uv, normal_uv,
        occlusion_uv, transmission_uv, material_index,
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
    sampled_base_color = base_color_roughness.xyz * base_color_atlas.sample_with(
        base_color_sampler, base_color_uv,
    ).xyz
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
    hooked = ordinarylight_raster_material_hook(
        RasterSurface(
            sampled_base_color, sampled_emission, normal, metallic,
            roughness, transmission, occlusion,
        ),
        RasterMaterialContext(
            base_color_uv, world_position, view,
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
    surface_alpha = material.w
    f0 = osh.mix(
        osh.vec3(0.04), surface_base_color, osh.vec3(surface_metallic),
    )
    fresnel = f0 + (osh.vec3(1.0) - f0) * osh.power(
        osh.vec3(1.0 - vdoth), osh.vec3(5.0),
    )
    alpha = surface_roughness * surface_roughness
    alpha2 = alpha * alpha
    denominator = ndoth * ndoth * (alpha2 - 1.0) + 1.0
    distribution = alpha2 / osh.maximum(
        3.14159265 * denominator * denominator, 0.000001,
    )
    k = (surface_roughness + 1.0) * (surface_roughness + 1.0) / 8.0
    geometry_v = ndotv / osh.maximum(ndotv * (1.0 - k) + k, 0.000001)
    geometry_l = ndotl / osh.maximum(ndotl * (1.0 - k) + k, 0.000001)
    specular = distribution * geometry_v * geometry_l * fresnel / osh.maximum(
        4.0 * ndotv * ndotl, 0.000001,
    )
    diffuse_weight = (1.0 - surface_metallic) * (1.0 - surface_transmission)
    diffuse = (
        (osh.vec3(1.0) - fresnel) * surface_base_color
        * diffuse_weight / 3.14159265
    )
    attenuation = 1.0 / (distance * distance)
    direct = (
        (diffuse + specular) * light_color_ambient.xyz
        * (ndotl * attenuation * shadow_visibility * shadow_map_visibility)
    )
    ambient = (
        surface_base_color * diffuse_weight
        + f0 * (1.0 - 0.5 * surface_roughness)
        + osh.vec3(surface_transmission) * (osh.vec3(1.0) - f0)
    ) * light_color_ambient.w * surface_occlusion
    shaded = ambient + direct + surface_emission
    return osh.vec4(surface_base_color if unlit_program else shaded, surface_alpha)


__all__ = [
    "RasterCamera", "RasterMaterial", "RasterMaterialContext", "RasterSurface",
    "SceneVertexOutput", "ShadowVertexOutput", "blend_raster_surfaces",
    "default_raster_material_hook", "scene_fragment",
    "scene_vertex", "shadow_fragment", "shadow_vertex",
]
