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
class GeometryProductVertexOutput:
    """Varyings for the renderer-neutral native geometry-product pass."""

    position: osh.invariant(osh.builtin(osh.vec4, "position"))
    world_normal: osh.location(osh.vec3, 0)
    object_id: osh.location(osh.f32, 1)
    current_clip: osh.location(osh.vec4, 2)
    previous_clip: osh.location(osh.vec4, 3)
    world_position: osh.location(osh.vec3, 4)


@osh.structure
class GeometryProductOutput:
    """Native MRT products shared by Vulkan and WebGPU raster targets."""

    normal_depth: osh.location(osh.vec4, 0)
    motion_object: osh.location(osh.vec4, 1)


@osh.structure
class RasterCamera:
    view_projection: osh.mat4
    position_exposure: osh.vec4
    viewport_optics: osh.vec4
    optical_diagnostic: osh.vec4


@osh.structure
class GeometryProductCamera:
    current_view_projection: osh.mat4
    previous_view_projection: osh.mat4
    viewport: osh.vec4
    camera_position: osh.vec4


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
    probe_position_radius: osh.vec4
    probe_box_min_mode: osh.vec4
    probe_box_max_blend: osh.vec4
    environment_rect_secondary: osh.vec4
    environment_rotation_log_range_secondary: osh.vec4
    probe_position_radius_secondary: osh.vec4
    probe_box_min_mode_secondary: osh.vec4
    probe_box_max_blend_secondary: osh.vec4


@osh.structure
class RasterLight:
    """Portable analytic-light record shared by Vulkan and WebGPU."""

    position_type: osh.vec4
    direction_range: osh.vec4
    color_intensity: osh.vec4
    spot: osh.vec4


@osh.structure
class RasterShadow:
    view_projection: osh.mat4
    atlas: osh.vec4
    parameters: osh.vec4


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
def geometry_product_vertex(
    world_position: osh.location(osh.vec3, 0),
    world_normal: osh.location(osh.vec3, 1),
    object_id: osh.location(osh.f32, 2),
    previous_world_position: osh.location(osh.vec3, 3),
    camera: osh.uniform_buffer(GeometryProductCamera, binding=0),
) -> GeometryProductVertexOutput:
    current_clip = camera.current_view_projection * osh.vec4(
        world_position, 1.0,
    )
    previous_clip = camera.previous_view_projection * osh.vec4(
        previous_world_position, 1.0,
    )
    return GeometryProductVertexOutput(
        current_clip, world_normal, object_id, current_clip, previous_clip,
        world_position,
    )


@osh.fragment
def geometry_product_fragment(
    world_normal: osh.location(osh.vec3, 0),
    object_id: osh.location(osh.f32, 1),
    current_clip: osh.location(osh.vec4, 2),
    previous_clip: osh.location(osh.vec4, 3),
    world_position: osh.location(osh.vec3, 4),
    camera: osh.uniform_buffer(GeometryProductCamera, binding=0),
) -> GeometryProductOutput:
    current_ndc = current_clip.xy / osh.maximum(
        osh.absolute(current_clip.w), 0.000001,
    )
    previous_ndc = previous_clip.xy / osh.maximum(
        osh.absolute(previous_clip.w), 0.000001,
    )
    # Public motion vectors use image-space pixels (+x right, +y down).
    motion = (current_ndc - previous_ndc) * osh.vec2(
        camera.viewport.x * 0.5, -camera.viewport.y * 0.5,
    )
    return GeometryProductOutput(
        osh.vec4(
            osh.normalize(world_normal),
            osh.length(world_position - camera.camera_position.xyz),
        ),
        osh.vec4(motion, osh.round(object_id), 0.0),
    )


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
    lights: osh.storage_buffer(RasterLight, access="read", binding=10),
    shadows: osh.storage_buffer(RasterShadow, access="read", binding=11),
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
    optical_layer_role = materials[material_id].sheen_color.w
    advanced_subsurface_color = materials[material_id].subsurface_color.xyz
    advanced_texture_indices = materials[material_id].advanced_texture_indices
    optical = materials[material_id].optical
    environment_rect = materials[material_id].environment_rect
    environment_color_intensity = materials[material_id].environment_color_intensity
    environment_rotation_log_range = materials[material_id].environment_rotation_log_range
    probe_position_radius = materials[material_id].probe_position_radius
    probe_box_min_mode = materials[material_id].probe_box_min_mode
    probe_box_max_blend = materials[material_id].probe_box_max_blend
    environment_rect_secondary = materials[material_id].environment_rect_secondary
    environment_rotation_log_range_secondary = materials[material_id].environment_rotation_log_range_secondary
    probe_position_radius_secondary = materials[material_id].probe_position_radius_secondary
    probe_box_min_mode_secondary = materials[material_id].probe_box_min_mode_secondary
    probe_box_max_blend_secondary = materials[material_id].probe_box_max_blend_secondary
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
    surface_sheen_color = hooked.sheen_color
    surface_sheen_roughness = osh.maximum(
        0.0, osh.minimum(1.0, hooked.sheen_roughness),
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
    first_packed_light = lights[osh.u32(0)]
    first_has_packed_light = camera.optical_diagnostic.y > 0.5
    first_light_type = (
        first_packed_light.position_type.w
        if first_has_packed_light else light_position_type.w
    )
    first_light_position = (
        first_packed_light.position_type.xyz
        if first_has_packed_light else light_position_type.xyz
    )
    first_light_direction = (
        first_packed_light.direction_range.xyz
        if first_has_packed_light else light_position_type.xyz
    )
    first_light_radiance = (
        first_packed_light.color_intensity.xyz
        * first_packed_light.color_intensity.w
        if first_has_packed_light else light_color_ambient.xyz
    )
    light_delta = first_light_position - world_position
    point_distance = osh.maximum(osh.length(light_delta), 0.000001)
    point_incoming = light_delta / point_distance
    direction = -first_light_direction
    directional_incoming = direction / osh.maximum(osh.length(direction), 0.000001)
    incoming = (
        directional_incoming
        if first_light_type > 0.5 and first_light_type < 1.5
        else point_incoming
    )
    distance = (
        1.0
        if first_light_type > 0.5 and first_light_type < 1.5
        else point_distance
    )
    half_delta = incoming + view
    half_vector = half_delta / osh.maximum(osh.length(half_delta), 0.000001)
    signed_ndotl = (
        surface_normal.x * incoming.x + surface_normal.y * incoming.y
        + surface_normal.z * incoming.z
    )
    ndotl = osh.maximum(signed_ndotl, 0.0)
    receiver_bias = osh.maximum(0.00002, (1.0 - ndotl) * 0.0001)
    pcf_visibility = shadow_map.sample_compare_with(
        shadow_sampler, projected_shadow.xy,
        projected_shadow.z - receiver_bias,
    )
    shadow_map_visibility = (
        1.0 if osh.absolute(shadow_coordinate.w) < 0.000001 else
        pcf_visibility
    )
    if camera.optical_diagnostic.z > 0.5:
        shadow_map_visibility = 1.0
        for shadow_index in range(24):
            if shadow_index < osh.i32(camera.optical_diagnostic.z):
                shadow_record = shadows[osh.u32(shadow_index)]
                if osh.absolute(shadow_record.parameters.x) < 0.25:
                    shadow_face_matches = True
                    if shadow_record.parameters.w > 1.5:
                        point_shadow_delta = (
                            world_position - first_light_position
                        )
                        point_shadow_axis = osh.absolute(point_shadow_delta)
                        point_shadow_face = 0.0
                        if (
                            point_shadow_axis.y >= point_shadow_axis.x
                            and point_shadow_axis.y >= point_shadow_axis.z
                        ):
                            point_shadow_face = (
                                2.0 if point_shadow_delta.y >= 0.0 else 3.0
                            )
                        elif point_shadow_axis.z >= point_shadow_axis.x:
                            point_shadow_face = (
                                4.0 if point_shadow_delta.z >= 0.0 else 5.0
                            )
                        else:
                            point_shadow_face = (
                                0.0 if point_shadow_delta.x >= 0.0 else 1.0
                            )
                        shadow_face_matches = osh.absolute(
                            shadow_record.parameters.y - point_shadow_face
                        ) < 0.25
                    if not shadow_face_matches:
                        continue
                    biased_shadow_position = world_position + (
                        incoming * shadow_record.parameters.z
                    )
                    shadow_clip = shadow_record.view_projection * osh.vec4(
                        biased_shadow_position, 1.0,
                    )
                    shadow_clip_w = osh.maximum(
                        osh.absolute(shadow_clip.w), 0.000001,
                    )
                    shadow_ndc = shadow_clip.xyz / shadow_clip_w
                    if (
                        shadow_clip.w > 0.0
                        and osh.absolute(shadow_ndc.x) <= 1.0001
                        and osh.absolute(shadow_ndc.y) <= 1.0001
                        and shadow_ndc.z >= 0.0 and shadow_ndc.z <= 1.0
                    ):
                        shadow_uv = osh.vec2(
                            shadow_record.atlas.x
                            + shadow_record.atlas.y * shadow_ndc.x,
                            shadow_record.atlas.z
                            - shadow_record.atlas.w * shadow_ndc.y,
                        )
                        # Point-light perspective depth becomes tightly packed
                        # near one. A large fixed projected-depth bias erases
                        # valid long shadows, so rely primarily on the
                        # world-space receiver offset and retain only a small
                        # comparison epsilon here.
                        shadow_compare_bias = (
                            0.00002 if shadow_record.parameters.w > 1.5
                            else receiver_bias
                        )
                        shadow_map_visibility = shadow_map.sample_compare_with(
                            shadow_sampler, shadow_uv,
                            shadow_ndc.z - shadow_compare_bias,
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
    vdoth = osh.minimum(
        1.0, osh.maximum(
            view.x * half_vector.x + view.y * half_vector.y
            + view.z * half_vector.z,
            0.0,
        ),
    )
    sheen_weight = (
        (1.0 - surface_sheen_roughness)
        * osh.power(1.0 - vdoth, 5.0)
    )
    sheen = surface_sheen_color * sheen_weight
    raw_surface_alpha = material.w * base_color_sample.w
    masked_surface_alpha = 1.0 if raw_surface_alpha >= optical.z else 0.0
    surface_alpha = (
        masked_surface_alpha
        if optical.w > 0.5 and optical.w < 1.5 else raw_surface_alpha
    )
    dielectric_ior = osh.maximum(ior_distance_program_flags.x, 1.0001)
    dielectric_f0_ratio = (
        (dielectric_ior - 1.0) / (dielectric_ior + 1.0)
    )
    dielectric_f0 = dielectric_f0_ratio * dielectric_f0_ratio
    f0 = osh.mix(
        osh.vec3(dielectric_f0), surface_base_color,
        osh.vec3(surface_metallic),
    )
    fresnel = f0 + (osh.vec3(1.0) - f0) * osh.power(
        osh.vec3(1.0 - vdoth), osh.vec3(5.0),
    )
    # Direct-light microfacet Fresnel uses V.H above.  Environment reflection
    # versus transmission is an interface-energy split and therefore uses
    # the view incidence V.N.  Reusing the direct-light term here makes the
    # split depend on light direction and can leave a strong unrefracted lobe.
    optical_fresnel = f0 + (osh.vec3(1.0) - f0) * osh.power(
        osh.vec3(1.0 - ndotv), osh.vec3(5.0),
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
    # Apply the wrap to signed N.L. Clamping first gives every back-facing
    # fragment the same positive value and leaves the shadow map to create an
    # abrupt, triangle-shaped terminator. Signed wrapping instead rolls the
    # lobe continuously through zero, as the approximation requires.
    wrapped_ndotl = osh.maximum(
        0.0, (signed_ndotl + surface_subsurface_radius)
        / (1.0 + surface_subsurface_radius),
    )
    diffuse_ndotl = osh.mix(ndotl, wrapped_ndotl, surface_subsurface)
    attenuation = 1.0 / (distance * distance)
    if first_has_packed_light and first_packed_light.direction_range.w > 0.0:
        first_range_ratio = (
            point_distance / first_packed_light.direction_range.w
        )
        first_range_falloff = osh.maximum(1.0 - first_range_ratio, 0.0)
        attenuation = attenuation * first_range_falloff * first_range_falloff
    if first_has_packed_light and first_light_type > 1.5:
        first_spot_axis = first_light_direction / osh.maximum(
            osh.length(first_light_direction), 0.000001,
        )
        first_spot_cosine = -(
            incoming.x * first_spot_axis.x
            + incoming.y * first_spot_axis.y
            + incoming.z * first_spot_axis.z
        )
        first_spot_outer = osh.cosine(first_packed_light.spot.y)
        first_spot_inner = osh.cosine(first_packed_light.spot.x)
        attenuation = attenuation * osh.maximum(
            0.0, osh.minimum(
                1.0,
                (first_spot_cosine - first_spot_outer)
                / osh.maximum(first_spot_inner - first_spot_outer, 0.00001),
            ),
        )
    # Keep the ordinary surface response fully shadowed, but allow the added
    # wrapped component to cross the geometric/self-shadow terminator. This is
    # the portable approximation of local subsurface diffusion; treating that
    # component as opaque direct light produces a serrated shadow-map fringe.
    wrapped_contribution = osh.maximum(diffuse_ndotl - ndotl, 0.0)
    hard_shadow_visibility = shadow_map_visibility * shadow_visibility
    scattered_shadow_visibility = osh.mix(
        hard_shadow_visibility, 1.0, surface_subsurface,
    )
    diffuse_visibility = (
        ndotl * hard_shadow_visibility
        + wrapped_contribution * scattered_shadow_visibility
    )
    direct = (
        (diffuse + sheen) * base_energy * first_light_radiance
        * (diffuse_visibility * attenuation)
        + (specular * base_energy + clearcoat_specular) * first_light_radiance
        * (ndotl * hard_shadow_visibility * attenuation)
    )
    # The legacy first-light attributes remain in the vertex ABI while wheel
    # consumers migrate, but every further analytic light is evaluated from
    # the renderer-neutral GPU light array.  Shadow visibility for these
    # additional lights is introduced by the shadow-array pass; until then
    # they are deliberately unshadowed rather than silently ignored.
    for light_index in range(1, 8):
        if light_index < osh.i32(camera.optical_diagnostic.y):
            packed_light = lights[osh.u32(light_index)]
            packed_type = packed_light.position_type.w
            packed_delta = packed_light.position_type.xyz - world_position
            packed_distance = osh.maximum(osh.length(packed_delta), 0.000001)
            packed_point_incoming = packed_delta / packed_distance
            packed_direction = -packed_light.direction_range.xyz
            packed_directional_incoming = packed_direction / osh.maximum(
                osh.length(packed_direction), 0.000001,
            )
            packed_incoming = (
                packed_directional_incoming
                if packed_type > 0.5 and packed_type < 1.5
                else packed_point_incoming
            )
            packed_light_distance = (
                1.0 if packed_type > 0.5 and packed_type < 1.5
                else packed_distance
            )
            packed_attenuation = 1.0 / (
                packed_light_distance * packed_light_distance
            )
            if packed_light.direction_range.w > 0.0:
                range_ratio = packed_distance / packed_light.direction_range.w
                range_falloff = osh.maximum(1.0 - range_ratio, 0.0)
                packed_attenuation = (
                    packed_attenuation * range_falloff * range_falloff
                )
            if packed_type > 1.5:
                spot_axis = packed_light.direction_range.xyz / osh.maximum(
                    osh.length(packed_light.direction_range.xyz), 0.000001,
                )
                spot_cosine = -(
                    packed_incoming.x * spot_axis.x
                    + packed_incoming.y * spot_axis.y
                    + packed_incoming.z * spot_axis.z
                )
                spot_outer = osh.cosine(packed_light.spot.y)
                spot_inner = osh.cosine(packed_light.spot.x)
                packed_attenuation = packed_attenuation * osh.maximum(
                    0.0, osh.minimum(
                        1.0,
                        (spot_cosine - spot_outer)
                        / osh.maximum(spot_inner - spot_outer, 0.00001),
                    ),
                )
            packed_half_delta = packed_incoming + view
            packed_half = packed_half_delta / osh.maximum(
                osh.length(packed_half_delta), 0.000001,
            )
            packed_signed_ndotl = (
                surface_normal.x * packed_incoming.x
                + surface_normal.y * packed_incoming.y
                + surface_normal.z * packed_incoming.z
            )
            packed_ndotl = osh.maximum(packed_signed_ndotl, 0.0)
            packed_ndoth = osh.maximum(
                surface_normal.x * packed_half.x
                + surface_normal.y * packed_half.y
                + surface_normal.z * packed_half.z,
                0.0,
            )
            packed_vdoth = osh.minimum(
                1.0, osh.maximum(
                    view.x * packed_half.x + view.y * packed_half.y
                    + view.z * packed_half.z,
                    0.0,
                ),
            )
            packed_fresnel = f0 + (osh.vec3(1.0) - f0) * osh.power(
                osh.vec3(1.0 - packed_vdoth), osh.vec3(5.0),
            )
            packed_tangent_dot_half = (
                tangent.x * packed_half.x + tangent.y * packed_half.y
                + tangent.z * packed_half.z
            )
            packed_bitangent_dot_half = (
                bitangent.x * packed_half.x + bitangent.y * packed_half.y
                + bitangent.z * packed_half.z
            )
            packed_anisotropic_denominator = (
                packed_tangent_dot_half * packed_tangent_dot_half
                / (alpha_x * alpha_x)
                + packed_bitangent_dot_half * packed_bitangent_dot_half
                / (alpha_y * alpha_y)
                + packed_ndoth * packed_ndoth
            )
            packed_distribution = 1.0 / osh.maximum(
                3.14159265 * alpha_x * alpha_y
                * packed_anisotropic_denominator
                * packed_anisotropic_denominator,
                0.000001,
            )
            packed_geometry_l = packed_ndotl / osh.maximum(
                packed_ndotl * (1.0 - k) + k, 0.000001,
            )
            packed_specular = (
                packed_distribution * geometry_v * packed_geometry_l
                * packed_fresnel / osh.maximum(
                    4.0 * ndotv * packed_ndotl, 0.000001,
                )
            )
            packed_coat_denominator = (
                packed_ndoth * packed_ndoth * (coat_alpha2 - 1.0) + 1.0
            )
            packed_coat_distribution = coat_alpha2 / osh.maximum(
                3.14159265 * packed_coat_denominator
                * packed_coat_denominator,
                0.000001,
            )
            packed_coat_fresnel = (
                0.04 + 0.96 * osh.power(1.0 - packed_vdoth, 5.0)
            )
            packed_coat_geometry_l = packed_ndotl / osh.maximum(
                packed_ndotl * (1.0 - coat_k) + coat_k, 0.000001,
            )
            packed_clearcoat = osh.vec3(
                surface_clearcoat * packed_coat_distribution
                * coat_geometry_v * packed_coat_geometry_l
                * packed_coat_fresnel / osh.maximum(
                    4.0 * ndotv * packed_ndotl, 0.000001,
                )
            )
            packed_base_energy = 1.0 - surface_clearcoat * packed_coat_fresnel
            packed_diffuse_base = (
                (osh.vec3(1.0) - packed_fresnel) * surface_base_color
                * diffuse_weight / 3.14159265
            )
            packed_diffuse = osh.mix(
                packed_diffuse_base,
                packed_diffuse_base * hooked.subsurface_color,
                surface_subsurface,
            )
            packed_wrapped_ndotl = osh.maximum(
                0.0, (packed_signed_ndotl + surface_subsurface_radius)
                / (1.0 + surface_subsurface_radius),
            )
            packed_diffuse_ndotl = osh.mix(
                packed_ndotl, packed_wrapped_ndotl, surface_subsurface,
            )
            packed_sheen = surface_sheen_color * (
                (1.0 - surface_sheen_roughness)
                * osh.power(1.0 - packed_vdoth, 5.0)
            )
            packed_radiance = (
                packed_light.color_intensity.xyz
                * packed_light.color_intensity.w
            )
            packed_shadow_visibility = 1.0
            for packed_shadow_index in range(24):
                if packed_shadow_index < osh.i32(camera.optical_diagnostic.z):
                    packed_shadow = shadows[osh.u32(packed_shadow_index)]
                    if osh.absolute(
                        packed_shadow.parameters.x - osh.f32(light_index)
                    ) < 0.25:
                        packed_shadow_face_matches = True
                        if packed_shadow.parameters.w > 1.5:
                            packed_point_shadow_delta = (
                                world_position - packed_light.position_type.xyz
                            )
                            packed_point_shadow_axis = osh.absolute(
                                packed_point_shadow_delta
                            )
                            packed_point_shadow_face = 0.0
                            if (
                                packed_point_shadow_axis.y
                                >= packed_point_shadow_axis.x
                                and packed_point_shadow_axis.y
                                >= packed_point_shadow_axis.z
                            ):
                                packed_point_shadow_face = (
                                    2.0 if packed_point_shadow_delta.y >= 0.0
                                    else 3.0
                                )
                            elif (
                                packed_point_shadow_axis.z
                                >= packed_point_shadow_axis.x
                            ):
                                packed_point_shadow_face = (
                                    4.0 if packed_point_shadow_delta.z >= 0.0
                                    else 5.0
                                )
                            else:
                                packed_point_shadow_face = (
                                    0.0 if packed_point_shadow_delta.x >= 0.0
                                    else 1.0
                                )
                            packed_shadow_face_matches = osh.absolute(
                                packed_shadow.parameters.y
                                - packed_point_shadow_face
                            ) < 0.25
                        if not packed_shadow_face_matches:
                            continue
                        packed_shadow_position = world_position + (
                            packed_incoming * packed_shadow.parameters.z
                        )
                        packed_shadow_clip = (
                            packed_shadow.view_projection
                            * osh.vec4(packed_shadow_position, 1.0)
                        )
                        packed_shadow_w = osh.maximum(
                            osh.absolute(packed_shadow_clip.w), 0.000001,
                        )
                        packed_shadow_ndc = (
                            packed_shadow_clip.xyz / packed_shadow_w
                        )
                        if (
                            packed_shadow_clip.w > 0.0
                            and osh.absolute(packed_shadow_ndc.x) <= 1.0001
                            and osh.absolute(packed_shadow_ndc.y) <= 1.0001
                            and packed_shadow_ndc.z >= 0.0
                            and packed_shadow_ndc.z <= 1.0
                        ):
                            packed_shadow_uv = osh.vec2(
                                packed_shadow.atlas.x
                                + packed_shadow.atlas.y * packed_shadow_ndc.x,
                                packed_shadow.atlas.z
                                - packed_shadow.atlas.w * packed_shadow_ndc.y,
                            )
                            packed_shadow_compare_bias = (
                                0.00002 if packed_shadow.parameters.w > 1.5
                                else 0.00005
                            )
                            packed_shadow_visibility = (
                                shadow_map.sample_compare_with(
                                    shadow_sampler, packed_shadow_uv,
                                    packed_shadow_ndc.z
                                    - packed_shadow_compare_bias,
                                )
                            )
            packed_wrapped_contribution = osh.maximum(
                packed_diffuse_ndotl - packed_ndotl, 0.0,
            )
            packed_scattered_visibility = osh.mix(
                packed_shadow_visibility, 1.0, surface_subsurface,
            )
            packed_diffuse_visibility = (
                packed_ndotl * packed_shadow_visibility
                + packed_wrapped_contribution * packed_scattered_visibility
            )
            direct = direct + (
                (packed_diffuse + packed_sheen) * packed_base_energy
                * packed_radiance * packed_diffuse_visibility
                + (packed_specular * packed_base_energy + packed_clearcoat)
                * packed_radiance * packed_ndotl * packed_shadow_visibility
            ) * packed_attenuation
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
    # A thick closed dielectric has two interfaces.  Use the authored
    # thickness as the diameter of a local osculating sphere to recover a
    # stable exit point and exit normal.  This is exact for spheres, a useful
    # local approximation for smooth convex objects, and is bypassed for
    # explicitly thin-walled materials.  The later screen-space exit march
    # can refine this proxy when a matching back surface is available.
    proxy_radius = osh.maximum(optical_thickness * 0.5, 0.0001)
    proxy_center = world_position - surface_normal * proxy_radius
    proxy_center_delta = world_position - proxy_center
    proxy_exit_distance = osh.maximum(
        -2.0 * (
            refracted.x * proxy_center_delta.x
            + refracted.y * proxy_center_delta.y
            + refracted.z * proxy_center_delta.z
        ),
        0.0,
    )
    proxy_exit_position = world_position + refracted * proxy_exit_distance
    proxy_exit_delta = proxy_exit_position - proxy_center
    proxy_exit_normal = proxy_exit_delta / osh.maximum(
        osh.length(proxy_exit_delta), 0.000001,
    )
    raw_secondary_refracted = osh.refract(
        refracted, -proxy_exit_normal,
        osh.maximum(ior_distance_program_flags.x, 1.0001),
    )
    secondary_reflected = refracted - (-proxy_exit_normal) * (
        2.0 * (
            refracted.x * (-proxy_exit_normal.x)
            + refracted.y * (-proxy_exit_normal.y)
            + refracted.z * (-proxy_exit_normal.z)
        )
    )
    secondary_refracted = secondary_reflected if (
        raw_secondary_refracted.x * raw_secondary_refracted.x
        + raw_secondary_refracted.y * raw_secondary_refracted.y
        + raw_secondary_refracted.z * raw_secondary_refracted.z < 0.0001
    ) else raw_secondary_refracted
    closed_refracted = (
        secondary_refracted if surface_thin_walled < 0.5 else refracted
    )
    closed_exit_position = (
        proxy_exit_position if surface_thin_walled < 0.5 else world_position
    )
    # Local probes describe radiance at a finite position rather than at
    # infinity. Intersect each outgoing ray with the probe influence sphere
    # and sample the direction from the probe center to that point. This
    # spherical parallax correction is stable for arbitrary camera poses and
    # reduces the sliding produced by treating a room probe as an environment.
    probe_reflected = reflected
    probe_refracted = closed_refracted
    if probe_position_radius.w > 0.0:
        reflected_offset = world_position - probe_position_radius.xyz
        reflected_b = (
            reflected_offset.x * reflected.x
            + reflected_offset.y * reflected.y
            + reflected_offset.z * reflected.z
        )
        reflected_c = (
            reflected_offset.x * reflected_offset.x
            + reflected_offset.y * reflected_offset.y
            + reflected_offset.z * reflected_offset.z
            - probe_position_radius.w * probe_position_radius.w
        )
        reflected_t = osh.maximum(
            0.0, -reflected_b + osh.sqrt(osh.maximum(
                0.0, reflected_b * reflected_b - reflected_c,
            )),
        )
        probe_reflected = (
            world_position + reflected * reflected_t
            - probe_position_radius.xyz
        )
        probe_reflected = probe_reflected / osh.maximum(
            osh.length(probe_reflected), 0.000001,
        )
        refracted_offset = closed_exit_position - probe_position_radius.xyz
        refracted_b = (
            refracted_offset.x * closed_refracted.x
            + refracted_offset.y * closed_refracted.y
            + refracted_offset.z * closed_refracted.z
        )
        refracted_c = (
            refracted_offset.x * refracted_offset.x
            + refracted_offset.y * refracted_offset.y
            + refracted_offset.z * refracted_offset.z
            - probe_position_radius.w * probe_position_radius.w
        )
        refracted_t = osh.maximum(
            0.0, -refracted_b + osh.sqrt(osh.maximum(
                0.0, refracted_b * refracted_b - refracted_c,
            )),
        )
        probe_refracted = (
            closed_exit_position + closed_refracted * refracted_t
            - probe_position_radius.xyz
        )
        probe_refracted = probe_refracted / osh.maximum(
            osh.length(probe_refracted), 0.000001,
        )
        if probe_box_min_mode.w > 0.5:
            reflected_safe = osh.vec3(
                reflected.x if osh.absolute(reflected.x) > 0.000001 else 0.000001,
                reflected.y if osh.absolute(reflected.y) > 0.000001 else 0.000001,
                reflected.z if osh.absolute(reflected.z) > 0.000001 else 0.000001,
            )
            reflected_to_min = (probe_box_min_mode.xyz - world_position) / reflected_safe
            reflected_to_max = (probe_box_max_blend.xyz - world_position) / reflected_safe
            reflected_far = osh.maximum(reflected_to_min, reflected_to_max)
            reflected_box_t = osh.minimum(
                reflected_far.x, osh.minimum(reflected_far.y, reflected_far.z),
            )
            probe_reflected = (
                world_position + reflected * reflected_box_t
                - probe_position_radius.xyz
            )
            probe_reflected = probe_reflected / osh.maximum(
                osh.length(probe_reflected), 0.000001,
            )
            refracted_safe = osh.vec3(
                closed_refracted.x if osh.absolute(closed_refracted.x) > 0.000001 else 0.000001,
                closed_refracted.y if osh.absolute(closed_refracted.y) > 0.000001 else 0.000001,
                closed_refracted.z if osh.absolute(closed_refracted.z) > 0.000001 else 0.000001,
            )
            refracted_to_min = (probe_box_min_mode.xyz - closed_exit_position) / refracted_safe
            refracted_to_max = (probe_box_max_blend.xyz - closed_exit_position) / refracted_safe
            refracted_far = osh.maximum(refracted_to_min, refracted_to_max)
            refracted_box_t = osh.minimum(
                refracted_far.x, osh.minimum(refracted_far.y, refracted_far.z),
            )
            probe_refracted = (
                closed_exit_position + closed_refracted * refracted_box_t
                - probe_position_radius.xyz
            )
            probe_refracted = probe_refracted / osh.maximum(
                osh.length(probe_refracted), 0.000001,
            )
    # A secondary local probe must be projected from its own capture origin.
    # Blending images first and applying the primary transform afterwards
    # makes nearby geometry appear detached because the two captures have
    # different parallax.
    probe_reflected_secondary = reflected
    probe_refracted_secondary = closed_refracted
    if probe_position_radius_secondary.w > 0.0:
        reflected_offset_secondary = (
            world_position - probe_position_radius_secondary.xyz
        )
        reflected_b_secondary = (
            reflected_offset_secondary.x * reflected.x
            + reflected_offset_secondary.y * reflected.y
            + reflected_offset_secondary.z * reflected.z
        )
        reflected_c_secondary = (
            reflected_offset_secondary.x * reflected_offset_secondary.x
            + reflected_offset_secondary.y * reflected_offset_secondary.y
            + reflected_offset_secondary.z * reflected_offset_secondary.z
            - probe_position_radius_secondary.w
            * probe_position_radius_secondary.w
        )
        reflected_sphere_t_secondary = osh.maximum(
            0.0, -reflected_b_secondary + osh.sqrt(osh.maximum(
                0.0,
                reflected_b_secondary * reflected_b_secondary
                - reflected_c_secondary,
            )),
        )
        probe_reflected_secondary = (
            world_position + reflected * reflected_sphere_t_secondary
            - probe_position_radius_secondary.xyz
        )
        probe_reflected_secondary = probe_reflected_secondary / osh.maximum(
            osh.length(probe_reflected_secondary), 0.000001,
        )
        refracted_offset_secondary = (
            closed_exit_position - probe_position_radius_secondary.xyz
        )
        refracted_b_secondary = (
            refracted_offset_secondary.x * closed_refracted.x
            + refracted_offset_secondary.y * closed_refracted.y
            + refracted_offset_secondary.z * closed_refracted.z
        )
        refracted_c_secondary = (
            refracted_offset_secondary.x * refracted_offset_secondary.x
            + refracted_offset_secondary.y * refracted_offset_secondary.y
            + refracted_offset_secondary.z * refracted_offset_secondary.z
            - probe_position_radius_secondary.w
            * probe_position_radius_secondary.w
        )
        refracted_sphere_t_secondary = osh.maximum(
            0.0, -refracted_b_secondary + osh.sqrt(osh.maximum(
                0.0,
                refracted_b_secondary * refracted_b_secondary
                - refracted_c_secondary,
            )),
        )
        probe_refracted_secondary = (
            closed_exit_position
            + closed_refracted * refracted_sphere_t_secondary
            - probe_position_radius_secondary.xyz
        )
        probe_refracted_secondary = probe_refracted_secondary / osh.maximum(
            osh.length(probe_refracted_secondary), 0.000001,
        )
    if probe_box_min_mode_secondary.w > 0.5:
        reflected_safe_secondary = osh.vec3(
            reflected.x if osh.absolute(reflected.x) > 0.000001 else 0.000001,
            reflected.y if osh.absolute(reflected.y) > 0.000001 else 0.000001,
            reflected.z if osh.absolute(reflected.z) > 0.000001 else 0.000001,
        )
        reflected_min_secondary = (
            probe_box_min_mode_secondary.xyz - world_position
        ) / reflected_safe_secondary
        reflected_max_secondary = (
            probe_box_max_blend_secondary.xyz - world_position
        ) / reflected_safe_secondary
        reflected_far_secondary = osh.maximum(
            reflected_min_secondary, reflected_max_secondary,
        )
        reflected_t_secondary = osh.minimum(
            reflected_far_secondary.x,
            osh.minimum(reflected_far_secondary.y, reflected_far_secondary.z),
        )
        probe_reflected_secondary = (
            world_position + reflected * reflected_t_secondary
            - probe_position_radius_secondary.xyz
        )
        probe_reflected_secondary = probe_reflected_secondary / osh.maximum(
            osh.length(probe_reflected_secondary), 0.000001,
        )
        refracted_safe_secondary = osh.vec3(
            closed_refracted.x if osh.absolute(closed_refracted.x) > 0.000001 else 0.000001,
            closed_refracted.y if osh.absolute(closed_refracted.y) > 0.000001 else 0.000001,
            closed_refracted.z if osh.absolute(closed_refracted.z) > 0.000001 else 0.000001,
        )
        refracted_min_secondary = (
            probe_box_min_mode_secondary.xyz - closed_exit_position
        ) / refracted_safe_secondary
        refracted_max_secondary = (
            probe_box_max_blend_secondary.xyz - closed_exit_position
        ) / refracted_safe_secondary
        refracted_far_secondary = osh.maximum(
            refracted_min_secondary, refracted_max_secondary,
        )
        refracted_t_secondary = osh.minimum(
            refracted_far_secondary.x,
            osh.minimum(refracted_far_secondary.y, refracted_far_secondary.z),
        )
        probe_refracted_secondary = (
            closed_exit_position + closed_refracted * refracted_t_secondary
            - probe_position_radius_secondary.xyz
        )
        probe_refracted_secondary = probe_refracted_secondary / osh.maximum(
            osh.length(probe_refracted_secondary), 0.000001,
        )
    reflection_uv = osh.vec2(
        osh.fraction(
            osh.arctangent2(probe_reflected.z, probe_reflected.x) / 6.28318531
            + 0.5 + environment_rotation_log_range.x / 6.28318531
        ),
        osh.arccosine(osh.maximum(-1.0, osh.minimum(1.0, probe_reflected.y)))
        / 3.14159265,
    )
    refraction_uv = osh.vec2(
        osh.fraction(
            osh.arctangent2(probe_refracted.z, probe_refracted.x) / 6.28318531
            + 0.5 + environment_rotation_log_range.x / 6.28318531
        ),
        osh.arccosine(osh.maximum(-1.0, osh.minimum(1.0, probe_refracted.y)))
        / 3.14159265,
    )
    reflection_uv_secondary = osh.vec2(
        osh.fraction(
            osh.arctangent2(
                probe_reflected_secondary.z, probe_reflected_secondary.x,
            ) / 6.28318531 + 0.5
            + environment_rotation_log_range_secondary.x / 6.28318531
        ),
        osh.arccosine(osh.maximum(
            -1.0, osh.minimum(1.0, probe_reflected_secondary.y),
        )) / 3.14159265,
    )
    refraction_uv_secondary = osh.vec2(
        osh.fraction(
            osh.arctangent2(
                probe_refracted_secondary.z, probe_refracted_secondary.x,
            ) / 6.28318531 + 0.5
            + environment_rotation_log_range_secondary.x / 6.28318531
        ),
        osh.arccosine(osh.maximum(
            -1.0, osh.minimum(1.0, probe_refracted_secondary.y),
        )) / 3.14159265,
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
    reflection_uv_secondary_low = osh.vec2(
        (reflection_uv_secondary.x + environment_level_low) * 0.25,
        reflection_uv_secondary.y,
    )
    reflection_uv_secondary_high = osh.vec2(
        (reflection_uv_secondary.x + environment_level_high) * 0.25,
        reflection_uv_secondary.y,
    )
    refraction_uv_secondary_low = osh.vec2(
        (refraction_uv_secondary.x + environment_level_low) * 0.25,
        refraction_uv_secondary.y,
    )
    refraction_uv_secondary_high = osh.vec2(
        (refraction_uv_secondary.x + environment_level_high) * 0.25,
        refraction_uv_secondary.y,
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
    reflected_secondary_low = base_color_atlas.sample_with(
        base_color_sampler,
        environment_rect_secondary.xy
        + reflection_uv_secondary_low * environment_rect_secondary.zw,
    ).xyz
    reflected_secondary_high = base_color_atlas.sample_with(
        base_color_sampler,
        environment_rect_secondary.xy
        + reflection_uv_secondary_high * environment_rect_secondary.zw,
    ).xyz
    refracted_secondary_low = base_color_atlas.sample_with(
        base_color_sampler,
        environment_rect_secondary.xy
        + refraction_uv_secondary_low * environment_rect_secondary.zw,
    ).xyz
    refracted_secondary_high = base_color_atlas.sample_with(
        base_color_sampler,
        environment_rect_secondary.xy
        + refraction_uv_secondary_high * environment_rect_secondary.zw,
    ).xyz
    reflected_encoded = osh.mix(
        reflected_encoded_low, reflected_encoded_high,
        environment_level_mix,
    )
    refracted_encoded = osh.mix(
        refracted_encoded_low, refracted_encoded_high,
        environment_level_mix,
    ).xyz
    reflected_secondary_encoded = osh.mix(
        reflected_secondary_low, reflected_secondary_high,
        environment_level_mix,
    )
    refracted_secondary_encoded = osh.mix(
        refracted_secondary_low, refracted_secondary_high,
        environment_level_mix,
    )
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
    secondary_enabled = (
        1.0 if environment_rect_secondary.z > 0.0 else 0.0
    )
    secondary_reflected_environment = (
        osh.power(
            osh.vec3(2.0),
            reflected_secondary_encoded
            * environment_rotation_log_range_secondary.y,
        ) - osh.vec3(1.0)
    )
    secondary_refracted_environment = (
        osh.power(
            osh.vec3(2.0),
            refracted_secondary_encoded
            * environment_rotation_log_range_secondary.y,
        ) - osh.vec3(1.0)
    )
    primary_weight = (
        probe_box_max_blend.w if secondary_enabled > 0.5 else 1.0
    )
    secondary_weight = probe_box_max_blend_secondary.w * secondary_enabled
    weight_sum = osh.maximum(primary_weight + secondary_weight, 0.000001)
    reflected_environment = (
        reflected_environment * primary_weight
        + secondary_reflected_environment * secondary_weight
    ) / weight_sum
    refracted_environment = (
        refracted_environment * primary_weight
        + secondary_refracted_environment * secondary_weight
    ) / weight_sum
    screen_clip = camera.view_projection * osh.vec4(world_position, 1.0)
    screen_ndc = screen_clip.xyz / osh.maximum(
        osh.absolute(screen_clip.w), 0.000001,
    )
    screen_uv = osh.vec2(
        screen_ndc.x * 0.5 + 0.5,
        0.5 - screen_ndc.y * 0.5,
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
    # Refraction is a world-space direction.  Adding ``refracted.xy`` directly
    # to screen UV only works when the camera happens to align with the world
    # XY axes; changing camera azimuth or IOR then sends otherwise identical
    # objects toward unrelated parts of the scene buffer. Project a point
    # reached by the refracted ray instead, so the displacement follows the
    # active view/projection transform.
    refraction_world_distance = (
        0.30 + optical_thickness * 0.18
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
            0.5 - ray_ndc.y * 0.5,
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
                    0.5 - middle_ndc.y * 0.5,
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
    # Follow the outgoing ray until it crosses the completed opaque depth
    # buffer. A fixed-distance projection is not a scene intersection: its UV
    # can land on arbitrary dark geometry as IOR changes even though the ray
    # would eventually reach a bright wall or floor. The depth march makes the
    # selected scene-color sample correspond to an actual visible receiver.
    refraction_origin = closed_exit_position + closed_refracted * 0.03
    refraction_extent = osh.maximum(8.0, osh.length(view_delta) * 3.0)
    refraction_screen_uv = screen_uv
    nested_outer_layer = osh.maximum(
        0.0, osh.minimum(
            1.0,
            camera.optical_diagnostic.w
            * osh.maximum(optical_layer_role, 0.0),
        ),
    )
    # The accumulated source depth contains the already-rendered inner
    # interface.  It lies *inside* the current outer shell, so an outgoing ray
    # starting at ``closed_exit_position`` can never discover it.  March the
    # entry refraction through the bounded interior first; pixels that miss
    # the inner interface continue to use the ordinary outgoing march below.
    nested_internal_uv = screen_uv
    nested_internal_hit = 0.0
    nested_previous_delta = -1.0
    nested_previous_fraction = 0.0
    if nested_outer_layer > 0.5:
        for nested_step in range(1, 25):
            nested_fraction = osh.f32(nested_step) / 24.0
            nested_distance = (
                0.02 + nested_fraction * nested_fraction
                * osh.maximum(proxy_exit_distance - 0.04, 0.04)
            )
            nested_world = world_position + refracted * nested_distance
            nested_clip = (
                camera.view_projection * osh.vec4(nested_world, 1.0)
            )
            if nested_clip.w <= 0.000001:
                break
            nested_ndc = nested_clip.xyz / nested_clip.w
            nested_uv_candidate = osh.vec2(
                nested_ndc.x * 0.5 + 0.5,
                0.5 - nested_ndc.y * 0.5,
            )
            if (
                nested_uv_candidate.x <= 0.001
                or nested_uv_candidate.x >= 0.999
                or nested_uv_candidate.y <= 0.001
                or nested_uv_candidate.y >= 0.999
            ):
                break
            nested_scene_depth = scene_depth.sample_depth_level_with(
                scene_depth_sampler, nested_uv_candidate, 0,
            )
            nested_delta = nested_ndc.z - nested_scene_depth
            if nested_previous_delta <= 0.00001 and nested_delta > 0.00001:
                nested_lower = nested_previous_fraction
                nested_upper = nested_fraction
                nested_refined_uv = nested_uv_candidate
                for nested_refine_step in range(4):
                    nested_middle = (nested_lower + nested_upper) * 0.5
                    nested_middle_distance = (
                        0.02 + nested_middle * nested_middle
                        * osh.maximum(proxy_exit_distance - 0.04, 0.04)
                    )
                    nested_middle_world = (
                        world_position + refracted * nested_middle_distance
                    )
                    nested_middle_clip = (
                        camera.view_projection
                        * osh.vec4(nested_middle_world, 1.0)
                    )
                    nested_middle_ndc = (
                        nested_middle_clip.xyz
                        / osh.maximum(nested_middle_clip.w, 0.000001)
                    )
                    nested_middle_uv = osh.vec2(
                        nested_middle_ndc.x * 0.5 + 0.5,
                        0.5 - nested_middle_ndc.y * 0.5,
                    )
                    nested_middle_depth = scene_depth.sample_depth_level_with(
                        scene_depth_sampler, nested_middle_uv, 0,
                    )
                    nested_middle_delta = (
                        nested_middle_ndc.z - nested_middle_depth
                    )
                    if nested_middle_delta > 0.00001:
                        nested_upper = nested_middle
                        nested_refined_uv = nested_middle_uv
                    else:
                        nested_lower = nested_middle
                nested_internal_uv = nested_refined_uv
                nested_internal_hit = 1.0
                break
            nested_previous_fraction = nested_fraction
            nested_previous_delta = nested_delta
    refraction_hit = 0.0
    previous_refraction_fraction = 0.0
    previous_refraction_delta = -1.0
    for refraction_step in range(1, 25):
        refraction_fraction = osh.f32(refraction_step) / 24.0
        refraction_distance = (
            0.08
            + refraction_fraction * refraction_fraction * refraction_extent
        )
        refraction_world = (
            refraction_origin + closed_refracted * refraction_distance
        )
        refraction_clip = (
            camera.view_projection * osh.vec4(refraction_world, 1.0)
        )
        if refraction_clip.w <= 0.000001:
            break
        refraction_ndc = refraction_clip.xyz / refraction_clip.w
        refraction_uv_candidate = osh.vec2(
            refraction_ndc.x * 0.5 + 0.5,
            0.5 - refraction_ndc.y * 0.5,
        )
        if (
            refraction_uv_candidate.x <= 0.001
            or refraction_uv_candidate.x >= 0.999
            or refraction_uv_candidate.y <= 0.001
            or refraction_uv_candidate.y >= 0.999
        ):
            break
        refraction_scene_depth = scene_depth.sample_depth_level_with(
            scene_depth_sampler, refraction_uv_candidate, 0,
        )
        refraction_delta = refraction_ndc.z - refraction_scene_depth
        if (
            previous_refraction_delta <= 0.00001
            and refraction_delta > 0.00001
        ):
            lower_refraction_fraction = previous_refraction_fraction
            upper_refraction_fraction = refraction_fraction
            refined_refraction_uv = refraction_uv_candidate
            refined_refraction_delta = refraction_delta
            for refraction_refine_step in range(4):
                middle_refraction_fraction = (
                    lower_refraction_fraction + upper_refraction_fraction
                ) * 0.5
                middle_refraction_distance = (
                    0.08 + middle_refraction_fraction
                    * middle_refraction_fraction * refraction_extent
                )
                middle_refraction_world = (
                    refraction_origin
                    + closed_refracted * middle_refraction_distance
                )
                middle_refraction_clip = (
                    camera.view_projection
                    * osh.vec4(middle_refraction_world, 1.0)
                )
                middle_refraction_ndc = (
                    middle_refraction_clip.xyz
                    / osh.maximum(middle_refraction_clip.w, 0.000001)
                )
                middle_refraction_uv = osh.vec2(
                    middle_refraction_ndc.x * 0.5 + 0.5,
                    0.5 - middle_refraction_ndc.y * 0.5,
                )
                middle_refraction_depth = (
                    scene_depth.sample_depth_level_with(
                        scene_depth_sampler, middle_refraction_uv, 0,
                    )
                )
                middle_refraction_delta = (
                    middle_refraction_ndc.z - middle_refraction_depth
                )
                if middle_refraction_delta > 0.00001:
                    upper_refraction_fraction = middle_refraction_fraction
                    refined_refraction_uv = middle_refraction_uv
                    refined_refraction_delta = middle_refraction_delta
                else:
                    lower_refraction_fraction = middle_refraction_fraction
            refraction_thickness = (
                0.00006 + upper_refraction_fraction * 0.00030
            )
            if refined_refraction_delta < refraction_thickness:
                refraction_screen_uv = refined_refraction_uv
                refraction_hit = 1.0
            break
        previous_refraction_fraction = refraction_fraction
        previous_refraction_delta = refraction_delta
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
    nested_internal_refracted = scene_color.sample_level_with(
        scene_sampler, nested_internal_uv, 0.0,
    ).xyz
    reflected_source = osh.mix(
        reflected_environment, screen_reflected,
        reflection_hit * screen_enabled * (1.0 - surface_roughness),
    )
    refraction_edge_distance = osh.minimum(
        osh.minimum(refraction_screen_uv.x, 1.0 - refraction_screen_uv.x),
        osh.minimum(refraction_screen_uv.y, 1.0 - refraction_screen_uv.y),
    )
    refraction_edge_confidence = osh.maximum(
        0.0, osh.minimum(1.0, refraction_edge_distance * 16.0),
    )
    refraction_angle_confidence = osh.maximum(
        0.0, osh.minimum(1.0, (ndotv - 0.08) * 2.5),
    )
    refraction_confidence = (
        refraction_hit * refraction_edge_confidence
        * refraction_angle_confidence
    )
    screen_transmitted_source = osh.mix(
        refracted_environment, screen_refracted,
        refraction_confidence,
    )
    screen_transmitted_source = osh.mix(
        screen_transmitted_source, nested_internal_refracted,
        nested_internal_hit * nested_outer_layer,
    )
    screen_transmission_confidence = osh.maximum(
        refraction_confidence,
        nested_internal_hit * nested_outer_layer,
    )
    refracted_source = osh.mix(
        refracted_environment, screen_transmitted_source,
        screen_transmission_confidence * screen_enabled
        * (1.0 - surface_roughness),
    )
    environment_enabled = environment_rotation_log_range.z
    reflection_source_enabled = osh.maximum(
        environment_enabled, reflection_hit * screen_enabled,
    )
    refraction_source_enabled = osh.maximum(
        environment_enabled,
        screen_transmission_confidence * screen_enabled,
    )
    ambient = (
        surface_base_color * diffuse_weight
        + f0 * (1.0 - 0.5 * surface_roughness)
        + transmission_tint * surface_transmission * (osh.vec3(1.0) - f0)
    ) * light_color_ambient.w * surface_occlusion
    base_shaded = ambient + direct + surface_emission
    transmission_weight = (
        surface_transmission * refraction_source_enabled
    )
    # Transmission is not an additional straight-through color for a
    # dielectric.  It selects the refracted radiance path, whose remaining
    # energy is 1-F after the reflected interface lobe is accounted for.
    # Thin-walled and closed materials differ in how ``refracted_source`` is
    # constructed above, but share this energy-conserving composition.
    dielectric_refracted = (
        refracted_source * transmission_tint
        * (osh.vec3(1.0) - optical_fresnel)
    )
    transmitted_shaded = osh.mix(
        base_shaded,
        dielectric_refracted,
        transmission_weight,
    )
    shaded = (
        transmitted_shaded
        + reflected_source * optical_fresnel * reflection_source_enabled
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
    if diagnostic_mode > 5.5 and diagnostic_mode < 6.5:
        return osh.vec4(
            diagnostic_depth_trace,
            osh.fraction(diagnostic_depth_trace),
            diagnostic_ray_step,
            1.0,
        )
    if diagnostic_mode > 6.5 and diagnostic_mode < 7.5:
        return osh.vec4(osh.vec3(refraction_hit), 1.0)
    if diagnostic_mode > 7.5 and diagnostic_mode < 8.5:
        return osh.vec4(refraction_screen_uv, refraction_hit, 1.0)
    if diagnostic_mode > 8.5:
        return osh.vec4(refracted_source, 1.0)
    return result


__all__ = [
    "RasterCamera", "RasterMaterial", "RasterMaterialContext", "RasterSurface",
    "SceneVertexOutput", "ShadowVertexOutput", "blend_raster_surfaces",
    "default_raster_material_hook", "scene_fragment",
    "scene_vertex", "shadow_fragment", "shadow_vertex",
]
