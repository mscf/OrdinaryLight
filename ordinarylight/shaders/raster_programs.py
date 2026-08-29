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


@osh.structure
class ShadowVertexOutput:
    position: osh.builtin(osh.vec4, "position")
    clip_depth: osh.location(osh.vec2, 0)


@osh.structure
class RasterCamera:
    view_projection: osh.mat4
    position_exposure: osh.vec4


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
    camera: osh.uniform_buffer(RasterCamera, binding=3),
) -> SceneVertexOutput:
    return SceneVertexOutput(
        camera.view_projection * position,
        base_color, world_normal, world_position, material,
        emission, camera.position_exposure.xyz,
        light_position_type, light_color_ambient,
        base_color_uv, shadow_coordinate, shadow_visibility,
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
    base_color_atlas: osh.sampled_texture_2d(binding=0),
    base_color_sampler: osh.sampler(binding=1),
    shadow_map: osh.sampled_depth_texture_2d(binding=2),
    shadow_sampler: osh.comparison_sampler(binding=4),
) -> osh.location(osh.vec4, 0):
    sampled_base_color = base_color * base_color_atlas.sample_with(
        base_color_sampler, base_color_uv,
    ).xyz
    shadow_w = osh.maximum(osh.absolute(shadow_coordinate.w), 0.000001)
    projected_shadow = shadow_coordinate.xyz / shadow_w
    normal = world_normal / osh.maximum(osh.length(world_normal), 0.000001)
    view_delta = camera_position - world_position
    view = view_delta / osh.maximum(osh.length(view_delta), 0.000001)
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
        normal.x * incoming.x + normal.y * incoming.y + normal.z * incoming.z,
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
        normal.x * view.x + normal.y * view.y + normal.z * view.z, 0.0,
    )
    ndoth = osh.maximum(
        normal.x * half_vector.x + normal.y * half_vector.y
        + normal.z * half_vector.z,
        0.0,
    )
    vdoth = osh.maximum(
        view.x * half_vector.x + view.y * half_vector.y
        + view.z * half_vector.z,
        0.0,
    )
    metallic = material.x
    roughness = osh.maximum(material.y, 0.04)
    transmission = material.z
    surface_alpha = material.w
    f0 = osh.mix(osh.vec3(0.04), sampled_base_color, osh.vec3(metallic))
    fresnel = f0 + (osh.vec3(1.0) - f0) * osh.power(
        osh.vec3(1.0 - vdoth), osh.vec3(5.0),
    )
    alpha = roughness * roughness
    alpha2 = alpha * alpha
    denominator = ndoth * ndoth * (alpha2 - 1.0) + 1.0
    distribution = alpha2 / osh.maximum(
        3.14159265 * denominator * denominator, 0.000001,
    )
    k = (roughness + 1.0) * (roughness + 1.0) / 8.0
    geometry_v = ndotv / osh.maximum(ndotv * (1.0 - k) + k, 0.000001)
    geometry_l = ndotl / osh.maximum(ndotl * (1.0 - k) + k, 0.000001)
    specular = distribution * geometry_v * geometry_l * fresnel / osh.maximum(
        4.0 * ndotv * ndotl, 0.000001,
    )
    diffuse_weight = (1.0 - metallic) * (1.0 - transmission)
    diffuse = (
        (osh.vec3(1.0) - fresnel) * sampled_base_color * diffuse_weight / 3.14159265
    )
    attenuation = 1.0 / (distance * distance)
    direct = (
        (diffuse + specular) * light_color_ambient.xyz
        * (ndotl * attenuation * shadow_visibility * shadow_map_visibility)
    )
    ambient = (
        sampled_base_color * diffuse_weight + f0 * (1.0 - 0.5 * roughness)
        + osh.vec3(transmission) * (osh.vec3(1.0) - f0)
    ) * light_color_ambient.w
    return osh.vec4(ambient + direct + emission, surface_alpha)


__all__ = [
    "RasterCamera", "SceneVertexOutput", "ShadowVertexOutput", "scene_fragment",
    "scene_vertex", "shadow_fragment", "shadow_vertex",
]
