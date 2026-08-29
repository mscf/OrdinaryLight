@group(0) @binding(0) var base_color_atlas: texture_2d<f32>;
@group(0) @binding(1) var base_color_sampler: sampler;
@group(0) @binding(2) var shadow_map: texture_depth_2d;
@group(0) @binding(4) var shadow_sampler: sampler_comparison;

@fragment
fn main(
    @location(0) base_color: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_position: vec3<f32>,
    @location(3) material: vec4<f32>,
    @location(4) emission: vec3<f32>,
    @location(5) camera_position: vec3<f32>,
    @location(6) light_position_type: vec4<f32>,
    @location(7) light_color_ambient: vec4<f32>,
    @location(8) base_color_uv: vec2<f32>,
    @location(9) shadow_coordinate: vec4<f32>,
    @location(10) shadow_visibility: f32
) -> @location(0) vec4<f32> {
    let sampled_base_color: vec3<f32> = (base_color * textureSample(base_color_atlas, base_color_sampler, base_color_uv).xyz);
    let shadow_w: f32 = max(abs(shadow_coordinate.w), 1e-06);
    let projected_shadow: vec3<f32> = (shadow_coordinate.xyz / shadow_w);
    let normal: vec3<f32> = (world_normal / max(length(world_normal), 1e-06));
    let view_delta: vec3<f32> = (camera_position - world_position);
    let view: vec3<f32> = (view_delta / max(length(view_delta), 1e-06));
    let light_delta: vec3<f32> = (light_position_type.xyz - world_position);
    let point_distance: f32 = max(length(light_delta), 1e-06);
    let point_incoming: vec3<f32> = (light_delta / point_distance);
    let direction: vec3<f32> = (-light_position_type.xyz);
    let directional_incoming: vec3<f32> = (direction / max(length(direction), 1e-06));
    let incoming: vec3<f32> = select(point_incoming, directional_incoming, (light_position_type.w > 0.5));
    let distance: f32 = select(point_distance, 1.0, (light_position_type.w > 0.5));
    let half_delta: vec3<f32> = (incoming + view);
    let half_vector: vec3<f32> = (half_delta / max(length(half_delta), 1e-06));
    let ndotl: f32 = max((((normal.x * incoming.x) + (normal.y * incoming.y)) + (normal.z * incoming.z)), 0.0);
    let receiver_bias: f32 = max(2e-05, ((1.0 - ndotl) * 0.0001));
    let pcf_visibility: f32 = textureSampleCompare(shadow_map, shadow_sampler, projected_shadow.xy, (projected_shadow.z - receiver_bias));
    let shadow_map_visibility: f32 = select(pcf_visibility, 1.0, (abs(shadow_coordinate.w) < 1e-06));
    let ndotv: f32 = max((((normal.x * view.x) + (normal.y * view.y)) + (normal.z * view.z)), 0.0);
    let ndoth: f32 = max((((normal.x * half_vector.x) + (normal.y * half_vector.y)) + (normal.z * half_vector.z)), 0.0);
    let vdoth: f32 = max((((view.x * half_vector.x) + (view.y * half_vector.y)) + (view.z * half_vector.z)), 0.0);
    let metallic: f32 = material.x;
    let roughness: f32 = max(material.y, 0.04);
    let transmission: f32 = material.z;
    let surface_alpha: f32 = material.w;
    let f0: vec3<f32> = mix(vec3<f32>(0.04), sampled_base_color, vec3<f32>(metallic));
    let fresnel: vec3<f32> = (f0 + ((vec3<f32>(1.0) - f0) * pow(vec3<f32>((1.0 - vdoth)), vec3<f32>(5.0))));
    let alpha: f32 = (roughness * roughness);
    let alpha2: f32 = (alpha * alpha);
    let denominator: f32 = (((ndoth * ndoth) * (alpha2 - 1.0)) + 1.0);
    let distribution: f32 = (alpha2 / max(((3.14159265 * denominator) * denominator), 1e-06));
    let k: f32 = (((roughness + 1.0) * (roughness + 1.0)) / 8.0);
    let geometry_v: f32 = (ndotv / max(((ndotv * (1.0 - k)) + k), 1e-06));
    let geometry_l: f32 = (ndotl / max(((ndotl * (1.0 - k)) + k), 1e-06));
    let specular: vec3<f32> = ((((distribution * geometry_v) * geometry_l) * fresnel) / max(((4.0 * ndotv) * ndotl), 1e-06));
    let diffuse_weight: f32 = ((1.0 - metallic) * (1.0 - transmission));
    let diffuse: vec3<f32> = ((((vec3<f32>(1.0) - fresnel) * sampled_base_color) * diffuse_weight) / 3.14159265);
    let attenuation: f32 = (1.0 / (distance * distance));
    let direct: vec3<f32> = (((diffuse + specular) * light_color_ambient.xyz) * (((ndotl * attenuation) * shadow_visibility) * shadow_map_visibility));
    let ambient: vec3<f32> = ((((sampled_base_color * diffuse_weight) + (f0 * (1.0 - (0.5 * roughness)))) + (vec3<f32>(transmission) * (vec3<f32>(1.0) - f0))) * light_color_ambient.w);
    return vec4<f32>(((ambient + direct) + emission), surface_alpha);
}
