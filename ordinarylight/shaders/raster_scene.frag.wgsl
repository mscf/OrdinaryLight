struct RasterMaterial {
    base_color_roughness: vec4<f32>,
    emission_metallic: vec4<f32>,
    attenuation_transmission: vec4<f32>,
    ior_distance_program_flags: vec4<f32>,
    texture_indices: vec4<f32>,
    normal_occlusion_transmission: vec4<f32>,
}

@group(0) @binding(0) var base_color_atlas: texture_2d<f32>;
@group(0) @binding(1) var base_color_sampler: sampler;
@group(0) @binding(2) var shadow_map: texture_depth_2d;
@group(0) @binding(4) var shadow_sampler: sampler_comparison;
@group(0) @binding(5) var<storage, read> materials: array<RasterMaterial>;

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
    @location(10) shadow_visibility: f32,
    @location(11) world_tangent: vec4<f32>,
    @location(12) metallic_roughness_uv: vec2<f32>,
    @location(13) emissive_uv: vec2<f32>,
    @location(14) normal_uv: vec2<f32>,
    @location(15) occlusion_uv: vec2<f32>,
    @location(16) transmission_uv: vec2<f32>,
    @location(17) material_index: f32
) -> @location(0) vec4<f32> {
    if ((material_index < (-0.5))) {
        return vec4<f32>((base_color + emission), material.w);
    }
    let material_id: u32 = u32((material_index + 0.5));
    let base_color_roughness: vec4<f32> = materials[material_id].base_color_roughness;
    let emission_metallic: vec4<f32> = materials[material_id].emission_metallic;
    let attenuation_transmission: vec4<f32> = materials[material_id].attenuation_transmission;
    let ior_distance_program_flags: vec4<f32> = materials[material_id].ior_distance_program_flags;
    let normal_occlusion_transmission: vec4<f32> = materials[material_id].normal_occlusion_transmission;
    let sampled_base_color: vec3<f32> = (base_color_roughness.xyz * textureSample(base_color_atlas, base_color_sampler, base_color_uv).xyz);
    let metallic_roughness_sample: vec4<f32> = textureSample(base_color_atlas, base_color_sampler, metallic_roughness_uv);
    let sampled_emission: vec3<f32> = (emission_metallic.xyz * textureSample(base_color_atlas, base_color_sampler, emissive_uv).xyz);
    let normal_sample: vec3<f32> = ((textureSample(base_color_atlas, base_color_sampler, normal_uv).xyz * 2.0) - vec3<f32>(1.0));
    let occlusion_sample: f32 = textureSample(base_color_atlas, base_color_sampler, occlusion_uv).x;
    let transmission_sample: f32 = textureSample(base_color_atlas, base_color_sampler, transmission_uv).x;
    let shadow_w: f32 = max(abs(shadow_coordinate.w), 1e-06);
    let projected_shadow: vec3<f32> = (shadow_coordinate.xyz / shadow_w);
    let geometric_normal: vec3<f32> = (world_normal / max(length(world_normal), 1e-06));
    let tangent: vec3<f32> = (world_tangent.xyz / max(length(world_tangent.xyz), 1e-06));
    let bitangent: vec3<f32> = (cross(geometric_normal, tangent) * world_tangent.w);
    let scaled_normal_sample: vec3<f32> = vec3<f32>((normal_sample.x * normal_occlusion_transmission.x), (normal_sample.y * normal_occlusion_transmission.x), normal_sample.z);
    let mapped_normal: vec3<f32> = (((tangent * scaled_normal_sample.x) + (bitangent * scaled_normal_sample.y)) + (geometric_normal * scaled_normal_sample.z));
    let normal: vec3<f32> = (mapped_normal / max(length(mapped_normal), 1e-06));
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
    let program_kind: f32 = floor(ior_distance_program_flags.w);
    let mirror_program: bool = ((program_kind > 1.5) && (program_kind < 2.5));
    let glass_program: bool = ((program_kind > 2.5) && (program_kind < 3.5));
    let unlit_program: bool = (program_kind > 3.5);
    let metallic: f32 = select((emission_metallic.w * metallic_roughness_sample.z), 1.0, mirror_program);
    let base_roughness: f32 = max((base_color_roughness.w * metallic_roughness_sample.y), 0.04);
    let roughness: f32 = select(base_roughness, 0.04, (mirror_program || glass_program));
    let base_transmission: f32 = (attenuation_transmission.w * transmission_sample);
    let transmission: f32 = select(base_transmission, 1.0, glass_program);
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
    let occlusion: f32 = mix(1.0, occlusion_sample, normal_occlusion_transmission.z);
    let ambient: vec3<f32> = (((((sampled_base_color * diffuse_weight) + (f0 * (1.0 - (0.5 * roughness)))) + (vec3<f32>(transmission) * (vec3<f32>(1.0) - f0))) * light_color_ambient.w) * occlusion);
    let shaded: vec3<f32> = ((ambient + direct) + sampled_emission);
    return vec4<f32>(select(shaded, sampled_base_color, unlit_program), surface_alpha);
}
