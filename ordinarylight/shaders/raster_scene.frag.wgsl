struct RasterMaterial {
    base_color_roughness: vec4<f32>,
    emission_metallic: vec4<f32>,
    attenuation_transmission: vec4<f32>,
    ior_distance_program_flags: vec4<f32>,
    texture_indices: vec4<f32>,
    normal_occlusion_transmission: vec4<f32>,
    advanced0: vec4<f32>,
    advanced1: vec4<f32>,
    sheen_color: vec4<f32>,
    subsurface_color: vec4<f32>,
    advanced_texture_indices: vec4<f32>,
    optical: vec4<f32>,
    environment_rect: vec4<f32>,
    environment_color_intensity: vec4<f32>,
    environment_rotation_log_range: vec4<f32>,
}

struct SurfaceParameters {
    base_color: vec3<f32>,
    emission: vec3<f32>,
    normal: vec3<f32>,
    metallic: f32,
    roughness: f32,
    transmission: f32,
    occlusion: f32,
    clearcoat: f32,
    clearcoat_roughness: f32,
    sheen_color: vec3<f32>,
    sheen_roughness: f32,
    anisotropy: f32,
    thin_walled: f32,
    subsurface: f32,
    subsurface_color: vec3<f32>,
    subsurface_radius: f32,
}

struct SurfaceContext {
    uv: vec2<f32>,
    normal: vec3<f32>,
    view_direction: vec3<f32>,
    program_id: f32,
}

@group(0) @binding(0) var base_color_atlas: texture_2d<f32>;
@group(0) @binding(1) var base_color_sampler: sampler;
@group(0) @binding(2) var shadow_map: texture_depth_2d;
@group(0) @binding(4) var shadow_sampler: sampler_comparison;
@group(0) @binding(5) var<storage, read> materials: array<RasterMaterial>;

fn blend_surface_parameters(base: SurfaceParameters, layer: SurfaceParameters, weight: f32) -> SurfaceParameters {
    let amount: f32 = max(0.0, min(1.0, weight));
    return SurfaceParameters(mix(base.base_color, layer.base_color, amount), (base.emission + (layer.emission * amount)), normalize(mix(base.normal, layer.normal, amount)), mix(base.metallic, layer.metallic, amount), mix(base.roughness, layer.roughness, amount), mix(base.transmission, layer.transmission, amount), mix(base.occlusion, layer.occlusion, amount), mix(base.clearcoat, layer.clearcoat, amount), mix(base.clearcoat_roughness, layer.clearcoat_roughness, amount), mix(base.sheen_color, layer.sheen_color, amount), mix(base.sheen_roughness, layer.sheen_roughness, amount), mix(base.anisotropy, layer.anisotropy, amount), mix(base.thin_walled, layer.thin_walled, amount), mix(base.subsurface, layer.subsurface, amount), mix(base.subsurface_color, layer.subsurface_color, amount), mix(base.subsurface_radius, layer.subsurface_radius, amount));
}

fn ordinarylight_material_modifier(surface: SurfaceParameters, context: SurfaceContext) -> SurfaceParameters {
    return surface;
}

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
    @location(17) material_index: f32,
    @location(18) thickness_uv: vec2<f32>,
    @location(19) clearcoat_uv: vec2<f32>,
    @location(20) sheen_uv: vec2<f32>,
    @location(21) anisotropy_uv: vec2<f32>,
    @location(22) subsurface_uv: vec2<f32>
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
    let advanced0: vec4<f32> = materials[material_id].advanced0;
    let advanced1: vec4<f32> = materials[material_id].advanced1;
    let advanced_sheen_color: vec3<f32> = materials[material_id].sheen_color.xyz;
    let advanced_subsurface_color: vec3<f32> = materials[material_id].subsurface_color.xyz;
    let advanced_texture_indices: vec4<f32> = materials[material_id].advanced_texture_indices;
    let optical: vec4<f32> = materials[material_id].optical;
    let environment_rect: vec4<f32> = materials[material_id].environment_rect;
    let environment_color_intensity: vec4<f32> = materials[material_id].environment_color_intensity;
    let environment_rotation_log_range: vec4<f32> = materials[material_id].environment_rotation_log_range;
    let base_color_sample: vec4<f32> = textureSample(base_color_atlas, base_color_sampler, base_color_uv);
    let sampled_base_color: vec3<f32> = (base_color_roughness.xyz * base_color_sample.xyz);
    let metallic_roughness_sample: vec4<f32> = textureSample(base_color_atlas, base_color_sampler, metallic_roughness_uv);
    let sampled_emission: vec3<f32> = (emission_metallic.xyz * textureSample(base_color_atlas, base_color_sampler, emissive_uv).xyz);
    let normal_sample: vec3<f32> = ((textureSample(base_color_atlas, base_color_sampler, normal_uv).xyz * 2.0) - vec3<f32>(1.0));
    let occlusion_sample: f32 = textureSample(base_color_atlas, base_color_sampler, occlusion_uv).x;
    let transmission_sample: f32 = textureSample(base_color_atlas, base_color_sampler, transmission_uv).x;
    let thickness_sample: f32 = select(1.0, textureSample(base_color_atlas, base_color_sampler, thickness_uv).x, (optical.x >= 0.0));
    let clearcoat_sample: f32 = select(1.0, textureSample(base_color_atlas, base_color_sampler, clearcoat_uv).x, (advanced_texture_indices.x >= 0.0));
    let sheen_sample: vec3<f32> = select(vec3<f32>(1.0), textureSample(base_color_atlas, base_color_sampler, sheen_uv).xyz, (advanced_texture_indices.y >= 0.0));
    let anisotropy_sample: f32 = select(1.0, textureSample(base_color_atlas, base_color_sampler, anisotropy_uv).x, (advanced_texture_indices.z >= 0.0));
    let subsurface_sample: f32 = select(1.0, textureSample(base_color_atlas, base_color_sampler, subsurface_uv).x, (advanced_texture_indices.w >= 0.0));
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
    let program_kind: f32 = floor(ior_distance_program_flags.w);
    let mirror_program: bool = ((program_kind > 1.5) && (program_kind < 2.5));
    let glass_program: bool = ((program_kind > 2.5) && (program_kind < 3.5));
    let unlit_program: bool = (program_kind > 3.5);
    let metallic: f32 = select((emission_metallic.w * metallic_roughness_sample.z), 1.0, mirror_program);
    let base_roughness: f32 = max((base_color_roughness.w * metallic_roughness_sample.y), 0.04);
    let roughness: f32 = select(base_roughness, 0.04, (mirror_program || glass_program));
    let base_transmission: f32 = (attenuation_transmission.w * transmission_sample);
    let transmission: f32 = select(base_transmission, 1.0, glass_program);
    let occlusion: f32 = mix(1.0, occlusion_sample, normal_occlusion_transmission.z);
    let hooked: SurfaceParameters = ordinarylight_material_modifier(SurfaceParameters(sampled_base_color, sampled_emission, normal, metallic, roughness, transmission, occlusion, (advanced0.x * clearcoat_sample), advanced0.y, (advanced_sheen_color * sheen_sample), advanced0.z, (advanced0.w * anisotropy_sample), advanced1.z, (advanced1.x * subsurface_sample), advanced_subsurface_color, advanced1.y), SurfaceContext(base_color_uv, normal, view, ior_distance_program_flags.z));
    let surface_base_color: vec3<f32> = hooked.base_color;
    let surface_emission: vec3<f32> = hooked.emission;
    let surface_normal: vec3<f32> = (hooked.normal / max(length(hooked.normal), 1e-06));
    let surface_metallic: f32 = max(0.0, min(1.0, hooked.metallic));
    let surface_roughness: f32 = max(0.04, min(1.0, hooked.roughness));
    let surface_transmission: f32 = max(0.0, min(1.0, hooked.transmission));
    let surface_occlusion: f32 = max(0.0, min(1.0, hooked.occlusion));
    let surface_clearcoat: f32 = max(0.0, min(1.0, hooked.clearcoat));
    let surface_clearcoat_roughness: f32 = max(0.04, min(1.0, hooked.clearcoat_roughness));
    let surface_sheen: vec3<f32> = (hooked.sheen_color * (1.0 - max(0.0, min(1.0, hooked.sheen_roughness))));
    let surface_anisotropy: f32 = max((-1.0), min(1.0, hooked.anisotropy));
    let surface_thin_walled: f32 = max(0.0, min(1.0, hooked.thin_walled));
    let surface_subsurface: f32 = max(0.0, min(1.0, hooked.subsurface));
    let surface_subsurface_radius: f32 = max(0.0, min(1.0, hooked.subsurface_radius));
    let light_delta: vec3<f32> = (light_position_type.xyz - world_position);
    let point_distance: f32 = max(length(light_delta), 1e-06);
    let point_incoming: vec3<f32> = (light_delta / point_distance);
    let direction: vec3<f32> = (-light_position_type.xyz);
    let directional_incoming: vec3<f32> = (direction / max(length(direction), 1e-06));
    let incoming: vec3<f32> = select(point_incoming, directional_incoming, (light_position_type.w > 0.5));
    let distance: f32 = select(point_distance, 1.0, (light_position_type.w > 0.5));
    let half_delta: vec3<f32> = (incoming + view);
    let half_vector: vec3<f32> = (half_delta / max(length(half_delta), 1e-06));
    let ndotl: f32 = max((((surface_normal.x * incoming.x) + (surface_normal.y * incoming.y)) + (surface_normal.z * incoming.z)), 0.0);
    let receiver_bias: f32 = max(2e-05, ((1.0 - ndotl) * 0.0001));
    let pcf_visibility: f32 = textureSampleCompare(shadow_map, shadow_sampler, projected_shadow.xy, (projected_shadow.z - receiver_bias));
    let shadow_map_visibility: f32 = select(pcf_visibility, 1.0, (abs(shadow_coordinate.w) < 1e-06));
    let ndotv: f32 = max((((surface_normal.x * view.x) + (surface_normal.y * view.y)) + (surface_normal.z * view.z)), 0.0);
    let ndoth: f32 = max((((surface_normal.x * half_vector.x) + (surface_normal.y * half_vector.y)) + (surface_normal.z * half_vector.z)), 0.0);
    let vdoth: f32 = max((((view.x * half_vector.x) + (view.y * half_vector.y)) + (view.z * half_vector.z)), 0.0);
    let raw_surface_alpha: f32 = (material.w * base_color_sample.w);
    let masked_surface_alpha: f32 = select(0.0, 1.0, (raw_surface_alpha >= optical.z));
    let surface_alpha: f32 = select(raw_surface_alpha, masked_surface_alpha, ((optical.w > 0.5) && (optical.w < 1.5)));
    let f0: vec3<f32> = mix(vec3<f32>(0.04), surface_base_color, vec3<f32>(surface_metallic));
    let fresnel: vec3<f32> = (f0 + ((vec3<f32>(1.0) - f0) * pow(vec3<f32>((1.0 - vdoth)), vec3<f32>(5.0))));
    let alpha: f32 = (surface_roughness * surface_roughness);
    let alpha_x: f32 = max(0.02, (alpha * (1.0 - (0.7 * surface_anisotropy))));
    let alpha_y: f32 = max(0.02, (alpha * (1.0 + (0.7 * surface_anisotropy))));
    let tangent_dot_half: f32 = (((tangent.x * half_vector.x) + (tangent.y * half_vector.y)) + (tangent.z * half_vector.z));
    let bitangent_dot_half: f32 = (((bitangent.x * half_vector.x) + (bitangent.y * half_vector.y)) + (bitangent.z * half_vector.z));
    let anisotropic_denominator: f32 = ((((tangent_dot_half * tangent_dot_half) / (alpha_x * alpha_x)) + ((bitangent_dot_half * bitangent_dot_half) / (alpha_y * alpha_y))) + (ndoth * ndoth));
    let distribution: f32 = (1.0 / max(((((3.14159265 * alpha_x) * alpha_y) * anisotropic_denominator) * anisotropic_denominator), 1e-06));
    let k: f32 = (((surface_roughness + 1.0) * (surface_roughness + 1.0)) / 8.0);
    let geometry_v: f32 = (ndotv / max(((ndotv * (1.0 - k)) + k), 1e-06));
    let geometry_l: f32 = (ndotl / max(((ndotl * (1.0 - k)) + k), 1e-06));
    let specular: vec3<f32> = ((((distribution * geometry_v) * geometry_l) * fresnel) / max(((4.0 * ndotv) * ndotl), 1e-06));
    let coat_alpha: f32 = (surface_clearcoat_roughness * surface_clearcoat_roughness);
    let coat_alpha2: f32 = (coat_alpha * coat_alpha);
    let coat_denominator: f32 = (((ndoth * ndoth) * (coat_alpha2 - 1.0)) + 1.0);
    let coat_distribution: f32 = (coat_alpha2 / max(((3.14159265 * coat_denominator) * coat_denominator), 1e-06));
    let coat_fresnel: f32 = (0.04 + (0.96 * pow((1.0 - vdoth), 5.0)));
    let coat_k: f32 = (((surface_clearcoat_roughness + 1.0) * (surface_clearcoat_roughness + 1.0)) / 8.0);
    let coat_geometry_v: f32 = (ndotv / max(((ndotv * (1.0 - coat_k)) + coat_k), 1e-06));
    let coat_geometry_l: f32 = (ndotl / max(((ndotl * (1.0 - coat_k)) + coat_k), 1e-06));
    let clearcoat_specular: vec3<f32> = vec3<f32>((((((surface_clearcoat * coat_distribution) * coat_geometry_v) * coat_geometry_l) * coat_fresnel) / max(((4.0 * ndotv) * ndotl), 1e-06)));
    let base_energy: f32 = (1.0 - (surface_clearcoat * coat_fresnel));
    let diffuse_weight: f32 = ((1.0 - surface_metallic) * (1.0 - surface_transmission));
    let diffuse_base: vec3<f32> = ((((vec3<f32>(1.0) - fresnel) * surface_base_color) * diffuse_weight) / 3.14159265);
    let diffuse: vec3<f32> = mix(diffuse_base, (diffuse_base * hooked.subsurface_color), surface_subsurface);
    let wrapped_ndotl: f32 = max(0.0, ((ndotl + surface_subsurface_radius) / (1.0 + surface_subsurface_radius)));
    let diffuse_ndotl: f32 = mix(ndotl, wrapped_ndotl, surface_subsurface);
    let attenuation: f32 = (1.0 / (distance * distance));
    let visibility: f32 = ((attenuation * shadow_visibility) * shadow_map_visibility);
    let direct: vec3<f32> = (((((diffuse + surface_sheen) * base_energy) * light_color_ambient.xyz) * (diffuse_ndotl * visibility)) + ((((specular * base_energy) + clearcoat_specular) * light_color_ambient.xyz) * (ndotl * visibility)));
    let optical_thickness: f32 = (advanced1.w * thickness_sample);
    let absorption_exponent: f32 = (optical_thickness / max(ior_distance_program_flags.y, 1e-06));
    let transmission_tint: vec3<f32> = mix(pow(max(attenuation_transmission.xyz, vec3<f32>(1e-06)), vec3<f32>(absorption_exponent)), surface_base_color, surface_thin_walled);
    let incident: vec3<f32> = (-view);
    let reflected: vec3<f32> = (incident - (surface_normal * (2.0 * (((incident.x * surface_normal.x) + (incident.y * surface_normal.y)) + (incident.z * surface_normal.z)))));
    let raw_refracted: vec3<f32> = refract(incident, surface_normal, (1.0 / max(ior_distance_program_flags.x, 1.0001)));
    let refracted: vec3<f32> = select(raw_refracted, reflected, ((((raw_refracted.x * raw_refracted.x) + (raw_refracted.y * raw_refracted.y)) + (raw_refracted.z * raw_refracted.z)) < 0.0001));
    let reflection_uv: vec2<f32> = vec2<f32>(fract((((atan2(reflected.z, reflected.x) / 6.28318531) + 0.5) + (environment_rotation_log_range.x / 6.28318531))), (acos(max((-1.0), min(1.0, reflected.y))) / 3.14159265));
    let refraction_uv: vec2<f32> = vec2<f32>(fract((((atan2(refracted.z, refracted.x) / 6.28318531) + 0.5) + (environment_rotation_log_range.x / 6.28318531))), (acos(max((-1.0), min(1.0, refracted.y))) / 3.14159265));
    let environment_level: f32 = min((surface_roughness * 5.0), 3.0);
    let environment_level_low: f32 = floor(environment_level);
    let environment_level_high: f32 = min((environment_level_low + 1.0), 3.0);
    let environment_level_mix: f32 = fract(environment_level);
    let reflection_uv_low: vec2<f32> = vec2<f32>(((reflection_uv.x + environment_level_low) * 0.25), reflection_uv.y);
    let reflection_uv_high: vec2<f32> = vec2<f32>(((reflection_uv.x + environment_level_high) * 0.25), reflection_uv.y);
    let refraction_uv_low: vec2<f32> = vec2<f32>(((refraction_uv.x + environment_level_low) * 0.25), refraction_uv.y);
    let refraction_uv_high: vec2<f32> = vec2<f32>(((refraction_uv.x + environment_level_high) * 0.25), refraction_uv.y);
    let reflected_encoded_low: vec3<f32> = textureSample(base_color_atlas, base_color_sampler, (environment_rect.xy + (reflection_uv_low * environment_rect.zw))).xyz;
    let reflected_encoded_high: vec3<f32> = textureSample(base_color_atlas, base_color_sampler, (environment_rect.xy + (reflection_uv_high * environment_rect.zw))).xyz;
    let refracted_encoded_low: vec3<f32> = textureSample(base_color_atlas, base_color_sampler, (environment_rect.xy + (refraction_uv_low * environment_rect.zw))).xyz;
    let refracted_encoded_high: vec3<f32> = textureSample(base_color_atlas, base_color_sampler, (environment_rect.xy + (refraction_uv_high * environment_rect.zw))).xyz;
    let reflected_encoded: vec3<f32> = mix(reflected_encoded_low, reflected_encoded_high, environment_level_mix);
    let refracted_encoded: vec3<f32> = mix(refracted_encoded_low, refracted_encoded_high, environment_level_mix).xyz;
    let reflected_environment: vec3<f32> = (((pow(vec3<f32>(2.0), (reflected_encoded * environment_rotation_log_range.y)) - vec3<f32>(1.0)) * environment_color_intensity.xyz) * environment_color_intensity.w);
    let refracted_environment: vec3<f32> = (((pow(vec3<f32>(2.0), (refracted_encoded * environment_rotation_log_range.y)) - vec3<f32>(1.0)) * environment_color_intensity.xyz) * environment_color_intensity.w);
    let environment_enabled: f32 = environment_rotation_log_range.z;
    let ambient: vec3<f32> = (((((surface_base_color * diffuse_weight) + (f0 * (1.0 - (0.5 * surface_roughness)))) + ((transmission_tint * surface_transmission) * (vec3<f32>(1.0) - f0))) * light_color_ambient.w) * surface_occlusion);
    let base_shaded: vec3<f32> = ((ambient + direct) + surface_emission);
    let transmitted_shaded: vec3<f32> = mix(base_shaded, (refracted_environment * transmission_tint), (surface_transmission * environment_enabled));
    let shaded: vec3<f32> = (transmitted_shaded + ((reflected_environment * fresnel) * environment_enabled));
    return vec4<f32>(select(shaded, surface_base_color, unlit_program), surface_alpha);
}
