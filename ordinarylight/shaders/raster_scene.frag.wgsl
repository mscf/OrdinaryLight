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
    probe_position_radius: vec4<f32>,
    probe_box_min_mode: vec4<f32>,
    probe_box_max_blend: vec4<f32>,
    environment_rect_secondary: vec4<f32>,
    environment_rotation_log_range_secondary: vec4<f32>,
    probe_position_radius_secondary: vec4<f32>,
    probe_box_min_mode_secondary: vec4<f32>,
    probe_box_max_blend_secondary: vec4<f32>,
}

struct RasterCamera {
    view_projection: mat4x4<f32>,
    position_exposure: vec4<f32>,
    viewport_optics: vec4<f32>,
    optical_diagnostic: vec4<f32>,
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
@group(0) @binding(6) var scene_color: texture_2d<f32>;
@group(0) @binding(7) var scene_depth: texture_depth_2d;
@group(0) @binding(8) var scene_sampler: sampler;
@group(0) @binding(9) var scene_depth_sampler: sampler;
@group(0) @binding(3) var<uniform> camera: RasterCamera;

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
    @location(22) subsurface_uv: vec2<f32>,
    @location(23) object_id: f32
) -> @location(0) vec4<f32> {
    if ((material_index < (-0.5))) {
        return vec4<f32>((base_color + emission), material.w);
    }
    let material_id: u32 = u32((material_index + 0.5));
    let object_tag: f32 = (floor((object_id + 0.5)) + 1.0);
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
    let probe_position_radius: vec4<f32> = materials[material_id].probe_position_radius;
    let probe_box_min_mode: vec4<f32> = materials[material_id].probe_box_min_mode;
    let probe_box_max_blend: vec4<f32> = materials[material_id].probe_box_max_blend;
    let environment_rect_secondary: vec4<f32> = materials[material_id].environment_rect_secondary;
    let environment_rotation_log_range_secondary: vec4<f32> = materials[material_id].environment_rotation_log_range_secondary;
    let probe_position_radius_secondary: vec4<f32> = materials[material_id].probe_position_radius_secondary;
    let probe_box_min_mode_secondary: vec4<f32> = materials[material_id].probe_box_min_mode_secondary;
    let probe_box_max_blend_secondary: vec4<f32> = materials[material_id].probe_box_max_blend_secondary;
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
    let surface_sheen_color: vec3<f32> = hooked.sheen_color;
    let surface_sheen_roughness: f32 = max(0.0, min(1.0, hooked.sheen_roughness));
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
    let sheen_weight: f32 = ((1.0 - surface_sheen_roughness) * pow((1.0 - vdoth), 5.0));
    let sheen: vec3<f32> = (surface_sheen_color * sheen_weight);
    let raw_surface_alpha: f32 = (material.w * base_color_sample.w);
    let masked_surface_alpha: f32 = select(0.0, 1.0, (raw_surface_alpha >= optical.z));
    let surface_alpha: f32 = select(raw_surface_alpha, masked_surface_alpha, ((optical.w > 0.5) && (optical.w < 1.5)));
    let dielectric_ior: f32 = max(ior_distance_program_flags.x, 1.0001);
    let dielectric_f0_ratio: f32 = ((dielectric_ior - 1.0) / (dielectric_ior + 1.0));
    let dielectric_f0: f32 = (dielectric_f0_ratio * dielectric_f0_ratio);
    let f0: vec3<f32> = mix(vec3<f32>(dielectric_f0), surface_base_color, vec3<f32>(surface_metallic));
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
    let direct: vec3<f32> = (((((diffuse + sheen) * base_energy) * light_color_ambient.xyz) * (diffuse_ndotl * visibility)) + ((((specular * base_energy) + clearcoat_specular) * light_color_ambient.xyz) * (ndotl * visibility)));
    let optical_thickness: f32 = (advanced1.w * thickness_sample);
    let absorption_exponent: f32 = (optical_thickness / max(ior_distance_program_flags.y, 1e-06));
    let transmission_tint: vec3<f32> = mix(pow(max(attenuation_transmission.xyz, vec3<f32>(1e-06)), vec3<f32>(absorption_exponent)), surface_base_color, surface_thin_walled);
    let incident: vec3<f32> = (-view);
    let reflected: vec3<f32> = (incident - (surface_normal * (2.0 * (((incident.x * surface_normal.x) + (incident.y * surface_normal.y)) + (incident.z * surface_normal.z)))));
    let raw_refracted: vec3<f32> = refract(incident, surface_normal, (1.0 / max(ior_distance_program_flags.x, 1.0001)));
    let refracted: vec3<f32> = select(raw_refracted, reflected, ((((raw_refracted.x * raw_refracted.x) + (raw_refracted.y * raw_refracted.y)) + (raw_refracted.z * raw_refracted.z)) < 0.0001));
    let proxy_radius: f32 = max((optical_thickness * 0.5), 0.0001);
    let proxy_center: vec3<f32> = (world_position - (surface_normal * proxy_radius));
    let proxy_center_delta: vec3<f32> = (world_position - proxy_center);
    let proxy_exit_distance: f32 = max(((-2.0) * (((refracted.x * proxy_center_delta.x) + (refracted.y * proxy_center_delta.y)) + (refracted.z * proxy_center_delta.z))), 0.0);
    let proxy_exit_position: vec3<f32> = (world_position + (refracted * proxy_exit_distance));
    let proxy_exit_delta: vec3<f32> = (proxy_exit_position - proxy_center);
    let proxy_exit_normal: vec3<f32> = (proxy_exit_delta / max(length(proxy_exit_delta), 1e-06));
    let raw_secondary_refracted: vec3<f32> = refract(refracted, (-proxy_exit_normal), max(ior_distance_program_flags.x, 1.0001));
    let secondary_reflected: vec3<f32> = (refracted - ((-proxy_exit_normal) * (2.0 * (((refracted.x * (-proxy_exit_normal.x)) + (refracted.y * (-proxy_exit_normal.y))) + (refracted.z * (-proxy_exit_normal.z))))));
    let secondary_refracted: vec3<f32> = select(raw_secondary_refracted, secondary_reflected, ((((raw_secondary_refracted.x * raw_secondary_refracted.x) + (raw_secondary_refracted.y * raw_secondary_refracted.y)) + (raw_secondary_refracted.z * raw_secondary_refracted.z)) < 0.0001));
    let closed_refracted: vec3<f32> = select(refracted, secondary_refracted, (surface_thin_walled < 0.5));
    let closed_exit_position: vec3<f32> = select(world_position, proxy_exit_position, (surface_thin_walled < 0.5));
    var probe_reflected: vec3<f32> = reflected;
    var probe_refracted: vec3<f32> = closed_refracted;
    if ((probe_position_radius.w > 0.0)) {
        let reflected_offset: vec3<f32> = (world_position - probe_position_radius.xyz);
        let reflected_b: f32 = (((reflected_offset.x * reflected.x) + (reflected_offset.y * reflected.y)) + (reflected_offset.z * reflected.z));
        let reflected_c: f32 = ((((reflected_offset.x * reflected_offset.x) + (reflected_offset.y * reflected_offset.y)) + (reflected_offset.z * reflected_offset.z)) - (probe_position_radius.w * probe_position_radius.w));
        let reflected_t: f32 = max(0.0, ((-reflected_b) + sqrt(max(0.0, ((reflected_b * reflected_b) - reflected_c)))));
        probe_reflected = ((world_position + (reflected * reflected_t)) - probe_position_radius.xyz);
        probe_reflected = (probe_reflected / max(length(probe_reflected), 1e-06));
        let refracted_offset: vec3<f32> = (closed_exit_position - probe_position_radius.xyz);
        let refracted_b: f32 = (((refracted_offset.x * closed_refracted.x) + (refracted_offset.y * closed_refracted.y)) + (refracted_offset.z * closed_refracted.z));
        let refracted_c: f32 = ((((refracted_offset.x * refracted_offset.x) + (refracted_offset.y * refracted_offset.y)) + (refracted_offset.z * refracted_offset.z)) - (probe_position_radius.w * probe_position_radius.w));
        let refracted_t: f32 = max(0.0, ((-refracted_b) + sqrt(max(0.0, ((refracted_b * refracted_b) - refracted_c)))));
        probe_refracted = ((closed_exit_position + (closed_refracted * refracted_t)) - probe_position_radius.xyz);
        probe_refracted = (probe_refracted / max(length(probe_refracted), 1e-06));
        if ((probe_box_min_mode.w > 0.5)) {
            let reflected_safe: vec3<f32> = vec3<f32>(select(1e-06, reflected.x, (abs(reflected.x) > 1e-06)), select(1e-06, reflected.y, (abs(reflected.y) > 1e-06)), select(1e-06, reflected.z, (abs(reflected.z) > 1e-06)));
            let reflected_to_min: vec3<f32> = ((probe_box_min_mode.xyz - world_position) / reflected_safe);
            let reflected_to_max: vec3<f32> = ((probe_box_max_blend.xyz - world_position) / reflected_safe);
            let reflected_far: vec3<f32> = max(reflected_to_min, reflected_to_max);
            let reflected_box_t: f32 = min(reflected_far.x, min(reflected_far.y, reflected_far.z));
            probe_reflected = ((world_position + (reflected * reflected_box_t)) - probe_position_radius.xyz);
            probe_reflected = (probe_reflected / max(length(probe_reflected), 1e-06));
            let refracted_safe: vec3<f32> = vec3<f32>(select(1e-06, closed_refracted.x, (abs(closed_refracted.x) > 1e-06)), select(1e-06, closed_refracted.y, (abs(closed_refracted.y) > 1e-06)), select(1e-06, closed_refracted.z, (abs(closed_refracted.z) > 1e-06)));
            let refracted_to_min: vec3<f32> = ((probe_box_min_mode.xyz - closed_exit_position) / refracted_safe);
            let refracted_to_max: vec3<f32> = ((probe_box_max_blend.xyz - closed_exit_position) / refracted_safe);
            let refracted_far: vec3<f32> = max(refracted_to_min, refracted_to_max);
            let refracted_box_t: f32 = min(refracted_far.x, min(refracted_far.y, refracted_far.z));
            probe_refracted = ((closed_exit_position + (closed_refracted * refracted_box_t)) - probe_position_radius.xyz);
            probe_refracted = (probe_refracted / max(length(probe_refracted), 1e-06));
        }
    }
    var probe_reflected_secondary: vec3<f32> = reflected;
    var probe_refracted_secondary: vec3<f32> = closed_refracted;
    if ((probe_position_radius_secondary.w > 0.0)) {
        let reflected_offset_secondary: vec3<f32> = (world_position - probe_position_radius_secondary.xyz);
        let reflected_b_secondary: f32 = (((reflected_offset_secondary.x * reflected.x) + (reflected_offset_secondary.y * reflected.y)) + (reflected_offset_secondary.z * reflected.z));
        let reflected_c_secondary: f32 = ((((reflected_offset_secondary.x * reflected_offset_secondary.x) + (reflected_offset_secondary.y * reflected_offset_secondary.y)) + (reflected_offset_secondary.z * reflected_offset_secondary.z)) - (probe_position_radius_secondary.w * probe_position_radius_secondary.w));
        let reflected_sphere_t_secondary: f32 = max(0.0, ((-reflected_b_secondary) + sqrt(max(0.0, ((reflected_b_secondary * reflected_b_secondary) - reflected_c_secondary)))));
        probe_reflected_secondary = ((world_position + (reflected * reflected_sphere_t_secondary)) - probe_position_radius_secondary.xyz);
        probe_reflected_secondary = (probe_reflected_secondary / max(length(probe_reflected_secondary), 1e-06));
        let refracted_offset_secondary: vec3<f32> = (closed_exit_position - probe_position_radius_secondary.xyz);
        let refracted_b_secondary: f32 = (((refracted_offset_secondary.x * closed_refracted.x) + (refracted_offset_secondary.y * closed_refracted.y)) + (refracted_offset_secondary.z * closed_refracted.z));
        let refracted_c_secondary: f32 = ((((refracted_offset_secondary.x * refracted_offset_secondary.x) + (refracted_offset_secondary.y * refracted_offset_secondary.y)) + (refracted_offset_secondary.z * refracted_offset_secondary.z)) - (probe_position_radius_secondary.w * probe_position_radius_secondary.w));
        let refracted_sphere_t_secondary: f32 = max(0.0, ((-refracted_b_secondary) + sqrt(max(0.0, ((refracted_b_secondary * refracted_b_secondary) - refracted_c_secondary)))));
        probe_refracted_secondary = ((closed_exit_position + (closed_refracted * refracted_sphere_t_secondary)) - probe_position_radius_secondary.xyz);
        probe_refracted_secondary = (probe_refracted_secondary / max(length(probe_refracted_secondary), 1e-06));
    }
    if ((probe_box_min_mode_secondary.w > 0.5)) {
        let reflected_safe_secondary: vec3<f32> = vec3<f32>(select(1e-06, reflected.x, (abs(reflected.x) > 1e-06)), select(1e-06, reflected.y, (abs(reflected.y) > 1e-06)), select(1e-06, reflected.z, (abs(reflected.z) > 1e-06)));
        let reflected_min_secondary: vec3<f32> = ((probe_box_min_mode_secondary.xyz - world_position) / reflected_safe_secondary);
        let reflected_max_secondary: vec3<f32> = ((probe_box_max_blend_secondary.xyz - world_position) / reflected_safe_secondary);
        let reflected_far_secondary: vec3<f32> = max(reflected_min_secondary, reflected_max_secondary);
        let reflected_t_secondary: f32 = min(reflected_far_secondary.x, min(reflected_far_secondary.y, reflected_far_secondary.z));
        probe_reflected_secondary = ((world_position + (reflected * reflected_t_secondary)) - probe_position_radius_secondary.xyz);
        probe_reflected_secondary = (probe_reflected_secondary / max(length(probe_reflected_secondary), 1e-06));
        let refracted_safe_secondary: vec3<f32> = vec3<f32>(select(1e-06, closed_refracted.x, (abs(closed_refracted.x) > 1e-06)), select(1e-06, closed_refracted.y, (abs(closed_refracted.y) > 1e-06)), select(1e-06, closed_refracted.z, (abs(closed_refracted.z) > 1e-06)));
        let refracted_min_secondary: vec3<f32> = ((probe_box_min_mode_secondary.xyz - closed_exit_position) / refracted_safe_secondary);
        let refracted_max_secondary: vec3<f32> = ((probe_box_max_blend_secondary.xyz - closed_exit_position) / refracted_safe_secondary);
        let refracted_far_secondary: vec3<f32> = max(refracted_min_secondary, refracted_max_secondary);
        let refracted_t_secondary: f32 = min(refracted_far_secondary.x, min(refracted_far_secondary.y, refracted_far_secondary.z));
        probe_refracted_secondary = ((closed_exit_position + (closed_refracted * refracted_t_secondary)) - probe_position_radius_secondary.xyz);
        probe_refracted_secondary = (probe_refracted_secondary / max(length(probe_refracted_secondary), 1e-06));
    }
    let reflection_uv: vec2<f32> = vec2<f32>(fract((((atan2(probe_reflected.z, probe_reflected.x) / 6.28318531) + 0.5) + (environment_rotation_log_range.x / 6.28318531))), (acos(max((-1.0), min(1.0, probe_reflected.y))) / 3.14159265));
    let refraction_uv: vec2<f32> = vec2<f32>(fract((((atan2(probe_refracted.z, probe_refracted.x) / 6.28318531) + 0.5) + (environment_rotation_log_range.x / 6.28318531))), (acos(max((-1.0), min(1.0, probe_refracted.y))) / 3.14159265));
    let reflection_uv_secondary: vec2<f32> = vec2<f32>(fract((((atan2(probe_reflected_secondary.z, probe_reflected_secondary.x) / 6.28318531) + 0.5) + (environment_rotation_log_range_secondary.x / 6.28318531))), (acos(max((-1.0), min(1.0, probe_reflected_secondary.y))) / 3.14159265));
    let refraction_uv_secondary: vec2<f32> = vec2<f32>(fract((((atan2(probe_refracted_secondary.z, probe_refracted_secondary.x) / 6.28318531) + 0.5) + (environment_rotation_log_range_secondary.x / 6.28318531))), (acos(max((-1.0), min(1.0, probe_refracted_secondary.y))) / 3.14159265));
    let environment_level: f32 = min((surface_roughness * 5.0), 3.0);
    let environment_level_low: f32 = floor(environment_level);
    let environment_level_high: f32 = min((environment_level_low + 1.0), 3.0);
    let environment_level_mix: f32 = fract(environment_level);
    let reflection_uv_low: vec2<f32> = vec2<f32>(((reflection_uv.x + environment_level_low) * 0.25), reflection_uv.y);
    let reflection_uv_high: vec2<f32> = vec2<f32>(((reflection_uv.x + environment_level_high) * 0.25), reflection_uv.y);
    let refraction_uv_low: vec2<f32> = vec2<f32>(((refraction_uv.x + environment_level_low) * 0.25), refraction_uv.y);
    let refraction_uv_high: vec2<f32> = vec2<f32>(((refraction_uv.x + environment_level_high) * 0.25), refraction_uv.y);
    let reflection_uv_secondary_low: vec2<f32> = vec2<f32>(((reflection_uv_secondary.x + environment_level_low) * 0.25), reflection_uv_secondary.y);
    let reflection_uv_secondary_high: vec2<f32> = vec2<f32>(((reflection_uv_secondary.x + environment_level_high) * 0.25), reflection_uv_secondary.y);
    let refraction_uv_secondary_low: vec2<f32> = vec2<f32>(((refraction_uv_secondary.x + environment_level_low) * 0.25), refraction_uv_secondary.y);
    let refraction_uv_secondary_high: vec2<f32> = vec2<f32>(((refraction_uv_secondary.x + environment_level_high) * 0.25), refraction_uv_secondary.y);
    let reflected_encoded_low: vec3<f32> = textureSample(base_color_atlas, base_color_sampler, (environment_rect.xy + (reflection_uv_low * environment_rect.zw))).xyz;
    let reflected_encoded_high: vec3<f32> = textureSample(base_color_atlas, base_color_sampler, (environment_rect.xy + (reflection_uv_high * environment_rect.zw))).xyz;
    let refracted_encoded_low: vec3<f32> = textureSample(base_color_atlas, base_color_sampler, (environment_rect.xy + (refraction_uv_low * environment_rect.zw))).xyz;
    let refracted_encoded_high: vec3<f32> = textureSample(base_color_atlas, base_color_sampler, (environment_rect.xy + (refraction_uv_high * environment_rect.zw))).xyz;
    let reflected_secondary_low: vec3<f32> = textureSample(base_color_atlas, base_color_sampler, (environment_rect_secondary.xy + (reflection_uv_secondary_low * environment_rect_secondary.zw))).xyz;
    let reflected_secondary_high: vec3<f32> = textureSample(base_color_atlas, base_color_sampler, (environment_rect_secondary.xy + (reflection_uv_secondary_high * environment_rect_secondary.zw))).xyz;
    let refracted_secondary_low: vec3<f32> = textureSample(base_color_atlas, base_color_sampler, (environment_rect_secondary.xy + (refraction_uv_secondary_low * environment_rect_secondary.zw))).xyz;
    let refracted_secondary_high: vec3<f32> = textureSample(base_color_atlas, base_color_sampler, (environment_rect_secondary.xy + (refraction_uv_secondary_high * environment_rect_secondary.zw))).xyz;
    let reflected_encoded: vec3<f32> = mix(reflected_encoded_low, reflected_encoded_high, environment_level_mix);
    let refracted_encoded: vec3<f32> = mix(refracted_encoded_low, refracted_encoded_high, environment_level_mix).xyz;
    let reflected_secondary_encoded: vec3<f32> = mix(reflected_secondary_low, reflected_secondary_high, environment_level_mix);
    let refracted_secondary_encoded: vec3<f32> = mix(refracted_secondary_low, refracted_secondary_high, environment_level_mix);
    var reflected_environment: vec3<f32> = (((pow(vec3<f32>(2.0), (reflected_encoded * environment_rotation_log_range.y)) - vec3<f32>(1.0)) * environment_color_intensity.xyz) * environment_color_intensity.w);
    var refracted_environment: vec3<f32> = (((pow(vec3<f32>(2.0), (refracted_encoded * environment_rotation_log_range.y)) - vec3<f32>(1.0)) * environment_color_intensity.xyz) * environment_color_intensity.w);
    let secondary_enabled: f32 = select(0.0, 1.0, (environment_rect_secondary.z > 0.0));
    let secondary_reflected_environment: vec3<f32> = (pow(vec3<f32>(2.0), (reflected_secondary_encoded * environment_rotation_log_range_secondary.y)) - vec3<f32>(1.0));
    let secondary_refracted_environment: vec3<f32> = (pow(vec3<f32>(2.0), (refracted_secondary_encoded * environment_rotation_log_range_secondary.y)) - vec3<f32>(1.0));
    let primary_weight: f32 = select(1.0, probe_box_max_blend.w, (secondary_enabled > 0.5));
    let secondary_weight: f32 = (probe_box_max_blend_secondary.w * secondary_enabled);
    let weight_sum: f32 = max((primary_weight + secondary_weight), 1e-06);
    reflected_environment = (((reflected_environment * primary_weight) + (secondary_reflected_environment * secondary_weight)) / weight_sum);
    refracted_environment = (((refracted_environment * primary_weight) + (secondary_refracted_environment * secondary_weight)) / weight_sum);
    let screen_clip: vec4<f32> = (camera.view_projection * vec4<f32>(world_position, 1.0));
    let screen_ndc: vec3<f32> = (screen_clip.xyz / max(abs(screen_clip.w), 1e-06));
    let screen_uv: vec2<f32> = vec2<f32>(((screen_ndc.x * 0.5) + 0.5), (0.5 - (screen_ndc.y * 0.5)));
    let screen_pass_enabled: f32 = select(0.0, 1.0, ((abs(camera.viewport_optics.z) > 0.5) && (abs(camera.viewport_optics.z) < 1.5)));
    let screen_enabled: f32 = (screen_pass_enabled * max(surface_transmission, surface_metallic));
    let quality_distance: f32 = min((camera.viewport_optics.w / 24.0), 2.0);
    let refraction_world_distance: f32 = ((0.3 + (optical_thickness * 0.18)) * quality_distance);
    var reflection_screen_uv: vec2<f32> = screen_uv;
    var reflection_hit: f32 = 0.0;
    let reflection_origin: vec3<f32> = (world_position + (surface_normal * 0.06));
    let reflection_extent: f32 = max(8.0, (length(view_delta) * 3.0));
    var previous_ray_fraction: f32 = 0.0;
    var previous_depth_delta: f32 = (-1.0);
    var diagnostic_depth_delta: f32 = (-1.0);
    var diagnostic_ray_step: f32 = 0.0;
    var diagnostic_confidence: f32 = 0.0;
    var diagnostic_depth_trace: f32 = 0.0;
    for (var ray_step: i32 = 1; ray_step < 25; ray_step += 1) {
        let ray_fraction: f32 = (f32(ray_step) / 24.0);
        let ray_distance: f32 = (0.12 + ((ray_fraction * ray_fraction) * reflection_extent));
        let ray_world: vec3<f32> = (reflection_origin + (reflected * ray_distance));
        let ray_clip: vec4<f32> = (camera.view_projection * vec4<f32>(ray_world, 1.0));
        if ((ray_clip.w <= 1e-06)) {
            break;
        }
        let ray_ndc: vec3<f32> = (ray_clip.xyz / ray_clip.w);
        let ray_uv: vec2<f32> = vec2<f32>(((ray_ndc.x * 0.5) + 0.5), (0.5 - (ray_ndc.y * 0.5)));
        if (((((ray_uv.x <= 0.001) || (ray_uv.x >= 0.999)) || (ray_uv.y <= 0.001)) || (ray_uv.y >= 0.999))) {
            break;
        }
        let sampled_depth: f32 = textureSampleLevel(scene_depth, scene_depth_sampler, ray_uv, 0);
        diagnostic_depth_trace = (diagnostic_depth_trace + (sampled_depth * ray_fraction));
        let depth_delta: f32 = (ray_ndc.z - sampled_depth);
        diagnostic_depth_delta = depth_delta;
        diagnostic_ray_step = ray_fraction;
        let depth_thickness: f32 = (8e-05 + (ray_fraction * 0.00042));
        if (((previous_depth_delta <= 1e-05) && (depth_delta > 1e-05))) {
            var lower_fraction: f32 = previous_ray_fraction;
            var upper_fraction: f32 = ray_fraction;
            var refined_uv: vec2<f32> = ray_uv;
            var refined_delta: f32 = depth_delta;
            for (var refine_step: i32 = 0; refine_step < 4; refine_step += 1) {
                let middle_fraction: f32 = ((lower_fraction + upper_fraction) * 0.5);
                let middle_distance: f32 = (0.12 + ((middle_fraction * middle_fraction) * reflection_extent));
                let middle_world: vec3<f32> = (reflection_origin + (reflected * middle_distance));
                let middle_clip: vec4<f32> = (camera.view_projection * vec4<f32>(middle_world, 1.0));
                let middle_ndc: vec3<f32> = (middle_clip.xyz / max(middle_clip.w, 1e-06));
                let middle_uv: vec2<f32> = vec2<f32>(((middle_ndc.x * 0.5) + 0.5), (0.5 - (middle_ndc.y * 0.5)));
                let middle_depth: f32 = textureSampleLevel(scene_depth, scene_depth_sampler, middle_uv, 0);
                let middle_delta: f32 = (middle_ndc.z - middle_depth);
                if ((middle_delta > 1e-05)) {
                    upper_fraction = middle_fraction;
                    refined_uv = middle_uv;
                    refined_delta = middle_delta;
                } else {
                    lower_fraction = middle_fraction;
                }
            }
            let refined_thickness: f32 = (6e-05 + (upper_fraction * 0.0003));
            let edge_distance: f32 = min(min(refined_uv.x, (1.0 - refined_uv.x)), min(refined_uv.y, (1.0 - refined_uv.y)));
            let edge_confidence: f32 = clamp(((edge_distance - 0.002) / 0.018), 0.0, 1.0);
            if ((refined_delta < refined_thickness)) {
                let candidate: vec4<f32> = textureSampleLevel(scene_color, scene_sampler, refined_uv, 0.0);
                let different_object: f32 = select(0.0, 1.0, (abs((candidate.w - object_tag)) > 0.25));
                let hit_texel: vec2<f32> = vec2<f32>((1.0 / max(camera.viewport_optics.x, 1.0)), (1.0 / max(camera.viewport_optics.y, 1.0)));
                let hit_depth: f32 = textureSampleLevel(scene_depth, scene_depth_sampler, refined_uv, 0);
                let depth_left: f32 = textureSampleLevel(scene_depth, scene_depth_sampler, (refined_uv - vec2<f32>(hit_texel.x, 0.0)), 0);
                let depth_right: f32 = textureSampleLevel(scene_depth, scene_depth_sampler, (refined_uv + vec2<f32>(hit_texel.x, 0.0)), 0);
                let depth_down: f32 = textureSampleLevel(scene_depth, scene_depth_sampler, (refined_uv - vec2<f32>(0.0, hit_texel.y)), 0);
                let depth_up: f32 = textureSampleLevel(scene_depth, scene_depth_sampler, (refined_uv + vec2<f32>(0.0, hit_texel.y)), 0);
                let depth_spread: f32 = max(max(abs((depth_left - hit_depth)), abs((depth_right - hit_depth))), max(abs((depth_down - hit_depth)), abs((depth_up - hit_depth))));
                let depth_confidence: f32 = clamp((1.0 - (depth_spread / 0.00075)), 0.0, 1.0);
                reflection_screen_uv = refined_uv;
                reflection_hit = ((edge_confidence * different_object) * depth_confidence);
                diagnostic_confidence = (edge_confidence * depth_confidence);
            }
            if ((camera.optical_diagnostic.x < 5.5)) {
                break;
            }
        }
        previous_ray_fraction = ray_fraction;
        previous_depth_delta = depth_delta;
    }
    let refraction_origin: vec3<f32> = (closed_exit_position + (closed_refracted * 0.03));
    let refraction_extent: f32 = max(8.0, (length(view_delta) * 3.0));
    var refraction_screen_uv: vec2<f32> = screen_uv;
    var refraction_hit: f32 = 0.0;
    var previous_refraction_fraction: f32 = 0.0;
    var previous_refraction_delta: f32 = (-1.0);
    for (var refraction_step: i32 = 1; refraction_step < 25; refraction_step += 1) {
        let refraction_fraction: f32 = (f32(refraction_step) / 24.0);
        let refraction_distance: f32 = (0.08 + ((refraction_fraction * refraction_fraction) * refraction_extent));
        let refraction_world: vec3<f32> = (refraction_origin + (closed_refracted * refraction_distance));
        let refraction_clip: vec4<f32> = (camera.view_projection * vec4<f32>(refraction_world, 1.0));
        if ((refraction_clip.w <= 1e-06)) {
            break;
        }
        let refraction_ndc: vec3<f32> = (refraction_clip.xyz / refraction_clip.w);
        let refraction_uv_candidate: vec2<f32> = vec2<f32>(((refraction_ndc.x * 0.5) + 0.5), (0.5 - (refraction_ndc.y * 0.5)));
        if (((((refraction_uv_candidate.x <= 0.001) || (refraction_uv_candidate.x >= 0.999)) || (refraction_uv_candidate.y <= 0.001)) || (refraction_uv_candidate.y >= 0.999))) {
            break;
        }
        let refraction_scene_depth: f32 = textureSampleLevel(scene_depth, scene_depth_sampler, refraction_uv_candidate, 0);
        let refraction_delta: f32 = (refraction_ndc.z - refraction_scene_depth);
        if (((previous_refraction_delta <= 1e-05) && (refraction_delta > 1e-05))) {
            var lower_refraction_fraction: f32 = previous_refraction_fraction;
            var upper_refraction_fraction: f32 = refraction_fraction;
            var refined_refraction_uv: vec2<f32> = refraction_uv_candidate;
            var refined_refraction_delta: f32 = refraction_delta;
            for (var refraction_refine_step: i32 = 0; refraction_refine_step < 4; refraction_refine_step += 1) {
                let middle_refraction_fraction: f32 = ((lower_refraction_fraction + upper_refraction_fraction) * 0.5);
                let middle_refraction_distance: f32 = (0.08 + ((middle_refraction_fraction * middle_refraction_fraction) * refraction_extent));
                let middle_refraction_world: vec3<f32> = (refraction_origin + (closed_refracted * middle_refraction_distance));
                let middle_refraction_clip: vec4<f32> = (camera.view_projection * vec4<f32>(middle_refraction_world, 1.0));
                let middle_refraction_ndc: vec3<f32> = (middle_refraction_clip.xyz / max(middle_refraction_clip.w, 1e-06));
                let middle_refraction_uv: vec2<f32> = vec2<f32>(((middle_refraction_ndc.x * 0.5) + 0.5), (0.5 - (middle_refraction_ndc.y * 0.5)));
                let middle_refraction_depth: f32 = textureSampleLevel(scene_depth, scene_depth_sampler, middle_refraction_uv, 0);
                let middle_refraction_delta: f32 = (middle_refraction_ndc.z - middle_refraction_depth);
                if ((middle_refraction_delta > 1e-05)) {
                    upper_refraction_fraction = middle_refraction_fraction;
                    refined_refraction_uv = middle_refraction_uv;
                    refined_refraction_delta = middle_refraction_delta;
                } else {
                    lower_refraction_fraction = middle_refraction_fraction;
                }
            }
            let refraction_thickness: f32 = (6e-05 + (upper_refraction_fraction * 0.0003));
            if ((refined_refraction_delta < refraction_thickness)) {
                refraction_screen_uv = refined_refraction_uv;
                refraction_hit = 1.0;
            }
            break;
        }
        previous_refraction_fraction = refraction_fraction;
        previous_refraction_delta = refraction_delta;
    }
    let reflection_texel: vec2<f32> = vec2<f32>((1.0 / max(camera.viewport_optics.x, 1.0)), (1.0 / max(camera.viewport_optics.y, 1.0)));
    let reflection_radius: f32 = (1.0 + ((surface_roughness * surface_roughness) * 10.0));
    let reflection_offset: vec2<f32> = (reflection_texel * reflection_radius);
    let screen_reflected: vec3<f32> = ((((((textureSampleLevel(scene_color, scene_sampler, reflection_screen_uv, 0.0).xyz * 4.0) + textureSampleLevel(scene_color, scene_sampler, (reflection_screen_uv + vec2<f32>(reflection_offset.x, 0.0)), 0.0).xyz) + textureSampleLevel(scene_color, scene_sampler, (reflection_screen_uv - vec2<f32>(reflection_offset.x, 0.0)), 0.0).xyz) + textureSampleLevel(scene_color, scene_sampler, (reflection_screen_uv + vec2<f32>(0.0, reflection_offset.y)), 0.0).xyz) + textureSampleLevel(scene_color, scene_sampler, (reflection_screen_uv - vec2<f32>(0.0, reflection_offset.y)), 0.0).xyz) * 0.125);
    let screen_refracted: vec3<f32> = textureSampleLevel(scene_color, scene_sampler, refraction_screen_uv, 0.0).xyz;
    let reflected_source: vec3<f32> = mix(reflected_environment, screen_reflected, ((reflection_hit * screen_enabled) * (1.0 - surface_roughness)));
    let refraction_edge_distance: f32 = min(min(refraction_screen_uv.x, (1.0 - refraction_screen_uv.x)), min(refraction_screen_uv.y, (1.0 - refraction_screen_uv.y)));
    let refraction_edge_confidence: f32 = max(0.0, min(1.0, (refraction_edge_distance * 16.0)));
    let refraction_angle_confidence: f32 = max(0.0, min(1.0, ((ndotv - 0.08) * 2.5)));
    let refraction_confidence: f32 = ((refraction_hit * refraction_edge_confidence) * refraction_angle_confidence);
    let refracted_source: vec3<f32> = mix(refracted_environment, screen_refracted, ((refraction_confidence * screen_enabled) * (1.0 - surface_roughness)));
    let environment_enabled: f32 = environment_rotation_log_range.z;
    let ambient: vec3<f32> = (((((surface_base_color * diffuse_weight) + (f0 * (1.0 - (0.5 * surface_roughness)))) + ((transmission_tint * surface_transmission) * (vec3<f32>(1.0) - f0))) * light_color_ambient.w) * surface_occlusion);
    let base_shaded: vec3<f32> = ((ambient + direct) + surface_emission);
    let transmitted_shaded: vec3<f32> = mix(base_shaded, (refracted_source * transmission_tint), (surface_transmission * environment_enabled));
    let shaded: vec3<f32> = (transmitted_shaded + ((reflected_source * fresnel) * environment_enabled));
    let prepass_alpha: f32 = select(surface_alpha, object_tag, (abs(camera.viewport_optics.z) > 1.5));
    let result: vec4<f32> = vec4<f32>(select(shaded, surface_base_color, unlit_program), prepass_alpha);
    let diagnostic_mode: f32 = camera.optical_diagnostic.x;
    if (((diagnostic_mode > 0.5) && (diagnostic_mode < 1.5))) {
        return vec4<f32>(vec3<f32>(reflection_hit), 1.0);
    }
    if (((diagnostic_mode > 1.5) && (diagnostic_mode < 2.5))) {
        return vec4<f32>(reflection_screen_uv, diagnostic_ray_step, 1.0);
    }
    if (((diagnostic_mode > 2.5) && (diagnostic_mode < 3.5))) {
        return vec4<f32>(diagnostic_depth_delta, abs(diagnostic_depth_delta), diagnostic_ray_step, 1.0);
    }
    if (((diagnostic_mode > 3.5) && (diagnostic_mode < 4.5))) {
        return vec4<f32>(diagnostic_confidence, reflection_hit, diagnostic_ray_step, 1.0);
    }
    if (((diagnostic_mode > 4.5) && (diagnostic_mode < 5.5))) {
        return vec4<f32>(object_tag, material_index, 0.0, 1.0);
    }
    if (((diagnostic_mode > 5.5) && (diagnostic_mode < 6.5))) {
        return vec4<f32>(diagnostic_depth_trace, fract(diagnostic_depth_trace), diagnostic_ray_step, 1.0);
    }
    if (((diagnostic_mode > 6.5) && (diagnostic_mode < 7.5))) {
        return vec4<f32>(vec3<f32>(refraction_hit), 1.0);
    }
    if (((diagnostic_mode > 7.5) && (diagnostic_mode < 8.5))) {
        return vec4<f32>(refraction_screen_uv, refraction_hit, 1.0);
    }
    if ((diagnostic_mode > 8.5)) {
        return vec4<f32>(refracted_source, 1.0);
    }
    return result;
}
