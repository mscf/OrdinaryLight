#include "wavefront_srgb.glsl"

#ifndef WAVE_ORDINARYSHADE_TEXTURE_APPLICATION
#define WAVE_ORDINARYSHADE_TEXTURE_APPLICATION 1
#endif
#ifndef WAVE_ORDINARYSHADE_PBR
#define WAVE_ORDINARYSHADE_PBR 1
#endif
#ifndef WAVE_ORDINARYSHADE_ANALYTIC_LIGHTS
#define WAVE_ORDINARYSHADE_ANALYTIC_LIGHTS 1
#endif
#ifndef WAVE_ORDINARYSHADE_AREA_LIGHTS
#define WAVE_ORDINARYSHADE_AREA_LIGHTS 1
#endif
#ifndef WAVE_ORDINARYSHADE_ENVIRONMENT_LIGHTS
#define WAVE_ORDINARYSHADE_ENVIRONMENT_LIGHTS 1
#endif
#ifndef WAVE_ORDINARYSHADE_UNIFIED_NEE
#define WAVE_ORDINARYSHADE_UNIFIED_NEE 1
#endif
#ifndef WAVE_ORDINARYSHADE_EMISSIVE_MIS
#define WAVE_ORDINARYSHADE_EMISSIVE_MIS 1
#endif
#ifndef WAVE_ORDINARYSHADE_SECONDARY_TRANSPORT
#define WAVE_ORDINARYSHADE_SECONDARY_TRANSPORT 1
#endif
#ifndef WAVE_ORDINARYSHADE_SECONDARY_TRANSMISSION
#define WAVE_ORDINARYSHADE_SECONDARY_TRANSMISSION 1
#endif
#ifndef WAVE_ORDINARYSHADE_SECONDARY_SURFACE
#define WAVE_ORDINARYSHADE_SECONDARY_SURFACE 1
#endif
#ifndef WAVE_ORDINARYSHADE_SECONDARY_CONTROL
#define WAVE_ORDINARYSHADE_SECONDARY_CONTROL 1
#endif
#ifndef WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION
#define WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION 1
#endif
#if WAVE_ORDINARYSHADE_TEXTURE_APPLICATION || WAVE_ORDINARYSHADE_PBR || \
        WAVE_ORDINARYSHADE_ANALYTIC_LIGHTS || WAVE_ORDINARYSHADE_AREA_LIGHTS || \
        WAVE_ORDINARYSHADE_ENVIRONMENT_LIGHTS || WAVE_ORDINARYSHADE_UNIFIED_NEE || \
        WAVE_ORDINARYSHADE_EMISSIVE_MIS || WAVE_ORDINARYSHADE_SECONDARY_TRANSPORT || \
        WAVE_ORDINARYSHADE_SECONDARY_TRANSMISSION || \
        WAVE_ORDINARYSHADE_SECONDARY_SURFACE || WAVE_ORDINARYSHADE_SECONDARY_CONTROL || \
        WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION
#define ordinarylight_output_queue_count output_queue.count
#define ordinarylight_output_queue output_queue
#define ordinarylight_vertices vertices
#define ordinarylight_attributes attributes
#define ordinarylight_materials materials
#define ordinarylight_medium_stacks stacks
#define ordinarylight_paths paths
#define ordinarylight_secondary_paths secondary_paths
#include "ordinaryshade_primary.glsl"
#undef ordinarylight_secondary_paths
#undef ordinarylight_paths
#undef ordinarylight_medium_stacks
#undef ordinarylight_materials
#undef ordinarylight_attributes
#undef ordinarylight_vertices
#undef ordinarylight_output_queue
#undef ordinarylight_output_queue_count
#endif

float wrapTextureCoordinate(float value, uint mode)
{
    if (mode == 1u)
        return clamp(value, 0.0, 0.99999994);
    if (mode == 2u) {
        float period = mod(value, 2.0);
        if (period < 0.0)
            period += 2.0;
        return period <= 1.0 ? period : 2.0 - period;
    }
    return fract(value);
}

int wrapTextureIndex(int value, int size, uint mode)
{
    if (mode == 1u)
        return clamp(value, 0, size - 1);
    int period = mode == 2u ? size * 2 : size;
    int wrapped = value % period;
    if (wrapped < 0)
        wrapped += period;
    return mode == 2u && wrapped >= size ? period - 1 - wrapped : wrapped;
}

vec4 decodeTextureTexel(uint packed, bool srgb)
{
    vec4 value = unpackUnorm4x8(packed);
    if (!srgb)
        return value;
    // Packed source channels are 8-bit, so this table is the exact glTF sRGB
    // transfer evaluated once for every representable input value.
    value.r = SRGB_TO_LINEAR[packed & 255u];
    value.g = SRGB_TO_LINEAR[(packed >> 8u) & 255u];
    value.b = SRGB_TO_LINEAR[(packed >> 16u) & 255u];
    return value;
}

vec4 fetchTextureTexel(
    uint offset, ivec2 size, ivec2 coordinate, uvec2 wrap, bool srgb)
{
    int x = wrapTextureIndex(coordinate.x, size.x, wrap.x);
    int y = wrapTextureIndex(coordinate.y, size.y, wrap.y);
    return decodeTextureTexel(
        texture_words[offset + uint(y * size.x + x)], srgb);
}

uint textureMipOffset(uint base_offset, ivec2 base_size, uint level)
{
    uint offset = base_offset;
    ivec2 size = base_size;
    for (uint current = 0u; current < level; ++current) {
        offset += uint(size.x * size.y);
        size = max((size + 1) / 2, ivec2(1));
    }
    return offset;
}

vec4 sampleTextureLevel(
    uint base_offset, ivec2 base_size, uint level, vec2 uv,
    uvec2 wrap, bool srgb, bool linear_filter)
{
    ivec2 size = max(
        (base_size + (ivec2(1) << int(level)) - 1) >> int(level), ivec2(1));
    uint offset = textureMipOffset(base_offset, base_size, level);
    vec2 wrapped_uv = vec2(
        wrapTextureCoordinate(uv.x, wrap.x),
        wrapTextureCoordinate(uv.y, wrap.y));
    if (!linear_filter) {
        ivec2 coordinate = ivec2(floor(wrapped_uv * vec2(size)));
        return fetchTextureTexel(offset, size, coordinate, wrap, srgb);
    }
    vec2 texel = wrapped_uv * vec2(size) - 0.5;
    ivec2 base = ivec2(floor(texel));
    vec2 fraction = fract(texel);
    vec4 top = mix(
        fetchTextureTexel(offset, size, base, wrap, srgb),
        fetchTextureTexel(offset, size, base + ivec2(1, 0), wrap, srgb),
        fraction.x);
    vec4 bottom = mix(
        fetchTextureTexel(offset, size, base + ivec2(0, 1), wrap, srgb),
        fetchTextureTexel(offset, size, base + ivec2(1, 1), wrap, srgb),
        fraction.x);
    return mix(top, bottom, fraction.y);
}

vec4 sampleSceneTexture(int texture_index, vec2 uv)
{
    if (texture_index < 0 || uint(texture_index) >= texture_words[0])
        return vec4(0.0);
#if WAVE_NATIVE_TEXTURES
    int descriptor_index = texture_index * 2 + 1;
    return textureLod(
        native_textures[nonuniformEXT(descriptor_index)], uv, 0.0);
#else
    uint metadata = 1u + uint(texture_index) * 8u;
    uint offset = texture_words[metadata + 1u];
    ivec2 size = ivec2(
        texture_words[metadata + 2u], texture_words[metadata + 3u]);
    uint flags = texture_words[metadata + 4u];
    uvec2 wrap = uvec2(flags & 3u, (flags >> 2u) & 3u);
    return sampleTextureLevel(
        offset, size, 0u, uv, wrap, false, (flags & 16u) != 0u);
#endif
}

vec4 sampleMaterialTexture(
    float binding_index_value, vec2 uv0, vec2 uv1, bool srgb,
    float uv0_footprint, float uv1_footprint)
{
    int binding_index = int(binding_index_value);
    if (binding_index < 0)
        return vec4(1.0);
    TextureBindingData binding = texture_bindings[binding_index];
    int texture_index = int(binding.texture_rotation.x);
    if (texture_index < 0 || uint(texture_index) >= texture_words[0])
        return vec4(1.0);
#if WAVE_WORK_COUNTERS
    profileWork(2u, 1u);
#endif
    bool use_uv1 = binding.texture_rotation.w > 0.5;
    vec2 uv = use_uv1 ? uv1 : uv0;
    float uv_footprint = use_uv1 ? uv1_footprint : uv0_footprint;
    float cosine = binding.texture_rotation.y;
    float sine = binding.texture_rotation.z;
    vec2 scaled_uv = uv * binding.offset_scale.zw;
    uv = binding.offset_scale.xy + vec2(
        cosine * scaled_uv.x - sine * scaled_uv.y,
        sine * scaled_uv.x + cosine * scaled_uv.y);
    uv_footprint *= max(
        abs(binding.offset_scale.z), abs(binding.offset_scale.w));
#if WAVE_NATIVE_TEXTURES
    int descriptor_index = texture_index * 2 + (srgb ? 0 : 1);
    ivec2 size = textureSize(
        native_textures[nonuniformEXT(descriptor_index)], 0);
    int level_count = textureQueryLevels(
        native_textures[nonuniformEXT(descriptor_index)]);
    float lod = clamp(log2(max(
        uv_footprint * float(max(size.x, size.y)), 1.0)),
        0.0, float(max(level_count, 1) - 1));
    return textureLod(
        native_textures[nonuniformEXT(descriptor_index)], uv, lod);
#else
    uint metadata = 1u + uint(texture_index) * 8u;
    uint offset = texture_words[metadata + (srgb ? 0u : 1u)];
    ivec2 size = ivec2(texture_words[metadata + 2u], texture_words[metadata + 3u]);
    uint flags = texture_words[metadata + 4u];
    uint level_count = texture_words[metadata + 5u];
    uvec2 wrap = uvec2(flags & 3u, (flags >> 2u) & 3u);
    float lod = clamp(log2(max(
        uv_footprint * float(max(size.x, size.y)), 1.0)),
        0.0, float(max(level_count, 1u) - 1u));
    uint lower = uint(floor(lod));
    bool linear_filter = (flags & 16u) != 0u;
    vec4 first = sampleTextureLevel(
        offset, size, lower, uv, wrap, srgb, linear_filter);
    if (!linear_filter || lower + 1u >= level_count)
        return first;
    vec4 second = sampleTextureLevel(
        offset, size, lower + 1u, uv, wrap, srgb, true);
    return mix(first, second, fract(lod));
#endif
}

float triangleUvDensity(
    vec3 a, vec3 b, vec3 c, vec2 uv_a, vec2 uv_b, vec2 uv_c)
{
    vec2 first_uv = uv_b - uv_a;
    vec2 second_uv = uv_c - uv_a;
    float uv_area = abs(first_uv.x * second_uv.y - first_uv.y * second_uv.x);
    float world_area = length(cross(b - a, c - a));
    return sqrt(uv_area / max(world_area, 0.00000001));
}

bool materialHasTextures(MaterialData material)
{
#if WAVE_UNTEXTURED_SCENE
    return false;
#else
    return any(greaterThanEqual(material.texture_indices, vec4(0.0)))
        || material.texture_parameters.y >= 0.0
        || material.texture_parameters.w >= 0.0
        || any(greaterThanEqual(material.advanced_texture_indices, vec4(0.0)));
#endif
}

void applyMaterialTextures(
    inout MaterialData material, vec2 uv0, vec2 uv1,
    float uv0_footprint, float uv1_footprint)
{
#if WAVE_UNTEXTURED_SCENE
    // Once material textures have been evaluated, this component stores the
    // resolved occlusion multiplier rather than the transmission binding.
    material.texture_parameters.w = 1.0;
    return;
#else
    // Resolve transmission first. Metallic/roughness and occlusion only feed
    // the opaque PBR branch, so transmissive paths need not fetch them.
#if WAVE_WORK_COUNTERS
    if (material.texture_parameters.w >= 0.0) profileWork(5u, 1u);
#endif
    float transmission = sampleMaterialTexture(
        material.texture_parameters.w, uv0, uv1, false,
        uv0_footprint, uv1_footprint).r;
#if WAVE_ORDINARYSHADE_TEXTURE_APPLICATION
    material.attenuation_transmission.a = ordinarylight_texture_apply_scalar(
        material.attenuation_transmission.a, transmission);
#else
    material.attenuation_transmission.a *= transmission;
#endif
#if WAVE_WORK_COUNTERS
    if (material.texture_indices.x >= 0.0) profileWork(6u, 1u);
#endif
    vec3 base_color_sample = sampleMaterialTexture(
        material.texture_indices.x, uv0, uv1, true,
        uv0_footprint, uv1_footprint).rgb;
#if WAVE_ORDINARYSHADE_TEXTURE_APPLICATION
    material.base_roughness.rgb = ordinarylight_texture_apply_rgb(
        material.base_roughness.rgb, base_color_sample);
#else
    material.base_roughness.rgb *= base_color_sample;
#endif
#if WAVE_WORK_COUNTERS
    if (material.texture_indices.z >= 0.0) profileWork(8u, 1u);
#endif
    vec3 emissive_sample = sampleMaterialTexture(
        material.texture_indices.z, uv0, uv1, true,
        uv0_footprint, uv1_footprint).rgb;
#if WAVE_ORDINARYSHADE_TEXTURE_APPLICATION
    material.emission_metallic.rgb = ordinarylight_texture_apply_rgb(
        material.emission_metallic.rgb, emissive_sample);
#else
    material.emission_metallic.rgb *= emissive_sample;
#endif
    vec4 clearcoat_sample = sampleMaterialTexture(
        material.advanced_texture_indices.x, uv0, uv1, false,
        uv0_footprint, uv1_footprint);
    vec4 sheen_sample = sampleMaterialTexture(
        material.advanced_texture_indices.y, uv0, uv1, true,
        uv0_footprint, uv1_footprint);
    vec4 anisotropy_sample = sampleMaterialTexture(
        material.advanced_texture_indices.z, uv0, uv1, false,
        uv0_footprint, uv1_footprint);
    vec4 subsurface_sample = sampleMaterialTexture(
        material.advanced_texture_indices.w, uv0, uv1, false,
        uv0_footprint, uv1_footprint);
    material.advanced0.x *= clearcoat_sample.r;
    material.advanced0.y *= max(clearcoat_sample.g, clearcoat_sample.r);
    material.sheen_color.rgb *= sheen_sample.rgb;
    material.advanced0.z *= sheen_sample.a;
    material.advanced0.w *= anisotropy_sample.r;
    material.advanced1.x *= subsurface_sample.r;
    if (material.attenuation_transmission.a > 0.001)
        return;
#if WAVE_WORK_COUNTERS
    if (material.texture_indices.y >= 0.0) profileWork(7u, 1u);
#endif
    vec4 metallic_roughness = sampleMaterialTexture(
        material.texture_indices.y, uv0, uv1, false,
        uv0_footprint, uv1_footprint);
#if WAVE_ORDINARYSHADE_TEXTURE_APPLICATION
    material.base_roughness.a = ordinarylight_texture_apply_scalar(
        material.base_roughness.a, metallic_roughness.g);
    material.emission_metallic.a = ordinarylight_texture_apply_scalar(
        material.emission_metallic.a, metallic_roughness.b);
#else
    material.base_roughness.a *= metallic_roughness.g;
    material.emission_metallic.a *= metallic_roughness.b;
#endif
#if WAVE_WORK_COUNTERS
    if (material.texture_parameters.y >= 0.0) profileWork(9u, 1u);
#endif
    float occlusion = sampleMaterialTexture(
        material.texture_parameters.y, uv0, uv1, false,
        uv0_footprint, uv1_footprint).r;
#if WAVE_ORDINARYSHADE_TEXTURE_APPLICATION
    material.texture_parameters.w = ordinarylight_texture_apply_occlusion(
        occlusion, material.texture_parameters.z);
#else
    material.texture_parameters.w = mix(
        1.0, occlusion, material.texture_parameters.z);
#endif
#endif
}

vec3 applyNormalTexture(
    MaterialData material, vec2 uv0, vec2 uv1,
    float uv0_footprint, float uv1_footprint,
    vec3 shading_normal, vec4 tangent_data)
{
#if WAVE_UNTEXTURED_SCENE
    return shading_normal;
#else
    int texture_index = int(material.texture_indices.w);
    if (texture_index < 0)
        return shading_normal;
#if WAVE_WORK_COUNTERS
    profileWork(10u, 1u);
#endif
    vec3 tangent_normal = sampleMaterialTexture(
        material.texture_indices.w, uv0, uv1, false,
        uv0_footprint, uv1_footprint).xyz;
#if WAVE_ORDINARYSHADE_TEXTURE_APPLICATION
    TextureBindingData binding = texture_bindings[texture_index];
    return ordinarylight_texture_apply_normal(
        tangent_normal, material.texture_parameters.x,
        shading_normal, tangent_data, binding.texture_rotation.yz,
        binding.offset_scale.zw);
#else
    tangent_normal = tangent_normal * 2.0 - 1.0;
    tangent_normal.xy *= material.texture_parameters.x;
    tangent_normal = normalize(tangent_normal);
    vec3 tangent = normalize(tangent_data.xyz
        - shading_normal * dot(shading_normal, tangent_data.xyz));
    vec3 bitangent = cross(shading_normal, tangent) * tangent_data.w;
    TextureBindingData binding = texture_bindings[texture_index];
    float cosine = binding.texture_rotation.y;
    float sine = binding.texture_rotation.z;
    float inverse_x = 1.0 / binding.offset_scale.z;
    float inverse_y = 1.0 / binding.offset_scale.w;
    vec3 transformed_tangent = normalize(
        tangent * (cosine * inverse_x)
        + bitangent * (-sine * inverse_y));
    vec3 transformed_bitangent = normalize(
        tangent * (sine * inverse_x)
        + bitangent * (cosine * inverse_y));
    tangent = transformed_tangent;
    bitangent = transformed_bitangent;
    return normalize(tangent * tangent_normal.x
        + bitangent * tangent_normal.y + shading_normal * tangent_normal.z);
#endif
#endif
}

bool textureBindingUsesUv1(float binding_index_value)
{
    int binding_index = int(binding_index_value);
    return binding_index >= 0
        && texture_bindings[binding_index].texture_rotation.w > 0.5;
}

bool materialUsesUv1(MaterialData material)
{
    return fract(material.ior_distance.z) > 0.125;
}

vec4 triangleTangent(
    vec3 a, vec3 b, vec3 c, vec2 uv_a, vec2 uv_b, vec2 uv_c,
    vec3 shading_normal)
{
    vec3 edge_a = b - a;
    vec3 edge_b = c - a;
    vec2 delta_a = uv_b - uv_a;
    vec2 delta_b = uv_c - uv_a;
    float determinant = delta_a.x * delta_b.y - delta_a.y * delta_b.x;
    if (abs(determinant) < 0.00000001) {
        vec3 fallback = normalize(abs(shading_normal.z) < 0.999
            ? cross(shading_normal, vec3(0.0, 0.0, 1.0))
            : cross(shading_normal, vec3(0.0, 1.0, 0.0)));
        return vec4(fallback, 1.0);
    }
    float inverse = 1.0 / determinant;
    vec3 tangent = normalize(
        (edge_a * delta_b.y - edge_b * delta_a.y) * inverse);
    vec3 bitangent = normalize(
        (edge_b * delta_a.x - edge_a * delta_b.x) * inverse);
    float handedness = dot(cross(shading_normal, tangent), bitangent) < 0.0
        ? -1.0 : 1.0;
    return vec4(tangent, handedness);
}
