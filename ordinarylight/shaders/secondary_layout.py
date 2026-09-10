"""Mechanical resource ABI and variant guards. Algorithms are typed helpers."""
LAYOUT = r'''#version 460
#extension GL_EXT_ray_query : require
#extension GL_GOOGLE_include_directive : require
#include "transport_v1/material_contracts.glsl"
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require
#if WAVE_NATIVE_TEXTURES
#extension GL_EXT_nonuniform_qualifier : require
#endif

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

const uint WAVE_MAX_MEDIUM_STACK_DEPTH = 16u;
const uint PATH_ACTIVE_BIT = 1u;
const uint PATH_PREVIOUS_DIFFUSE_BIT = 2u;
const uint PATH_PREVIOUS_UNIFIED_NEE_BIT = 4u;
const uint PATH_INDIRECT_CAPTURE_BIT = 8u;

struct WaveRay {
    vec4 origin_tmin;
    vec4 direction_tmax;
    uint path_index;
    uint padding_a;
    uint padding_b;
    uint padding_c;
};
struct WaveHit {
    vec4 position_t;
    vec3 geometric_normal;
    uint primitive_index;
    vec2 barycentrics;
    uint ray_index;
    uint path_index;
};
struct WavePathState {
    vec4 throughput;
    vec4 radiance;
    uvec4 metadata;
};
struct SecondaryPathState {
    vec4 position_valid;
    vec4 normal_pdf;
    vec4 primary_throughput;
    vec4 primary_radiance;
    vec4 diffuse_radiance_hit_distance;
    vec4 specular_radiance_hit_distance;
    vec4 primary_position;
    vec4 primary_geometry;
};

@pathRng@
@setPathRng@
@pathBounce@
@setPathBounce@
@pathPreviousPdf@
@setPathPreviousPdf@
struct WaveMediumStack {
    float ior[WAVE_MAX_MEDIUM_STACK_DEPTH];
};
struct MaterialData {
    vec4 base_roughness;
    vec4 emission_metallic;
    vec4 attenuation_transmission;
    vec4 ior_distance;
    vec4 texture_indices;
    vec4 texture_parameters;
    vec4 advanced0;
    vec4 advanced1;
    vec4 sheen_color;
    vec4 subsurface_color;
    vec4 advanced_texture_indices;
    vec4 optical;
};

struct VertexAttributeData { vec4 normal; vec4 texcoord; vec4 tangent; };
struct PointLightData {
    vec4 position_type;
    vec4 direction_range;
    vec4 color_intensity;
    vec4 spot_parameters;
};
struct AreaLightData {
    vec4 a; vec4 b; vec4 c; vec4 emission_area; vec4 distribution;
};
struct TextureBindingData { vec4 texture_rotation; vec4 offset_scale; };

layout(set = 0, binding = 0, std430) readonly buffer HitQueue {
    uint count; uint capacity; uint overflow; uint queue_padding;
    WaveHit hits[];
} hit_queue;
layout(set = 0, binding = 1, std430) readonly buffer InputRayQueue {
    uint count; uint capacity; uint overflow; uint queue_padding;
    WaveRay rays[];
} input_queue;
layout(set = 0, binding = 2, std430) buffer PathStates {
    WavePathState paths[];
};
layout(set = 0, binding = 3, std430) readonly buffer MaterialBuffer {
    MaterialData materials[];
};
layout(set = 0, binding = 4, std430) readonly buffer VertexBuffer {
    vec4 vertices[];
};
layout(set = 0, binding = 5, std430) readonly buffer AttributeBuffer {
    VertexAttributeData attributes[];
};
layout(set = 0, binding = 6, std430) buffer OutputRayQueue {
    uint count; uint capacity; uint overflow; uint queue_padding;
    WaveRay rays[];
} output_queue;
layout(set = 0, binding = 7, std430) buffer MediumStacks {
    WaveMediumStack stacks[];
};
layout(set = 0, binding = 8) uniform accelerationStructureEXT scene_tlas;
layout(set = 0, binding = 9, std430) readonly buffer PointLightBuffer {
    PointLightData point_lights[];
};
layout(set = 0, binding = 10, std430) readonly buffer AreaLightBuffer {
    AreaLightData area_lights[];
};
layout(set = 0, binding = 11, std430) readonly buffer TextureBuffer {
    uint texture_words[];
};
layout(set = 0, binding = 12, std430) readonly buffer TextureBindingBuffer {
    TextureBindingData texture_bindings[];
};
#if WAVE_NATIVE_TEXTURES
layout(set = 0, binding = 13) uniform sampler2D native_textures[128];
#endif
#if WAVE_WORK_COUNTERS
layout(set = 0, binding = 14, std430) buffer WorkCounterBuffer {
    uint work_counters[];
};
uint profile_bounce = 0u;
@profileWork@
#endif
layout(set = 0, binding = 15, std430) buffer SecondaryPathStates {
    SecondaryPathState secondary_paths[];
};

layout(push_constant) uniform PushConstants {
    uint max_bounces;
    uint point_light_count;
    uint area_light_count;
    uint area_light_samples;
    uint secondary_area_light_samples;
    float area_light_weight;
    uint environment_samples;
    uint russian_roulette_start;
    float russian_roulette_min_survival;
    uint fused_intersection;
    uint subgroup_enqueue;
    float secondary_nee_probability;
    uint unified_secondary_nee;
    uint indirect_secondary_capture;
} push;

@reserveOutputIndex@

#define WAVE_VOLUME_HEADER_BINDING 17
#define WAVE_VOLUME_SCALAR_BINDING 18
#define WAVE_VOLUME_TRANSFER_BINDING 19
#define WAVE_VOLUME_TRIANGLE_BINDING 20
#define WAVE_VOLUME_SAMPLER_BINDING 21
#include "wavefront_textures.glsl"
#include "wavefront_volumes.glsl"
#include "wavefront_lighting.glsl"

@terminatePath@

@main@
'''
