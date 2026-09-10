"""Mechanical resource ABI and variant guards. Algorithms are typed helpers."""
LAYOUT = r'''#include "transport_v1/material_contracts.glsl"

#if !defined(WAVE_SHARED_PRIMARY_RESERVOIRS)
#define WAVE_SHARED_PRIMARY_RESERVOIRS 0
#endif
#if !defined(WAVE_LOCAL_SIZE_X)
#define WAVE_LOCAL_SIZE_X 8
#endif
#if !defined(WAVE_LOCAL_SIZE_Y)
#define WAVE_LOCAL_SIZE_Y 8
#endif
#if !defined(WAVE_RAYGEN)
#define WAVE_RAYGEN 0
#endif
#if !defined(WAVE_CUSTOM_MATERIAL_PROGRAM)
#define WAVE_CUSTOM_MATERIAL_PROGRAM 0
#endif
#if !defined(WAVE_UNTEXTURED_PRIMARY)
#define WAVE_UNTEXTURED_PRIMARY WAVE_UNTEXTURED_SCENE
#endif
#if !defined(WAVE_UNTEXTURED_SECONDARY)
#define WAVE_UNTEXTURED_SECONDARY WAVE_UNTEXTURED_SCENE
#endif
#if !WAVE_RAYGEN
layout(
    local_size_x = WAVE_LOCAL_SIZE_X,
    local_size_y = WAVE_LOCAL_SIZE_Y,
    local_size_z = 1
) in;
#endif

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
struct WaveMediumStack { float ior[WAVE_MAX_MEDIUM_STACK_DEPTH]; };
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

layout(set = 0, binding = 0) uniform accelerationStructureEXT scene_tlas;
layout(set = 0, binding = 1, std430) buffer PathStates {
    WavePathState paths[];
};
layout(set = 0, binding = 2, std430) readonly buffer MaterialBuffer {
    MaterialData materials[];
};
layout(set = 0, binding = 3, std430) readonly buffer VertexBuffer {
    vec4 vertices[];
};
layout(set = 0, binding = 4, std430) readonly buffer AttributeBuffer {
    VertexAttributeData attributes[];
};
layout(set = 0, binding = 5, std430) buffer OutputRayQueue {
    uint count; uint capacity; uint overflow; uint queue_padding;
    WaveRay rays[];
} output_queue;
layout(set = 0, binding = 6, std430) buffer MediumStacks {
    WaveMediumStack stacks[];
};
layout(set = 0, binding = 7, std430) readonly buffer CameraData {
    vec4 origin;
    vec4 forward;
    vec4 right;
    vec4 up;
#if WAVE_CAMERA_RESTIR_POLICY
    uvec4 restir_policy;
#endif
} camera;
layout(set = 0, binding = 8, r32f) uniform writeonly image2D position_image;
layout(set = 0, binding = 9, r32ui) uniform writeonly uimage2D normal_image;
layout(set = 0, binding = 18, std430) readonly buffer PreviousCameraData {
    vec4 origin;
    vec4 forward;
    vec4 right;
    vec4 up;
} previous_camera;
layout(set = 0, binding = 19, r32f) uniform readonly image2D previous_position_image;
layout(set = 0, binding = 20, r32ui) uniform readonly uimage2D previous_normal_image;
layout(set = 0, binding = 21, r32ui) uniform writeonly uimage2D material_image;
layout(set = 0, binding = 22, r32ui) uniform readonly uimage2D previous_material_image;
layout(set = 0, binding = 23, std430) buffer SecondaryPathStates {
    SecondaryPathState secondary_paths[];
};
layout(set = 0, binding = 10, std430) readonly buffer PointLightBuffer {
    PointLightData point_lights[];
};
layout(set = 0, binding = 11, std430) readonly buffer AreaLightBuffer {
    AreaLightData area_lights[];
};
layout(set = 0, binding = 12, std430) readonly buffer TextureBuffer {
    uint texture_words[];
};
layout(set = 0, binding = 13, std430) readonly buffer TextureBindingBuffer {
    TextureBindingData texture_bindings[];
};
#if WAVE_NATIVE_TEXTURES
layout(set = 0, binding = 14) uniform sampler2D native_textures[128];
#endif
#if WAVE_WORK_COUNTERS
layout(set = 0, binding = 15, std430) buffer WorkCounterBuffer {
    uint work_counters[];
};
uint profile_bounce = 0u;
@profileWork@
#endif

layout(push_constant) uniform PushConstants {
    uvec4 image_tile;
    uvec4 tile_frame;
    uint max_bounces;
    uint point_light_count;
    uint area_light_count;
    uint area_light_samples;
    uint secondary_area_light_samples;
    float area_light_weight;
    uint gbuffer_enabled;
    uint environment_samples;
    uint subgroup_enqueue;
    uint russian_roulette_start;
    float russian_roulette_min_survival;
    float secondary_nee_probability;
    uint inline_bounces;
    uint restir_di;
    uint restir_history_valid;
    uint restir_history_limit;
    uint restir_candidate_count;
    uint restir_spatial_reuse;
    uint restir_spatial_neighbors;
    uint restir_spatial_radius;
    uint restir_pairwise_mis;
    uint restir_generalized_mis;
    float restir_generalized_balance_cap;
    uint unified_secondary_nee;
    uint unified_primary_restir;
    uint stratified_primary_restir;
    uint indirect_secondary_capture;
    uint indirect_capture_stride;
    uvec4 object_effect_ranges[2];
} push;

// The production specialization removes optional estimators that are disabled
// in the selected RendererConfig. The push-constant ABI stays unchanged so
// specialized and general pipelines remain interchangeable.
#if WAVE_PRODUCTION_RESTIR
#define WAVE_GENERALIZED_RESTIR 0u
#define WAVE_UNIFIED_PRIMARY_RESTIR 0u
#define WAVE_STRATIFIED_PRIMARY_RESTIR 0u
#else
#define WAVE_GENERALIZED_RESTIR push.restir_generalized_mis
#define WAVE_UNIFIED_PRIMARY_RESTIR push.unified_primary_restir
#define WAVE_STRATIFIED_PRIMARY_RESTIR push.stratified_primary_restir
#endif

uint hashValue(uint value);

@restirEncodeNormal@

@restirDecodeNormal@

@restirPackNormalClass@

@restirUnpackNormalClass@

@restirSpatialOffset@

@restirMaterialSignature@

@reprojectRestir@

@restirPreviousWorldPosition@

@restirHistorySurfaceCompatible@

@restirHistorySurfaceMatches@

uint ordinarylight_reserve_output_index(uint subgroup_enqueue);

@reserveOutputIndex@

@hashValue@

#define WAVE_RESTIR_PRIMARY 1
#include "wavefront_textures.glsl"
#include "wavefront_volumes.glsl"
#include "wavefront_lighting.glsl"

#define WAVE_MEDIUM_IOR(path_index, depth)      ordinarylight_medium_ior(path_index, depth)
#define WAVE_SET_MEDIUM_IOR(path_index, depth, value)      ordinarylight_set_medium_ior(path_index, depth, value)

#define WAVE_STORE_PATH(path_index, path)      ordinarylight_store_path(path_index, path)
#define WAVE_DEACTIVATE_STORED_PATH(path_index)      ordinarylight_deactivate_stored_path(path_index)
#define WAVE_SECONDARY_PRIMARY_VALID(path_index)      ordinarylight_secondary_primary_valid(path_index)

#define WAVE_PROFILE_WORK(counter, amount)      ordinarylight_profile_work(counter, amount)
#define WAVE_INTEGRATE_VOLUMES(origin, direction, distance, radiance, throughput)      ordinarylight_integrate_secondary_volumes(          origin, direction, distance, radiance, throughput)

#if !defined(WAVE_GROUP_SWIZZLE_WIDTH)
#define WAVE_GROUP_SWIZZLE_WIDTH 1
#endif
#include "ordinaryshade_primary.glsl"

#if WAVE_MEGAKERNEL || WAVE_HYBRID
@ordinarylightSecondaryBounce@

@traceRemaining@
#endif

#if !WAVE_CONTINUATION
@primaryRestirHistoryValid@

@primaryRestirHistoryLimit@

@processPrimaryPixel@
#endif

#if !WAVE_CONTINUATION
#if WAVE_PERSISTENT_COARSE && !1
shared uint persistent_tile_index;
#endif

@main@
#endif
'''
