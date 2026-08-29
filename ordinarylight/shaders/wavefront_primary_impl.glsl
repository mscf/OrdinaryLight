
#ifndef WAVE_LOCAL_SIZE_X
#define WAVE_LOCAL_SIZE_X 8
#endif
#ifndef WAVE_LOCAL_SIZE_Y
#define WAVE_LOCAL_SIZE_Y 8
#endif
#ifndef WAVE_RAYGEN
#define WAVE_RAYGEN 0
#endif
#ifndef WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION
#define WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION 1
#endif
#ifndef WAVE_CUSTOM_MATERIAL_PROGRAM
#define WAVE_CUSTOM_MATERIAL_PROGRAM 0
#endif
#ifndef WAVE_UNTEXTURED_PRIMARY
#define WAVE_UNTEXTURED_PRIMARY WAVE_UNTEXTURED_SCENE
#endif
#ifndef WAVE_UNTEXTURED_SECONDARY
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
};

uint pathRng(WavePathState path) { return path.metadata.z; }
void setPathRng(inout WavePathState path, uint rng) {
    path.metadata.z = rng;
}
uint pathBounce(WavePathState path) { return uint(path.throughput.w); }
void setPathBounce(inout WavePathState path, uint bounce) {
    path.throughput.w = float(bounce);
}
float pathPreviousPdf(WavePathState path) { return path.radiance.w; }
void setPathPreviousPdf(inout WavePathState path, float pdf) {
    path.radiance.w = pdf;
}
struct WaveMediumStack { float ior[WAVE_MAX_MEDIUM_STACK_DEPTH]; };
struct MaterialData {
    vec4 base_roughness;
    vec4 emission_metallic;
    vec4 attenuation_transmission;
    vec4 ior_distance;
    vec4 texture_indices;
    vec4 texture_parameters;
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
void profileWork(uint counter, uint amount)
{
    atomicAdd(work_counters[counter], amount);
    uint bounce = min(profile_bounce, 7u);
    if (counter == 0u)
        atomicAdd(work_counters[16u + bounce], amount);
    else if (counter == 1u)
        atomicAdd(work_counters[24u + bounce], amount);
    else if (counter == 3u)
        atomicAdd(work_counters[32u + bounce], amount);
}
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

vec2 restirEncodeNormal(vec3 normal)
{
    normal /= abs(normal.x) + abs(normal.y) + abs(normal.z);
    vec2 encoded = normal.xy;
    if (normal.z < 0.0)
        encoded = (1.0 - abs(encoded.yx)) * sign(encoded.xy);
    return encoded;
}

vec3 restirDecodeNormal(vec2 encoded)
{
    vec3 normal = vec3(encoded, 1.0 - abs(encoded.x) - abs(encoded.y));
    if (normal.z < 0.0)
        normal.xy = (1.0 - abs(normal.yx)) * sign(normal.xy);
    return normalize(normal);
}

uint restirPackNormalClass(vec3 normal, float surface_class)
{
    vec2 unit = restirEncodeNormal(normal) * 0.5 + 0.5;
    uvec2 quantized = uvec2(round(clamp(unit, 0.0, 1.0) * 32767.0));
    uint classification = uint(clamp(round(surface_class), 0.0, 3.0));
    return quantized.x | (quantized.y << 15u) | (classification << 30u);
}

vec4 restirUnpackNormalClass(uint packed)
{
    vec2 unit = vec2(
        float(packed & 0x7fffu),
        float((packed >> 15u) & 0x7fffu)
    ) / 32767.0;
    return vec4(
        restirDecodeNormal(unit * 2.0 - 1.0),
        float(packed >> 30u));
}

ivec2 restirSpatialOffset(uint index, int radius)
{
    const ivec2 directions[8] = ivec2[8](
        ivec2(1, 0), ivec2(-1, 0), ivec2(0, 1), ivec2(0, -1),
        ivec2(1, 1), ivec2(-1, 1), ivec2(1, -1), ivec2(-1, -1));
    return directions[index & 7u] * radius;
}

uint restirMaterialSignature(MaterialData material)
{
    uint signature = hashValue(floatBitsToUint(material.base_roughness.x));
    signature ^= hashValue(floatBitsToUint(material.base_roughness.y));
    signature ^= hashValue(floatBitsToUint(material.base_roughness.z));
    signature ^= hashValue(floatBitsToUint(material.base_roughness.w));
    signature ^= hashValue(floatBitsToUint(material.emission_metallic.w));
    signature ^= hashValue(floatBitsToUint(
        material.attenuation_transmission.w));
    signature ^= hashValue(floatBitsToUint(material.ior_distance.x));
    signature ^= hashValue(floatBitsToUint(material.texture_indices.x));
    signature ^= hashValue(floatBitsToUint(material.texture_indices.y));
    signature ^= hashValue(floatBitsToUint(material.texture_indices.z));
    signature ^= hashValue(floatBitsToUint(material.texture_indices.w));
    return signature;
}

bool reprojectRestir(vec3 world_position, out ivec2 previous_pixel)
{
    vec3 offset = world_position - previous_camera.origin.xyz;
    float previous_depth = dot(offset, previous_camera.forward.xyz);
    float scale = length(previous_camera.up.xyz);
    float aspect = float(push.image_tile.x) / float(push.image_tile.y);
    if (previous_depth <= 0.0001 || scale <= 0.0001)
        return false;
    vec2 ndc = vec2(
        dot(offset, normalize(previous_camera.right.xyz)) /
            (previous_depth * aspect * scale),
        -dot(offset, normalize(previous_camera.up.xyz)) /
            (previous_depth * scale));
    vec2 pixel = (ndc * 0.5 + 0.5) * vec2(push.image_tile.xy) - 0.5;
    previous_pixel = ivec2(round(pixel));
    return all(greaterThanEqual(previous_pixel, ivec2(0)))
        && all(lessThan(previous_pixel, ivec2(push.image_tile.xy)));
}

vec3 restirPreviousWorldPosition(ivec2 pixel, float ray_distance)
{
    vec2 ndc = ((vec2(pixel) + 0.5) / vec2(push.image_tile.xy))
        * 2.0 - 1.0;
    float aspect = float(push.image_tile.x) / float(push.image_tile.y);
    vec3 direction = normalize(previous_camera.forward.xyz
        + ndc.x * aspect * previous_camera.right.xyz
        - ndc.y * previous_camera.up.xyz);
    return previous_camera.origin.xyz + direction * ray_distance;
}

bool restirHistorySurfaceCompatible(
    ivec2 history_pixel, vec3 position, vec3 shading_normal,
    float distance, float surface_class, uint material_signature,
    bool temporal_center, out vec4 old_position, out vec4 old_normal)
{
    float old_distance = imageLoad(
        previous_position_image, history_pixel).x;
    old_position = vec4(
        restirPreviousWorldPosition(history_pixel, old_distance),
        old_distance);
    old_normal = restirUnpackNormalClass(imageLoad(
        previous_normal_image, history_pixel).x);
    uint old_material = imageLoad(
        previous_material_image, history_pixel).x;
    vec3 position_delta = old_position.xyz - position;
    float position_tolerance = max(0.03, distance * 0.01);
    float position_error = temporal_center
        ? length(position_delta)
        : abs(dot(position_delta, normalize(shading_normal)));
    float normal_agreement = dot(
        normalize(old_normal.xyz), normalize(shading_normal));
    return old_position.w >= 0.0
        && position_error <= position_tolerance
        && normal_agreement > 0.9
        && abs(old_normal.w - surface_class) < 0.25
        && old_material == material_signature;
}

bool restirHistorySurfaceMatches(
    ivec2 history_pixel, vec3 position, vec3 shading_normal,
    float distance, float surface_class, uint material_signature,
    bool temporal_center)
{
    vec4 history_position;
    vec4 history_normal;
    return restirHistorySurfaceCompatible(
        history_pixel, position, shading_normal, distance, surface_class,
        material_signature, temporal_center, history_position, history_normal);
}

#if WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION
uint ordinarylight_reserve_output_index(uint subgroup_enqueue);
#endif

uint reserveOutputIndex()
{
#if WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION
    return ordinarylight_reserve_output_index(push.subgroup_enqueue);
#else
    if (push.subgroup_enqueue == 0u)
        return atomicAdd(output_queue.count, 1u);
    uvec4 active_lanes = subgroupBallot(true);
    uint base = 0u;
    if (subgroupElect())
        base = atomicAdd(
            output_queue.count, subgroupBallotBitCount(active_lanes));
    base = subgroupBroadcastFirst(base);
    return base + subgroupBallotExclusiveBitCount(active_lanes);
#endif
}

uint hashValue(uint value)
{
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    value ^= value >> 16;
    return value;
}

#define WAVE_RESTIR_PRIMARY 1
#include "wavefront_textures.glsl"
#include "wavefront_volumes.glsl"
#include "wavefront_lighting.glsl"

#if WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION
#define WAVE_MEDIUM_IOR(path_index, depth) \
    ordinarylight_medium_ior(path_index, depth)
#define WAVE_SET_MEDIUM_IOR(path_index, depth, value) \
    ordinarylight_set_medium_ior(path_index, depth, value)
#else
#define WAVE_MEDIUM_IOR(path_index, depth) stacks[path_index].ior[depth]
#define WAVE_SET_MEDIUM_IOR(path_index, depth, value) \
    stacks[path_index].ior[depth] = value
#endif

#if WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION
#define WAVE_STORE_PATH(path_index, path) \
    ordinarylight_store_path(path_index, path)
#define WAVE_DEACTIVATE_STORED_PATH(path_index) \
    ordinarylight_deactivate_stored_path(path_index)
#define WAVE_SECONDARY_PRIMARY_VALID(path_index) \
    ordinarylight_secondary_primary_valid(path_index)
#else
#define WAVE_STORE_PATH(path_index, path) paths[path_index] = path
#define WAVE_DEACTIVATE_STORED_PATH(path_index) \
    paths[path_index].metadata.w &= ~PATH_ACTIVE_BIT
#define WAVE_SECONDARY_PRIMARY_VALID(path_index) \
    secondary_paths[path_index].primary_throughput.w
#endif

#if WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION
#define WAVE_PROFILE_WORK(counter, amount) \
    ordinarylight_profile_work(counter, amount)
#define WAVE_INTEGRATE_VOLUMES(origin, direction, distance, radiance, throughput) \
    ordinarylight_integrate_secondary_volumes( \
        origin, direction, distance, radiance, throughput)
#else
#define WAVE_PROFILE_WORK(counter, amount) profileWork(counter, amount)
#define WAVE_INTEGRATE_VOLUMES(origin, direction, distance, radiance, throughput) \
    integrateVolumesBeforeSurface( \
        origin, direction, distance, radiance, throughput)
#endif

#ifndef WAVE_ORDINARYSHADE_PRIMARY_CAMERA
#define WAVE_ORDINARYSHADE_PRIMARY_CAMERA 1
#endif
#ifndef WAVE_GROUP_SWIZZLE_WIDTH
#define WAVE_GROUP_SWIZZLE_WIDTH 1
#endif
#ifndef WAVE_ORDINARYSHADE_PRIMARY_STATE
#define WAVE_ORDINARYSHADE_PRIMARY_STATE 1
#endif
#ifndef WAVE_ORDINARYSHADE_PRIMARY_SURFACE
#define WAVE_ORDINARYSHADE_PRIMARY_SURFACE 1
#endif
#ifndef WAVE_ORDINARYSHADE_PRIMARY_TEXTURE_STATE
#define WAVE_ORDINARYSHADE_PRIMARY_TEXTURE_STATE 1
#endif
#ifndef WAVE_ORDINARYSHADE_PRIMARY_OUTPUT
#define WAVE_ORDINARYSHADE_PRIMARY_OUTPUT 1
#endif
#ifndef WAVE_ORDINARYSHADE_PRIMARY_TRANSMISSION
#define WAVE_ORDINARYSHADE_PRIMARY_TRANSMISSION 1
#endif
#ifndef WAVE_ORDINARYSHADE_PRIMARY_CONTINUATION
#define WAVE_ORDINARYSHADE_PRIMARY_CONTINUATION 1
#endif
#if WAVE_ORDINARYSHADE_PRIMARY_CAMERA || WAVE_ORDINARYSHADE_PRIMARY_STATE \
        || WAVE_ORDINARYSHADE_PRIMARY_SURFACE \
        || WAVE_ORDINARYSHADE_PRIMARY_TEXTURE_STATE \
        || WAVE_ORDINARYSHADE_PRIMARY_OUTPUT \
        || WAVE_ORDINARYSHADE_PRIMARY_TRANSMISSION \
        || WAVE_ORDINARYSHADE_PRIMARY_CONTINUATION
#include "ordinaryshade_primary.glsl"
#endif

#if WAVE_MEGAKERNEL || WAVE_HYBRID
bool ordinarylightSecondaryBounce(
    inout WavePathState path, inout vec3 origin, inout vec3 direction,
    uint path_index, inout uint medium_depth, inout uint rng,
    inout float cone_width, inout float cone_spread)
{
    uint bounce = pathBounce(path);
#if WAVE_WORK_COUNTERS
    profile_bounce = bounce;
    WAVE_PROFILE_WORK(0u, 1u);
#endif
#if WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION
        bool surface_hit = false;
        float distance = 1.0e30;
        uint primitive = 0u;
        vec2 barycentrics = vec2(0.0);
        ordinarylight_secondary_trace_query(
            origin, direction, surface_hit, distance, primitive,
            barycentrics);
#else
        rayQueryEXT query;
        rayQueryInitializeEXT(
            query, scene_tlas, gl_RayFlagsOpaqueEXT, 0x01,
            origin, 0.001, direction, 1.0e30
        );
        while (rayQueryProceedEXT(query)) {}
        bool surface_hit = rayQueryGetIntersectionTypeEXT(query, true)
            == gl_RayQueryCommittedIntersectionTriangleEXT;
        float distance = surface_hit
            ? rayQueryGetIntersectionTEXT(query, true) : 1.0e30;
#endif
        WAVE_INTEGRATE_VOLUMES(
            origin, direction, distance,
            path.radiance.rgb, path.throughput.rgb);
#if WAVE_ORDINARYSHADE_SECONDARY_CONTROL
        if (!ordinarylight_secondary_throughput_visible(
                path.throughput.rgb)) {
#else
        if (max(path.throughput.r,
                max(path.throughput.g, path.throughput.b)) < 1e-4) {
#endif
#if WAVE_ORDINARYSHADE_SECONDARY_CONTROL
            path.metadata.w = ordinarylight_primary_deactivate(
                path.metadata.w);
#else
            path.metadata.w &= ~PATH_ACTIVE_BIT;
#endif
            return false;
        }
        if (!surface_hit) {
#if WAVE_WORK_COUNTERS
            WAVE_PROFILE_WORK(4u, 1u);
#endif
#if WAVE_ORDINARYSHADE_EMISSIVE_MIS
            float environment_mis = ordinarylight_environment_miss_mis(
                (path.metadata.w & PATH_PREVIOUS_DIFFUSE_BIT) != 0u,
                push.environment_samples, pathPreviousPdf(path),
                (path.metadata.w & PATH_PREVIOUS_UNIFIED_NEE_BIT) != 0u,
                unifiedAreaDomainProbability());
#else
            float environment_mis = 1.0;
            if ((path.metadata.w & PATH_PREVIOUS_DIFFUSE_BIT) != 0u
                    && push.environment_samples > 0u) {
                float pdf = pathPreviousPdf(path);
                float light_pdf = pdf * float(push.environment_samples);
                if ((path.metadata.w & PATH_PREVIOUS_UNIFIED_NEE_BIT) != 0u)
                    light_pdf = pdf * (1.0 - unifiedAreaDomainProbability());
                environment_mis = powerHeuristic(pdf, light_pdf);
            }
#endif
#if WAVE_ORDINARYSHADE_SECONDARY_CONTROL
            path.radiance.rgb += ordinarylight_secondary_miss_contribution(
                path.throughput.rgb, environmentColor(direction),
                environment_mis);
            path.metadata.w = ordinarylight_primary_deactivate(
                path.metadata.w);
#else
            path.radiance.rgb += path.throughput.rgb
                * environmentColor(direction) * environment_mis;
            path.metadata.w &= ~PATH_ACTIVE_BIT;
#endif
            return false;
        }
#if WAVE_WORK_COUNTERS
        WAVE_PROFILE_WORK(3u, 1u);
#endif

#if !WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION
        uint primitive = rayQueryGetIntersectionPrimitiveIndexEXT(query, true)
            + rayQueryGetIntersectionInstanceCustomIndexEXT(query, true);
        vec2 barycentrics = rayQueryGetIntersectionBarycentricsEXT(query, true);
#endif
#if WAVE_ORDINARYSHADE_SECONDARY_SURFACE
        cone_width = ordinarylight_secondary_cone_width(
            cone_width, distance, cone_spread);
#else
        cone_width += distance * cone_spread;
#endif
#if WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION
        vec3 a = ordinarylight_secondary_vertex_position(primitive, 0u);
        vec3 b = ordinarylight_secondary_vertex_position(primitive, 1u);
        vec3 c = ordinarylight_secondary_vertex_position(primitive, 2u);
        VertexAttributeData attribute_a =
            ordinarylight_secondary_vertex_attribute(primitive, 0u);
        VertexAttributeData attribute_b =
            ordinarylight_secondary_vertex_attribute(primitive, 1u);
        VertexAttributeData attribute_c =
            ordinarylight_secondary_vertex_attribute(primitive, 2u);
#else
        vec3 a = vertices[primitive * 3u + 0u].xyz;
        vec3 b = vertices[primitive * 3u + 1u].xyz;
        vec3 c = vertices[primitive * 3u + 2u].xyz;
        VertexAttributeData attribute_a = attributes[primitive * 3u + 0u];
        VertexAttributeData attribute_b = attributes[primitive * 3u + 1u];
        VertexAttributeData attribute_c = attributes[primitive * 3u + 2u];
#endif
#if WAVE_ORDINARYSHADE_SECONDARY_SURFACE
        vec3 hit = ordinarylight_primary_hit_position(
            origin, direction, distance);
        vec3 geometric_normal = ordinarylight_primary_geometric_normal(a, b, c);
        vec3 weights = ordinarylight_primary_barycentric_weights(barycentrics);
        vec3 shading_normal = ordinarylight_primary_shading_normal(
            attribute_a.normal.xyz,
            attribute_b.normal.xyz,
            attribute_c.normal.xyz,
            weights, geometric_normal);
        bool entering = ordinarylight_primary_is_entering(
            direction, geometric_normal);
        vec3 normal = ordinarylight_primary_oriented_normal(
            shading_normal, entering);
#else
        vec3 hit = origin + distance * direction;
        vec3 geometric_normal = normalize(cross(b - a, c - a));
        vec3 weights = vec3(
            1.0 - barycentrics.x - barycentrics.y,
            barycentrics.x, barycentrics.y);
        vec3 shading_normal = normalize(
            attribute_a.normal.xyz * weights.x
            + attribute_b.normal.xyz * weights.y
            + attribute_c.normal.xyz * weights.z);
        if (dot(shading_normal, geometric_normal) < 0.0)
            shading_normal = -shading_normal;
        bool entering = dot(direction, geometric_normal) < 0.0;
        vec3 normal = entering ? shading_normal : -shading_normal;
#endif
#if WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION
        MaterialData material = ordinarylight_secondary_material(primitive);
#else
        MaterialData material = materials[primitive];
#endif
#if WAVE_SER
#if WAVE_ORDINARYSHADE_SECONDARY_CONTROL
        uint ser_hint = ordinarylight_secondary_ser_hint(
            material.attenuation_transmission.a,
            material.emission_metallic.a, material.emission_metallic.w,
            material.base_roughness.w, materialHasTextures(material));
#else
        uint ser_hint = uint(material.attenuation_transmission.a > 0.001);
        ser_hint |= uint(material.emission_metallic.a > 0.5) << 1u;
        ser_hint |= uint(material.emission_metallic.w > 0.5) << 2u;
        ser_hint |= uint(clamp(
            material.base_roughness.w * 7.0, 0.0, 7.0)) << 3u;
        ser_hint |= uint(materialHasTextures(material)) << 6u;
#endif
#if WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION
        ordinarylight_secondary_reorder(ser_hint);
#else
        reorderThreadNV(ser_hint, 7u);
#endif
#endif
#if !WAVE_UNTEXTURED_SECONDARY
#if WAVE_ORDINARYSHADE_SECONDARY_SURFACE
        vec2 uv0 = ordinarylight_primary_interpolate_vec4(
            attribute_a.texcoord, attribute_b.texcoord,
            attribute_c.texcoord, weights).xy;
        vec2 uv1 = ordinarylight_primary_interpolate_vec4(
            attribute_a.texcoord, attribute_b.texcoord,
            attribute_c.texcoord, weights).zw;
#else
        vec2 uv0 =
            attribute_a.texcoord.xy * weights.x
            + attribute_b.texcoord.xy * weights.y
            + attribute_c.texcoord.xy * weights.z;
        vec2 uv1 =
            attribute_a.texcoord.zw * weights.x
            + attribute_b.texcoord.zw * weights.y
            + attribute_c.texcoord.zw * weights.z;
#endif
#if WAVE_ORDINARYSHADE_SECONDARY_SURFACE
        bool textured_material = materialHasTextures(material);
        float uv0_footprint = ordinarylight_secondary_texture_footprint(
            cone_width, attribute_a.normal.w,
            textured_material);
        float uv1_density = ordinarylight_primary_uv_density(
            a, b, c,
            attribute_a.texcoord.zw, attribute_b.texcoord.zw,
            attribute_c.texcoord.zw);
        float uv1_footprint = ordinarylight_secondary_texture_footprint(
            cone_width, uv1_density, textured_material);
#else
        float uv0_footprint = 0.0;
        float uv1_footprint = 0.0;
        if (materialHasTextures(material)) {
            uv0_footprint = cone_width
                * attribute_a.normal.w;
            uv1_footprint = cone_width * triangleUvDensity(
                a, b, c,
                attribute_a.texcoord.zw, attribute_b.texcoord.zw,
                attribute_c.texcoord.zw);
        }
#endif
        applyMaterialTextures(
            material, uv0, uv1, uv0_footprint, uv1_footprint);
#if WAVE_ORDINARYSHADE_SECONDARY_SURFACE
        vec4 tangent_data = ordinarylight_primary_interpolate_vec4(
            attribute_a.tangent, attribute_b.tangent,
            attribute_c.tangent, weights);
#else
        vec4 tangent_data =
            attribute_a.tangent * weights.x
            + attribute_b.tangent * weights.y
            + attribute_c.tangent * weights.z;
#endif
        if (textureBindingUsesUv1(material.texture_indices.w))
#if WAVE_ORDINARYSHADE_SECONDARY_SURFACE
            tangent_data = ordinarylight_primary_triangle_tangent(
#else
            tangent_data = triangleTangent(
#endif
                a, b, c,
                attribute_a.texcoord.zw, attribute_b.texcoord.zw,
                attribute_c.texcoord.zw,
                shading_normal);
        shading_normal = applyNormalTexture(
            material, uv0, uv1, uv0_footprint, uv1_footprint,
            shading_normal, tangent_data);
#if WAVE_ORDINARYSHADE_SECONDARY_SURFACE
        shading_normal = ordinarylight_secondary_correct_shading_normal(
            shading_normal, geometric_normal);
        normal = ordinarylight_primary_oriented_normal(
            shading_normal, entering);
#else
        if (dot(shading_normal, geometric_normal) < 0.0)
            shading_normal = -shading_normal;
        normal = entering ? shading_normal : -shading_normal;
#endif
#else
        material.texture_parameters.w = 1.0;
#endif
        // WAVE_MATERIAL_APPLICATION_SECONDARY
#if WAVE_ORDINARYSHADE_SECONDARY_CONTROL
        if (ordinarylight_secondary_capture_hit(
                path.metadata.w, bounce,
                WAVE_SECONDARY_PRIMARY_VALID(path_index))) {
#else
        if ((path.metadata.w & PATH_INDIRECT_CAPTURE_BIT) != 0u && bounce == 1u
                && WAVE_SECONDARY_PRIMARY_VALID(path_index) > 0.5) {
#endif
#if WAVE_ORDINARYSHADE_SECONDARY_CONTROL
            vec4 captured_position =
                ordinarylight_secondary_capture_position(hit);
#else
            vec4 captured_position = vec4(hit, 1.0);
#endif
#if WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION
            ordinarylight_store_secondary_hit(
                path_index, captured_position, normal);
#else
            secondary_paths[path_index].position_valid = captured_position;
            secondary_paths[path_index].normal_pdf.xyz = normal;
#endif
        }

#if WAVE_ORDINARYSHADE_SECONDARY_CONTROL
        bool emission_visible = ordinarylight_secondary_emission_visible(
            entering, material.ior_distance.w);
#else
        bool emission_visible = entering || material.ior_distance.w > 0.5;
#endif
        if (emission_visible) {
#if WAVE_ORDINARYSHADE_EMISSIVE_MIS
            float emission_mis = ordinarylight_emissive_hit_mis(
                (path.metadata.w & PATH_PREVIOUS_DIFFUSE_BIT) != 0u,
                material.emission_metallic.rgb, a, b, c, geometric_normal,
                direction, distance, material.ior_distance.w > 0.5,
                push.area_light_weight, push.secondary_area_light_samples,
                (path.metadata.w & PATH_PREVIOUS_UNIFIED_NEE_BIT) != 0u,
                unifiedAreaDomainProbability(), pathPreviousPdf(path));
#else
            float emission_mis = 1.0;
            if ((path.metadata.w & PATH_PREVIOUS_DIFFUSE_BIT) != 0u
                    && dot(material.emission_metallic.rgb,
                           material.emission_metallic.rgb) > 0.0) {
                float area = 0.5 * length(cross(b - a, c - a));
                float raw_cosine = dot(geometric_normal, -direction);
                float light_cosine = material.ior_distance.w > 0.5
                    ? abs(raw_cosine) : max(raw_cosine, 0.0);
                float light_power = dot(material.emission_metallic.rgb,
                    vec3(0.2126, 0.7152, 0.0722));
                float selection_pdf = area * light_power /
                    max(push.area_light_weight, 0.000001);
                float light_pdf = selection_pdf * distance * distance /
                    max(light_cosine * area, 0.000001);
                float sampled_light_pdf = light_pdf * float(max(
                    push.secondary_area_light_samples, 1u));
                if ((path.metadata.w & PATH_PREVIOUS_UNIFIED_NEE_BIT) != 0u)
                    sampled_light_pdf = light_pdf
                        * unifiedAreaDomainProbability();
                emission_mis = powerHeuristic(
                    pathPreviousPdf(path), sampled_light_pdf);
            }
#endif
#if WAVE_ORDINARYSHADE_EMISSIVE_MIS
            path.radiance.rgb += ordinarylight_emission_contribution(
                path.throughput.rgb, material.emission_metallic.rgb,
                emission_mis);
#else
            path.radiance.rgb += path.throughput.rgb
                * material.emission_metallic.rgb * emission_mis;
#endif
        }

#if WAVE_ORDINARYSHADE_SECONDARY_CONTROL
        uint next_bounce = ordinarylight_secondary_next_bounce(bounce);
        if (ordinarylight_secondary_bounce_terminates(
                next_bounce, push.max_bounces)) {
#else
        uint next_bounce = bounce + 1u;
        if (next_bounce >= push.max_bounces) {
#endif
            setPathBounce(path, next_bounce);
#if WAVE_ORDINARYSHADE_SECONDARY_CONTROL
            path.metadata.w = ordinarylight_primary_deactivate(
                path.metadata.w);
#else
            path.metadata.w &= ~PATH_ACTIVE_BIT;
#endif
            return false;
        }

        vec3 next_direction;
        float bsdf_pdf = 0.0;
#if WAVE_OPAQUE_SCENE
        float transmission = 0.0;
#else
        float transmission = material.attenuation_transmission.a;
#endif
        if (transmission > 0.001) {
#if WAVE_ORDINARYSHADE_SECONDARY_TRANSMISSION
            float current_ior = WAVE_MEDIUM_IOR(
                path_index, medium_depth - 1u);
            float previous_medium_ior = medium_depth > 1u
                ? WAVE_MEDIUM_IOR(path_index, medium_depth - 2u) : 1.0;
            float target_ior = ordinarylight_secondary_target_ior(
                entering, material.ior_distance.x, previous_medium_ior,
                medium_depth);
            vec3 refracted = ordinarylight_secondary_refracted_direction(
                direction, normal, current_ior, target_ior);
            next_direction = ordinarylight_primary_resolve_transmission_direction(
                refracted, direction, normal);
            if (ordinarylight_secondary_enters_medium(
                    refracted, entering, medium_depth,
                    WAVE_MAX_MEDIUM_STACK_DEPTH))
                WAVE_SET_MEDIUM_IOR(path_index, medium_depth, target_ior);
            medium_depth = ordinarylight_secondary_medium_depth(
                refracted, entering, medium_depth,
                WAVE_MAX_MEDIUM_STACK_DEPTH);
            path.throughput.rgb =
                ordinarylight_secondary_transmission_throughput(
                    path.throughput.rgb, material.base_roughness.rgb,
                    transmission);
#else
            float current_ior = WAVE_MEDIUM_IOR(
                path_index, medium_depth - 1u);
            float target_ior = entering ? max(material.ior_distance.x, 1.0001)
                : (medium_depth > 1u
                    ? WAVE_MEDIUM_IOR(path_index, medium_depth - 2u) : 1.0);
            next_direction = refract(
                direction, normal, current_ior / target_ior);
            if (dot(next_direction, next_direction) < 0.01)
                next_direction = reflect(direction, normal);
            else if (entering && medium_depth < WAVE_MAX_MEDIUM_STACK_DEPTH) {
                WAVE_SET_MEDIUM_IOR(path_index, medium_depth, target_ior);
                medium_depth++;
            } else if (!entering && medium_depth > 1u) {
                medium_depth--;
            }
            path.throughput.rgb *= mix(
                vec3(1.0), material.base_roughness.rgb, 0.2) * transmission;
#endif
        } else {
#if WAVE_ORDINARYSHADE_SECONDARY_CONTROL
            float nee_probability = ordinarylight_secondary_nee_probability(
                push.secondary_nee_probability);
#else
            float nee_probability = clamp(
                push.secondary_nee_probability, 0.000001, 1.0);
#endif
            bool sample_direct = selectSecondaryNee(
                nee_probability, path.metadata.x, path.metadata.y,
                next_bounce);
            if (sample_direct) {
                vec3 direct = samplePointLights(
                    hit, normal, direction, material);
                if (push.unified_secondary_nee != 0u) {
                    direct += sampleUnifiedSecondaryLight(
                        hit, normal, direction, material, rng);
                } else {
#if WAVE_ORDINARYSHADE_SECONDARY_CONTROL
                    uint light_samples =
                        ordinarylight_secondary_area_sample_count(
                            push.secondary_area_light_samples);
#else
                    uint light_samples = clamp(
                        push.secondary_area_light_samples, 1u, 16u);
#endif
                    vec3 area_direct = vec3(0.0);
                    for (uint sample_index = 0u;
                            sample_index < light_samples; ++sample_index)
                        area_direct += sampleAreaLight(
                            hit, normal, direction, material, rng,
                            sample_index, light_samples);
#if WAVE_ORDINARYSHADE_SECONDARY_CONTROL
                    direct += ordinarylight_secondary_average_contribution(
                        area_direct, light_samples);
                    uint environment_samples =
                        ordinarylight_secondary_environment_sample_count(
                            push.environment_samples);
#else
                    direct += area_direct / float(light_samples);
                    uint environment_samples = min(
                        push.environment_samples, 4u);
#endif
                    if (environment_samples > 0u) {
                        vec3 environment_direct = vec3(0.0);
                        for (uint sample_index = 0u;
                                sample_index < environment_samples;
                                ++sample_index)
                            environment_direct += sampleEnvironment(
                                hit, normal, direction, material, rng,
                                environment_samples);
#if WAVE_ORDINARYSHADE_SECONDARY_CONTROL
                        direct += ordinarylight_secondary_average_contribution(
                            environment_direct, environment_samples);
#else
                        direct += environment_direct /
                            float(environment_samples);
#endif
                    }
                }
#if WAVE_ORDINARYSHADE_SECONDARY_TRANSPORT
                path.radiance.rgb += ordinarylight_secondary_direct_contribution(
                    path.throughput.rgb, direct, nee_probability);
#else
                path.radiance.rgb += path.throughput.rgb
                    * direct / nee_probability;
#endif
            }
            vec3 bsdf_weight;
            samplePbr(material, normal, direction, rng,
                next_direction, bsdf_weight, bsdf_pdf);
#if WAVE_ORDINARYSHADE_SECONDARY_TRANSPORT
            path.throughput.rgb = ordinarylight_secondary_scatter_throughput(
                path.throughput.rgb, bsdf_weight);
            cone_spread = ordinarylight_primary_scattered_cone_spread(
                cone_spread, material.base_roughness.a);
#else
            path.throughput.rgb *= bsdf_weight;
            cone_spread += material.base_roughness.a * 0.25;
#endif
        }

#if WAVE_ORDINARYSHADE_SECONDARY_TRANSPORT
        next_direction = ordinarylight_primary_continuation_direction(
            next_direction);
        setPathBounce(path, next_bounce);
        path.metadata.w = ordinarylight_primary_continuation_flags(
            path.metadata.w, medium_depth, transmission,
            push.unified_secondary_nee != 0u);
        setPathPreviousPdf(path, ordinarylight_primary_previous_pdf(
            pathPreviousPdf(path), bsdf_pdf, transmission));
#else
        next_direction = normalize(next_direction);
        setPathBounce(path, next_bounce);
        path.metadata.w = PATH_ACTIVE_BIT | (medium_depth << 8u)
            | (path.metadata.w & PATH_INDIRECT_CAPTURE_BIT);
        if (transmission <= 0.001) {
            path.metadata.w |= PATH_PREVIOUS_DIFFUSE_BIT;
            if (push.unified_secondary_nee != 0u)
                path.metadata.w |= PATH_PREVIOUS_UNIFIED_NEE_BIT;
            setPathPreviousPdf(path, bsdf_pdf);
        }
#endif
#if WAVE_ORDINARYSHADE_SECONDARY_TRANSPORT
        if (ordinarylight_secondary_roulette_enabled(
                push.russian_roulette_start, next_bounce, transmission)) {
            float survival = ordinarylight_secondary_survival_probability(
                path.throughput.rgb,
                push.russian_roulette_min_survival);
            if (!ordinarylight_secondary_survives(
                    randomFloat(rng), survival)) {
                path.metadata.w = ordinarylight_primary_deactivate(
                    path.metadata.w);
                return false;
            }
            path.throughput.rgb = ordinarylight_secondary_survival_throughput(
                path.throughput.rgb, survival);
        }
#else
        if (push.russian_roulette_start > 0u
                && next_bounce >= push.russian_roulette_start
                && transmission <= 0.001) {
            float survival = clamp(max(path.throughput.r,
                max(path.throughput.g, path.throughput.b)),
                push.russian_roulette_min_survival, 0.95);
            if (randomFloat(rng) >= survival) {
                path.metadata.w &= ~PATH_ACTIVE_BIT;
                return false;
            }
            path.throughput.rgb /= survival;
        }
#endif
#if WAVE_ORDINARYSHADE_SECONDARY_TRANSPORT
        origin = ordinarylight_primary_continuation_origin(
            hit, next_direction);
#else
        origin = hit + next_direction * 0.002;
#endif
        direction = next_direction;
        return true;
}

void traceRemaining(
    inout WavePathState path, inout vec3 origin, inout vec3 direction,
    uint path_index, uint medium_depth, inout uint rng,
    inout float cone_width, inout float cone_spread)
{
#if WAVE_ORDINARYSHADE_SECONDARY_CONTROL
    uint stop_bounce = ordinarylight_secondary_stop_bounce(
        WAVE_HYBRID != 0, push.inline_bounces, push.max_bounces);
#else
    uint stop_bounce = WAVE_HYBRID != 0
        ? min(push.inline_bounces, push.max_bounces) : push.max_bounces;
#endif
#if WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION
    ordinarylight_trace_remaining(
        path, origin, direction, path_index, medium_depth, rng,
        cone_width, cone_spread, stop_bounce);
#else
    while (pathBounce(path) < stop_bounce) {
        if (!ordinarylightSecondaryBounce(
                path, origin, direction, path_index, medium_depth, rng,
                cone_width, cone_spread))
            break;
    }
    setPathRng(path, rng);
#endif
}
#endif

#if !WAVE_CONTINUATION
void processPrimaryPixel(uvec2 local_pixel)
{
#if WAVE_WORK_COUNTERS
    profile_bounce = 0u;
#endif
    if (any(greaterThanEqual(local_pixel, push.tile_frame.xy)))
        return;
    uvec2 pixel = push.image_tile.zw + local_pixel;
    if (any(greaterThanEqual(pixel, push.image_tile.xy)))
        return;
    uint path_index = local_pixel.y * push.tile_frame.x + local_pixel.x;
    if (path_index >= output_queue.capacity)
        return;

    uint pixel_index = pixel.y * push.image_tile.x + pixel.x;
    if (push.restir_di != 0u)
        storeCurrentDirectLightReservoir(
            pixel_index, emptyDirectLightReservoir());
    // camera.origin.w duplicates tile_frame.z. Including both XOR terms
    // cancels frame variation and leaves a fixed screen-space noise pattern.
#if WAVE_ORDINARYSHADE_PRIMARY_STATE
    uint rng = ordinarylight_primary_rng_seed(
        pixel_index, push.tile_frame.z, push.tile_frame.w);
    rng = ordinarylight_primary_rng_step(rng);
    float jitter_x = ordinarylight_primary_rng_value(rng);
    rng = ordinarylight_primary_rng_step(rng);
    float jitter_y = ordinarylight_primary_rng_value(rng);
    vec2 jitter = vec2(jitter_x, jitter_y);
#else
    uint rng = hashValue(pixel_index ^ hashValue(push.tile_frame.z)
                         ^ hashValue(push.tile_frame.w + 1u));
    vec2 jitter = vec2(randomFloat(rng), randomFloat(rng));
#endif
    vec2 ndc = ((vec2(pixel) + jitter) / vec2(push.image_tile.xy)) * 2.0 - 1.0;
    float aspect = float(push.image_tile.x) / float(push.image_tile.y);
    int camera_projection = int(camera.up.w + 0.5);
#if WAVE_ORDINARYSHADE_PRIMARY_CAMERA
    vec3 ray_origin = ordinarylight_primary_ray_origin(
        camera.origin.xyz, camera.right.xyz, camera.up.xyz,
        ndc, aspect, camera_projection);
    vec3 incoming = ordinarylight_primary_ray_direction(
        camera.forward.xyz, camera.right.xyz, camera.up.xyz,
        ndc, aspect, camera_projection);
#else
    vec3 ray_origin = camera.origin.xyz;
    vec3 incoming;
    if (camera_projection == 1) {
        ray_origin += ndc.x * aspect * camera.right.xyz
            - ndc.y * camera.up.xyz;
        incoming = normalize(camera.forward.xyz);
    } else if (camera_projection == 2) {
        float yaw = ndc.x * length(camera.right.xyz);
        float pitch = -ndc.y * length(camera.up.xyz);
        incoming = normalize(
            normalize(camera.forward.xyz) * cos(pitch) * cos(yaw)
            + normalize(camera.right.xyz) * cos(pitch) * sin(yaw)
            + normalize(camera.up.xyz) * sin(pitch));
    } else {
        incoming = normalize(camera.forward.xyz
            + ndc.x * aspect * camera.right.xyz - ndc.y * camera.up.xyz);
    }
#endif

    WavePathState path;
    path.throughput = vec4(1.0);
    path.radiance = vec4(0.0);
    bool indirect_capture_pixel = push.indirect_secondary_capture != 0u;
    if (indirect_capture_pixel && push.indirect_capture_stride > 1u) {
        uvec2 capture_offset = uvec2(push.indirect_capture_stride / 2u);
        indirect_capture_pixel = all(equal(
            pixel % push.indirect_capture_stride, capture_offset));
    }
#if WAVE_ORDINARYSHADE_PRIMARY_STATE
    path.metadata = uvec4(
        pixel_index,
        ordinarylight_primary_path_identity(
            push.tile_frame.z, push.tile_frame.w),
        rng, ordinarylight_primary_path_flags(indirect_capture_pixel));
#else
    path.metadata = uvec4(
        pixel_index,
        (push.tile_frame.z << 8u) | (push.tile_frame.w & 255u),
        rng, 257u | (indirect_capture_pixel ? PATH_INDIRECT_CAPTURE_BIT : 0u));
#endif
    setPathBounce(path, 0u);
#if !WAVE_OPAQUE_SCENE
#if WAVE_ORDINARYSHADE_PRIMARY_TRANSMISSION
    WAVE_SET_MEDIUM_IOR(
        path_index, 0u, ordinarylight_primary_initial_medium_ior());
#else
    WAVE_SET_MEDIUM_IOR(path_index, 0u, 1.0);
#endif
#endif
    if (indirect_capture_pixel) {
#if WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION
        ordinarylight_clear_secondary_path(path_index);
#else
        secondary_paths[path_index].position_valid = vec4(0.0);
        secondary_paths[path_index].normal_pdf = vec4(0.0);
        secondary_paths[path_index].primary_throughput = vec4(0.0);
        secondary_paths[path_index].primary_radiance = vec4(0.0);
#endif
    }

    rayQueryEXT query;
#if WAVE_WORK_COUNTERS
    profileWork(0u, 1u);
#endif
    rayQueryInitializeEXT(
        query, scene_tlas, gl_RayFlagsOpaqueEXT, 0x01,
        ray_origin, 0.001, incoming, 1.0e30
    );
    while (rayQueryProceedEXT(query)) {}
    bool surface_hit = rayQueryGetIntersectionTypeEXT(query, true)
        == gl_RayQueryCommittedIntersectionTriangleEXT;
    float distance = surface_hit
        ? rayQueryGetIntersectionTEXT(query, true) : 1.0e30;
    integrateVolumesBeforeSurface(
        ray_origin, incoming, distance,
        path.radiance.rgb, path.throughput.rgb);
    if (!surface_hit || max(path.throughput.r,
            max(path.throughput.g, path.throughput.b)) < 1e-4) {
#if WAVE_WORK_COUNTERS
        profileWork(4u, 1u);
#endif
        if (push.gbuffer_enabled != 0u) {
#if WAVE_ORDINARYSHADE_PRIMARY_OUTPUT
            imageStore(position_image, ivec2(pixel),
                ordinarylight_primary_invalid_position());
            imageStore(normal_image, ivec2(pixel),
                ordinarylight_primary_packed_payload(0u));
            imageStore(material_image, ivec2(pixel),
                ordinarylight_primary_invalid_material());
#else
            imageStore(position_image, ivec2(pixel), vec4(-1.0));
            imageStore(normal_image, ivec2(pixel), uvec4(0u));
            imageStore(material_image, ivec2(pixel), uvec4(0xffffffffu));
#endif
        }
        if (!surface_hit)
            path.radiance.rgb += path.throughput.rgb
                * environmentColor(incoming);
#if WAVE_ORDINARYSHADE_PRIMARY_OUTPUT
        path.metadata.w = ordinarylight_primary_deactivate(path.metadata.w);
#else
        path.metadata.w &= ~PATH_ACTIVE_BIT;
#endif
        WAVE_STORE_PATH(path_index, path);
        return;
    }
#if WAVE_WORK_COUNTERS
    profileWork(3u, 1u);
#endif
    uint primitive = rayQueryGetIntersectionPrimitiveIndexEXT(query, true)
        + rayQueryGetIntersectionInstanceCustomIndexEXT(query, true);
    vec2 barycentrics = rayQueryGetIntersectionBarycentricsEXT(query, true);
#if WAVE_ORDINARYSHADE_PRIMARY_SURFACE
    float cone_spread = ordinarylight_primary_cone_spread(
        camera.up.xyz, push.image_tile.y);
#else
    float cone_spread = 2.0 * length(camera.up.xyz)
        / float(max(push.image_tile.y, 1u));
#endif
    float cone_width = distance * cone_spread;
    vec3 a = vertices[primitive * 3u + 0u].xyz;
    vec3 b = vertices[primitive * 3u + 1u].xyz;
    vec3 c = vertices[primitive * 3u + 2u].xyz;
#if WAVE_ORDINARYSHADE_PRIMARY_SURFACE
    vec3 position = ordinarylight_primary_hit_position(
        ray_origin, incoming, distance);
    vec3 geometric_normal = ordinarylight_primary_geometric_normal(a, b, c);
    vec3 weights = ordinarylight_primary_barycentric_weights(barycentrics);
    vec3 shading_normal = ordinarylight_primary_shading_normal(
        attributes[primitive * 3u + 0u].normal.xyz,
        attributes[primitive * 3u + 1u].normal.xyz,
        attributes[primitive * 3u + 2u].normal.xyz,
        weights, geometric_normal);
    bool entering = ordinarylight_primary_is_entering(
        incoming, geometric_normal);
    vec3 normal = ordinarylight_primary_oriented_normal(
        shading_normal, entering);
#else
    vec3 position = ray_origin + distance * incoming;
    vec3 geometric_normal = normalize(cross(b - a, c - a));
    vec3 weights = vec3(1.0 - barycentrics.x - barycentrics.y,
                        barycentrics.x, barycentrics.y);
    vec3 shading_normal = normalize(
        attributes[primitive * 3u + 0u].normal.xyz * weights.x
        + attributes[primitive * 3u + 1u].normal.xyz * weights.y
        + attributes[primitive * 3u + 2u].normal.xyz * weights.z);
    if (dot(shading_normal, geometric_normal) < 0.0)
        shading_normal = -shading_normal;
    bool entering = dot(incoming, geometric_normal) < 0.0;
    vec3 normal = entering ? shading_normal : -shading_normal;
#endif
    MaterialData material = materials[primitive];
#if WAVE_SER
    uint ser_hint = uint(material.attenuation_transmission.a > 0.001);
    ser_hint |= uint(material.emission_metallic.a > 0.5) << 1u;
    ser_hint |= uint(material.emission_metallic.w > 0.5) << 2u;
    ser_hint |= uint(clamp(
        material.base_roughness.w * 7.0, 0.0, 7.0)) << 3u;
    ser_hint |= uint(materialHasTextures(material)) << 6u;
#if WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION
    ordinarylight_secondary_reorder(ser_hint);
#else
    reorderThreadNV(ser_hint, 7u);
#endif
#endif
    uint material_signature = restirMaterialSignature(material);
    if (push.object_effect_ranges[0].x < push.object_effect_ranges[0].y) {
        material_signature &= 0x1fffffffu;
        for (uint effect_index = 0u; effect_index < 4u; ++effect_index) {
            uvec4 ranges = push.object_effect_ranges[effect_index >> 1u];
            uint start = ranges[(effect_index & 1u) * 2u];
            uint end = ranges[(effect_index & 1u) * 2u + 1u];
            if (start < end && primitive >= start && primitive < end) {
                material_signature |= (effect_index + 1u) << 29u;
                break;
            }
        }
    }
#if WAVE_UNTEXTURED_PRIMARY
    const bool material_textured = false;
    material.texture_parameters.w = 1.0;
#else
    bool material_textured = materialHasTextures(material);
#if WAVE_ORDINARYSHADE_PRIMARY_TEXTURE_STATE
    vec4 interpolated_uv = ordinarylight_primary_interpolate_vec4(
        attributes[primitive * 3u + 0u].texcoord,
        attributes[primitive * 3u + 1u].texcoord,
        attributes[primitive * 3u + 2u].texcoord, weights);
    vec2 uv0 = interpolated_uv.xy;
    vec2 uv1 = interpolated_uv.zw;
#else
    vec2 uv0 = attributes[primitive * 3u + 0u].texcoord.xy * weights.x
        + attributes[primitive * 3u + 1u].texcoord.xy * weights.y
        + attributes[primitive * 3u + 2u].texcoord.xy * weights.z;
    vec2 uv1 = attributes[primitive * 3u + 0u].texcoord.zw * weights.x
        + attributes[primitive * 3u + 1u].texcoord.zw * weights.y
        + attributes[primitive * 3u + 2u].texcoord.zw * weights.z;
#endif
    float uv0_footprint = 0.0;
    float uv1_footprint = 0.0;
    if (material_textured) {
        uv0_footprint = cone_width
            * attributes[primitive * 3u + 0u].normal.w;
#if WAVE_ORDINARYSHADE_PRIMARY_TEXTURE_STATE
        uv1_footprint = cone_width * ordinarylight_primary_uv_density(
#else
        uv1_footprint = cone_width * triangleUvDensity(
#endif
            a, b, c,
            attributes[primitive * 3u + 0u].texcoord.zw,
            attributes[primitive * 3u + 1u].texcoord.zw,
            attributes[primitive * 3u + 2u].texcoord.zw);
    }
    applyMaterialTextures(material, uv0, uv1, uv0_footprint, uv1_footprint);
#if WAVE_ORDINARYSHADE_PRIMARY_TEXTURE_STATE
    vec4 tangent_data = ordinarylight_primary_interpolate_vec4(
        attributes[primitive * 3u + 0u].tangent,
        attributes[primitive * 3u + 1u].tangent,
        attributes[primitive * 3u + 2u].tangent, weights);
#else
    vec4 tangent_data = attributes[primitive * 3u + 0u].tangent * weights.x
        + attributes[primitive * 3u + 1u].tangent * weights.y
        + attributes[primitive * 3u + 2u].tangent * weights.z;
#endif
    if (textureBindingUsesUv1(material.texture_indices.w))
#if WAVE_ORDINARYSHADE_PRIMARY_TEXTURE_STATE
        tangent_data = ordinarylight_primary_triangle_tangent(
#else
        tangent_data = triangleTangent(
#endif
            a, b, c,
            attributes[primitive * 3u + 0u].texcoord.zw,
            attributes[primitive * 3u + 1u].texcoord.zw,
            attributes[primitive * 3u + 2u].texcoord.zw,
            shading_normal);
    shading_normal = applyNormalTexture(
        material, uv0, uv1, uv0_footprint, uv1_footprint,
        shading_normal, tangent_data);
#if WAVE_ORDINARYSHADE_PRIMARY_TEXTURE_STATE
    shading_normal = ordinarylight_primary_correct_mapped_normal(
        shading_normal, geometric_normal);
#else
    if (dot(shading_normal, geometric_normal) < 0.0)
        shading_normal = -shading_normal;
#endif
    normal = entering ? shading_normal : -shading_normal;
#endif

#if WAVE_ORDINARYSHADE_PRIMARY_SURFACE
#if WAVE_OPAQUE_SCENE
    float surface_class = ordinarylight_primary_surface_class(
        material.attenuation_transmission.a,
        material.emission_metallic.a, true);
#else
    float surface_class = ordinarylight_primary_surface_class(
        material.attenuation_transmission.a,
        material.emission_metallic.a, false);
#endif
#elif WAVE_OPAQUE_SCENE
    float surface_class = material.emission_metallic.a > 0.5 ? 1.0 : 0.0;
#else
    float surface_class = material.attenuation_transmission.a > 0.001 ? 2.0
        : (material.emission_metallic.a > 0.5 ? 1.0 : 0.0);
#endif
    if (push.gbuffer_enabled != 0u) {
#if WAVE_ORDINARYSHADE_PRIMARY_OUTPUT
        imageStore(position_image, ivec2(pixel),
            ordinarylight_primary_hit_position_payload(distance));
        imageStore(normal_image, ivec2(pixel),
            ordinarylight_primary_packed_payload(
                restirPackNormalClass(shading_normal, surface_class)));
        imageStore(material_image, ivec2(pixel),
            ordinarylight_primary_packed_payload(material_signature));
#else
        imageStore(position_image, ivec2(pixel), vec4(distance));
        imageStore(normal_image, ivec2(pixel), uvec4(
            restirPackNormalClass(shading_normal, surface_class)));
        imageStore(material_image, ivec2(pixel), uvec4(material_signature));
#endif
    }
#if WAVE_ORDINARYSHADE_PRIMARY_OUTPUT
    path.radiance.rgb += ordinarylight_primary_emission(
        material.emission_metallic.rgb, entering,
        material.ior_distance.w > 0.5);
#else
    if (entering || material.ior_distance.w > 0.5)
        path.radiance.rgb += material.emission_metallic.rgb;
#endif

#if WAVE_ORDINARYSHADE_PRIMARY_OUTPUT
    if (ordinarylight_primary_should_terminate(push.max_bounces)) {
#else
    if (push.max_bounces <= 1u) {
#endif
        setPathBounce(path, 1u);
#if WAVE_ORDINARYSHADE_PRIMARY_OUTPUT
        path.metadata.w = ordinarylight_primary_deactivate(path.metadata.w);
#else
        path.metadata.w &= ~PATH_ACTIVE_BIT;
#endif
        WAVE_STORE_PATH(path_index, path);
        return;
    }

    vec3 next_direction;
    float bsdf_pdf = 0.0;
#if WAVE_ORDINARYSHADE_PRIMARY_TRANSMISSION
#if WAVE_OPAQUE_SCENE
    float transmission = ordinarylight_primary_transmission(
        material.attenuation_transmission.a, true);
#else
    float transmission = ordinarylight_primary_transmission(
        material.attenuation_transmission.a, false);
#endif
#elif WAVE_OPAQUE_SCENE
    float transmission = 0.0;
#else
    float transmission = material.attenuation_transmission.a;
#endif
    uint medium_depth = 1u;
    if (transmission > 0.001) {
#if WAVE_ORDINARYSHADE_PRIMARY_TRANSMISSION
        float target_ior = ordinarylight_primary_target_ior(
            material.ior_distance.x);
        vec3 refracted_direction = ordinarylight_primary_refracted_direction(
            incoming, normal, target_ior);
        next_direction = ordinarylight_primary_resolve_transmission_direction(
            refracted_direction, incoming, normal);
        bool entered_medium = ordinarylight_primary_enters_medium(
            refracted_direction, entering);
        if (entered_medium)
            WAVE_SET_MEDIUM_IOR(path_index, 1u, target_ior);
        medium_depth = ordinarylight_primary_medium_depth(entered_medium);
        path.throughput.rgb *= ordinarylight_primary_transmission_weight(
            material.base_roughness.rgb, transmission);
#else
        float target_ior = max(material.ior_distance.x, 1.0001);
        next_direction = refract(incoming, normal, 1.0 / target_ior);
        if (dot(next_direction, next_direction) < 0.01)
            next_direction = reflect(incoming, normal);
        else if (entering) {
            WAVE_SET_MEDIUM_IOR(path_index, 1u, target_ior);
            medium_depth = 2u;
        }
        path.throughput.rgb *= mix(vec3(1.0), material.base_roughness.rgb, 0.2)
            * transmission;
#endif
    } else {
        path.radiance.rgb += samplePointLights(
            position, normal, incoming, material);
        uint light_samples = clamp(push.area_light_samples, 1u, 16u);
        if (push.restir_di != 0u && push.area_light_count > 0u) {
            uint candidate_count = clamp(
                push.restir_candidate_count, 1u, 4u);
            DirectLightReservoir reservoir = emptyDirectLightReservoir();
            for (uint sample_index = 0u; sample_index < candidate_count;
                    ++sample_index) {
                AreaLightCandidate candidate =
                    WAVE_UNIFIED_PRIMARY_RESTIR != 0u
                    ? generateUnifiedPrimaryCandidate(
                        position, normal, incoming, material, rng,
                        sample_index, candidate_count)
                    : generateAreaLightCandidate(
                        position, normal, incoming, material, rng,
                        sample_index, candidate_count);
                updateDirectLightReservoir(
                    reservoir, candidate.light_index,
                    candidate.barycentrics, candidate.target,
                    candidate.target, 1.0, randomFloat(rng));
            }
            if (push.restir_history_valid != 0u) {
                ivec2 previous_pixel;
                if (reprojectRestir(position, previous_pixel)) {
                    uint spatial_count = push.restir_spatial_reuse != 0u
                        ? clamp(push.restir_spatial_neighbors, 1u, 8u) : 0u;
                    uint spatial_rotation = hashValue(
                        pixel_index ^ push.tile_frame.z) & 7u;
                    for (uint reuse_index = 0u;
                            reuse_index <= spatial_count; ++reuse_index) {
                        ivec2 history_pixel = previous_pixel;
                        if (reuse_index > 0u) {
                            history_pixel += restirSpatialOffset(
                                reuse_index - 1u + spatial_rotation,
                                int(push.restir_spatial_radius));
                        }
                        bool inside = all(greaterThanEqual(
                            history_pixel, ivec2(0))) && all(lessThan(
                                history_pixel, ivec2(push.image_tile.xy)));
                        bool geometry_valid = false;
                        bool history_source_present = false;
                        DirectLightReservoir history =
                            emptyDirectLightReservoir();
                        if (inside) {
                            uint previous_index = uint(history_pixel.y)
                                * push.image_tile.x + uint(history_pixel.x);
                            history = loadPreviousDirectLightReservoir(
                                previous_index);
                            history_source_present =
                                history.data.x != 0xffffffffu;
                            if (history_source_present) {
                                geometry_valid = restirHistorySurfaceMatches(
                                    history_pixel, position, shading_normal,
                                    distance, surface_class,
                                    material_signature, reuse_index == 0u);
                            }
                        }
                        if (geometry_valid) {
                            float current_count =
                                unpackHalf2x16(reservoir.data.w).y;
                            float remaining_history = max(
                                float(push.restir_history_limit)
                                    - current_count, 0.0);
                            float source_history_limit = spatial_count > 0u
                                ? min(remaining_history, 1.0)
                                : remaining_history;
                            history = limitDirectLightReservoir(
                                history, source_history_limit);
                            if (unpackHalf2x16(history.data.w).y <= 0.0)
                                history = emptyDirectLightReservoir();
                        }
                        if (history.data.x != 0xffffffffu) {
#if WAVE_WORK_COUNTERS
                            profileWork(11u, 1u);
#endif
                            AreaLightCandidate history_candidate;
                            history_candidate.light_index = history.data.x;
                            history_candidate.barycentrics =
                                unpackHalf2x16(history.data.y);
                            history_candidate.target = 0.0;
                            vec3 history_direction;
                            float history_distance;
                            vec3 history_contribution =
                                WAVE_UNIFIED_PRIMARY_RESTIR != 0u
                                ? evaluateUnifiedPrimaryCandidate(
                                    history_candidate, position, normal,
                                    incoming, material, candidate_count,
                                    history_direction, history_distance)
                                : evaluateAreaLightCandidate(
                                    history_candidate, position, normal,
                                    incoming, material, candidate_count,
                                    history_direction, history_distance);
                            float history_target = max(dot(
                                history_contribution,
                                vec3(0.2126, 0.7152, 0.0722)), 0.0);
                            bool pairwise = push.restir_pairwise_mis != 0u
                                && reuse_index > 0u
                                && !material_textured;
                            // The reservoir carries the selected proposal's
                            // target density at its source surface. Reusing
                            // that value is exact for this pair, works for
                            // textured/custom materials, and avoids storing
                            // another full evaluated-material G-buffer.
                            float source_target =
                                unpackHalf2x16(history.data.w).x;
                            bool generalized =
                                WAVE_GENERALIZED_RESTIR != 0u
                                && reuse_index > 0u
                                && !material_textured;
                            float generalized_density_normalization = 1.0;
                            if (generalized) {
                                float target_sum = history_target;
                                float active_proposals = history_target > 0.0
                                    ? 1.0 : 0.0;
                                for (uint proposal_index = 0u;
                                        proposal_index <= spatial_count;
                                        ++proposal_index) {
                                    ivec2 proposal_pixel = previous_pixel;
                                    if (proposal_index > 0u) {
                                        proposal_pixel += restirSpatialOffset(
                                            proposal_index - 1u
                                                + spatial_rotation,
                                            int(push.restir_spatial_radius));
                                    }
                                    bool proposal_inside = all(
                                        greaterThanEqual(proposal_pixel,
                                            ivec2(0))) && all(lessThan(
                                        proposal_pixel,
                                        ivec2(push.image_tile.xy)));
                                    if (!proposal_inside)
                                        continue;
                                    uint proposal_flat_index =
                                        uint(proposal_pixel.y)
                                            * push.image_tile.x
                                            + uint(proposal_pixel.x);
                                    DirectLightReservoir proposal =
                                        loadPreviousDirectLightReservoir(
                                            proposal_flat_index);
                                    if (proposal.data.x == 0xffffffffu)
                                        continue;
                                    vec4 proposal_position;
                                    vec4 proposal_normal;
                                    if (!restirHistorySurfaceCompatible(
                                            proposal_pixel, position,
                                            shading_normal, distance,
                                            surface_class, material_signature,
                                            proposal_index == 0u,
                                            proposal_position,
                                            proposal_normal))
                                        continue;
                                    float proposal_target = source_target;
                                    if (proposal_index != reuse_index) {
                                        vec3 proposal_direction;
                                        float proposal_distance;
                                        vec3 proposal_incoming = normalize(
                                            proposal_position.xyz
                                                - previous_camera.origin.xyz);
                                        vec3 proposal_contribution =
                                            WAVE_UNIFIED_PRIMARY_RESTIR != 0u
                                            ? evaluateUnifiedPrimaryCandidate(
                                                history_candidate,
                                                proposal_position.xyz,
                                                normalize(
                                                    proposal_normal.xyz),
                                                proposal_incoming, material,
                                                candidate_count,
                                                proposal_direction,
                                                proposal_distance)
                                            : evaluateAreaLightCandidate(
                                                history_candidate,
                                                proposal_position.xyz,
                                                normalize(
                                                    proposal_normal.xyz),
                                                proposal_incoming, material,
                                                candidate_count,
                                                proposal_direction,
                                                proposal_distance);
                                        proposal_target = max(dot(
                                            proposal_contribution,
                                            vec3(0.2126, 0.7152, 0.0722)),
                                            0.0);
                                    }
                                    target_sum += proposal_target;
                                    if (proposal_target > 0.0)
                                        active_proposals += 1.0;
                                }
                                // Repeated spatiotemporal reuse can otherwise
                                // multiply a high-variance generalized factor
                                // by as many as the full proposal count each
                                // generation. Bound it relative to canonical
                                // reuse, just as pairwise balance is bounded by
                                // two, while retaining multi-proposal downweighting.
                                float canonical_normalization =
                                    source_target > 0.0
                                    ? 1.0 / source_target : 0.0;
                                generalized_density_normalization =
                                    target_sum > 0.0
                                    ? min(active_proposals / target_sum,
                                        push.restir_generalized_balance_cap
                                            * canonical_normalization)
                                    : 0.0;
                            }
                            bool history_selected = reuse_index == 0u
                                ? mergeDirectLightReservoir(
                                    reservoir, history, history_target,
                                    randomFloat(rng))
                                : generalized
                                ? mergeBalancedDirectLightReservoir(
                                    reservoir, history, history_target,
                                    generalized_density_normalization,
                                    randomFloat(rng))
                                : pairwise
                                ? mergePairwiseDirectLightReservoir(
                                    reservoir, history, history_target,
                                    source_target, randomFloat(rng))
                                : mergeCanonicalDirectLightReservoir(
                                    reservoir, history, history_target,
                                    randomFloat(rng));
                            if (history_selected) {
#if WAVE_WORK_COUNTERS
                                profileWork(12u, 1u);
#endif
                            }
                        } else {
#if WAVE_WORK_COUNTERS
                            profileWork(13u, 1u);
                            profileWork(history_source_present ? 14u : 15u,
                                1u);
#endif
                            // An empty temporal center normally means the
                            // previous pixel was environment or otherwise had
                            // no direct-light proposal. Avoid four speculative
                            // neighbor reads at these disocclusions. A present
                            // but geometry-incompatible center still permits
                            // spatial recovery from compatible neighbors.
                            if (reuse_index == 0u
                                    && !history_source_present)
                                break;
                        }
                    }
                }
            }
            storeCurrentDirectLightReservoir(pixel_index, reservoir);
            if (reservoir.data.x != 0xffffffffu) {
                AreaLightCandidate selected;
                selected.light_index = reservoir.data.x;
                selected.barycentrics = unpackHalf2x16(reservoir.data.y);
                selected.target = unpackHalf2x16(reservoir.data.w).x;
                vec3 selected_direction;
                float selected_distance;
                vec3 selected_contribution =
                    WAVE_UNIFIED_PRIMARY_RESTIR != 0u
                    ? evaluateUnifiedPrimaryCandidate(
                        selected, position, normal, incoming, material,
                        candidate_count, selected_direction,
                        selected_distance)
                    : evaluateAreaLightCandidate(
                        selected, position, normal, incoming, material,
                        candidate_count, selected_direction,
                        selected_distance);
                float visibility = areaLightCandidateVisibility(
                    position, normal, selected_direction,
                    selected_distance);
                path.radiance.rgb += selected_contribution * visibility
                    * directLightReservoirNormalization(reservoir);
            }
        } else {
            vec3 area_direct = vec3(0.0);
            for (uint sample_index = 0u; sample_index < light_samples;
                    ++sample_index)
                area_direct += sampleAreaLight(
                    position, normal, incoming, material, rng,
                    sample_index, light_samples);
            path.radiance.rgb += area_direct / float(light_samples);
        }
        if (push.restir_di != 0u
                && WAVE_STRATIFIED_PRIMARY_RESTIR != 0u
                && push.environment_samples > 0u) {
            uint reservoir_pixel_count = push.image_tile.x * push.image_tile.y;
            DirectLightReservoir environment_reservoir =
                emptyDirectLightReservoir();
            uint environment_candidates = clamp(
                push.restir_candidate_count, 1u, 4u);
            for (uint sample_index = 0u;
                    sample_index < environment_candidates; ++sample_index) {
                AreaLightCandidate candidate;
                vec3 candidate_direction = cosineHemisphere(
                    normal, randomFloat(rng), randomFloat(rng));
                candidate.light_index = ENVIRONMENT_LIGHT_CANDIDATE_INDEX;
                candidate.barycentrics = encodeEnvironmentCandidateDirection(
                    candidate_direction);
                float candidate_distance;
                vec3 contribution = evaluateEnvironmentCandidate(
                    candidate.barycentrics, position, normal, incoming,
                    material, environment_candidates, 1.0,
                    candidate_direction, candidate_distance);
                candidate.target = max(dot(contribution,
                    vec3(0.2126, 0.7152, 0.0722)), 0.0);
                updateDirectLightReservoir(
                    environment_reservoir, candidate.light_index,
                    candidate.barycentrics, candidate.target,
                    candidate.target, 1.0, randomFloat(rng));
            }
            if (push.restir_history_valid != 0u) {
                ivec2 history_pixel;
                if (reprojectRestir(position, history_pixel)
                        && all(greaterThanEqual(history_pixel, ivec2(0)))
                        && all(lessThan(history_pixel,
                            ivec2(push.image_tile.xy)))) {
                    if (restirHistorySurfaceMatches(
                            history_pixel, position, shading_normal, distance,
                            surface_class, material_signature, true)) {
                        uint history_index = uint(history_pixel.y)
                            * push.image_tile.x + uint(history_pixel.x);
                        DirectLightReservoir history =
                            loadPreviousEnvironmentReservoir(
                                history_index, reservoir_pixel_count);
                        history = limitDirectLightReservoir(
                            history, max(float(push.restir_history_limit)
                                - unpackHalf2x16(
                                    environment_reservoir.data.w).y, 0.0));
                        if (history.data.x != 0xffffffffu) {
                            AreaLightCandidate history_candidate;
                            history_candidate.light_index = history.data.x;
                            history_candidate.barycentrics =
                                unpackHalf2x16(history.data.y);
                            vec3 history_direction;
                            float history_distance;
                            vec3 history_contribution =
                                evaluateEnvironmentCandidate(
                                    history_candidate.barycentrics,
                                    position, normal, incoming, material,
                                    environment_candidates, 1.0,
                                    history_direction, history_distance);
                            float history_target = max(dot(
                                history_contribution,
                                vec3(0.2126, 0.7152, 0.0722)), 0.0);
                            mergeDirectLightReservoir(
                                environment_reservoir, history,
                                history_target, randomFloat(rng));
                        }
                    }
                }
            }
            storeCurrentEnvironmentReservoir(
                pixel_index, reservoir_pixel_count, environment_reservoir);
            if (environment_reservoir.data.x != 0xffffffffu) {
                vec3 selected_direction;
                float selected_distance;
                vec3 selected_contribution = evaluateEnvironmentCandidate(
                    unpackHalf2x16(environment_reservoir.data.y),
                    position, normal, incoming, material,
                    environment_candidates, 1.0,
                    selected_direction, selected_distance);
                float visibility = areaLightCandidateVisibility(
                    position, normal, selected_direction,
                    selected_distance);
                path.radiance.rgb += selected_contribution * visibility
                    * directLightReservoirNormalization(
                        environment_reservoir);
            }
        }
        uint environment_samples = push.restir_di != 0u
                && (WAVE_UNIFIED_PRIMARY_RESTIR != 0u
                    || WAVE_STRATIFIED_PRIMARY_RESTIR != 0u)
            ? 0u : min(push.environment_samples, 4u);
        if (environment_samples > 0u) {
            vec3 environment_direct = vec3(0.0);
            for (uint sample_index = 0u;
                    sample_index < environment_samples; ++sample_index)
                environment_direct += sampleEnvironment(
                    position, normal, incoming, material, rng,
                    environment_samples);
            path.radiance.rgb += environment_direct /
                float(environment_samples);
        }
        vec3 bsdf_weight;
        samplePbr(material, normal, incoming, rng,
            next_direction, bsdf_weight, bsdf_pdf);
#if WAVE_ORDINARYSHADE_PRIMARY_CONTINUATION
        path.throughput.rgb = ordinarylight_primary_apply_bsdf_weight(
            path.throughput.rgb, bsdf_weight);
        cone_spread = ordinarylight_primary_scattered_cone_spread(
            cone_spread, material.base_roughness.a);
#else
        path.throughput.rgb *= bsdf_weight;
        cone_spread += material.base_roughness.a * 0.25;
#endif
    }
#if WAVE_ORDINARYSHADE_PRIMARY_CONTINUATION
    next_direction = ordinarylight_primary_continuation_direction(
        next_direction);
#else
    next_direction = normalize(next_direction);
#endif
    setPathBounce(path, 1u);
#if WAVE_ORDINARYSHADE_PRIMARY_CONTINUATION
    path.metadata.w = ordinarylight_primary_continuation_flags(
        path.metadata.w, medium_depth, transmission,
        push.restir_di != 0u && WAVE_UNIFIED_PRIMARY_RESTIR != 0u);
    setPathPreviousPdf(path, ordinarylight_primary_previous_pdf(
        pathPreviousPdf(path), bsdf_pdf, transmission));
#else
    path.metadata.w = PATH_ACTIVE_BIT | (medium_depth << 8u)
        | (path.metadata.w & PATH_INDIRECT_CAPTURE_BIT);
    if (transmission <= 0.001) {
        path.metadata.w |= PATH_PREVIOUS_DIFFUSE_BIT;
        if (push.restir_di != 0u && WAVE_UNIFIED_PRIMARY_RESTIR != 0u)
            path.metadata.w |= PATH_PREVIOUS_UNIFIED_NEE_BIT;
        setPathPreviousPdf(path, bsdf_pdf);
    }
#endif
    setPathRng(path, rng);
#if WAVE_ORDINARYSHADE_PRIMARY_CONTINUATION
    if (ordinarylight_primary_capture_secondary(
            path.metadata.w, transmission)) {
#else
    if ((path.metadata.w & PATH_INDIRECT_CAPTURE_BIT) != 0u
            && transmission <= 0.001) {
#endif
#if WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION
        ordinarylight_store_secondary_primary(
            path_index, path.throughput.rgb, path.radiance.rgb, bsdf_pdf);
#else
        secondary_paths[path_index].primary_throughput =
            vec4(path.throughput.rgb, 1.0);
        secondary_paths[path_index].primary_radiance =
            vec4(path.radiance.rgb, 1.0);
        secondary_paths[path_index].normal_pdf.w = bsdf_pdf;
#endif
    }
    WAVE_STORE_PATH(path_index, path);

#if WAVE_ORDINARYSHADE_PRIMARY_CONTINUATION
    vec3 continuation_origin = ordinarylight_primary_continuation_origin(
        position, next_direction);
#else
    vec3 continuation_origin = position + next_direction * 0.002;
#endif
    vec3 continuation_direction = next_direction;
#if WAVE_MEGAKERNEL || WAVE_HYBRID
    traceRemaining(
        path, continuation_origin, continuation_direction,
        path_index, medium_depth, rng, cone_width, cone_spread);
    WAVE_STORE_PATH(path_index, path);
#if WAVE_MEGAKERNEL
    return;
#endif
#endif
#if !WAVE_MEGAKERNEL
    if ((path.metadata.w & PATH_ACTIVE_BIT) == 0u)
        return;
    uint output_index = reserveOutputIndex();
#if WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION
#if WAVE_ORDINARYSHADE_PRIMARY_CONTINUATION
    vec4 queued_origin = ordinarylight_primary_ray_origin_payload(
        continuation_origin);
    vec4 queued_direction = ordinarylight_primary_ray_direction_payload(
        continuation_direction);
#else
    vec4 queued_origin = vec4(continuation_origin, 0.001);
    vec4 queued_direction = vec4(continuation_direction, 1.0e30);
#endif
    if (!ordinarylight_enqueue_continuation(
            output_index, queued_origin, queued_direction, path_index,
            cone_width, cone_spread)) {
        WAVE_DEACTIVATE_STORED_PATH(path_index);
        return;
    }
#else
    if (output_index >= output_queue.capacity) {
        atomicAdd(output_queue.overflow, 1u);
        WAVE_DEACTIVATE_STORED_PATH(path_index);
        return;
    }
#if WAVE_ORDINARYSHADE_PRIMARY_CONTINUATION
    output_queue.rays[output_index].origin_tmin =
        ordinarylight_primary_ray_origin_payload(continuation_origin);
    output_queue.rays[output_index].direction_tmax =
        ordinarylight_primary_ray_direction_payload(continuation_direction);
#else
    output_queue.rays[output_index].origin_tmin = vec4(
        continuation_origin, 0.001);
    output_queue.rays[output_index].direction_tmax = vec4(
        continuation_direction, 1.0e30);
#endif
    output_queue.rays[output_index].path_index = path_index;
    output_queue.rays[output_index].padding_a = floatBitsToUint(cone_width);
    output_queue.rays[output_index].padding_b = floatBitsToUint(cone_spread);
    output_queue.rays[output_index].padding_c = 0u;
#endif
#endif
}
#endif

#if !WAVE_CONTINUATION
#if WAVE_PERSISTENT_COARSE && !WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION
shared uint persistent_tile_index;
#endif

void main()
{
#if WAVE_RAYGEN
    uvec2 pixel = gl_LaunchIDEXT.xy;
    processPrimaryPixel(pixel);
#elif WAVE_PERSISTENT_COARSE
#if WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION
    ordinarylight_persistent_coarse_schedule(push.tile_frame.xy);
#else
    uvec2 tile_count = (push.tile_frame.xy + uvec2(7u)) / 8u;
    uint total_tiles = tile_count.x * tile_count.y;
    for (;;) {
        if (gl_LocalInvocationIndex == 0u)
            persistent_tile_index = atomicAdd(output_queue.count, 1u);
        barrier();
        uint tile_index = persistent_tile_index;
        if (tile_index >= total_tiles)
            return;
        uvec2 tile = uvec2(
            tile_index % tile_count.x, tile_index / tile_count.x);
        processPrimaryPixel(tile * 8u + gl_LocalInvocationID.xy);
        barrier();
    }
#endif
#else
#if WAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION
    uvec2 group_id = ordinarylight_primary_scheduled_group(
        gl_WorkGroupID.xy, gl_NumWorkGroups.xy,
        uint(WAVE_GROUP_SWIZZLE_WIDTH));
#else
    uvec2 group_id = gl_WorkGroupID.xy;
#if WAVE_GROUP_SWIZZLE_WIDTH > 1
    const uint swizzle_width = uint(WAVE_GROUP_SWIZZLE_WIDTH);
    uint full_tiles = gl_NumWorkGroups.x / swizzle_width;
    uint full_tile_groups = full_tiles * swizzle_width
        * gl_NumWorkGroups.y;
    uint linear_group = gl_WorkGroupID.y * gl_NumWorkGroups.x
        + gl_WorkGroupID.x;
    if (linear_group < full_tile_groups) {
        uint groups_per_tile = swizzle_width * gl_NumWorkGroups.y;
        uint tile = linear_group / groups_per_tile;
        uint local = linear_group - tile * groups_per_tile;
        group_id = uvec2(
            tile * swizzle_width + local % swizzle_width,
            local / swizzle_width);
    } else {
        uint tail_width = gl_NumWorkGroups.x
            - full_tiles * swizzle_width;
        uint local = linear_group - full_tile_groups;
        group_id = uvec2(
            full_tiles * swizzle_width + local % tail_width,
            local / tail_width);
    }
#endif
#endif
    processPrimaryPixel(
        group_id * gl_WorkGroupSize.xy + gl_LocalInvocationID.xy);
#endif
}
#endif
