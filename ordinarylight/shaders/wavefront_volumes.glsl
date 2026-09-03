#ifndef WAVE_VOLUME_HEADER_BINDING
#define WAVE_VOLUME_HEADER_BINDING 25
#define WAVE_VOLUME_SCALAR_BINDING 26
#define WAVE_VOLUME_TRANSFER_BINDING 27
#define WAVE_VOLUME_TRIANGLE_BINDING 28
#define WAVE_VOLUME_SAMPLER_BINDING 29
#endif

#ifndef WAVE_OVERLAPPING_VOLUMES
#define WAVE_OVERLAPPING_VOLUMES 0
#endif
#ifndef WAVE_VOLUME_SCATTERING
#define WAVE_VOLUME_SCATTERING 0
#endif
#ifndef WAVE_VOLUME_MULTIPLE_SCATTERING
#define WAVE_VOLUME_MULTIPLE_SCATTERING 0
#endif
#ifndef WAVE_VOLUME_EMPTY_SPACE_SKIPPING
#define WAVE_VOLUME_EMPTY_SPACE_SKIPPING 0
#endif

struct VolumeHeader {
    mat4 world_to_local;
    uvec4 dimensions_offset;
    vec4 value_parameters;
    vec4 render_parameters;
    vec4 scattering_parameters;
    vec4 phase_parameters;
    vec4 multiple_scattering_parameters;
    uvec4 acceleration_parameters;
    uvec4 clip_parameters;
    vec4 clip_planes[8];
};

layout(set = 0, binding = WAVE_VOLUME_HEADER_BINDING, std430)
readonly buffer VolumeHeaderBuffer { VolumeHeader volume_headers[]; };
layout(set = 0, binding = WAVE_VOLUME_SCALAR_BINDING, std430)
readonly buffer VolumeScalarBuffer { float volume_scalars[]; };
layout(set = 0, binding = WAVE_VOLUME_TRANSFER_BINDING, std430)
readonly buffer VolumeTransferBuffer { vec4 volume_transfer[]; };
layout(set = 0, binding = WAVE_VOLUME_TRIANGLE_BINDING, std430)
readonly buffer TriangleVolumeBuffer { uint triangle_volumes[]; };
layout(set = 0, binding = WAVE_VOLUME_SAMPLER_BINDING)
uniform sampler3D volume_textures[16];

bool isVolumePrimitive(uint primitive)
{
    return triangle_volumes[primitive] != 0xffffffffu;
}

vec2 volumeInterval(VolumeHeader header, vec3 origin, vec3 direction)
{
    vec3 local_origin = (header.world_to_local * vec4(origin, 1.0)).xyz;
    vec3 local_direction = (header.world_to_local * vec4(direction, 0.0)).xyz;
    vec3 safe_direction = vec3(
        abs(local_direction.x) > 1e-10 ? local_direction.x : 1e-10,
        abs(local_direction.y) > 1e-10 ? local_direction.y : 1e-10,
        abs(local_direction.z) > 1e-10 ? local_direction.z : 1e-10);
    vec3 first = (vec3(0.0) - local_origin) / safe_direction;
    vec3 second = (vec3(1.0) - local_origin) / safe_direction;
    vec3 lower = min(first, second);
    vec3 upper = max(first, second);
    return vec2(max(max(lower.x, lower.y), lower.z),
                min(min(upper.x, upper.y), upper.z));
}

vec3 volumeSliceDistances(VolumeHeader header, vec3 origin, vec3 direction)
{
    vec3 local_origin = (header.world_to_local * vec4(origin, 1.0)).xyz;
    vec3 local_direction = (header.world_to_local * vec4(direction, 0.0)).xyz;
    vec3 distances = vec3(1.0e30);
    for (uint axis = 0u; axis < 3u; ++axis) {
        if ((header.clip_parameters.z & (1u << axis)) != 0u
                && abs(local_direction[axis]) > 1.0e-10)
            distances[axis] = (header.clip_planes[7][axis]
                - local_origin[axis]) / local_direction[axis];
    }
    if (distances.x > distances.y) {
        float temporary = distances.x; distances.x = distances.y;
        distances.y = temporary;
    }
    if (distances.y > distances.z) {
        float temporary = distances.y; distances.y = distances.z;
        distances.z = temporary;
    }
    if (distances.x > distances.y) {
        float temporary = distances.x; distances.x = distances.y;
        distances.y = temporary;
    }
    return distances;
}

float sampleVolumeTexture(uint index, vec3 coordinate)
{
    switch (index) {
    case 0u: return texture(volume_textures[0], coordinate).r;
    case 1u: return texture(volume_textures[1], coordinate).r;
    case 2u: return texture(volume_textures[2], coordinate).r;
    case 3u: return texture(volume_textures[3], coordinate).r;
    case 4u: return texture(volume_textures[4], coordinate).r;
    case 5u: return texture(volume_textures[5], coordinate).r;
    case 6u: return texture(volume_textures[6], coordinate).r;
    case 7u: return texture(volume_textures[7], coordinate).r;
    case 8u: return texture(volume_textures[8], coordinate).r;
    case 9u: return texture(volume_textures[9], coordinate).r;
    case 10u: return texture(volume_textures[10], coordinate).r;
    case 11u: return texture(volume_textures[11], coordinate).r;
    case 12u: return texture(volume_textures[12], coordinate).r;
    case 13u: return texture(volume_textures[13], coordinate).r;
    case 14u: return texture(volume_textures[14], coordinate).r;
    default: return texture(volume_textures[15], coordinate).r;
    }
}

float volumeScalar(uint volume_index, VolumeHeader header, vec3 world_position)
{
    for (uint plane_index = 0u;
         plane_index < min(header.clip_parameters.x, 8u); ++plane_index) {
        vec4 plane = header.clip_planes[plane_index];
        if (dot(plane.xyz, world_position) < plane.w)
            return -1.0;
    }
    vec3 local = clamp(
        (header.world_to_local * vec4(world_position, 1.0)).xyz,
        vec3(0.0), vec3(1.0));
    float physical = sampleVolumeTexture(volume_index, local);
    if (header.clip_parameters.y != 0u && isnan(physical))
        return -2.0;
    float mapped = physical;
    uint mapping = uint(header.phase_parameters.z);
    if (mapping == 1u)
        mapped = physical > 0.0 ? log(physical) : -3.402823e38;
    else if (mapping == 2u)
        mapped = sign(physical) * log(
            1.0 + abs(physical) / header.phase_parameters.w);
    return (mapped - header.render_parameters.y)
        * header.render_parameters.w;
}

vec4 volumeTransferSample(VolumeHeader header, float value)
{
    uint offset = uint(header.value_parameters.x);
    if (value < -1.5)
        return volume_transfer[offset];
    if (value < 0.0)
        return vec4(0.0);
    uint count = uint(header.value_parameters.y);
    uint reserved = min(header.clip_parameters.y, 1u);
    offset += reserved;
    count -= reserved;
    float coordinate = clamp(value, 0.0, 1.0) * float(max(count, 1u) - 1u);
    uint lower = uint(floor(coordinate));
    uint upper = min(lower + 1u, count - 1u);
    return mix(volume_transfer[offset + lower], volume_transfer[offset + upper],
               fract(coordinate));
}

#if WAVE_VOLUME_EMPTY_SPACE_SKIPPING
uvec3 volumeBrickIndexFromVoxel(VolumeHeader header, vec3 voxel_position)
{
    uvec3 brick_grid = header.acceleration_parameters.yzw;
    return min(
        uvec3(floor(max(voxel_position, vec3(0.0)) / 8.0)),
        brick_grid - uvec3(1u));
}

bool volumeBrickOccupiedAtVoxel(
    VolumeHeader header, vec3 voxel_position)
{
    uvec3 brick_grid = header.acceleration_parameters.yzw;
    if (any(equal(brick_grid, uvec3(0u))))
        return true;
    uvec3 brick = volumeBrickIndexFromVoxel(header, voxel_position);
    uint linear_index = brick.x + brick_grid.x
        * (brick.y + brick_grid.y * brick.z);
    return volume_scalars[
        header.acceleration_parameters.x + linear_index] > 0.5;
}

void volumeVoxelRay(
    VolumeHeader header, vec3 origin, vec3 direction,
    out vec3 voxel_origin, out vec3 voxel_direction)
{
    vec3 local_direction = (
        header.world_to_local * vec4(direction, 0.0)).xyz;
    vec3 voxel_extent = vec3(header.dimensions_offset.xyz) - vec3(1.0);
    voxel_origin = (header.world_to_local * vec4(origin, 1.0)).xyz
        * voxel_extent;
    voxel_direction = local_direction * voxel_extent;
}

float volumeBrickExitDistanceAtVoxel(
    VolumeHeader header, vec3 voxel_position, vec3 voxel_direction,
    float distance)
{
    vec3 voxel_extent = vec3(header.dimensions_offset.xyz) - vec3(1.0);
    voxel_position = clamp(voxel_position, vec3(0.0), voxel_extent);
    uvec3 brick = volumeBrickIndexFromVoxel(header, voxel_position);
    vec3 lower = vec3(brick) * 8.0;
    vec3 upper = min(lower + vec3(8.0), voxel_extent);
    float next_delta = 1.0e30;
    for (uint axis = 0u; axis < 3u; ++axis) {
        if (abs(voxel_direction[axis]) <= 1.0e-10)
            continue;
        float boundary = voxel_direction[axis] > 0.0
            ? upper[axis] : lower[axis];
        float candidate = (
            boundary - voxel_position[axis]) / voxel_direction[axis];
        // A negative-going ray may be exactly on the lower face selected by
        // floor().  Treat that face as an immediate exit so the next march
        // sample reclassifies in the adjacent brick instead of jumping across
        // the remainder of the volume.
        if (candidate >= -1.0e-6)
            next_delta = min(next_delta, max(candidate, 0.0));
    }
    return distance + next_delta;
}

bool volumeBrickOccupied(VolumeHeader header, vec3 world_position)
{
    vec3 voxel_origin;
    vec3 voxel_direction;
    volumeVoxelRay(
        header, world_position, vec3(0.0), voxel_origin, voxel_direction);
    return volumeBrickOccupiedAtVoxel(header, voxel_origin);
}

float volumeBrickExitDistance(
    VolumeHeader header, vec3 origin, vec3 direction, float distance)
{
    vec3 voxel_origin;
    vec3 voxel_direction;
    volumeVoxelRay(
        header, origin, direction, voxel_origin, voxel_direction);
    return volumeBrickExitDistanceAtVoxel(
        header, voxel_origin + voxel_direction * distance,
        voxel_direction, distance);
}
#endif

#if WAVE_VOLUME_SCATTERING
vec3 environmentColor(vec3 direction);

float volumePhase(VolumeHeader header, float cosine)
{
    if (header.phase_parameters.y < 0.5)
        return 0.0795774715459;
    float g = clamp(header.phase_parameters.x, -0.99, 0.99);
    float denominator = max(
        1.0 + g * g - 2.0 * g * clamp(cosine, -1.0, 1.0), 1e-8);
    return (1.0 - g * g)
        / (12.5663706144 * denominator * sqrt(denominator));
}

float volumeOpaqueVisibility(
    vec3 world_position, vec3 direction, float maximum_distance)
{
    float shadow_distance = maximum_distance < 1.0e29
        ? max(maximum_distance - 0.004, 0.001) : 1.0e30;
    rayQueryEXT shadow;
    rayQueryInitializeEXT(
        shadow, scene_tlas,
        gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT,
        0x01, world_position + direction * 0.002, 0.001,
        direction, shadow_distance);
    while (rayQueryProceedEXT(shadow)) {}
    return rayQueryGetIntersectionTypeEXT(shadow, true)
        == gl_RayQueryCommittedIntersectionNoneEXT ? 1.0 : 0.0;
}

float approximateVolumeLightTransmittance(
    vec3 world_position, vec3 light_direction, float light_distance)
{
    uint volume_count = min(
        uint(volume_headers[0].render_parameters.z), 16u);
    float optical_depth = 0.0;
    for (uint volume_index = 0u; volume_index < volume_count; ++volume_index) {
        VolumeHeader medium = volume_headers[volume_index];
        vec2 interval = volumeInterval(
            medium, world_position, light_direction);
        float entry = max(interval.x, 0.0);
        float exit_distance = min(interval.y, light_distance);
        if (exit_distance <= entry)
            continue;
        float midpoint = 0.5 * (entry + exit_distance);
        vec4 sample_value = volumeTransferSample(
            medium, volumeScalar(
                volume_index, medium,
                world_position + light_direction * midpoint));
        float reference_alpha = clamp(
            sample_value.a * medium.value_parameters.z, 0.0, 0.999999);
        float extinction = -log(1.0 - reference_alpha)
            / max(medium.render_parameters.x, 1e-5);
        optical_depth += extinction * (exit_distance - entry);
    }
    return exp(-optical_depth);
}

vec3 volumePointScattering(
    VolumeHeader header, vec3 world_position, vec3 ray_direction
#if WAVE_VOLUME_MULTIPLE_SCATTERING
    , float optical_depth
#endif
    )
{
    float scattering_scale = header.scattering_parameters.w;
    if (scattering_scale <= 0.0)
        return vec3(0.0);
    vec3 scattered = vec3(0.0);
#if WAVE_VOLUME_MULTIPLE_SCATTERING
    vec3 isotropic_scattered = vec3(0.0);
#endif
    vec3 outgoing = -ray_direction;
    for (uint light_index = 0u;
            light_index < min(push.point_light_count, 64u); ++light_index) {
        PointLightData light = point_lights[light_index];
        int light_type = int(light.position_type.w + 0.5);
        if (light_type == 3) continue;
        vec3 incoming;
        float distance_to_light = 10000.0;
        float attenuation = 1.0;
        if (light_type == 1) {
            incoming = -normalize(light.direction_range.xyz);
        } else {
            vec3 offset = light.position_type.xyz - world_position;
            float distance_squared = max(dot(offset, offset), 1e-6);
            distance_to_light = sqrt(distance_squared);
            incoming = offset / distance_to_light;
            if (light.direction_range.w > 0.0 &&
                    distance_to_light > light.direction_range.w) continue;
            attenuation = 1.0 / distance_squared;
            if (light_type == 2) {
                float cone = dot(normalize(light.direction_range.xyz), -incoming);
                float spot = smoothstep(
                    light.spot_parameters.y, light.spot_parameters.x, cone);
                if (spot <= 0.0) continue;
                attenuation *= spot;
            }
        }
        vec3 incident = light.color_intensity.rgb * light.color_intensity.a
            * attenuation;
        incident *= approximateVolumeLightTransmittance(
            world_position, incoming, distance_to_light);
        incident *= volumeOpaqueVisibility(
            world_position, incoming, distance_to_light);
        scattered += incident * volumePhase(
            header, dot(-incoming, outgoing));
#if WAVE_VOLUME_MULTIPLE_SCATTERING
        isotropic_scattered += incident * 0.0795774715459;
#endif
    }

    // A centroid quadrature sample per emissive triangle is deterministic and
    // avoids introducing a second stochastic sequence into the volume march.
    for (uint light_index = 0u;
            light_index < min(push.area_light_count, 64u); ++light_index) {
        AreaLightData light = area_lights[light_index];
        vec3 light_position = (light.a.xyz + light.b.xyz + light.c.xyz) / 3.0;
        vec3 offset = light_position - world_position;
        float distance_squared = max(dot(offset, offset), 1e-6);
        float distance_to_light = sqrt(distance_squared);
        vec3 incoming = offset / distance_to_light;
        vec3 light_normal = normalize(cross(
            light.b.xyz - light.a.xyz, light.c.xyz - light.a.xyz));
        float raw_cosine = dot(light_normal, -incoming);
        float light_cosine = light.distribution.z > 0.5
            ? abs(raw_cosine) : max(raw_cosine, 0.0);
        if (light_cosine <= 1e-6)
            continue;
        float visibility = volumeOpaqueVisibility(
            world_position, incoming, distance_to_light);
        float transmittance = approximateVolumeLightTransmittance(
            world_position, incoming, distance_to_light);
        vec3 incident = light.emission_area.rgb * light.emission_area.a
            * light_cosine / distance_squared;
        incident *= visibility * transmittance;
        scattered += incident
            * volumePhase(header, dot(-incoming, outgoing));
#if WAVE_VOLUME_MULTIPLE_SCATTERING
        isotropic_scattered += incident * 0.0795774715459;
#endif
    }

    const vec3 environment_directions[4] = vec3[4](
        vec3(0.577350269, 0.577350269, 0.577350269),
        vec3(-0.577350269, -0.577350269, 0.577350269),
        vec3(-0.577350269, 0.577350269, -0.577350269),
        vec3(0.577350269, -0.577350269, -0.577350269));
    uint environment_count = min(push.environment_samples, 4u);
    for (uint sample_index = 0u;
            sample_index < environment_count; ++sample_index) {
        vec3 incoming = environment_directions[sample_index];
        float visibility = volumeOpaqueVisibility(
            world_position, incoming, 1.0e30);
        float transmittance = approximateVolumeLightTransmittance(
            world_position, incoming, 1.0e30);
        vec3 incident = environmentColor(incoming) * visibility * transmittance
            * (12.5663706144 / float(environment_count));
        scattered += incident
            * volumePhase(header, dot(-incoming, outgoing));
#if WAVE_VOLUME_MULTIPLE_SCATTERING
        isotropic_scattered += incident * 0.0795774715459;
#endif
    }
#if WAVE_VOLUME_MULTIPLE_SCATTERING
    uint scattering_orders = clamp(
        uint(header.multiple_scattering_parameters.w + 0.5), 1u, 8u);
    vec3 ratio = clamp(
        header.multiple_scattering_parameters.rgb
            * (1.0 - exp(-max(optical_depth, 0.0))),
        vec3(0.0), vec3(0.999));
    vec3 order_weight = ratio;
    for (uint order = 2u; order <= scattering_orders; ++order) {
        scattered += isotropic_scattered * order_weight;
        order_weight *= ratio;
    }
#endif
    return scattered * header.scattering_parameters.rgb * scattering_scale;
}
#endif

float integrateVolumeUntil(
    uint volume_index, vec3 origin, vec3 direction,
    float maximum_distance,
    inout vec3 radiance, inout vec3 throughput)
{
    VolumeHeader header = volume_headers[volume_index];
    vec2 interval = volumeInterval(header, origin, direction);
    float entry = max(interval.x, 0.0);
    float exit_distance = min(interval.y, maximum_distance);
    if (exit_distance <= entry)
        return entry;
    if (header.clip_parameters.z != 0u) {
        vec3 distances = volumeSliceDistances(header, origin, direction);
        if ((header.clip_parameters.w & 1u) != 0u) {
            // Combined mode inserts these exact samples into the march below.
        } else {
        for (uint slice_index = 0u; slice_index < 3u; ++slice_index) {
            float slice_distance = distances[slice_index];
            if (slice_distance < entry || slice_distance > exit_distance)
                continue;
            vec4 sample_value = volumeTransferSample(
                header, volumeScalar(
                    volume_index, header,
                    origin + direction * slice_distance));
            float alpha = clamp(
                sample_value.a * header.value_parameters.z, 0.0, 1.0);
            radiance += throughput * alpha * sample_value.rgb
                * header.value_parameters.w;
            throughput *= 1.0 - alpha;
        }
        return exit_distance;
        }
    }
    float reference_step = max(header.render_parameters.x, 1e-5);
    uint steps = min(uint(ceil((exit_distance - entry) / reference_step)), 4096u);
    float step_size = (exit_distance - entry) / float(max(steps, 1u));
    float transmittance = 1.0;
    vec3 integrated = vec3(0.0);
    bool isosurface_enabled = (header.clip_parameters.w & 2u) != 0u;
    bool volume_enabled = header.clip_parameters.w != 2u;
    float isovalue = header.clip_planes[7].w;
    float previous_distance = entry;
    float previous_scalar = volumeScalar(
        volume_index, header, origin + direction * previous_distance);
#if WAVE_VOLUME_SCATTERING
#if WAVE_VOLUME_MULTIPLE_SCATTERING
    float scattering_midpoint = 0.5 * (entry + exit_distance);
    vec4 scattering_sample = volumeTransferSample(
        header, volumeScalar(
            volume_index, header,
            origin + direction * scattering_midpoint));
    float scattering_alpha = clamp(
        scattering_sample.a * header.value_parameters.z, 0.0, 0.999999);
    float scattering_extinction = -log(1.0 - scattering_alpha)
        / reference_step;
    vec3 scattering_source = volumePointScattering(
        header, origin + direction * scattering_midpoint, direction,
        scattering_extinction * (exit_distance - entry));
#else
    vec3 scattering_source = volumePointScattering(
        header, origin + direction * (0.5 * (entry + exit_distance)),
        direction);
#endif
#endif
    uint step_index = 0u;
#if WAVE_VOLUME_EMPTY_SPACE_SKIPPING
    vec3 voxel_origin;
    vec3 voxel_direction;
    volumeVoxelRay(header, origin, direction, voxel_origin, voxel_direction);
    float occupied_until = -1.0;
#endif
    while (step_index < steps) {
        float distance = entry + (float(step_index) + 0.5) * step_size;
#if WAVE_VOLUME_EMPTY_SPACE_SKIPPING
        vec3 world_position = origin + direction * distance;
        vec3 voxel_position = voxel_origin + voxel_direction * distance;
        if (!isosurface_enabled && distance + 1.0e-7 >= occupied_until) {
            float brick_exit = volumeBrickExitDistanceAtVoxel(
                header, voxel_position, voxel_direction, distance);
            if (!volumeBrickOccupiedAtVoxel(header, voxel_position)) {
                uint jump = uint(clamp(
                    ceil((brick_exit - distance) / step_size), 1.0,
                    float(steps - step_index)));
                step_index = min(step_index + jump, steps);
                continue;
            }
            occupied_until = brick_exit;
        }
#endif
        float scalar = volumeScalar(
            volume_index, header, origin + direction * distance);
        vec4 sample_value = volumeTransferSample(header, scalar);
        if (volume_enabled) {
        float reference_alpha = clamp(
            sample_value.a * header.value_parameters.z, 0.0, 0.999999);
        float alpha = 1.0 - pow(
            1.0 - reference_alpha, step_size / reference_step);
        vec3 source = sample_value.rgb * header.value_parameters.w;
#if WAVE_VOLUME_SCATTERING
        source += scattering_source;
#endif
        integrated += transmittance * alpha * source;
        transmittance *= 1.0 - alpha;
        }
        if (isosurface_enabled && previous_scalar >= 0.0 && scalar >= 0.0
                && ((previous_scalar < isovalue && scalar >= isovalue)
                    || (previous_scalar > isovalue && scalar <= isovalue))) {
            float lower_distance = previous_distance;
            float upper_distance = distance;
            float lower_scalar = previous_scalar;
            for (uint refinement = 0u; refinement < 8u; ++refinement) {
                float middle_distance = 0.5 * (lower_distance + upper_distance);
                float middle_scalar = volumeScalar(
                    volume_index, header,
                    origin + direction * middle_distance);
                if (middle_scalar < 0.0)
                    break;
                if ((lower_scalar < isovalue && middle_scalar < isovalue)
                        || (lower_scalar > isovalue && middle_scalar > isovalue)) {
                    lower_distance = middle_distance;
                    lower_scalar = middle_scalar;
                } else {
                    upper_distance = middle_distance;
                }
            }
            vec4 surface_sample = volumeTransferSample(header, isovalue);
            float surface_alpha = clamp(
                surface_sample.a * header.value_parameters.z, 0.0, 1.0);
            integrated += transmittance * surface_alpha * surface_sample.rgb
                * header.value_parameters.w;
            transmittance *= 1.0 - surface_alpha;
            if (!volume_enabled || transmittance <= 0.001)
                break;
        }
        previous_distance = distance;
        previous_scalar = scalar;
        if ((header.clip_parameters.w & 1u) != 0u) {
            vec3 slice_distances = volumeSliceDistances(
                header, origin, direction);
            float half_step = 0.5 * step_size + 1.0e-7;
            for (uint slice_index = 0u; slice_index < 3u; ++slice_index) {
                float slice_distance = slice_distances[slice_index];
                if (abs(slice_distance - distance) > half_step)
                    continue;
                vec4 slice_sample = volumeTransferSample(
                    header, volumeScalar(
                        volume_index, header,
                        origin + direction * slice_distance));
                float slice_alpha = clamp(
                    slice_sample.a * header.value_parameters.z, 0.0, 1.0);
                integrated += transmittance * slice_alpha * slice_sample.rgb
                    * header.value_parameters.w;
                transmittance *= 1.0 - slice_alpha;
            }
        }
        if (transmittance < 1e-4)
            break;
        step_index += 1u;
    }
    radiance += throughput * integrated;
    throughput *= transmittance;
    return exit_distance;
}

float integrateVolume(
    uint volume_index, vec3 origin, vec3 direction,
    inout vec3 radiance, inout vec3 throughput)
{
    return integrateVolumeUntil(
        volume_index, origin, direction, 1.0e30, radiance, throughput);
}

#if WAVE_OVERLAPPING_VOLUMES
void integrateOverlappingVolumesBeforeSurface(
    vec3 origin, vec3 direction, float surface_distance,
    inout vec3 radiance, inout vec3 throughput)
{
    const uint maximum_volumes = 16u;
    uint volume_count = min(
        uint(volume_headers[0].render_parameters.z), maximum_volumes);
    float entries[16];
    float exits[16];
    for (uint volume_index = 0u; volume_index < maximum_volumes; ++volume_index) {
        entries[volume_index] = 1.0;
        exits[volume_index] = 0.0;
        if (volume_index >= volume_count)
            continue;
        vec2 interval = volumeInterval(
            volume_headers[volume_index], origin, direction);
        float entry = max(interval.x, 0.0);
        float exit_distance = min(interval.y, surface_distance);
        if (exit_distance > entry) {
            entries[volume_index] = entry;
            exits[volume_index] = exit_distance;
        }
    }

    float union_entry = surface_distance;
    float union_exit = 0.0;
    float reference_step = 1.0e30;
#if WAVE_VOLUME_SCATTERING
    vec3 scattering_sources[16];
#endif
    for (uint volume_index = 0u; volume_index < volume_count; ++volume_index) {
#if WAVE_VOLUME_SCATTERING
        scattering_sources[volume_index] = vec3(0.0);
#endif
        if (exits[volume_index] <= entries[volume_index])
            continue;
        union_entry = min(union_entry, entries[volume_index]);
        union_exit = max(union_exit, exits[volume_index]);
        reference_step = min(
            reference_step,
            max(volume_headers[volume_index].render_parameters.x, 1e-5));
#if WAVE_VOLUME_SCATTERING
#if WAVE_VOLUME_MULTIPLE_SCATTERING
        VolumeHeader scattering_header = volume_headers[volume_index];
        float scattering_midpoint = 0.5 * (
            entries[volume_index] + exits[volume_index]);
        vec3 scattering_position = origin + direction * scattering_midpoint;
        vec4 scattering_sample = volumeTransferSample(
            scattering_header, volumeScalar(
                volume_index, scattering_header, scattering_position));
        float scattering_alpha = clamp(
            scattering_sample.a * scattering_header.value_parameters.z,
            0.0, 0.999999);
        float scattering_extinction = -log(1.0 - scattering_alpha)
            / max(scattering_header.render_parameters.x, 1e-5);
        scattering_sources[volume_index] = volumePointScattering(
            scattering_header, scattering_position, direction,
            scattering_extinction
                * (exits[volume_index] - entries[volume_index]));
#else
        scattering_sources[volume_index] = volumePointScattering(
            volume_headers[volume_index],
            origin + direction * (0.5 * (
                entries[volume_index] + exits[volume_index])),
            direction);
#endif
#endif
    }
    if (union_exit <= union_entry)
        return;

    uint steps = min(
        uint(ceil((union_exit - union_entry) / reference_step)), 4096u);
    float step_size = (union_exit - union_entry) / float(max(steps, 1u));
    uint step_index = 0u;
    while (step_index < steps) {
        float distance = union_entry + (float(step_index) + 0.5) * step_size;
        vec3 world_position = origin + direction * distance;
#if WAVE_VOLUME_EMPTY_SPACE_SKIPPING
        bool any_occupied = false;
        float empty_exit = 1.0e30;
        for (uint volume_index = 0u;
                volume_index < volume_count; ++volume_index) {
            if (distance < entries[volume_index]
                    || distance >= exits[volume_index])
                continue;
            VolumeHeader header = volume_headers[volume_index];
            if (volumeBrickOccupied(header, world_position)) {
                any_occupied = true;
                break;
            }
            empty_exit = min(
                empty_exit,
                volumeBrickExitDistance(
                    header, origin, direction, distance));
        }
        if (!any_occupied) {
            uint jump = uint(clamp(
                ceil((empty_exit - distance) / step_size), 1.0,
                float(steps - step_index)));
            step_index = min(step_index + jump, steps);
            continue;
        }
#endif
        float extinction = 0.0;
        vec3 emission_extinction = vec3(0.0);
        for (uint volume_index = 0u; volume_index < volume_count; ++volume_index) {
            if (distance < entries[volume_index]
                    || distance >= exits[volume_index])
                continue;
            VolumeHeader header = volume_headers[volume_index];
            vec4 sample_value = volumeTransferSample(
                header, volumeScalar(volume_index, header, world_position));
            float reference_alpha = clamp(
                sample_value.a * header.value_parameters.z,
                0.0, 0.999999);
            float medium_extinction = -log(1.0 - reference_alpha)
                / max(header.render_parameters.x, 1e-5);
            extinction += medium_extinction;
            emission_extinction += medium_extinction
                * sample_value.rgb * header.value_parameters.w;
#if WAVE_VOLUME_SCATTERING
            emission_extinction += medium_extinction
                * scattering_sources[volume_index];
#endif
        }
        if (extinction > 1e-8) {
            float alpha = 1.0 - exp(-extinction * step_size);
            radiance += throughput * alpha * emission_extinction / extinction;
            throughput *= 1.0 - alpha;
        }
        if (max(throughput.r, max(throughput.g, throughput.b)) < 1e-4)
            return;
        step_index += 1u;
    }
}
#endif

void integrateVolumesBeforeSurface(
    vec3 origin, vec3 direction, float surface_distance,
    inout vec3 radiance, inout vec3 throughput)
{
    // Empty scenes bind one zero-initialized header to keep descriptors valid.
    if (volume_headers[0].dimensions_offset.x == 0u)
        return;
#if WAVE_OVERLAPPING_VOLUMES
    if (uint(volume_headers[0].render_parameters.z) > 1u) {
        integrateOverlappingVolumesBeforeSurface(
            origin, direction, surface_distance, radiance, throughput);
        return;
    }
#endif
    vec3 traversal_origin = origin;
    float traveled = 0.0;
    for (uint traversal = 0u; traversal < 32u; ++traversal) {
        float remaining = surface_distance - traveled;
        if (remaining <= 0.001)
            break;
        rayQueryEXT volume_query;
        rayQueryInitializeEXT(
            volume_query, scene_tlas, gl_RayFlagsOpaqueEXT, 0x02,
            traversal_origin, 0.001, direction, remaining);
        while (rayQueryProceedEXT(volume_query)) {}
        if (rayQueryGetIntersectionTypeEXT(volume_query, true)
                != gl_RayQueryCommittedIntersectionTriangleEXT)
            break;
        uint primitive = rayQueryGetIntersectionPrimitiveIndexEXT(
            volume_query, true)
            + rayQueryGetIntersectionInstanceCustomIndexEXT(volume_query, true);
        uint volume_index = triangle_volumes[primitive];
        vec2 interval = volumeInterval(
            volume_headers[volume_index], traversal_origin, direction);
        float segment_end = min(interval.y, remaining);
        integrateVolumeUntil(
            volume_index, traversal_origin, direction, segment_end,
            radiance, throughput);
        if (segment_end >= remaining - 0.001
                || max(throughput.r, max(throughput.g, throughput.b)) < 1e-4)
            break;
        float advance = interval.y + 0.002;
        traversal_origin += direction * advance;
        traveled += advance;
    }
}

float volumeShadowTransmittance(
    vec3 origin, vec3 direction, float maximum_distance)
{
    vec3 ignored_radiance = vec3(0.0);
    vec3 transmittance = vec3(1.0);
    integrateVolumesBeforeSurface(
        origin, direction, maximum_distance,
        ignored_radiance, transmittance);
    return clamp(transmittance.r, 0.0, 1.0);
}
