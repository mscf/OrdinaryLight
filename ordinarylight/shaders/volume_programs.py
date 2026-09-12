"""Typed transport helpers. Generated artifacts must be rebuilt from this source."""
import ordinaryshade as osh
from ordinarylight.shaders.native_intersection_programs import NativeIntersection, nativeTraceSurface, nativeIntersectionMiss, nativeIntersectCandidate, nativeSurfaceMask
from .transport_programs import PointLightData

@osh.structure
class VolumeHeader:
    world_to_local: osh.mat4
    dimensions_offset: osh.uvec4
    value_parameters: osh.vec4
    render_parameters: osh.vec4
    scattering_parameters: osh.vec4
    phase_parameters: osh.vec4
    multiple_scattering_parameters: osh.vec4
    acceleration_parameters: osh.uvec4
    clip_parameters: osh.uvec4
    clip_planes: osh.array(osh.vec4, 8)

@osh.function
def isVolumePrimitive(primitive: osh.u32) -> osh.boolean:
    return triangle_volumes[primitive] != osh.u32(4294967295)

@osh.function
def volumeInterval(header: VolumeHeader, origin: osh.vec3, direction: osh.vec3) -> osh.vec2:
    local_origin = (header.world_to_local * osh.vec4(origin, 1.0)).xyz
    local_direction = (header.world_to_local * osh.vec4(direction, 0.0)).xyz
    safe_direction = osh.vec3(local_direction.x if osh.absolute(local_direction.x) > 1e-10 else 1e-10, local_direction.y if osh.absolute(local_direction.y) > 1e-10 else 1e-10, local_direction.z if osh.absolute(local_direction.z) > 1e-10 else 1e-10)
    first = (osh.vec3(0.0) - local_origin) / safe_direction
    second = (osh.vec3(1.0) - local_origin) / safe_direction
    lower = osh.minimum(first, second)
    upper = osh.maximum(first, second)
    return osh.vec2(osh.maximum(osh.maximum(lower.x, lower.y), lower.z), osh.minimum(osh.minimum(upper.x, upper.y), upper.z))

@osh.function
def volumeSliceDistances(header: VolumeHeader, origin: osh.vec3, direction: osh.vec3) -> osh.vec3:
    local_origin = (header.world_to_local * osh.vec4(origin, 1.0)).xyz
    local_direction = (header.world_to_local * osh.vec4(direction, 0.0)).xyz
    distances = osh.vec3(1e+30)
    axis = osh.u32(0)
    while axis < osh.u32(3):
        if header.clip_parameters.z & osh.u32(1) << axis != osh.u32(0) and osh.absolute(local_direction[axis]) > 1e-10:
            distances[axis] = (header.clip_planes[7][axis] - local_origin[axis]) / local_direction[axis]
        axis = axis + 1
    if distances.x > distances.y:
        temporary = distances.x
        distances.x = distances.y
        distances.y = temporary
    if distances.y > distances.z:
        temporary = distances.y
        distances.y = distances.z
        distances.z = temporary
    if distances.x > distances.y:
        temporary = distances.x
        distances.x = distances.y
        distances.y = temporary
    return distances

@osh.function
def sampleVolumeTexture(index: osh.u32, coordinate: osh.vec3) -> osh.f32:
    if index == osh.u32(0):
        return volume_textures.sample(0, coordinate)
    elif index == osh.u32(1):
        return volume_textures.sample(1, coordinate)
    elif index == osh.u32(2):
        return volume_textures.sample(2, coordinate)
    elif index == osh.u32(3):
        return volume_textures.sample(3, coordinate)
    elif index == osh.u32(4):
        return volume_textures.sample(4, coordinate)
    elif index == osh.u32(5):
        return volume_textures.sample(5, coordinate)
    elif index == osh.u32(6):
        return volume_textures.sample(6, coordinate)
    elif index == osh.u32(7):
        return volume_textures.sample(7, coordinate)
    elif index == osh.u32(8):
        return volume_textures.sample(8, coordinate)
    elif index == osh.u32(9):
        return volume_textures.sample(9, coordinate)
    elif index == osh.u32(10):
        return volume_textures.sample(10, coordinate)
    elif index == osh.u32(11):
        return volume_textures.sample(11, coordinate)
    elif index == osh.u32(12):
        return volume_textures.sample(12, coordinate)
    elif index == osh.u32(13):
        return volume_textures.sample(13, coordinate)
    elif index == osh.u32(14):
        return volume_textures.sample(14, coordinate)
    else:
        return volume_textures.sample(15, coordinate)

@osh.function
def volumeScalar(volume_index: osh.u32, header: VolumeHeader, world_position: osh.vec3) -> osh.f32:
    plane_index = osh.u32(0)
    while plane_index < osh.minimum(header.clip_parameters.x, osh.u32(8)):
        plane = header.clip_planes[plane_index]
        if osh.dot(plane.xyz, world_position) < plane.w:
            return -1.0
        plane_index = plane_index + 1
    local = osh.clamp((header.world_to_local * osh.vec4(world_position, 1.0)).xyz, osh.vec3(0.0), osh.vec3(1.0))
    physical = sampleVolumeTexture(volume_index, local)
    if header.clip_parameters.y != osh.u32(0) and osh.is_nan(physical):
        return -2.0
    mapped = physical
    mapping = osh.u32(header.phase_parameters.z)
    if mapping == osh.u32(1):
        mapped = osh.logarithm(physical) if physical > 0.0 else -3.402823e+38
    elif mapping == osh.u32(2):
        mapped = osh.sign(physical) * osh.logarithm(1.0 + osh.absolute(physical) / header.phase_parameters.w)
    return (mapped - header.render_parameters.y) * header.render_parameters.w

@osh.function
def volumeTransferSample(header: VolumeHeader, value: osh.f32) -> osh.vec4:
    offset = osh.u32(header.value_parameters.x)
    if value < -1.5:
        return volume_transfer[offset]
    if value < 0.0:
        return osh.vec4(0.0)
    count = osh.u32(header.value_parameters.y)
    reserved = osh.minimum(header.clip_parameters.y, osh.u32(1))
    offset = offset + reserved
    count = count - reserved
    coordinate = osh.clamp(value, 0.0, 1.0) * osh.f32(osh.maximum(count, osh.u32(1)) - osh.u32(1))
    lower = osh.u32(osh.floor(coordinate))
    upper = osh.minimum(lower + osh.u32(1), count - osh.u32(1))
    return osh.mix(volume_transfer[offset + lower], volume_transfer[offset + upper], osh.fraction(coordinate))

@osh.function
def volumeBrickIndexFromVoxel(header: VolumeHeader, voxel_position: osh.vec3) -> osh.uvec3:
    brick_grid = header.acceleration_parameters.yzw
    return osh.minimum(osh.uvec3(osh.floor(osh.maximum(voxel_position, osh.vec3(0.0)) / 8.0)), brick_grid - osh.uvec3(osh.u32(1)))

@osh.function
def volumeBrickOccupiedAtVoxel(header: VolumeHeader, voxel_position: osh.vec3) -> osh.boolean:
    brick_grid = header.acceleration_parameters.yzw
    if osh.any_value(brick_grid == osh.uvec3(osh.u32(0))):
        return True
    brick = volumeBrickIndexFromVoxel(header, voxel_position)
    linear_index = brick.x + brick_grid.x * (brick.y + brick_grid.y * brick.z)
    return volume_scalars[header.acceleration_parameters.x + linear_index] > 0.5

@osh.function
def volumeVoxelRay(header: VolumeHeader, origin: osh.vec3, direction: osh.vec3, voxel_origin: osh.out(osh.vec3), voxel_direction: osh.out(osh.vec3)) -> osh.void:
    local_direction = (header.world_to_local * osh.vec4(direction, 0.0)).xyz
    voxel_extent = osh.vec3(header.dimensions_offset.xyz) - osh.vec3(1.0)
    voxel_origin = (header.world_to_local * osh.vec4(origin, 1.0)).xyz * voxel_extent
    voxel_direction = local_direction * voxel_extent

@osh.function
def volumeBrickExitDistanceAtVoxel(header: VolumeHeader, voxel_position: osh.vec3, voxel_direction: osh.vec3, distance: osh.f32) -> osh.f32:
    voxel_extent = osh.vec3(header.dimensions_offset.xyz) - osh.vec3(1.0)
    voxel_position = osh.clamp(voxel_position, osh.vec3(0.0), voxel_extent)
    brick = volumeBrickIndexFromVoxel(header, voxel_position)
    lower = osh.vec3(brick) * 8.0
    upper = osh.minimum(lower + osh.vec3(8.0), voxel_extent)
    next_delta = 1e+30
    axis = osh.u32(0)
    while axis < osh.u32(3):
        if osh.absolute(voxel_direction[axis]) <= 1e-10:
            axis = axis + 1
            continue
        boundary = upper[axis] if voxel_direction[axis] > 0.0 else lower[axis]
        candidate = (boundary - voxel_position[axis]) / voxel_direction[axis]
        if candidate >= -1e-06:
            next_delta = osh.minimum(next_delta, osh.maximum(candidate, 0.0))
        axis = axis + 1
    return distance + next_delta

@osh.function
def volumeBrickOccupied(header: VolumeHeader, world_position: osh.vec3) -> osh.boolean:
    voxel_origin = osh.vec3(0)
    voxel_direction = osh.vec3(0)
    volumeVoxelRay(header, world_position, osh.vec3(0.0), voxel_origin, voxel_direction)
    return volumeBrickOccupiedAtVoxel(header, voxel_origin)

@osh.function
def volumeBrickExitDistance(header: VolumeHeader, origin: osh.vec3, direction: osh.vec3, distance: osh.f32) -> osh.f32:
    voxel_origin = osh.vec3(0)
    voxel_direction = osh.vec3(0)
    volumeVoxelRay(header, origin, direction, voxel_origin, voxel_direction)
    return volumeBrickExitDistanceAtVoxel(header, voxel_origin + voxel_direction * distance, voxel_direction, distance)

@osh.function
def volumePhase(header: VolumeHeader, cosine: osh.f32) -> osh.f32:
    if header.phase_parameters.y < 0.5:
        return 0.0795774715459
    g = osh.clamp(header.phase_parameters.x, -0.99, 0.99)
    denominator = osh.maximum(1.0 + g * g - 2.0 * g * osh.clamp(cosine, -1.0, 1.0), 1e-08)
    return (1.0 - g * g) / (12.5663706144 * denominator * osh.sqrt(denominator))

@osh.function
def volumeOpaqueVisibility(world_position: osh.vec3, direction: osh.vec3, maximum_distance: osh.f32) -> osh.f32:
    shadow_distance = osh.maximum(maximum_distance - 0.004, 0.001) if maximum_distance < 1e+29 else 1e+30
    shadow = nativeTraceSurface(world_position + direction * 0.002, 0.001, direction, shadow_distance, True, nativeSurfaceMask())
    return 1.0 if shadow.address.w == gl_RayQueryCommittedIntersectionNoneEXT else 0.0

@osh.function
def approximateVolumeLightTransmittance(world_position: osh.vec3, light_direction: osh.vec3, light_distance: osh.f32) -> osh.f32:
    volume_count = osh.minimum(osh.u32(volume_headers[0].render_parameters.z), osh.u32(16))
    optical_depth = 0.0
    volume_index = osh.u32(0)
    while volume_index < volume_count:
        medium = volume_headers[volume_index]
        interval = volumeInterval(medium, world_position, light_direction)
        entry = osh.maximum(interval.x, 0.0)
        exit_distance = osh.minimum(interval.y, light_distance)
        if exit_distance <= entry:
            volume_index = volume_index + 1
            continue
        midpoint = 0.5 * (entry + exit_distance)
        sample_value = volumeTransferSample(medium, volumeScalar(volume_index, medium, world_position + light_direction * midpoint))
        reference_alpha = osh.clamp(sample_value.a * medium.value_parameters.z, 0.0, 0.999999)
        extinction = -osh.logarithm(1.0 - reference_alpha) / osh.maximum(medium.render_parameters.x, 1e-05)
        optical_depth = optical_depth + extinction * (exit_distance - entry)
        volume_index = volume_index + 1
    return osh.exp(-optical_depth)

@osh.function
def volumePointScattering(header: VolumeHeader, world_position: osh.vec3, ray_direction: osh.vec3, optical_depth: osh.f32) -> osh.vec3:
    scattering_scale = header.scattering_parameters.w
    if scattering_scale <= 0.0:
        return osh.vec3(0.0)
    scattered = osh.vec3(0.0)
    if osh.specialization('WAVE_VOLUME_MULTIPLE_SCATTERING'):
        isotropic_scattered = osh.vec3(0.0)
    outgoing = -ray_direction
    light_index = osh.u32(0)
    while light_index < osh.minimum(push.point_light_count, osh.u32(64)):
        light = point_lights[light_index]
        light_type = osh.i32(light.position_type.w + 0.5)
        if light_type == 3:
            light_index = light_index + 1
            continue
        incoming = osh.vec3(0)
        distance_to_light = 10000.0
        attenuation = 1.0
        if light_type == 1:
            incoming = -osh.normalize(light.direction_range.xyz)
        else:
            offset = light.position_type.xyz - world_position
            distance_squared = osh.maximum(osh.dot(offset, offset), 1e-06)
            distance_to_light = osh.sqrt(distance_squared)
            incoming = offset / distance_to_light
            if light.direction_range.w > 0.0 and distance_to_light > light.direction_range.w:
                light_index = light_index + 1
                continue
            attenuation = 1.0 / distance_squared
            if light_type == 2:
                cone = osh.dot(osh.normalize(light.direction_range.xyz), -incoming)
                spot = osh.smoothstep(light.spot_parameters.y, light.spot_parameters.x, cone)
                if spot <= 0.0:
                    light_index = light_index + 1
                    continue
                attenuation = attenuation * spot
        incident = light.color_intensity.rgb * light.color_intensity.a * attenuation
        incident = incident * approximateVolumeLightTransmittance(world_position, incoming, distance_to_light)
        incident = incident * volumeOpaqueVisibility(world_position, incoming, distance_to_light)
        scattered = scattered + incident * volumePhase(header, osh.dot(-incoming, outgoing))
        if osh.specialization('WAVE_VOLUME_MULTIPLE_SCATTERING'):
            isotropic_scattered = isotropic_scattered + incident * 0.0795774715459
        light_index = light_index + 1
    light_index = osh.u32(0)
    while light_index < osh.minimum(push.area_light_count, osh.u32(64)):
        light = area_lights[light_index]
        light_position = (light.a.xyz + light.b.xyz + light.c.xyz) / 3.0
        offset = light_position - world_position
        distance_squared = osh.maximum(osh.dot(offset, offset), 1e-06)
        distance_to_light = osh.sqrt(distance_squared)
        incoming = offset / distance_to_light
        light_normal = osh.normalize(osh.cross(light.b.xyz - light.a.xyz, light.c.xyz - light.a.xyz))
        raw_cosine = osh.dot(light_normal, -incoming)
        light_cosine = osh.absolute(raw_cosine) if light.distribution.z > 0.5 else osh.maximum(raw_cosine, 0.0)
        if light_cosine <= 1e-06:
            light_index = light_index + 1
            continue
        visibility = volumeOpaqueVisibility(world_position, incoming, distance_to_light)
        transmittance = approximateVolumeLightTransmittance(world_position, incoming, distance_to_light)
        incident = light.emission_area.rgb * light.emission_area.a * light_cosine / distance_squared
        incident = incident * (visibility * transmittance)
        scattered = scattered + incident * volumePhase(header, osh.dot(-incoming, outgoing))
        if osh.specialization('WAVE_VOLUME_MULTIPLE_SCATTERING'):
            isotropic_scattered = isotropic_scattered + incident * 0.0795774715459
        light_index = light_index + 1
    environment_directions = osh.local_array(osh.vec3, 4)
    environment_directions[0] = osh.vec3(0.577350269, 0.577350269, 0.577350269)
    environment_directions[1] = osh.vec3(-0.577350269, -0.577350269, 0.577350269)
    environment_directions[2] = osh.vec3(-0.577350269, 0.577350269, -0.577350269)
    environment_directions[3] = osh.vec3(0.577350269, -0.577350269, -0.577350269)
    environment_count = osh.minimum(push.environment_samples, osh.u32(4))
    sample_index = osh.u32(0)
    while sample_index < environment_count:
        incoming = environment_directions[sample_index]
        visibility = volumeOpaqueVisibility(world_position, incoming, 1e+30)
        transmittance = approximateVolumeLightTransmittance(world_position, incoming, 1e+30)
        incident = environmentColor(incoming) * visibility * transmittance * (12.5663706144 / osh.f32(environment_count))
        scattered = scattered + incident * volumePhase(header, osh.dot(-incoming, outgoing))
        if osh.specialization('WAVE_VOLUME_MULTIPLE_SCATTERING'):
            isotropic_scattered = isotropic_scattered + incident * 0.0795774715459
        sample_index = sample_index + 1
    if osh.specialization('WAVE_VOLUME_MULTIPLE_SCATTERING'):
        scattering_orders = osh.clamp(osh.u32(header.multiple_scattering_parameters.w + 0.5), osh.u32(1), osh.u32(8))
        ratio = osh.clamp(header.multiple_scattering_parameters.rgb * (1.0 - osh.exp(-osh.maximum(optical_depth, 0.0))), osh.vec3(0.0), osh.vec3(0.999))
        order_weight = ratio
        order = osh.u32(2)
        while order <= scattering_orders:
            scattered = scattered + isotropic_scattered * order_weight
            order_weight = order_weight * ratio
            order = order + 1
    return scattered * header.scattering_parameters.rgb * scattering_scale

@osh.function
def integrateVolumeUntil(volume_index: osh.u32, origin: osh.vec3, direction: osh.vec3, maximum_distance: osh.f32, radiance: osh.inout(osh.vec3), throughput: osh.inout(osh.vec3)) -> osh.f32:
    header = volume_headers[volume_index]
    interval = volumeInterval(header, origin, direction)
    entry = osh.maximum(interval.x, 0.0)
    exit_distance = osh.minimum(interval.y, maximum_distance)
    if exit_distance <= entry:
        return entry
    if header.clip_parameters.z != osh.u32(0):
        distances = volumeSliceDistances(header, origin, direction)
        if header.clip_parameters.w & osh.u32(1) != osh.u32(0):
            pass
        else:
            slice_index = osh.u32(0)
            while slice_index < osh.u32(3):
                slice_distance = distances[slice_index]
                if slice_distance < entry or slice_distance > exit_distance:
                    slice_index = slice_index + 1
                    continue
                sample_value = volumeTransferSample(header, volumeScalar(volume_index, header, origin + direction * slice_distance))
                alpha = osh.clamp(sample_value.a * header.value_parameters.z, 0.0, 1.0)
                radiance = radiance + throughput * alpha * sample_value.rgb * header.value_parameters.w
                throughput = throughput * (1.0 - alpha)
                slice_index = slice_index + 1
            return exit_distance
    reference_step = osh.maximum(header.render_parameters.x, 1e-05)
    steps = osh.minimum(osh.u32(osh.ceiling((exit_distance - entry) / reference_step)), osh.u32(4096))
    step_size = (exit_distance - entry) / osh.f32(osh.maximum(steps, osh.u32(1)))
    transmittance = 1.0
    integrated = osh.vec3(0.0)
    isosurface_enabled = header.clip_parameters.w & osh.u32(2) != osh.u32(0)
    volume_enabled = header.clip_parameters.w != osh.u32(2)
    isovalue = header.clip_planes[7].w
    previous_distance = entry
    previous_scalar = volumeScalar(volume_index, header, origin + direction * previous_distance)
    if osh.specialization('WAVE_VOLUME_SCATTERING'):
        if osh.specialization('WAVE_VOLUME_MULTIPLE_SCATTERING'):
            scattering_midpoint = 0.5 * (entry + exit_distance)
            scattering_sample = volumeTransferSample(header, volumeScalar(volume_index, header, origin + direction * scattering_midpoint))
            scattering_alpha = osh.clamp(scattering_sample.a * header.value_parameters.z, 0.0, 0.999999)
            scattering_extinction = -osh.logarithm(1.0 - scattering_alpha) / reference_step
            scattering_source = volumePointScattering(header, origin + direction * scattering_midpoint, direction, scattering_extinction * (exit_distance - entry))
        else:
            scattering_source = volumePointScattering(header, origin + direction * (0.5 * (entry + exit_distance)), direction, 0.0)
    step_index = osh.u32(0)
    if osh.specialization('WAVE_VOLUME_EMPTY_SPACE_SKIPPING'):
        voxel_origin = osh.vec3(0)
        voxel_direction = osh.vec3(0)
        volumeVoxelRay(header, origin, direction, voxel_origin, voxel_direction)
        occupied_until = -1.0
    while step_index < steps:
        distance = entry + (osh.f32(step_index) + 0.5) * step_size
        if osh.specialization('WAVE_VOLUME_EMPTY_SPACE_SKIPPING'):
            world_position = origin + direction * distance
            voxel_position = voxel_origin + voxel_direction * distance
            if not isosurface_enabled and distance + 1e-07 >= occupied_until:
                brick_exit = volumeBrickExitDistanceAtVoxel(header, voxel_position, voxel_direction, distance)
                if not volumeBrickOccupiedAtVoxel(header, voxel_position):
                    jump = osh.u32(osh.clamp(osh.ceiling((brick_exit - distance) / step_size), 1.0, osh.f32(steps - step_index)))
                    step_index = osh.minimum(step_index + jump, steps)
                    continue
                occupied_until = brick_exit
        scalar = volumeScalar(volume_index, header, origin + direction * distance)
        sample_value = volumeTransferSample(header, scalar)
        if volume_enabled:
            reference_alpha = osh.clamp(sample_value.a * header.value_parameters.z, 0.0, 0.999999)
            alpha = 1.0 - osh.power(1.0 - reference_alpha, step_size / reference_step)
            source = sample_value.rgb * header.value_parameters.w
            if osh.specialization('WAVE_VOLUME_SCATTERING'):
                source = source + scattering_source
            integrated = integrated + transmittance * alpha * source
            transmittance = transmittance * (1.0 - alpha)
        if ((isosurface_enabled and previous_scalar >= 0.0) and scalar >= 0.0) and (previous_scalar < isovalue and scalar >= isovalue or (previous_scalar > isovalue and scalar <= isovalue)):
            lower_distance = previous_distance
            upper_distance = distance
            lower_scalar = previous_scalar
            refinement = osh.u32(0)
            while refinement < osh.u32(8):
                middle_distance = 0.5 * (lower_distance + upper_distance)
                middle_scalar = volumeScalar(volume_index, header, origin + direction * middle_distance)
                if middle_scalar < 0.0:
                    break
                if lower_scalar < isovalue and middle_scalar < isovalue or (lower_scalar > isovalue and middle_scalar > isovalue):
                    lower_distance = middle_distance
                    lower_scalar = middle_scalar
                else:
                    upper_distance = middle_distance
                refinement = refinement + 1
            surface_sample = volumeTransferSample(header, isovalue)
            surface_alpha = osh.clamp(surface_sample.a * header.value_parameters.z, 0.0, 1.0)
            integrated = integrated + transmittance * surface_alpha * surface_sample.rgb * header.value_parameters.w
            transmittance = transmittance * (1.0 - surface_alpha)
            if not volume_enabled or transmittance <= 0.001:
                break
        previous_distance = distance
        previous_scalar = scalar
        if header.clip_parameters.w & osh.u32(1) != osh.u32(0):
            slice_distances = volumeSliceDistances(header, origin, direction)
            half_step = 0.5 * step_size + 1e-07
            slice_index = osh.u32(0)
            while slice_index < osh.u32(3):
                slice_distance = slice_distances[slice_index]
                if osh.absolute(slice_distance - distance) > half_step:
                    slice_index = slice_index + 1
                    continue
                slice_sample = volumeTransferSample(header, volumeScalar(volume_index, header, origin + direction * slice_distance))
                slice_alpha = osh.clamp(slice_sample.a * header.value_parameters.z, 0.0, 1.0)
                integrated = integrated + transmittance * slice_alpha * slice_sample.rgb * header.value_parameters.w
                transmittance = transmittance * (1.0 - slice_alpha)
                slice_index = slice_index + 1
        if transmittance < 0.0001:
            break
        step_index = step_index + osh.u32(1)
    radiance = radiance + throughput * integrated
    throughput = throughput * transmittance
    return exit_distance

@osh.function
def integrateVolume(volume_index: osh.u32, origin: osh.vec3, direction: osh.vec3, radiance: osh.inout(osh.vec3), throughput: osh.inout(osh.vec3)) -> osh.f32:
    return integrateVolumeUntil(volume_index, origin, direction, 1e+30, radiance, throughput)

@osh.function
def integrateOverlappingVolumesBeforeSurface(origin: osh.vec3, direction: osh.vec3, surface_distance: osh.f32, radiance: osh.inout(osh.vec3), throughput: osh.inout(osh.vec3)) -> osh.void:
    maximum_volumes = osh.u32(16)
    volume_count = osh.minimum(osh.u32(volume_headers[0].render_parameters.z), maximum_volumes)
    entries = osh.local_array(osh.f32, 16)
    exits = osh.local_array(osh.f32, 16)
    volume_index = osh.u32(0)
    while volume_index < maximum_volumes:
        entries[volume_index] = 1.0
        exits[volume_index] = 0.0
        if volume_index >= volume_count:
            volume_index = volume_index + 1
            continue
        interval = volumeInterval(volume_headers[volume_index], origin, direction)
        entry = osh.maximum(interval.x, 0.0)
        exit_distance = osh.minimum(interval.y, surface_distance)
        if exit_distance > entry:
            entries[volume_index] = entry
            exits[volume_index] = exit_distance
        volume_index = volume_index + 1
    union_entry = surface_distance
    union_exit = 0.0
    reference_step = 1e+30
    if osh.specialization('WAVE_VOLUME_SCATTERING'):
        scattering_sources = osh.local_array(osh.vec3, 16)
    volume_index = osh.u32(0)
    while volume_index < volume_count:
        if osh.specialization('WAVE_VOLUME_SCATTERING'):
            scattering_sources[volume_index] = osh.vec3(0.0)
        if exits[volume_index] <= entries[volume_index]:
            volume_index = volume_index + 1
            continue
        union_entry = osh.minimum(union_entry, entries[volume_index])
        union_exit = osh.maximum(union_exit, exits[volume_index])
        reference_step = osh.minimum(reference_step, osh.maximum(volume_headers[volume_index].render_parameters.x, 1e-05))
        if osh.specialization('WAVE_VOLUME_SCATTERING'):
            if osh.specialization('WAVE_VOLUME_MULTIPLE_SCATTERING'):
                scattering_header = volume_headers[volume_index]
                scattering_midpoint = 0.5 * (entries[volume_index] + exits[volume_index])
                scattering_position = origin + direction * scattering_midpoint
                scattering_sample = volumeTransferSample(scattering_header, volumeScalar(volume_index, scattering_header, scattering_position))
                scattering_alpha = osh.clamp(scattering_sample.a * scattering_header.value_parameters.z, 0.0, 0.999999)
                scattering_extinction = -osh.logarithm(1.0 - scattering_alpha) / osh.maximum(scattering_header.render_parameters.x, 1e-05)
                scattering_sources[volume_index] = volumePointScattering(scattering_header, scattering_position, direction, scattering_extinction * (exits[volume_index] - entries[volume_index]))
            else:
                scattering_sources[volume_index] = volumePointScattering(volume_headers[volume_index], origin + direction * (0.5 * (entries[volume_index] + exits[volume_index])), direction, 0.0)
        volume_index = volume_index + 1
    if union_exit <= union_entry:
        return
    steps = osh.minimum(osh.u32(osh.ceiling((union_exit - union_entry) / reference_step)), osh.u32(4096))
    step_size = (union_exit - union_entry) / osh.f32(osh.maximum(steps, osh.u32(1)))
    step_index = osh.u32(0)
    while step_index < steps:
        distance = union_entry + (osh.f32(step_index) + 0.5) * step_size
        world_position = origin + direction * distance
        if osh.specialization('WAVE_VOLUME_EMPTY_SPACE_SKIPPING'):
            any_occupied = False
            empty_exit = 1e+30
            volume_index = osh.u32(0)
            while volume_index < volume_count:
                if distance < entries[volume_index] or distance >= exits[volume_index]:
                    volume_index = volume_index + 1
                    continue
                header = volume_headers[volume_index]
                if volumeBrickOccupied(header, world_position):
                    any_occupied = True
                    break
                empty_exit = osh.minimum(empty_exit, volumeBrickExitDistance(header, origin, direction, distance))
                volume_index = volume_index + 1
            if not any_occupied:
                jump = osh.u32(osh.clamp(osh.ceiling((empty_exit - distance) / step_size), 1.0, osh.f32(steps - step_index)))
                step_index = osh.minimum(step_index + jump, steps)
                continue
        extinction = 0.0
        emission_extinction = osh.vec3(0.0)
        volume_index = osh.u32(0)
        while volume_index < volume_count:
            if distance < entries[volume_index] or distance >= exits[volume_index]:
                volume_index = volume_index + 1
                continue
            header = volume_headers[volume_index]
            sample_value = volumeTransferSample(header, volumeScalar(volume_index, header, world_position))
            reference_alpha = osh.clamp(sample_value.a * header.value_parameters.z, 0.0, 0.999999)
            medium_extinction = -osh.logarithm(1.0 - reference_alpha) / osh.maximum(header.render_parameters.x, 1e-05)
            extinction = extinction + medium_extinction
            emission_extinction = emission_extinction + medium_extinction * sample_value.rgb * header.value_parameters.w
            if osh.specialization('WAVE_VOLUME_SCATTERING'):
                emission_extinction = emission_extinction + medium_extinction * scattering_sources[volume_index]
            volume_index = volume_index + 1
        if extinction > 1e-08:
            alpha = 1.0 - osh.exp(-extinction * step_size)
            radiance = radiance + throughput * alpha * emission_extinction / extinction
            throughput = throughput * (1.0 - alpha)
        if osh.maximum(throughput.r, osh.maximum(throughput.g, throughput.b)) < 0.0001:
            return
        step_index = step_index + osh.u32(1)

@osh.function
def integrateVolumesBeforeSurface(origin: osh.vec3, direction: osh.vec3, surface_distance: osh.f32, radiance: osh.inout(osh.vec3), throughput: osh.inout(osh.vec3)) -> osh.void:
    if volume_headers[0].dimensions_offset.x == osh.u32(0):
        return
    if osh.specialization('WAVE_OVERLAPPING_VOLUMES'):
        if osh.u32(volume_headers[0].render_parameters.z) > osh.u32(1):
            integrateOverlappingVolumesBeforeSurface(origin, direction, surface_distance, radiance, throughput)
            return
    traversal_origin = origin
    traveled = 0.0
    traversal = osh.u32(0)
    while traversal < osh.u32(32):
        remaining = surface_distance - traveled
        if remaining <= 0.001:
            break
        volume_query = osh.ray_query()
        volume_query.initialize(scene_tlas, gl_RayFlagsOpaqueEXT, 2, traversal_origin, 0.001, direction, remaining)
        while volume_query.proceed():
            pass
        if volume_query.intersection_type(True) != gl_RayQueryCommittedIntersectionTriangleEXT:
            break
        primitive = volume_query.primitive_index(True) + volume_query.instance_custom_index(True)
        volume_index = triangle_volumes[primitive]
        interval = volumeInterval(volume_headers[volume_index], traversal_origin, direction)
        segment_end = osh.minimum(interval.y, remaining)
        integrateVolumeUntil(volume_index, traversal_origin, direction, segment_end, radiance, throughput)
        if segment_end >= remaining - 0.001 or osh.maximum(throughput.r, osh.maximum(throughput.g, throughput.b)) < 0.0001:
            break
        advance = interval.y + 0.002
        traversal_origin = traversal_origin + direction * advance
        traveled = traveled + advance
        traversal = traversal + 1

@osh.function
def volumeShadowTransmittance(origin: osh.vec3, direction: osh.vec3, maximum_distance: osh.f32) -> osh.f32:
    if osh.specialization("WAVE_SURFACE_ONLY"):
        return 1.0
    ignored_radiance = osh.vec3(0.0)
    transmittance = osh.vec3(1.0)
    integrateVolumesBeforeSurface(origin, direction, maximum_distance, ignored_radiance, transmittance)
    return osh.clamp(transmittance.r, 0.0, 1.0)

@osh.structure
class VolumeLightingContext:
    point_light_count: osh.u32
    area_light_count: osh.u32
    environment_samples: osh.u32
