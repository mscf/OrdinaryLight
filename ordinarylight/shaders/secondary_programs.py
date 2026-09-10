"""Typed transport helpers. Generated artifacts must be rebuilt from this source."""
import ordinaryshade as osh
from .lighting_programs import AreaLightCandidate
from .restir_programs import DirectLightReservoir
WAVE_MAX_MEDIUM_STACK_DEPTH = 16
from .transport_programs import MaterialEvaluation

@osh.structure
class WaveRay:
    origin_tmin: osh.vec4
    direction_tmax: osh.vec4
    path_index: osh.u32
    padding_a: osh.u32
    padding_b: osh.u32
    padding_c: osh.u32

@osh.structure
class WaveHit:
    position_t: osh.vec4
    geometric_normal: osh.vec3
    primitive_index: osh.u32
    barycentrics: osh.vec2
    ray_index: osh.u32
    path_index: osh.u32

@osh.structure
class WavePathState:
    throughput: osh.vec4
    radiance: osh.vec4
    metadata: osh.uvec4

@osh.structure
class SecondaryPathState:
    position_valid: osh.vec4
    normal_pdf: osh.vec4
    primary_throughput: osh.vec4
    primary_radiance: osh.vec4
    diffuse_radiance_hit_distance: osh.vec4
    specular_radiance_hit_distance: osh.vec4
    primary_position: osh.vec4
    primary_geometry: osh.vec4

@osh.structure
class WaveMediumStack:
    ior: osh.array(osh.f32, WAVE_MAX_MEDIUM_STACK_DEPTH)

@osh.structure
class MaterialData:
    base_roughness: osh.vec4
    emission_metallic: osh.vec4
    attenuation_transmission: osh.vec4
    ior_distance: osh.vec4
    texture_indices: osh.vec4
    texture_parameters: osh.vec4
    advanced0: osh.vec4
    advanced1: osh.vec4
    sheen_color: osh.vec4
    subsurface_color: osh.vec4
    advanced_texture_indices: osh.vec4
    optical: osh.vec4

@osh.structure
class VertexAttributeData:
    normal: osh.vec4
    texcoord: osh.vec4
    tangent: osh.vec4

@osh.structure
class PointLightData:
    position_type: osh.vec4
    direction_range: osh.vec4
    color_intensity: osh.vec4
    spot_parameters: osh.vec4

@osh.structure
class AreaLightData:
    a: osh.vec4
    b: osh.vec4
    c: osh.vec4
    emission_area: osh.vec4
    distribution: osh.vec4

@osh.structure
class TextureBindingData:
    texture_rotation: osh.vec4
    offset_scale: osh.vec4

@osh.function
def pathRng(path: WavePathState) -> osh.u32:
    return path.metadata.z

@osh.function
def setPathRng(path: osh.inout(WavePathState), rng: osh.u32) -> osh.void:
    path.metadata.z = rng

@osh.function
def pathBounce(path: WavePathState) -> osh.u32:
    return osh.u32(path.throughput.w)

@osh.function
def setPathBounce(path: osh.inout(WavePathState), bounce: osh.u32) -> osh.void:
    path.throughput.w = osh.f32(bounce)

@osh.function
def pathPreviousPdf(path: WavePathState) -> osh.f32:
    return path.radiance.w

@osh.function
def setPathPreviousPdf(path: osh.inout(WavePathState), pdf: osh.f32) -> osh.void:
    path.radiance.w = pdf

@osh.function
def profileWork(counter: osh.u32, amount: osh.u32) -> osh.void:
    osh.atomic_add(work_counters[counter], amount)
    bounce = osh.minimum(profile_bounce, osh.u32(7))
    if counter == osh.u32(0):
        osh.atomic_add(work_counters[osh.u32(16) + bounce], amount)
    elif counter == osh.u32(1):
        osh.atomic_add(work_counters[osh.u32(24) + bounce], amount)
    elif counter == osh.u32(3):
        osh.atomic_add(work_counters[osh.u32(32) + bounce], amount)

@osh.function
def reserveOutputIndex() -> osh.u32:
    if push.subgroup_enqueue == osh.u32(0):
        return osh.atomic_add(output_queue.count, osh.u32(1))
    active_lanes = osh.subgroup_ballot(True)
    base = osh.u32(0)
    if osh.subgroup_elect():
        base = osh.atomic_add(output_queue.count, osh.subgroup_ballot_bit_count(active_lanes))
    base = osh.subgroup_broadcast_first(base)
    return base + osh.subgroup_ballot_exclusive_bit_count(active_lanes)

@osh.function
def terminatePath(path_index: osh.u32) -> osh.void:
    paths[path_index].metadata.w = paths[path_index].metadata.w & ~PATH_ACTIVE_BIT

@osh.function
def main() -> osh.void:
    capture_secondary = push.indirect_secondary_capture != osh.u32(0)
    if osh.specialization('WAVE_DENOISER_SIGNAL_CAPTURE'):
        capture_secondary = True
    hit_index = gl_GlobalInvocationID.x
    hit = WaveHit(osh.vec4(0), osh.vec3(0), osh.u32(0), osh.vec2(0), osh.u32(0), osh.u32(0))
    input_ray = WaveRay(osh.vec4(0), osh.vec4(0), osh.u32(0), osh.u32(0), osh.u32(0), osh.u32(0))
    if push.fused_intersection != osh.u32(0):
        if hit_index >= osh.minimum(input_queue.count, input_queue.capacity):
            return
        input_ray = input_queue.rays[hit_index]
        if osh.specialization('WAVE_WORK_COUNTERS'):
            profile_bounce = osh.minimum(pathBounce(paths[input_ray.path_index]), osh.u32(7))
            profileWork(osh.u32(0), osh.u32(1))
        query = osh.ray_query()
        query.initialize(scene_tlas, gl_RayFlagsOpaqueEXT, 1, input_ray.origin_tmin.xyz, input_ray.origin_tmin.w, input_ray.direction_tmax.xyz, input_ray.direction_tmax.w)
        while query.proceed():
            pass
        hit.path_index = input_ray.path_index
        hit.ray_index = hit_index
        if query.intersection_type(True) == gl_RayQueryCommittedIntersectionTriangleEXT:
            distance = query.intersection_t(True)
            primitive = query.primitive_index(True) + query.instance_custom_index(True)
            a = vertices[primitive * osh.u32(3) + osh.u32(0)].xyz
            b = vertices[primitive * osh.u32(3) + osh.u32(1)].xyz
            c = vertices[primitive * osh.u32(3) + osh.u32(2)].xyz
            hit.position_t = osh.vec4(input_ray.origin_tmin.xyz + distance * input_ray.direction_tmax.xyz, distance)
            hit.geometric_normal = osh.normalize(osh.cross(b - a, c - a))
            hit.primitive_index = primitive
            hit.barycentrics = query.barycentrics(True)
        else:
            if osh.specialization('WAVE_WORK_COUNTERS'):
                profileWork(osh.u32(4), osh.u32(1))
            hit.position_t = osh.vec4(0.0, 0.0, 0.0, -1.0)
            hit.geometric_normal = osh.vec3(0.0)
            hit.primitive_index = osh.u32(4294967295)
            hit.barycentrics = osh.vec2(0.0)
    else:
        if hit_index >= osh.minimum(hit_queue.count, hit_queue.capacity):
            return
        hit = hit_queue.hits[hit_index]
        input_ray = input_queue.rays[hit.ray_index]
    path_index = hit.path_index
    path = paths[path_index]
    if osh.specialization('WAVE_WORK_COUNTERS'):
        profile_bounce = osh.minimum(pathBounce(path), osh.u32(7))
    incoming = input_ray.direction_tmax.xyz
    surface_distance = input_ray.direction_tmax.w if hit.primitive_index == osh.u32(4294967295) else hit.position_t.w
    integrateVolumesBeforeSurface(input_ray.origin_tmin.xyz, incoming, surface_distance, path.radiance.rgb, path.throughput.rgb)
    if osh.maximum(path.throughput.r, osh.maximum(path.throughput.g, path.throughput.b)) < 0.0001:
        path.metadata.w = path.metadata.w & ~PATH_ACTIVE_BIT
        paths[path_index] = path
        return
    if hit.primitive_index == osh.u32(4294967295):
        environment_mis = 1.0
        if path.metadata.w & PATH_PREVIOUS_DIFFUSE_BIT != osh.u32(0) and push.environment_samples > osh.u32(0):
            pdf = pathPreviousPdf(path)
            light_pdf = pdf * osh.f32(push.environment_samples)
            if path.metadata.w & PATH_PREVIOUS_UNIFIED_NEE_BIT != osh.u32(0):
                light_pdf = pdf * (1.0 - unifiedAreaDomainProbability())
            environment_mis = powerHeuristic(pdf, light_pdf)
        path.radiance.rgb = path.radiance.rgb + path.throughput.rgb * environmentColor(incoming) * environment_mis
        path.metadata.w = path.metadata.w & ~PATH_ACTIVE_BIT
        paths[path_index] = path
        return
    if osh.specialization('WAVE_WORK_COUNTERS'):
        profileWork(osh.u32(3), osh.u32(1))
    primitive = hit.primitive_index
    cone_spread = osh.uint_bits_to_float(input_ray.padding_b)
    cone_width = osh.uint_bits_to_float(input_ray.padding_a) + hit.position_t.w * cone_spread
    weights = osh.vec3(1.0 - hit.barycentrics.x - hit.barycentrics.y, hit.barycentrics.x, hit.barycentrics.y)
    shading_normal = osh.normalize(attributes[primitive * osh.u32(3) + osh.u32(0)].normal.xyz * weights.x + attributes[primitive * osh.u32(3) + osh.u32(1)].normal.xyz * weights.y + attributes[primitive * osh.u32(3) + osh.u32(2)].normal.xyz * weights.z)
    geometric_normal = hit.geometric_normal
    if osh.dot(shading_normal, geometric_normal) < 0.0:
        shading_normal = -shading_normal
    entering = osh.dot(incoming, geometric_normal) < 0.0
    normal = shading_normal if entering else -shading_normal
    material = materials[primitive]
    a = vertices[primitive * osh.u32(3) + osh.u32(0)].xyz
    b = vertices[primitive * osh.u32(3) + osh.u32(1)].xyz
    c = vertices[primitive * osh.u32(3) + osh.u32(2)].xyz
    if materialHasTextures(material):
        uv0 = attributes[primitive * osh.u32(3) + osh.u32(0)].texcoord.xy * weights.x + attributes[primitive * osh.u32(3) + osh.u32(1)].texcoord.xy * weights.y + attributes[primitive * osh.u32(3) + osh.u32(2)].texcoord.xy * weights.z
        uv1 = osh.vec2(0.0)
        uv0_footprint = cone_width * attributes[primitive * osh.u32(3) + osh.u32(0)].normal.w
        uv1_footprint = 0.0
        if materialUsesUv1(material):
            uv1 = attributes[primitive * osh.u32(3) + osh.u32(0)].texcoord.zw * weights.x + attributes[primitive * osh.u32(3) + osh.u32(1)].texcoord.zw * weights.y + attributes[primitive * osh.u32(3) + osh.u32(2)].texcoord.zw * weights.z
            uv1_footprint = cone_width * triangleUvDensity(a, b, c, attributes[primitive * osh.u32(3) + osh.u32(0)].texcoord.zw, attributes[primitive * osh.u32(3) + osh.u32(1)].texcoord.zw, attributes[primitive * osh.u32(3) + osh.u32(2)].texcoord.zw)
        applyMaterialTextures(material, uv0, uv1, uv0_footprint, uv1_footprint)
        tangent_data = attributes[primitive * osh.u32(3) + osh.u32(0)].tangent * weights.x + attributes[primitive * osh.u32(3) + osh.u32(1)].tangent * weights.y + attributes[primitive * osh.u32(3) + osh.u32(2)].tangent * weights.z
        if textureBindingUsesUv1(material.texture_indices.w):
            tangent_data = triangleTangent(a, b, c, attributes[primitive * osh.u32(3) + osh.u32(0)].texcoord.zw, attributes[primitive * osh.u32(3) + osh.u32(1)].texcoord.zw, attributes[primitive * osh.u32(3) + osh.u32(2)].texcoord.zw, shading_normal)
        shading_normal = applyNormalTexture(material, uv0, uv1, uv0_footprint, uv1_footprint, shading_normal, tangent_data)
        if osh.dot(shading_normal, geometric_normal) < 0.0:
            shading_normal = -shading_normal
        normal = shading_normal if entering else -shading_normal
    wave_surface_response = MaterialEvaluation(osh.vec3(0.0), osh.vec3(0.0), 0.0, 0.0, 0.0, 0.0, osh.vec3(0.0), 0.0, 0.0, osh.vec3(0.0), osh.vec3(0.0), 0.0, 0.0)
    wave_surface_rng = pathRng(path)
    if osh.specialization('WAVE_CUSTOM_MATERIAL_PROGRAM'):
        wave_material_uv = attributes[primitive * osh.u32(3)].texcoord.xy * weights.x + attributes[primitive * osh.u32(3) + osh.u32(1)].texcoord.xy * weights.y + attributes[primitive * osh.u32(3) + osh.u32(2)].texcoord.xy * weights.z
        wave_surface_response = waveApplyMaterialProgram(material, normal, wave_material_uv, incoming, entering, primitive, weights, osh.f32(pathBounce(path)))
        if wave_surface_response.custom_scattering > 0.5:
            wave_medium_depth = osh.maximum(path.metadata.w >> osh.u32(8), osh.u32(1))
            wave_current_ior = stacks[path_index].ior[wave_medium_depth - osh.u32(1)]
            wave_exterior_ior = osh.maximum(material.ior_distance.x, 1.0001) if entering else stacks[path_index].ior[wave_medium_depth - osh.u32(2)] if wave_medium_depth > osh.u32(1) else 1.0
            wave_random_u = randomFloat(wave_surface_rng)
            wave_random_v = randomFloat(wave_surface_rng)
            wave_surface_response = evaluateMaterial(material, normal, wave_material_uv, incoming, entering, wave_random_u, wave_random_v, osh.f32(pathBounce(path)), wave_current_ior, wave_exterior_ior)
    if (capture_secondary and pathBounce(path) == osh.u32(1)) and secondary_paths[path_index].primary_throughput.w > 0.5:
        secondary_paths[path_index].position_valid = osh.vec4(hit.position_t.xyz, 1.0)
        secondary_paths[path_index].normal_pdf.xyz = normal
    emission_visible = entering or material.ior_distance.w > 0.5
    if emission_visible:
        emission_mis = 1.0
        if path.metadata.w & PATH_PREVIOUS_DIFFUSE_BIT != osh.u32(0) and osh.dot(material.emission_metallic.rgb, material.emission_metallic.rgb) > 0.0:
            area = 0.5 * osh.length(osh.cross(b - a, c - a))
            light_cosine_raw = osh.dot(geometric_normal, -incoming)
            light_cosine = osh.absolute(light_cosine_raw) if material.ior_distance.w > 0.5 else osh.maximum(light_cosine_raw, 0.0)
            luminance = osh.dot(material.emission_metallic.rgb, osh.vec3(0.2126, 0.7152, 0.0722))
            selection_pdf = area * luminance / osh.maximum(push.area_light_weight, 1e-06)
            light_pdf = selection_pdf * hit.position_t.w * hit.position_t.w / osh.maximum(light_cosine * area, 1e-06)
            sampled_light_pdf = light_pdf * osh.f32(osh.maximum(push.secondary_area_light_samples, osh.u32(1)))
            if path.metadata.w & PATH_PREVIOUS_UNIFIED_NEE_BIT != osh.u32(0):
                sampled_light_pdf = light_pdf * unifiedAreaDomainProbability()
            emission_mis = powerHeuristic(pathPreviousPdf(path), sampled_light_pdf)
        path.radiance.rgb = path.radiance.rgb + path.throughput.rgb * material.emission_metallic.rgb * emission_mis
    next_bounce = pathBounce(path) + osh.u32(1)
    if next_bounce >= push.max_bounces:
        setPathBounce(path, next_bounce)
        path.metadata.w = path.metadata.w & ~PATH_ACTIVE_BIT
        paths[path_index] = path
        return
    rng = wave_surface_rng if wave_surface_response.custom_scattering > 0.5 else pathRng(path)
    next_direction = osh.vec3(0)
    bsdf_pdf = 0.0
    sampled_specular = 0.0
    transmission = material.attenuation_transmission.a
    medium_depth = osh.maximum(path.metadata.w >> osh.u32(8), osh.u32(1))
    if wave_surface_response.custom_scattering > 0.5:
        wave_event = osh.i32(wave_surface_response.event + 0.5)
        if wave_event == 0:
            path.metadata.w = path.metadata.w & ~PATH_ACTIVE_BIT
            setPathRng(path, rng)
            paths[path_index] = path
            return
        next_direction = osh.normalize(wave_surface_response.next_direction)
        bsdf_pdf = osh.maximum(wave_surface_response.pdf, 1e-06)
        path.throughput.rgb = path.throughput.rgb * (wave_surface_response.weight / bsdf_pdf)
        transmission = 1.0 if wave_event == 3 else 0.0
        if wave_event == 3:
            target_ior = osh.maximum(material.ior_distance.x, 1.0001)
            if entering and medium_depth < WAVE_MAX_MEDIUM_STACK_DEPTH:
                stacks[path_index].ior[medium_depth] = target_ior
                medium_depth = medium_depth + osh.u32(1)
            elif not entering and medium_depth > osh.u32(1):
                medium_depth = medium_depth - osh.u32(1)
    elif transmission > 0.001:
        current_ior = stacks[path_index].ior[medium_depth - osh.u32(1)]
        target_ior = osh.maximum(material.ior_distance.x, 1.0001) if entering else stacks[path_index].ior[medium_depth - osh.u32(2)] if medium_depth > osh.u32(1) else 1.0
        next_direction = osh.refract(incoming, normal, current_ior / target_ior)
        if osh.dot(next_direction, next_direction) < 0.01:
            next_direction = osh.reflect(incoming, normal)
        elif entering and medium_depth < WAVE_MAX_MEDIUM_STACK_DEPTH:
            stacks[path_index].ior[medium_depth] = target_ior
            medium_depth = medium_depth + 1
        elif not entering and medium_depth > osh.u32(1):
            medium_depth = medium_depth - 1
        path.throughput.rgb = path.throughput.rgb * (osh.mix(osh.vec3(1.0), material.base_roughness.rgb, 0.2) * transmission)
    else:
        nee_probability = osh.clamp(push.secondary_nee_probability, 1e-06, 1.0)
        sample_direct = selectSecondaryNee(nee_probability, path.metadata.x, path.metadata.y, next_bounce)
        if sample_direct:
            direct = samplePointLights(hit.position_t.xyz, normal, incoming, material)
            if push.unified_secondary_nee != osh.u32(0):
                direct = direct + sampleUnifiedSecondaryLight(hit.position_t.xyz, normal, incoming, material, rng)
            else:
                light_samples = osh.clamp(push.secondary_area_light_samples, osh.u32(1), osh.u32(16))
                area_direct = osh.vec3(0.0)
                sample_index = osh.u32(0)
                while sample_index < light_samples:
                    area_direct = area_direct + sampleAreaLight(hit.position_t.xyz, normal, incoming, material, rng, sample_index, light_samples)
                    sample_index = sample_index + 1
                direct = direct + area_direct / osh.f32(light_samples)
                environment_samples = osh.minimum(push.environment_samples, osh.u32(4))
                if environment_samples > osh.u32(0):
                    environment_direct = osh.vec3(0.0)
                    sample_index = osh.u32(0)
                    while sample_index < environment_samples:
                        environment_direct = environment_direct + sampleEnvironment(hit.position_t.xyz, normal, incoming, material, rng, environment_samples)
                        sample_index = sample_index + 1
                    direct = direct + environment_direct / osh.f32(environment_samples)
            path.radiance.rgb = path.radiance.rgb + path.throughput.rgb * direct / nee_probability
        bsdf_weight = osh.vec3(0)
        samplePbr(material, normal, incoming, rng, next_direction, bsdf_weight, bsdf_pdf, sampled_specular)
        path.throughput.rgb = path.throughput.rgb * bsdf_weight
        cone_spread = cone_spread + material.base_roughness.a * 0.25
    if (capture_secondary and pathBounce(path) == osh.u32(0)) and transmission <= 0.001:
        secondary_paths[path_index].primary_throughput = osh.vec4(path.throughput.rgb, 1.0 + sampled_specular)
        secondary_paths[path_index].primary_radiance = osh.vec4(path.radiance.rgb, pbrSpecularProbability(material))
        secondary_paths[path_index].normal_pdf.w = bsdf_pdf
        secondary_paths[path_index].primary_position = osh.vec4(hit.position_t.xyz, 1.0)
        secondary_paths[path_index].primary_geometry = osh.vec4(osh.uint_bits_to_float(hit.primitive_index), hit.barycentrics, 0.0)
    next_direction = osh.normalize(next_direction)
    setPathBounce(path, next_bounce)
    path.metadata.w = PATH_ACTIVE_BIT | medium_depth << osh.u32(8) | path.metadata.w & PATH_INDIRECT_CAPTURE_BIT
    if transmission <= 0.001:
        path.metadata.w = path.metadata.w | PATH_PREVIOUS_DIFFUSE_BIT
        if push.unified_secondary_nee != osh.u32(0):
            path.metadata.w = path.metadata.w | PATH_PREVIOUS_UNIFIED_NEE_BIT
        setPathPreviousPdf(path, bsdf_pdf)
    if (push.russian_roulette_start > osh.u32(0) and next_bounce >= push.russian_roulette_start) and transmission <= 0.001:
        survival = osh.clamp(osh.maximum(path.throughput.r, osh.maximum(path.throughput.g, path.throughput.b)), push.russian_roulette_min_survival, 0.95)
        if randomFloat(rng) >= survival:
            path.metadata.w = path.metadata.w & ~PATH_ACTIVE_BIT
            setPathRng(path, rng)
            paths[path_index] = path
            return
        path.throughput.rgb = path.throughput.rgb / survival
    setPathRng(path, rng)
    paths[path_index] = path
    output_index = reserveOutputIndex()
    if output_index >= output_queue.capacity:
        osh.atomic_add(output_queue.overflow, osh.u32(1))
        terminatePath(path_index)
        return
    output_queue.rays[output_index].origin_tmin = osh.vec4(hit.position_t.xyz + next_direction * 0.002, 0.001)
    output_queue.rays[output_index].direction_tmax = osh.vec4(next_direction, 1e+30)
    output_queue.rays[output_index].path_index = path_index
    output_queue.rays[output_index].padding_a = osh.float_bits_to_uint(cone_width)
    output_queue.rays[output_index].padding_b = osh.float_bits_to_uint(cone_spread)
    output_queue.rays[output_index].padding_c = osh.u32(0)

@osh.structure
class SecondaryConstants:
    max_bounces: osh.u32
    point_light_count: osh.u32
    area_light_count: osh.u32
    area_light_samples: osh.u32
    secondary_area_light_samples: osh.u32
    area_light_weight: osh.f32
    environment_samples: osh.u32
    russian_roulette_start: osh.u32
    russian_roulette_min_survival: osh.f32
    fused_intersection: osh.u32
    subgroup_enqueue: osh.u32
    secondary_nee_probability: osh.f32
    unified_secondary_nee: osh.u32
    indirect_secondary_capture: osh.u32

@osh.structure
class CameraData:
    origin: osh.vec4
    forward: osh.vec4
    right: osh.vec4
    up: osh.vec4

@osh.structure
class OutputQueue:
    count: osh.u32
    capacity: osh.u32
    overflow: osh.u32
    queue_padding: osh.u32

@osh.structure
class RayQueue:
    count: osh.u32
    capacity: osh.u32
    overflow: osh.u32
    queue_padding: osh.u32
    rays: osh.runtime_array(WaveRay)

@osh.structure
class HitQueue:
    count: osh.u32
    capacity: osh.u32
    overflow: osh.u32
    queue_padding: osh.u32
    hits: osh.runtime_array(WaveHit)
