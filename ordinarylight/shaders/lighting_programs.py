"""Typed transport helpers. Generated artifacts must be rebuilt from this source."""
import ordinaryshade as osh
from .transport_programs import MaterialData, AreaLightData, PointLightData
from .restir_programs import DirectLightReservoir

@osh.structure
class AreaLightCandidate:
    light_index: osh.u32
    barycentrics: osh.vec2
    target: osh.f32

@osh.function
def pbrSpecularProbability(material: MaterialData) -> osh.f32:
    f0 = osh.mix(osh.vec3(0.04), material.base_roughness.rgb, material.emission_metallic.a)
    return osh.clamp(osh.maximum(f0.r, osh.maximum(f0.g, f0.b)), 0.1, 0.9)

@osh.function
def evaluatePbr(material: MaterialData, normal: osh.vec3, view: osh.vec3, outgoing: osh.vec3) -> osh.vec3:
    n_dot_v = osh.maximum(osh.dot(normal, view), 0.0)
    n_dot_l = osh.maximum(osh.dot(normal, outgoing), 0.0)
    transportLastSpecularFraction = osh.vec3(0.0)
    if n_dot_v <= 0.0 or n_dot_l <= 0.0:
        return osh.vec3(0.0)
    half_vector = osh.normalize(view + outgoing)
    n_dot_h = osh.maximum(osh.dot(normal, half_vector), 0.0)
    v_dot_h = osh.maximum(osh.dot(view, half_vector), 0.0)
    metallic = material.emission_metallic.a
    f0 = osh.mix(osh.vec3(0.04), material.base_roughness.rgb, metallic)
    fresnel = pbrFresnel(f0, v_dot_h)
    anisotropy = osh.clamp(material.advanced0.w, -1.0, 1.0)
    tangent = osh.normalize(osh.cross(normal, osh.vec3(0.0, 0.0, 1.0)) if osh.absolute(normal.z) < 0.999 else osh.cross(normal, osh.vec3(0.0, 1.0, 0.0)))
    bitangent = osh.cross(normal, tangent)
    alpha = osh.maximum(material.base_roughness.a * material.base_roughness.a, 0.02)
    alpha_x = osh.maximum(alpha * (1.0 - 0.7 * anisotropy), 0.02)
    alpha_y = osh.maximum(alpha * (1.0 + 0.7 * anisotropy), 0.02)
    anisotropic_denominator = osh.dot(half_vector, tangent) * osh.dot(half_vector, tangent) / (alpha_x * alpha_x) + osh.dot(half_vector, bitangent) * osh.dot(half_vector, bitangent) / (alpha_y * alpha_y) + n_dot_h * n_dot_h
    distribution = 1.0 / osh.maximum(3.14159265359 * alpha_x * alpha_y * anisotropic_denominator * anisotropic_denominator, 1e-06)
    geometry = ggxSmithComponent(n_dot_v, material.base_roughness.a) * ggxSmithComponent(n_dot_l, material.base_roughness.a)
    specular = fresnel * distribution * geometry / osh.maximum(4.0 * n_dot_v * n_dot_l, 1e-06)
    diffuse = (osh.vec3(1.0) - fresnel) * (1.0 - metallic) * material.base_roughness.rgb / 3.14159265359
    subsurface = osh.clamp(material.advanced1.x, 0.0, 1.0)
    diffuse = osh.mix(diffuse, diffuse * material.subsurface_color.rgb, subsurface)
    sheen_weight = (1.0 - osh.clamp(material.advanced0.z, 0.0, 1.0)) * osh.power(1.0 - v_dot_h, 5.0)
    sheen = material.sheen_color.rgb * sheen_weight
    clearcoat = osh.clamp(material.advanced0.x, 0.0, 1.0)
    coat_roughness = osh.clamp(material.advanced0.y, 0.02, 1.0)
    coat_distribution = ggxDistribution(n_dot_h, coat_roughness)
    coat_geometry = ggxSmithComponent(n_dot_v, coat_roughness) * ggxSmithComponent(n_dot_l, coat_roughness)
    coat_fresnel = 0.04 + 0.96 * osh.power(1.0 - v_dot_h, 5.0)
    coat = osh.vec3(clearcoat * coat_distribution * coat_geometry * coat_fresnel / osh.maximum(4.0 * n_dot_v * n_dot_l, 1e-06))
    base_energy = 1.0 - clearcoat * coat_fresnel
    combined = (diffuse + specular + sheen) * base_energy + coat
    reflected = (specular + sheen) * base_energy + coat
    transportLastSpecularFraction = osh.clamp(reflected / osh.maximum(combined, osh.vec3(1e-30)), osh.vec3(0.0), osh.vec3(1.0))
    return combined

@osh.function
def pbrPdf(material: MaterialData, normal: osh.vec3, view: osh.vec3, outgoing: osh.vec3) -> osh.f32:
    n_dot_l = osh.maximum(osh.dot(normal, outgoing), 0.0)
    if n_dot_l <= 0.0:
        return 0.0
    half_vector = osh.normalize(view + outgoing)
    n_dot_h = osh.maximum(osh.dot(normal, half_vector), 0.0)
    v_dot_h = osh.maximum(osh.dot(view, half_vector), 1e-06)
    specular_pdf = ggxDistribution(n_dot_h, material.base_roughness.a) * n_dot_h / (4.0 * v_dot_h)
    diffuse_pdf = n_dot_l / 3.14159265359
    probability = osh.clamp(pbrSpecularProbability(material) + material.advanced0.x * 0.15, 0.1, 0.95)
    return osh.mix(diffuse_pdf, specular_pdf, probability)

@osh.function
def sampleGgxHalfVector(normal: osh.vec3, roughness: osh.f32, random_u: osh.f32, random_v: osh.f32) -> osh.vec3:
    alpha = osh.maximum(roughness * roughness, 0.0009)
    alpha_squared = alpha * alpha
    phi = 6.28318530718 * random_v
    cosine = osh.sqrt((1.0 - random_u) / osh.maximum(1.0 + (alpha_squared - 1.0) * random_u, 1e-06))
    sine = osh.sqrt(osh.maximum(0.0, 1.0 - cosine * cosine))
    tangent = osh.normalize(osh.cross(normal, osh.vec3(0.0, 0.0, 1.0)) if osh.absolute(normal.z) < 0.999 else osh.cross(normal, osh.vec3(0.0, 1.0, 0.0)))
    bitangent = osh.cross(normal, tangent)
    return osh.normalize(tangent * sine * osh.cosine(phi) + bitangent * sine * osh.sine(phi) + normal * cosine)

@osh.function
def samplePbr(material: MaterialData, normal: osh.vec3, incoming: osh.vec3, rng: osh.inout(osh.u32), outgoing: osh.out(osh.vec3), weight: osh.out(osh.vec3), pdf: osh.out(osh.f32), sampled_specular: osh.out(osh.f32)) -> osh.void:
    view = -incoming
    probability = ordinarylight_pbr_specular_probability(material.base_roughness.rgb, material.emission_metallic.a)
    specular = randomFloat(rng) < probability
    sampled_specular = 1.0 if specular else 0.0
    random_u = randomFloat(rng)
    random_v = randomFloat(rng)
    outgoing = ordinarylight_pbr_cosine_hemisphere(normal, random_u, random_v)
    if specular:
        half_vector = ordinarylight_pbr_sample_half_vector(normal, material.base_roughness.a, random_u, random_v)
        outgoing = ordinarylight_pbr_reflect(incoming, half_vector)
    if osh.dot(normal, outgoing) <= 0.0:
        outgoing = ordinarylight_pbr_cosine_hemisphere(normal, randomFloat(rng), randomFloat(rng))
    outgoing = osh.normalize(outgoing)
    pdf = osh.maximum(ordinarylight_pbr_pdf(material.base_roughness.rgb, material.base_roughness.a, material.emission_metallic.a, normal, view, outgoing), 1e-06)
    evaluated = ordinarylight_pbr_evaluate(material.base_roughness.rgb, material.base_roughness.a, material.emission_metallic.a, normal, view, outgoing)
    half_vector = osh.normalize(view + outgoing)
    view_half = osh.maximum(osh.dot(view, half_vector), 0.0)
    f0 = osh.mix(osh.vec3(0.04), material.base_roughness.rgb, material.emission_metallic.a)
    fresnel = f0 + (osh.vec3(1.0) - f0) * osh.power(1.0 - osh.clamp(view_half, 0.0, 1.0), 5.0)
    diffuse = (osh.vec3(1.0) - fresnel) * (1.0 - material.emission_metallic.a) * material.base_roughness.rgb / 3.14159265359
    transportLastSpecularFraction = osh.clamp((evaluated - diffuse) / osh.maximum(evaluated, osh.vec3(1e-30)), osh.vec3(0.0), osh.vec3(1.0))
    weight = ordinarylight_pbr_weight(evaluated, material.emission_metallic.a, material.texture_parameters.w, normal, view, outgoing, pdf)

@osh.function
def samplePointLights(hit: osh.vec3, normal: osh.vec3, incoming: osh.vec3, material: MaterialData) -> osh.vec3:
    direct = osh.vec3(0.0)
    transportPointSpecular = osh.vec3(0.0)
    index = osh.u32(0)
    while index < osh.minimum(OL_TRANSPORT_POINT_LIGHT_COUNT, osh.u32(64)):
        light = point_lights[index]
        light_type = osh.i32(light.position_type.w + 0.5)
        if light_type == 3:
            index = index + 1
            continue
        distance_squared = ordinarylight_analytic_light_distance_squared(light_type, light.position_type.xyz, hit)
        distance_to_light = 10000.0 if light_type == 1 else osh.sqrt(distance_squared)
        direction = ordinarylight_analytic_light_direction(light_type, light.position_type.xyz, light.direction_range.xyz, hit)
        if (light_type != 1 and light.direction_range.w > 0.0) and distance_to_light > light.direction_range.w:
            index = index + 1
            continue
        attenuation = ordinarylight_analytic_light_attenuation(light_type, distance_squared, direction, light.direction_range.xyz, light.spot_parameters.x, light.spot_parameters.y)
        if attenuation <= 0.0:
            index = index + 1
            continue
        cosine = ordinarylight_analytic_light_cosine(normal, direction)
        if cosine <= 0.0:
            index = index + 1
            continue
        shadow_origin = OL_TRANSPORT_RAY_ORIGIN(hit, normal)
        shadow_distance = ordinarylight_analytic_light_shadow_distance(light_type, distance_to_light)
        shadow = osh.ray_query()
        if osh.specialization('WAVE_WORK_COUNTERS'):
            profileWork(osh.u32(1), osh.u32(1))
        shadow.initialize(scene_tlas, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT, 1, shadow_origin, 0.001, direction, shadow_distance)
        while shadow.proceed():
            pass
        if shadow.intersection_type(True) != gl_RayQueryCommittedIntersectionNoneEXT:
            index = index + 1
            continue
        incident = ordinarylight_analytic_light_incident(light.color_intensity.rgb, light.color_intensity.a, attenuation, volumeShadowTransmittance(shadow_origin, direction, shadow_distance))
        contribution = ordinarylight_analytic_light_contribution(evaluatePbr(material, normal, -incoming, direction), incident, cosine)
        direct = direct + contribution
        transportPointSpecular = transportPointSpecular + contribution * transportLastSpecularFraction
        index = index + 1
    return direct

@osh.function
def sampleAreaLightTechnique(hit: osh.vec3, normal: osh.vec3, incoming: osh.vec3, material: MaterialData, rng: osh.inout(osh.u32), sample_index: osh.u32, sample_count: osh.u32, technique_probability: osh.f32) -> osh.vec3:
    if OL_TRANSPORT_AREA_LIGHT_COUNT == osh.u32(0):
        return osh.vec3(0.0)
    selection = (osh.f32(sample_index) + randomFloat(rng)) / osh.f32(sample_count)
    lower = osh.u32(0)
    upper = OL_TRANSPORT_AREA_LIGHT_COUNT - osh.u32(1)
    step = osh.u32(0)
    while step < osh.u32(32) and lower < upper:
        middle = lower + (upper - lower) / osh.u32(2)
        if selection <= area_lights[middle].distribution.x:
            upper = middle
        else:
            lower = middle + osh.u32(1)
        step = step + 1
    light = area_lights[lower]
    root_u = osh.sqrt(randomFloat(rng))
    v = randomFloat(rng)
    light_position = ordinarylight_area_light_position(light.a.xyz, light.b.xyz, light.c.xyz, root_u, v)
    offset = light_position - hit
    distance_squared = osh.dot(offset, offset)
    distance_to_light = osh.sqrt(distance_squared)
    direction = offset / osh.maximum(distance_to_light, 1e-06)
    surface_cosine = osh.maximum(osh.dot(normal, direction), 0.0)
    light_cosine = ordinarylight_area_light_cosine(light.a.xyz, light.b.xyz, light.c.xyz, direction, light.distribution.z > 0.5)
    if surface_cosine <= 0.0 or light_cosine <= 1e-06:
        return osh.vec3(0.0)
    shadow_origin = OL_TRANSPORT_RAY_ORIGIN(hit, normal)
    shadow_distance = osh.maximum(distance_to_light - 0.004, 0.001)
    shadow = osh.ray_query()
    if osh.specialization('WAVE_WORK_COUNTERS'):
        profileWork(osh.u32(1), osh.u32(1))
    shadow.initialize(scene_tlas, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT, 1, shadow_origin, 0.001, direction, shadow_distance)
    while shadow.proceed():
        pass
    if shadow.intersection_type(True) != gl_RayQueryCommittedIntersectionNoneEXT:
        return osh.vec3(0.0)
    effective_pdf = ordinarylight_area_light_pdf(light.distribution.y, distance_squared, light_cosine, light.emission_area.a, technique_probability)
    bsdf_pdf = pbrPdf(material, normal, -incoming, direction)
    mis = ordinarylight_area_light_mis(effective_pdf, osh.f32(sample_count), bsdf_pdf)
    transmittance = volumeShadowTransmittance(shadow_origin, direction, shadow_distance)
    return ordinarylight_area_light_contribution(evaluatePbr(material, normal, -incoming, direction), light.emission_area.rgb, surface_cosine, mis, transmittance, effective_pdf)

@osh.function
def sampleAreaLight(hit: osh.vec3, normal: osh.vec3, incoming: osh.vec3, material: MaterialData, rng: osh.inout(osh.u32), sample_index: osh.u32, sample_count: osh.u32) -> osh.vec3:
    return sampleAreaLightTechnique(hit, normal, incoming, material, rng, sample_index, sample_count, 1.0)

@osh.function
def unifiedAreaDomainProbability() -> osh.f32:
    return ordinarylight_unified_area_probability(OL_TRANSPORT_AREA_LIGHT_COUNT, OL_TRANSPORT_ENVIRONMENT_SAMPLES, OL_TRANSPORT_AREA_LIGHT_WEIGHT)

@osh.function
def evaluateAreaLightCandidateTechnique(candidate: AreaLightCandidate, hit: osh.vec3, normal: osh.vec3, incoming: osh.vec3, material: MaterialData, sample_count: osh.u32, technique_probability: osh.f32, direction: osh.out(osh.vec3), distance_to_light: osh.out(osh.f32)) -> osh.vec3:
    if candidate.light_index >= OL_TRANSPORT_AREA_LIGHT_COUNT:
        direction = normal
        distance_to_light = 0.0
        return osh.vec3(0.0)
    light = area_lights[candidate.light_index]
    light_position = ordinarylight_area_light_barycentric_position(light.a.xyz, light.b.xyz, light.c.xyz, candidate.barycentrics)
    offset = light_position - hit
    distance_squared = osh.dot(offset, offset)
    distance_to_light = osh.sqrt(distance_squared)
    direction = offset / osh.maximum(distance_to_light, 1e-06)
    surface_cosine = osh.maximum(osh.dot(normal, direction), 0.0)
    light_cosine = ordinarylight_area_light_cosine(light.a.xyz, light.b.xyz, light.c.xyz, direction, light.distribution.z > 0.5)
    if surface_cosine <= 0.0 or light_cosine <= 1e-06:
        return osh.vec3(0.0)
    effective_pdf = ordinarylight_area_light_pdf(light.distribution.y, distance_squared, light_cosine, light.emission_area.a, technique_probability)
    bsdf_pdf = pbrPdf(material, normal, -incoming, direction)
    mis = ordinarylight_area_light_mis(effective_pdf, osh.f32(sample_count), bsdf_pdf)
    return ordinarylight_area_light_contribution(evaluatePbr(material, normal, -incoming, direction), light.emission_area.rgb, surface_cosine, mis, 1.0, effective_pdf)

@osh.function
def evaluateAreaLightCandidate(candidate: AreaLightCandidate, hit: osh.vec3, normal: osh.vec3, incoming: osh.vec3, material: MaterialData, sample_count: osh.u32, direction: osh.out(osh.vec3), distance_to_light: osh.out(osh.f32)) -> osh.vec3:
    return evaluateAreaLightCandidateTechnique(candidate, hit, normal, incoming, material, sample_count, 1.0, direction, distance_to_light)

@osh.function
def generateAreaLightCandidate(hit: osh.vec3, normal: osh.vec3, incoming: osh.vec3, material: MaterialData, rng: osh.inout(osh.u32), sample_index: osh.u32, sample_count: osh.u32) -> AreaLightCandidate:
    candidate = AreaLightCandidate(osh.u32(0), osh.vec2(0), osh.f32(0.0))
    candidate.light_index = osh.u32(4294967295)
    candidate.barycentrics = osh.vec2(0.0)
    candidate.target = 0.0
    if OL_TRANSPORT_AREA_LIGHT_COUNT == osh.u32(0):
        return candidate
    selection = (osh.f32(sample_index) + randomFloat(rng)) / osh.f32(sample_count)
    lower = osh.u32(0)
    upper = OL_TRANSPORT_AREA_LIGHT_COUNT - osh.u32(1)
    step = osh.u32(0)
    while step < osh.u32(32) and lower < upper:
        middle = lower + (upper - lower) / osh.u32(2)
        if selection <= area_lights[middle].distribution.x:
            upper = middle
        else:
            lower = middle + osh.u32(1)
        step = step + 1
    root_u = osh.sqrt(randomFloat(rng))
    v = randomFloat(rng)
    candidate.light_index = lower
    candidate.barycentrics = osh.vec2(root_u * (1.0 - v), root_u * v)
    unused_direction = osh.vec3(0)
    unused_distance = osh.f32(0.0)
    contribution = evaluateAreaLightCandidate(candidate, hit, normal, incoming, material, sample_count, unused_direction, unused_distance)
    candidate.target = ordinarylight_light_candidate_target(contribution)
    return candidate

@osh.function
def areaLightCandidateVisibility(hit: osh.vec3, normal: osh.vec3, direction: osh.vec3, distance_to_light: osh.f32) -> osh.f32:
    if distance_to_light <= 0.0:
        return 0.0
    shadow_origin = OL_TRANSPORT_RAY_ORIGIN(hit, normal)
    shadow_distance = osh.maximum(distance_to_light - 0.004, 0.001)
    shadow = osh.ray_query()
    if osh.specialization('WAVE_WORK_COUNTERS'):
        profileWork(osh.u32(1), osh.u32(1))
    shadow.initialize(scene_tlas, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT, 1, shadow_origin, 0.001, direction, shadow_distance)
    while shadow.proceed():
        pass
    if shadow.intersection_type(True) != gl_RayQueryCommittedIntersectionNoneEXT:
        return 0.0
    return volumeShadowTransmittance(shadow_origin, direction, shadow_distance)

@osh.function
def environmentColor(direction: osh.vec3) -> osh.vec3:
    index = osh.u32(0)
    while index < osh.minimum(OL_TRANSPORT_POINT_LIGHT_COUNT, osh.u32(64)):
        light = point_lights[index]
        if osh.i32(light.position_type.w + 0.5) != 3:
            index = index + 1
            continue
        radiance = osh.vec3(1.0)
        texture_index = -1 if light.spot_parameters.x < 0.0 else osh.i32(light.spot_parameters.x + 0.5)
        if texture_index >= 0:
            uv = ordinarylight_environment_uv(direction, light.spot_parameters.y)
            encoded = sampleSceneTexture(texture_index, uv).rgb
            radiance = ordinarylight_environment_radiance(encoded, light.spot_parameters.z, osh.vec3(1.0), 1.0, True)
        return ordinarylight_environment_radiance(radiance, 0.0, light.color_intensity.rgb, light.color_intensity.a, False)
        index = index + 1
    return ordinarylight_environment_analytic(direction)

@osh.function
def sampleEnvironmentTechnique(hit: osh.vec3, normal: osh.vec3, incoming: osh.vec3, material: MaterialData, rng: osh.inout(osh.u32), sample_count: osh.u32, technique_probability: osh.f32) -> osh.vec3:
    direction = ordinarylight_pbr_cosine_hemisphere(normal, randomFloat(rng), randomFloat(rng))
    shadow_origin = OL_TRANSPORT_RAY_ORIGIN(hit, normal)
    shadow = osh.ray_query()
    if osh.specialization('WAVE_WORK_COUNTERS'):
        profileWork(osh.u32(1), osh.u32(1))
    shadow.initialize(scene_tlas, gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT, 1, shadow_origin, 0.001, direction, 1e+30)
    while shadow.proceed():
        pass
    if shadow.intersection_type(True) != gl_RayQueryCommittedIntersectionNoneEXT:
        return osh.vec3(0.0)
    cosine = ordinarylight_analytic_light_cosine(normal, direction)
    pdf = cosine / 3.14159265359
    effective_pdf = ordinarylight_environment_effective_pdf(cosine, technique_probability)
    mis = ordinarylight_environment_mis(cosine, effective_pdf, osh.f32(sample_count))
    transmittance = volumeShadowTransmittance(shadow_origin, direction, 1e+30)
    return ordinarylight_environment_contribution(evaluatePbr(material, normal, -incoming, direction), environmentColor(direction), cosine, mis, transmittance, effective_pdf, material.texture_parameters.w, material.emission_metallic.a)

@osh.function
def sampleEnvironment(hit: osh.vec3, normal: osh.vec3, incoming: osh.vec3, material: MaterialData, rng: osh.inout(osh.u32), sample_count: osh.u32) -> osh.vec3:
    return sampleEnvironmentTechnique(hit, normal, incoming, material, rng, sample_count, 1.0)

@osh.function
def sampleUnifiedSecondaryLight(hit: osh.vec3, normal: osh.vec3, incoming: osh.vec3, material: MaterialData, rng: osh.inout(osh.u32)) -> osh.vec3:
    area_enabled = OL_TRANSPORT_AREA_LIGHT_COUNT > osh.u32(0)
    environment_enabled = OL_TRANSPORT_ENVIRONMENT_SAMPLES > osh.u32(0)
    if not area_enabled and (not environment_enabled):
        return osh.vec3(0.0)
    area_probability = ordinarylight_unified_secondary_area_probability(area_enabled, environment_enabled, OL_TRANSPORT_AREA_LIGHT_WEIGHT, OL_TRANSPORT_SECONDARY_AREA_LIGHT_SAMPLES, OL_TRANSPORT_ENVIRONMENT_SAMPLES)
    if randomFloat(rng) < area_probability:
        return sampleAreaLightTechnique(hit, normal, incoming, material, rng, osh.u32(0), osh.u32(1), area_probability)
    return sampleEnvironmentTechnique(hit, normal, incoming, material, rng, osh.u32(1), 1.0 - area_probability)

@osh.function
def encodeEnvironmentCandidateDirection(direction: osh.vec3) -> osh.vec2:
    return ordinarylight_environment_encode_direction(direction)

@osh.function
def decodeEnvironmentCandidateDirection(encoded: osh.vec2) -> osh.vec3:
    return ordinarylight_environment_decode_direction(encoded)

@osh.function
def evaluateEnvironmentCandidate(encoded_direction: osh.vec2, hit: osh.vec3, normal: osh.vec3, incoming: osh.vec3, material: MaterialData, sample_count: osh.u32, technique_probability: osh.f32, direction: osh.out(osh.vec3), distance_to_light: osh.out(osh.f32)) -> osh.vec3:
    direction = decodeEnvironmentCandidateDirection(encoded_direction)
    distance_to_light = 1e+30
    cosine = osh.maximum(osh.dot(normal, direction), 0.0)
    if cosine <= 0.0:
        return osh.vec3(0.0)
    pdf = cosine / 3.14159265359
    effective_pdf = ordinarylight_environment_effective_pdf(cosine, technique_probability)
    mis = ordinarylight_environment_mis(cosine, effective_pdf, osh.f32(sample_count))
    return ordinarylight_environment_contribution(evaluatePbr(material, normal, -incoming, direction), environmentColor(direction), cosine, mis, 1.0, effective_pdf, material.texture_parameters.w, material.emission_metallic.a)

@osh.function
def generateUnifiedPrimaryCandidate(hit: osh.vec3, normal: osh.vec3, incoming: osh.vec3, material: MaterialData, rng: osh.inout(osh.u32), sample_index: osh.u32, sample_count: osh.u32) -> AreaLightCandidate:
    area_probability = unifiedAreaDomainProbability()
    if OL_TRANSPORT_ENVIRONMENT_SAMPLES == osh.u32(0) or randomFloat(rng) < area_probability:
        candidate = generateAreaLightCandidate(hit, normal, incoming, material, rng, sample_index, sample_count)
        unused_direction = osh.vec3(0)
        unused_distance = osh.f32(0.0)
        contribution = evaluateAreaLightCandidateTechnique(candidate, hit, normal, incoming, material, sample_count, area_probability, unused_direction, unused_distance)
        candidate.target = ordinarylight_light_candidate_target(contribution)
        return candidate
    candidate = AreaLightCandidate(osh.u32(0), osh.vec2(0), osh.f32(0.0))
    direction = cosineHemisphere(normal, randomFloat(rng), randomFloat(rng))
    candidate.light_index = ENVIRONMENT_LIGHT_CANDIDATE_INDEX
    candidate.barycentrics = encodeEnvironmentCandidateDirection(direction)
    unused_distance = osh.f32(0.0)
    contribution = evaluateEnvironmentCandidate(candidate.barycentrics, hit, normal, incoming, material, sample_count, 1.0 - area_probability, direction, unused_distance)
    candidate.target = ordinarylight_light_candidate_target(contribution)
    return candidate

@osh.function
def evaluateUnifiedPrimaryCandidate(candidate: AreaLightCandidate, hit: osh.vec3, normal: osh.vec3, incoming: osh.vec3, material: MaterialData, sample_count: osh.u32, direction: osh.out(osh.vec3), distance_to_light: osh.out(osh.f32)) -> osh.vec3:
    area_probability = unifiedAreaDomainProbability()
    if candidate.light_index == ENVIRONMENT_LIGHT_CANDIDATE_INDEX:
        return evaluateEnvironmentCandidate(candidate.barycentrics, hit, normal, incoming, material, sample_count, 1.0 - area_probability, direction, distance_to_light)
    return evaluateAreaLightCandidateTechnique(candidate, hit, normal, incoming, material, sample_count, area_probability, direction, distance_to_light)

@osh.function
def ordinarylightDefaultRayOrigin(hit: osh.vec3, normal: osh.vec3) -> osh.vec3:
    return hit + normal * 0.002

@osh.external
def OL_TRANSPORT_RAY_ORIGIN(hit: osh.vec3, normal: osh.vec3) -> osh.vec3:
    pass

@osh.external
def volumeShadowTransmittance(origin: osh.vec3, direction: osh.vec3, maximum_distance: osh.f32) -> osh.f32:
    pass
