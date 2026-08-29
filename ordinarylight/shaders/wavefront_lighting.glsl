// Shared wavefront BSDF sampling and next-event estimation.
// The including shader defines MaterialData, light buffers, scene_tlas, and
// a push block exposing the common lighting fields.
#include "wavefront_restir.glsl"

float randomFloat(inout uint state)
{
    state = state * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    word = (word >> 22u) ^ word;
    return float(word) * (1.0 / 4294967296.0);
}

uint secondaryNeeHash(uint value)
{
#if WAVE_ORDINARYSHADE_UNIFIED_NEE
    return ordinarylight_secondary_nee_hash(value);
#else
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    return value ^ (value >> 16u);
#endif
}

bool selectSecondaryNee(
    float probability, uint pixel_index, uint frame_sample, uint bounce)
{
#if WAVE_ORDINARYSHADE_UNIFIED_NEE
    return ordinarylight_secondary_nee_select(
        probability, pixel_index, frame_sample, bounce);
#else
    if (probability >= 0.999999)
        return true;

    // A digitally shifted base-2 sequence alternates exactly at p=0.5 and
    // remains well stratified at other probabilities.  Pixel, bounce, and
    // sample identity scramble neighboring paths without consuming BSDF RNG.
    uint frame_index = frame_sample >> 8u;
    uint sample_index = frame_sample & 255u;
    uint scramble = secondaryNeeHash(
        pixel_index ^ secondaryNeeHash(bounce + 1u)
        ^ secondaryNeeHash(sample_index + 1u));
    uint sequence = bitfieldReverse(frame_index) ^ scramble;
    float selector = (float(sequence) + 0.5) * (1.0 / 4294967296.0);
    return selector < probability;
#endif
}

vec3 cosineHemisphere(vec3 normal, float random_u, float random_v)
{
    float radius = sqrt(random_u);
    float phi = 6.28318530718 * random_v;
    vec3 tangent = normalize(abs(normal.z) < 0.999
        ? cross(normal, vec3(0.0, 0.0, 1.0))
        : cross(normal, vec3(0.0, 1.0, 0.0)));
    vec3 bitangent = cross(normal, tangent);
    return normalize(tangent * radius * cos(phi) + bitangent * radius * sin(phi)
                     + normal * sqrt(max(0.0, 1.0 - random_u)));
}

float powerHeuristic(float first_pdf, float second_pdf)
{
    float a = first_pdf * first_pdf;
    float b = second_pdf * second_pdf;
    return a / max(a + b, 0.000001);
}

vec3 pbrFresnel(vec3 f0, float cosine)
{
    return f0 + (vec3(1.0) - f0) * pow(1.0 - clamp(cosine, 0.0, 1.0), 5.0);
}

float ggxDistribution(float n_dot_h, float roughness)
{
    float alpha = max(roughness * roughness, 0.0009);
    float alpha_squared = alpha * alpha;
    float denominator = n_dot_h * n_dot_h * (alpha_squared - 1.0) + 1.0;
    return alpha_squared /
        max(3.14159265359 * denominator * denominator, 0.000001);
}

float ggxSmithComponent(float n_dot_direction, float roughness)
{
    float alpha = max(roughness * roughness, 0.0009);
    float alpha_squared = alpha * alpha;
    return 2.0 * n_dot_direction / max(
        n_dot_direction + sqrt(alpha_squared
            + (1.0 - alpha_squared) * n_dot_direction * n_dot_direction),
        0.000001);
}

float pbrSpecularProbability(MaterialData material)
{
    vec3 f0 = mix(vec3(0.04), material.base_roughness.rgb,
                  material.emission_metallic.a);
    return clamp(max(f0.r, max(f0.g, f0.b)), 0.1, 0.9);
}

vec3 evaluatePbr(
    MaterialData material, vec3 normal, vec3 view, vec3 outgoing)
{
    float n_dot_v = max(dot(normal, view), 0.0);
    float n_dot_l = max(dot(normal, outgoing), 0.0);
    if (n_dot_v <= 0.0 || n_dot_l <= 0.0)
        return vec3(0.0);
    vec3 half_vector = normalize(view + outgoing);
    float n_dot_h = max(dot(normal, half_vector), 0.0);
    float v_dot_h = max(dot(view, half_vector), 0.0);
    float metallic = material.emission_metallic.a;
    vec3 f0 = mix(vec3(0.04), material.base_roughness.rgb, metallic);
    vec3 fresnel = pbrFresnel(f0, v_dot_h);
    float distribution = ggxDistribution(n_dot_h, material.base_roughness.a);
    float geometry = ggxSmithComponent(n_dot_v, material.base_roughness.a)
        * ggxSmithComponent(n_dot_l, material.base_roughness.a);
    vec3 specular = fresnel * distribution * geometry /
        max(4.0 * n_dot_v * n_dot_l, 0.000001);
    vec3 diffuse = (vec3(1.0) - fresnel) * (1.0 - metallic)
        * material.base_roughness.rgb / 3.14159265359;
    return diffuse + specular;
}

float pbrPdf(MaterialData material, vec3 normal, vec3 view, vec3 outgoing)
{
    float n_dot_l = max(dot(normal, outgoing), 0.0);
    if (n_dot_l <= 0.0)
        return 0.0;
    vec3 half_vector = normalize(view + outgoing);
    float n_dot_h = max(dot(normal, half_vector), 0.0);
    float v_dot_h = max(dot(view, half_vector), 0.000001);
    float specular_pdf = ggxDistribution(
        n_dot_h, material.base_roughness.a) * n_dot_h / (4.0 * v_dot_h);
    float diffuse_pdf = n_dot_l / 3.14159265359;
    float probability = pbrSpecularProbability(material);
    return mix(diffuse_pdf, specular_pdf, probability);
}

vec3 sampleGgxHalfVector(
    vec3 normal, float roughness, float random_u, float random_v)
{
    float alpha = max(roughness * roughness, 0.0009);
    float alpha_squared = alpha * alpha;
    float phi = 6.28318530718 * random_v;
    float cosine = sqrt((1.0 - random_u) /
        max(1.0 + (alpha_squared - 1.0) * random_u, 0.000001));
    float sine = sqrt(max(0.0, 1.0 - cosine * cosine));
    vec3 tangent = normalize(abs(normal.z) < 0.999
        ? cross(normal, vec3(0.0, 0.0, 1.0))
        : cross(normal, vec3(0.0, 1.0, 0.0)));
    vec3 bitangent = cross(normal, tangent);
    return normalize(tangent * sine * cos(phi)
        + bitangent * sine * sin(phi) + normal * cosine);
}

void samplePbr(
    MaterialData material, vec3 normal, vec3 incoming, inout uint rng,
    out vec3 outgoing, out vec3 weight, out float pdf)
{
#if WAVE_ORDINARYSHADE_PBR
    vec3 view = -incoming;
    float probability = ordinarylight_pbr_specular_probability(
        material.base_roughness.rgb, material.emission_metallic.a);
    bool specular = randomFloat(rng) < probability;
    float random_u = randomFloat(rng);
    float random_v = randomFloat(rng);
    outgoing = ordinarylight_pbr_cosine_hemisphere(
        normal, random_u, random_v);
    if (specular) {
        vec3 half_vector = ordinarylight_pbr_sample_half_vector(
            normal, material.base_roughness.a, random_u, random_v);
        outgoing = ordinarylight_pbr_reflect(incoming, half_vector);
    }
    if (dot(normal, outgoing) <= 0.0) {
        outgoing = ordinarylight_pbr_cosine_hemisphere(
            normal, randomFloat(rng), randomFloat(rng));
    }
    outgoing = normalize(outgoing);
    pdf = max(ordinarylight_pbr_pdf(
        material.base_roughness.rgb, material.base_roughness.a,
        material.emission_metallic.a, normal, view, outgoing), 0.000001);
    vec3 evaluated = ordinarylight_pbr_evaluate(
        material.base_roughness.rgb, material.base_roughness.a,
        material.emission_metallic.a, normal, view, outgoing);
    weight = ordinarylight_pbr_weight(
        evaluated, material.emission_metallic.a,
        material.texture_parameters.w, normal, view, outgoing, pdf);
#else
    vec3 view = -incoming;
    float probability = pbrSpecularProbability(material);
    bool specular = randomFloat(rng) < probability;
    if (specular) {
        vec3 half_vector = sampleGgxHalfVector(
            normal, material.base_roughness.a,
            randomFloat(rng), randomFloat(rng));
        outgoing = reflect(incoming, half_vector);
    } else {
        outgoing = cosineHemisphere(
            normal, randomFloat(rng), randomFloat(rng));
    }
    if (dot(normal, outgoing) <= 0.0) {
        outgoing = cosineHemisphere(
            normal, randomFloat(rng), randomFloat(rng));
    }
    outgoing = normalize(outgoing);
    pdf = max(pbrPdf(material, normal, view, outgoing), 0.000001);
    weight = evaluatePbr(material, normal, view, outgoing)
        * max(dot(normal, outgoing), 0.0) / pdf;
    weight *= mix(
        material.texture_parameters.w, 1.0,
        material.emission_metallic.a);
#endif
}

vec3 samplePointLights(
    vec3 hit, vec3 normal, vec3 incoming, MaterialData material)
{
    vec3 direct = vec3(0.0);
    for (uint index = 0u; index < min(push.point_light_count, 64u); ++index) {
        PointLightData light = point_lights[index];
        int light_type = int(light.position_type.w + 0.5);
        if (light_type == 3) continue;
#if WAVE_ORDINARYSHADE_ANALYTIC_LIGHTS
        float distance_squared = ordinarylight_analytic_light_distance_squared(
            light_type, light.position_type.xyz, hit);
        float distance_to_light = light_type == 1
            ? 10000.0 : sqrt(distance_squared);
        vec3 direction = ordinarylight_analytic_light_direction(
            light_type, light.position_type.xyz,
            light.direction_range.xyz, hit);
        if (light_type != 1 && light.direction_range.w > 0.0 &&
                distance_to_light > light.direction_range.w) continue;
        float attenuation = ordinarylight_analytic_light_attenuation(
            light_type, distance_squared, direction,
            light.direction_range.xyz, light.spot_parameters.x,
            light.spot_parameters.y);
        if (attenuation <= 0.0) continue;
#else
        vec3 direction;
        float distance_squared = 1.0;
        float distance_to_light = 10000.0;
        float attenuation = 1.0;
        if (light_type == 1) {
            direction = -normalize(light.direction_range.xyz);
        } else {
            vec3 offset = light.position_type.xyz - hit;
            distance_squared = max(dot(offset, offset), 0.000001);
            distance_to_light = sqrt(distance_squared);
            direction = offset / distance_to_light;
            if (light.direction_range.w > 0.0 &&
                    distance_to_light > light.direction_range.w) continue;
            attenuation = 1.0 / distance_squared;
            if (light_type == 2) {
                float cone = dot(normalize(light.direction_range.xyz), -direction);
                float spot = smoothstep(
                    light.spot_parameters.y, light.spot_parameters.x, cone);
                if (spot <= 0.0) continue;
                attenuation *= spot;
            }
        }
#endif
#if WAVE_ORDINARYSHADE_ANALYTIC_LIGHTS
        float cosine = ordinarylight_analytic_light_cosine(
            normal, direction);
#else
        float cosine = max(dot(normal, direction), 0.0);
#endif
        if (cosine <= 0.0) continue;
        vec3 shadow_origin = hit + normal * 0.002;
#if WAVE_ORDINARYSHADE_ANALYTIC_LIGHTS
        float shadow_distance = ordinarylight_analytic_light_shadow_distance(
            light_type, distance_to_light);
#else
        float shadow_distance = light_type == 1
            ? 10000.0 : max(distance_to_light - 0.004, 0.001);
#endif
        rayQueryEXT shadow;
#if WAVE_WORK_COUNTERS
        profileWork(1u, 1u);
#endif
        rayQueryInitializeEXT(
            shadow, scene_tlas,
            gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT,
            0x01, shadow_origin, 0.001, direction, shadow_distance);
        while (rayQueryProceedEXT(shadow)) {}
        if (rayQueryGetIntersectionTypeEXT(shadow, true) !=
                gl_RayQueryCommittedIntersectionNoneEXT) continue;
#if WAVE_ORDINARYSHADE_ANALYTIC_LIGHTS
        vec3 incident = ordinarylight_analytic_light_incident(
            light.color_intensity.rgb, light.color_intensity.a, attenuation,
            volumeShadowTransmittance(
                shadow_origin, direction, shadow_distance));
        direct += ordinarylight_analytic_light_contribution(
            evaluatePbr(material, normal, -incoming, direction),
            incident, cosine);
#else
        vec3 incident = light.color_intensity.rgb * light.color_intensity.a
            * attenuation;
        incident *= volumeShadowTransmittance(
            shadow_origin, direction, shadow_distance);
        direct += evaluatePbr(material, normal, -incoming, direction)
            * incident * cosine;
#endif
    }
    return direct;
}

vec3 sampleAreaLightTechnique(
    vec3 hit, vec3 normal, vec3 incoming, MaterialData material, inout uint rng,
    uint sample_index, uint sample_count, float technique_probability)
{
    if (push.area_light_count == 0u) return vec3(0.0);
    float selection = (float(sample_index) + randomFloat(rng)) /
        float(sample_count);
    uint lower = 0u, upper = push.area_light_count - 1u;
    for (uint step = 0u; step < 32u && lower < upper; ++step) {
        uint middle = lower + (upper - lower) / 2u;
        if (selection <= area_lights[middle].distribution.x) upper = middle;
        else lower = middle + 1u;
    }
    AreaLightData light = area_lights[lower];
    float root_u = sqrt(randomFloat(rng));
    float v = randomFloat(rng);
#if WAVE_ORDINARYSHADE_AREA_LIGHTS
    vec3 light_position = ordinarylight_area_light_position(
        light.a.xyz, light.b.xyz, light.c.xyz, root_u, v);
#else
    vec3 light_position = (1.0 - root_u) * light.a.xyz
        + root_u * (1.0 - v) * light.b.xyz + root_u * v * light.c.xyz;
#endif
    vec3 offset = light_position - hit;
    float distance_squared = dot(offset, offset);
    float distance_to_light = sqrt(distance_squared);
    vec3 direction = offset / max(distance_to_light, 0.000001);
    float surface_cosine = max(dot(normal, direction), 0.0);
#if WAVE_ORDINARYSHADE_AREA_LIGHTS
    float light_cosine = ordinarylight_area_light_cosine(
        light.a.xyz, light.b.xyz, light.c.xyz, direction,
        light.distribution.z > 0.5);
#else
    vec3 light_normal = normalize(cross(
        light.b.xyz - light.a.xyz, light.c.xyz - light.a.xyz));
    float raw_light_cosine = dot(light_normal, -direction);
    float light_cosine = light.distribution.z > 0.5
        ? abs(raw_light_cosine) : max(raw_light_cosine, 0.0);
#endif
    if (surface_cosine <= 0.0 || light_cosine <= 0.000001)
        return vec3(0.0);
    vec3 shadow_origin = hit + normal * 0.002;
    float shadow_distance = max(distance_to_light - 0.004, 0.001);
    rayQueryEXT shadow;
#if WAVE_WORK_COUNTERS
    profileWork(1u, 1u);
#endif
    rayQueryInitializeEXT(
        shadow, scene_tlas,
        gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT,
        0x01, shadow_origin, 0.001, direction, shadow_distance);
    while (rayQueryProceedEXT(shadow)) {}
    if (rayQueryGetIntersectionTypeEXT(shadow, true) !=
            gl_RayQueryCommittedIntersectionNoneEXT) return vec3(0.0);
#if WAVE_ORDINARYSHADE_AREA_LIGHTS
    float effective_pdf = ordinarylight_area_light_pdf(
        light.distribution.y, distance_squared, light_cosine,
        light.emission_area.a, technique_probability);
    float bsdf_pdf = pbrPdf(material, normal, -incoming, direction);
    float mis = ordinarylight_area_light_mis(
        effective_pdf, float(sample_count), bsdf_pdf);
#else
    float light_pdf = light.distribution.y * distance_squared /
        max(light_cosine * light.emission_area.a, 0.000001);
    float bsdf_pdf = pbrPdf(material, normal, -incoming, direction);
    float effective_pdf = light_pdf * max(technique_probability, 0.000001);
    float mis = powerHeuristic(
        effective_pdf * float(sample_count), bsdf_pdf);
#endif
    float transmittance = volumeShadowTransmittance(
        shadow_origin, direction, shadow_distance);
#if WAVE_ORDINARYSHADE_AREA_LIGHTS
    return ordinarylight_area_light_contribution(
        evaluatePbr(material, normal, -incoming, direction),
        light.emission_area.rgb, surface_cosine, mis, transmittance,
        effective_pdf);
#else
    return evaluatePbr(material, normal, -incoming, direction)
        * light.emission_area.rgb * surface_cosine * mis
        * transmittance / max(effective_pdf, 0.000001);
#endif
}

vec3 sampleAreaLight(
    vec3 hit, vec3 normal, vec3 incoming, MaterialData material, inout uint rng,
    uint sample_index, uint sample_count)
{
    return sampleAreaLightTechnique(
        hit, normal, incoming, material, rng,
        sample_index, sample_count, 1.0);
}

struct AreaLightCandidate {
    uint light_index;
    vec2 barycentrics;
    float target;
};

const uint ENVIRONMENT_LIGHT_CANDIDATE_INDEX = 0x01fffffeu;

float unifiedAreaDomainProbability()
{
#if WAVE_ORDINARYSHADE_UNIFIED_NEE
    return ordinarylight_unified_area_probability(
        push.area_light_count, push.environment_samples,
        push.area_light_weight);
#else
    if (push.area_light_count == 0u)
        return 0.0;
    if (push.environment_samples == 0u)
        return 1.0;
    float area_weight = sqrt(max(push.area_light_weight, 0.000001));
    return area_weight / (area_weight + 1.0);
#endif
}

vec3 evaluateAreaLightCandidateTechnique(
    AreaLightCandidate candidate, vec3 hit, vec3 normal, vec3 incoming,
    MaterialData material, uint sample_count, float technique_probability,
    out vec3 direction, out float distance_to_light)
{
    if (candidate.light_index >= push.area_light_count) {
        direction = normal;
        distance_to_light = 0.0;
        return vec3(0.0);
    }
    AreaLightData light = area_lights[candidate.light_index];
#if WAVE_ORDINARYSHADE_AREA_LIGHTS
    vec3 light_position = ordinarylight_area_light_barycentric_position(
        light.a.xyz, light.b.xyz, light.c.xyz, candidate.barycentrics);
#else
    float a_weight = max(
        1.0 - candidate.barycentrics.x - candidate.barycentrics.y, 0.0);
    vec3 light_position = a_weight * light.a.xyz
        + candidate.barycentrics.x * light.b.xyz
        + candidate.barycentrics.y * light.c.xyz;
#endif
    vec3 offset = light_position - hit;
    float distance_squared = dot(offset, offset);
    distance_to_light = sqrt(distance_squared);
    direction = offset / max(distance_to_light, 0.000001);
    float surface_cosine = max(dot(normal, direction), 0.0);
#if WAVE_ORDINARYSHADE_AREA_LIGHTS
    float light_cosine = ordinarylight_area_light_cosine(
        light.a.xyz, light.b.xyz, light.c.xyz, direction,
        light.distribution.z > 0.5);
#else
    vec3 light_normal = normalize(cross(
        light.b.xyz - light.a.xyz, light.c.xyz - light.a.xyz));
    float raw_light_cosine = dot(light_normal, -direction);
    float light_cosine = light.distribution.z > 0.5
        ? abs(raw_light_cosine) : max(raw_light_cosine, 0.0);
#endif
    if (surface_cosine <= 0.0 || light_cosine <= 0.000001)
        return vec3(0.0);
#if WAVE_ORDINARYSHADE_AREA_LIGHTS
    float effective_pdf = ordinarylight_area_light_pdf(
        light.distribution.y, distance_squared, light_cosine,
        light.emission_area.a, technique_probability);
    float bsdf_pdf = pbrPdf(material, normal, -incoming, direction);
    float mis = ordinarylight_area_light_mis(
        effective_pdf, float(sample_count), bsdf_pdf);
    return ordinarylight_area_light_contribution(
        evaluatePbr(material, normal, -incoming, direction),
        light.emission_area.rgb, surface_cosine, mis, 1.0, effective_pdf);
#else
    float light_pdf = light.distribution.y * distance_squared /
        max(light_cosine * light.emission_area.a, 0.000001);
    float bsdf_pdf = pbrPdf(material, normal, -incoming, direction);
    float effective_pdf = light_pdf * max(technique_probability, 0.000001);
    float mis = powerHeuristic(
        effective_pdf * float(sample_count), bsdf_pdf);
    return evaluatePbr(material, normal, -incoming, direction)
        * light.emission_area.rgb * surface_cosine * mis
        / max(effective_pdf, 0.000001);
#endif
}

vec3 evaluateAreaLightCandidate(
    AreaLightCandidate candidate, vec3 hit, vec3 normal, vec3 incoming,
    MaterialData material, uint sample_count,
    out vec3 direction, out float distance_to_light)
{
    return evaluateAreaLightCandidateTechnique(
        candidate, hit, normal, incoming, material, sample_count, 1.0,
        direction, distance_to_light);
}

AreaLightCandidate generateAreaLightCandidate(
    vec3 hit, vec3 normal, vec3 incoming, MaterialData material,
    inout uint rng, uint sample_index, uint sample_count)
{
    AreaLightCandidate candidate;
    candidate.light_index = 0xffffffffu;
    candidate.barycentrics = vec2(0.0);
    candidate.target = 0.0;
    if (push.area_light_count == 0u)
        return candidate;
    float selection = (float(sample_index) + randomFloat(rng))
        / float(sample_count);
    uint lower = 0u, upper = push.area_light_count - 1u;
    for (uint step = 0u; step < 32u && lower < upper; ++step) {
        uint middle = lower + (upper - lower) / 2u;
        if (selection <= area_lights[middle].distribution.x) upper = middle;
        else lower = middle + 1u;
    }
    float root_u = sqrt(randomFloat(rng));
    float v = randomFloat(rng);
    candidate.light_index = lower;
    candidate.barycentrics = vec2(root_u * (1.0 - v), root_u * v);
    vec3 unused_direction;
    float unused_distance;
    vec3 contribution = evaluateAreaLightCandidate(
        candidate, hit, normal, incoming, material, sample_count,
        unused_direction, unused_distance);
#if WAVE_ORDINARYSHADE_UNIFIED_NEE
    candidate.target = ordinarylight_light_candidate_target(contribution);
#else
    candidate.target = max(dot(
        contribution, vec3(0.2126, 0.7152, 0.0722)), 0.0);
#endif
    return candidate;
}

float areaLightCandidateVisibility(
    vec3 hit, vec3 normal, vec3 direction, float distance_to_light)
{
    if (distance_to_light <= 0.0)
        return 0.0;
    vec3 shadow_origin = hit + normal * 0.002;
    float shadow_distance = max(distance_to_light - 0.004, 0.001);
    rayQueryEXT shadow;
#if WAVE_WORK_COUNTERS
    profileWork(1u, 1u);
#endif
    rayQueryInitializeEXT(
        shadow, scene_tlas,
        gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT,
        0x01, shadow_origin, 0.001, direction, shadow_distance);
    while (rayQueryProceedEXT(shadow)) {}
    if (rayQueryGetIntersectionTypeEXT(shadow, true)
            != gl_RayQueryCommittedIntersectionNoneEXT)
        return 0.0;
    return volumeShadowTransmittance(
        shadow_origin, direction, shadow_distance);
}

vec3 environmentColor(vec3 direction)
{
    for (uint index = 0u; index < min(push.point_light_count, 64u); ++index) {
        PointLightData light = point_lights[index];
        if (int(light.position_type.w + 0.5) != 3) continue;
        vec3 radiance = vec3(1.0);
        int texture_index = light.spot_parameters.x < 0.0
            ? -1 : int(light.spot_parameters.x + 0.5);
        if (texture_index >= 0) {
#if WAVE_ORDINARYSHADE_ENVIRONMENT_LIGHTS
            vec2 uv = ordinarylight_environment_uv(
                direction, light.spot_parameters.y);
#else
            float longitude = atan(direction.z, direction.x)
                + light.spot_parameters.y;
            vec2 uv = vec2(
                fract(longitude * 0.15915494309189535 + 0.5),
                acos(clamp(direction.y, -1.0, 1.0)) * 0.3183098861837907);
#endif
            vec3 encoded = sampleSceneTexture(texture_index, uv).rgb;
#if WAVE_ORDINARYSHADE_ENVIRONMENT_LIGHTS
            radiance = ordinarylight_environment_radiance(
                encoded, light.spot_parameters.z, vec3(1.0), 1.0, true);
#else
            radiance = exp2(encoded * light.spot_parameters.z) - 1.0;
#endif
        }
#if WAVE_ORDINARYSHADE_ENVIRONMENT_LIGHTS
        return ordinarylight_environment_radiance(
            radiance, 0.0, light.color_intensity.rgb,
            light.color_intensity.a, false);
#else
        return radiance * light.color_intensity.rgb
            * light.color_intensity.a;
#endif
    }
#if WAVE_ORDINARYSHADE_ENVIRONMENT_LIGHTS
    return ordinarylight_environment_analytic(direction);
#else
    float sky = max(direction.y, 0.0);
    return mix(vec3(0.018, 0.022, 0.032), vec3(0.32, 0.46, 0.72), sky);
#endif
}

vec3 sampleEnvironmentTechnique(
    vec3 hit, vec3 normal, vec3 incoming, MaterialData material,
    inout uint rng, uint sample_count, float technique_probability)
{
#if WAVE_ORDINARYSHADE_ENVIRONMENT_LIGHTS
    vec3 direction = ordinarylight_pbr_cosine_hemisphere(
        normal, randomFloat(rng), randomFloat(rng)
    );
#else
    vec3 direction = cosineHemisphere(
        normal, randomFloat(rng), randomFloat(rng)
    );
#endif
    vec3 shadow_origin = hit + normal * 0.002;
    rayQueryEXT shadow;
#if WAVE_WORK_COUNTERS
    profileWork(1u, 1u);
#endif
    rayQueryInitializeEXT(
        shadow, scene_tlas,
        gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT,
        0x01, shadow_origin, 0.001, direction, 1.0e30
    );
    while (rayQueryProceedEXT(shadow)) {}
    if (rayQueryGetIntersectionTypeEXT(shadow, true) !=
            gl_RayQueryCommittedIntersectionNoneEXT)
        return vec3(0.0);
#if WAVE_ORDINARYSHADE_ENVIRONMENT_LIGHTS
    float cosine = ordinarylight_analytic_light_cosine(normal, direction);
    float pdf = cosine / 3.14159265359;
    float effective_pdf = ordinarylight_environment_effective_pdf(
        cosine, technique_probability);
    float mis = ordinarylight_environment_mis(
        cosine, effective_pdf, float(sample_count));
#else
    float pdf = max(dot(normal, direction), 0.0) / 3.14159265359;
    float effective_pdf = pdf * max(technique_probability, 0.000001);
    float mis = powerHeuristic(effective_pdf * float(sample_count), pdf);
#endif
    float transmittance = volumeShadowTransmittance(
        shadow_origin, direction, 1.0e30);
#if WAVE_ORDINARYSHADE_ENVIRONMENT_LIGHTS
    return ordinarylight_environment_contribution(
        evaluatePbr(material, normal, -incoming, direction),
        environmentColor(direction), cosine, mis, transmittance,
        effective_pdf, material.texture_parameters.w,
        material.emission_metallic.a);
#else
    return evaluatePbr(material, normal, -incoming, direction)
        * environmentColor(direction) * max(dot(normal, direction), 0.0)
        * mis * transmittance / max(effective_pdf, 0.000001)
        * mix(material.texture_parameters.w, 1.0,
              material.emission_metallic.a);
#endif
}

vec3 sampleEnvironment(
    vec3 hit, vec3 normal, vec3 incoming, MaterialData material,
    inout uint rng, uint sample_count)
{
    return sampleEnvironmentTechnique(
        hit, normal, incoming, material, rng, sample_count, 1.0);
}

vec3 sampleUnifiedSecondaryLight(
    vec3 hit, vec3 normal, vec3 incoming, MaterialData material,
    inout uint rng)
{
    bool area_enabled = push.area_light_count > 0u;
    bool environment_enabled = push.environment_samples > 0u;
    if (!area_enabled && !environment_enabled)
        return vec3(0.0);
    // Allocate mixture draws in proportion to the square root of estimated
    // domain power. Direct proportional allocation can starve a dim domain;
    // equal allocation wastes half the visibility budget when emissive
    // geometry dominates the environment, as it commonly does indoors.
#if WAVE_ORDINARYSHADE_UNIFIED_NEE
    float area_probability = ordinarylight_unified_secondary_area_probability(
        area_enabled, environment_enabled, push.area_light_weight,
        push.secondary_area_light_samples, push.environment_samples);
#else
    float area_weight = area_enabled
        ? sqrt(max(push.area_light_weight, 0.000001))
            * float(max(push.secondary_area_light_samples, 1u)) : 0.0;
    float environment_weight = environment_enabled
        ? float(min(push.environment_samples, 4u)) : 0.0;
    float area_probability = area_weight /
        max(area_weight + environment_weight, 0.000001);
#endif
    if (randomFloat(rng) < area_probability)
        return sampleAreaLightTechnique(
            hit, normal, incoming, material, rng, 0u, 1u,
            area_probability);
    return sampleEnvironmentTechnique(
        hit, normal, incoming, material, rng, 1u,
        1.0 - area_probability);
}

vec2 encodeEnvironmentCandidateDirection(vec3 direction)
{
#if WAVE_ORDINARYSHADE_ENVIRONMENT_LIGHTS
    return ordinarylight_environment_encode_direction(direction);
#else
    direction /= abs(direction.x) + abs(direction.y) + abs(direction.z);
    vec2 encoded = direction.xy;
    if (direction.z < 0.0)
        encoded = (1.0 - abs(encoded.yx)) * sign(encoded.xy);
    return encoded * 0.5 + 0.5;
#endif
}

vec3 decodeEnvironmentCandidateDirection(vec2 encoded)
{
#if WAVE_ORDINARYSHADE_ENVIRONMENT_LIGHTS
    return ordinarylight_environment_decode_direction(encoded);
#else
    vec2 value = encoded * 2.0 - 1.0;
    vec3 direction = vec3(value, 1.0 - abs(value.x) - abs(value.y));
    if (direction.z < 0.0)
        direction.xy = (1.0 - abs(direction.yx)) * sign(direction.xy);
    return normalize(direction);
#endif
}

vec3 evaluateEnvironmentCandidate(
    vec2 encoded_direction, vec3 hit, vec3 normal, vec3 incoming,
    MaterialData material, uint sample_count, float technique_probability,
    out vec3 direction, out float distance_to_light)
{
    direction = decodeEnvironmentCandidateDirection(encoded_direction);
    distance_to_light = 1.0e30;
    float cosine = max(dot(normal, direction), 0.0);
    if (cosine <= 0.0)
        return vec3(0.0);
#if WAVE_ORDINARYSHADE_ENVIRONMENT_LIGHTS
    float pdf = cosine / 3.14159265359;
    float effective_pdf = ordinarylight_environment_effective_pdf(
        cosine, technique_probability);
    float mis = ordinarylight_environment_mis(
        cosine, effective_pdf, float(sample_count));
    return ordinarylight_environment_contribution(
        evaluatePbr(material, normal, -incoming, direction),
        environmentColor(direction), cosine, mis, 1.0, effective_pdf,
        material.texture_parameters.w, material.emission_metallic.a);
#else
    float pdf = cosine / 3.14159265359;
    float effective_pdf = pdf * max(technique_probability, 0.000001);
    float mis = powerHeuristic(
        effective_pdf * float(sample_count), pdf);
    return evaluatePbr(material, normal, -incoming, direction)
        * environmentColor(direction) * cosine * mis
        / max(effective_pdf, 0.000001)
        * mix(material.texture_parameters.w, 1.0,
              material.emission_metallic.a);
#endif
}

AreaLightCandidate generateUnifiedPrimaryCandidate(
    vec3 hit, vec3 normal, vec3 incoming, MaterialData material,
    inout uint rng, uint sample_index, uint sample_count)
{
    float area_probability = unifiedAreaDomainProbability();
    if (push.environment_samples == 0u
            || randomFloat(rng) < area_probability) {
        AreaLightCandidate candidate = generateAreaLightCandidate(
            hit, normal, incoming, material, rng, sample_index, sample_count);
        vec3 unused_direction;
        float unused_distance;
        vec3 contribution = evaluateAreaLightCandidateTechnique(
            candidate, hit, normal, incoming, material, sample_count,
            area_probability, unused_direction, unused_distance);
#if WAVE_ORDINARYSHADE_UNIFIED_NEE
        candidate.target = ordinarylight_light_candidate_target(contribution);
#else
        candidate.target = max(dot(
            contribution, vec3(0.2126, 0.7152, 0.0722)), 0.0);
#endif
        return candidate;
    }
    AreaLightCandidate candidate;
    vec3 direction = cosineHemisphere(
        normal, randomFloat(rng), randomFloat(rng));
    candidate.light_index = ENVIRONMENT_LIGHT_CANDIDATE_INDEX;
    candidate.barycentrics = encodeEnvironmentCandidateDirection(direction);
    float unused_distance;
    vec3 contribution = evaluateEnvironmentCandidate(
        candidate.barycentrics, hit, normal, incoming, material, sample_count,
        1.0 - area_probability, direction, unused_distance);
#if WAVE_ORDINARYSHADE_UNIFIED_NEE
    candidate.target = ordinarylight_light_candidate_target(contribution);
#else
    candidate.target = max(dot(
        contribution, vec3(0.2126, 0.7152, 0.0722)), 0.0);
#endif
    return candidate;
}

vec3 evaluateUnifiedPrimaryCandidate(
    AreaLightCandidate candidate, vec3 hit, vec3 normal, vec3 incoming,
    MaterialData material, uint sample_count,
    out vec3 direction, out float distance_to_light)
{
    float area_probability = unifiedAreaDomainProbability();
    if (candidate.light_index == ENVIRONMENT_LIGHT_CANDIDATE_INDEX)
        return evaluateEnvironmentCandidate(
            candidate.barycentrics, hit, normal, incoming, material,
            sample_count, 1.0 - area_probability,
            direction, distance_to_light);
    return evaluateAreaLightCandidateTechnique(
        candidate, hit, normal, incoming, material, sample_count,
        area_probability, direction, distance_to_light);
}
