#ifndef ORDINARYLIGHT_SAMPLING_V1
#define ORDINARYLIGHT_SAMPLING_V1 1
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


#endif
