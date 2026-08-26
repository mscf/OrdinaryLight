// Compact indirect-light reservoir ABI shared by probe and production shaders.
// Storage is six raw words (24 bytes) per reservoir; using a GLSL struct array
// would round the std430 stride up to 32 bytes because of uvec4 alignment.

struct IndirectLightSample {
    vec3 secondary_position;
    float proposal_pdf;
    vec3 secondary_normal;
    float target;
    vec3 radiance;
};

struct IndirectLightReservoir {
    IndirectLightSample selected;
    float weight_sum;
    uint sample_count;
    bool valid;
    uint debug_flags;
};

#if WAVE_INDIRECT_REUSE_STORAGE
#ifndef WAVE_INDIRECT_REUSE_BINDING
#define WAVE_INDIRECT_REUSE_BINDING 0
#endif
layout(set = 0, binding = WAVE_INDIRECT_REUSE_BINDING, std430) buffer IndirectReservoirBuffer {
    uint indirect_reservoir_words[];
};
#endif

vec2 indirectEncodeNormal(vec3 normal)
{
    normal /= abs(normal.x) + abs(normal.y) + abs(normal.z);
    if (normal.z < 0.0)
        normal.xy = (1.0 - abs(normal.yx)) * sign(normal.xy);
    return normal.xy;
}

vec3 indirectDecodeNormal(vec2 encoded)
{
    vec3 normal = vec3(
        encoded, 1.0 - abs(encoded.x) - abs(encoded.y));
    if (normal.z < 0.0)
        normal.xy = (1.0 - abs(normal.yx)) * sign(normal.xy);
    return normalize(normal);
}

uint indirectPackRgb9e5(vec3 color)
{
    color = clamp(color, vec3(0.0), vec3(65408.0));
    float maximum = max(color.r, max(color.g, color.b));
    if (maximum < exp2(-16.0))
        return 0u;
    int exponent = min(max(-16, int(floor(log2(maximum)))) + 1, 16);
    float scale = exp2(float(exponent - 9));
    uvec3 mantissa = uvec3(round(color / scale));
    if (max(mantissa.r, max(mantissa.g, mantissa.b)) > 511u
            && exponent < 16) {
        exponent += 1;
        scale *= 2.0;
        mantissa = uvec3(round(color / scale));
    }
    mantissa = min(mantissa, uvec3(511u));
    return mantissa.r | (mantissa.g << 9u) | (mantissa.b << 18u)
        | (uint(exponent + 15) << 27u);
}

vec3 indirectUnpackRgb9e5(uint packed)
{
    int exponent = int((packed >> 27u) & 0x1fu) - 15;
    float scale = exp2(float(exponent - 9));
    return vec3(
        float(packed & 0x1ffu),
        float((packed >> 9u) & 0x1ffu),
        float((packed >> 18u) & 0x1ffu)) * scale;
}

IndirectLightReservoir emptyIndirectLightReservoir()
{
    IndirectLightReservoir reservoir;
    reservoir.selected.secondary_position = vec3(0.0);
    reservoir.selected.proposal_pdf = 1.0;
    reservoir.selected.secondary_normal = vec3(0.0, 1.0, 0.0);
    reservoir.selected.target = 0.0;
    reservoir.selected.radiance = vec3(0.0);
    reservoir.weight_sum = 0.0;
    reservoir.sample_count = 0u;
    reservoir.valid = false;
    reservoir.debug_flags = 0u;
    return reservoir;
}

#if WAVE_INDIRECT_REUSE_STORAGE
void storeIndirectLightReservoir(
    uint reservoir_index, IndirectLightReservoir reservoir,
    vec3 camera_origin)
{
    uint word = reservoir_index * 6u;
    if (!reservoir.valid || reservoir.sample_count == 0u) {
        for (uint index = 0u; index < 6u; ++index)
            indirect_reservoir_words[word + index] = 0u;
        return;
    }
    vec3 relative_position =
        reservoir.selected.secondary_position - camera_origin;
    indirect_reservoir_words[word] = packHalf2x16(relative_position.xy);
    indirect_reservoir_words[word + 1u] = packHalf2x16(vec2(
        relative_position.z,
        clamp(reservoir.selected.proposal_pdf, exp2(-24.0), 65504.0)));
    indirect_reservoir_words[word + 2u] = packUnorm2x16(
        indirectEncodeNormal(reservoir.selected.secondary_normal) * 0.5 + 0.5);
    indirect_reservoir_words[word + 3u] = indirectPackRgb9e5(
        reservoir.selected.radiance);
    indirect_reservoir_words[word + 4u] = packHalf2x16(vec2(
        clamp(reservoir.weight_sum, 0.0, 65504.0),
        clamp(reservoir.selected.target, 0.0, 65504.0)));
    indirect_reservoir_words[word + 5u] = 0x80000000u
        | (reservoir.debug_flags & 0x007fff00u)
        | min(reservoir.sample_count, 127u);
}

IndirectLightReservoir loadIndirectLightReservoir(
    uint reservoir_index, vec3 camera_origin)
{
    uint word = reservoir_index * 6u;
    uint header = indirect_reservoir_words[word + 5u];
    uint sample_count = header & 0x7fu;
    if ((header & 0x80000000u) == 0u || sample_count == 0u)
        return emptyIndirectLightReservoir();
    vec2 position_xy = unpackHalf2x16(indirect_reservoir_words[word]);
    vec2 position_z_pdf = unpackHalf2x16(
        indirect_reservoir_words[word + 1u]);
    vec2 weight_target = unpackHalf2x16(
        indirect_reservoir_words[word + 4u]);
    IndirectLightReservoir reservoir;
    reservoir.selected.secondary_position = camera_origin
        + vec3(position_xy, position_z_pdf.x);
    reservoir.selected.proposal_pdf = position_z_pdf.y;
    reservoir.selected.secondary_normal = indirectDecodeNormal(
        unpackUnorm2x16(indirect_reservoir_words[word + 2u]) * 2.0 - 1.0);
    reservoir.selected.target = weight_target.y;
    reservoir.selected.radiance = indirectUnpackRgb9e5(
        indirect_reservoir_words[word + 3u]);
    reservoir.weight_sum = weight_target.x;
    reservoir.sample_count = sample_count;
    reservoir.valid = true;
    reservoir.debug_flags = header & 0x007fff00u;
    return reservoir;
}
#endif
