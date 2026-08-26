// Shared weighted-reservoir operations for direct-light sample reuse.
// The arithmetic representation remains four words so reservoir operations can
// retain a full-precision weight sum. Persistent history uses a tightly packed
// 12-byte ABI: 25-bit light index + 7-bit sample count, packed half
// barycentrics, and packed half weight-sum/selected-target. The history limit is
// 64 samples, so seven count bits are sufficient; 0x01ffffff is the packed
// empty-light sentinel.
struct DirectLightReservoir {
    uvec4 data;
};

#if WAVE_RESTIR_PRIMARY
layout(set = 0, binding = 16, std430) buffer CurrentReservoirBuffer {
    uint current_reservoir_words[];
};
layout(set = 0, binding = 17, std430) readonly buffer PreviousReservoirBuffer {
    uint previous_reservoir_words[];
};

const uint DIRECT_LIGHT_STORAGE_INDEX_BITS = 25u;
const uint DIRECT_LIGHT_STORAGE_INDEX_MASK = 0x01ffffffu;
const uint DIRECT_LIGHT_STORAGE_COUNT_SHIFT = 25u;

DirectLightReservoir unpackStoredDirectLightReservoir(
    uint header, uint barycentrics, uint weight_target)
{
    DirectLightReservoir reservoir;
    uint stored_index = header & DIRECT_LIGHT_STORAGE_INDEX_MASK;
    uint sample_count = header >> DIRECT_LIGHT_STORAGE_COUNT_SHIFT;
    vec2 unpacked = unpackHalf2x16(weight_target);
    reservoir.data = uvec4(
        stored_index == DIRECT_LIGHT_STORAGE_INDEX_MASK
            ? 0xffffffffu : stored_index,
        barycentrics,
        floatBitsToUint(unpacked.x),
        packHalf2x16(vec2(unpacked.y, float(sample_count))));
    return reservoir;
}

DirectLightReservoir loadPreviousDirectLightReservoir(uint reservoir_index)
{
    uint word = reservoir_index * 3u;
    return unpackStoredDirectLightReservoir(
        previous_reservoir_words[word],
        previous_reservoir_words[word + 1u],
        previous_reservoir_words[word + 2u]);
}

void storeCurrentDirectLightReservoir(
    uint reservoir_index, DirectLightReservoir reservoir)
{
    vec2 target_count = unpackHalf2x16(reservoir.data.w);
    uint light_index = reservoir.data.x == 0xffffffffu
        ? DIRECT_LIGHT_STORAGE_INDEX_MASK
        : min(reservoir.data.x, DIRECT_LIGHT_STORAGE_INDEX_MASK - 1u);
    uint sample_count = uint(clamp(round(target_count.y), 0.0, 127.0));
    float weight_sum = clamp(
        uintBitsToFloat(reservoir.data.z), 0.0, 65504.0);
    uint word = reservoir_index * 3u;
    current_reservoir_words[word] = light_index
        | (sample_count << DIRECT_LIGHT_STORAGE_COUNT_SHIFT);
    current_reservoir_words[word + 1u] = reservoir.data.y;
    current_reservoir_words[word + 2u] = packHalf2x16(
        vec2(weight_sum, target_count.x));
}

// Environment reservoirs do not need a light index. Store a 12-bit octahedral
// direction per component plus the seven-bit represented-sample count in one
// word, followed by half-precision weight sum and selected target. This keeps
// the independent environment history to eight bytes per pixel.
DirectLightReservoir loadPreviousEnvironmentReservoir(
    uint reservoir_index, uint pixel_count)
{
    uint word = pixel_count * 3u + reservoir_index * 2u;
    uint header = previous_reservoir_words[word];
    uint sample_count = (header >> 24u) & 0x7fu;
    if (sample_count == 0u) {
        DirectLightReservoir empty_reservoir;
        empty_reservoir.data = uvec4(0xffffffffu, 0u, 0u, 0u);
        return empty_reservoir;
    }
    vec2 encoded = vec2(
        float(header & 0xfffu),
        float((header >> 12u) & 0xfffu)) / 4095.0;
    vec2 weight_target = unpackHalf2x16(
        previous_reservoir_words[word + 1u]);
    DirectLightReservoir reservoir;
    reservoir.data = uvec4(
        0x01fffffeu,
        packHalf2x16(encoded),
        floatBitsToUint(weight_target.x),
        packHalf2x16(vec2(weight_target.y, float(sample_count))));
    return reservoir;
}

void storeCurrentEnvironmentReservoir(
    uint reservoir_index, uint pixel_count, DirectLightReservoir reservoir)
{
    uint word = pixel_count * 3u + reservoir_index * 2u;
    vec2 target_count = unpackHalf2x16(reservoir.data.w);
    uint sample_count = reservoir.data.x == 0xffffffffu
        ? 0u : uint(clamp(round(target_count.y), 1.0, 127.0));
    vec2 encoded = clamp(unpackHalf2x16(reservoir.data.y), 0.0, 1.0);
    uvec2 quantized = uvec2(round(encoded * 4095.0));
    current_reservoir_words[word] = quantized.x
        | (quantized.y << 12u) | (sample_count << 24u);
    current_reservoir_words[word + 1u] = packHalf2x16(vec2(
        clamp(uintBitsToFloat(reservoir.data.z), 0.0, 65504.0),
        clamp(target_count.x, 0.0, 65504.0)));
}
#endif

DirectLightReservoir emptyDirectLightReservoir()
{
    DirectLightReservoir reservoir;
    reservoir.data = uvec4(0xffffffffu, 0u, 0u, 0u);
    return reservoir;
}

bool updateDirectLightReservoir(
    inout DirectLightReservoir reservoir, uint light_index,
    vec2 barycentrics, float target, float candidate_weight,
    float represented_samples, float random_value)
{
    candidate_weight = max(candidate_weight, 0.0);
    float weight_sum = uintBitsToFloat(reservoir.data.z) + candidate_weight;
    vec2 target_count = unpackHalf2x16(reservoir.data.w);
    float sample_count = target_count.y + represented_samples;
    bool selected = candidate_weight > 0.0
        && (reservoir.data.x == 0xffffffffu
            || random_value * weight_sum < candidate_weight);
    if (selected) {
        reservoir.data.x = light_index;
        reservoir.data.y = packHalf2x16(clamp(barycentrics, 0.0, 1.0));
        target_count.x = min(max(target, 0.0), 65504.0);
    }
    reservoir.data.z = floatBitsToUint(weight_sum);
    reservoir.data.w = packHalf2x16(vec2(target_count.x, sample_count));
    return selected;
}

bool mergeDirectLightReservoir(
    inout DirectLightReservoir destination,
    DirectLightReservoir source, float target_at_current_surface,
    float random_value)
{
    vec2 source_target_count = unpackHalf2x16(source.data.w);
    if (source.data.x == 0xffffffffu || source_target_count.y <= 0.0)
        return false;
    float reuse_weight = source_target_count.x > 0.0
        ? max(target_at_current_surface, 0.0)
            * uintBitsToFloat(source.data.z) / source_target_count.x
        : 0.0;
    return updateDirectLightReservoir(
        destination, source.data.x, unpackHalf2x16(source.data.y),
        target_at_current_surface, reuse_weight, source_target_count.y,
        random_value);
}

bool mergeCanonicalDirectLightReservoir(
    inout DirectLightReservoir destination,
    DirectLightReservoir source, float target_at_current_surface,
    float random_value)
{
    vec2 source_target_count = unpackHalf2x16(source.data.w);
    if (source.data.x == 0xffffffffu || source_target_count.x <= 0.0
            || source_target_count.y <= 0.0)
        return false;
    // A spatial neighbor is a correlated reservoir summary, not another
    // independent batch of M observations. Collapse it to one canonical
    // representative using its normalized source weight.
    float canonical_weight = max(target_at_current_surface, 0.0)
        * uintBitsToFloat(source.data.z)
        / (source_target_count.x * source_target_count.y);
    return updateDirectLightReservoir(
        destination, source.data.x, unpackHalf2x16(source.data.y),
        target_at_current_surface, canonical_weight, 1.0, random_value);
}

bool mergePairwiseDirectLightReservoir(
    inout DirectLightReservoir destination,
    DirectLightReservoir source, float target_at_current_surface,
    float target_at_source_surface, float random_value)
{
    vec2 source_target_count = unpackHalf2x16(source.data.w);
    if (source.data.x == 0xffffffffu || source_target_count.x <= 0.0
            || source_target_count.y <= 0.0)
        return false;
    // Balance the source proposal against the destination proposal. The
    // factor is normalized to one when both surfaces assign equal density,
    // preserving the canonical merge in the locally stationary case.
    float target_sum = max(target_at_current_surface, 0.0)
        + max(target_at_source_surface, 0.0);
    float pairwise_balance = target_sum > 0.0
        ? 2.0 * max(target_at_source_surface, 0.0) / target_sum : 0.0;
    float pairwise_weight = max(target_at_current_surface, 0.0)
        * uintBitsToFloat(source.data.z)
        / (source_target_count.x * source_target_count.y)
        * pairwise_balance;
    return updateDirectLightReservoir(
        destination, source.data.x, unpackHalf2x16(source.data.y),
        target_at_current_surface, pairwise_weight, 1.0, random_value);
}

bool mergeBalancedDirectLightReservoir(
    inout DirectLightReservoir destination,
    DirectLightReservoir source, float target_at_current_surface,
    float active_proposals_over_target_sum, float random_value)
{
    vec2 source_target_count = unpackHalf2x16(source.data.w);
    if (source.data.x == 0xffffffffu || source_target_count.x <= 0.0
            || source_target_count.y <= 0.0)
        return false;
    // This is the cancellation-free form of the generalized balance merge:
    //
    //   current_target * W / (source_target * M)
    //       * active_proposals * source_target / target_sum
    //
    // Computing those factors separately can overflow when the half-precision
    // source target is tiny even though their product is well behaved.
    float weight = max(target_at_current_surface, 0.0)
        * uintBitsToFloat(source.data.z)
        / source_target_count.y
        * max(active_proposals_over_target_sum, 0.0);
    return updateDirectLightReservoir(
        destination, source.data.x, unpackHalf2x16(source.data.y),
        target_at_current_surface, weight, 1.0, random_value);
}

float directLightReservoirNormalization(DirectLightReservoir reservoir)
{
    vec2 target_count = unpackHalf2x16(reservoir.data.w);
    if (reservoir.data.x == 0xffffffffu || target_count.x <= 0.0
            || target_count.y <= 0.0)
        return 0.0;
    return uintBitsToFloat(reservoir.data.z)
        / (target_count.y * target_count.x);
}

DirectLightReservoir limitDirectLightReservoir(
    DirectLightReservoir reservoir, float maximum_samples)
{
    vec2 target_count = unpackHalf2x16(reservoir.data.w);
    maximum_samples = max(maximum_samples, 0.0);
    if (target_count.y > maximum_samples && target_count.y > 0.0) {
        float scale = maximum_samples / target_count.y;
        reservoir.data.z = floatBitsToUint(
            uintBitsToFloat(reservoir.data.z) * scale);
        reservoir.data.w = packHalf2x16(vec2(target_count.x, maximum_samples));
    }
    return reservoir;
}
