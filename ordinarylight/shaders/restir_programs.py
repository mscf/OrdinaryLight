"""Typed OrdinaryShade direct-light reservoir arithmetic and packed history ABI."""
import ordinaryshade as osh


@osh.structure
class DirectLightReservoir:
    data: osh.uvec4


@osh.function
def emptyDirectLightReservoir() -> DirectLightReservoir:
    return DirectLightReservoir(osh.uvec4(0xFFFFFFFF, 0, 0, 0))


@osh.function
def updateDirectLightReservoir(
    reservoir: osh.inout(DirectLightReservoir), light_index: osh.u32,
    barycentrics: osh.vec2, target: osh.f32, candidate_weight: osh.f32,
    represented_samples: osh.f32, random_value: osh.f32,
) -> osh.boolean:
    candidate_weight = osh.maximum(candidate_weight, 0.0)
    weight_sum = osh.uint_bits_to_float(reservoir.data.z) + candidate_weight
    target_count = osh.unpack_half2x16(reservoir.data.w)
    sample_count = target_count.y + represented_samples
    selected = candidate_weight > 0.0 and (reservoir.data.x == osh.u32(0xFFFFFFFF)
                                         or random_value * weight_sum < candidate_weight)
    if selected:
        reservoir.data.x = light_index
        reservoir.data.y = osh.pack_half2x16(osh.clamp(barycentrics, osh.vec2(0.0), osh.vec2(1.0)))
        target_count.x = osh.minimum(osh.maximum(target, 0.0), 65504.0)
    reservoir.data.z = osh.float_bits_to_uint(weight_sum)
    reservoir.data.w = osh.pack_half2x16(osh.vec2(target_count.x, sample_count))
    return selected


@osh.function
def mergeDirectLightReservoir(destination: osh.inout(DirectLightReservoir),
                              source: DirectLightReservoir,
                              target_at_current_surface: osh.f32, random_value: osh.f32) -> osh.boolean:
    tc = osh.unpack_half2x16(source.data.w)
    if source.data.x == osh.u32(0xFFFFFFFF) or tc.y <= 0.0:
        return False
    weight = 0.0
    if tc.x > 0.0:
        weight = osh.maximum(target_at_current_surface, 0.0) * osh.uint_bits_to_float(source.data.z) / tc.x
    return updateDirectLightReservoir(destination, source.data.x, osh.unpack_half2x16(source.data.y),
                                      target_at_current_surface, weight, tc.y, random_value)


@osh.function
def mergeCanonicalDirectLightReservoir(destination: osh.inout(DirectLightReservoir),
                                       source: DirectLightReservoir,
                                       target_at_current_surface: osh.f32, random_value: osh.f32) -> osh.boolean:
    tc = osh.unpack_half2x16(source.data.w)
    if source.data.x == osh.u32(0xFFFFFFFF) or tc.x <= 0.0 or tc.y <= 0.0:
        return False
    weight = osh.maximum(target_at_current_surface, 0.0) * osh.uint_bits_to_float(source.data.z) / (tc.x * tc.y)
    return updateDirectLightReservoir(destination, source.data.x, osh.unpack_half2x16(source.data.y),
                                      target_at_current_surface, weight, 1.0, random_value)


@osh.function
def mergePairwiseDirectLightReservoir(destination: osh.inout(DirectLightReservoir),
                                      source: DirectLightReservoir,
                                      target_at_current_surface: osh.f32,
                                      target_at_source_surface: osh.f32, random_value: osh.f32) -> osh.boolean:
    tc = osh.unpack_half2x16(source.data.w)
    if source.data.x == osh.u32(0xFFFFFFFF) or tc.x <= 0.0 or tc.y <= 0.0:
        return False
    total = osh.maximum(target_at_current_surface, 0.0) + osh.maximum(target_at_source_surface, 0.0)
    balance = 0.0
    if total > 0.0:
        balance = 2.0 * osh.maximum(target_at_source_surface, 0.0) / total
    weight = osh.maximum(target_at_current_surface, 0.0) * osh.uint_bits_to_float(source.data.z) / (tc.x * tc.y) * balance
    return updateDirectLightReservoir(destination, source.data.x, osh.unpack_half2x16(source.data.y),
                                      target_at_current_surface, weight, 1.0, random_value)


@osh.function
def mergeBalancedDirectLightReservoir(destination: osh.inout(DirectLightReservoir),
                                      source: DirectLightReservoir,
                                      target_at_current_surface: osh.f32,
                                      active_proposals_over_target_sum: osh.f32, random_value: osh.f32) -> osh.boolean:
    # Preserve the cancellation-free form: source target is already canceled
    # from the proposal balance, avoiding overflow for tiny half-float targets.
    tc = osh.unpack_half2x16(source.data.w)
    if source.data.x == osh.u32(0xFFFFFFFF) or tc.x <= 0.0 or tc.y <= 0.0:
        return False
    weight = osh.maximum(target_at_current_surface, 0.0) * osh.uint_bits_to_float(source.data.z) / tc.y * osh.maximum(active_proposals_over_target_sum, 0.0)
    return updateDirectLightReservoir(destination, source.data.x, osh.unpack_half2x16(source.data.y),
                                      target_at_current_surface, weight, 1.0, random_value)


@osh.function
def directLightReservoirNormalization(reservoir: DirectLightReservoir) -> osh.f32:
    tc = osh.unpack_half2x16(reservoir.data.w)
    if reservoir.data.x == osh.u32(0xFFFFFFFF) or tc.x <= 0.0 or tc.y <= 0.0:
        return 0.0
    return osh.uint_bits_to_float(reservoir.data.z) / (tc.y * tc.x)


@osh.function
def limitDirectLightReservoir(reservoir: DirectLightReservoir, maximum_samples: osh.f32) -> DirectLightReservoir:
    tc = osh.unpack_half2x16(reservoir.data.w)
    maximum_samples = osh.maximum(maximum_samples, 0.0)
    if tc.y > maximum_samples and tc.y > 0.0:
        scale = maximum_samples / tc.y
        reservoir.data.z = osh.float_bits_to_uint(osh.uint_bits_to_float(reservoir.data.z) * scale)
        reservoir.data.w = osh.pack_half2x16(osh.vec2(tc.x, maximum_samples))
    return reservoir


@osh.function
def unpackStoredDirectLightReservoir(header: osh.u32, barycentrics: osh.u32, weight_target: osh.u32) -> DirectLightReservoir:
    light = header & osh.u32(0x01FFFFFF)
    count = header >> osh.u32(25)
    unpacked = osh.unpack_half2x16(weight_target)
    if light == osh.u32(0x01FFFFFF):
        light = osh.u32(0xFFFFFFFF)
    return DirectLightReservoir(osh.uvec4(light, barycentrics, osh.float_bits_to_uint(unpacked.x),
                                         osh.pack_half2x16(osh.vec2(unpacked.y, osh.f32(count)))))


@osh.function
def loadPreviousDirectLightReservoir(reservoir_index: osh.u32) -> DirectLightReservoir:
    word = reservoir_index * osh.u32(3)
    return unpackStoredDirectLightReservoir(previous_reservoir_words[word], previous_reservoir_words[word + osh.u32(1)],
                                            previous_reservoir_words[word + osh.u32(2)])


@osh.function
def storeCurrentDirectLightReservoir(reservoir_index: osh.u32, reservoir: DirectLightReservoir) -> osh.void:
    tc = osh.unpack_half2x16(reservoir.data.w)
    light = osh.u32(0x01FFFFFF)
    if reservoir.data.x != osh.u32(0xFFFFFFFF):
        light = osh.minimum(reservoir.data.x, osh.u32(0x01FFFFFE))
    count = osh.u32(osh.clamp(osh.round(tc.y), 0.0, 127.0))
    weight = osh.clamp(osh.uint_bits_to_float(reservoir.data.z), 0.0, 65504.0)
    word = reservoir_index * osh.u32(3)
    current_reservoir_words[word] = light | (count << osh.u32(25))
    current_reservoir_words[word + osh.u32(1)] = reservoir.data.y
    current_reservoir_words[word + osh.u32(2)] = osh.pack_half2x16(osh.vec2(weight, tc.x))


@osh.function
def loadPreviousEnvironmentReservoir(reservoir_index: osh.u32, pixel_count: osh.u32) -> DirectLightReservoir:
    word = pixel_count * osh.u32(3) + reservoir_index * osh.u32(2)
    header = previous_reservoir_words[word]
    count = (header >> osh.u32(24)) & osh.u32(0x7F)
    if count == osh.u32(0):
        return emptyDirectLightReservoir()
    encoded = osh.vec2(osh.f32(header & osh.u32(0xFFF)), osh.f32((header >> osh.u32(12)) & osh.u32(0xFFF))) / 4095.0
    wt = osh.unpack_half2x16(previous_reservoir_words[word + osh.u32(1)])
    return DirectLightReservoir(osh.uvec4(0x01FFFFFE, osh.pack_half2x16(encoded), osh.float_bits_to_uint(wt.x),
                                         osh.pack_half2x16(osh.vec2(wt.y, osh.f32(count)))))


@osh.function
def storeCurrentEnvironmentReservoir(reservoir_index: osh.u32, pixel_count: osh.u32, reservoir: DirectLightReservoir) -> osh.void:
    word = pixel_count * osh.u32(3) + reservoir_index * osh.u32(2)
    tc = osh.unpack_half2x16(reservoir.data.w)
    count = osh.u32(0)
    if reservoir.data.x != osh.u32(0xFFFFFFFF):
        count = osh.u32(osh.clamp(osh.round(tc.y), 1.0, 127.0))
    encoded = osh.clamp(osh.unpack_half2x16(reservoir.data.y), osh.vec2(0.0), osh.vec2(1.0))
    quantized = osh.uvec2(osh.round(encoded * 4095.0))
    current_reservoir_words[word] = quantized.x | (quantized.y << osh.u32(12)) | (count << osh.u32(24))
    current_reservoir_words[word + osh.u32(1)] = osh.pack_half2x16(osh.vec2(
        osh.clamp(osh.uint_bits_to_float(reservoir.data.z), 0.0, 65504.0), osh.clamp(tc.x, 0.0, 65504.0)))


COMMON = (emptyDirectLightReservoir, updateDirectLightReservoir, mergeDirectLightReservoir,
          mergeCanonicalDirectLightReservoir, mergePairwiseDirectLightReservoir, mergeBalancedDirectLightReservoir,
          directLightReservoirNormalization, limitDirectLightReservoir)
STORAGE = (unpackStoredDirectLightReservoir, loadPreviousDirectLightReservoir, storeCurrentDirectLightReservoir,
           loadPreviousEnvironmentReservoir, storeCurrentEnvironmentReservoir)


@osh.compute()
def reservoir_module(
    current_reservoir_words: osh.storage_buffer(osh.u32, binding=16),
    previous_reservoir_words: osh.storage_buffer(osh.u32, access='read', binding=17),
):
    pass
