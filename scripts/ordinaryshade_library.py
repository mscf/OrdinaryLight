"""Reusable typed shader helpers shared by generated Ordinary Light stages."""

import ordinaryshade as osh


@osh.function
def acesApproximation(color: osh.vec3) -> osh.vec3:
    return osh.clamp(
        (color * (2.51 * color + 0.03))
        / (color * (2.43 * color + 0.59) + 0.14),
        osh.vec3(0.0),
        osh.vec3(1.0),
    )


@osh.function
def linearToSrgb(color: osh.vec3) -> osh.vec3:
    low = color <= osh.vec3(0.0031308)
    lower = color * 12.92
    upper = 1.055 * osh.power(color, osh.vec3(1.0 / 2.4)) - 0.055
    return osh.select(low, lower, upper)


@osh.function
def nv12ByteValue(value: osh.f32) -> osh.u32:
    return osh.u32(osh.clamp(osh.round(value), 0.0, 255.0))


@osh.function
def nv12Luma(rgb: osh.vec3) -> osh.u32:
    return nv12ByteValue(
        16.0 + 219.0 * osh.dot(rgb, osh.vec3(0.2126, 0.7152, 0.0722))
    )


@osh.function
def nv12Chroma(rgb: osh.vec3) -> osh.uvec2:
    cb = osh.dot(rgb, osh.vec3(-0.114572, -0.385428, 0.5))
    cr = osh.dot(rgb, osh.vec3(0.5, -0.454153, -0.045847))
    return osh.uvec2(
        nv12ByteValue(128.0 + 224.0 * cb),
        nv12ByteValue(128.0 + 224.0 * cr),
    )


@osh.function
def nv12Pixel(
    x: osh.u32, y: osh.u32, width: osh.u32, height: osh.u32
) -> osh.ivec2:
    return osh.clamp(
        osh.ivec2(osh.i32(x), osh.i32(y)),
        osh.ivec2(0),
        osh.ivec2(osh.i32(width), osh.i32(height)) - osh.ivec2(1),
    )


@osh.function
def pack4Bytes(a: osh.u32, b: osh.u32, c: osh.u32, d: osh.u32) -> osh.u32:
    return (
        a
        | (b << osh.u32(8))
        | (c << osh.u32(16))
        | (d << osh.u32(24))
    )


@osh.function
def p010TenBitValue(value: osh.f32) -> osh.u32:
    return osh.u32(osh.clamp(osh.round(value), 0.0, 1023.0))


@osh.function
def p010Luma(rgb: osh.vec3) -> osh.u32:
    return p010TenBitValue(
        64.0 + 876.0 * osh.dot(rgb, osh.vec3(0.2126, 0.7152, 0.0722))
    ) << osh.u32(6)


@osh.function
def p010Chroma(rgb: osh.vec3) -> osh.uvec2:
    cb = osh.dot(rgb, osh.vec3(-0.114572, -0.385428, 0.5))
    cr = osh.dot(rgb, osh.vec3(0.5, -0.454153, -0.045847))
    return osh.uvec2(
        p010TenBitValue(512.0 + 896.0 * cb) << osh.u32(6),
        p010TenBitValue(512.0 + 896.0 * cr) << osh.u32(6),
    )


@osh.function
def p010SourcePixel(
    x: osh.u32, y: osh.u32, source_size: osh.ivec2,
    width: osh.u32, height: osh.u32,
) -> osh.ivec2:
    return osh.clamp(
        osh.ivec2(
            osh.i32(x) * source_size.x / osh.i32(width),
            osh.i32(y) * source_size.y / osh.i32(height),
        ),
        osh.ivec2(0),
        source_size - osh.ivec2(1),
    )


@osh.function
def p010Color(linear: osh.vec3, exposure: osh.f32) -> osh.vec3:
    return linearToSrgb(
        acesApproximation(osh.maximum(linear, osh.vec3(0.0)) * exposure)
    )


@osh.function
def pack2x16(a: osh.u32, b: osh.u32) -> osh.u32:
    return a | (b << osh.u32(16))


@osh.function
def decodeAtrousNormal(encoded: osh.vec2) -> osh.vec3:
    normal = osh.vec3(
        encoded, 1.0 - osh.absolute(encoded.x) - osh.absolute(encoded.y)
    )
    if normal.z < 0.0:
        normal.xy = (
            1.0 - osh.absolute(normal.yx)
        ) * osh.sign(normal.xy)
    return osh.normalize(normal)


@osh.function
def atrousKernel(index: osh.i32) -> osh.f32:
    if index == -2:
        return 1.0
    if index == -1:
        return 4.0
    if index == 0:
        return 6.0
    if index == 1:
        return 4.0
    return 1.0


@osh.function
def overlayLetterPixel(letter: osh.i32, pixel: osh.ivec2) -> osh.boolean:
    if pixel.x < 0 or pixel.y < 0 or pixel.x >= 5 or pixel.y >= 7:
        return False
    if letter == 0:
        return pixel.x == 0 or pixel.y == 0 or (pixel.y == 3 and pixel.x < 4)
    if letter == 1:
        return (
            pixel.x == 0
            or ((pixel.y == 0 or pixel.y == 3) and pixel.x < 4)
            or (pixel.x == 4 and pixel.y > 0 and pixel.y < 3)
        )
    return (
        pixel.y == 0 or pixel.y == 3 or pixel.y == 6
        or (pixel.x == 0 and pixel.y > 0 and pixel.y < 3)
        or (pixel.x == 4 and pixel.y > 3 and pixel.y < 6)
    )


@osh.function
def overlayDigitMask(digit: osh.i32) -> osh.u32:
    if digit == 0:
        return osh.u32(0x3F)
    if digit == 1:
        return osh.u32(0x06)
    if digit == 2:
        return osh.u32(0x5B)
    if digit == 3:
        return osh.u32(0x4F)
    if digit == 4:
        return osh.u32(0x66)
    if digit == 5:
        return osh.u32(0x6D)
    if digit == 6:
        return osh.u32(0x7D)
    if digit == 7:
        return osh.u32(0x07)
    if digit == 8:
        return osh.u32(0x7F)
    return osh.u32(0x6F)


@osh.function
def overlayDigitPixel(digit: osh.i32, pixel: osh.ivec2) -> osh.boolean:
    if pixel.x < 0 or pixel.y < 0 or pixel.x >= 5 or pixel.y >= 9:
        return False
    mask = overlayDigitMask(osh.clamp(digit, 0, 9))
    segment_a = (mask & osh.u32(0x01)) != osh.u32(0) and pixel.y <= 1 and pixel.x > 0 and pixel.x < 4
    segment_b = (mask & osh.u32(0x02)) != osh.u32(0) and pixel.x >= 3 and pixel.y > 0 and pixel.y < 4
    segment_c = (mask & osh.u32(0x04)) != osh.u32(0) and pixel.x >= 3 and pixel.y > 4 and pixel.y < 8
    segment_d = (mask & osh.u32(0x08)) != osh.u32(0) and pixel.y >= 7 and pixel.x > 0 and pixel.x < 4
    segment_e = (mask & osh.u32(0x10)) != osh.u32(0) and pixel.x <= 1 and pixel.y > 4 and pixel.y < 8
    segment_f = (mask & osh.u32(0x20)) != osh.u32(0) and pixel.x <= 1 and pixel.y > 0 and pixel.y < 4
    segment_g = (mask & osh.u32(0x40)) != osh.u32(0) and pixel.y >= 3 and pixel.y <= 5 and pixel.x > 0 and pixel.x < 4
    return segment_a or segment_b or segment_c or segment_d or segment_e or segment_f or segment_g


@osh.function
def fpsOverlay(
    pixel: osh.uvec2, size: osh.uvec2, color: osh.vec3, overlay: osh.vec4
) -> osh.vec3:
    if overlay.y < 0.5:
        return color
    scale = osh.clamp(osh.i32(osh.f32(size.y) / 360.0 + 0.5), 2, 6)
    logical = osh.ivec2(pixel) / scale
    if logical.x >= 49 or logical.y >= 13:
        return color
    color = color * 0.22
    point = logical - osh.ivec2(2)
    ink = (
        overlayLetterPixel(0, point)
        or overlayLetterPixel(1, point - osh.ivec2(6, 0))
        or overlayLetterPixel(2, point - osh.ivec2(12, 0))
    )
    fps = osh.clamp(osh.i32(overlay.x + 0.5), 0, 9999)
    digits = 4 if fps >= 1000 else (3 if fps >= 100 else (2 if fps >= 10 else 1))
    for index in range(4):
        if index >= digits:
            break
        divisor = 1000 if index == 0 else (100 if index == 1 else (10 if index == 2 else 1))
        if digits < 4:
            divisor = (100 if index == 0 else (10 if index == 1 else 1)) if digits == 3 else ((10 if index == 0 else 1) if digits == 2 else 1)
        digit = (fps / divisor) % 10
        ink = ink or overlayDigitPixel(
            digit, point - osh.ivec2(20 + index * 6, 0)
        )
    return osh.vec3(0.35, 1.0, 0.45) if ink else color


@osh.function
def waveHash(value: osh.u32) -> osh.u32:
    value = value ^ (value >> osh.u32(16))
    value = value * osh.u32(0x7FEB352D)
    value = value ^ (value >> osh.u32(15))
    value = value * osh.u32(0x846CA68B)
    value = value ^ (value >> osh.u32(16))
    return value


@osh.function
def waveRandomFloat(value: osh.u32) -> osh.f32:
    return osh.f32(value) * (1.0 / 4294967296.0)


@osh.structure
class IndirectLightSample:
    secondary_position: osh.vec3
    proposal_pdf: osh.f32
    secondary_normal: osh.vec3
    target: osh.f32
    radiance: osh.vec3


@osh.structure
class IndirectLightReservoir:
    selected: IndirectLightSample
    weight_sum: osh.f32
    sample_count: osh.u32
    valid: osh.boolean
    debug_flags: osh.u32


@osh.function
def indirectEncodeNormal(normal: osh.vec3) -> osh.vec2:
    normal = normal / (
        osh.absolute(normal.x) + osh.absolute(normal.y) + osh.absolute(normal.z)
    )
    if normal.z < 0.0:
        normal.xy = (1.0 - osh.absolute(normal.yx)) * osh.sign(normal.xy)
    return normal.xy


@osh.function
def indirectDecodeNormal(encoded: osh.vec2) -> osh.vec3:
    normal = osh.vec3(
        encoded, 1.0 - osh.absolute(encoded.x) - osh.absolute(encoded.y)
    )
    if normal.z < 0.0:
        normal.xy = (1.0 - osh.absolute(normal.yx)) * osh.sign(normal.xy)
    return osh.normalize(normal)


@osh.function
def indirectPackRgb9e5(color: osh.vec3) -> osh.u32:
    color = osh.clamp(color, osh.vec3(0.0), osh.vec3(65408.0))
    maximum = osh.maximum(color.r, osh.maximum(color.g, color.b))
    if maximum < osh.exp2(-16.0):
        return osh.u32(0)
    exponent = osh.minimum(
        osh.maximum(-16, osh.i32(osh.floor(osh.log2(maximum)))) + 1, 16
    )
    scale = osh.exp2(osh.f32(exponent - 9))
    mantissa = osh.uvec3(osh.round(color / scale))
    if (
        osh.maximum(mantissa.r, osh.maximum(mantissa.g, mantissa.b))
        > osh.u32(511) and exponent < 16
    ):
        exponent = exponent + 1
        scale = scale * 2.0
        mantissa = osh.uvec3(osh.round(color / scale))
    mantissa = osh.minimum(mantissa, osh.uvec3(osh.u32(511)))
    return (
        mantissa.r | (mantissa.g << osh.u32(9))
        | (mantissa.b << osh.u32(18))
        | (osh.u32(exponent + 15) << osh.u32(27))
    )


@osh.function
def indirectUnpackRgb9e5(packed: osh.u32) -> osh.vec3:
    exponent = osh.i32((packed >> osh.u32(27)) & osh.u32(0x1F)) - 15
    scale = osh.exp2(osh.f32(exponent - 9))
    return osh.vec3(
        osh.f32(packed & osh.u32(0x1FF)),
        osh.f32((packed >> osh.u32(9)) & osh.u32(0x1FF)),
        osh.f32((packed >> osh.u32(18)) & osh.u32(0x1FF)),
    ) * scale


@osh.function
def emptyIndirectLightReservoir() -> IndirectLightReservoir:
    return IndirectLightReservoir(
        IndirectLightSample(
            osh.vec3(0.0), 1.0, osh.vec3(0.0, 1.0, 0.0), 0.0,
            osh.vec3(0.0),
        ),
        0.0, osh.u32(0), False, osh.u32(0),
    )


@osh.function
def storeIndirectLightReservoir(
    reservoir_index: osh.u32, reservoir: IndirectLightReservoir,
    camera_origin: osh.vec3,
) -> osh.void:
    word = reservoir_index * osh.u32(6)
    if not reservoir.valid or reservoir.sample_count == osh.u32(0):
        for index in range(6):
            indirect_reservoir_words[word + osh.u32(index)] = osh.u32(0)
        return
    relative_position = reservoir.selected.secondary_position - camera_origin
    indirect_reservoir_words[word] = osh.pack_half2x16(relative_position.xy)
    indirect_reservoir_words[word + osh.u32(1)] = osh.pack_half2x16(osh.vec2(
        relative_position.z,
        osh.clamp(reservoir.selected.proposal_pdf, osh.exp2(-24.0), 65504.0),
    ))
    indirect_reservoir_words[word + osh.u32(2)] = osh.pack_unorm2x16(
        indirectEncodeNormal(reservoir.selected.secondary_normal) * 0.5 + 0.5
    )
    indirect_reservoir_words[word + osh.u32(3)] = indirectPackRgb9e5(
        reservoir.selected.radiance
    )
    indirect_reservoir_words[word + osh.u32(4)] = osh.pack_half2x16(osh.vec2(
        osh.clamp(reservoir.weight_sum, 0.0, 65504.0),
        osh.clamp(reservoir.selected.target, 0.0, 65504.0),
    ))
    indirect_reservoir_words[word + osh.u32(5)] = (
        osh.u32(0x80000000)
        | (reservoir.debug_flags & osh.u32(0x007FFF00))
        | osh.minimum(reservoir.sample_count, osh.u32(127))
    )


@osh.function
def loadIndirectLightReservoir(
    reservoir_index: osh.u32, camera_origin: osh.vec3,
) -> IndirectLightReservoir:
    word = reservoir_index * osh.u32(6)
    header = indirect_reservoir_words[word + osh.u32(5)]
    sample_count = header & osh.u32(0x7F)
    if (
        (header & osh.u32(0x80000000)) == osh.u32(0)
        or sample_count == osh.u32(0)
    ):
        return emptyIndirectLightReservoir()
    position_xy = osh.unpack_half2x16(indirect_reservoir_words[word])
    position_z_pdf = osh.unpack_half2x16(
        indirect_reservoir_words[word + osh.u32(1)]
    )
    weight_target = osh.unpack_half2x16(
        indirect_reservoir_words[word + osh.u32(4)]
    )
    return IndirectLightReservoir(
        IndirectLightSample(
            camera_origin + osh.vec3(position_xy, position_z_pdf.x),
            position_z_pdf.y,
            indirectDecodeNormal(osh.unpack_unorm2x16(
                indirect_reservoir_words[word + osh.u32(2)]
            ) * 2.0 - 1.0),
            weight_target.y,
            indirectUnpackRgb9e5(
                indirect_reservoir_words[word + osh.u32(3)]
            ),
        ),
        weight_target.x, sample_count, True,
        header & osh.u32(0x007FFF00),
    )
