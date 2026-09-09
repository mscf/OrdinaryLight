"""OrdinaryShade FP32 port of AMD FSR 1 EASU (without RCAS).

Derived from GPUOpen-Effects/FidelityFX-FSR revision
 a21ffb8f6c13233ba336352bdff293894c706575, ffx_fsr1.h and ffx_a.h.

Copyright (c) 2021 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import ordinaryshade as osh


@osh.function
def easu_rcp(value: osh.f32) -> osh.f32:
    return osh.uint_bits_to_float(osh.u32(0x7EF07EBB) - osh.float_bits_to_uint(value))


@osh.function
def easu_luma(color: osh.vec3) -> osh.f32:
    return color.b * 0.5 + (color.r * 0.5 + color.g)


@osh.function
def easu_direction(
    weight: osh.f32, a: osh.f32, b: osh.f32, c: osh.f32, d: osh.f32, e: osh.f32
) -> osh.vec4:
    lx = easu_rcp(osh.maximum(osh.absolute(d - c), osh.absolute(c - b)))
    dx = d - b
    lx = osh.clamp(osh.absolute(dx) * lx, 0.0, 1.0)
    lx = lx * lx
    ly = easu_rcp(osh.maximum(osh.absolute(e - c), osh.absolute(c - a)))
    dy = e - a
    ly = osh.clamp(osh.absolute(dy) * ly, 0.0, 1.0)
    ly = ly * ly
    return osh.vec4(dx * weight, dy * weight, lx * weight, ly * weight)


@osh.function
def easu_tap(
    offset: osh.vec2,
    direction: osh.vec2,
    stretch: osh.vec2,
    lobe: osh.f32,
    clip: osh.f32,
    color: osh.vec3,
) -> osh.vec4:
    v = osh.vec2(
        offset.x * direction.x + offset.y * direction.y,
        offset.x * (-direction.y) + offset.y * direction.x,
    )
    v = v * stretch
    distance = osh.minimum(v.x * v.x + v.y * v.y, clip)
    base = 0.4 * distance - 1.0
    window = lobe * distance - 1.0
    base = base * base
    window = window * window
    base = 1.5625 * base - 0.5625
    weight = base * window
    return osh.vec4(color * weight, weight)


@osh.function
def easu_resolve(
    pp: osh.vec2,
    b: osh.vec3,
    c: osh.vec3,
    e: osh.vec3,
    f: osh.vec3,
    g: osh.vec3,
    h: osh.vec3,
    i: osh.vec3,
    j: osh.vec3,
    k: osh.vec3,
    l: osh.vec3,
    n: osh.vec3,
    o: osh.vec3,
) -> osh.vec3:
    # Footprint:    b c
    #            e f g h
    #            i j k l
    #              n o
    first = easu_direction(
        (1.0 - pp.x) * (1.0 - pp.y),
        easu_luma(b),
        easu_luma(e),
        easu_luma(f),
        easu_luma(g),
        easu_luma(j),
    )
    second = easu_direction(
        pp.x * (1.0 - pp.y),
        easu_luma(c),
        easu_luma(f),
        easu_luma(g),
        easu_luma(h),
        easu_luma(k),
    )
    third = easu_direction(
        (1.0 - pp.x) * pp.y,
        easu_luma(f),
        easu_luma(i),
        easu_luma(j),
        easu_luma(k),
        easu_luma(n),
    )
    fourth = easu_direction(
        pp.x * pp.y,
        easu_luma(g),
        easu_luma(j),
        easu_luma(k),
        easu_luma(l),
        easu_luma(o),
    )
    direction = ((first.xy + second.xy) + third.xy) + fourth.xy
    length = first.z + first.w
    length = length + second.z
    length = length + second.w
    length = length + third.z
    length = length + third.w
    length = length + fourth.z
    length = length + fourth.w
    squared = direction * direction
    magnitude = squared.x + squared.y
    reciprocal = osh.uint_bits_to_float(
        osh.u32(0x5F347D74) - (osh.float_bits_to_uint(magnitude) >> osh.u32(1))
    )
    if magnitude < (1.0 / 32768.0):
        reciprocal = 1.0
        direction.x = 1.0
    direction = direction * reciprocal
    length = length * 0.5
    length = length * length
    stretch = (direction.x * direction.x + direction.y * direction.y) * easu_rcp(
        osh.maximum(osh.absolute(direction.x), osh.absolute(direction.y))
    )
    anisotropy = osh.vec2(1.0 + (stretch - 1.0) * length, 1.0 - 0.5 * length)
    lobe = 0.5 + ((1.0 / 4.0 - 0.04) - 0.5) * length
    clip = easu_rcp(lobe)
    low = osh.minimum(osh.minimum(osh.minimum(f, g), j), k)
    high = osh.maximum(osh.maximum(osh.maximum(f, g), j), k)
    total = easu_tap(osh.vec2(0.0, -1.0) - pp, direction, anisotropy, lobe, clip, b)
    total = total + easu_tap(
        osh.vec2(1.0, -1.0) - pp, direction, anisotropy, lobe, clip, c
    )
    total = total + easu_tap(
        osh.vec2(-1.0, 1.0) - pp, direction, anisotropy, lobe, clip, i
    )
    total = total + easu_tap(
        osh.vec2(0.0, 1.0) - pp, direction, anisotropy, lobe, clip, j
    )
    total = total + easu_tap(
        osh.vec2(0.0, 0.0) - pp, direction, anisotropy, lobe, clip, f
    )
    total = total + easu_tap(
        osh.vec2(-1.0, 0.0) - pp, direction, anisotropy, lobe, clip, e
    )
    total = total + easu_tap(
        osh.vec2(1.0, 1.0) - pp, direction, anisotropy, lobe, clip, k
    )
    total = total + easu_tap(
        osh.vec2(2.0, 1.0) - pp, direction, anisotropy, lobe, clip, l
    )
    total = total + easu_tap(
        osh.vec2(2.0, 0.0) - pp, direction, anisotropy, lobe, clip, h
    )
    total = total + easu_tap(
        osh.vec2(1.0, 0.0) - pp, direction, anisotropy, lobe, clip, g
    )
    total = total + easu_tap(
        osh.vec2(1.0, 2.0) - pp, direction, anisotropy, lobe, clip, o
    )
    total = total + easu_tap(
        osh.vec2(0.0, 2.0) - pp, direction, anisotropy, lobe, clip, n
    )
    return osh.clamp(total.rgb * (1.0 / total.w), low, high)


EASU_HELPERS = (easu_rcp, easu_luma, easu_direction, easu_tap, easu_resolve)
