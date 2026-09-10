"""Typed sampling and microfacet helpers shared by transport backends."""
import ordinaryshade as osh


@osh.function
def randomFloat(state: osh.inout(osh.u32)) -> osh.f32:
    state = state * osh.u32(747796405) + osh.u32(2891336453)
    word = ((state >> ((state >> osh.u32(28)) + osh.u32(4))) ^ state) * osh.u32(277803737)
    word = (word >> osh.u32(22)) ^ word
    return osh.f32(word) * (1.0 / 4294967296.0)


@osh.function
def secondaryNeeHash(value: osh.u32) -> osh.u32:
    value = value ^ (value >> osh.u32(16))
    value = value * osh.u32(0x7feb352d)
    value = value ^ (value >> osh.u32(15))
    value = value * osh.u32(0x846ca68b)
    return value ^ (value >> osh.u32(16))


@osh.function
def selectSecondaryNee(probability: osh.f32, pixel_index: osh.u32,
                       frame_sample: osh.u32, bounce: osh.u32) -> osh.boolean:
    if probability >= 0.999999:
        return True
    # Digitally shifted base-2 sequence; no consumption of the BSDF RNG.
    frame_index = frame_sample >> osh.u32(8)
    sample_index = frame_sample & osh.u32(255)
    scramble = secondaryNeeHash(pixel_index ^ secondaryNeeHash(bounce + osh.u32(1))
                                ^ secondaryNeeHash(sample_index + osh.u32(1)))
    sequence = osh.bitfield_reverse(frame_index) ^ scramble
    selector = (osh.f32(sequence) + 0.5) * (1.0 / 4294967296.0)
    return selector < probability


@osh.function
def cosineHemisphere(normal: osh.vec3, random_u: osh.f32, random_v: osh.f32) -> osh.vec3:
    radius = osh.sqrt(random_u)
    phi = 6.28318530718 * random_v
    tangent = osh.normalize(osh.select(osh.absolute(normal.z) < 0.999,
        osh.cross(normal, osh.vec3(0.0, 0.0, 1.0)),
        osh.cross(normal, osh.vec3(0.0, 1.0, 0.0))))
    bitangent = osh.cross(normal, tangent)
    return osh.normalize(tangent * radius * osh.cosine(phi) + bitangent * radius * osh.sine(phi)
                         + normal * osh.sqrt(osh.maximum(0.0, 1.0 - random_u)))


@osh.function
def powerHeuristic(first_pdf: osh.f32, second_pdf: osh.f32) -> osh.f32:
    a = first_pdf * first_pdf
    b = second_pdf * second_pdf
    return a / osh.maximum(a + b, 0.000001)


@osh.function
def pbrFresnel(f0: osh.vec3, cosine: osh.f32) -> osh.vec3:
    return f0 + (osh.vec3(1.0) - f0) * osh.power(1.0 - osh.clamp(cosine, 0.0, 1.0), 5.0)


@osh.function
def ggxDistribution(n_dot_h: osh.f32, roughness: osh.f32) -> osh.f32:
    alpha = osh.maximum(roughness * roughness, 0.0009)
    alpha_squared = alpha * alpha
    denominator = n_dot_h * n_dot_h * (alpha_squared - 1.0) + 1.0
    return alpha_squared / osh.maximum(3.14159265359 * denominator * denominator, 0.000001)


@osh.function
def ggxSmithComponent(n_dot_direction: osh.f32, roughness: osh.f32) -> osh.f32:
    alpha = osh.maximum(roughness * roughness, 0.0009)
    alpha_squared = alpha * alpha
    return 2.0 * n_dot_direction / osh.maximum(n_dot_direction + osh.sqrt(alpha_squared
        + (1.0 - alpha_squared) * n_dot_direction * n_dot_direction), 0.000001)


@osh.compute()
def sampling_module():
    pass


HELPERS = (randomFloat, secondaryNeeHash, selectSecondaryNee, cosineHemisphere,
           powerHeuristic, pbrFresnel, ggxDistribution, ggxSmithComponent)
