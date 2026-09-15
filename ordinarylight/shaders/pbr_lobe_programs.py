"""Experimental separable continuation estimator, authored in OrdinaryShade.

Matches the base isotropic PBR BRDF used by samplePbr. This is not an advanced
material evaluator or an optical-boundary sampler. Direct-light MIS must use
matching per-lobe densities before this can replace native continuation.

Callers supply finite material parameters in their physical ranges, normalized
normal/incoming vectors, and independent random values in [0, 1).
"""
import ordinaryshade as osh


@osh.structure
class PbrLobeSample:
    direction_pdf: osh.vec4
    throughput_lobe: osh.vec4


@osh.function
def pbrLobeProbability(base: osh.vec3, metallic: osh.f32) -> osh.f32:
    f0 = osh.mix(osh.vec3(0.04), base, metallic)
    return osh.clamp(osh.maximum(f0.r, osh.maximum(f0.g, f0.b)), 0.1, 0.9)


@osh.function
def pbrLobeValue(base: osh.vec3, roughness: osh.f32, metallic: osh.f32,
                 normal: osh.vec3, view: osh.vec3, outgoing: osh.vec3,
                 specular: osh.boolean) -> osh.vec3:
    nv = osh.maximum(osh.dot(normal, view), 0.0)
    nl = osh.maximum(osh.dot(normal, outgoing), 0.0)
    if nv <= 0.0 or nl <= 0.0:
        return osh.vec3(0.0)
    half_vector = osh.normalize(view + outgoing)
    nh = osh.maximum(osh.dot(normal, half_vector), 0.0)
    vh = osh.maximum(osh.dot(view, half_vector), 0.0)
    f0 = osh.mix(osh.vec3(0.04), base, metallic)
    fresnel = f0 + (osh.vec3(1.0) - f0) * osh.power(1.0 - osh.clamp(vh, 0.0, 1.0), 5.0)
    if not specular:
        return (osh.vec3(1.0) - fresnel) * (1.0 - metallic) * base / 3.14159265359
    alpha = osh.maximum(roughness * roughness, 0.0009)
    a2 = alpha * alpha
    denominator = nh * nh * (a2 - 1.0) + 1.0
    # Preserve the maintained BRDF's regularization. This is intentionally
    # separate from the normalized distribution used by the sampler PDF.
    distribution = a2 / osh.maximum(3.14159265359 * denominator * denominator, 0.000001)
    gv = 2.0 * nv / osh.maximum(nv + osh.sqrt(a2 + (1.0 - a2) * nv * nv), 0.000001)
    gl = 2.0 * nl / osh.maximum(nl + osh.sqrt(a2 + (1.0 - a2) * nl * nl), 0.000001)
    return fresnel * distribution * (gv * gl) / osh.maximum(4.0 * nv * nl, 0.000001)


@osh.function
def pbrLobePdf(roughness: osh.f32, normal: osh.vec3, view: osh.vec3,
              outgoing: osh.vec3, specular: osh.boolean) -> osh.f32:
    nl = osh.maximum(osh.dot(normal, outgoing), 0.0)
    if nl <= 0.0:
        return 0.0
    if not specular:
        return nl / 3.14159265359
    half_vector = osh.normalize(view + outgoing)
    nh = osh.maximum(osh.dot(normal, half_vector), 0.0)
    vh = osh.dot(view, half_vector)
    if vh <= 0.0:
        return 0.0
    alpha = osh.maximum(roughness * roughness, 0.0009)
    a2 = alpha * alpha
    denominator = (1.0 - nh * nh) + a2 * nh * nh
    distribution = a2 / osh.maximum(3.14159265359 * denominator * denominator, 1e-30)
    return distribution * nh / (4.0 * vh)


@osh.function
def samplePbrLobe(base: osh.vec3, roughness: osh.f32, metallic: osh.f32,
                  occlusion: osh.f32, normal: osh.vec3, incoming: osh.vec3,
                  randoms: osh.vec3, allow_diffuse: osh.boolean) -> PbrLobeSample:
    probability = pbrLobeProbability(base, metallic)
    specular = randoms.x < probability
    lobe = 1.0 if specular else 0.0
    result = PbrLobeSample(osh.vec4(normal, 0.0), osh.vec4(0.0, 0.0, 0.0, lobe))
    if not specular and not allow_diffuse:
        return result
    tangent = osh.normalize(osh.cross(normal, osh.vec3(0.0, 0.0, 1.0)) if osh.absolute(normal.z) < 0.999 else osh.cross(normal, osh.vec3(0.0, 1.0, 0.0)))
    bitangent = osh.cross(normal, tangent)
    phi = 6.28318530718 * randoms.z
    cosine = osh.sqrt(1.0 - randoms.y)
    if specular:
        alpha = osh.maximum(roughness * roughness, 0.0009)
        cosine = osh.sqrt((1.0 - randoms.y) / osh.maximum((1.0 - randoms.y) + alpha * alpha * randoms.y, 1e-30))
    sine = osh.sqrt(osh.maximum(0.0, 1.0 - cosine * cosine))
    sampled = osh.normalize(tangent * (sine * osh.cosine(phi)) + bitangent * (sine * osh.sine(phi)) + normal * cosine)
    outgoing = sampled
    if specular:
        outgoing = osh.normalize(incoming - 2.0 * osh.dot(sampled, incoming) * sampled)
    view = -incoming
    # A below-surface GGX draw has zero contribution. Replacing it with a cosine
    # draw while retaining the GGX PDF would change the sampling distribution.
    if osh.dot(normal, view) <= 0.0 or osh.dot(normal, outgoing) <= 0.0:
        return result
    conditional_pdf = pbrLobePdf(roughness, normal, view, outgoing, specular)
    pdf = conditional_pdf * (probability if specular else 1.0 - probability)
    if pdf <= 0.0:
        return result
    value = pbrLobeValue(base, roughness, metallic, normal, view, outgoing, specular)
    weight = value * (osh.dot(normal, outgoing) / pdf) * osh.mix(occlusion, 1.0, metallic)
    result.direction_pdf = osh.vec4(outgoing, pdf)
    result.throughput_lobe = osh.vec4(weight, lobe)
    return result


@osh.function
def pbrLobeMisWeights(light_density: osh.f32, continuation_density: osh.f32) -> osh.vec2:
    """Complementary power-heuristic weights for one lobe, light then BSDF.

    Densities include technique/event probabilities and sample counts. Call
    separately for diffuse and specular; a single combined-BRDF weight is not
    generally valid after separating their sampling distributions.
    """
    densities = osh.maximum(osh.vec2(light_density, continuation_density), osh.vec2(0.0))
    scale = osh.maximum(densities.x, densities.y)
    if scale <= 0.0:
        return osh.vec2(0.0)
    normalized = densities / scale
    squared = normalized * normalized
    return squared / (squared.x + squared.y)
