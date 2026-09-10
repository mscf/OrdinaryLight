"""Typed transport helpers. Generated artifacts must be rebuilt from this source."""
import ordinaryshade as osh
OL_PI = 3.141592653589793

@osh.structure
class MaterialData:
    base_roughness: osh.vec4
    emission_metallic: osh.vec4
    attenuation_transmission: osh.vec4
    ior_distance: osh.vec4
    texture_indices: osh.vec4
    texture_parameters: osh.vec4
    advanced0: osh.vec4
    advanced1: osh.vec4
    sheen_color: osh.vec4
    subsurface_color: osh.vec4
    advanced_texture_indices: osh.vec4
    optical: osh.vec4

@osh.structure
class PointLightData:
    position_type: osh.vec4
    direction_range: osh.vec4
    color_intensity: osh.vec4
    spot_parameters: osh.vec4

@osh.structure
class AreaLightData:
    a: osh.vec4
    b: osh.vec4
    c: osh.vec4
    emission_area: osh.vec4
    distribution: osh.vec4

@osh.structure
class MaterialEvaluation:
    base_color: osh.vec3
    emission: osh.vec3
    metallic: osh.f32
    roughness: osh.f32
    transmission: osh.f32
    ior: osh.f32
    attenuation_color: osh.vec3
    attenuation_distance: osh.f32
    custom_scattering: osh.f32
    weight: osh.vec3
    next_direction: osh.vec3
    event: osh.f32
    pdf: osh.f32

@osh.structure
class OrdinaryLightSurfaceSample:
    position: osh.vec4
    geometric_normal: osh.vec4
    shading_normal: osh.vec4
    incoming: osh.vec4
    identity: osh.uvec4
    media: osh.uvec4

@osh.structure
class OrdinaryLightDielectricEvent:
    direction: osh.vec3
    throughput: osh.f32
    reflected: osh.boolean
    tir: osh.boolean
    fresnel: osh.f32

@osh.function
def ordinarylightEnteringMedium(direction: osh.vec3, geometric_normal: osh.vec3) -> osh.boolean:
    return osh.dot(direction, geometric_normal) < 0.0

@osh.function
def ordinarylightDestinationMedium(surface: OrdinaryLightSurfaceSample, direction: osh.vec3) -> osh.u32:
    return surface.media.y if ordinarylightEnteringMedium(direction, surface.geometric_normal.xyz) else surface.media.x

@osh.function
def ordinarylightBeer(absorption: osh.vec3, distance: osh.f32) -> osh.vec3:
    return osh.exp(-absorption * distance)

@osh.function
def ordinarylightDielectric(direction: osh.vec3, incident_normal: osh.vec3, eta_i: osh.f32, eta_t: osh.f32, selector: osh.f32) -> OrdinaryLightDielectricEvent:
    event = OrdinaryLightDielectricEvent(osh.vec3(0), osh.f32(0.0), osh.boolean(False), osh.boolean(False), osh.f32(0.0))
    ci = osh.clamp(-osh.dot(direction, incident_normal), 0.0, 1.0)
    eta = eta_i / eta_t
    st2 = eta * eta * osh.maximum(0.0, 1.0 - ci * ci)
    event.tir = st2 >= 1.0
    ct = osh.sqrt(osh.maximum(0.0, 1.0 - st2))
    rs = (eta_i * ci - eta_t * ct) / osh.maximum(eta_i * ci + eta_t * ct, 1e-20)
    rp = (eta_t * ci - eta_i * ct) / osh.maximum(eta_t * ci + eta_i * ct, 1e-20)
    event.fresnel = 1.0 if event.tir else 0.0 if eta_i == eta_t else 0.5 * (rs * rs + rp * rp)
    event.reflected = event.tir or selector < event.fresnel
    event.direction = osh.normalize(osh.reflect(direction, incident_normal) if event.reflected else eta * direction + (eta * ci - ct) * incident_normal)
    event.throughput = 1.0 if event.reflected else eta * eta
    return event

@osh.function
def ordinarylightSampleSphere_pdf(parameters: osh.vec4, position: osh.vec3, normal: osh.vec3) -> osh.f32:
    if parameters.w <= 0.0:
        return 0.0
    return 1.0 / (12.566370614359172 * parameters.w * parameters.w)

@osh.function
def ordinarylightSampleSphere(parameters: osh.vec4, randoms: osh.vec3, position: osh.out(osh.vec3), normal: osh.out(osh.vec3), area_pdf: osh.out(osh.f32)) -> osh.u32:
    if parameters.w <= 0.0:
        return osh.u32(2)
    z = 1.0 - 2.0 * randoms.x
    phi = 6.283185307179586 * randoms.y
    r = osh.sqrt(osh.maximum(0.0, 1.0 - z * z))
    normal = osh.vec3(r * osh.cosine(phi), r * osh.sine(phi), z)
    position = parameters.xyz + parameters.w * normal
    area_pdf = ordinarylightSampleSphere_pdf(parameters, position, normal)
    return osh.u32(1)

@osh.function
def ordinarylightSdfSphere(origin: osh.vec3, direction: osh.vec3, t_min: osh.f32, t_max: osh.f32, parameters: osh.vec4, tolerance: osh.f32, max_steps: osh.u32, distance: osh.out(osh.f32), geometric_normal: osh.out(osh.vec3)) -> osh.u32:
    distance = t_min
    start_delta = origin + distance * direction - parameters.xyz
    start_value = osh.length(start_delta) - parameters.w
    origin_value = osh.length(origin - parameters.xyz) - parameters.w
    leaving_origin_root = (osh.absolute(origin_value) <= tolerance and osh.absolute(start_value) <= tolerance) and start_value * osh.dot(start_delta, direction) > 0.0
    step = osh.u32(0)
    while step < max_steps:
        delta = origin + distance * direction - parameters.xyz
        value = osh.length(delta) - parameters.w
        if leaving_origin_root and osh.absolute(value) > tolerance:
            leaving_origin_root = False
        if osh.absolute(value) <= tolerance and (not leaving_origin_root):
            geometric_normal = osh.normalize(delta)
            return osh.u32(1)
        distance = distance + (osh.maximum(osh.absolute(value), tolerance) if leaving_origin_root else osh.absolute(value))
        if distance > t_max + tolerance:
            return osh.u32(0)
        step = step + 1
    return osh.u32(2)

@osh.function
def olDistribution(cosine: osh.f32, roughness: osh.f32) -> osh.f32:
    if cosine <= 0.0:
        return 0.0
    a = osh.maximum(roughness * roughness, 0.0001)
    a2 = a * a
    q = 1.0 - cosine * cosine + a2 * cosine * cosine
    return a2 / (3.141592653589793 * q * q)

@osh.function
def olMask(cosine: osh.f32, roughness: osh.f32) -> osh.f32:
    cosine = osh.absolute(cosine)
    if cosine == 0.0:
        return 0.0
    a = osh.maximum(roughness * roughness, 0.0001)
    return 2.0 * cosine / (cosine + osh.sqrt(a * a + (1.0 - a * a) * cosine * cosine))

@osh.function
def olMicroNormal(normal: osh.vec3, roughness: osh.f32, randoms: osh.vec2) -> osh.vec3:
    a = osh.maximum(roughness * roughness, 0.0001)
    u = osh.minimum(randoms.x, 0.99999994)
    cosine = osh.sqrt((1.0 - u) / (1.0 - u + a * a * u))
    sine = osh.sqrt(osh.maximum(0.0, 1.0 - cosine * cosine))
    t = osh.normalize(osh.cross(normal, osh.vec3(0, 0, 1) if osh.absolute(normal.z) < 0.99 else osh.vec3(0, 1, 0)))
    return osh.normalize(normal * cosine + sine * (t * osh.cosine(2.0 * 3.141592653589793 * randoms.y) + osh.cross(normal, t) * osh.sine(2.0 * 3.141592653589793 * randoms.y)))

@osh.function
def olBsdf(material: MaterialEvaluation, kind: osh.u32, normal: osh.vec3, view: osh.vec3, outgoing: osh.vec3, eta_i: osh.f32, eta_t: osh.f32) -> osh.vec4:
    cv = osh.dot(normal, view)
    co = osh.dot(normal, outgoing)
    if (cv <= 0.0 or co == 0.0) or kind == osh.u32(3):
        return osh.vec4(0)
    if kind == osh.u32(0):
        return osh.vec4(material.base_color / 3.141592653589793, co / 3.141592653589793) if co > 0.0 else osh.vec4(0)
    if material.roughness == 0.0:
        if (kind != osh.u32(2) or co <= 0.0) or material.metallic == 1.0:
            return osh.vec4(0)
        f0 = osh.mix(osh.vec3(osh.power((material.ior - 1.0) / (material.ior + 1.0), 2.0)), material.base_color, material.metallic)
        fresnel = f0 + (osh.vec3(1) - f0) * osh.power(1.0 - cv, 5.0)
        exit_fresnel = f0 + (osh.vec3(1) - f0) * osh.power(1.0 - co, 5.0)
        return osh.vec4((osh.vec3(1) - fresnel) * (osh.vec3(1) - exit_fresnel) * (1.0 - material.metallic) * material.base_color / 3.141592653589793, 0.5 * co / 3.141592653589793)
    reflected = co > 0.0
    if kind != osh.u32(1) and (not reflected):
        return osh.vec4(0)
    eta = eta_t / eta_i
    sum = view + outgoing * (1.0 if reflected else eta)
    if osh.dot(sum, sum) < 1e-20:
        return osh.vec4(0)
    half_vector = osh.normalize(sum)
    if osh.dot(normal, half_vector) < 0.0:
        half_vector = -half_vector
    vh = osh.dot(view, half_vector)
    oh = osh.dot(outgoing, half_vector)
    nh = osh.dot(normal, half_vector)
    if vh <= 0.0 or oh * co <= 0.0:
        return osh.vec4(0)
    d = olDistribution(nh, material.roughness)
    g = olMask(cv, material.roughness) * olMask(co, material.roughness)
    normal_pdf = d * nh
    if kind == osh.u32(1):
        fresnel = ordinarylightDielectric(-view, half_vector, eta_i, eta_t, 0.0).fresnel
        if reflected:
            return osh.vec4(osh.vec3(fresnel * d * g / (4.0 * cv * co)), normal_pdf * fresnel / (4.0 * vh))
        denominator = oh + vh / eta
        denominator = denominator * denominator
        if denominator < 1e-30:
            return osh.vec4(0)
        pdf = normal_pdf * (1.0 - fresnel) * osh.absolute(oh) / denominator
        value = (1.0 - fresnel) * d * g * osh.absolute(vh * oh / (cv * co * denominator)) / (eta * eta)
        return osh.vec4(osh.vec3(value), pdf)
    f0 = osh.mix(osh.vec3(osh.power((material.ior - 1.0) / (material.ior + 1.0), 2.0)), material.base_color, material.metallic)
    fresnel = f0 + (osh.vec3(1) - f0) * osh.power(1.0 - vh, 5.0)
    value = fresnel * d * g / (4.0 * cv * co) + (osh.vec3(1) - fresnel) * (1.0 - material.metallic) * material.base_color / 3.141592653589793
    probability = 1.0 if material.metallic == 1.0 else 0.5
    return osh.vec4(value, osh.mix(co / 3.141592653589793, normal_pdf / (4.0 * vh), probability))

@osh.function
def olSampleBsdf(material: MaterialEvaluation, kind: osh.u32, normal: osh.vec3, view: osh.vec3, eta_i: osh.f32, eta_t: osh.f32, randoms: osh.vec3, weight: osh.out(osh.vec3), pdf: osh.out(osh.f32), reflected: osh.out(osh.boolean), tir: osh.out(osh.boolean)) -> osh.vec3:
    weight = osh.vec3(0)
    pdf = 0.0
    reflected = True
    tir = False
    outgoing = osh.vec3(0)
    if kind == osh.u32(0):
        outgoing = cosineHemisphere(normal, randoms.x, randoms.y)
    elif kind == osh.u32(1):
        if eta_i == eta_t:
            weight = osh.vec3(1)
            pdf = 1.0
            reflected = False
            return -view
        micro = normal if material.roughness == 0.0 else olMicroNormal(normal, material.roughness, randoms.xy)
        if osh.dot(view, micro) <= 0.0:
            return normal
        event = ordinarylightDielectric(-view, micro, eta_i, eta_t, randoms.z)
        outgoing = event.direction
        reflected = event.reflected
        tir = event.tir
        if (osh.dot(outgoing, normal) > 0.0) != reflected:
            return outgoing
        if material.roughness == 0.0 or eta_i == eta_t:
            weight = osh.vec3(event.throughput)
            pdf = 1.0
            return outgoing
    elif kind == osh.u32(2):
        probability = 1.0 if material.metallic == 1.0 else 0.5
        if material.roughness == 0.0:
            f0 = osh.mix(osh.vec3(osh.power((material.ior - 1.0) / (material.ior + 1.0), 2.0)), material.base_color, material.metallic)
            fresnel = f0 + (osh.vec3(1) - f0) * osh.power(1.0 - osh.dot(normal, view), 5.0)
            if randoms.z < probability:
                weight = fresnel / probability
                pdf = 1.0
                return osh.reflect(-view, normal)
            outgoing = cosineHemisphere(normal, randoms.x, randoms.y)
            exit_fresnel = f0 + (osh.vec3(1) - f0) * osh.power(1.0 - osh.dot(normal, outgoing), 5.0)
            weight = (osh.vec3(1) - fresnel) * (osh.vec3(1) - exit_fresnel) * (1.0 - material.metallic) * material.base_color / (1.0 - probability)
            pdf = (1.0 - probability) * osh.maximum(osh.dot(normal, outgoing), 0.0) / 3.141592653589793
            return outgoing
        outgoing = osh.reflect(-view, olMicroNormal(normal, material.roughness, randoms.xy)) if randoms.z < probability else cosineHemisphere(normal, randoms.x, randoms.y)
    else:
        return normal
    value = olBsdf(material, kind, normal, view, outgoing, eta_i, eta_t)
    pdf = value.a
    if pdf > 0.0:
        weight = value.rgb * osh.absolute(osh.dot(normal, outgoing)) / pdf
    return outgoing

@osh.structure
class CustomRecord:
    lower: osh.vec4
    upper: osh.vec4
    parameters: osh.vec4
    metadata: osh.uvec4

@osh.structure
class TransportMaterialRecord:
    albedo_kind: osh.vec4
    emission: osh.vec4
    optics: osh.vec4

@osh.structure
class OrdinaryLightHit:
    position_distance: osh.vec4
    geometric_normal: osh.vec4
    shading_normal: osh.vec4
    identity: osh.uvec4
    boundary: osh.uvec4

@osh.structure
class OrdinaryLightCustomHit:
    distance: osh.f32
    geometric_normal: osh.vec3
    flags: osh.u32
    material: osh.u32
    boundary: osh.u32
    identity: osh.u32
    uv: osh.vec2
    shading_normal: osh.vec3

@osh.structure
class OrdinaryLightEmitterSample:
    position: osh.vec3
    normal: osh.vec3
    area_pdf: osh.f32
    kind: osh.u32
    primitive: osh.u32

@osh.function
def ordinarylightValidAreaPdf(pdf: osh.f32) -> osh.boolean:
    return (not osh.is_nan(pdf) and (not osh.is_inf(pdf))) and pdf >= 0.0

@osh.function
def ordinarylightTriangleArea(index: osh.u32) -> osh.f32:
    a = transport_vertices[index * osh.u32(3)].xyz
    b = transport_vertices[index * osh.u32(3) + osh.u32(1)].xyz
    c = transport_vertices[index * osh.u32(3) + osh.u32(2)].xyz
    return 0.5 * osh.length(osh.cross(b - a, c - a))

@osh.function
def ordinarylightSampleSurface(randoms: osh.vec4, selected: osh.out(OrdinaryLightEmitterSample)) -> osh.u32:
    randoms = osh.minimum(randoms, osh.vec4(0.99999994))
    slot = osh.minimum(osh.u32(randoms.x * osh.f32(OL_SURFACE_SLOT_COUNT)), OL_SURFACE_SLOT_COUNT - osh.u32(1))
    status = osh.u32(0)
    if slot < OL_TRIANGLE_COUNT:
        a = transport_vertices[slot * osh.u32(3)].xyz
        b = transport_vertices[slot * osh.u32(3) + osh.u32(1)].xyz
        c = transport_vertices[slot * osh.u32(3) + osh.u32(2)].xyz
        area = ordinarylightTriangleArea(slot)
        if area <= 0.0:
            return osh.u32(0)
        root = osh.sqrt(randoms.y)
        selected.position = a * (1.0 - root) + b * (root * (1.0 - randoms.z)) + c * (root * randoms.z)
        selected.normal = osh.normalize(osh.cross(b - a, c - a))
        selected.area_pdf = 1.0 / area
        selected.kind = osh.u32(1)
        selected.primitive = slot
        status = osh.u32(1)
    else:
        index = slot - OL_TRIANGLE_COUNT
        geometry = custom_geometry[index]
        if geometry.metadata.x == osh.u32(4294967295):
            return osh.u32(0)
        selected.kind = osh.u32(2)
        selected.primitive = index
        status = ordinarylightCustomSurfaceSample(geometry.metadata.x, geometry.parameters, randoms.yzw, selected.position, selected.normal, selected.area_pdf)
        if status != osh.u32(1):
            return osh.u32(2) if status > osh.u32(1) else osh.u32(0)
        if osh.any_value(selected.position < geometry.lower.xyz) or osh.any_value(selected.position > geometry.upper.xyz):
            return osh.u32(2)
        reverse_pdf = ordinarylightCustomSurfacePdf(geometry.metadata.x, geometry.parameters, selected.position, selected.normal)
        if not ordinarylightValidAreaPdf(reverse_pdf) or osh.absolute(reverse_pdf - selected.area_pdf) > 0.0001 * osh.maximum(reverse_pdf, selected.area_pdf):
            return osh.u32(2)
    if (((((osh.any_value(osh.is_nan(selected.position)) or osh.any_value(osh.is_inf(selected.position))) or osh.any_value(osh.is_nan(selected.normal))) or osh.any_value(osh.is_inf(selected.normal))) or osh.absolute(osh.dot(selected.normal, selected.normal) - 1.0) > 0.001) or not ordinarylightValidAreaPdf(selected.area_pdf)) or selected.area_pdf == 0.0:
        return osh.u32(2)
    return status

@osh.function
def ordinarylightSurfaceLightPdf(origin: osh.vec3, hit: OrdinaryLightHit, status: osh.inout(osh.u32)) -> osh.f32:
    area_pdf = 0.0
    if hit.identity.x == osh.u32(1):
        area = ordinarylightTriangleArea(hit.identity.y)
        if area > 0.0:
            area_pdf = 1.0 / area
    elif hit.identity.x == osh.u32(2):
        geometry = custom_geometry[hit.identity.y]
        area_pdf = ordinarylightCustomSurfacePdf(geometry.metadata.x, geometry.parameters, hit.position_distance.xyz, hit.geometric_normal.xyz)
    if not ordinarylightValidAreaPdf(area_pdf):
        status = status | osh.u32(1)
        return 0.0
    offset = hit.position_distance.xyz - origin
    distance_squared = osh.dot(offset, offset)
    if area_pdf == 0.0 or distance_squared == 0.0:
        return 0.0
    cosine = osh.absolute(osh.dot(hit.geometric_normal.xyz, -osh.normalize(offset)))
    if cosine <= 0.0:
        return 0.0
    pdf = area_pdf * distance_squared / (osh.f32(OL_SURFACE_SLOT_COUNT) * cosine)
    if not ordinarylightValidAreaPdf(pdf):
        status = status | osh.u32(1)
        return 0.0
    return pdf

@osh.function
def ordinarylightValidTransportMaterial(evaluated: MaterialEvaluation, material: TransportMaterialRecord, hit: OrdinaryLightHit) -> osh.boolean:
    return not (((((((((((((((osh.is_nan(evaluated.metallic) or osh.is_inf(evaluated.metallic)) or osh.is_nan(evaluated.roughness)) or osh.is_inf(evaluated.roughness)) or osh.is_nan(evaluated.ior)) or osh.is_inf(evaluated.ior)) or evaluated.ior <= 0.0) or (material.albedo_kind.w == 0.0 and (evaluated.metallic != 0.0 or evaluated.roughness != 0.0))) or (material.albedo_kind.w == 1.0 and (evaluated.metallic != 0.0 or evaluated.ior != optical_media[hit.boundary.z].a))) or osh.any_value(evaluated.attenuation_color != osh.vec3(1))) or evaluated.attenuation_distance != 1e+30) or evaluated.transmission != osh.f32(material.albedo_kind.w == 1.0)) or osh.any_value(osh.is_nan(evaluated.base_color))) or osh.any_value(osh.is_inf(evaluated.base_color))) or osh.any_value(osh.is_nan(evaluated.emission))) or osh.any_value(osh.is_inf(evaluated.emission)))

@osh.function
def ordinarylightEmissiveMisWeight(first_pdf: osh.f32, second_pdf: osh.f32) -> osh.f32:
    scale = osh.maximum(first_pdf, second_pdf)
    if scale <= 0.0:
        return 0.0
    a = first_pdf / scale
    b = second_pdf / scale
    return a * a / (a * a + b * b)


# These declarations link application-provided surface samplers; their dispatch
# implementations are emitted by the scene-specific transport shader builder.
@osh.external
def ordinarylightCustomSurfaceSample(program: osh.u32, parameters: osh.vec4, randoms: osh.vec3,
                                     position: osh.out(osh.vec3), normal: osh.out(osh.vec3),
                                     area_pdf: osh.out(osh.f32)) -> osh.u32:
    pass


@osh.external
def ordinarylightCustomSurfacePdf(program: osh.u32, parameters: osh.vec4,
                                  position: osh.vec3, normal: osh.vec3) -> osh.f32:
    pass
