"""Typed material attribute interpolation and staged evaluation support."""
import ordinaryshade as osh
from .transport_programs import MaterialData, MaterialEvaluation


@osh.function
def waveFresnelSchlick(cosine: osh.f32, ior_from: osh.f32, ior_to: osh.f32) -> osh.f32:
    ratio = (ior_from - ior_to) / osh.maximum(ior_from + ior_to, 0.000001)
    r0 = ratio * ratio
    one_minus_cosine = 1.0 - osh.clamp(cosine, 0.0, 1.0)
    return r0 + (1.0 - r0) * one_minus_cosine * one_minus_cosine * one_minus_cosine * one_minus_cosine * one_minus_cosine


@osh.function
def waveVertexAttribute4(slot: osh.u32) -> osh.vec4:
    base = wave_attribute_primitive * (osh.u32(3) * WAVE_VERTEX_CHANNEL_COUNT) + slot
    return (wave_custom_attributes[base] * wave_attribute_weights.x
            + wave_custom_attributes[base + WAVE_VERTEX_CHANNEL_COUNT] * wave_attribute_weights.y
            + wave_custom_attributes[base + osh.u32(2) * WAVE_VERTEX_CHANNEL_COUNT] * wave_attribute_weights.z)


@osh.function
def waveVertexAttribute1(slot: osh.u32) -> osh.f32:
    return waveVertexAttribute4(slot).x


@osh.function
def waveVertexAttribute2(slot: osh.u32) -> osh.vec2:
    return waveVertexAttribute4(slot).xy


@osh.function
def waveVertexAttribute3(slot: osh.u32) -> osh.vec3:
    return waveVertexAttribute4(slot).xyz


@osh.external
def evaluateMaterial(material: osh.inout(MaterialData), normal: osh.inout(osh.vec3), uv: osh.vec2,
                     direction: osh.vec3, entering: osh.boolean, random_u: osh.f32, random_v: osh.f32,
                     bounce_index: osh.f32, current_ior: osh.f32, exterior_ior: osh.f32) -> MaterialEvaluation:
    pass


@osh.function
def waveApplyMaterialProgram(material: osh.inout(MaterialData), normal: osh.vec3, uv: osh.vec2,
                             direction: osh.vec3, entering: osh.boolean, primitive: osh.u32,
                             weights: osh.vec3, bounce_index: osh.f32) -> MaterialEvaluation:
    wave_attribute_primitive = primitive
    wave_attribute_weights = weights
    evaluated = evaluateMaterial(material, normal, uv, direction, entering, 0.5, 0.5,
                                 bounce_index, 1.0, material.ior_distance.x)
    material.base_roughness = osh.vec4(evaluated.base_color, evaluated.roughness)
    material.emission_metallic = osh.vec4(evaluated.emission, evaluated.metallic)
    material.attenuation_transmission = osh.vec4(evaluated.attenuation_color, evaluated.transmission)
    material.ior_distance.xy = osh.vec2(evaluated.ior, evaluated.attenuation_distance)
    return evaluated


@osh.function
def waveSetMaterialAttributes(primitive: osh.u32, weights: osh.vec3) -> osh.void:
    wave_attribute_primitive = primitive
    wave_attribute_weights = weights
