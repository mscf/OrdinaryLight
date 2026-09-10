"""Typed transport helpers. Generated artifacts must be rebuilt from this source."""
import ordinaryshade as osh
from .transport_programs import MaterialData

@osh.structure
class TextureBindingData:
    texture_rotation: osh.vec4
    offset_scale: osh.vec4

@osh.function
def wrapTextureCoordinate(value: osh.f32, mode: osh.u32) -> osh.f32:
    if mode == osh.u32(1):
        return osh.clamp(value, 0.0, 0.99999994)
    if mode == osh.u32(2):
        period = osh.modulo(value, 2.0)
        if period < 0.0:
            period = period + 2.0
        return period if period <= 1.0 else 2.0 - period
    return osh.fraction(value)

@osh.function
def wrapTextureIndex(value: osh.i32, size: osh.i32, mode: osh.u32) -> osh.i32:
    if mode == osh.u32(1):
        return osh.clamp(value, 0, size - 1)
    period = size * 2 if mode == osh.u32(2) else size
    wrapped = value % period
    if wrapped < 0:
        wrapped = wrapped + period
    return period - 1 - wrapped if mode == osh.u32(2) and wrapped >= size else wrapped

@osh.function
def decodeTextureTexel(packed: osh.u32, srgb: osh.boolean) -> osh.vec4:
    value = osh.unpack_unorm4x8(packed)
    if not srgb:
        return value
    value.r = SRGB_TO_LINEAR[packed & osh.u32(255)]
    value.g = SRGB_TO_LINEAR[packed >> osh.u32(8) & osh.u32(255)]
    value.b = SRGB_TO_LINEAR[packed >> osh.u32(16) & osh.u32(255)]
    return value

@osh.function
def fetchTextureTexel(offset: osh.u32, size: osh.ivec2, coordinate: osh.ivec2, wrap: osh.uvec2, srgb: osh.boolean) -> osh.vec4:
    x = wrapTextureIndex(coordinate.x, size.x, wrap.x)
    y = wrapTextureIndex(coordinate.y, size.y, wrap.y)
    return decodeTextureTexel(texture_words[offset + osh.u32(y * size.x + x)], srgb)

@osh.function
def textureMipOffset(base_offset: osh.u32, base_size: osh.ivec2, level: osh.u32) -> osh.u32:
    offset = base_offset
    size = base_size
    current = osh.u32(0)
    while current < level:
        offset = offset + osh.u32(size.x * size.y)
        size = osh.maximum((size + 1) / 2, osh.ivec2(1))
        current = current + 1
    return offset

@osh.function
def sampleTextureLevel(base_offset: osh.u32, base_size: osh.ivec2, level: osh.u32, uv: osh.vec2, wrap: osh.uvec2, srgb: osh.boolean, linear_filter: osh.boolean) -> osh.vec4:
    size = osh.maximum(base_size + (osh.ivec2(1) << osh.i32(level)) - 1 >> osh.i32(level), osh.ivec2(1))
    offset = textureMipOffset(base_offset, base_size, level)
    wrapped_uv = osh.vec2(wrapTextureCoordinate(uv.x, wrap.x), wrapTextureCoordinate(uv.y, wrap.y))
    if not linear_filter:
        coordinate = osh.ivec2(osh.floor(wrapped_uv * osh.vec2(size)))
        return fetchTextureTexel(offset, size, coordinate, wrap, srgb)
    texel = wrapped_uv * osh.vec2(size) - 0.5
    base = osh.ivec2(osh.floor(texel))
    fraction = osh.fraction(texel)
    top = osh.mix(fetchTextureTexel(offset, size, base, wrap, srgb), fetchTextureTexel(offset, size, base + osh.ivec2(1, 0), wrap, srgb), fraction.x)
    bottom = osh.mix(fetchTextureTexel(offset, size, base + osh.ivec2(0, 1), wrap, srgb), fetchTextureTexel(offset, size, base + osh.ivec2(1, 1), wrap, srgb), fraction.x)
    return osh.mix(top, bottom, fraction.y)

@osh.function
def sampleSceneTexture(texture_index: osh.i32, uv: osh.vec2) -> osh.vec4:
    if texture_index < 0 or osh.u32(texture_index) >= texture_words[0]:
        return osh.vec4(0.0)
    if osh.specialization('WAVE_NATIVE_TEXTURES'):
        descriptor_index = texture_index * 2 + 1
        return native_textures.sample_lod(descriptor_index, uv, 0.0)
    else:
        metadata = osh.u32(1) + osh.u32(texture_index) * osh.u32(8)
        offset = texture_words[metadata + osh.u32(1)]
        size = osh.ivec2(texture_words[metadata + osh.u32(2)], texture_words[metadata + osh.u32(3)])
        flags = texture_words[metadata + osh.u32(4)]
        wrap = osh.uvec2(flags & osh.u32(3), flags >> osh.u32(2) & osh.u32(3))
        return sampleTextureLevel(offset, size, osh.u32(0), uv, wrap, False, flags & osh.u32(16) != osh.u32(0))

@osh.function
def sampleMaterialTexture(binding_index_value: osh.f32, uv0: osh.vec2, uv1: osh.vec2, srgb: osh.boolean, uv0_footprint: osh.f32, uv1_footprint: osh.f32) -> osh.vec4:
    binding_index = osh.i32(binding_index_value)
    if binding_index < 0:
        return osh.vec4(1.0)
    binding = texture_bindings[binding_index]
    texture_index = osh.i32(binding.texture_rotation.x)
    if texture_index < 0 or osh.u32(texture_index) >= texture_words[0]:
        return osh.vec4(1.0)
    if osh.specialization('WAVE_WORK_COUNTERS'):
        profileWork(osh.u32(2), osh.u32(1))
    use_uv1 = binding.texture_rotation.w > 0.5
    uv = uv1 if use_uv1 else uv0
    uv_footprint = uv1_footprint if use_uv1 else uv0_footprint
    cosine = binding.texture_rotation.y
    sine = binding.texture_rotation.z
    scaled_uv = uv * binding.offset_scale.zw
    uv = binding.offset_scale.xy + osh.vec2(cosine * scaled_uv.x - sine * scaled_uv.y, sine * scaled_uv.x + cosine * scaled_uv.y)
    uv_footprint = uv_footprint * osh.maximum(osh.absolute(binding.offset_scale.z), osh.absolute(binding.offset_scale.w))
    if osh.specialization('WAVE_NATIVE_TEXTURES'):
        descriptor_index = texture_index * 2 + (0 if srgb else 1)
        size = native_textures.size(descriptor_index, 0)
        level_count = native_textures.levels(descriptor_index)
        lod = osh.clamp(osh.log2(osh.maximum(uv_footprint * osh.f32(osh.maximum(size.x, size.y)), 1.0)), 0.0, osh.f32(osh.maximum(level_count, 1) - 1))
        return native_textures.sample_lod(descriptor_index, uv, lod)
    else:
        metadata = osh.u32(1) + osh.u32(texture_index) * osh.u32(8)
        offset = texture_words[metadata + (osh.u32(0) if srgb else osh.u32(1))]
        size = osh.ivec2(texture_words[metadata + osh.u32(2)], texture_words[metadata + osh.u32(3)])
        flags = texture_words[metadata + osh.u32(4)]
        level_count = texture_words[metadata + osh.u32(5)]
        wrap = osh.uvec2(flags & osh.u32(3), flags >> osh.u32(2) & osh.u32(3))
        lod = osh.clamp(osh.log2(osh.maximum(uv_footprint * osh.f32(osh.maximum(size.x, size.y)), 1.0)), 0.0, osh.f32(osh.maximum(level_count, osh.u32(1)) - osh.u32(1)))
        lower = osh.u32(osh.floor(lod))
        linear_filter = flags & osh.u32(16) != osh.u32(0)
        first = sampleTextureLevel(offset, size, lower, uv, wrap, srgb, linear_filter)
        if not linear_filter or lower + osh.u32(1) >= level_count:
            return first
        second = sampleTextureLevel(offset, size, lower + osh.u32(1), uv, wrap, srgb, True)
        return osh.mix(first, second, osh.fraction(lod))

@osh.function
def triangleUvDensity(a: osh.vec3, b: osh.vec3, c: osh.vec3, uv_a: osh.vec2, uv_b: osh.vec2, uv_c: osh.vec2) -> osh.f32:
    first_uv = uv_b - uv_a
    second_uv = uv_c - uv_a
    uv_area = osh.absolute(first_uv.x * second_uv.y - first_uv.y * second_uv.x)
    world_area = osh.length(osh.cross(b - a, c - a))
    return osh.sqrt(uv_area / osh.maximum(world_area, 1e-08))

@osh.function
def materialHasTextures(material: MaterialData) -> osh.boolean:
    if osh.specialization('WAVE_UNTEXTURED_SCENE'):
        return False
    else:
        return (((osh.any_value(material.texture_indices >= osh.vec4(0.0)) or material.texture_parameters.y >= 0.0) or material.texture_parameters.w >= 0.0) or osh.any_value(material.advanced_texture_indices >= osh.vec4(0.0))) or material.optical.x >= 0.0

@osh.function
def applyMaterialTextures(material: osh.inout(MaterialData), uv0: osh.vec2, uv1: osh.vec2, uv0_footprint: osh.f32, uv1_footprint: osh.f32) -> osh.void:
    if osh.specialization('WAVE_UNTEXTURED_SCENE'):
        material.texture_parameters.w = 1.0
        return
    else:
        if osh.specialization('WAVE_WORK_COUNTERS'):
            if material.texture_parameters.w >= 0.0:
                profileWork(osh.u32(5), osh.u32(1))
        transmission = sampleMaterialTexture(material.texture_parameters.w, uv0, uv1, False, uv0_footprint, uv1_footprint).r
        material.attenuation_transmission.a = ordinarylight_texture_apply_scalar(material.attenuation_transmission.a, transmission)
        if osh.specialization('WAVE_WORK_COUNTERS'):
            if material.texture_indices.x >= 0.0:
                profileWork(osh.u32(6), osh.u32(1))
        base_color_sample = sampleMaterialTexture(material.texture_indices.x, uv0, uv1, True, uv0_footprint, uv1_footprint).rgb
        material.base_roughness.rgb = ordinarylight_texture_apply_rgb(material.base_roughness.rgb, base_color_sample)
        if osh.specialization('WAVE_WORK_COUNTERS'):
            if material.texture_indices.z >= 0.0:
                profileWork(osh.u32(8), osh.u32(1))
        emissive_sample = sampleMaterialTexture(material.texture_indices.z, uv0, uv1, True, uv0_footprint, uv1_footprint).rgb
        material.emission_metallic.rgb = ordinarylight_texture_apply_rgb(material.emission_metallic.rgb, emissive_sample)
        clearcoat_sample = sampleMaterialTexture(material.advanced_texture_indices.x, uv0, uv1, False, uv0_footprint, uv1_footprint)
        sheen_sample = sampleMaterialTexture(material.advanced_texture_indices.y, uv0, uv1, True, uv0_footprint, uv1_footprint)
        anisotropy_sample = sampleMaterialTexture(material.advanced_texture_indices.z, uv0, uv1, False, uv0_footprint, uv1_footprint)
        subsurface_sample = sampleMaterialTexture(material.advanced_texture_indices.w, uv0, uv1, False, uv0_footprint, uv1_footprint)
        thickness_sample = sampleMaterialTexture(material.optical.x, uv0, uv1, False, uv0_footprint, uv1_footprint)
        material.advanced0.x = material.advanced0.x * clearcoat_sample.r
        material.advanced0.y = material.advanced0.y * osh.maximum(clearcoat_sample.g, clearcoat_sample.r)
        material.sheen_color.rgb = material.sheen_color.rgb * sheen_sample.rgb
        material.advanced0.z = material.advanced0.z * sheen_sample.a
        material.advanced0.w = material.advanced0.w * anisotropy_sample.r
        material.advanced1.x = material.advanced1.x * subsurface_sample.r
        material.advanced1.w = material.advanced1.w * thickness_sample.g
        if material.attenuation_transmission.a > 0.001:
            return
        if osh.specialization('WAVE_WORK_COUNTERS'):
            if material.texture_indices.y >= 0.0:
                profileWork(osh.u32(7), osh.u32(1))
        metallic_roughness = sampleMaterialTexture(material.texture_indices.y, uv0, uv1, False, uv0_footprint, uv1_footprint)
        material.base_roughness.a = ordinarylight_texture_apply_scalar(material.base_roughness.a, metallic_roughness.g)
        material.emission_metallic.a = ordinarylight_texture_apply_scalar(material.emission_metallic.a, metallic_roughness.b)
        if osh.specialization('WAVE_WORK_COUNTERS'):
            if material.texture_parameters.y >= 0.0:
                profileWork(osh.u32(9), osh.u32(1))
        occlusion = sampleMaterialTexture(material.texture_parameters.y, uv0, uv1, False, uv0_footprint, uv1_footprint).r
        material.texture_parameters.w = ordinarylight_texture_apply_occlusion(occlusion, material.texture_parameters.z)

@osh.function
def applyNormalTexture(material: MaterialData, uv0: osh.vec2, uv1: osh.vec2, uv0_footprint: osh.f32, uv1_footprint: osh.f32, shading_normal: osh.vec3, tangent_data: osh.vec4) -> osh.vec3:
    if osh.specialization('WAVE_UNTEXTURED_SCENE'):
        return shading_normal
    else:
        texture_index = osh.i32(material.texture_indices.w)
        if texture_index < 0:
            return shading_normal
        if osh.specialization('WAVE_WORK_COUNTERS'):
            profileWork(osh.u32(10), osh.u32(1))
        tangent_normal = sampleMaterialTexture(material.texture_indices.w, uv0, uv1, False, uv0_footprint, uv1_footprint).xyz
        binding = texture_bindings[texture_index]
        return ordinarylight_texture_apply_normal(tangent_normal, material.texture_parameters.x, shading_normal, tangent_data, binding.texture_rotation.yz, binding.offset_scale.zw)

@osh.function
def textureBindingUsesUv1(binding_index_value: osh.f32) -> osh.boolean:
    binding_index = osh.i32(binding_index_value)
    return binding_index >= 0 and texture_bindings[binding_index].texture_rotation.w > 0.5

@osh.function
def materialUsesUv1(material: MaterialData) -> osh.boolean:
    return osh.fraction(material.ior_distance.z) > 0.125

@osh.function
def triangleTangent(a: osh.vec3, b: osh.vec3, c: osh.vec3, uv_a: osh.vec2, uv_b: osh.vec2, uv_c: osh.vec2, shading_normal: osh.vec3) -> osh.vec4:
    edge_a = b - a
    edge_b = c - a
    delta_a = uv_b - uv_a
    delta_b = uv_c - uv_a
    determinant = delta_a.x * delta_b.y - delta_a.y * delta_b.x
    if osh.absolute(determinant) < 1e-08:
        fallback = osh.normalize(osh.cross(shading_normal, osh.vec3(0.0, 0.0, 1.0)) if osh.absolute(shading_normal.z) < 0.999 else osh.cross(shading_normal, osh.vec3(0.0, 1.0, 0.0)))
        return osh.vec4(fallback, 1.0)
    inverse = 1.0 / determinant
    tangent = osh.normalize((edge_a * delta_b.y - edge_b * delta_a.y) * inverse)
    bitangent = osh.normalize((edge_b * delta_a.x - edge_a * delta_b.x) * inverse)
    handedness = -1.0 if osh.dot(osh.cross(shading_normal, tangent), bitangent) < 0.0 else 1.0
    return osh.vec4(tangent, handedness)
