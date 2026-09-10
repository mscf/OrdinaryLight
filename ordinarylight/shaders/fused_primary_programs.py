"""Typed transport helpers. Generated artifacts must be rebuilt from this source."""
import ordinaryshade as osh
from .lighting_programs import AreaLightCandidate
from .restir_programs import DirectLightReservoir
from .transport_programs import MaterialEvaluation
WAVE_MAX_MEDIUM_STACK_DEPTH = 16

@osh.structure
class WaveRay:
    origin_tmin: osh.vec4
    direction_tmax: osh.vec4
    path_index: osh.u32
    padding_a: osh.u32
    padding_b: osh.u32
    padding_c: osh.u32

@osh.structure
class WavePathState:
    throughput: osh.vec4
    radiance: osh.vec4
    metadata: osh.uvec4

@osh.structure
class SecondaryPathState:
    position_valid: osh.vec4
    normal_pdf: osh.vec4
    primary_throughput: osh.vec4
    primary_radiance: osh.vec4
    diffuse_radiance_hit_distance: osh.vec4
    specular_radiance_hit_distance: osh.vec4
    primary_position: osh.vec4
    primary_geometry: osh.vec4

@osh.structure
class WaveMediumStack:
    ior: osh.array(osh.f32, WAVE_MAX_MEDIUM_STACK_DEPTH)

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
class VertexAttributeData:
    normal: osh.vec4
    texcoord: osh.vec4
    tangent: osh.vec4

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
class TextureBindingData:
    texture_rotation: osh.vec4
    offset_scale: osh.vec4

@osh.function
def pathRng(path: WavePathState) -> osh.u32:
    return path.metadata.z

@osh.function
def setPathRng(path: osh.inout(WavePathState), rng: osh.u32) -> osh.void:
    path.metadata.z = rng

@osh.function
def pathBounce(path: WavePathState) -> osh.u32:
    return osh.u32(path.throughput.w)

@osh.function
def setPathBounce(path: osh.inout(WavePathState), bounce: osh.u32) -> osh.void:
    path.throughput.w = osh.f32(bounce)

@osh.function
def pathPreviousPdf(path: WavePathState) -> osh.f32:
    return path.radiance.w

@osh.function
def setPathPreviousPdf(path: osh.inout(WavePathState), pdf: osh.f32) -> osh.void:
    path.radiance.w = pdf

@osh.function
def profileWork(counter: osh.u32, amount: osh.u32) -> osh.void:
    osh.atomic_add(work_counters[counter], amount)
    bounce = osh.minimum(profile_bounce, osh.u32(7))
    if counter == osh.u32(0):
        osh.atomic_add(work_counters[osh.u32(16) + bounce], amount)
    elif counter == osh.u32(1):
        osh.atomic_add(work_counters[osh.u32(24) + bounce], amount)
    elif counter == osh.u32(3):
        osh.atomic_add(work_counters[osh.u32(32) + bounce], amount)

@osh.function
def restirEncodeNormal(normal: osh.vec3) -> osh.vec2:
    normal = normal / (osh.absolute(normal.x) + osh.absolute(normal.y) + osh.absolute(normal.z))
    encoded = normal.xy
    if normal.z < 0.0:
        encoded = (1.0 - osh.absolute(encoded.yx)) * osh.sign(encoded.xy)
    return encoded

@osh.function
def restirDecodeNormal(encoded: osh.vec2) -> osh.vec3:
    normal = osh.vec3(encoded, 1.0 - osh.absolute(encoded.x) - osh.absolute(encoded.y))
    if normal.z < 0.0:
        normal.xy = (1.0 - osh.absolute(normal.yx)) * osh.sign(normal.xy)
    return osh.normalize(normal)

@osh.function
def restirPackNormalClass(normal: osh.vec3, surface_class: osh.f32) -> osh.u32:
    unit = restirEncodeNormal(normal) * 0.5 + 0.5
    quantized = osh.uvec2(osh.round(osh.clamp(unit, 0.0, 1.0) * 32767.0))
    classification = osh.u32(osh.clamp(osh.round(surface_class), 0.0, 3.0))
    return quantized.x | quantized.y << osh.u32(15) | classification << osh.u32(30)

@osh.function
def restirUnpackNormalClass(packed: osh.u32) -> osh.vec4:
    unit = osh.vec2(osh.f32(packed & osh.u32(32767)), osh.f32(packed >> osh.u32(15) & osh.u32(32767))) / 32767.0
    return osh.vec4(restirDecodeNormal(unit * 2.0 - 1.0), osh.f32(packed >> osh.u32(30)))

@osh.function
def restirSpatialOffset(index: osh.u32, radius: osh.i32) -> osh.ivec2:
    directions = osh.local_array(osh.ivec2, 8)
    directions[0] = osh.ivec2(1, 0)
    directions[1] = osh.ivec2(-1, 0)
    directions[2] = osh.ivec2(0, 1)
    directions[3] = osh.ivec2(0, -1)
    directions[4] = osh.ivec2(1, 1)
    directions[5] = osh.ivec2(-1, 1)
    directions[6] = osh.ivec2(1, -1)
    directions[7] = osh.ivec2(-1, -1)
    return directions[index & osh.u32(7)] * radius

@osh.function
def restirMaterialSignature(material: MaterialData) -> osh.u32:
    signature = hashValue(osh.float_bits_to_uint(material.base_roughness.x))
    signature = signature ^ hashValue(osh.float_bits_to_uint(material.base_roughness.y))
    signature = signature ^ hashValue(osh.float_bits_to_uint(material.base_roughness.z))
    signature = signature ^ hashValue(osh.float_bits_to_uint(material.base_roughness.w))
    signature = signature ^ hashValue(osh.float_bits_to_uint(material.emission_metallic.w))
    signature = signature ^ hashValue(osh.float_bits_to_uint(material.attenuation_transmission.w))
    signature = signature ^ hashValue(osh.float_bits_to_uint(material.ior_distance.x))
    signature = signature ^ hashValue(osh.float_bits_to_uint(material.texture_indices.x))
    signature = signature ^ hashValue(osh.float_bits_to_uint(material.texture_indices.y))
    signature = signature ^ hashValue(osh.float_bits_to_uint(material.texture_indices.z))
    signature = signature ^ hashValue(osh.float_bits_to_uint(material.texture_indices.w))
    return signature

@osh.function
def reprojectRestir(world_position: osh.vec3, previous_pixel: osh.out(osh.ivec2)) -> osh.boolean:
    offset = world_position - previous_camera.origin.xyz
    previous_depth = osh.dot(offset, previous_camera.forward.xyz)
    scale = osh.length(previous_camera.up.xyz)
    aspect = osh.f32(push.image_tile.x) / osh.f32(push.image_tile.y)
    if previous_depth <= 0.0001 or scale <= 0.0001:
        return False
    ndc = osh.vec2(osh.dot(offset, osh.normalize(previous_camera.right.xyz)) / (previous_depth * aspect * scale), -osh.dot(offset, osh.normalize(previous_camera.up.xyz)) / (previous_depth * scale))
    pixel = (ndc * 0.5 + 0.5) * osh.vec2(push.image_tile.xy) - 0.5
    previous_pixel = osh.ivec2(osh.round(pixel))
    return osh.all_value(previous_pixel >= osh.ivec2(0)) and osh.all_value(previous_pixel < osh.ivec2(push.image_tile.xy))

@osh.function
def restirPreviousWorldPosition(pixel: osh.ivec2, ray_distance: osh.f32) -> osh.vec3:
    ndc = (osh.vec2(pixel) + 0.5) / osh.vec2(push.image_tile.xy) * 2.0 - 1.0
    aspect = osh.f32(push.image_tile.x) / osh.f32(push.image_tile.y)
    direction = osh.normalize(previous_camera.forward.xyz + ndc.x * aspect * previous_camera.right.xyz - ndc.y * previous_camera.up.xyz)
    return previous_camera.origin.xyz + direction * ray_distance

@osh.function
def restirHistorySurfaceCompatible(history_pixel: osh.ivec2, position: osh.vec3, shading_normal: osh.vec3, distance: osh.f32, surface_class: osh.f32, material_signature: osh.u32, temporal_center: osh.boolean, old_position: osh.out(osh.vec4), old_normal: osh.out(osh.vec4)) -> osh.boolean:
    old_distance = previous_position_image.load(history_pixel).x
    old_position = osh.vec4(restirPreviousWorldPosition(history_pixel, old_distance), old_distance)
    old_normal = restirUnpackNormalClass(previous_normal_image.load(history_pixel).x)
    old_material = previous_material_image.load(history_pixel).x
    position_delta = old_position.xyz - position
    position_tolerance = osh.maximum(0.03, distance * 0.01)
    position_error = osh.length(position_delta) if temporal_center else osh.absolute(osh.dot(position_delta, osh.normalize(shading_normal)))
    normal_agreement = osh.dot(osh.normalize(old_normal.xyz), osh.normalize(shading_normal))
    return (((old_position.w >= 0.0 and position_error <= position_tolerance) and normal_agreement > 0.9) and osh.absolute(old_normal.w - surface_class) < 0.25) and old_material == material_signature

@osh.function
def restirHistorySurfaceMatches(history_pixel: osh.ivec2, position: osh.vec3, shading_normal: osh.vec3, distance: osh.f32, surface_class: osh.f32, material_signature: osh.u32, temporal_center: osh.boolean) -> osh.boolean:
    history_position = osh.vec4(0)
    history_normal = osh.vec4(0)
    return restirHistorySurfaceCompatible(history_pixel, position, shading_normal, distance, surface_class, material_signature, temporal_center, history_position, history_normal)

@osh.function
def reserveOutputIndex() -> osh.u32:
    return ordinarylight_reserve_output_index(push.subgroup_enqueue)

@osh.function
def hashValue(value: osh.u32) -> osh.u32:
    value = value ^ value >> 16
    value = value * osh.u32(2146121005)
    value = value ^ value >> 15
    value = value * osh.u32(2221713035)
    value = value ^ value >> 16
    return value

@osh.function
def ordinarylightSecondaryBounce(path: osh.inout(WavePathState), origin: osh.inout(osh.vec3), direction: osh.inout(osh.vec3), path_index: osh.u32, medium_depth: osh.inout(osh.u32), rng: osh.inout(osh.u32), cone_width: osh.inout(osh.f32), cone_spread: osh.inout(osh.f32)) -> osh.boolean:
    bounce = pathBounce(path)
    if osh.specialization('WAVE_WORK_COUNTERS'):
        profile_bounce = bounce
        ordinarylight_profile_work(osh.u32(0), osh.u32(1))
    surface_hit = False
    distance = 1e+30
    primitive = osh.u32(0)
    barycentrics = osh.vec2(0.0)
    ordinarylight_secondary_trace_query(origin, direction, surface_hit, distance, primitive, barycentrics)
    ordinarylight_integrate_secondary_volumes(origin, direction, distance, path.radiance.rgb, path.throughput.rgb)
    if not ordinarylight_secondary_throughput_visible(path.throughput.rgb):
        path.metadata.w = ordinarylight_primary_deactivate(path.metadata.w)
        return False
    if not surface_hit:
        if osh.specialization('WAVE_WORK_COUNTERS'):
            ordinarylight_profile_work(osh.u32(4), osh.u32(1))
        environment_mis = ordinarylight_environment_miss_mis(path.metadata.w & PATH_PREVIOUS_DIFFUSE_BIT != osh.u32(0), push.environment_samples, pathPreviousPdf(path), path.metadata.w & PATH_PREVIOUS_UNIFIED_NEE_BIT != osh.u32(0), unifiedAreaDomainProbability())
        path.radiance.rgb = path.radiance.rgb + ordinarylight_secondary_miss_contribution(path.throughput.rgb, environmentColor(direction), environment_mis)
        path.metadata.w = ordinarylight_primary_deactivate(path.metadata.w)
        return False
    if osh.specialization('WAVE_WORK_COUNTERS'):
        ordinarylight_profile_work(osh.u32(3), osh.u32(1))
    cone_width = ordinarylight_secondary_cone_width(cone_width, distance, cone_spread)
    a = ordinarylight_secondary_vertex_position(primitive, osh.u32(0))
    b = ordinarylight_secondary_vertex_position(primitive, osh.u32(1))
    c = ordinarylight_secondary_vertex_position(primitive, osh.u32(2))
    attribute_a = ordinarylight_secondary_vertex_attribute(primitive, osh.u32(0))
    attribute_b = ordinarylight_secondary_vertex_attribute(primitive, osh.u32(1))
    attribute_c = ordinarylight_secondary_vertex_attribute(primitive, osh.u32(2))
    hit = ordinarylight_primary_hit_position(origin, direction, distance)
    geometric_normal = ordinarylight_primary_geometric_normal(a, b, c)
    weights = ordinarylight_primary_barycentric_weights(barycentrics)
    shading_normal = ordinarylight_primary_shading_normal(attribute_a.normal.xyz, attribute_b.normal.xyz, attribute_c.normal.xyz, weights, geometric_normal)
    entering = ordinarylight_primary_is_entering(direction, geometric_normal)
    normal = ordinarylight_primary_oriented_normal(shading_normal, entering)
    material = ordinarylight_secondary_material(primitive)
    if osh.specialization('WAVE_SER'):
        ser_hint = ordinarylight_secondary_ser_hint(material.attenuation_transmission.a, material.emission_metallic.a, material.emission_metallic.w, material.base_roughness.w, materialHasTextures(material))
        ordinarylight_secondary_reorder(ser_hint)
    if osh.specialization('!WAVE_UNTEXTURED_SECONDARY'):
        uv0 = ordinarylight_primary_interpolate_vec4(attribute_a.texcoord, attribute_b.texcoord, attribute_c.texcoord, weights).xy
        uv1 = ordinarylight_primary_interpolate_vec4(attribute_a.texcoord, attribute_b.texcoord, attribute_c.texcoord, weights).zw
        textured_material = materialHasTextures(material)
        uv0_footprint = ordinarylight_secondary_texture_footprint(cone_width, attribute_a.normal.w, textured_material)
        uv1_density = ordinarylight_primary_uv_density(a, b, c, attribute_a.texcoord.zw, attribute_b.texcoord.zw, attribute_c.texcoord.zw)
        uv1_footprint = ordinarylight_secondary_texture_footprint(cone_width, uv1_density, textured_material)
        applyMaterialTextures(material, uv0, uv1, uv0_footprint, uv1_footprint)
        tangent_data = ordinarylight_primary_interpolate_vec4(attribute_a.tangent, attribute_b.tangent, attribute_c.tangent, weights)
        if textureBindingUsesUv1(material.texture_indices.w):
            tangent_data = ordinarylight_primary_triangle_tangent(a, b, c, attribute_a.texcoord.zw, attribute_b.texcoord.zw, attribute_c.texcoord.zw, shading_normal)
        shading_normal = applyNormalTexture(material, uv0, uv1, uv0_footprint, uv1_footprint, shading_normal, tangent_data)
        shading_normal = ordinarylight_secondary_correct_shading_normal(shading_normal, geometric_normal)
        normal = ordinarylight_primary_oriented_normal(shading_normal, entering)
    else:
        material.texture_parameters.w = 1.0
    if ordinarylight_secondary_capture_hit(path.metadata.w, bounce, ordinarylight_secondary_primary_valid(path_index)):
        captured_position = ordinarylight_secondary_capture_position(hit)
        ordinarylight_store_secondary_hit(path_index, captured_position, normal)
    wave_surface_response = MaterialEvaluation(osh.vec3(0.0), osh.vec3(0.0), 0.0, 0.0, 0.0, 0.0, osh.vec3(0.0), 0.0, 0.0, osh.vec3(0.0), osh.vec3(0.0), 0.0, 0.0)
    if osh.specialization('WAVE_CUSTOM_MATERIAL_PROGRAM'):
        wave_material_uv = attributes[primitive * osh.u32(3)].texcoord.xy * weights.x + attributes[primitive * osh.u32(3) + osh.u32(1)].texcoord.xy * weights.y + attributes[primitive * osh.u32(3) + osh.u32(2)].texcoord.xy * weights.z
        wave_surface_response = waveApplyMaterialProgram(material, normal, wave_material_uv, direction, entering, primitive, weights, osh.f32(bounce))
        if wave_surface_response.custom_scattering > 0.5:
            wave_current_ior = stacks[path_index].ior[medium_depth - osh.u32(1)]
            wave_exterior_ior = osh.maximum(material.ior_distance.x, 1.0001) if entering else stacks[path_index].ior[medium_depth - osh.u32(2)] if medium_depth > osh.u32(1) else 1.0
            wave_random_u = randomFloat(rng)
            wave_random_v = randomFloat(rng)
            wave_surface_response = evaluateMaterial(material, normal, wave_material_uv, direction, entering, wave_random_u, wave_random_v, osh.f32(bounce), wave_current_ior, wave_exterior_ior)
    emission_visible = ordinarylight_secondary_emission_visible(entering, material.ior_distance.w)
    if emission_visible:
        emission_mis = ordinarylight_emissive_hit_mis(path.metadata.w & PATH_PREVIOUS_DIFFUSE_BIT != osh.u32(0), material.emission_metallic.rgb, a, b, c, geometric_normal, direction, distance, material.ior_distance.w > 0.5, push.area_light_weight, push.secondary_area_light_samples, path.metadata.w & PATH_PREVIOUS_UNIFIED_NEE_BIT != osh.u32(0), unifiedAreaDomainProbability(), pathPreviousPdf(path))
        path.radiance.rgb = path.radiance.rgb + ordinarylight_emission_contribution(path.throughput.rgb, material.emission_metallic.rgb, emission_mis)
    next_bounce = ordinarylight_secondary_next_bounce(bounce)
    if ordinarylight_secondary_bounce_terminates(next_bounce, push.max_bounces):
        setPathBounce(path, next_bounce)
        path.metadata.w = ordinarylight_primary_deactivate(path.metadata.w)
        return False
    next_direction = osh.vec3(0)
    bsdf_pdf = 0.0
    if osh.specialization('WAVE_OPAQUE_SCENE'):
        transmission = 0.0
    else:
        transmission = material.attenuation_transmission.a
    if wave_surface_response.custom_scattering > 0.5:
        wave_event = osh.i32(wave_surface_response.event + 0.5)
        if wave_event == 0:
            path.metadata.w = path.metadata.w & ~PATH_ACTIVE_BIT
            return False
        next_direction = osh.normalize(wave_surface_response.next_direction)
        bsdf_pdf = osh.maximum(wave_surface_response.pdf, 1e-06)
        path.throughput.rgb = path.throughput.rgb * (wave_surface_response.weight / bsdf_pdf)
        transmission = 1.0 if wave_event == 3 else 0.0
        if wave_event == 3:
            target_ior = osh.maximum(material.ior_distance.x, 1.0001)
            if entering and medium_depth < WAVE_MAX_MEDIUM_STACK_DEPTH:
                stacks[path_index].ior[medium_depth] = target_ior
                medium_depth = medium_depth + osh.u32(1)
            elif not entering and medium_depth > osh.u32(1):
                medium_depth = medium_depth - osh.u32(1)
    elif transmission > 0.001:
        current_ior = ordinarylight_medium_ior(path_index, medium_depth - osh.u32(1))
        previous_medium_ior = ordinarylight_medium_ior(path_index, medium_depth - osh.u32(2)) if medium_depth > osh.u32(1) else 1.0
        target_ior = ordinarylight_secondary_target_ior(entering, material.ior_distance.x, previous_medium_ior, medium_depth)
        refracted = ordinarylight_secondary_refracted_direction(direction, normal, current_ior, target_ior)
        next_direction = ordinarylight_primary_resolve_transmission_direction(refracted, direction, normal)
        if ordinarylight_secondary_enters_medium(refracted, entering, medium_depth, WAVE_MAX_MEDIUM_STACK_DEPTH):
            ordinarylight_set_medium_ior(path_index, medium_depth, target_ior)
        medium_depth = ordinarylight_secondary_medium_depth(refracted, entering, medium_depth, WAVE_MAX_MEDIUM_STACK_DEPTH)
        optical_distance = 0.0 if entering else distance
        if material.advanced1.z > 0.5:
            optical_distance = material.advanced1.w
        absorption = osh.power(osh.maximum(material.attenuation_transmission.rgb, osh.vec3(1e-06)), osh.vec3(optical_distance / osh.maximum(material.ior_distance.y, 1e-06)))
        tint = osh.mix(osh.vec3(1.0), material.base_roughness.rgb, material.advanced1.z)
        path.throughput.rgb = path.throughput.rgb * (absorption * tint * transmission)
    else:
        nee_probability = ordinarylight_secondary_nee_probability(push.secondary_nee_probability)
        sample_direct = selectSecondaryNee(nee_probability, path.metadata.x, path.metadata.y, next_bounce)
        if sample_direct:
            direct = samplePointLights(hit, normal, direction, material)
            if push.unified_secondary_nee != osh.u32(0):
                direct = direct + sampleUnifiedSecondaryLight(hit, normal, direction, material, rng)
            else:
                light_samples = ordinarylight_secondary_area_sample_count(push.secondary_area_light_samples)
                area_direct = osh.vec3(0.0)
                sample_index = osh.u32(0)
                while sample_index < light_samples:
                    area_direct = area_direct + sampleAreaLight(hit, normal, direction, material, rng, sample_index, light_samples)
                    sample_index = sample_index + 1
                direct = direct + ordinarylight_secondary_average_contribution(area_direct, light_samples)
                environment_samples = ordinarylight_secondary_environment_sample_count(push.environment_samples)
                if environment_samples > osh.u32(0):
                    environment_direct = osh.vec3(0.0)
                    sample_index = osh.u32(0)
                    while sample_index < environment_samples:
                        environment_direct = environment_direct + sampleEnvironment(hit, normal, direction, material, rng, environment_samples)
                        sample_index = sample_index + 1
                    direct = direct + ordinarylight_secondary_average_contribution(environment_direct, environment_samples)
            path.radiance.rgb = path.radiance.rgb + ordinarylight_secondary_direct_contribution(path.throughput.rgb, direct, nee_probability)
        bsdf_weight = osh.vec3(0)
        sampled_specular = osh.f32(0.0)
        samplePbr(material, normal, direction, rng, next_direction, bsdf_weight, bsdf_pdf, sampled_specular)
        path.throughput.rgb = ordinarylight_secondary_scatter_throughput(path.throughput.rgb, bsdf_weight)
        cone_spread = ordinarylight_primary_scattered_cone_spread(cone_spread, material.base_roughness.a)
    next_direction = ordinarylight_primary_continuation_direction(next_direction)
    setPathBounce(path, next_bounce)
    path.metadata.w = ordinarylight_primary_continuation_flags(path.metadata.w, medium_depth, transmission, push.unified_secondary_nee != osh.u32(0))
    setPathPreviousPdf(path, ordinarylight_primary_previous_pdf(pathPreviousPdf(path), bsdf_pdf, transmission))
    if ordinarylight_secondary_roulette_enabled(push.russian_roulette_start, next_bounce, transmission):
        survival = ordinarylight_secondary_survival_probability(path.throughput.rgb, push.russian_roulette_min_survival)
        if not ordinarylight_secondary_survives(randomFloat(rng), survival):
            path.metadata.w = ordinarylight_primary_deactivate(path.metadata.w)
            return False
        path.throughput.rgb = ordinarylight_secondary_survival_throughput(path.throughput.rgb, survival)
    origin = ordinarylight_primary_continuation_origin(hit, next_direction)
    direction = next_direction
    return True

@osh.function
def traceRemaining(path: osh.inout(WavePathState), origin: osh.inout(osh.vec3), direction: osh.inout(osh.vec3), path_index: osh.u32, medium_depth: osh.u32, rng: osh.inout(osh.u32), cone_width: osh.inout(osh.f32), cone_spread: osh.inout(osh.f32)) -> osh.void:
    stop_bounce = ordinarylight_secondary_stop_bounce(WAVE_HYBRID != 0, push.inline_bounces, push.max_bounces)
    ordinarylight_trace_remaining(path, origin, direction, path_index, medium_depth, rng, cone_width, cone_spread, stop_bounce)

@osh.function
def primaryRestirHistoryValid() -> osh.u32:
    if osh.specialization("WAVE_CAMERA_RESTIR_POLICY"):
        if camera.restir_policy.x != osh.u32(0):
            return camera.restir_policy.y
    return push.restir_history_valid

@osh.function
def primaryRestirHistoryLimit() -> osh.u32:
    if osh.specialization("WAVE_CAMERA_RESTIR_POLICY"):
        if camera.restir_policy.x != osh.u32(0):
            return camera.restir_policy.z
    return push.restir_history_limit

@osh.function
def processPrimaryPixel(local_pixel: osh.uvec2) -> osh.void:
    if osh.specialization('WAVE_WORK_COUNTERS'):
        profile_bounce = osh.u32(0)
    if osh.any_value(local_pixel >= push.tile_frame.xy):
        return
    pixel = push.image_tile.zw + local_pixel
    if osh.any_value(pixel >= push.image_tile.xy):
        return
    path_index = local_pixel.y * push.tile_frame.x + local_pixel.x
    if path_index >= output_queue.capacity:
        return
    pixel_index = pixel.y * push.image_tile.x + pixel.x
    restir_reservoir_count = osh.maximum(push.tile_frame.z, osh.u32(1))
    restir_stream = osh.minimum(push.tile_frame.w, restir_reservoir_count - osh.u32(1))
    restir_reservoir_index = pixel_index * restir_reservoir_count + restir_stream
    if osh.specialization('WAVE_SHARED_PRIMARY_RESERVOIRS > 0'):
        direct_stream_count = osh.u32(WAVE_SHARED_PRIMARY_RESERVOIRS) if push.restir_di != osh.u32(0) else osh.u32(1)
        restir_reservoir_count = restir_reservoir_count * direct_stream_count
        restir_stream = push.tile_frame.w * direct_stream_count
        restir_reservoir_index = pixel_index * restir_reservoir_count + restir_stream
    else:
        direct_stream_count = osh.u32(1)
    frame_index = osh.u32(camera.origin.w + 0.5)
    if push.restir_di != osh.u32(0):
        stream = osh.u32(0)
        while stream < direct_stream_count:
            storeCurrentDirectLightReservoir(restir_reservoir_index + stream, emptyDirectLightReservoir())
            stream = stream + 1
    rng = ordinarylight_primary_rng_seed(pixel_index, frame_index, push.tile_frame.w)
    rng = ordinarylight_primary_rng_step(rng)
    jitter_x = ordinarylight_primary_rng_value(rng)
    rng = ordinarylight_primary_rng_step(rng)
    jitter_y = ordinarylight_primary_rng_value(rng)
    jitter = osh.vec2(jitter_x, jitter_y)
    if osh.float_bits_to_uint(camera.right.w) != osh.u32(0):
        jitter = osh.unpack_half2x16(osh.float_bits_to_uint(camera.right.w))
    ndc = (osh.vec2(pixel) + jitter) / osh.vec2(push.image_tile.xy) * 2.0 - 1.0
    aspect = osh.f32(push.image_tile.x) / osh.f32(push.image_tile.y)
    camera_projection = osh.i32(camera.up.w + 0.5)
    ray_origin = ordinarylight_primary_ray_origin(camera.origin.xyz, camera.right.xyz, camera.up.xyz, ndc, aspect, camera_projection)
    incoming = ordinarylight_primary_ray_direction(camera.forward.xyz, camera.right.xyz, camera.up.xyz, ndc, aspect, camera_projection)
    path = WavePathState(osh.vec4(0), osh.vec4(0), osh.uvec4(0))
    path.throughput = osh.vec4(1.0)
    path.radiance = osh.vec4(0.0)
    indirect_capture_pixel = push.indirect_secondary_capture != osh.u32(0)
    if indirect_capture_pixel and push.indirect_capture_stride > osh.u32(1):
        capture_offset = osh.uvec2(push.indirect_capture_stride / osh.u32(2))
        indirect_capture_pixel = osh.all_value(pixel % push.indirect_capture_stride == capture_offset)
    path.metadata = osh.uvec4(pixel_index, ordinarylight_primary_path_identity(frame_index, push.tile_frame.w), rng, ordinarylight_primary_path_flags(indirect_capture_pixel))
    setPathBounce(path, osh.u32(0))
    if osh.specialization('!WAVE_OPAQUE_SCENE'):
        ordinarylight_set_medium_ior(path_index, osh.u32(0), ordinarylight_primary_initial_medium_ior())
    if indirect_capture_pixel:
        ordinarylight_clear_secondary_path(path_index)
    query = osh.ray_query()
    if osh.specialization('WAVE_WORK_COUNTERS'):
        profileWork(osh.u32(0), osh.u32(1))
    query.initialize(scene_tlas, gl_RayFlagsOpaqueEXT, 1, ray_origin, 0.001, incoming, 1e+30)
    while query.proceed():
        pass
    surface_hit = query.intersection_type(True) == gl_RayQueryCommittedIntersectionTriangleEXT
    distance = query.intersection_t(True) if surface_hit else 1e+30
    if osh.specialization("!WAVE_SURFACE_ONLY"):
        integrateVolumesBeforeSurface(ray_origin, incoming, distance, path.radiance.rgb, path.throughput.rgb)
    if not surface_hit or osh.maximum(path.throughput.r, osh.maximum(path.throughput.g, path.throughput.b)) < 0.0001:
        if osh.specialization('WAVE_WORK_COUNTERS'):
            profileWork(osh.u32(4), osh.u32(1))
        if push.gbuffer_enabled != osh.u32(0):
            position_image.store(osh.ivec2(pixel), ordinarylight_primary_invalid_position())
            normal_image.store(osh.ivec2(pixel), ordinarylight_primary_packed_payload(osh.u32(0)))
            material_image.store(osh.ivec2(pixel), ordinarylight_primary_invalid_material())
        if not surface_hit:
            path.radiance.rgb = path.radiance.rgb + path.throughput.rgb * environmentColor(incoming)
        path.metadata.w = ordinarylight_primary_deactivate(path.metadata.w)
        ordinarylight_store_path(path_index, path)
        return
    if osh.specialization('WAVE_WORK_COUNTERS'):
        profileWork(osh.u32(3), osh.u32(1))
    instance_key = query.instance_custom_index(True)
    primitive = query.primitive_index(True) + instance_key
    barycentrics = query.barycentrics(True)
    cone_spread = ordinarylight_primary_cone_spread(camera.up.xyz, push.image_tile.y)
    cone_width = distance * cone_spread
    a = vertices[primitive * osh.u32(3) + osh.u32(0)].xyz
    b = vertices[primitive * osh.u32(3) + osh.u32(1)].xyz
    c = vertices[primitive * osh.u32(3) + osh.u32(2)].xyz
    position = ordinarylight_primary_hit_position(ray_origin, incoming, distance)
    geometric_normal = ordinarylight_primary_geometric_normal(a, b, c)
    weights = ordinarylight_primary_barycentric_weights(barycentrics)
    shading_normal = ordinarylight_primary_shading_normal(attributes[primitive * osh.u32(3) + osh.u32(0)].normal.xyz, attributes[primitive * osh.u32(3) + osh.u32(1)].normal.xyz, attributes[primitive * osh.u32(3) + osh.u32(2)].normal.xyz, weights, geometric_normal)
    entering = ordinarylight_primary_is_entering(incoming, geometric_normal)
    normal = ordinarylight_primary_oriented_normal(shading_normal, entering)
    material = materials[primitive]
    if osh.specialization('WAVE_SER'):
        ser_hint = osh.u32(material.attenuation_transmission.a > 0.001)
        ser_hint = ser_hint | osh.u32(material.emission_metallic.a > 0.5) << osh.u32(1)
        ser_hint = ser_hint | osh.u32(material.emission_metallic.w > 0.5) << osh.u32(2)
        ser_hint = ser_hint | osh.u32(osh.clamp(material.base_roughness.w * 7.0, 0.0, 7.0)) << osh.u32(3)
        ser_hint = ser_hint | osh.u32(materialHasTextures(material)) << osh.u32(6)
        ordinarylight_secondary_reorder(ser_hint)
    material_signature = restirMaterialSignature(material)
    if push.object_effect_ranges[0].x < push.object_effect_ranges[0].y:
        material_signature = material_signature & osh.u32(536870911)
        effect_index = osh.u32(0)
        while effect_index < osh.u32(4):
            ranges = push.object_effect_ranges[effect_index >> osh.u32(1)]
            start = ranges[(effect_index & osh.u32(1)) * osh.u32(2)]
            end = ranges[(effect_index & osh.u32(1)) * osh.u32(2) + osh.u32(1)]
            if (start < end and primitive >= start) and primitive < end:
                material_signature = material_signature | effect_index + osh.u32(1) << osh.u32(29)
                break
            effect_index = effect_index + 1
    if osh.specialization('WAVE_UNTEXTURED_PRIMARY'):
        material_textured = False
        material.texture_parameters.w = 1.0
    else:
        material_textured = materialHasTextures(material)
        interpolated_uv = ordinarylight_primary_interpolate_vec4(attributes[primitive * osh.u32(3) + osh.u32(0)].texcoord, attributes[primitive * osh.u32(3) + osh.u32(1)].texcoord, attributes[primitive * osh.u32(3) + osh.u32(2)].texcoord, weights)
        uv0 = interpolated_uv.xy
        uv1 = interpolated_uv.zw
        uv0_footprint = 0.0
        uv1_footprint = 0.0
        if material_textured:
            uv0_footprint = cone_width * attributes[primitive * osh.u32(3) + osh.u32(0)].normal.w
            uv1_footprint = cone_width * ordinarylight_primary_uv_density(a, b, c, attributes[primitive * osh.u32(3) + osh.u32(0)].texcoord.zw, attributes[primitive * osh.u32(3) + osh.u32(1)].texcoord.zw, attributes[primitive * osh.u32(3) + osh.u32(2)].texcoord.zw)
        applyMaterialTextures(material, uv0, uv1, uv0_footprint, uv1_footprint)
        tangent_data = ordinarylight_primary_interpolate_vec4(attributes[primitive * osh.u32(3) + osh.u32(0)].tangent, attributes[primitive * osh.u32(3) + osh.u32(1)].tangent, attributes[primitive * osh.u32(3) + osh.u32(2)].tangent, weights)
        if textureBindingUsesUv1(material.texture_indices.w):
            tangent_data = ordinarylight_primary_triangle_tangent(a, b, c, attributes[primitive * osh.u32(3) + osh.u32(0)].texcoord.zw, attributes[primitive * osh.u32(3) + osh.u32(1)].texcoord.zw, attributes[primitive * osh.u32(3) + osh.u32(2)].texcoord.zw, shading_normal)
        shading_normal = applyNormalTexture(material, uv0, uv1, uv0_footprint, uv1_footprint, shading_normal, tangent_data)
        shading_normal = ordinarylight_primary_correct_mapped_normal(shading_normal, geometric_normal)
        normal = shading_normal if entering else -shading_normal
    wave_surface_response = MaterialEvaluation(osh.vec3(0.0), osh.vec3(0.0), 0.0, 0.0, 0.0, 0.0, osh.vec3(0.0), 0.0, 0.0, osh.vec3(0.0), osh.vec3(0.0), 0.0, 0.0)
    if osh.specialization('WAVE_CUSTOM_MATERIAL_PROGRAM'):
        wave_material_uv = attributes[primitive * osh.u32(3)].texcoord.xy * weights.x + attributes[primitive * osh.u32(3) + osh.u32(1)].texcoord.xy * weights.y + attributes[primitive * osh.u32(3) + osh.u32(2)].texcoord.xy * weights.z
        wave_surface_response = waveApplyMaterialProgram(material, normal, wave_material_uv, incoming, entering, primitive, weights, 0.0)
        if wave_surface_response.custom_scattering > 0.5:
            wave_current_ior = 1.0
            wave_exterior_ior = osh.maximum(material.ior_distance.x, 1.0001)
            wave_random_u = randomFloat(rng)
            wave_random_v = randomFloat(rng)
            wave_surface_response = evaluateMaterial(material, normal, wave_material_uv, incoming, entering, wave_random_u, wave_random_v, 0.0, wave_current_ior, wave_exterior_ior)
    if osh.specialization('WAVE_OPAQUE_SCENE'):
        surface_class = ordinarylight_primary_surface_class(material.attenuation_transmission.a, material.emission_metallic.a, True)
    else:
        surface_class = ordinarylight_primary_surface_class(material.attenuation_transmission.a, material.emission_metallic.a, False)
    if push.gbuffer_enabled != osh.u32(0):
        position_image.store(osh.ivec2(pixel), ordinarylight_primary_hit_position_payload(distance))
        normal_image.store(osh.ivec2(pixel), ordinarylight_primary_packed_payload(restirPackNormalClass(shading_normal, surface_class)))
        material_image.store(osh.ivec2(pixel), ordinarylight_primary_packed_payload(material_signature))
    path.radiance.rgb = path.radiance.rgb + ordinarylight_primary_emission(material.emission_metallic.rgb, entering, material.ior_distance.w > 0.5)
    if ordinarylight_primary_should_terminate(push.max_bounces):
        setPathBounce(path, osh.u32(1))
        path.metadata.w = ordinarylight_primary_deactivate(path.metadata.w)
        ordinarylight_store_path(path_index, path)
        return
    next_direction = osh.vec3(0)
    bsdf_pdf = 0.0
    if osh.specialization('WAVE_OPAQUE_SCENE'):
        transmission = ordinarylight_primary_transmission(material.attenuation_transmission.a, True)
    else:
        transmission = ordinarylight_primary_transmission(material.attenuation_transmission.a, False)
    medium_depth = osh.u32(1)
    sampled_specular = 0.0
    primary_specular = osh.vec3(0.0)
    indirect_specular_fraction = osh.vec3(1.0)
    if wave_surface_response.custom_scattering > 0.5:
        wave_event = osh.i32(wave_surface_response.event + 0.5)
        if wave_event == 0:
            path.metadata.w = path.metadata.w & ~PATH_ACTIVE_BIT
            setPathRng(path, rng)
            paths[path_index] = path
            return
        next_direction = osh.normalize(wave_surface_response.next_direction)
        bsdf_pdf = osh.maximum(wave_surface_response.pdf, 1e-06)
        path.throughput.rgb = path.throughput.rgb * (wave_surface_response.weight / bsdf_pdf)
        transmission = 1.0 if wave_event == 3 else 0.0
        if wave_event == 3:
            target_ior = osh.maximum(material.ior_distance.x, 1.0001)
            if entering and medium_depth < WAVE_MAX_MEDIUM_STACK_DEPTH:
                stacks[path_index].ior[medium_depth] = target_ior
                medium_depth = medium_depth + osh.u32(1)
            elif not entering and medium_depth > osh.u32(1):
                medium_depth = medium_depth - osh.u32(1)
    elif transmission > 0.001:
        target_ior = ordinarylight_primary_target_ior(material.ior_distance.x)
        refracted_direction = ordinarylight_primary_refracted_direction(incoming, normal, target_ior)
        next_direction = ordinarylight_primary_resolve_transmission_direction(refracted_direction, incoming, normal)
        entered_medium = ordinarylight_primary_enters_medium(refracted_direction, entering)
        if entered_medium:
            ordinarylight_set_medium_ior(path_index, osh.u32(1), target_ior)
        medium_depth = ordinarylight_primary_medium_depth(entered_medium)
        optical_distance = material.advanced1.w if material.advanced1.z > 0.5 else 0.0
        absorption = osh.power(osh.maximum(material.attenuation_transmission.rgb, osh.vec3(1e-06)), osh.vec3(optical_distance / osh.maximum(material.ior_distance.y, 1e-06)))
        tint = osh.mix(osh.vec3(1.0), material.base_roughness.rgb, material.advanced1.z)
        path.throughput.rgb = path.throughput.rgb * (absorption * tint * transmission)
    else:
        if osh.specialization("WAVE_PRODUCTION_RESTIR && WAVE_PREPARED_PRIMARY_PBR"):
            preparePrimaryPbr(material, normal, -incoming)
        path.radiance.rgb = path.radiance.rgb + samplePointLights(position, normal, incoming, material)
        primary_specular = primary_specular + transportPointSpecular
        light_samples = osh.clamp(push.area_light_samples, osh.u32(1), osh.u32(16))
        first_restir_stream = restir_stream
        direct_stream = osh.u32(0)
        while direct_stream < direct_stream_count:
            restir_stream = first_restir_stream + direct_stream
            restir_reservoir_index = pixel_index * restir_reservoir_count + restir_stream
            radiance_before_direct = path.radiance.rgb
            specular_before_direct = primary_specular
            if push.restir_di != osh.u32(0) and push.area_light_count > osh.u32(0):
                candidate_count = osh.clamp(push.restir_candidate_count, osh.u32(1), osh.u32(4))
                reservoir = emptyDirectLightReservoir()
                sample_index = osh.u32(0)
                while sample_index < candidate_count:
                    candidate = generateUnifiedPrimaryCandidate(position, normal, incoming, material, rng, sample_index, candidate_count) if WAVE_UNIFIED_PRIMARY_RESTIR != osh.u32(0) else generateAreaLightCandidate(position, normal, incoming, material, rng, sample_index, candidate_count)
                    updateDirectLightReservoir(reservoir, candidate.light_index, candidate.barycentrics, candidate.target, candidate.target, 1.0, randomFloat(rng))
                    sample_index = sample_index + 1
                if primaryRestirHistoryValid() != osh.u32(0):
                    previous_pixel = osh.ivec2(0)
                    if reprojectRestir(position, previous_pixel):
                        spatial_count = osh.clamp(push.restir_spatial_neighbors, osh.u32(1), osh.u32(8)) if push.restir_spatial_reuse != osh.u32(0) else osh.u32(0)
                        spatial_rotation = hashValue(pixel_index ^ frame_index) & osh.u32(7)
                        reuse_index = osh.u32(0)
                        while reuse_index <= spatial_count:
                            history_pixel = previous_pixel
                            if reuse_index > osh.u32(0):
                                history_pixel = history_pixel + restirSpatialOffset(reuse_index - osh.u32(1) + spatial_rotation, osh.i32(push.restir_spatial_radius))
                            inside = osh.all_value(history_pixel >= osh.ivec2(0)) and osh.all_value(history_pixel < osh.ivec2(push.image_tile.xy))
                            geometry_valid = False
                            history_source_present = False
                            history = emptyDirectLightReservoir()
                            if inside:
                                previous_index = osh.u32(history_pixel.y) * push.image_tile.x + osh.u32(history_pixel.x)
                                history = loadPreviousDirectLightReservoir(previous_index * restir_reservoir_count + restir_stream)
                                history_source_present = history.data.x != osh.u32(4294967295)
                                if history_source_present:
                                    geometry_valid = restirHistorySurfaceMatches(history_pixel, position, shading_normal, distance, surface_class, material_signature, reuse_index == osh.u32(0))
                            if osh.specialization('WAVE_SHARED_PRIMARY_RESERVOIRS > 0'):
                                if not geometry_valid:
                                    history = emptyDirectLightReservoir()
                            if geometry_valid:
                                current_count = osh.unpack_half2x16(reservoir.data.w).y
                                remaining_history = osh.maximum(osh.f32(primaryRestirHistoryLimit()) - current_count, 0.0)
                                source_history_limit = osh.minimum(remaining_history, 1.0) if spatial_count > osh.u32(0) else remaining_history
                                history = limitDirectLightReservoir(history, source_history_limit)
                                if osh.unpack_half2x16(history.data.w).y <= 0.0:
                                    history = emptyDirectLightReservoir()
                            if history.data.x != osh.u32(4294967295):
                                if osh.specialization('WAVE_WORK_COUNTERS'):
                                    profileWork(osh.u32(11), osh.u32(1))
                                history_candidate = AreaLightCandidate(osh.u32(0), osh.vec2(0), osh.f32(0))
                                history_candidate.light_index = history.data.x
                                history_candidate.barycentrics = osh.unpack_half2x16(history.data.y)
                                history_candidate.target = 0.0
                                history_direction = osh.vec3(0)
                                history_distance = osh.f32(0.0)
                                history_contribution = evaluateUnifiedPrimaryCandidate(history_candidate, position, normal, incoming, material, candidate_count, history_direction, history_distance) if WAVE_UNIFIED_PRIMARY_RESTIR != osh.u32(0) else evaluateAreaLightCandidate(history_candidate, position, normal, incoming, material, candidate_count, history_direction, history_distance)
                                history_target = osh.maximum(osh.dot(history_contribution, osh.vec3(0.2126, 0.7152, 0.0722)), 0.0)
                                pairwise = (push.restir_pairwise_mis != osh.u32(0) and reuse_index > osh.u32(0)) and (not material_textured)
                                source_target = osh.unpack_half2x16(history.data.w).x
                                generalized = (WAVE_GENERALIZED_RESTIR != osh.u32(0) and reuse_index > osh.u32(0)) and (not material_textured)
                                generalized_density_normalization = 1.0
                                if generalized:
                                    target_sum = history_target
                                    active_proposals = 1.0 if history_target > 0.0 else 0.0
                                    proposal_index = osh.u32(0)
                                    while proposal_index <= spatial_count:
                                        proposal_pixel = previous_pixel
                                        if proposal_index > osh.u32(0):
                                            proposal_pixel = proposal_pixel + restirSpatialOffset(proposal_index - osh.u32(1) + spatial_rotation, osh.i32(push.restir_spatial_radius))
                                        proposal_inside = osh.all_value(proposal_pixel >= osh.ivec2(0)) and osh.all_value(proposal_pixel < osh.ivec2(push.image_tile.xy))
                                        if not proposal_inside:
                                            proposal_index = proposal_index + 1
                                            continue
                                        proposal_flat_index = osh.u32(proposal_pixel.y) * push.image_tile.x + osh.u32(proposal_pixel.x)
                                        proposal = loadPreviousDirectLightReservoir(proposal_flat_index * restir_reservoir_count + restir_stream)
                                        if proposal.data.x == osh.u32(4294967295):
                                            proposal_index = proposal_index + 1
                                            continue
                                        proposal_position = osh.vec4(0)
                                        proposal_normal = osh.vec4(0)
                                        if not restirHistorySurfaceCompatible(proposal_pixel, position, shading_normal, distance, surface_class, material_signature, proposal_index == osh.u32(0), proposal_position, proposal_normal):
                                            proposal_index = proposal_index + 1
                                            continue
                                        proposal_target = source_target
                                        if proposal_index != reuse_index:
                                            proposal_direction = osh.vec3(0)
                                            proposal_distance = osh.f32(0.0)
                                            proposal_incoming = osh.normalize(proposal_position.xyz - previous_camera.origin.xyz)
                                            proposal_contribution = evaluateUnifiedPrimaryCandidate(history_candidate, proposal_position.xyz, osh.normalize(proposal_normal.xyz), proposal_incoming, material, candidate_count, proposal_direction, proposal_distance) if WAVE_UNIFIED_PRIMARY_RESTIR != osh.u32(0) else evaluateAreaLightCandidate(history_candidate, proposal_position.xyz, osh.normalize(proposal_normal.xyz), proposal_incoming, material, candidate_count, proposal_direction, proposal_distance)
                                            proposal_target = osh.maximum(osh.dot(proposal_contribution, osh.vec3(0.2126, 0.7152, 0.0722)), 0.0)
                                        target_sum = target_sum + proposal_target
                                        if proposal_target > 0.0:
                                            active_proposals = active_proposals + 1.0
                                        proposal_index = proposal_index + 1
                                    canonical_normalization = 1.0 / source_target if source_target > 0.0 else 0.0
                                    generalized_density_normalization = osh.minimum(active_proposals / target_sum, push.restir_generalized_balance_cap * canonical_normalization) if target_sum > 0.0 else 0.0
                                history_selected = mergeDirectLightReservoir(reservoir, history, history_target, randomFloat(rng)) if reuse_index == osh.u32(0) else mergeBalancedDirectLightReservoir(reservoir, history, history_target, generalized_density_normalization, randomFloat(rng)) if generalized else mergePairwiseDirectLightReservoir(reservoir, history, history_target, source_target, randomFloat(rng)) if pairwise else mergeCanonicalDirectLightReservoir(reservoir, history, history_target, randomFloat(rng))
                                if history_selected:
                                    if osh.specialization('WAVE_WORK_COUNTERS'):
                                        profileWork(osh.u32(12), osh.u32(1))
                            else:
                                if osh.specialization('WAVE_WORK_COUNTERS'):
                                    profileWork(osh.u32(13), osh.u32(1))
                                    profileWork(osh.u32(14) if history_source_present else osh.u32(15), osh.u32(1))
                                if reuse_index == osh.u32(0) and (not history_source_present):
                                    break
                            reuse_index = reuse_index + 1
                storeCurrentDirectLightReservoir(restir_reservoir_index, reservoir)
                if reservoir.data.x != osh.u32(4294967295):
                    selected = AreaLightCandidate(osh.u32(0), osh.vec2(0), osh.f32(0))
                    selected.light_index = reservoir.data.x
                    selected.barycentrics = osh.unpack_half2x16(reservoir.data.y)
                    selected.target = osh.unpack_half2x16(reservoir.data.w).x
                    selected_direction = osh.vec3(0)
                    selected_distance = osh.f32(0.0)
                    selected_contribution = evaluateUnifiedPrimaryCandidate(selected, position, normal, incoming, material, candidate_count, selected_direction, selected_distance) if WAVE_UNIFIED_PRIMARY_RESTIR != osh.u32(0) else evaluateAreaLightCandidate(selected, position, normal, incoming, material, candidate_count, selected_direction, selected_distance)
                    selected_specular = selected_contribution * transportLastSpecularFraction
                    visibility = areaLightCandidateVisibility(position, normal, selected_direction, selected_distance)
                    path.radiance.rgb = path.radiance.rgb + selected_contribution * visibility * directLightReservoirNormalization(reservoir)
                    primary_specular = primary_specular + selected_specular * visibility * directLightReservoirNormalization(reservoir)
            else:
                area_direct = osh.vec3(0.0)
                sample_index = osh.u32(0)
                while sample_index < light_samples:
                    contribution = sampleAreaLight(position, normal, incoming, material, rng, sample_index, light_samples)
                    area_direct = area_direct + contribution
                    primary_specular = primary_specular + contribution * transportLastSpecularFraction / osh.f32(light_samples)
                    sample_index = sample_index + 1
                path.radiance.rgb = path.radiance.rgb + area_direct / osh.f32(light_samples)
            if (push.restir_di != osh.u32(0) and WAVE_STRATIFIED_PRIMARY_RESTIR != osh.u32(0)) and push.environment_samples > osh.u32(0):
                reservoir_pixel_count = push.image_tile.x * push.image_tile.y * restir_reservoir_count
                environment_reservoir = emptyDirectLightReservoir()
                environment_candidates = osh.clamp(push.restir_candidate_count, osh.u32(1), osh.u32(4))
                sample_index = osh.u32(0)
                while sample_index < environment_candidates:
                    candidate = AreaLightCandidate(osh.u32(0), osh.vec2(0), osh.f32(0))
                    candidate_direction = cosineHemisphere(normal, randomFloat(rng), randomFloat(rng))
                    candidate.light_index = ENVIRONMENT_LIGHT_CANDIDATE_INDEX
                    candidate.barycentrics = encodeEnvironmentCandidateDirection(candidate_direction)
                    candidate_distance = osh.f32(0.0)
                    contribution = evaluateEnvironmentCandidate(candidate.barycentrics, position, normal, incoming, material, environment_candidates, 1.0, candidate_direction, candidate_distance)
                    candidate.target = osh.maximum(osh.dot(contribution, osh.vec3(0.2126, 0.7152, 0.0722)), 0.0)
                    updateDirectLightReservoir(environment_reservoir, candidate.light_index, candidate.barycentrics, candidate.target, candidate.target, 1.0, randomFloat(rng))
                    sample_index = sample_index + 1
                if primaryRestirHistoryValid() != osh.u32(0):
                    history_pixel = osh.ivec2(0)
                    if (reprojectRestir(position, history_pixel) and osh.all_value(history_pixel >= osh.ivec2(0))) and osh.all_value(history_pixel < osh.ivec2(push.image_tile.xy)):
                        if restirHistorySurfaceMatches(history_pixel, position, shading_normal, distance, surface_class, material_signature, True):
                            history_index = osh.u32(history_pixel.y) * push.image_tile.x + osh.u32(history_pixel.x)
                            history = loadPreviousEnvironmentReservoir(history_index * restir_reservoir_count + restir_stream, reservoir_pixel_count)
                            history = limitDirectLightReservoir(history, osh.maximum(osh.f32(primaryRestirHistoryLimit()) - osh.unpack_half2x16(environment_reservoir.data.w).y, 0.0))
                            if history.data.x != osh.u32(4294967295):
                                history_candidate = AreaLightCandidate(osh.u32(0), osh.vec2(0), osh.f32(0))
                                history_candidate.light_index = history.data.x
                                history_candidate.barycentrics = osh.unpack_half2x16(history.data.y)
                                history_direction = osh.vec3(0)
                                history_distance = osh.f32(0.0)
                                history_contribution = evaluateEnvironmentCandidate(history_candidate.barycentrics, position, normal, incoming, material, environment_candidates, 1.0, history_direction, history_distance)
                                history_target = osh.maximum(osh.dot(history_contribution, osh.vec3(0.2126, 0.7152, 0.0722)), 0.0)
                                mergeDirectLightReservoir(environment_reservoir, history, history_target, randomFloat(rng))
                storeCurrentEnvironmentReservoir(restir_reservoir_index, reservoir_pixel_count, environment_reservoir)
                if environment_reservoir.data.x != osh.u32(4294967295):
                    selected_direction = osh.vec3(0)
                    selected_distance = osh.f32(0.0)
                    selected_contribution = evaluateEnvironmentCandidate(osh.unpack_half2x16(environment_reservoir.data.y), position, normal, incoming, material, environment_candidates, 1.0, selected_direction, selected_distance)
                    selected_specular = selected_contribution * transportLastSpecularFraction
                    visibility = areaLightCandidateVisibility(position, normal, selected_direction, selected_distance)
                    path.radiance.rgb = path.radiance.rgb + selected_contribution * visibility * directLightReservoirNormalization(environment_reservoir)
                    primary_specular = primary_specular + selected_specular * visibility * directLightReservoirNormalization(environment_reservoir)
            path.radiance.rgb = radiance_before_direct + (path.radiance.rgb - radiance_before_direct) / osh.f32(direct_stream_count)
            primary_specular = specular_before_direct + (primary_specular - specular_before_direct) / osh.f32(direct_stream_count)
            direct_stream = direct_stream + 1
        environment_samples = osh.u32(0) if push.restir_di != osh.u32(0) and (WAVE_UNIFIED_PRIMARY_RESTIR != osh.u32(0) or WAVE_STRATIFIED_PRIMARY_RESTIR != osh.u32(0)) else osh.minimum(push.environment_samples, osh.u32(4))
        if environment_samples > osh.u32(0):
            environment_direct = osh.vec3(0.0)
            sample_index = osh.u32(0)
            while sample_index < environment_samples:
                contribution = sampleEnvironment(position, normal, incoming, material, rng, environment_samples)
                environment_direct = environment_direct + contribution
                primary_specular = primary_specular + contribution * transportLastSpecularFraction / osh.f32(environment_samples)
                sample_index = sample_index + 1
            path.radiance.rgb = path.radiance.rgb + environment_direct / osh.f32(environment_samples)
        transportPbrPrepared = False
        bsdf_weight = osh.vec3(0)
        samplePbr(material, normal, incoming, rng, next_direction, bsdf_weight, bsdf_pdf, sampled_specular)
        indirect_specular_fraction = transportLastSpecularFraction
        path.throughput.rgb = ordinarylight_primary_apply_bsdf_weight(path.throughput.rgb, bsdf_weight)
        cone_spread = ordinarylight_primary_scattered_cone_spread(cone_spread, material.base_roughness.a)
    next_direction = ordinarylight_primary_continuation_direction(next_direction)
    setPathBounce(path, osh.u32(1))
    path.metadata.w = ordinarylight_primary_continuation_flags(path.metadata.w, medium_depth, transmission, push.restir_di != osh.u32(0) and WAVE_UNIFIED_PRIMARY_RESTIR != osh.u32(0))
    setPathPreviousPdf(path, ordinarylight_primary_previous_pdf(pathPreviousPdf(path), bsdf_pdf, transmission))
    setPathRng(path, rng)
    capture_secondary = osh.boolean(False)
    if osh.specialization('WAVE_DENOISER_SIGNAL_CAPTURE'):
        capture_secondary = path.metadata.w & PATH_INDIRECT_CAPTURE_BIT != osh.u32(0)
    else:
        capture_secondary = ordinarylight_primary_capture_secondary(path.metadata.w, transmission)
    if capture_secondary:
        ordinarylight_store_secondary_primary(path_index, path.throughput.rgb, path.radiance.rgb, bsdf_pdf, sampled_specular, pbrSpecularProbability(material), position, material.base_roughness.a, primitive, barycentrics, instance_key)
        if osh.specialization('WAVE_DENOISER_SIGNAL_CAPTURE'):
            if material.attenuation_transmission.a > 0.001:
                secondary_paths[path_index].primary_geometry.y = osh.uint_bits_to_float(osh.float_bits_to_uint(secondary_paths[path_index].primary_geometry.y) | osh.u32(2147483648))
        secondary_paths[path_index].specular_radiance_hit_distance = osh.vec4(primary_specular, -1.0)
        secondary_paths[path_index].diffuse_radiance_hit_distance = osh.vec4(indirect_specular_fraction, -1.0)
    ordinarylight_store_path(path_index, path)
    continuation_origin = ordinarylight_primary_continuation_origin(position, next_direction)
    continuation_direction = next_direction
    if osh.specialization('WAVE_MEGAKERNEL || WAVE_HYBRID'):
        traceRemaining(path, continuation_origin, continuation_direction, path_index, medium_depth, rng, cone_width, cone_spread)
        ordinarylight_store_path(path_index, path)
        if osh.specialization('WAVE_MEGAKERNEL'):
            return
    if osh.specialization('!WAVE_MEGAKERNEL'):
        if path.metadata.w & PATH_ACTIVE_BIT == osh.u32(0):
            return
        output_index = reserveOutputIndex()
        queued_origin = ordinarylight_primary_ray_origin_payload(continuation_origin)
        queued_direction = ordinarylight_primary_ray_direction_payload(continuation_direction)
        if not ordinarylight_enqueue_continuation(output_index, queued_origin, queued_direction, path_index, cone_width, cone_spread):
            ordinarylight_deactivate_stored_path(path_index)
            return

@osh.function
def main() -> osh.void:
    if osh.specialization('WAVE_RAYGEN'):
        pixel = gl_LaunchIDEXT.xy
        processPrimaryPixel(pixel)
    elif osh.specialization('WAVE_PERSISTENT_COARSE'):
        ordinarylight_persistent_coarse_schedule(push.tile_frame.xy)
    else:
        group_id = ordinarylight_primary_scheduled_group(gl_WorkGroupID.xy, gl_NumWorkGroups.xy, osh.u32(WAVE_GROUP_SWIZZLE_WIDTH))
        processPrimaryPixel(group_id * gl_WorkGroupSize.xy + gl_LocalInvocationID.xy)

@osh.structure
class PrimaryConstants:
    image_tile: osh.uvec4
    tile_frame: osh.uvec4
    max_bounces: osh.u32
    point_light_count: osh.u32
    area_light_count: osh.u32
    area_light_samples: osh.u32
    secondary_area_light_samples: osh.u32
    area_light_weight: osh.f32
    gbuffer_enabled: osh.u32
    environment_samples: osh.u32
    subgroup_enqueue: osh.u32
    russian_roulette_start: osh.u32
    russian_roulette_min_survival: osh.f32
    secondary_nee_probability: osh.f32
    inline_bounces: osh.u32
    restir_di: osh.u32
    restir_history_valid: osh.u32
    restir_history_limit: osh.u32
    restir_candidate_count: osh.u32
    restir_spatial_reuse: osh.u32
    restir_spatial_neighbors: osh.u32
    restir_spatial_radius: osh.u32
    restir_pairwise_mis: osh.u32
    restir_generalized_mis: osh.u32
    restir_generalized_balance_cap: osh.f32
    unified_secondary_nee: osh.u32
    unified_primary_restir: osh.u32
    stratified_primary_restir: osh.u32
    indirect_secondary_capture: osh.u32
    indirect_capture_stride: osh.u32
    object_effect_ranges: osh.array(osh.uvec4, 2)

@osh.structure
class CameraData:
    origin: osh.vec4
    forward: osh.vec4
    right: osh.vec4
    up: osh.vec4
    restir_policy: osh.uvec4

@osh.structure
class OutputQueue:
    count: osh.u32
    capacity: osh.u32
    overflow: osh.u32
    queue_padding: osh.u32
