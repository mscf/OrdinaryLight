"""Generate complete Ordinary Light shader stages with Ordinary Shade."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
DEFAULT_ORDINARYSHADE = ROOT.parent / "ordinaryshade"
if DEFAULT_ORDINARYSHADE.is_dir():
    sys.path.insert(0, str(DEFAULT_ORDINARYSHADE))

import ordinaryshade as osh
from ordinaryshade_library import (
    acesApproximation, atrousKernel, decodeAtrousNormal, fpsOverlay,
    linearToSrgb,
    nv12ByteValue, nv12Chroma, nv12Luma, nv12Pixel, p010Chroma, p010Color,
    IndirectLightReservoir, IndirectLightSample, emptyIndirectLightReservoir,
    indirectDecodeNormal, indirectEncodeNormal, indirectPackRgb9e5,
    indirectUnpackRgb9e5, loadIndirectLightReservoir, overlayDigitMask,
    overlayDigitPixel, overlayLetterPixel, p010Luma,
    p010SourcePixel, p010TenBitValue, pack2x16, pack4Bytes, waveHash,
    storeIndirectLightReservoir, waveRandomFloat,
)
from generate_effect_shaders import (
    ordinarylight_emissive, ordinarylight_isolation, ordinarylight_outline,
    ordinarylight_tint,
)


@osh.structure
class IndirectClearConstants:
    reservoir_count: osh.u32


@osh.structure
class WorkQueue:
    count: osh.u32
    capacity: osh.u32
    overflow: osh.u32
    queue_padding: osh.u32


@osh.structure
class DispatchArguments:
    group_count_x: osh.u32
    group_count_y: osh.u32
    group_count_z: osh.u32


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


@osh.structure
class PathToHdrCamera:
    origin: osh.vec4
    forward: osh.vec4
    right: osh.vec4
    up: osh.vec4


@osh.structure
class PathToHdrConstants:
    path_count: osh.u32
    image_width: osh.u32
    image_height: osh.u32
    sample_index: osh.u32
    sample_count: osh.u32
    indirect_secondary_capture: osh.u32
    reservoir_width: osh.u32
    reservoir_height: osh.u32


@osh.structure
class CandidateCamera:
    origin: osh.vec4
    forward: osh.vec4
    right: osh.vec4
    up: osh.vec4


@osh.structure
class CandidateConstants:
    source_width: osh.u32
    source_height: osh.u32
    reservoir_width: osh.u32
    reservoir_height: osh.u32
    history_valid: osh.u32
    frame_index: osh.u32
    spatial_enabled: osh.u32
    profiling_enabled: osh.u32
    history_limit: osh.u32


@osh.structure
class CandidateMergeResult:
    reservoir: IndirectLightReservoir
    history_was_clamped: osh.boolean


@osh.structure
class CandidateProjection:
    valid: osh.boolean
    reservoir_index: osh.u32
    rejection: osh.u32


@osh.structure
class CandidateCompatibility:
    valid: osh.boolean
    rejection: osh.u32


@osh.structure
class IndirectCorrectionResult:
    valid: osh.boolean
    correction: osh.vec3
    confidence: osh.f32


@osh.structure
class DebugConstants:
    output_width: osh.u32
    output_height: osh.u32
    reservoir_width: osh.u32
    reservoir_height: osh.u32
    mode: osh.u32
    history_limit: osh.u32
    apply_strength: osh.f32


@osh.structure
class ReconstructConstants:
    exposure: osh.f32
    source_width: osh.u32
    source_height: osh.u32
    temporal_enabled: osh.u32
    history_weight: osh.f32
    history_valid: osh.u32
    diffuse_filter_enabled: osh.u32
    diffuse_filter_strength: osh.f32
    variance_confidence_enabled: osh.u32
    variance_confidence_strength: osh.f32
    material_confidence_enabled: osh.u32
    transmission_history_scale: osh.f32
    reprojection_search_enabled: osh.u32
    outlier_confidence_enabled: osh.u32
    outlier_confidence_strength: osh.f32
    effect_padding: osh.u32
    kind0: osh.u32
    radius0: osh.u32
    strength0: osh.f32
    object0: osh.u32
    kind1: osh.u32
    radius1: osh.u32
    strength1: osh.f32
    object1: osh.u32
    kind2: osh.u32
    radius2: osh.u32
    strength2: osh.f32
    object2: osh.u32
    kind3: osh.u32
    radius3: osh.u32
    strength3: osh.f32
    object3: osh.u32
    color0: osh.vec4
    color1: osh.vec4
    color2: osh.vec4
    color3: osh.vec4
    rect0: osh.vec4
    rect1: osh.vec4
    rect2: osh.vec4
    rect3: osh.vec4


@osh.structure
class ReprojectionResult:
    valid: osh.boolean
    pixel: osh.vec2
    depth: osh.f32


@osh.structure
class EffectSettings:
    kind: osh.u32
    radius: osh.u32
    strength: osh.f32
    color: osh.vec4
    rect: osh.vec4


@osh.structure
class ResolvedPixel:
    radiance: osh.vec4
    metadata: osh.uvec4


@osh.structure
class ResolveConstants:
    path_count: osh.u32
    padding_a: osh.u32
    padding_b: osh.u32
    padding_c: osh.u32


@osh.structure
class ToneMapConstants:
    path_count: osh.u32
    exposure: osh.f32
    padding_a: osh.u32
    padding_b: osh.u32


@osh.structure
class ToneMapImageConstants:
    path_count: osh.u32
    exposure: osh.f32
    image_width: osh.u32
    image_height: osh.u32


@osh.structure
class Nv12Constants:
    width: osh.u32
    height: osh.u32
    pitch: osh.u32
    reserved: osh.u32


@osh.structure
class P010Constants:
    width: osh.u32
    height: osh.u32
    pitch: osh.u32
    exposure: osh.f32


@osh.structure
class DenoiseCamera:
    origin: osh.vec4
    forward: osh.vec4
    right: osh.vec4
    up: osh.vec4
    overlay: osh.vec4
    accumulation: osh.vec4
    previous_origin: osh.vec4
    previous_forward: osh.vec4
    previous_right: osh.vec4
    previous_up: osh.vec4
    lighting: osh.vec4


@osh.structure
class WaveRay:
    origin_tmin: osh.vec4
    direction_tmax: osh.vec4
    path_index: osh.u32
    padding_a: osh.u32
    padding_b: osh.u32
    padding_c: osh.u32


@osh.structure
class RayQueue:
    count: osh.u32
    capacity: osh.u32
    overflow: osh.u32
    queue_padding: osh.u32
    rays: osh.runtime_array(WaveRay)


@osh.structure
class WaveHit:
    position_t: osh.vec4
    geometric_normal: osh.vec3
    primitive_index: osh.u32
    barycentrics: osh.vec2
    ray_index: osh.u32
    path_index: osh.u32


@osh.structure
class HitQueue:
    count: osh.u32
    capacity: osh.u32
    overflow: osh.u32
    queue_padding: osh.u32
    hits: osh.runtime_array(WaveHit)


@osh.structure
class ShadeMediumStack:
    ior_0_3: osh.vec4
    ior_4_7: osh.vec4
    ior_8_11: osh.vec4
    ior_12_15: osh.vec4


@osh.structure
class ShadeConstants:
    max_bounces: osh.u32
    point_light_count: osh.u32
    area_light_count: osh.u32
    area_light_samples: osh.u32
    secondary_area_light_samples: osh.u32
    area_light_weight: osh.f32
    environment_samples: osh.u32
    russian_roulette_start: osh.u32
    russian_roulette_min_survival: osh.f32
    fused_intersection: osh.u32
    subgroup_enqueue: osh.u32
    secondary_nee_probability: osh.f32
    unified_secondary_nee: osh.u32
    indirect_secondary_capture: osh.u32


@osh.structure
class VolumeHeader:
    world_to_local: osh.mat4
    dimensions_offset: osh.uvec4
    value_parameters: osh.vec4
    render_parameters: osh.vec4
    scattering_parameters: osh.vec4
    phase_parameters: osh.vec4
    multiple_scattering_parameters: osh.vec4
    acceleration_parameters: osh.uvec4


@osh.structure
class BucketRayQueue:
    ray_count: osh.u32
    ray_capacity: osh.u32
    ray_overflow: osh.u32
    ray_padding: osh.u32
    rays: osh.runtime_array(WaveRay)


@osh.structure
class PlainHitQueue:
    plain_count: osh.u32
    plain_capacity: osh.u32
    plain_overflow: osh.u32
    plain_padding: osh.u32
    plain_hits: osh.runtime_array(WaveHit)


@osh.structure
class TexturedHitQueue:
    textured_count: osh.u32
    textured_capacity: osh.u32
    textured_overflow: osh.u32
    textured_padding: osh.u32
    textured_hits: osh.runtime_array(WaveHit)


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


@osh.structure
class RayQueryCamera:
    origin: osh.vec4
    forward: osh.vec4
    right: osh.vec4
    up: osh.vec4
    overlay: osh.vec4


@osh.structure
class RayQueryImageCamera:
    origin: osh.vec4
    forward: osh.vec4
    right: osh.vec4
    up: osh.vec4
    overlay: osh.vec4
    accumulation: osh.vec4
    previous_origin: osh.vec4
    previous_forward: osh.vec4
    previous_right: osh.vec4
    previous_up: osh.vec4
    lighting: osh.vec4


@osh.structure
class PrimaryRayResult:
    origin: osh.vec3
    direction: osh.vec3


@osh.structure
class AreaLightSampleResult:
    radiance: osh.vec3
    random_state: osh.u32


@osh.structure
class RayQueryTraceResult:
    radiance: osh.vec3
    primary_depth: osh.f32
    primary_normal: osh.vec3
    primary_id: osh.f32


@osh.structure
class PbrSampleResult:
    outgoing: osh.vec3
    weight: osh.vec3
    pdf: osh.f32
    random_state: osh.u32
    sampled_specular: osh.boolean


@osh.structure
class PbrLobeResult:
    diffuse: osh.vec3
    specular: osh.vec3


@osh.structure
class ShadeSurface:
    material: MaterialData
    normal: osh.vec3
    geometric_normal: osh.vec3
    weights: osh.vec3
    uv: osh.vec2
    vertex_a: osh.vec3
    vertex_b: osh.vec3
    vertex_c: osh.vec3
    entering: osh.boolean


@osh.structure
class ShadeHitInput:
    valid: osh.boolean
    hit: WaveHit
    ray: WaveRay


@osh.structure
class ShadeMissResult:
    path: WavePathState
    environment_mis: osh.f32


@osh.structure
class ShadeTransmissionResult:
    path: WavePathState
    stack: ShadeMediumStack
    direction: osh.vec3
    medium_depth: osh.u32


@osh.structure
class ShadeRouletteResult:
    path: WavePathState
    random_state: osh.u32
    survived: osh.boolean


@osh.structure
class ShadeContinuationResult:
    path: WavePathState
    ray: WaveRay


@osh.structure
class ShadeEnqueueResult:
    path: WavePathState
    enqueued: osh.boolean


@osh.structure
class ShadePointLightSample:
    direction: osh.vec3
    shadow_origin: osh.vec3
    incident: osh.vec3
    cosine: osh.f32
    shadow_distance: osh.f32
    valid: osh.boolean


@osh.structure
class ShadeAreaLightSample:
    direction: osh.vec3
    shadow_origin: osh.vec3
    emission: osh.vec3
    surface_cosine: osh.f32
    effective_pdf: osh.f32
    sample_count: osh.f32
    shadow_distance: osh.f32
    random_state: osh.u32
    valid: osh.boolean


@osh.structure
class ShadeEmissionResult:
    contribution: osh.vec3
    mis_weight: osh.f32


@osh.structure
class ShadeEnvironmentSample:
    direction: osh.vec3
    shadow_origin: osh.vec3
    cosine: osh.f32
    effective_pdf: osh.f32
    sample_count: osh.f32
    random_state: osh.u32
    valid: osh.boolean


@osh.structure
class ShadeUnifiedDomainSelection:
    random_state: osh.u32
    area_probability: osh.f32
    area_selected: osh.boolean
    valid: osh.boolean


@osh.structure
class ShadeEnvironmentDescriptor:
    color_intensity: osh.vec4
    texture_parameters: osh.vec4
    valid: osh.boolean


@osh.structure
class ShadeOpaqueScatterResult:
    path: WavePathState
    direction: osh.vec3
    pdf: osh.f32
    random_state: osh.u32
    cone_spread: osh.f32
    sampled_specular: osh.boolean


@osh.structure
class ShadeVolumeMarchState:
    integrated: osh.vec3
    transmittance: osh.f32


@osh.structure
class ShadeVolumeIntegrationResult:
    state: ShadeVolumeMarchState
    exit_distance: osh.f32


@osh.structure
class ShadeVolumeVoxelRay:
    origin: osh.vec3
    direction: osh.vec3


@osh.structure
class ShadeVolumeBrickStep:
    exit_distance: osh.f32
    occupied: osh.boolean


@osh.structure
class ShadeVolumeUnionBounds:
    entry: osh.f32
    exit_distance: osh.f32
    reference_step: osh.f32
    valid: osh.boolean


@osh.structure
class ShadeOverlappingMedium:
    extinction: osh.f32
    emission_extinction: osh.vec3


@osh.structure
class CameraData:
    camera_origin: osh.vec4
    camera_forward: osh.vec4
    camera_right: osh.vec4
    camera_up: osh.vec4


@osh.structure
class GenerateConstants:
    image_tile: osh.uvec4
    tile_frame: osh.uvec4


@osh.function
def candidateInstrumentedPixel() -> osh.boolean:
    index = (
        osh.global_invocation_id.y * push.reservoir_width
        + osh.global_invocation_id.x
    )
    return (
        push.profiling_enabled != osh.u32(0)
        and (index & osh.u32(63)) == osh.u32(0)
    )


@osh.function
def candidateCountEvent(event: osh.u32) -> osh.void:
    if candidateInstrumentedPixel():
        osh.atomic_add(counters[event], osh.u32(1))


@osh.function
def candidateRejectionDebugFlag(counter: osh.u32) -> osh.u32:
    if counter == osh.u32(5) or counter == osh.u32(11):
        return osh.u32(1) << osh.u32(10)
    if counter == osh.u32(6) or counter == osh.u32(12):
        return osh.u32(1) << osh.u32(11)
    if counter == osh.u32(7) or counter == osh.u32(13):
        return osh.u32(1) << osh.u32(12)
    return osh.u32(1) << osh.u32(13)


@osh.function
def candidateLoadPreviousReservoir(
    reservoir_index: osh.u32, camera_origin: osh.vec3,
) -> IndirectLightReservoir:
    word = reservoir_index * osh.u32(6)
    header = previous_reservoir_words[word + osh.u32(5)]
    sample_count = header & osh.u32(0x7F)
    if (
        (header & osh.u32(0x80000000)) == osh.u32(0)
        or sample_count == osh.u32(0)
    ):
        return emptyIndirectLightReservoir()
    position_xy = osh.unpack_half2x16(previous_reservoir_words[word])
    position_z_pdf = osh.unpack_half2x16(
        previous_reservoir_words[word + osh.u32(1)]
    )
    weight_target = osh.unpack_half2x16(
        previous_reservoir_words[word + osh.u32(4)]
    )
    reservoir = emptyIndirectLightReservoir()
    reservoir.selected.secondary_position = camera_origin + osh.vec3(
        position_xy, position_z_pdf.x
    )
    reservoir.selected.proposal_pdf = position_z_pdf.y
    reservoir.selected.secondary_normal = indirectDecodeNormal(
        osh.unpack_unorm2x16(
            previous_reservoir_words[word + osh.u32(2)]
        ) * 2.0 - 1.0
    )
    reservoir.selected.target = weight_target.y
    reservoir.selected.radiance = indirectUnpackRgb9e5(
        previous_reservoir_words[word + osh.u32(3)]
    )
    reservoir.weight_sum = weight_target.x
    reservoir.sample_count = sample_count
    reservoir.valid = True
    return reservoir


@osh.function
def candidateRandom(pixel: osh.uvec2, salt: osh.u32) -> osh.f32:
    frame_index = osh.u32(camera.origin.w + 0.5)
    value = (
        pixel.x * osh.u32(0x9E3779B9)
        ^ pixel.y * osh.u32(0x85EBCA6B)
        ^ frame_index * osh.u32(0xC2B2AE35)
        ^ salt * osh.u32(0x27D4EB2D)
    )
    value = value ^ (value >> osh.u32(16))
    value = value * osh.u32(0x7FEB352D)
    value = value ^ (value >> osh.u32(15))
    return osh.f32(value) * (1.0 / 4294967296.0)


@osh.function
def candidateMergePrevious(
    reservoir: IndirectLightReservoir,
    old: IndirectLightReservoir,
    current_target: osh.f32,
    reservoir_pixel: osh.uvec2,
    salt: osh.u32,
    history_was_clamped: osh.boolean,
) -> CandidateMergeResult:
    if not old.valid or old.selected.target <= 0.0:
        return CandidateMergeResult(reservoir, history_was_clamped)
    reuse_weight = current_target * old.weight_sum / old.selected.target
    combined_weight = reservoir.weight_sum + reuse_weight
    if (
        reuse_weight > 0.0
        and candidateRandom(reservoir_pixel, salt) * combined_weight
        < reuse_weight
    ):
        reservoir.selected = old.selected
        reservoir.selected.target = current_target
    reservoir.weight_sum = combined_weight
    reservoir.sample_count = reservoir.sample_count + old.sample_count
    if reservoir.sample_count > push.history_limit:
        scale = osh.f32(push.history_limit) / osh.f32(reservoir.sample_count)
        reservoir.weight_sum = reservoir.weight_sum * scale
        reservoir.sample_count = push.history_limit
        history_was_clamped = True
    return CandidateMergeResult(reservoir, history_was_clamped)


@osh.function
def candidateDecodePrimaryNormal(packed: osh.u32) -> osh.vec3:
    encoded = osh.vec2(
        osh.f32(packed & osh.u32(0x7FFF)),
        osh.f32((packed >> osh.u32(15)) & osh.u32(0x7FFF)),
    ) / 32767.0 * 2.0 - 1.0
    return indirectDecodeNormal(encoded)


@osh.function
def candidatePrimaryWorldPosition(
    pixel: osh.ivec2, ray_distance: osh.f32,
) -> osh.vec3:
    ndc = (
        (osh.vec2(pixel) + 0.5)
        / osh.vec2(push.source_width, push.source_height)
    ) * 2.0 - 1.0
    aspect = osh.f32(push.source_width) / osh.f32(push.source_height)
    direction = osh.normalize(
        camera.forward.xyz + ndc.x * aspect * camera.right.xyz
        - ndc.y * camera.up.xyz
    )
    return camera.origin.xyz + direction * ray_distance


@osh.function
def candidatePreviousWorldPosition(
    pixel: osh.ivec2, ray_distance: osh.f32,
) -> osh.vec3:
    ndc = (
        (osh.vec2(pixel) + 0.5)
        / osh.vec2(push.source_width, push.source_height)
    ) * 2.0 - 1.0
    aspect = osh.f32(push.source_width) / osh.f32(push.source_height)
    direction = osh.normalize(
        previous_camera.forward.xyz
        + ndc.x * aspect * previous_camera.right.xyz
        - ndc.y * previous_camera.up.xyz
    )
    return previous_camera.origin.xyz + direction * ray_distance


@osh.function
def candidateSecondaryVisible(
    primary_position: osh.vec3,
    primary_normal: osh.vec3,
    secondary_position: osh.vec3,
) -> osh.boolean:
    offset = secondary_position - primary_position
    connection_distance = osh.length(offset)
    if connection_distance <= 0.006:
        return True
    direction = offset / connection_distance
    query = osh.ray_query()
    query.initialize(
        scene_tlas, osh.u32(5), osh.u32(0x01),
        primary_position + primary_normal * 0.002, 0.001,
        direction, osh.maximum(connection_distance - 0.004, 0.001),
    )
    while query.proceed():
        pass
    return query.intersection_type(True) == osh.u32(0)


@osh.function
def candidateReprojectPrevious(
    world_position: osh.vec3,
    world_normal: osh.vec3,
    material_signature: osh.u32,
) -> CandidateProjection:
    relative = world_position - previous_camera.origin.xyz
    forward_distance = osh.dot(relative, previous_camera.forward.xyz)
    if forward_distance <= 0.0:
        return CandidateProjection(False, osh.u32(0), osh.u32(5))
    aspect = osh.f32(push.source_width) / osh.f32(push.source_height)
    ndc_x = (
        osh.dot(relative, previous_camera.right.xyz)
        / osh.maximum(
            osh.dot(previous_camera.right.xyz, previous_camera.right.xyz),
            1.0e-8,
        ) / forward_distance / aspect
    )
    ndc_y = (
        -osh.dot(relative, previous_camera.up.xyz)
        / osh.maximum(
            osh.dot(previous_camera.up.xyz, previous_camera.up.xyz), 1.0e-8
        ) / forward_distance
    )
    previous_coordinate = (
        (osh.vec2(ndc_x, ndc_y) * 0.5 + 0.5)
        * osh.vec2(push.source_width, push.source_height) - 0.5
    )
    previous_pixel = osh.ivec2(osh.round(previous_coordinate))
    if (
        osh.any_value(previous_pixel < osh.ivec2(0))
        or osh.any_value(previous_pixel >= osh.ivec2(
            push.source_width, push.source_height
        ))
    ):
        return CandidateProjection(False, osh.u32(0), osh.u32(5))
    old_distance = previous_position.load(previous_pixel).x
    if not (old_distance > 0.0) or not (old_distance <= 3.402823e38):
        return CandidateProjection(False, osh.u32(0), osh.u32(5))
    old_position = candidatePreviousWorldPosition(previous_pixel, old_distance)
    position_tolerance = osh.maximum(0.02, old_distance * 0.02)
    if osh.length(old_position - world_position) > position_tolerance:
        return CandidateProjection(False, osh.u32(0), osh.u32(5))
    old_normal = candidateDecodePrimaryNormal(
        previous_normal.load(previous_pixel).x
    )
    if osh.dot(old_normal, world_normal) < 0.9:
        return CandidateProjection(False, osh.u32(0), osh.u32(6))
    if previous_material.load(previous_pixel).x != material_signature:
        return CandidateProjection(False, osh.u32(0), osh.u32(7))
    reservoir_pixel = osh.minimum(
        osh.uvec2(
            (osh.vec2(previous_pixel) + 0.5)
            * osh.vec2(push.reservoir_width, push.reservoir_height)
            / osh.vec2(push.source_width, push.source_height)
        ),
        osh.uvec2(
            push.reservoir_width - osh.u32(1),
            push.reservoir_height - osh.u32(1),
        ),
    )
    return CandidateProjection(
        True,
        reservoir_pixel.y * push.reservoir_width + reservoir_pixel.x,
        osh.u32(5),
    )


@osh.function
def candidateSpatialCompatibility(
    previous_reservoir_pixel: osh.uvec2,
    world_position: osh.vec3,
    world_normal: osh.vec3,
    material_signature: osh.u32,
) -> CandidateCompatibility:
    previous_pixel = osh.clamp(
        osh.ivec2(
            (osh.vec2(previous_reservoir_pixel) + 0.5)
            * osh.vec2(push.source_width, push.source_height)
            / osh.vec2(push.reservoir_width, push.reservoir_height)
        ),
        osh.ivec2(0),
        osh.ivec2(push.source_width, push.source_height) - 1,
    )
    old_distance = previous_position.load(previous_pixel).x
    if not (old_distance > 0.0) or not (old_distance <= 3.402823e38):
        return CandidateCompatibility(False, osh.u32(11))
    if previous_material.load(previous_pixel).x != material_signature:
        return CandidateCompatibility(False, osh.u32(13))
    old_normal = candidateDecodePrimaryNormal(
        previous_normal.load(previous_pixel).x
    )
    if osh.dot(old_normal, world_normal) < 0.9:
        return CandidateCompatibility(False, osh.u32(12))
    old_position = candidatePreviousWorldPosition(previous_pixel, old_distance)
    tolerance = osh.maximum(
        0.05,
        osh.length(world_position - previous_camera.origin.xyz) * 0.05,
    )
    return CandidateCompatibility(
        osh.length(old_position - world_position) <= tolerance, osh.u32(11)
    )


@osh.function
def indirectAcceptanceColor(flags: osh.u32) -> osh.vec3:
    temporal = (flags & (osh.u32(1) << osh.u32(8))) != osh.u32(0)
    spatial = (flags & (osh.u32(1) << osh.u32(9))) != osh.u32(0)
    position_rejected = (
        flags & (osh.u32(1) << osh.u32(10))
    ) != osh.u32(0)
    normal_rejected = (
        flags & (osh.u32(1) << osh.u32(11))
    ) != osh.u32(0)
    material_rejected = (
        flags & (osh.u32(1) << osh.u32(12))
    ) != osh.u32(0)
    empty = (flags & (osh.u32(1) << osh.u32(13))) != osh.u32(0)
    color = osh.vec3(0.04)
    if temporal:
        color.g = color.g + 0.7
    if spatial:
        color.b = color.b + 0.7
    if position_rejected:
        color.r = color.r + 0.8
    if normal_rejected:
        color = color + osh.vec3(0.7, 0.7, 0.0)
    if material_rejected:
        color = color + osh.vec3(0.7, 0.0, 0.7)
    if empty:
        color = color + osh.vec3(0.8, 0.25, 0.0)
    return osh.minimum(color, osh.vec3(1.0))


@osh.function
def indirectInvalidColor(pixel: osh.uvec2) -> osh.vec3:
    bright = (
        ((pixel.x >> osh.u32(4)) ^ (pixel.y >> osh.u32(4))) & osh.u32(1)
    ) != osh.u32(0)
    return (
        osh.vec3(1.0, 0.0, 1.0)
        if bright else osh.vec3(0.12, 0.0, 0.12)
    )


@osh.function
def indirectCorrection(
    reservoir_pixel: osh.ivec2, output_material: osh.u32,
) -> IndirectCorrectionResult:
    invalid = IndirectCorrectionResult(False, osh.vec3(0.0), 0.0)
    if (
        osh.any_value(reservoir_pixel < osh.ivec2(0))
        or osh.any_value(reservoir_pixel >= osh.ivec2(
            push.reservoir_width, push.reservoir_height
        ))
    ):
        return invalid
    reservoir_index = (
        osh.u32(reservoir_pixel.y) * push.reservoir_width
        + osh.u32(reservoir_pixel.x)
    )
    reservoir = loadIndirectLightReservoir(
        reservoir_index, osh.vec3(0.0)
    )
    if not reservoir.valid or reservoir.sample_count <= osh.u32(1):
        return invalid
    representative = osh.clamp(
        osh.ivec2(
            (osh.vec2(reservoir_pixel) + 0.5)
            * osh.vec2(push.output_width, push.output_height)
            / osh.vec2(push.reservoir_width, push.reservoir_height)
        ),
        osh.ivec2(0),
        osh.ivec2(push.output_width, push.output_height) - 1,
    )
    if (
        output_material == osh.u32(0xFFFFFFFF)
        or output_material != material_image.load(representative).x
    ):
        return invalid
    normalization = reservoir.weight_sum / osh.maximum(
        osh.f32(reservoir.sample_count) * reservoir.selected.target,
        1.0e-6,
    )
    reconstructed = reservoir.selected.radiance * osh.clamp(
        normalization, 0.25, 4.0
    )
    current_seed = indirectUnpackRgb9e5(
        indirect_seed_words[reservoir_index]
    )
    return IndirectCorrectionResult(
        True,
        reconstructed - current_seed,
        1.0 - 1.0 / osh.f32(reservoir.sample_count),
    )


@osh.function
def reconstructSmoothstep(a: osh.f32, b: osh.f32, value: osh.f32) -> osh.f32:
    t = osh.clamp((value - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


@osh.function
def reconstructDecodeNormal(encoded: osh.vec2) -> osh.vec3:
    normal = osh.vec3(
        encoded, 1.0 - osh.absolute(encoded.x) - osh.absolute(encoded.y)
    )
    if normal.z < 0.0:
        normal.xy = (1.0 - osh.absolute(normal.yx)) * osh.sign(normal.xy)
    return osh.normalize(normal)


@osh.function
def reconstructUnpackNormal(packed: osh.u32) -> osh.vec4:
    unit = osh.vec2(
        osh.f32(packed & osh.u32(0x7FFF)),
        osh.f32((packed >> osh.u32(15)) & osh.u32(0x7FFF)),
    ) / 32767.0
    return osh.vec4(
        reconstructDecodeNormal(unit * 2.0 - 1.0),
        osh.f32(packed >> osh.u32(30)),
    )


@osh.function
def reconstructClampSource(pixel: osh.ivec2) -> osh.ivec2:
    return osh.clamp(
        pixel, osh.ivec2(0),
        osh.ivec2(push.source_width, push.source_height) - 1,
    )


@osh.function
def reconstructLoadHdr(pixel: osh.ivec2) -> osh.vec3:
    return hdr_image.load(reconstructClampSource(pixel)).rgb


@osh.function
def reconstructEffectSlot(pixel: osh.ivec2) -> osh.u32:
    signature = material_image.load(reconstructClampSource(pixel)).x
    return (
        osh.u32(0) if signature == osh.u32(0xFFFFFFFF)
        else signature >> osh.u32(29)
    )


@osh.function
def reconstructWorldPosition(
    origin: osh.vec4, forward: osh.vec4, right: osh.vec4, up: osh.vec4,
    pixel: osh.ivec2, size: osh.ivec2, ray_distance: osh.f32,
) -> osh.vec3:
    ndc = ((osh.vec2(pixel) + 0.5) / osh.vec2(size)) * 2.0 - 1.0
    aspect = osh.f32(size.x) / osh.f32(size.y)
    direction = osh.normalize(
        forward.xyz + ndc.x * aspect * right.xyz - ndc.y * up.xyz
    )
    return origin.xyz + direction * ray_distance


@osh.function
def reconstructCurrentPosition(pixel: osh.ivec2, size: osh.ivec2) -> osh.vec4:
    ray_distance = position_image.load(pixel).x
    return osh.vec4(
        reconstructWorldPosition(
            current_camera.origin, current_camera.forward,
            current_camera.right, current_camera.up,
            pixel, size, ray_distance,
        ),
        ray_distance,
    )


@osh.function
def reconstructPreviousPosition(pixel: osh.ivec2, size: osh.ivec2) -> osh.vec4:
    ray_distance = previous_position.load(pixel).x
    return osh.vec4(
        reconstructWorldPosition(
            previous_camera.origin, previous_camera.forward,
            previous_camera.right, previous_camera.up,
            pixel, size, ray_distance,
        ),
        ray_distance,
    )


@osh.function
def reconstructBilinearHdr(source: osh.vec2) -> osh.vec3:
    floored = osh.floor(source)
    base = osh.ivec2(floored)
    fraction = source - floored
    top = osh.mix(
        reconstructLoadHdr(base), reconstructLoadHdr(base + osh.ivec2(1, 0)),
        fraction.x,
    )
    bottom = osh.mix(
        reconstructLoadHdr(base + osh.ivec2(0, 1)),
        reconstructLoadHdr(base + osh.ivec2(1, 1)), fraction.x,
    )
    return osh.mix(top, bottom, fraction.y)


@osh.function
def reconstructLuminance(color: osh.vec3) -> osh.f32:
    return osh.dot(color, osh.vec3(0.2126, 0.7152, 0.0722))


@osh.function
def reconstructReproject(
    world_position: osh.vec3, output_size: osh.ivec2,
) -> ReprojectionResult:
    offset = world_position - previous_camera.origin.xyz
    depth = osh.dot(offset, previous_camera.forward.xyz)
    scale = osh.length(previous_camera.up.xyz)
    aspect = osh.f32(output_size.x) / osh.f32(output_size.y)
    if depth <= 0.0001 or scale <= 0.0001:
        return ReprojectionResult(False, osh.vec2(0.0), depth)
    ndc = osh.vec2(
        osh.dot(offset, osh.normalize(previous_camera.right.xyz))
        / (depth * aspect * scale),
        -osh.dot(offset, osh.normalize(previous_camera.up.xyz))
        / (depth * scale),
    )
    pixel = (ndc * 0.5 + 0.5) * osh.vec2(output_size) - 0.5
    valid = (
        not osh.any_value(pixel < osh.vec2(-0.5))
        and not osh.any_value(pixel >= osh.vec2(output_size) - 0.5)
    )
    return ReprojectionResult(valid, pixel, depth)


@osh.function
def reconstructCurrentNormal(pixel: osh.ivec2) -> osh.vec4:
    return reconstructUnpackNormal(normal_image.load(pixel).x)


@osh.function
def reconstructPreviousNormal(pixel: osh.ivec2) -> osh.vec4:
    return reconstructUnpackNormal(previous_normal.load(pixel).x)


@osh.function
def reconstructFilterDiffuse(
    center_pixel: osh.ivec2, center_position: osh.vec4,
    center_normal: osh.vec4, unfiltered: osh.vec3,
) -> osh.vec3:
    weighted_color = osh.vec3(0.0)
    total_weight = 0.0
    luminance_sum = 0.0
    luminance_squared_sum = 0.0
    sample_weight_sum = 0.0
    distance_scale = osh.maximum(0.03, osh.absolute(center_position.w) * 0.02)
    for y in range(-1, 2):
        for x in range(-1, 2):
            sample_pixel = reconstructClampSource(
                center_pixel + osh.ivec2(x, y)
            )
            sample_position = reconstructCurrentPosition(
                sample_pixel, osh.ivec2(push.source_width, push.source_height)
            )
            sample_normal = reconstructCurrentNormal(sample_pixel)
            if (
                sample_position.w < 0.0 or sample_normal.w > 0.5
                or osh.dot(sample_normal.xyz, sample_normal.xyz) < 0.5
            ):
                continue
            normal_weight = osh.power(osh.maximum(osh.dot(
                osh.normalize(center_normal.xyz),
                osh.normalize(sample_normal.xyz),
            ), 0.0), 32.0)
            position_delta = osh.length(
                center_position.xyz - sample_position.xyz
            ) / distance_scale
            position_weight = osh.exp(-position_delta * position_delta)
            kernel_weight = 0.5
            if x == 0 or y == 0:
                kernel_weight = 0.75
            if x == 0 and y == 0:
                kernel_weight = 1.0
            weight = kernel_weight * normal_weight * position_weight
            sample_color = reconstructLoadHdr(sample_pixel)
            weighted_color = weighted_color + sample_color * weight
            total_weight = total_weight + weight
            value = reconstructLuminance(sample_color)
            luminance_sum = luminance_sum + value * kernel_weight
            luminance_squared_sum = (
                luminance_squared_sum + value * value * kernel_weight
            )
            sample_weight_sum = sample_weight_sum + kernel_weight
    if total_weight <= 0.0001 or sample_weight_sum <= 0.0001:
        return unfiltered
    mean = luminance_sum / sample_weight_sum
    variance = osh.maximum(
        luminance_squared_sum / sample_weight_sum - mean * mean, 0.0
    )
    relative_variance = variance / osh.maximum(mean * mean, 0.001)
    activation = push.diffuse_filter_strength * reconstructSmoothstep(
        0.002, 0.05, relative_variance
    )
    return osh.mix(unfiltered, weighted_color / total_weight, activation)


@osh.function
def reconstructEffect(index: osh.i32) -> EffectSettings:
    if index == 0:
        return EffectSettings(
            push.kind0, push.radius0, push.strength0, push.color0, push.rect0
        )
    if index == 1:
        return EffectSettings(
            push.kind1, push.radius1, push.strength1, push.color1, push.rect1
        )
    if index == 2:
        return EffectSettings(
            push.kind2, push.radius2, push.strength2, push.color2, push.rect2
        )
    return EffectSettings(
        push.kind3, push.radius3, push.strength3, push.color3, push.rect3
    )


@osh.function
def reconstructMain() -> osh.void:
    output_pixel = osh.ivec2(osh.global_invocation_id.xy)
    output_index = osh.clamp(osh.i32(current_camera.forward.w + 0.5), 0, 7)
    output_size = output_images[output_index].size()
    if osh.any_value(output_pixel >= output_size):
        return
    source = (
        (osh.vec2(output_pixel) + 0.5)
        * osh.vec2(push.source_width, push.source_height)
        / osh.vec2(output_size) - 0.5
    )
    source_pixel = reconstructClampSource(osh.ivec2(osh.round(source)))
    hdr = reconstructBilinearHdr(source)
    position_data = osh.vec4(0.0, 0.0, 0.0, -1.0)
    normal_data = osh.vec4(0.0)
    needs_gbuffer = (
        push.temporal_enabled != osh.u32(0)
        or push.diffuse_filter_enabled != osh.u32(0)
    )
    if needs_gbuffer:
        position_data = reconstructCurrentPosition(
            source_pixel, osh.ivec2(push.source_width, push.source_height)
        )
        normal_data = reconstructCurrentNormal(source_pixel)
    hit = (
        needs_gbuffer and position_data.w >= 0.0
        and osh.dot(normal_data.xyz, normal_data.xyz) > 0.5
    )
    current_depth = -1.0
    surface_class = -1.0
    if hit:
        current_depth = osh.dot(
            position_data.xyz - current_camera.origin.xyz,
            current_camera.forward.xyz,
        )
        surface_class = normal_data.w
    if (
        push.diffuse_filter_enabled != osh.u32(0)
        and hit and surface_class < 0.5
    ):
        hdr = reconstructFilterDiffuse(
            source_pixel, position_data, normal_data, hdr
        )
    accepted = False
    old_hdr = osh.vec3(0.0)
    history_confidence = 0.0
    if (
        push.temporal_enabled != osh.u32(0)
        and push.history_valid != osh.u32(0) and hit
    ):
        projection = reconstructReproject(position_data.xyz, output_size)
        if projection.valid:
            previous_source = (
                (projection.pixel + 0.5)
                * osh.vec2(push.source_width, push.source_height)
                / osh.vec2(output_size) - 0.5
            )
            previous_source_pixel = osh.ivec2(osh.round(previous_source))
            previous_color_pixel = osh.ivec2(osh.round(projection.pixel))
            candidate_found = True
            if push.reprojection_search_enabled != osh.u32(0):
                candidate_found = False
                best_score = 1.0e30
                source_size = osh.ivec2(push.source_width, push.source_height)
                base = osh.ivec2(osh.floor(previous_source))
                for y in range(2):
                    for x in range(2):
                        candidate = base + osh.ivec2(x, y)
                        if (
                            osh.any_value(candidate < osh.ivec2(0))
                            or osh.any_value(candidate >= source_size)
                        ):
                            continue
                        candidate_position = reconstructPreviousPosition(
                            candidate, source_size
                        )
                        candidate_normal = reconstructPreviousNormal(candidate)
                        if (
                            candidate_position.w < 0.0
                            or osh.dot(
                                candidate_normal.xyz, candidate_normal.xyz
                            ) < 0.5
                            or osh.absolute(
                                candidate_normal.w - surface_class
                            ) >= 0.25
                        ):
                            continue
                        position_error = osh.length(
                            candidate_position.xyz - position_data.xyz
                        )
                        normal_agreement = osh.dot(
                            osh.normalize(normal_data.xyz),
                            osh.normalize(candidate_normal.xyz),
                        )
                        tolerance = osh.maximum(
                            0.03, projection.depth * 0.01
                        )
                        if position_error > tolerance or normal_agreement <= 0.9:
                            continue
                        score = (
                            position_error / tolerance
                            + (1.0 - normal_agreement) * 10.0
                        )
                        if score < best_score:
                            best_score = score
                            candidate_found = True
                            previous_source_pixel = candidate
                            candidate_output = (
                                (osh.vec2(candidate) + 0.5)
                                * osh.vec2(output_size) / osh.vec2(source_size)
                                - 0.5
                            )
                            previous_color_pixel = osh.clamp(
                                osh.ivec2(osh.round(candidate_output)),
                                osh.ivec2(0), output_size - 1,
                            )
            if candidate_found:
                old_position = reconstructPreviousPosition(
                    previous_source_pixel,
                    osh.ivec2(push.source_width, push.source_height),
                )
                old_normal = reconstructPreviousNormal(previous_source_pixel)
                old_color = previous_color.load(previous_color_pixel)
                position_tolerance = osh.maximum(
                    0.03, projection.depth * 0.01
                )
                position_error = osh.length(
                    old_position.xyz - position_data.xyz
                )
                normal_agreement = osh.dot(
                    osh.normalize(normal_data.xyz),
                    osh.normalize(old_normal.xyz),
                )
                accepted = (
                    old_position.w >= 0.0
                    and position_error <= position_tolerance
                    and normal_agreement > 0.9
                    and osh.absolute(old_normal.w - surface_class) < 0.25
                )
                history_confidence = osh.minimum(
                    1.0 - reconstructSmoothstep(
                        0.0, 1.0, position_error / position_tolerance
                    ),
                    reconstructSmoothstep(0.9, 0.99, normal_agreement),
                )
                old_hdr = old_color.rgb
    if accepted:
        neighborhood_min = osh.vec3(1.0e30)
        neighborhood_max = osh.vec3(-1.0e30)
        luminance_sum = 0.0
        luminance_squared_sum = 0.0
        center_sample_luminance = 0.0
        for y in range(-1, 2):
            for x in range(-1, 2):
                sample_value = reconstructLoadHdr(
                    source_pixel + osh.ivec2(x, y)
                )
                neighborhood_min = osh.minimum(neighborhood_min, sample_value)
                neighborhood_max = osh.maximum(neighborhood_max, sample_value)
                sample_luminance = reconstructLuminance(sample_value)
                luminance_sum = luminance_sum + sample_luminance
                luminance_squared_sum = (
                    luminance_squared_sum + sample_luminance * sample_luminance
                )
                if x == 0 and y == 0:
                    center_sample_luminance = sample_luminance
        old_luminance = reconstructLuminance(old_hdr)
        old_hdr = osh.clamp(old_hdr, neighborhood_min, neighborhood_max)
        weight = push.history_weight * history_confidence
        if surface_class > 1.5 and push.material_confidence_enabled != osh.u32(0):
            current_luminance = reconstructLuminance(hdr)
            relative_delta = osh.absolute(
                old_luminance - current_luminance
            ) / osh.maximum(
                osh.maximum(old_luminance, current_luminance), 0.05
            )
            appearance_confidence = 1.0 - reconstructSmoothstep(
                0.05, 0.35, relative_delta
            )
            weight = weight * osh.mix(
                0.05, push.transmission_history_scale,
                appearance_confidence,
            )
        else:
            if surface_class > 0.5:
                weight = weight * 0.25
            else:
                if push.variance_confidence_enabled != osh.u32(0):
                    mean = luminance_sum / 9.0
                    variance = osh.maximum(
                        luminance_squared_sum / 9.0 - mean * mean, 0.0
                    )
                    relative_variance = variance / osh.maximum(
                        mean * mean, 0.001
                    )
                    noisy = reconstructSmoothstep(
                        0.002, 0.05, relative_variance
                    )
                    delta = reconstructLuminance(old_hdr) - reconstructLuminance(hdr)
                    agreement = osh.exp(
                        -(delta * delta) / osh.maximum(
                            4.0 * variance, 0.0005
                        )
                    )
                    adaptive_weight = osh.mix(
                        push.history_weight,
                        1.0 - (1.0 - push.history_weight)
                        * (1.0 - push.variance_confidence_strength),
                        noisy * agreement,
                    )
                    weight = adaptive_weight * history_confidence
        if push.outlier_confidence_enabled != osh.u32(0):
            neighbor_mean = (
                luminance_sum - center_sample_luminance
            ) / 8.0
            neighbor_variance = osh.maximum(
                (luminance_squared_sum
                 - center_sample_luminance * center_sample_luminance) / 8.0
                - neighbor_mean * neighbor_mean, 0.0,
            )
            sigma = osh.sqrt(osh.maximum(
                neighbor_variance, neighbor_mean * neighbor_mean * 0.0005
            ))
            current_deviation = osh.absolute(
                reconstructLuminance(hdr) - neighbor_mean
            ) / osh.maximum(sigma, 0.0005)
            history_deviation = osh.absolute(
                old_luminance - neighbor_mean
            ) / osh.maximum(sigma, 0.0005)
            threshold = 6.0 if surface_class > 0.5 else 4.0
            current_outlier = reconstructSmoothstep(
                threshold, threshold * 2.0, current_deviation
            )
            history_compatible = 1.0 - reconstructSmoothstep(
                threshold * 0.75, threshold * 1.5, history_deviation
            )
            protected_weight = osh.minimum(0.98, history_confidence)
            weight = osh.mix(
                weight, osh.maximum(weight, protected_weight),
                current_outlier * history_compatible
                * push.outlier_confidence_strength,
            )
        hdr = osh.mix(hdr, old_hdr, weight)
    history_color.store(
        output_pixel, osh.vec4(osh.maximum(hdr, osh.vec3(0.0)), 1.0)
    )
    mapped = acesApproximation(
        osh.maximum(hdr, osh.vec3(0.0)) * push.exposure
    )
    encoded = osh.vec4(linearToSrgb(mapped), 1.0)
    effect_slot = reconstructEffectSlot(source_pixel)
    output_uv = (osh.vec2(output_pixel) + 0.5) / osh.vec2(output_size)
    if push.kind0 != osh.u32(0):
        for effect_index in range(4):
            effect = reconstructEffect(effect_index)
            if effect.kind == osh.u32(4) and effect_slot != osh.u32(effect_index + 1):
                encoded.rgb = ordinarylight_isolation(
                    encoded.rgb, effect.strength
                )
            if effect.kind == osh.u32(5) or effect.kind == osh.u32(6):
                thickness = osh.vec2(effect.radius) / osh.vec2(output_size)
                valid_rect = (
                    effect.rect.z > effect.rect.x
                    and effect.rect.w > effect.rect.y
                )
                inside = (
                    valid_rect
                    and not osh.any_value(output_uv < effect.rect.xy)
                    and not osh.any_value(output_uv > effect.rect.zw)
                )
                border = inside and (
                    output_uv.x <= effect.rect.x + thickness.x
                    or output_uv.x >= effect.rect.z - thickness.x
                    or output_uv.y <= effect.rect.y + thickness.y
                    or output_uv.y >= effect.rect.w - thickness.y
                )
                if effect.kind == osh.u32(6) and inside:
                    encoded.rgb = ordinarylight_tint(
                        encoded.rgb, effect.color.rgb, effect.strength
                    )
                if border:
                    encoded.rgb = effect.color.rgb
    if effect_slot > osh.u32(0) and effect_slot <= osh.u32(4):
        effect_index = osh.i32(effect_slot - osh.u32(1))
        effect = reconstructEffect(effect_index)
        if effect.kind == osh.u32(2):
            encoded.rgb = ordinarylight_tint(
                encoded.rgb, effect.color.rgb, effect.strength
            )
        else:
            if effect.kind == osh.u32(3):
                encoded.rgb = ordinarylight_emissive(
                    encoded.rgb, effect.color.rgb, effect.strength
                )
            else:
                if effect.kind == osh.u32(1):
                    radius = osh.i32(effect.radius)
                    boundary = (
                        reconstructEffectSlot(source_pixel + osh.ivec2(-radius, 0)) != effect_slot
                        or reconstructEffectSlot(source_pixel + osh.ivec2(radius, 0)) != effect_slot
                        or reconstructEffectSlot(source_pixel + osh.ivec2(0, -radius)) != effect_slot
                        or reconstructEffectSlot(source_pixel + osh.ivec2(0, radius)) != effect_slot
                        or reconstructEffectSlot(source_pixel + osh.ivec2(-radius, -radius)) != effect_slot
                        or reconstructEffectSlot(source_pixel + osh.ivec2(radius, -radius)) != effect_slot
                        or reconstructEffectSlot(source_pixel + osh.ivec2(-radius, radius)) != effect_slot
                        or reconstructEffectSlot(source_pixel + osh.ivec2(radius, radius)) != effect_slot
                    )
                    encoded.rgb = ordinarylight_outline(
                        encoded.rgb, effect.color.rgb, boundary
                    )
    output_images[output_index].store(output_pixel, encoded)


@osh.compute(workgroup_size=(8, 8, 1))
def wavefront_reconstruct(
    hdr_image: osh.storage_image("rgba16f", access="read", binding=0),
    position_image: osh.storage_image("r32f", access="read", binding=1),
    normal_image: osh.storage_image("r32ui", access="read", binding=2),
    previous_color: osh.storage_image(
        "r11f_g11f_b10f", access="read", binding=3
    ),
    previous_position: osh.storage_image("r32f", access="read", binding=4),
    previous_normal: osh.storage_image("r32ui", access="read", binding=5),
    history_color: osh.storage_image(
        "r11f_g11f_b10f", access="write", binding=6
    ),
    output_images: osh.storage_image_array(
        "rgba8", 8, access="write", binding=7
    ),
    previous_camera: osh.storage_record(
        CandidateCamera, access="read", binding=8
    ),
    current_camera: osh.storage_record(
        CandidateCamera, access="read", binding=9
    ),
    material_image: osh.storage_image("r32ui", access="read", binding=10),
    push: osh.push_constants(ReconstructConstants),
):
    reconstructMain()


@osh.compute(workgroup_size=(8, 8, 1))
def wavefront_reconstruct_bgra(
    hdr_image: osh.storage_image("rgba16f", access="read", binding=0),
    position_image: osh.storage_image("r32f", access="read", binding=1),
    normal_image: osh.storage_image("r32ui", access="read", binding=2),
    previous_color: osh.storage_image(
        "r11f_g11f_b10f", access="read", binding=3
    ),
    previous_position: osh.storage_image("r32f", access="read", binding=4),
    previous_normal: osh.storage_image("r32ui", access="read", binding=5),
    history_color: osh.storage_image(
        "r11f_g11f_b10f", access="write", binding=6
    ),
    output_images: osh.storage_image_array(
        "unformatted", 8, access="write", binding=7
    ),
    previous_camera: osh.storage_record(
        CandidateCamera, access="read", binding=8
    ),
    current_camera: osh.storage_record(
        CandidateCamera, access="read", binding=9
    ),
    material_image: osh.storage_image("r32ui", access="read", binding=10),
    push: osh.push_constants(ReconstructConstants),
):
    reconstructMain()


@osh.compute(workgroup_size=(256, 1, 1))
def wavefront_indirect_clear(
    indirect_reservoirs: osh.storage_buffer(osh.u32, binding=0),
    clear_constants: osh.push_constants(IndirectClearConstants),
):
    reservoir_index = osh.global_invocation_id.x
    if reservoir_index >= clear_constants.reservoir_count:
        return
    base = reservoir_index * osh.u32(6)
    for word in range(6):
        indirect_reservoirs[base + osh.u32(word)] = osh.u32(0)


@osh.compute(workgroup_size=(1, 1, 1))
def wavefront_prepare_indirect(
    work_queue: osh.storage_record(WorkQueue, access="read", binding=0),
    dispatch_arguments: osh.storage_record(
        DispatchArguments, access="write", binding=1
    ),
):
    active_count = osh.minimum(work_queue.count, work_queue.capacity)
    dispatch_arguments.group_count_x = (
        active_count + osh.u32(63)
    ) / osh.u32(64)
    dispatch_arguments.group_count_y = osh.u32(1)
    dispatch_arguments.group_count_z = osh.u32(1)


@osh.compute(workgroup_size=(64, 1, 1))
def wavefront_resolve(
    paths: osh.storage_buffer(WavePathState, access="read", binding=0),
    pixels: osh.storage_buffer(ResolvedPixel, access="write", binding=1),
    push: osh.push_constants(ResolveConstants),
):
    path_index = osh.global_invocation_id.x
    if path_index >= push.path_count:
        return
    pixels[path_index].radiance = osh.vec4(paths[path_index].radiance.rgb, 1.0)
    pixels[path_index].metadata = osh.uvec4(
        paths[path_index].metadata.xy,
        osh.u32(paths[path_index].throughput.w),
        paths[path_index].metadata.w,
    )


@osh.compute(workgroup_size=(64, 1, 1))
def wavefront_path_to_hdr(
    paths: osh.storage_buffer(WavePathState, access="read", binding=0),
    hdr_image: osh.storage_image("rgba16f", binding=1),
    secondary_paths: osh.storage_buffer(
        SecondaryPathState, binding=2
    ),
    indirect_reservoir_words: osh.storage_buffer(osh.u32, binding=3),
    camera: osh.storage_record(PathToHdrCamera, access="read", binding=4),
    indirect_seed_words: osh.storage_buffer(osh.u32, binding=5),
    push: osh.push_constants(PathToHdrConstants),
):
    path_index = osh.global_invocation_id.x
    if path_index >= push.path_count:
        return
    pixel_index = paths[path_index].metadata.x
    pixel = osh.ivec2(
        osh.i32(pixel_index % push.image_width),
        osh.i32(pixel_index / push.image_width),
    )
    if pixel.y >= osh.i32(push.image_height):
        return
    contribution = (
        paths[path_index].radiance.rgb / osh.f32(push.sample_count)
    )
    accumulated = osh.vec3(0.0)
    if push.sample_index != osh.u32(0):
        accumulated = hdr_image.load(pixel).rgb
    hdr_image.store(pixel, osh.vec4(accumulated + contribution, 1.0))
    if (
        push.indirect_secondary_capture == osh.u32(0)
        or push.sample_index + osh.u32(1) != push.sample_count
    ):
        return
    secondary = secondary_paths[path_index]
    specular_probability = osh.clamp(
        secondary.primary_radiance.w, 0.0, 1.0
    )
    # Resolve from the complete multi-sample HDR value.  The secondary path
    # buffer is deliberately reset for every sample, so resolving sample zero
    # and then preparing later samples would replace valid signals with zeros.
    # Partitioning the accumulated radiance also guarantees that composing the
    # two denoised lobes starts energy-conserving before filtering.
    resolved_radiance = osh.maximum(
        accumulated + contribution, osh.vec3(0.0)
    )
    primary_radiance = osh.maximum(
        secondary.primary_radiance.rgb, osh.vec3(0.0)
    )
    indirect_contribution = osh.maximum(
        paths[path_index].radiance.rgb - primary_radiance,
        osh.vec3(0.0),
    )
    diffuse_radiance = resolved_radiance * (1.0 - specular_probability)
    specular_radiance = resolved_radiance * specular_probability
    hit_distance = 0.0
    if (
        secondary.primary_position.w > 0.5
        and secondary.position_valid.w > 0.5
    ):
        hit_distance = osh.length(
            secondary.position_valid.xyz - secondary.primary_position.xyz
        )
    secondary.diffuse_radiance_hit_distance = osh.vec4(
        diffuse_radiance, hit_distance
    )
    secondary.specular_radiance_hit_distance = osh.vec4(
        specular_radiance, hit_distance
    )
    secondary_paths[path_index] = secondary
    reservoir_pixel = osh.minimum(
        osh.uvec2(
            (osh.vec2(pixel) + 0.5)
            * osh.vec2(push.reservoir_width, push.reservoir_height)
            / osh.vec2(push.image_width, push.image_height)
        ),
        osh.uvec2(
            push.reservoir_width - osh.u32(1),
            push.reservoir_height - osh.u32(1),
        ),
    )
    representative = osh.ivec2(
        (osh.vec2(reservoir_pixel) + 0.5)
        * osh.vec2(push.image_width, push.image_height)
        / osh.vec2(push.reservoir_width, push.reservoir_height)
    )
    if pixel.x != representative.x or pixel.y != representative.y:
        return
    reservoir_index = (
        reservoir_pixel.y * push.reservoir_width + reservoir_pixel.x
    )
    target = osh.dot(
        indirect_contribution, osh.vec3(0.2126, 0.7152, 0.0722)
    )
    indirect_seed_words[reservoir_index] = indirectPackRgb9e5(
        indirect_contribution
    )
    # Positive comparisons reject NaN and infinity without requiring a
    # backend-specific classification intrinsic.
    if (
        secondary.position_valid.w < 0.5
        or not (target > 0.0)
        or not (target <= 3.402823e38)
    ):
        storeIndirectLightReservoir(
            reservoir_index, emptyIndirectLightReservoir(), camera.origin.xyz
        )
        return
    reservoir = emptyIndirectLightReservoir()
    reservoir.selected.secondary_position = secondary.position_valid.xyz
    reservoir.selected.secondary_normal = osh.normalize(
        secondary.normal_pdf.xyz
    )
    reservoir.selected.proposal_pdf = 1.0
    reservoir.selected.target = target
    reservoir.selected.radiance = indirect_contribution
    reservoir.weight_sum = target
    reservoir.sample_count = osh.u32(1)
    reservoir.valid = True
    storeIndirectLightReservoir(
        reservoir_index, reservoir, camera.origin.xyz
    )


@osh.compute(workgroup_size=(8, 8, 1))
def wavefront_indirect_candidates(
    indirect_reservoir_words: osh.storage_buffer(osh.u32, binding=0),
    hdr_image: osh.storage_image("rgba16f", access="read", binding=1),
    position_image: osh.storage_image("r32f", access="read", binding=2),
    normal_image: osh.storage_image("r32ui", access="read", binding=3),
    camera: osh.storage_record(CandidateCamera, access="read", binding=4),
    previous_reservoir_words: osh.storage_buffer(
        osh.u32, access="read", binding=5
    ),
    previous_position: osh.storage_image("r32f", access="read", binding=6),
    previous_normal: osh.storage_image("r32ui", access="read", binding=7),
    previous_camera: osh.storage_record(
        CandidateCamera, access="read", binding=8
    ),
    material_image: osh.storage_image("r32ui", access="read", binding=9),
    previous_material: osh.storage_image(
        "r32ui", access="read", binding=10
    ),
    counters: osh.storage_buffer(osh.u32, binding=11),
    scene_tlas: osh.acceleration_structure(binding=12),
    push: osh.push_constants(CandidateConstants),
):
    reservoir_pixel = osh.uvec2(osh.global_invocation_id.xy)
    if (
        reservoir_pixel.x >= push.reservoir_width
        or reservoir_pixel.y >= push.reservoir_height
    ):
        return
    reservoir_index = (
        reservoir_pixel.y * push.reservoir_width + reservoir_pixel.x
    )
    candidateCountEvent(osh.u32(0))
    source_coordinate = (
        (osh.vec2(reservoir_pixel) + 0.5)
        * osh.vec2(push.source_width, push.source_height)
        / osh.vec2(push.reservoir_width, push.reservoir_height)
    )
    source_pixel = osh.clamp(
        osh.ivec2(source_coordinate), osh.ivec2(0),
        osh.ivec2(push.source_width, push.source_height) - 1,
    )
    ray_distance = position_image.load(source_pixel).x
    reservoir = loadIndirectLightReservoir(
        reservoir_index, camera.origin.xyz
    )
    target = reservoir.selected.target
    if not (ray_distance > 0.0) or not (ray_distance <= 3.402823e38):
        candidateCountEvent(osh.u32(2))
        storeIndirectLightReservoir(
            reservoir_index, emptyIndirectLightReservoir(), camera.origin.xyz
        )
        return
    if (
        not reservoir.valid or not (target > 0.0)
        or not (target <= 3.402823e38)
    ):
        reservoir = emptyIndirectLightReservoir()
        target = 0.0
        candidateCountEvent(osh.u32(2))
    else:
        candidateCountEvent(osh.u32(1))
    current_position = candidatePrimaryWorldPosition(
        source_pixel, ray_distance
    )
    current_normal = candidateDecodePrimaryNormal(
        normal_image.load(source_pixel).x
    )
    history_was_clamped = False
    if push.history_valid != osh.u32(0):
        candidateCountEvent(osh.u32(3))
        material_signature = material_image.load(source_pixel).x
        projection = CandidateProjection(
            False, osh.u32(0), osh.u32(7)
        )
        if material_signature != osh.u32(0xFFFFFFFF):
            projection = candidateReprojectPrevious(
                current_position, current_normal, material_signature
            )
        if projection.valid:
            old = candidateLoadPreviousReservoir(
                projection.reservoir_index, previous_camera.origin.xyz
            )
            if (
                old.valid and not candidateSecondaryVisible(
                    current_position, current_normal,
                    old.selected.secondary_position,
                )
            ):
                old.valid = False
            old_target = osh.dot(
                old.selected.radiance, osh.vec3(0.2126, 0.7152, 0.0722)
            )
            merged = candidateMergePrevious(
                reservoir, old, old_target, reservoir_pixel, osh.u32(0),
                history_was_clamped,
            )
            reservoir = merged.reservoir
            history_was_clamped = merged.history_was_clamped
            if old.valid and old.selected.target > 0.0:
                candidateCountEvent(osh.u32(4))
                reservoir.debug_flags = (
                    reservoir.debug_flags | (osh.u32(1) << osh.u32(8))
                )
            else:
                candidateCountEvent(osh.u32(8))
                reservoir.debug_flags = (
                    reservoir.debug_flags
                    | candidateRejectionDebugFlag(osh.u32(8))
                )
            if push.spatial_enabled != osh.u32(0):
                previous_center = osh.uvec2(
                    projection.reservoir_index % push.reservoir_width,
                    projection.reservoir_index / push.reservoir_width,
                )
                sign_x = 1
                sign_y = 1
                if (
                    (osh.u32(camera.origin.w + 0.5) + reservoir_pixel.y)
                    & osh.u32(1)
                ) == osh.u32(0):
                    sign_x = -1
                if (
                    (osh.u32(camera.origin.w + 0.5) + reservoir_pixel.x)
                    & osh.u32(1)
                ) == osh.u32(0):
                    sign_y = -1
                for neighbor in range(4):
                    offset = osh.ivec2(0)
                    if neighbor == 0:
                        offset = osh.ivec2(sign_x * 2, 0)
                    else:
                        if neighbor == 1:
                            offset = osh.ivec2(0, sign_y * 2)
                        else:
                            if neighbor == 2:
                                offset = osh.ivec2(-sign_x * 4, 0)
                            else:
                                offset = osh.ivec2(0, -sign_y * 4)
                    coordinate = osh.ivec2(previous_center) + offset
                    if (
                        osh.any_value(coordinate < osh.ivec2(0))
                        or osh.any_value(coordinate >= osh.ivec2(
                            push.reservoir_width, push.reservoir_height
                        ))
                    ):
                        continue
                    candidateCountEvent(osh.u32(9))
                    spatial_pixel = osh.uvec2(coordinate)
                    compatibility = candidateSpatialCompatibility(
                        spatial_pixel, current_position, current_normal,
                        material_signature,
                    )
                    if not compatibility.valid:
                        candidateCountEvent(compatibility.rejection)
                        reservoir.debug_flags = (
                            reservoir.debug_flags
                            | candidateRejectionDebugFlag(
                                compatibility.rejection
                            )
                        )
                        continue
                    spatial_index = (
                        spatial_pixel.y * push.reservoir_width
                        + spatial_pixel.x
                    )
                    spatial = candidateLoadPreviousReservoir(
                        spatial_index, previous_camera.origin.xyz
                    )
                    if (
                        spatial.valid and not candidateSecondaryVisible(
                            current_position, current_normal,
                            spatial.selected.secondary_position,
                        )
                    ):
                        spatial.valid = False
                    spatial_target = osh.dot(
                        spatial.selected.radiance,
                        osh.vec3(0.2126, 0.7152, 0.0722),
                    )
                    merged = candidateMergePrevious(
                        reservoir, spatial, spatial_target, reservoir_pixel,
                        osh.u32(neighbor + 1), history_was_clamped,
                    )
                    reservoir = merged.reservoir
                    history_was_clamped = merged.history_was_clamped
                    if spatial.valid and spatial.selected.target > 0.0:
                        candidateCountEvent(osh.u32(10))
                        reservoir.debug_flags = (
                            reservoir.debug_flags
                            | (osh.u32(1) << osh.u32(9))
                        )
                    else:
                        candidateCountEvent(osh.u32(14))
                        reservoir.debug_flags = (
                            reservoir.debug_flags
                            | candidateRejectionDebugFlag(osh.u32(14))
                        )
        else:
            candidateCountEvent(projection.rejection)
            reservoir.debug_flags = (
                reservoir.debug_flags
                | candidateRejectionDebugFlag(projection.rejection)
            )
    if candidateInstrumentedPixel():
        osh.atomic_add(counters[osh.u32(15)], reservoir.sample_count)
        if history_was_clamped:
            osh.atomic_add(counters[osh.u32(16)], osh.u32(1))
        if reservoir.weight_sum >= 65504.0:
            osh.atomic_add(counters[osh.u32(17)], osh.u32(1))
    storeIndirectLightReservoir(
        reservoir_index, reservoir, camera.origin.xyz
    )


@osh.compute(workgroup_size=(8, 8, 1))
def wavefront_indirect_debug(
    indirect_reservoir_words: osh.storage_buffer(
        osh.u32, access="read", binding=0
    ),
    debug_image: osh.storage_image("rgba16f", binding=1),
    indirect_seed_words: osh.storage_buffer(
        osh.u32, access="read", binding=2
    ),
    material_image: osh.storage_image("r32ui", access="read", binding=3),
    push: osh.push_constants(DebugConstants),
):
    output_pixel = osh.uvec2(osh.global_invocation_id.xy)
    if (
        output_pixel.x >= push.output_width
        or output_pixel.y >= push.output_height
    ):
        return
    reservoir_pixel = osh.minimum(
        osh.uvec2(
            (osh.vec2(output_pixel) + 0.5)
            * osh.vec2(push.reservoir_width, push.reservoir_height)
            / osh.vec2(push.output_width, push.output_height)
        ),
        osh.uvec2(
            push.reservoir_width - osh.u32(1),
            push.reservoir_height - osh.u32(1),
        ),
    )
    reservoir_index = (
        reservoir_pixel.y * push.reservoir_width + reservoir_pixel.x
    )
    reservoir = loadIndirectLightReservoir(
        reservoir_index, osh.vec3(0.0)
    )
    color = osh.vec3(0.04)
    if push.mode == osh.u32(5):
        output_material = material_image.load(osh.ivec2(output_pixel)).x
        current = debug_image.load(osh.ivec2(output_pixel)).rgb
        reservoir_coordinate = (
            (osh.vec2(output_pixel) + 0.5)
            * osh.vec2(push.reservoir_width, push.reservoir_height)
            / osh.vec2(push.output_width, push.output_height) - 0.5
        )
        floored_coordinate = osh.floor(reservoir_coordinate)
        base = osh.ivec2(floored_coordinate)
        fraction = reservoir_coordinate - floored_coordinate
        filtered_correction = osh.vec3(0.0)
        for y in range(2):
            for x in range(2):
                result = indirectCorrection(
                    base + osh.ivec2(x, y), output_material
                )
                if not result.valid:
                    continue
                x_weight = fraction.x if x == 1 else 1.0 - fraction.x
                y_weight = fraction.y if y == 1 else 1.0 - fraction.y
                filter_weight = x_weight * y_weight
                filtered_correction = (
                    filtered_correction
                    + result.correction * filter_weight * result.confidence
                )
        color = osh.maximum(
            current + filtered_correction
            * osh.clamp(push.apply_strength, 0.0, 1.0),
            osh.vec3(0.0),
        )
        debug_image.store(osh.ivec2(output_pixel), osh.vec4(color, 1.0))
        return
    if push.mode == osh.u32(1) and reservoir.valid:
        color = reservoir.selected.radiance
    else:
        if push.mode == osh.u32(2) and reservoir.valid:
            color = osh.vec3(
                osh.f32(reservoir.sample_count)
                / osh.f32(osh.maximum(push.history_limit, osh.u32(1)))
            )
        else:
            if push.mode == osh.u32(3):
                color = (
                    osh.vec3(0.0, 1.0, 0.0)
                    if reservoir.valid else indirectInvalidColor(output_pixel)
                )
            else:
                if push.mode == osh.u32(4):
                    color = (
                        osh.maximum(
                            indirectAcceptanceColor(reservoir.debug_flags),
                            osh.vec3(0.04),
                        )
                        if reservoir.valid
                        else indirectInvalidColor(output_pixel)
                    )
    debug_image.store(osh.ivec2(output_pixel), osh.vec4(color, 1.0))


@osh.compute(workgroup_size=(64, 1, 1))
def wavefront_tone_map(
    pixels: osh.storage_buffer(ResolvedPixel, access="read", binding=0),
    rgba8: osh.storage_buffer(osh.u32, access="write", binding=1),
    push: osh.push_constants(ToneMapConstants),
):
    path_index = osh.global_invocation_id.x
    if path_index >= push.path_count:
        return
    mapped = acesApproximation(
        osh.maximum(pixels[path_index].radiance.rgb, osh.vec3(0.0))
        * push.exposure
    )
    rgba8[path_index] = osh.pack_unorm4x8(
        osh.vec4(linearToSrgb(mapped), 1.0)
    )


@osh.compute(workgroup_size=(64, 1, 1))
def wavefront_tone_map_image(
    paths: osh.storage_buffer(WavePathState, access="read", binding=0),
    output_image: osh.storage_image("rgba8", access="write", binding=1),
    push: osh.push_constants(ToneMapImageConstants),
):
    path_index = osh.global_invocation_id.x
    if path_index >= push.path_count:
        return
    pixel_index = paths[path_index].metadata.x
    source_pixel = osh.ivec2(
        osh.i32(pixel_index % push.image_width),
        osh.i32(pixel_index / push.image_width),
    )
    if source_pixel.y >= osh.i32(push.image_height):
        return
    mapped = acesApproximation(
        osh.maximum(paths[path_index].radiance.rgb, osh.vec3(0.0))
        * push.exposure
    )
    output_color = osh.vec4(linearToSrgb(mapped), 1.0)
    output_size = output_image.size()
    first = osh.ivec2(
        source_pixel.x * output_size.x / osh.i32(push.image_width),
        source_pixel.y * output_size.y / osh.i32(push.image_height),
    )
    last = osh.ivec2(
        (source_pixel.x + 1) * output_size.x / osh.i32(push.image_width),
        (source_pixel.y + 1) * output_size.y / osh.i32(push.image_height),
    )
    last = osh.maximum(last, first + osh.ivec2(1))
    for y in range(first.y, osh.minimum(last.y, output_size.y)):
        for x in range(first.x, osh.minimum(last.x, output_size.x)):
            output_image.store(osh.ivec2(x, y), output_color)


@osh.compute(workgroup_size=(8, 8, 1))
def rgba_to_nv12(
    rgba_input: osh.storage_image("rgba8", access="read", binding=0),
    words: osh.storage_buffer(osh.u32, access="write", binding=1),
    push: osh.push_constants(Nv12Constants),
):
    x = osh.global_invocation_id.x * osh.u32(4)
    y = osh.global_invocation_id.y * osh.u32(2)
    if x >= push.width:
        return
    if y >= push.height:
        return
    top0 = rgba_input.load(nv12Pixel(x, y, push.width, push.height)).rgb
    top1 = rgba_input.load(
        nv12Pixel(x + osh.u32(1), y, push.width, push.height)
    ).rgb
    top2 = rgba_input.load(
        nv12Pixel(x + osh.u32(2), y, push.width, push.height)
    ).rgb
    top3 = rgba_input.load(
        nv12Pixel(x + osh.u32(3), y, push.width, push.height)
    ).rgb
    bottom0 = rgba_input.load(
        nv12Pixel(x, y + osh.u32(1), push.width, push.height)
    ).rgb
    bottom1 = rgba_input.load(
        nv12Pixel(x + osh.u32(1), y + osh.u32(1), push.width, push.height)
    ).rgb
    bottom2 = rgba_input.load(
        nv12Pixel(x + osh.u32(2), y + osh.u32(1), push.width, push.height)
    ).rgb
    bottom3 = rgba_input.load(
        nv12Pixel(x + osh.u32(3), y + osh.u32(1), push.width, push.height)
    ).rgb
    words_per_row = push.pitch / osh.u32(4)
    words[y * words_per_row + x / osh.u32(4)] = pack4Bytes(
        nv12Luma(top0), nv12Luma(top1), nv12Luma(top2), nv12Luma(top3)
    )
    if y + osh.u32(1) < push.height:
        words[(y + osh.u32(1)) * words_per_row + x / osh.u32(4)] = pack4Bytes(
            nv12Luma(bottom0), nv12Luma(bottom1),
            nv12Luma(bottom2), nv12Luma(bottom3),
        )
    first = 0.25 * (top0 + top1 + bottom0 + bottom1)
    second = 0.25 * (top2 + top3 + bottom2 + bottom3)
    uv0 = nv12Chroma(first)
    uv1 = nv12Chroma(second)
    chroma_word = (
        push.pitch * push.height / osh.u32(4)
        + (y / osh.u32(2)) * words_per_row
        + x / osh.u32(4)
    )
    words[chroma_word] = pack4Bytes(uv0.x, uv0.y, uv1.x, uv1.y)


@osh.compute(workgroup_size=(8, 8, 1))
def hdr_to_p010(
    hdr_input: osh.storage_image("rgba16f", access="read", binding=0),
    words: osh.storage_buffer(osh.u32, access="write", binding=1),
    push: osh.push_constants(P010Constants),
):
    x = osh.global_invocation_id.x * osh.u32(4)
    y = osh.global_invocation_id.y * osh.u32(2)
    if x >= push.width:
        return
    if y >= push.height:
        return
    source_size = hdr_input.size()
    top0 = p010Color(hdr_input.load(
        p010SourcePixel(x, y, source_size, push.width, push.height)
    ).rgb, push.exposure)
    top1 = p010Color(hdr_input.load(p010SourcePixel(
        x + osh.u32(1), y, source_size, push.width, push.height
    )).rgb, push.exposure)
    top2 = p010Color(hdr_input.load(p010SourcePixel(
        x + osh.u32(2), y, source_size, push.width, push.height
    )).rgb, push.exposure)
    top3 = p010Color(hdr_input.load(p010SourcePixel(
        x + osh.u32(3), y, source_size, push.width, push.height
    )).rgb, push.exposure)
    bottom0 = p010Color(hdr_input.load(p010SourcePixel(
        x, y + osh.u32(1), source_size, push.width, push.height
    )).rgb, push.exposure)
    bottom1 = p010Color(hdr_input.load(p010SourcePixel(
        x + osh.u32(1), y + osh.u32(1), source_size,
        push.width, push.height
    )).rgb, push.exposure)
    bottom2 = p010Color(hdr_input.load(p010SourcePixel(
        x + osh.u32(2), y + osh.u32(1), source_size,
        push.width, push.height
    )).rgb, push.exposure)
    bottom3 = p010Color(hdr_input.load(p010SourcePixel(
        x + osh.u32(3), y + osh.u32(1), source_size,
        push.width, push.height
    )).rgb, push.exposure)
    words_per_row = push.pitch / osh.u32(4)
    word = y * words_per_row + x / osh.u32(2)
    words[word] = pack2x16(p010Luma(top0), p010Luma(top1))
    words[word + osh.u32(1)] = pack2x16(p010Luma(top2), p010Luma(top3))
    if y + osh.u32(1) < push.height:
        word = (y + osh.u32(1)) * words_per_row + x / osh.u32(2)
        words[word] = pack2x16(p010Luma(bottom0), p010Luma(bottom1))
        words[word + osh.u32(1)] = pack2x16(
            p010Luma(bottom2), p010Luma(bottom3)
        )
    uv0 = p010Chroma(0.25 * (top0 + top1 + bottom0 + bottom1))
    uv1 = p010Chroma(0.25 * (top2 + top3 + bottom2 + bottom3))
    chroma_word = (
        push.pitch * push.height / osh.u32(4)
        + (y / osh.u32(2)) * words_per_row
        + x / osh.u32(2)
    )
    words[chroma_word] = pack2x16(uv0.x, uv0.y)
    words[chroma_word + osh.u32(1)] = pack2x16(uv1.x, uv1.y)


@osh.compute(workgroup_size=(8, 8, 1))
def denoise_atrous(
    accumulation_a: osh.storage_image("rgba32f", access="read", binding=4),
    gbuffer_a: osh.storage_image("rgba32f", access="read", binding=5),
    accumulation_b: osh.storage_image("rgba32f", access="read", binding=6),
    gbuffer_b: osh.storage_image("rgba32f", access="read", binding=7),
    moment_a: osh.storage_image("r32f", access="read", binding=8),
    moment_b: osh.storage_image("r32f", access="read", binding=9),
    filtered_a: osh.storage_image("rgba32f", access="read_write", binding=12),
    filtered_b: osh.storage_image("rgba32f", access="read_write", binding=13),
    camera: osh.push_constants(DenoiseCamera),
):
    pixel = osh.ivec2(osh.global_invocation_id.xy)
    size = osh.ivec2(osh.i32(camera.origin.w), osh.i32(camera.forward.w))
    if pixel.x >= size.x:
        return
    if pixel.y >= size.y:
        return
    iteration = osh.maximum(osh.i32(camera.lighting.w) - 1, 0)
    step_width = 1 << iteration
    current_a = (osh.i32(camera.accumulation.x) & 1) == 0

    center = filtered_a.load(pixel)
    if iteration == 0:
        center = accumulation_a.load(pixel)
        if not current_a:
            center = accumulation_b.load(pixel)
    else:
        if ((iteration - 1) & 1) != 0:
            center = filtered_b.load(pixel)

    raw_history = accumulation_a.load(pixel)
    center_gbuffer = gbuffer_a.load(pixel)
    if not current_a:
        raw_history = accumulation_b.load(pixel)
        center_gbuffer = gbuffer_b.load(pixel)
    center_normal = decodeAtrousNormal(center_gbuffer.xy)
    center_depth = center_gbuffer.z
    center_background = center_depth < 0.0
    if center_gbuffer.w > 0.5:
        if (iteration & 1) == 0:
            filtered_a.store(pixel, center)
        else:
            filtered_b.store(pixel, center)
        return

    center_luminance = osh.dot(
        raw_history.rgb, osh.vec3(0.2126, 0.7152, 0.0722)
    )
    center_moment = moment_a.load(pixel).r
    if not current_a:
        center_moment = moment_b.load(pixel).r
    variance = osh.maximum(
        center_moment - center_luminance * center_luminance, 0.0
    )
    relative_variance = variance / osh.maximum(
        center_luminance * center_luminance, 0.01
    )
    variance_threshold = osh.maximum(camera.right.w, 0.000001)
    if relative_variance <= variance_threshold:
        if (iteration & 1) == 0:
            filtered_a.store(pixel, center)
        else:
            filtered_b.store(pixel, center)
        return

    filter_blend = osh.clamp(
        (relative_variance - variance_threshold) / (variance_threshold * 4.0),
        0.0, 1.0,
    )
    color_scale = osh.maximum(2.0 * osh.sqrt(variance), 0.015)
    total = osh.vec3(0.0)
    weight_sum = 0.0
    for y in range(-2, 3):
        for x in range(-2, 3):
            sample_pixel = pixel + osh.ivec2(x, y) * step_width
            if sample_pixel.x < 0:
                continue
            if sample_pixel.y < 0:
                continue
            if sample_pixel.x >= size.x:
                continue
            if sample_pixel.y >= size.y:
                continue
            sample_gbuffer = gbuffer_a.load(sample_pixel)
            if not current_a:
                sample_gbuffer = gbuffer_b.load(sample_pixel)
            sample_background = sample_gbuffer.z < 0.0
            if sample_background != center_background:
                continue
            sample_normal = decodeAtrousNormal(sample_gbuffer.xy)
            normal_weight = osh.power(
                osh.maximum(osh.dot(center_normal, sample_normal), 0.0), 64.0
            )
            depth_scale = osh.maximum(osh.absolute(center_depth) * 0.01, 0.005)
            depth_weight = osh.exp(
                -osh.absolute(sample_gbuffer.z - center_depth) / depth_scale
            )
            sample_color = filtered_a.load(sample_pixel)
            if iteration == 0:
                sample_color = accumulation_a.load(sample_pixel)
                if not current_a:
                    sample_color = accumulation_b.load(sample_pixel)
            else:
                if ((iteration - 1) & 1) != 0:
                    sample_color = filtered_b.load(sample_pixel)
            sample_luminance = osh.dot(
                sample_color.rgb, osh.vec3(0.2126, 0.7152, 0.0722)
            )
            color_weight = osh.exp(
                -osh.absolute(sample_luminance - center_luminance) / color_scale
            )
            weight = (
                atrousKernel(x) * atrousKernel(y) * normal_weight
                * depth_weight * color_weight
            )
            total = total + sample_color.rgb * weight
            weight_sum = weight_sum + weight
    filtered = total / osh.maximum(weight_sum, 0.000001)
    result = osh.vec4(osh.mix(center.rgb, filtered, filter_blend), center.a)
    if (iteration & 1) == 0:
        filtered_a.store(pixel, result)
    else:
        filtered_b.store(pixel, result)


@osh.compute(workgroup_size=(8, 8, 1))
def tone_map(
    output_image: osh.storage_image("rgba8", access="write", binding=1),
    accumulation_a: osh.storage_image("rgba32f", access="read", binding=4),
    accumulation_b: osh.storage_image("rgba32f", access="read", binding=6),
    filtered_a: osh.storage_image("rgba32f", access="read", binding=12),
    filtered_b: osh.storage_image("rgba32f", access="read", binding=13),
    camera: osh.push_constants(DenoiseCamera),
):
    pixel = osh.uvec2(osh.global_invocation_id.xy)
    size = osh.uvec2(osh.u32(camera.origin.w), osh.u32(camera.forward.w))
    if pixel.x >= size.x or pixel.y >= size.y:
        return
    read_a = (osh.i32(camera.accumulation.x) & 1) == 0
    denoise_iterations = osh.i32(camera.lighting.w)
    hdr = accumulation_a.load(osh.ivec2(pixel)).rgb
    if denoise_iterations > 0:
        if (denoise_iterations & 1) == 1:
            hdr = filtered_a.load(osh.ivec2(pixel)).rgb
        else:
            hdr = filtered_b.load(osh.ivec2(pixel)).rgb
    else:
        if not read_a:
            hdr = accumulation_b.load(osh.ivec2(pixel)).rgb
    color = osh.power(hdr / (osh.vec3(1.0) + hdr), osh.vec3(1.0 / 2.2))
    color = fpsOverlay(pixel, size, color, camera.overlay)
    output_image.store(osh.ivec2(pixel), osh.vec4(color, 1.0))


@osh.function
def waveFresnelSchlick(
    cosine: osh.f32, ior_from: osh.f32, ior_to: osh.f32,
) -> osh.f32:
    r0 = (ior_from - ior_to) / osh.maximum(
        ior_from + ior_to, 0.000001
    )
    r0 = r0 * r0
    return r0 + (1.0 - r0) * osh.power(
        1.0 - osh.clamp(cosine, 0.0, 1.0), 5.0
    )


@osh.function
def waveCosineHemisphere(
    normal: osh.vec3, random_u: osh.f32, random_v: osh.f32,
) -> osh.vec3:
    radius = osh.sqrt(random_u)
    phi = 6.28318530718 * random_v
    tangent = osh.cross(normal, osh.vec3(0.0, 1.0, 0.0))
    if osh.absolute(normal.z) < 0.999:
        tangent = osh.cross(normal, osh.vec3(0.0, 0.0, 1.0))
    tangent = osh.normalize(tangent)
    bitangent = osh.cross(normal, tangent)
    return osh.normalize(
        tangent * (radius * osh.cosine(phi))
        + bitangent * (radius * osh.sine(phi))
        + normal * osh.sqrt(osh.maximum(0.0, 1.0 - random_u))
    )


@osh.function
def evaluateMaterial(
    material: MaterialData, normal: osh.vec3, uv: osh.vec2,
    direction: osh.vec3, entering: osh.boolean, random_u: osh.f32,
    random_v: osh.f32, bounce_index: osh.f32, current_ior: osh.f32,
    exterior_ior: osh.f32,
) -> MaterialEvaluation:
    return MaterialEvaluation(
        material.base_roughness.rgb,
        material.emission_metallic.rgb,
        material.emission_metallic.a,
        material.base_roughness.a,
        material.attenuation_transmission.a,
        material.ior_distance.x,
        material.attenuation_transmission.rgb,
        material.ior_distance.y,
        0.0, osh.vec3(0.0), direction, 0.0, 1.0,
    )


@osh.function
def shadeApplyMaterialEvaluation(
    input_material: MaterialData, evaluated: MaterialEvaluation,
) -> MaterialData:
    material = input_material
    material.base_roughness = osh.vec4(
        evaluated.base_color, evaluated.roughness
    )
    material.emission_metallic = osh.vec4(
        evaluated.emission, evaluated.metallic
    )
    material.attenuation_transmission = osh.vec4(
        evaluated.attenuation_color, evaluated.transmission
    )
    material.ior_distance.xy = osh.vec2(
        evaluated.ior, evaluated.attenuation_distance
    )
    return material


@osh.function
def rayQueryEnvironment(direction: osh.vec3) -> osh.vec3:
    blend = osh.clamp(0.5 * (direction.y + 1.0), 0.0, 1.0)
    return osh.mix(osh.vec3(0.8), osh.vec3(0.2, 0.4, 0.9), blend)


@osh.function
def rayQueryPrimaryRay(screen: osh.vec2) -> PrimaryRayResult:
    projection = osh.i32(camera.up.w + 0.5)
    origin = camera.origin.xyz
    direction = camera.forward.xyz
    if projection == 1:
        origin = origin + screen.x * camera.right.xyz + screen.y * camera.up.xyz
        direction = osh.normalize(camera.forward.xyz)
    else:
        if projection == 2:
            yaw = screen.x * osh.length(camera.right.xyz)
            pitch = screen.y * osh.length(camera.up.xyz)
            direction = osh.normalize(
                osh.normalize(camera.forward.xyz)
                * osh.cosine(pitch) * osh.cosine(yaw)
                + osh.normalize(camera.right.xyz)
                * osh.cosine(pitch) * osh.sine(yaw)
                + osh.normalize(camera.up.xyz) * osh.sine(pitch)
            )
        else:
            direction = osh.normalize(
                camera.forward.xyz + screen.x * camera.right.xyz
                + screen.y * camera.up.xyz
            )
    return PrimaryRayResult(origin, direction)


@osh.function
def rayQueryRandomState(state: osh.u32) -> osh.u32:
    return state * osh.u32(747796405) + osh.u32(2891336453)


@osh.function
def rayQueryRandomValue(state: osh.u32) -> osh.f32:
    word = (
        ((state >> ((state >> osh.u32(28)) + osh.u32(4))) ^ state)
        * osh.u32(277803737)
    )
    word = (word >> osh.u32(22)) ^ word
    return osh.f32(word) * (1.0 / 4294967296.0)


@osh.function
def rayQueryReflect(direction: osh.vec3, normal: osh.vec3) -> osh.vec3:
    return direction - 2.0 * osh.dot(normal, direction) * normal


@osh.function
def rayQueryRefract(
    direction: osh.vec3, normal: osh.vec3, eta: osh.f32,
) -> osh.vec3:
    cosine = osh.dot(normal, direction)
    discriminant = 1.0 - eta * eta * (1.0 - cosine * cosine)
    if discriminant < 0.0:
        return osh.vec3(0.0)
    return eta * direction - (eta * cosine + osh.sqrt(discriminant)) * normal


@osh.function
def rayQueryEncodeNormal(normal_value: osh.vec3) -> osh.vec2:
    normal = normal_value / (
        osh.absolute(normal_value.x) + osh.absolute(normal_value.y)
        + osh.absolute(normal_value.z)
    )
    encoded = normal.xy
    if normal.z < 0.0:
        encoded = (1.0 - osh.absolute(encoded.yx)) * osh.sign(encoded.xy)
    return encoded


@osh.function
def rayQueryDecodeNormal(encoded: osh.vec2) -> osh.vec3:
    normal = osh.vec3(
        encoded, 1.0 - osh.absolute(encoded.x) - osh.absolute(encoded.y)
    )
    if normal.z < 0.0:
        normal.xy = (1.0 - osh.absolute(normal.yx)) * osh.sign(normal.xy)
    return osh.normalize(normal)


@osh.function
def rayQueryPowerHeuristic(
    first_pdf: osh.f32, second_pdf: osh.f32,
) -> osh.f32:
    first_squared = first_pdf * first_pdf
    second_squared = second_pdf * second_pdf
    return first_squared / osh.maximum(
        first_squared + second_squared, 0.000001
    )


@osh.function
def shadePbrFresnel(f0: osh.vec3, cosine: osh.f32) -> osh.vec3:
    return f0 + (osh.vec3(1.0) - f0) * osh.power(
        1.0 - osh.clamp(cosine, 0.0, 1.0), 5.0
    )


@osh.function
def shadeGgxDistribution(
    normal_half: osh.f32, roughness: osh.f32,
) -> osh.f32:
    alpha = osh.maximum(roughness * roughness, 0.0009)
    alpha_squared = alpha * alpha
    denominator = (
        normal_half * normal_half * (alpha_squared - 1.0) + 1.0
    )
    return alpha_squared / osh.maximum(
        3.14159265359 * denominator * denominator, 0.000001
    )


@osh.function
def shadeGgxSmithComponent(
    normal_direction: osh.f32, roughness: osh.f32,
) -> osh.f32:
    alpha = osh.maximum(roughness * roughness, 0.0009)
    alpha_squared = alpha * alpha
    return 2.0 * normal_direction / osh.maximum(
        normal_direction + osh.sqrt(
            alpha_squared + (1.0 - alpha_squared)
            * normal_direction * normal_direction
        ),
        0.000001,
    )


@osh.function
def shadePbrSpecularProbability(material: MaterialData) -> osh.f32:
    f0 = osh.mix(
        osh.vec3(0.04), material.base_roughness.rgb,
        material.emission_metallic.a,
    )
    return osh.clamp(
        osh.maximum(f0.r, osh.maximum(f0.g, f0.b))
        + material.advanced0.x * 0.15, 0.1, 0.95
    )


@osh.function
def shadeEvaluatePbrLobes(
    material: MaterialData, normal: osh.vec3, view: osh.vec3,
    outgoing: osh.vec3,
) -> PbrLobeResult:
    normal_view = osh.maximum(osh.dot(normal, view), 0.0)
    normal_light = osh.maximum(osh.dot(normal, outgoing), 0.0)
    if normal_view <= 0.0 or normal_light <= 0.0:
        return PbrLobeResult(osh.vec3(0.0), osh.vec3(0.0))
    half_vector = osh.normalize(view + outgoing)
    normal_half = osh.maximum(osh.dot(normal, half_vector), 0.0)
    view_half = osh.maximum(osh.dot(view, half_vector), 0.0)
    metallic = material.emission_metallic.a
    f0 = osh.mix(osh.vec3(0.04), material.base_roughness.rgb, metallic)
    fresnel = shadePbrFresnel(f0, view_half)
    effective_roughness = material.base_roughness.a * (
        1.0 - 0.25 * osh.absolute(osh.clamp(material.advanced0.w, -1.0, 1.0))
    )
    distribution = shadeGgxDistribution(normal_half, effective_roughness)
    geometry = (
        shadeGgxSmithComponent(normal_view, effective_roughness)
        * shadeGgxSmithComponent(normal_light, effective_roughness)
    )
    specular = fresnel * distribution * geometry / osh.maximum(
        4.0 * normal_view * normal_light, 0.000001
    )
    diffuse = (
        (osh.vec3(1.0) - fresnel) * (1.0 - metallic)
        * material.base_roughness.rgb / 3.14159265359
    )
    subsurface = osh.clamp(material.advanced1.x, 0.0, 1.0)
    diffuse = osh.mix(
        diffuse, diffuse * material.subsurface_color.rgb, subsurface
    )
    sheen = material.sheen_color.rgb * (
        (1.0 - osh.clamp(material.advanced0.z, 0.0, 1.0))
        * osh.power(1.0 - view_half, 5.0)
    )
    clearcoat = osh.clamp(material.advanced0.x, 0.0, 1.0)
    coat_roughness = osh.clamp(material.advanced0.y, 0.02, 1.0)
    coat_distribution = shadeGgxDistribution(normal_half, coat_roughness)
    coat_geometry = (
        shadeGgxSmithComponent(normal_view, coat_roughness)
        * shadeGgxSmithComponent(normal_light, coat_roughness)
    )
    coat_fresnel = 0.04 + 0.96 * osh.power(1.0 - view_half, 5.0)
    coat = osh.vec3(
        clearcoat * coat_distribution * coat_geometry * coat_fresnel
        / osh.maximum(4.0 * normal_view * normal_light, 0.000001)
    )
    layer_scale = 1.0 - clearcoat * coat_fresnel
    return PbrLobeResult(
        diffuse * layer_scale,
        (specular + sheen) * layer_scale + coat,
    )


@osh.function
def shadeEvaluatePbr(
    material: MaterialData, normal: osh.vec3, view: osh.vec3,
    outgoing: osh.vec3,
) -> osh.vec3:
    lobes = shadeEvaluatePbrLobes(material, normal, view, outgoing)
    return lobes.diffuse + lobes.specular


@osh.function
def shadePbrPdf(
    material: MaterialData, normal: osh.vec3, view: osh.vec3,
    outgoing: osh.vec3,
) -> osh.f32:
    normal_light = osh.maximum(osh.dot(normal, outgoing), 0.0)
    if normal_light <= 0.0:
        return 0.0
    half_vector = osh.normalize(view + outgoing)
    normal_half = osh.maximum(osh.dot(normal, half_vector), 0.0)
    view_half = osh.maximum(osh.dot(view, half_vector), 0.000001)
    specular_pdf = (
        shadeGgxDistribution(normal_half, material.base_roughness.a)
        * normal_half / (4.0 * view_half)
    )
    diffuse_pdf = normal_light / 3.14159265359
    return osh.mix(
        diffuse_pdf, specular_pdf, shadePbrSpecularProbability(material)
    )


@osh.function
def shadeSampleGgxHalfVector(
    normal: osh.vec3, roughness: osh.f32, random_u: osh.f32,
    random_v: osh.f32,
) -> osh.vec3:
    alpha = osh.maximum(roughness * roughness, 0.0009)
    alpha_squared = alpha * alpha
    phi = 6.28318530718 * random_v
    cosine = osh.sqrt(
        (1.0 - random_u) / osh.maximum(
            1.0 + (alpha_squared - 1.0) * random_u, 0.000001
        )
    )
    sine = osh.sqrt(osh.maximum(0.0, 1.0 - cosine * cosine))
    tangent = osh.cross(normal, osh.vec3(0.0, 1.0, 0.0))
    if osh.absolute(normal.z) < 0.999:
        tangent = osh.cross(normal, osh.vec3(0.0, 0.0, 1.0))
    tangent = osh.normalize(tangent)
    bitangent = osh.cross(normal, tangent)
    return osh.normalize(
        tangent * sine * osh.cosine(phi)
        + bitangent * sine * osh.sine(phi) + normal * cosine
    )


@osh.function
def shadeSamplePbr(
    material: MaterialData, normal: osh.vec3, incoming: osh.vec3,
    initial_random_state: osh.u32,
) -> PbrSampleResult:
    random_state = initial_random_state
    view = -incoming
    probability = shadePbrSpecularProbability(material)
    random_state = rayQueryRandomState(random_state)
    specular = rayQueryRandomValue(random_state) < probability
    random_state = rayQueryRandomState(random_state)
    first_random = rayQueryRandomValue(random_state)
    random_state = rayQueryRandomState(random_state)
    second_random = rayQueryRandomValue(random_state)
    outgoing = waveCosineHemisphere(normal, first_random, second_random)
    if specular:
        half_vector = shadeSampleGgxHalfVector(
            normal, material.base_roughness.a, first_random, second_random
        )
        outgoing = rayQueryReflect(incoming, half_vector)
    if osh.dot(normal, outgoing) <= 0.0:
        random_state = rayQueryRandomState(random_state)
        first_random = rayQueryRandomValue(random_state)
        random_state = rayQueryRandomState(random_state)
        second_random = rayQueryRandomValue(random_state)
        outgoing = waveCosineHemisphere(normal, first_random, second_random)
    outgoing = osh.normalize(outgoing)
    pdf = osh.maximum(
        shadePbrPdf(material, normal, view, outgoing), 0.000001
    )
    weight = (
        shadeEvaluatePbr(material, normal, view, outgoing)
        * osh.maximum(osh.dot(normal, outgoing), 0.0) / pdf
    )
    weight = weight * osh.mix(
        material.texture_parameters.w, 1.0,
        material.emission_metallic.a,
    )
    return PbrSampleResult(outgoing, weight, pdf, random_state, specular)


@osh.function
def shadeSrgbChannel(value: osh.f32) -> osh.f32:
    if value <= 0.04045:
        return value / 12.92
    return osh.power((value + 0.055) / 1.055, 2.4)


@osh.function
def shadeWrapTextureCoordinate(value: osh.f32, mode: osh.u32) -> osh.f32:
    if mode == osh.u32(1):
        return osh.clamp(value, 0.0, 0.99999994)
    if mode == osh.u32(2):
        period = value - osh.floor(value / 2.0) * 2.0
        if period < 0.0:
            period = period + 2.0
        if period <= 1.0:
            return period
        return 2.0 - period
    return value - osh.floor(value)


@osh.function
def shadeWrapTextureIndex(
    value: osh.i32, size: osh.i32, mode: osh.u32,
) -> osh.i32:
    if mode == osh.u32(1):
        return osh.clamp(value, 0, size - 1)
    period = size
    if mode == osh.u32(2):
        period = size * 2
    wrapped = value % period
    if wrapped < 0:
        wrapped = wrapped + period
    if mode == osh.u32(2) and wrapped >= size:
        return period - 1 - wrapped
    return wrapped


@osh.function
def shadeDecodeTextureTexel(packed: osh.u32, srgb: osh.boolean) -> osh.vec4:
    value = osh.vec4(
        osh.f32(packed & osh.u32(255)),
        osh.f32((packed >> osh.u32(8)) & osh.u32(255)),
        osh.f32((packed >> osh.u32(16)) & osh.u32(255)),
        osh.f32((packed >> osh.u32(24)) & osh.u32(255)),
    ) / 255.0
    if srgb:
        value.rgb = osh.vec3(
            shadeSrgbChannel(value.r), shadeSrgbChannel(value.g),
            shadeSrgbChannel(value.b),
        )
    return value


@osh.function
def shadeFetchTextureTexel(
    offset: osh.u32, size: osh.ivec2, coordinate: osh.ivec2,
    wrap: osh.uvec2, srgb: osh.boolean,
) -> osh.vec4:
    x = shadeWrapTextureIndex(coordinate.x, size.x, wrap.x)
    y = shadeWrapTextureIndex(coordinate.y, size.y, wrap.y)
    return shadeDecodeTextureTexel(
        texture_words[offset + osh.u32(y * size.x + x)], srgb
    )


@osh.function
def shadeTextureMipOffset(
    base_offset: osh.u32, base_size: osh.ivec2, level: osh.u32,
) -> osh.u32:
    offset = base_offset
    size = base_size
    for current in range(32):
        if osh.u32(current) >= level:
            break
        offset = offset + osh.u32(size.x * size.y)
        size = osh.maximum((size + 1) / 2, osh.ivec2(1))
    return offset


@osh.function
def shadeSampleTextureLevel(
    base_offset: osh.u32, base_size: osh.ivec2, level: osh.u32,
    uv: osh.vec2, wrap: osh.uvec2, srgb: osh.boolean,
    linear_filter: osh.boolean,
) -> osh.vec4:
    divisor = osh.i32(osh.u32(1) << level)
    size = osh.maximum(
        (base_size + osh.ivec2(divisor) - 1) / divisor, osh.ivec2(1)
    )
    offset = shadeTextureMipOffset(base_offset, base_size, level)
    wrapped_uv = osh.vec2(
        shadeWrapTextureCoordinate(uv.x, wrap.x),
        shadeWrapTextureCoordinate(uv.y, wrap.y),
    )
    if not linear_filter:
        coordinate = osh.ivec2(osh.floor(wrapped_uv * osh.vec2(size)))
        return shadeFetchTextureTexel(offset, size, coordinate, wrap, srgb)
    texel = wrapped_uv * osh.vec2(size) - 0.5
    base = osh.ivec2(osh.floor(texel))
    fraction = texel - osh.floor(texel)
    top = osh.mix(
        shadeFetchTextureTexel(offset, size, base, wrap, srgb),
        shadeFetchTextureTexel(
            offset, size, base + osh.ivec2(1, 0), wrap, srgb
        ), fraction.x,
    )
    bottom = osh.mix(
        shadeFetchTextureTexel(
            offset, size, base + osh.ivec2(0, 1), wrap, srgb
        ),
        shadeFetchTextureTexel(
            offset, size, base + osh.ivec2(1, 1), wrap, srgb
        ), fraction.x,
    )
    return osh.mix(top, bottom, fraction.y)


@osh.function
def shadeSampleMaterialTexture(
    binding_index_value: osh.f32, uv0: osh.vec2, uv1: osh.vec2,
    srgb: osh.boolean, uv0_footprint: osh.f32,
    uv1_footprint: osh.f32,
) -> osh.vec4:
    binding_index = osh.i32(binding_index_value)
    if binding_index < 0:
        return osh.vec4(1.0)
    binding = texture_bindings[binding_index]
    texture_index = osh.i32(binding.texture_rotation.x)
    if texture_index < 0 or osh.u32(texture_index) >= texture_words[osh.u32(0)]:
        return osh.vec4(1.0)
    use_uv1 = binding.texture_rotation.w > 0.5
    uv = uv0
    footprint = uv0_footprint
    if use_uv1:
        uv = uv1
        footprint = uv1_footprint
    cosine = binding.texture_rotation.y
    sine = binding.texture_rotation.z
    scaled_uv = uv * binding.offset_scale.zw
    uv = binding.offset_scale.xy + osh.vec2(
        cosine * scaled_uv.x - sine * scaled_uv.y,
        sine * scaled_uv.x + cosine * scaled_uv.y,
    )
    footprint = footprint * osh.maximum(
        osh.absolute(binding.offset_scale.z),
        osh.absolute(binding.offset_scale.w),
    )
    metadata = osh.u32(1) + osh.u32(texture_index) * osh.u32(8)
    offset = texture_words[metadata + (osh.u32(0) if srgb else osh.u32(1))]
    size = osh.ivec2(
        texture_words[metadata + osh.u32(2)],
        texture_words[metadata + osh.u32(3)],
    )
    flags = texture_words[metadata + osh.u32(4)]
    level_count = texture_words[metadata + osh.u32(5)]
    wrap = osh.uvec2(flags & osh.u32(3), (flags >> osh.u32(2)) & osh.u32(3))
    lod = osh.clamp(
        osh.log2(osh.maximum(
            footprint * osh.f32(osh.maximum(size.x, size.y)), 1.0
        )), 0.0, osh.f32(osh.maximum(level_count, osh.u32(1)) - osh.u32(1)),
    )
    lower = osh.u32(osh.floor(lod))
    linear_filter = (flags & osh.u32(16)) != osh.u32(0)
    first = shadeSampleTextureLevel(
        offset, size, lower, uv, wrap, srgb, linear_filter
    )
    if not linear_filter or lower + osh.u32(1) >= level_count:
        return first
    second = shadeSampleTextureLevel(
        offset, size, lower + osh.u32(1), uv, wrap, srgb, True
    )
    return osh.mix(first, second, lod - osh.floor(lod))


@osh.function
def shadeSampleNativeMaterialTexture(
    binding_index_value: osh.f32, uv0: osh.vec2, uv1: osh.vec2,
    srgb: osh.boolean, uv0_footprint: osh.f32,
    uv1_footprint: osh.f32,
) -> osh.vec4:
    binding_index = osh.i32(binding_index_value)
    if binding_index < 0:
        return osh.vec4(1.0)
    binding = texture_bindings[binding_index]
    texture_index = osh.i32(binding.texture_rotation.x)
    if texture_index < 0 or osh.u32(texture_index) >= texture_words[osh.u32(0)]:
        return osh.vec4(1.0)
    uv = uv0
    footprint = uv0_footprint
    if binding.texture_rotation.w > 0.5:
        uv = uv1
        footprint = uv1_footprint
    cosine = binding.texture_rotation.y
    sine = binding.texture_rotation.z
    scaled_uv = uv * binding.offset_scale.zw
    uv = binding.offset_scale.xy + osh.vec2(
        cosine * scaled_uv.x - sine * scaled_uv.y,
        sine * scaled_uv.x + cosine * scaled_uv.y,
    )
    footprint = footprint * osh.maximum(
        osh.absolute(binding.offset_scale.z),
        osh.absolute(binding.offset_scale.w),
    )
    descriptor_index = texture_index * 2 + (0 if srgb else 1)
    size = native_textures.size(descriptor_index, 0)
    level_count = native_textures.levels(descriptor_index)
    lod = osh.clamp(
        osh.log2(osh.maximum(
            footprint * osh.f32(osh.maximum(size.x, size.y)), 1.0
        )), 0.0, osh.f32(osh.maximum(level_count, 1) - 1),
    )
    return native_textures.sample_lod(descriptor_index, uv, lod)


@osh.function
def shadeTriangleUvDensity(
    vertex_a: osh.vec3, vertex_b: osh.vec3, vertex_c: osh.vec3,
    uv_a: osh.vec2, uv_b: osh.vec2, uv_c: osh.vec2,
) -> osh.f32:
    first_uv = uv_b - uv_a
    second_uv = uv_c - uv_a
    uv_area = osh.absolute(
        first_uv.x * second_uv.y - first_uv.y * second_uv.x
    )
    world_area = osh.length(osh.cross(
        vertex_b - vertex_a, vertex_c - vertex_a
    ))
    return osh.sqrt(uv_area / osh.maximum(world_area, 0.00000001))


@osh.function
def shadeMaterialHasTextures(material: MaterialData) -> osh.boolean:
    return (
        osh.any_value(material.texture_indices >= osh.vec4(0.0))
        or material.texture_parameters.y >= 0.0
        or material.texture_parameters.w >= 0.0
    )


@osh.function
def shadeTextureBindingUsesUv1(binding_value: osh.f32) -> osh.boolean:
    binding_index = osh.i32(binding_value)
    return (
        binding_index >= 0
        and texture_bindings[binding_index].texture_rotation.w > 0.5
    )


@osh.function
def shadeMaterialUsesUv1(material: MaterialData) -> osh.boolean:
    fractional = material.ior_distance.z - osh.floor(material.ior_distance.z)
    return fractional > 0.125


@osh.function
def shadeTriangleTangent(
    vertex_a: osh.vec3, vertex_b: osh.vec3, vertex_c: osh.vec3,
    uv_a: osh.vec2, uv_b: osh.vec2, uv_c: osh.vec2,
    shading_normal: osh.vec3,
) -> osh.vec4:
    edge_a = vertex_b - vertex_a
    edge_b = vertex_c - vertex_a
    delta_a = uv_b - uv_a
    delta_b = uv_c - uv_a
    determinant = delta_a.x * delta_b.y - delta_a.y * delta_b.x
    if osh.absolute(determinant) < 0.00000001:
        fallback = osh.cross(shading_normal, osh.vec3(0.0, 1.0, 0.0))
        if osh.absolute(shading_normal.z) < 0.999:
            fallback = osh.cross(shading_normal, osh.vec3(0.0, 0.0, 1.0))
        return osh.vec4(osh.normalize(fallback), 1.0)
    inverse = 1.0 / determinant
    tangent = osh.normalize(
        (edge_a * delta_b.y - edge_b * delta_a.y) * inverse
    )
    bitangent = osh.normalize(
        (edge_b * delta_a.x - edge_a * delta_b.x) * inverse
    )
    handedness = 1.0
    if osh.dot(osh.cross(shading_normal, tangent), bitangent) < 0.0:
        handedness = -1.0
    return osh.vec4(tangent, handedness)


@osh.function
def shadeApplyMaterialTextures(
    input_material: MaterialData, uv0: osh.vec2, uv1: osh.vec2,
    uv0_footprint: osh.f32, uv1_footprint: osh.f32,
) -> MaterialData:
    material = input_material
    transmission = shadeSampleMaterialTexture(
        material.texture_parameters.w, uv0, uv1, False,
        uv0_footprint, uv1_footprint,
    ).r
    material.attenuation_transmission.a = (
        material.attenuation_transmission.a * transmission
    )
    material.base_roughness.rgb = (
        material.base_roughness.rgb * shadeSampleMaterialTexture(
            material.texture_indices.x, uv0, uv1, True,
            uv0_footprint, uv1_footprint,
        ).rgb
    )
    material.emission_metallic.rgb = (
        material.emission_metallic.rgb * shadeSampleMaterialTexture(
            material.texture_indices.z, uv0, uv1, True,
            uv0_footprint, uv1_footprint,
        ).rgb
    )
    if material.attenuation_transmission.a > 0.001:
        return material
    metallic_roughness = shadeSampleMaterialTexture(
        material.texture_indices.y, uv0, uv1, False,
        uv0_footprint, uv1_footprint,
    )
    material.base_roughness.a = (
        material.base_roughness.a * metallic_roughness.g
    )
    material.emission_metallic.a = (
        material.emission_metallic.a * metallic_roughness.b
    )
    occlusion = shadeSampleMaterialTexture(
        material.texture_parameters.y, uv0, uv1, False,
        uv0_footprint, uv1_footprint,
    ).r
    material.texture_parameters.w = osh.mix(
        1.0, occlusion, material.texture_parameters.z
    )
    return material


@osh.function
def shadeApplyNormalTexture(
    material: MaterialData, uv0: osh.vec2, uv1: osh.vec2,
    uv0_footprint: osh.f32, uv1_footprint: osh.f32,
    shading_normal: osh.vec3, tangent_data: osh.vec4,
) -> osh.vec3:
    texture_index = osh.i32(material.texture_indices.w)
    if texture_index < 0:
        return shading_normal
    tangent_normal = (
        shadeSampleMaterialTexture(
            material.texture_indices.w, uv0, uv1, False,
            uv0_footprint, uv1_footprint,
        ).xyz * 2.0 - 1.0
    )
    tangent_normal.xy = (
        tangent_normal.xy * material.texture_parameters.x
    )
    tangent_normal = osh.normalize(tangent_normal)
    tangent = osh.normalize(
        tangent_data.xyz
        - shading_normal * osh.dot(shading_normal, tangent_data.xyz)
    )
    bitangent = osh.cross(shading_normal, tangent) * tangent_data.w
    binding = texture_bindings[texture_index]
    cosine = binding.texture_rotation.y
    sine = binding.texture_rotation.z
    inverse_x = 1.0 / binding.offset_scale.z
    inverse_y = 1.0 / binding.offset_scale.w
    transformed_tangent = osh.normalize(
        tangent * (cosine * inverse_x)
        + bitangent * (-sine * inverse_y)
    )
    transformed_bitangent = osh.normalize(
        tangent * (sine * inverse_x)
        + bitangent * (cosine * inverse_y)
    )
    return osh.normalize(
        transformed_tangent * tangent_normal.x
        + transformed_bitangent * tangent_normal.y
        + shading_normal * tangent_normal.z
    )


@osh.function
def shadePathRng(path: WavePathState) -> osh.u32:
    return path.metadata.z


@osh.function
def shadePathBounce(path: WavePathState) -> osh.u32:
    return osh.u32(path.throughput.w)


@osh.function
def shadePathPreviousPdf(path: WavePathState) -> osh.f32:
    return path.radiance.w


@osh.function
def shadeMediumIor(stack: ShadeMediumStack, index: osh.u32) -> osh.f32:
    value = stack.ior_0_3.x
    if index == osh.u32(1): value = stack.ior_0_3.y
    if index == osh.u32(2): value = stack.ior_0_3.z
    if index == osh.u32(3): value = stack.ior_0_3.w
    if index == osh.u32(4): value = stack.ior_4_7.x
    if index == osh.u32(5): value = stack.ior_4_7.y
    if index == osh.u32(6): value = stack.ior_4_7.z
    if index == osh.u32(7): value = stack.ior_4_7.w
    if index == osh.u32(8): value = stack.ior_8_11.x
    if index == osh.u32(9): value = stack.ior_8_11.y
    if index == osh.u32(10): value = stack.ior_8_11.z
    if index == osh.u32(11): value = stack.ior_8_11.w
    if index == osh.u32(12): value = stack.ior_12_15.x
    if index == osh.u32(13): value = stack.ior_12_15.y
    if index == osh.u32(14): value = stack.ior_12_15.z
    if index == osh.u32(15): value = stack.ior_12_15.w
    return value


@osh.function
def shadeSetMediumIor(
    input_stack: ShadeMediumStack, index: osh.u32, value: osh.f32,
) -> ShadeMediumStack:
    stack = input_stack
    if index == osh.u32(0): stack.ior_0_3.x = value
    if index == osh.u32(1): stack.ior_0_3.y = value
    if index == osh.u32(2): stack.ior_0_3.z = value
    if index == osh.u32(3): stack.ior_0_3.w = value
    if index == osh.u32(4): stack.ior_4_7.x = value
    if index == osh.u32(5): stack.ior_4_7.y = value
    if index == osh.u32(6): stack.ior_4_7.z = value
    if index == osh.u32(7): stack.ior_4_7.w = value
    if index == osh.u32(8): stack.ior_8_11.x = value
    if index == osh.u32(9): stack.ior_8_11.y = value
    if index == osh.u32(10): stack.ior_8_11.z = value
    if index == osh.u32(11): stack.ior_8_11.w = value
    if index == osh.u32(12): stack.ior_12_15.x = value
    if index == osh.u32(13): stack.ior_12_15.y = value
    if index == osh.u32(14): stack.ior_12_15.z = value
    if index == osh.u32(15): stack.ior_12_15.w = value
    return stack


@osh.function
def shadeResolveSurface(
    primitive: osh.u32, barycentrics: osh.vec2,
    input_geometric_normal: osh.vec3, incoming: osh.vec3,
    cone_width: osh.f32,
) -> ShadeSurface:
    weights = osh.vec3(
        1.0 - barycentrics.x - barycentrics.y,
        barycentrics.x, barycentrics.y,
    )
    shading_normal = osh.normalize(
        attributes[primitive * osh.u32(3)].normal.xyz * weights.x
        + attributes[primitive * osh.u32(3) + osh.u32(1)].normal.xyz * weights.y
        + attributes[primitive * osh.u32(3) + osh.u32(2)].normal.xyz * weights.z
    )
    geometric_normal = input_geometric_normal
    if osh.dot(shading_normal, geometric_normal) < 0.0:
        shading_normal = -shading_normal
    entering = osh.dot(incoming, geometric_normal) < 0.0
    normal = shading_normal if entering else -shading_normal
    material = materials[primitive]
    vertex_a = vertices[primitive * osh.u32(3)].xyz
    vertex_b = vertices[primitive * osh.u32(3) + osh.u32(1)].xyz
    vertex_c = vertices[primitive * osh.u32(3) + osh.u32(2)].xyz
    uv0 = (
        attributes[primitive * osh.u32(3)].texcoord.xy * weights.x
        + attributes[primitive * osh.u32(3) + osh.u32(1)].texcoord.xy * weights.y
        + attributes[primitive * osh.u32(3) + osh.u32(2)].texcoord.xy * weights.z
    )
    uv1 = osh.vec2(0.0)
    if shadeMaterialHasTextures(material):
        uv0_footprint = cone_width * attributes[primitive * osh.u32(3)].normal.w
        uv1_footprint = 0.0
        if shadeMaterialUsesUv1(material):
            uv1 = (
                attributes[primitive * osh.u32(3)].texcoord.zw * weights.x
                + attributes[primitive * osh.u32(3) + osh.u32(1)].texcoord.zw * weights.y
                + attributes[primitive * osh.u32(3) + osh.u32(2)].texcoord.zw * weights.z
            )
            uv1_footprint = cone_width * shadeTriangleUvDensity(
                vertex_a, vertex_b, vertex_c,
                attributes[primitive * osh.u32(3)].texcoord.zw,
                attributes[primitive * osh.u32(3) + osh.u32(1)].texcoord.zw,
                attributes[primitive * osh.u32(3) + osh.u32(2)].texcoord.zw,
            )
        material = shadeApplyMaterialTextures(
            material, uv0, uv1, uv0_footprint, uv1_footprint
        )
        tangent_data = (
            attributes[primitive * osh.u32(3)].tangent * weights.x
            + attributes[primitive * osh.u32(3) + osh.u32(1)].tangent * weights.y
            + attributes[primitive * osh.u32(3) + osh.u32(2)].tangent * weights.z
        )
        if shadeTextureBindingUsesUv1(material.texture_indices.w):
            tangent_data = shadeTriangleTangent(
                vertex_a, vertex_b, vertex_c,
                attributes[primitive * osh.u32(3)].texcoord.zw,
                attributes[primitive * osh.u32(3) + osh.u32(1)].texcoord.zw,
                attributes[primitive * osh.u32(3) + osh.u32(2)].texcoord.zw,
                shading_normal,
            )
        shading_normal = shadeApplyNormalTexture(
            material, uv0, uv1, uv0_footprint, uv1_footprint,
            shading_normal, tangent_data,
        )
        if osh.dot(shading_normal, geometric_normal) < 0.0:
            shading_normal = -shading_normal
        normal = shading_normal if entering else -shading_normal
    return ShadeSurface(
        material, normal, geometric_normal, weights, uv0,
        vertex_a, vertex_b, vertex_c, entering,
    )


@osh.function
def shadeLoadHit(hit_index: osh.u32) -> ShadeHitInput:
    empty_ray = WaveRay(
        osh.vec4(0.0), osh.vec4(0.0), osh.u32(0), osh.u32(0),
        osh.u32(0), osh.u32(0),
    )
    empty_hit = WaveHit(
        osh.vec4(0.0, 0.0, 0.0, -1.0), osh.vec3(0.0),
        osh.u32(0xFFFFFFFF), osh.vec2(0.0), osh.u32(0), osh.u32(0),
    )
    if push.fused_intersection == osh.u32(0):
        if hit_index >= osh.minimum(hit_queue.count, hit_queue.capacity):
            return ShadeHitInput(False, empty_hit, empty_ray)
        hit = hit_queue.hits[hit_index]
        return ShadeHitInput(True, hit, input_queue.rays[hit.ray_index])
    if hit_index >= osh.minimum(input_queue.count, input_queue.capacity):
        return ShadeHitInput(False, empty_hit, empty_ray)
    input_ray = input_queue.rays[hit_index]
    query = osh.ray_query()
    query.initialize(
        scene_tlas, osh.u32(1), osh.u32(0x01),
        input_ray.origin_tmin.xyz, input_ray.origin_tmin.w,
        input_ray.direction_tmax.xyz, input_ray.direction_tmax.w,
    )
    while query.proceed():
        pass
    hit = empty_hit
    hit.path_index = input_ray.path_index
    hit.ray_index = hit_index
    if query.intersection_type(True) == osh.u32(1):
        distance = query.intersection_t(True)
        primitive = query.primitive_index(True) + query.instance_custom_index(True)
        vertex_a = vertices[primitive * osh.u32(3)].xyz
        vertex_b = vertices[primitive * osh.u32(3) + osh.u32(1)].xyz
        vertex_c = vertices[primitive * osh.u32(3) + osh.u32(2)].xyz
        hit.position_t = osh.vec4(
            input_ray.origin_tmin.xyz
            + distance * input_ray.direction_tmax.xyz, distance,
        )
        hit.geometric_normal = osh.normalize(osh.cross(
            vertex_b - vertex_a, vertex_c - vertex_a
        ))
        hit.primitive_index = primitive
        hit.barycentrics = query.barycentrics(True)
    return ShadeHitInput(True, hit, input_ray)


@osh.function
def shadeRayCone(ray: WaveRay, distance: osh.f32) -> osh.vec2:
    cone_width = osh.uint_bits_to_float(ray.padding_a)
    cone_spread = osh.uint_bits_to_float(ray.padding_b)
    return osh.vec2(cone_width + distance * cone_spread, cone_spread)


@osh.function
def shadeSetRayCone(
    input_ray: WaveRay, cone_width: osh.f32, cone_spread: osh.f32,
) -> WaveRay:
    ray = input_ray
    ray.padding_a = osh.float_bits_to_uint(cone_width)
    ray.padding_b = osh.float_bits_to_uint(cone_spread)
    return ray


@osh.function
def shadeProfileWork(
    counter: osh.u32, amount: osh.u32, bounce_input: osh.u32,
) -> osh.void:
    osh.atomic_add(work_counters[counter], amount)
    bounce = osh.minimum(bounce_input, osh.u32(7))
    if counter == osh.u32(0):
        osh.atomic_add(work_counters[osh.u32(16) + bounce], amount)
    if counter == osh.u32(1):
        osh.atomic_add(work_counters[osh.u32(24) + bounce], amount)
    if counter == osh.u32(3):
        osh.atomic_add(work_counters[osh.u32(32) + bounce], amount)


@osh.function
def shadePowerHeuristic(pdf_a: osh.f32, pdf_b: osh.f32) -> osh.f32:
    squared_a = pdf_a * pdf_a
    squared_b = pdf_b * pdf_b
    return squared_a / osh.maximum(squared_a + squared_b, 0.000001)


@osh.function
def shadeResolveEnvironmentMiss(
    input_path: WavePathState,
    environment: osh.vec3,
    environment_samples: osh.u32,
    unified_area_probability: osh.f32,
) -> ShadeMissResult:
    path = input_path
    environment_mis = 1.0
    previous_diffuse = (path.metadata.w & osh.u32(2)) != osh.u32(0)
    if previous_diffuse and environment_samples > osh.u32(0):
        pdf = shadePathPreviousPdf(path)
        light_pdf = pdf * osh.f32(environment_samples)
        previous_unified = (path.metadata.w & osh.u32(4)) != osh.u32(0)
        if previous_unified:
            light_pdf = pdf * (1.0 - unified_area_probability)
        environment_mis = shadePowerHeuristic(pdf, light_pdf)
    path.radiance = osh.vec4(
        path.radiance.rgb + path.throughput.rgb * environment * environment_mis,
        path.radiance.w,
    )
    path.metadata.w = path.metadata.w & ~osh.u32(1)
    return ShadeMissResult(path, environment_mis)


@osh.function
def shadeAdvanceBounceOrTerminate(
    input_path: WavePathState, max_bounces: osh.u32,
) -> WavePathState:
    path = input_path
    next_bounce = shadePathBounce(path) + osh.u32(1)
    path.throughput.w = osh.f32(next_bounce)
    if next_bounce >= max_bounces:
        path.metadata.w = path.metadata.w & ~osh.u32(1)
    return path


@osh.function
def shadeTransmitPath(
    input_path: WavePathState,
    input_stack: ShadeMediumStack,
    material: MaterialData,
    normal: osh.vec3,
    incoming: osh.vec3,
    entering: osh.boolean,
    hit_distance: osh.f32,
) -> ShadeTransmissionResult:
    path = input_path
    stack = input_stack
    medium_depth = osh.maximum(path.metadata.w >> osh.u32(8), osh.u32(1))
    current_ior = shadeMediumIor(stack, medium_depth - osh.u32(1))
    target_ior = 1.0
    if entering:
        target_ior = osh.maximum(material.ior_distance.x, 1.0001)
    elif medium_depth > osh.u32(1):
        target_ior = shadeMediumIor(stack, medium_depth - osh.u32(2))
    direction = osh.refract(incoming, normal, current_ior / target_ior)
    if osh.dot(direction, direction) < 0.01:
        direction = rayQueryReflect(incoming, normal)
    elif entering and medium_depth < osh.u32(16):
        stack = shadeSetMediumIor(stack, medium_depth, target_ior)
        medium_depth = medium_depth + osh.u32(1)
    elif not entering and medium_depth > osh.u32(1):
        medium_depth = medium_depth - osh.u32(1)
    optical_distance = 0.0 if entering else hit_distance
    if material.advanced1.z > 0.5:
        optical_distance = material.advanced1.w
    attenuation_exponent = optical_distance / osh.maximum(
        material.ior_distance.y, 0.000001,
    )
    absorption = osh.power(
        osh.maximum(
            material.attenuation_transmission.rgb, osh.vec3(0.000001),
        ),
        osh.vec3(attenuation_exponent),
    )
    tint = osh.mix(
        osh.vec3(1.0), material.base_roughness.rgb, material.advanced1.z,
    )
    path.throughput = osh.vec4(
        path.throughput.rgb * absorption * tint
        * material.attenuation_transmission.a,
        path.throughput.w,
    )
    return ShadeTransmissionResult(path, stack, direction, medium_depth)


@osh.function
def shadeTrackCustomTransmission(
    input_path: WavePathState,
    input_stack: ShadeMediumStack,
    material: MaterialData,
    direction: osh.vec3,
    entering: osh.boolean,
) -> ShadeTransmissionResult:
    path = input_path
    stack = input_stack
    medium_depth = osh.maximum(path.metadata.w >> osh.u32(8), osh.u32(1))
    target_ior = osh.maximum(material.ior_distance.x, 1.0001)
    if entering and medium_depth < osh.u32(16):
        stack = shadeSetMediumIor(stack, medium_depth, target_ior)
        medium_depth = medium_depth + osh.u32(1)
    elif not entering and medium_depth > osh.u32(1):
        medium_depth = medium_depth - osh.u32(1)
    return ShadeTransmissionResult(path, stack, direction, medium_depth)


@osh.function
def shadeApplyRussianRoulette(
    input_path: WavePathState,
    initial_random_state: osh.u32,
    next_bounce: osh.u32,
    transmission: osh.f32,
    roulette_start: osh.u32,
    minimum_survival: osh.f32,
) -> ShadeRouletteResult:
    path = input_path
    random_state = initial_random_state
    if (
        roulette_start == osh.u32(0)
        or next_bounce < roulette_start
        or transmission > 0.001
    ):
        return ShadeRouletteResult(path, random_state, True)
    survival = osh.clamp(
        osh.maximum(
            path.throughput.r,
            osh.maximum(path.throughput.g, path.throughput.b),
        ),
        minimum_survival, 0.95,
    )
    random_state = rayQueryRandomState(random_state)
    if rayQueryRandomValue(random_state) >= survival:
        path.metadata.w = path.metadata.w & ~osh.u32(1)
        path.metadata.z = random_state
        return ShadeRouletteResult(path, random_state, False)
    path.throughput = osh.vec4(
        path.throughput.rgb / survival, path.throughput.w
    )
    path.metadata.z = random_state
    return ShadeRouletteResult(path, random_state, True)


@osh.function
def shadeBuildContinuation(
    input_path: WavePathState,
    hit_position: osh.vec3,
    input_direction: osh.vec3,
    path_index: osh.u32,
    next_bounce: osh.u32,
    medium_depth: osh.u32,
    transmission: osh.f32,
    bsdf_pdf: osh.f32,
    random_state: osh.u32,
    cone_width: osh.f32,
    cone_spread: osh.f32,
    unified_secondary_nee: osh.boolean,
) -> ShadeContinuationResult:
    path = input_path
    direction = osh.normalize(input_direction)
    capture = path.metadata.w & osh.u32(8)
    path.throughput.w = osh.f32(next_bounce)
    path.metadata.w = osh.u32(1) | (medium_depth << osh.u32(8)) | capture
    if transmission <= 0.001:
        path.metadata.w = path.metadata.w | osh.u32(2)
        if unified_secondary_nee:
            path.metadata.w = path.metadata.w | osh.u32(4)
        path.radiance.w = bsdf_pdf
    path.metadata.z = random_state
    ray = WaveRay(
        osh.vec4(hit_position + direction * 0.002, 0.001),
        osh.vec4(direction, 1.0e30), path_index,
        osh.float_bits_to_uint(cone_width),
        osh.float_bits_to_uint(cone_spread), osh.u32(0),
    )
    return ShadeContinuationResult(path, ray)


@osh.function
def shadeReserveOutputIndex(subgroup_enqueue: osh.boolean) -> osh.u32:
    if not subgroup_enqueue:
        return osh.atomic_add(output_queue.count, osh.u32(1))
    active_lanes = osh.subgroup_ballot(True)
    base = osh.u32(0)
    if osh.subgroup_elect():
        base = osh.atomic_add(
            output_queue.count,
            osh.subgroup_ballot_bit_count(active_lanes),
        )
    base = osh.subgroup_broadcast_first(base)
    return base + osh.subgroup_ballot_exclusive_bit_count(active_lanes)


@osh.function
def shadeEnqueueContinuation(
    continuation: ShadeContinuationResult, output_index: osh.u32,
) -> ShadeEnqueueResult:
    path = continuation.path
    if output_index >= output_queue.capacity:
        osh.atomic_add(output_queue.overflow, osh.u32(1))
        path.metadata.w = path.metadata.w & ~osh.u32(1)
        return ShadeEnqueueResult(path, False)
    output_queue.rays[output_index] = continuation.ray
    return ShadeEnqueueResult(path, True)


@osh.function
def shadeSmoothstep(edge_low: osh.f32, edge_high: osh.f32, value: osh.f32) -> osh.f32:
    amount = osh.clamp(
        (value - edge_low) / osh.maximum(edge_high - edge_low, 0.000001),
        0.0, 1.0,
    )
    return amount * amount * (3.0 - 2.0 * amount)


@osh.function
def shadePreparePointLight(
    light: PointLightData, hit: osh.vec3, normal: osh.vec3,
) -> ShadePointLightSample:
    light_type = osh.i32(light.position_type.w + 0.5)
    if light_type == 3:
        return ShadePointLightSample(
            osh.vec3(0.0), hit, osh.vec3(0.0), 0.0, 0.0, False
        )
    direction = osh.vec3(0.0)
    distance_squared = 1.0
    distance_to_light = 10000.0
    attenuation = 1.0
    if light_type == 1:
        direction = -osh.normalize(light.direction_range.xyz)
    else:
        offset = light.position_type.xyz - hit
        distance_squared = osh.maximum(osh.dot(offset, offset), 0.000001)
        distance_to_light = osh.sqrt(distance_squared)
        direction = offset / distance_to_light
        if (
            light.direction_range.w > 0.0
            and distance_to_light > light.direction_range.w
        ):
            return ShadePointLightSample(
                direction, hit, osh.vec3(0.0), 0.0, 0.0, False
            )
        attenuation = 1.0 / distance_squared
        if light_type == 2:
            cone = osh.dot(osh.normalize(light.direction_range.xyz), -direction)
            spot = shadeSmoothstep(
                light.spot_parameters.y, light.spot_parameters.x, cone
            )
            if spot <= 0.0:
                return ShadePointLightSample(
                    direction, hit, osh.vec3(0.0), 0.0, 0.0, False
                )
            attenuation = attenuation * spot
    cosine = osh.maximum(osh.dot(normal, direction), 0.0)
    if cosine <= 0.0:
        return ShadePointLightSample(
            direction, hit, osh.vec3(0.0), cosine, 0.0, False
        )
    shadow_origin = hit + normal * 0.002
    shadow_distance = osh.maximum(distance_to_light - 0.004, 0.001)
    if light_type == 1:
        shadow_distance = 10000.0
    incident = (
        light.color_intensity.rgb * light.color_intensity.a * attenuation
    )
    return ShadePointLightSample(
        direction, shadow_origin, incident, cosine, shadow_distance, True
    )


@osh.function
def shadePointLightVisible(sample: ShadePointLightSample) -> osh.boolean:
    if not sample.valid:
        return False
    shadow = osh.ray_query()
    shadow.initialize(
        scene_tlas, osh.u32(5), osh.u32(0x01), sample.shadow_origin, 0.001,
        sample.direction, sample.shadow_distance,
    )
    while shadow.proceed():
        pass
    return shadow.intersection_type(True) == osh.u32(0)


@osh.function
def shadePointLightContribution(
    sample: ShadePointLightSample, material: MaterialData,
    normal: osh.vec3, incoming: osh.vec3,
    volume_transmittance: osh.vec3,
) -> osh.vec3:
    return (
        shadeEvaluatePbr(material, normal, -incoming, sample.direction)
        * sample.incident * volume_transmittance * sample.cosine
    )


@osh.function
def shadeSelectAreaLight(area_light_count: osh.u32, selection: osh.f32) -> osh.u32:
    lower = osh.u32(0)
    upper = area_light_count - osh.u32(1)
    for step in range(32):
        if lower >= upper:
            break
        middle = lower + (upper - lower) / osh.u32(2)
        if selection <= area_lights[middle].distribution.x:
            upper = middle
        else:
            lower = middle + osh.u32(1)
    return lower


@osh.function
def shadePrepareAreaLight(
    hit: osh.vec3,
    normal: osh.vec3,
    initial_random_state: osh.u32,
    sample_index: osh.u32,
    sample_count: osh.u32,
    area_light_count: osh.u32,
    technique_probability: osh.f32,
) -> ShadeAreaLightSample:
    random_state = initial_random_state
    if area_light_count == osh.u32(0):
        return ShadeAreaLightSample(
            normal, hit, osh.vec3(0.0), 0.0, 0.0,
            osh.f32(sample_count), 0.0, random_state, False,
        )
    random_state = rayQueryRandomState(random_state)
    selection = (
        osh.f32(sample_index) + rayQueryRandomValue(random_state)
    ) / osh.f32(sample_count)
    light_index = shadeSelectAreaLight(area_light_count, selection)
    light = area_lights[light_index]
    random_state = rayQueryRandomState(random_state)
    root_u = osh.sqrt(rayQueryRandomValue(random_state))
    random_state = rayQueryRandomState(random_state)
    value_v = rayQueryRandomValue(random_state)
    light_position = (
        (1.0 - root_u) * light.a.xyz
        + root_u * (1.0 - value_v) * light.b.xyz
        + root_u * value_v * light.c.xyz
    )
    offset = light_position - hit
    distance_squared = osh.dot(offset, offset)
    distance_to_light = osh.sqrt(distance_squared)
    direction = offset / osh.maximum(distance_to_light, 0.000001)
    surface_cosine = osh.maximum(osh.dot(normal, direction), 0.0)
    light_normal = osh.normalize(osh.cross(
        light.b.xyz - light.a.xyz, light.c.xyz - light.a.xyz
    ))
    raw_light_cosine = osh.dot(light_normal, -direction)
    light_cosine = osh.maximum(raw_light_cosine, 0.0)
    if light.distribution.z > 0.5:
        light_cosine = osh.absolute(raw_light_cosine)
    if surface_cosine <= 0.0 or light_cosine <= 0.000001:
        return ShadeAreaLightSample(
            direction, hit, light.emission_area.rgb, surface_cosine, 0.0,
            osh.f32(sample_count), 0.0, random_state, False,
        )
    shadow_origin = hit + normal * 0.002
    shadow_distance = osh.maximum(distance_to_light - 0.004, 0.001)
    light_pdf = (
        light.distribution.y * distance_squared
        / osh.maximum(light_cosine * light.emission_area.a, 0.000001)
    )
    effective_pdf = light_pdf * osh.maximum(technique_probability, 0.000001)
    return ShadeAreaLightSample(
        direction, shadow_origin, light.emission_area.rgb, surface_cosine,
        effective_pdf, osh.f32(sample_count), shadow_distance, random_state, True,
    )


@osh.function
def shadeAreaLightVisible(sample: ShadeAreaLightSample) -> osh.boolean:
    if not sample.valid:
        return False
    shadow = osh.ray_query()
    shadow.initialize(
        scene_tlas, osh.u32(5), osh.u32(0x01), sample.shadow_origin, 0.001,
        sample.direction, sample.shadow_distance,
    )
    while shadow.proceed():
        pass
    return shadow.intersection_type(True) == osh.u32(0)


@osh.function
def shadeAreaLightContribution(
    sample: ShadeAreaLightSample,
    material: MaterialData,
    normal: osh.vec3,
    incoming: osh.vec3,
    volume_transmittance: osh.vec3,
) -> osh.vec3:
    bsdf_pdf = shadePbrPdf(material, normal, -incoming, sample.direction)
    mis = shadePowerHeuristic(
        sample.effective_pdf * sample.sample_count, bsdf_pdf
    )
    return (
        shadeEvaluatePbr(material, normal, -incoming, sample.direction)
        * sample.emission * sample.surface_cosine * mis
        * volume_transmittance / osh.maximum(sample.effective_pdf, 0.000001)
    )


@osh.function
def shadeUnifiedAreaDomainProbability(
    area_light_count: osh.u32,
    environment_samples: osh.u32,
    area_light_weight: osh.f32,
) -> osh.f32:
    if area_light_count == osh.u32(0):
        return 0.0
    if environment_samples == osh.u32(0):
        return 1.0
    area_weight = osh.sqrt(osh.maximum(area_light_weight, 0.000001))
    return area_weight / (area_weight + 1.0)


@osh.function
def shadeSecondaryNeeHash(value: osh.u32) -> osh.u32:
    value = value ^ (value >> osh.u32(16))
    value = value * osh.u32(0x7FEB352D)
    value = value ^ (value >> osh.u32(15))
    value = value * osh.u32(0x846CA68B)
    return value ^ (value >> osh.u32(16))


@osh.function
def shadeSelectSecondaryNee(
    probability: osh.f32,
    pixel_index: osh.u32,
    frame_sample: osh.u32,
    bounce: osh.u32,
) -> osh.boolean:
    if probability >= 0.999999:
        return True
    frame_index = frame_sample >> osh.u32(8)
    sample_index = frame_sample & osh.u32(255)
    scramble = shadeSecondaryNeeHash(
        pixel_index
        ^ shadeSecondaryNeeHash(bounce + osh.u32(1))
        ^ shadeSecondaryNeeHash(sample_index + osh.u32(1))
    )
    sequence = osh.bitfield_reverse(frame_index) ^ scramble
    selector = (osh.f32(sequence) + 0.5) * (1.0 / 4294967296.0)
    return selector < probability


@osh.function
def shadeEmissionContribution(
    path: WavePathState,
    material: MaterialData,
    vertex_a: osh.vec3,
    vertex_b: osh.vec3,
    vertex_c: osh.vec3,
    geometric_normal: osh.vec3,
    incoming: osh.vec3,
    hit_distance: osh.f32,
    entering: osh.boolean,
    area_light_weight: osh.f32,
    secondary_area_samples: osh.u32,
    unified_secondary_nee: osh.boolean,
    unified_area_probability: osh.f32,
) -> ShadeEmissionResult:
    emission_visible = entering or material.ior_distance.w > 0.5
    if not emission_visible:
        return ShadeEmissionResult(osh.vec3(0.0), 1.0)
    emission_mis = 1.0
    previous_diffuse = (path.metadata.w & osh.u32(2)) != osh.u32(0)
    emission = material.emission_metallic.rgb
    if previous_diffuse and osh.dot(emission, emission) > 0.0:
        area = 0.5 * osh.length(osh.cross(
            vertex_b - vertex_a, vertex_c - vertex_a
        ))
        raw_light_cosine = osh.dot(geometric_normal, -incoming)
        light_cosine = osh.maximum(raw_light_cosine, 0.0)
        if material.ior_distance.w > 0.5:
            light_cosine = osh.absolute(raw_light_cosine)
        luminance = osh.dot(emission, osh.vec3(0.2126, 0.7152, 0.0722))
        selection_pdf = (
            area * luminance / osh.maximum(area_light_weight, 0.000001)
        )
        light_pdf = (
            selection_pdf * hit_distance * hit_distance
            / osh.maximum(light_cosine * area, 0.000001)
        )
        sampled_light_pdf = light_pdf * osh.f32(
            osh.maximum(secondary_area_samples, osh.u32(1))
        )
        if unified_secondary_nee:
            sampled_light_pdf = light_pdf * unified_area_probability
        emission_mis = shadePowerHeuristic(
            shadePathPreviousPdf(path), sampled_light_pdf
        )
    return ShadeEmissionResult(
        path.throughput.rgb * emission * emission_mis, emission_mis
    )


@osh.function
def shadeAnalyticEnvironment(direction: osh.vec3) -> osh.vec3:
    sky = osh.maximum(direction.y, 0.0)
    return osh.mix(
        osh.vec3(0.018, 0.022, 0.032),
        osh.vec3(0.32, 0.46, 0.72), sky,
    )


@osh.function
def shadeFindEnvironmentLight(point_light_count: osh.u32) -> ShadeEnvironmentDescriptor:
    for light_index in range(64):
        if osh.u32(light_index) >= point_light_count:
            break
        light = point_lights[osh.u32(light_index)]
        if osh.i32(light.position_type.w + 0.5) == 3:
            return ShadeEnvironmentDescriptor(
                light.color_intensity, light.spot_parameters, True
            )
    return ShadeEnvironmentDescriptor(osh.vec4(1.0), osh.vec4(-1.0), False)


@osh.function
def shadeEnvironmentUv(direction: osh.vec3, rotation: osh.f32) -> osh.vec2:
    longitude = osh.arctangent2(direction.z, direction.x) + rotation
    return osh.vec2(
        osh.fraction(longitude * 0.15915494309189535 + 0.5),
        osh.arccosine(osh.clamp(direction.y, -1.0, 1.0))
        * 0.3183098861837907,
    )


@osh.function
def shadeDecodeEnvironmentRadiance(
    encoded: osh.vec3,
    exposure: osh.f32,
    color_intensity: osh.vec4,
    texture_index: osh.f32,
) -> osh.vec3:
    radiance = osh.vec3(1.0)
    if texture_index >= 0.0:
        radiance = osh.exp2(encoded * exposure) - osh.vec3(1.0)
    return radiance * color_intensity.rgb * color_intensity.a


@osh.function
def shadeSamplePackedSceneTexture(
    texture_index: osh.i32, uv: osh.vec2,
) -> osh.vec4:
    if texture_index < 0 or osh.u32(texture_index) >= texture_words[osh.u32(0)]:
        return osh.vec4(0.0)
    metadata = osh.u32(1) + osh.u32(texture_index) * osh.u32(8)
    offset = texture_words[metadata + osh.u32(1)]
    size = osh.ivec2(
        texture_words[metadata + osh.u32(2)],
        texture_words[metadata + osh.u32(3)],
    )
    flags = texture_words[metadata + osh.u32(4)]
    wrap = osh.uvec2(
        flags & osh.u32(3), (flags >> osh.u32(2)) & osh.u32(3)
    )
    return shadeSampleTextureLevel(
        offset, size, osh.u32(0), uv, wrap, False,
        (flags & osh.u32(16)) != osh.u32(0),
    )


@osh.function
def shadeSampleNativeSceneTexture(
    texture_index: osh.i32, uv: osh.vec2,
) -> osh.vec4:
    if texture_index < 0 or osh.u32(texture_index) >= texture_words[osh.u32(0)]:
        return osh.vec4(0.0)
    return native_textures.sample_lod(texture_index * 2 + 1, uv, 0.0)


@osh.function
def shadeEnvironmentRadiance(
    direction: osh.vec3, point_light_count: osh.u32,
) -> osh.vec3:
    descriptor = shadeFindEnvironmentLight(point_light_count)
    if not descriptor.valid:
        return shadeAnalyticEnvironment(direction)
    texture_index = osh.i32(descriptor.texture_parameters.x)
    encoded = osh.vec3(0.0)
    if texture_index >= 0:
        uv = shadeEnvironmentUv(
            direction, descriptor.texture_parameters.y
        )
        encoded = shadeSamplePackedSceneTexture(texture_index, uv).rgb
    return shadeDecodeEnvironmentRadiance(
        encoded, descriptor.texture_parameters.z,
        descriptor.color_intensity, descriptor.texture_parameters.x,
    )


@osh.function
def shadeAccumulateDirectContribution(
    input_path: WavePathState,
    contribution: osh.vec3,
    nee_probability: osh.f32,
) -> WavePathState:
    path = input_path
    path.radiance = osh.vec4(
        path.radiance.rgb
        + path.throughput.rgb * contribution
        / osh.maximum(nee_probability, 0.000001),
        path.radiance.w,
    )
    return path


@osh.function
def shadeScatterOpaquePath(
    input_path: WavePathState,
    material: MaterialData,
    normal: osh.vec3,
    incoming: osh.vec3,
    initial_random_state: osh.u32,
    input_cone_spread: osh.f32,
) -> ShadeOpaqueScatterResult:
    path = input_path
    sampled = shadeSamplePbr(
        material, normal, incoming, initial_random_state
    )
    path.throughput = osh.vec4(
        path.throughput.rgb * sampled.weight, path.throughput.w
    )
    return ShadeOpaqueScatterResult(
        path, sampled.outgoing, sampled.pdf, sampled.random_state,
        input_cone_spread + material.base_roughness.a * 0.25,
        sampled.sampled_specular,
    )


@osh.function
def shadeIsVolumePrimitive(primitive: osh.u32) -> osh.boolean:
    return triangle_volumes[primitive] != osh.u32(0xFFFFFFFF)


@osh.function
def shadeVolumeLocalCoordinate(
    header: VolumeHeader, world_position: osh.vec3,
) -> osh.vec3:
    return osh.clamp(
        (header.world_to_local * osh.vec4(world_position, 1.0)).xyz,
        osh.vec3(0.0), osh.vec3(1.0),
    )


@osh.function
def shadeVolumeInterval(
    header: VolumeHeader, origin: osh.vec3, direction: osh.vec3,
) -> osh.vec2:
    local_origin = (header.world_to_local * osh.vec4(origin, 1.0)).xyz
    local_direction = (header.world_to_local * osh.vec4(direction, 0.0)).xyz
    safe_direction = osh.vec3(
        local_direction.x if osh.absolute(local_direction.x) > 1.0e-10 else 1.0e-10,
        local_direction.y if osh.absolute(local_direction.y) > 1.0e-10 else 1.0e-10,
        local_direction.z if osh.absolute(local_direction.z) > 1.0e-10 else 1.0e-10,
    )
    first = (osh.vec3(0.0) - local_origin) / safe_direction
    second = (osh.vec3(1.0) - local_origin) / safe_direction
    lower = osh.minimum(first, second)
    upper = osh.maximum(first, second)
    return osh.vec2(
        osh.maximum(osh.maximum(lower.x, lower.y), lower.z),
        osh.minimum(osh.minimum(upper.x, upper.y), upper.z),
    )


@osh.function
def shadeVolumeTransferSample(
    header: VolumeHeader, value: osh.f32,
) -> osh.vec4:
    offset = osh.u32(header.value_parameters.x)
    count = osh.u32(header.value_parameters.y)
    coordinate = (
        osh.clamp(value, 0.0, 1.0)
        * osh.f32(osh.maximum(count, osh.u32(1)) - osh.u32(1))
    )
    lower = osh.u32(osh.floor(coordinate))
    upper = osh.minimum(lower + osh.u32(1), count - osh.u32(1))
    return osh.mix(
        volume_transfer[offset + lower], volume_transfer[offset + upper],
        osh.fraction(coordinate),
    )


@osh.function
def shadeVolumeScalar(
    volume_index: osh.u32,
    header: VolumeHeader,
    world_position: osh.vec3,
) -> osh.f32:
    local = shadeVolumeLocalCoordinate(header, world_position)
    return volume_textures.sample(volume_index, local)


@osh.function
def shadeVolumeBrickIndexFromVoxel(
    header: VolumeHeader, voxel_position: osh.vec3,
) -> osh.uvec3:
    brick_grid = header.acceleration_parameters.yzw
    unclamped = osh.uvec3(osh.floor(
        osh.maximum(voxel_position, osh.vec3(0.0)) / 8.0
    ))
    return osh.minimum(unclamped, brick_grid - osh.uvec3(1))


@osh.function
def shadeVolumeBrickOccupiedAtVoxel(
    header: VolumeHeader, voxel_position: osh.vec3,
) -> osh.boolean:
    brick_grid = header.acceleration_parameters.yzw
    if (
        brick_grid.x == osh.u32(0)
        or brick_grid.y == osh.u32(0)
        or brick_grid.z == osh.u32(0)
    ):
        return True
    brick = shadeVolumeBrickIndexFromVoxel(header, voxel_position)
    linear_index = brick.x + brick_grid.x * (
        brick.y + brick_grid.y * brick.z
    )
    return (
        volume_scalars[header.acceleration_parameters.x + linear_index] > 0.5
    )


@osh.function
def shadeVolumeVoxelRay(
    header: VolumeHeader, origin: osh.vec3, direction: osh.vec3,
) -> ShadeVolumeVoxelRay:
    local_direction = (
        header.world_to_local * osh.vec4(direction, 0.0)
    ).xyz
    voxel_extent = osh.vec3(header.dimensions_offset.xyz) - osh.vec3(1.0)
    voxel_origin = (
        header.world_to_local * osh.vec4(origin, 1.0)
    ).xyz * voxel_extent
    return ShadeVolumeVoxelRay(voxel_origin, local_direction * voxel_extent)


@osh.function
def shadeVolumeAxisExitDelta(
    position: osh.f32, direction: osh.f32,
    lower: osh.f32, upper: osh.f32,
) -> osh.f32:
    if osh.absolute(direction) <= 1.0e-10:
        return 1.0e30
    boundary = upper if direction > 0.0 else lower
    candidate = (boundary - position) / direction
    if candidate >= -1.0e-6:
        return osh.maximum(candidate, 0.0)
    return 1.0e30


@osh.function
def shadeVolumeBrickStepAtVoxel(
    header: VolumeHeader,
    voxel_position_input: osh.vec3,
    voxel_direction: osh.vec3,
    distance: osh.f32,
) -> ShadeVolumeBrickStep:
    voxel_extent = osh.vec3(header.dimensions_offset.xyz) - osh.vec3(1.0)
    voxel_position = osh.clamp(
        voxel_position_input, osh.vec3(0.0), voxel_extent
    )
    brick = shadeVolumeBrickIndexFromVoxel(header, voxel_position)
    lower = osh.vec3(brick) * 8.0
    upper = osh.minimum(lower + osh.vec3(8.0), voxel_extent)
    next_delta = osh.minimum(
        shadeVolumeAxisExitDelta(
            voxel_position.x, voxel_direction.x, lower.x, upper.x
        ),
        osh.minimum(
            shadeVolumeAxisExitDelta(
                voxel_position.y, voxel_direction.y, lower.y, upper.y
            ),
            shadeVolumeAxisExitDelta(
                voxel_position.z, voxel_direction.z, lower.z, upper.z
            ),
        ),
    )
    return ShadeVolumeBrickStep(
        distance + next_delta,
        shadeVolumeBrickOccupiedAtVoxel(header, voxel_position),
    )


@osh.function
def shadeExtendVolumeUnion(
    input_bounds: ShadeVolumeUnionBounds,
    interval: osh.vec2,
    header: VolumeHeader,
    surface_distance: osh.f32,
) -> ShadeVolumeUnionBounds:
    bounds = input_bounds
    entry = osh.maximum(interval.x, 0.0)
    exit_distance = osh.minimum(interval.y, surface_distance)
    if exit_distance <= entry:
        return bounds
    if not bounds.valid:
        return ShadeVolumeUnionBounds(
            entry, exit_distance,
            osh.maximum(header.render_parameters.x, 1.0e-5), True,
        )
    bounds.entry = osh.minimum(bounds.entry, entry)
    bounds.exit_distance = osh.maximum(bounds.exit_distance, exit_distance)
    bounds.reference_step = osh.minimum(
        bounds.reference_step,
        osh.maximum(header.render_parameters.x, 1.0e-5),
    )
    return bounds


@osh.function
def shadeAccumulateOverlappingMedium(
    input_medium: ShadeOverlappingMedium,
    header: VolumeHeader,
    sample_value: osh.vec4,
    scattering_source: osh.vec3,
) -> ShadeOverlappingMedium:
    medium = input_medium
    extinction = shadeVolumeExtinction(header, sample_value)
    medium.extinction = medium.extinction + extinction
    medium.emission_extinction = (
        medium.emission_extinction
        + extinction * (
            sample_value.rgb * header.value_parameters.w + scattering_source
        )
    )
    return medium


@osh.function
def shadeCompositeOverlappingStep(
    input_state: ShadeVolumeMarchState,
    medium: ShadeOverlappingMedium,
    step_size: osh.f32,
) -> ShadeVolumeMarchState:
    state = input_state
    if medium.extinction <= 1.0e-8:
        return state
    alpha = 1.0 - osh.exp(-medium.extinction * step_size)
    source = medium.emission_extinction / medium.extinction
    state.integrated = (
        state.integrated + state.transmittance * alpha * source
    )
    state.transmittance = state.transmittance * (1.0 - alpha)
    return state


@osh.function
def shadeVolumeOpaqueVisibility(
    world_position: osh.vec3,
    direction: osh.vec3,
    maximum_distance: osh.f32,
) -> osh.f32:
    shadow_distance = 1.0e30
    if maximum_distance < 1.0e29:
        shadow_distance = osh.maximum(maximum_distance - 0.004, 0.001)
    shadow = osh.ray_query()
    shadow.initialize(
        scene_tlas, osh.u32(5), osh.u32(0x01),
        world_position + direction * 0.002, 0.001,
        direction, shadow_distance,
    )
    while shadow.proceed():
        pass
    return 1.0 if shadow.intersection_type(True) == osh.u32(0) else 0.0


@osh.function
def shadeApproximateVolumeLightTransmittance(
    world_position: osh.vec3,
    light_direction: osh.vec3,
    light_distance: osh.f32,
) -> osh.f32:
    volume_count = osh.minimum(
        osh.u32(volume_headers[osh.u32(0)].render_parameters.z), osh.u32(16)
    )
    optical_depth = 0.0
    for volume_index in range(16):
        if osh.u32(volume_index) >= volume_count:
            break
        medium = volume_headers[osh.u32(volume_index)]
        interval = shadeVolumeInterval(
            medium, world_position, light_direction
        )
        entry = osh.maximum(interval.x, 0.0)
        exit_distance = osh.minimum(interval.y, light_distance)
        if exit_distance <= entry:
            continue
        midpoint = 0.5 * (entry + exit_distance)
        sample_value = shadeVolumeTransferSample(
            medium,
            shadeVolumeScalar(
                osh.u32(volume_index), medium,
                world_position + light_direction * midpoint,
            ),
        )
        optical_depth = (
            optical_depth
            + shadeVolumeExtinction(medium, sample_value)
            * (exit_distance - entry)
        )
    return osh.exp(-optical_depth)


@osh.function
def shadeVolumePointLightScattering(
    header: VolumeHeader,
    light: PointLightData,
    world_position: osh.vec3,
    ray_direction: osh.vec3,
) -> osh.vec3:
    light_type = osh.i32(light.position_type.w + 0.5)
    if light_type == 3:
        return osh.vec3(0.0)
    incoming = osh.vec3(0.0)
    distance_to_light = 10000.0
    attenuation = 1.0
    if light_type == 1:
        incoming = -osh.normalize(light.direction_range.xyz)
    else:
        offset = light.position_type.xyz - world_position
        distance_squared = osh.maximum(osh.dot(offset, offset), 1.0e-6)
        distance_to_light = osh.sqrt(distance_squared)
        incoming = offset / distance_to_light
        if (
            light.direction_range.w > 0.0
            and distance_to_light > light.direction_range.w
        ):
            return osh.vec3(0.0)
        attenuation = 1.0 / distance_squared
        if light_type == 2:
            cone = osh.dot(
                osh.normalize(light.direction_range.xyz), -incoming
            )
            spot = shadeSmoothstep(
                light.spot_parameters.y, light.spot_parameters.x, cone
            )
            if spot <= 0.0:
                return osh.vec3(0.0)
            attenuation = attenuation * spot
    incident = light.color_intensity.rgb * light.color_intensity.a * attenuation
    incident = incident * shadeApproximateVolumeLightTransmittance(
        world_position, incoming, distance_to_light
    )
    incident = incident * shadeVolumeOpaqueVisibility(
        world_position, incoming, distance_to_light
    )
    return incident * shadeVolumePhase(
        header, osh.dot(-incoming, -ray_direction)
    )


@osh.function
def shadeVolumeAreaLightScattering(
    header: VolumeHeader,
    light: AreaLightData,
    world_position: osh.vec3,
    ray_direction: osh.vec3,
) -> osh.vec3:
    light_position = (light.a.xyz + light.b.xyz + light.c.xyz) / 3.0
    offset = light_position - world_position
    distance_squared = osh.maximum(osh.dot(offset, offset), 1.0e-6)
    distance_to_light = osh.sqrt(distance_squared)
    incoming = offset / distance_to_light
    light_normal = osh.normalize(osh.cross(
        light.b.xyz - light.a.xyz, light.c.xyz - light.a.xyz
    ))
    raw_cosine = osh.dot(light_normal, -incoming)
    light_cosine = osh.maximum(raw_cosine, 0.0)
    if light.distribution.z > 0.5:
        light_cosine = osh.absolute(raw_cosine)
    if light_cosine <= 1.0e-6:
        return osh.vec3(0.0)
    visibility = shadeVolumeOpaqueVisibility(
        world_position, incoming, distance_to_light
    )
    transmittance = shadeApproximateVolumeLightTransmittance(
        world_position, incoming, distance_to_light
    )
    incident = (
        light.emission_area.rgb * light.emission_area.a * light_cosine
        / distance_squared * visibility * transmittance
    )
    return incident * shadeVolumePhase(
        header, osh.dot(-incoming, -ray_direction)
    )


@osh.function
def shadeVolumeEnvironmentScattering(
    header: VolumeHeader,
    environment_radiance: osh.vec3,
    incoming: osh.vec3,
    world_position: osh.vec3,
    ray_direction: osh.vec3,
    sample_count: osh.u32,
) -> osh.vec3:
    visibility = shadeVolumeOpaqueVisibility(
        world_position, incoming, 1.0e30
    )
    transmittance = shadeApproximateVolumeLightTransmittance(
        world_position, incoming, 1.0e30
    )
    incident = (
        environment_radiance * visibility * transmittance
        * (12.5663706144 / osh.f32(osh.maximum(sample_count, osh.u32(1))))
    )
    return incident * shadeVolumePhase(
        header, osh.dot(-incoming, -ray_direction)
    )


@osh.function
def shadeFinalizeVolumeScattering(
    header: VolumeHeader,
    single_scattered: osh.vec3,
    isotropic_scattered: osh.vec3,
    optical_depth: osh.f32,
) -> osh.vec3:
    scattered = single_scattered
    scattering_orders = osh.clamp(
        osh.u32(header.multiple_scattering_parameters.w + 0.5),
        osh.u32(1), osh.u32(8),
    )
    ratio = osh.clamp(
        header.multiple_scattering_parameters.rgb
        * (1.0 - osh.exp(-osh.maximum(optical_depth, 0.0))),
        osh.vec3(0.0), osh.vec3(0.999),
    )
    order_weight = ratio
    for order in range(2, 9):
        if osh.u32(order) > scattering_orders:
            break
        scattered = scattered + isotropic_scattered * order_weight
        order_weight = order_weight * ratio
    return (
        scattered * header.scattering_parameters.rgb
        * header.scattering_parameters.w
    )


@osh.function
def shadeVolumeScatteringSource(
    header: VolumeHeader,
    world_position: osh.vec3,
    ray_direction: osh.vec3,
    optical_depth: osh.f32,
    point_light_count: osh.u32,
    area_light_count: osh.u32,
    environment_sample_count: osh.u32,
    environment_radiance_0: osh.vec3,
    environment_radiance_1: osh.vec3,
    environment_radiance_2: osh.vec3,
    environment_radiance_3: osh.vec3,
    multiple_scattering: osh.boolean,
) -> osh.vec3:
    if header.scattering_parameters.w <= 0.0:
        return osh.vec3(0.0)
    single_scattered = osh.vec3(0.0)
    isotropic_scattered = osh.vec3(0.0)
    for light_index in range(64):
        if osh.u32(light_index) >= osh.minimum(point_light_count, osh.u32(64)):
            break
        light = point_lights[osh.u32(light_index)]
        contribution = shadeVolumePointLightScattering(
            header, light, world_position, ray_direction
        )
        single_scattered = single_scattered + contribution
        isotropic_scattered = isotropic_scattered + contribution
    for light_index in range(64):
        if osh.u32(light_index) >= osh.minimum(area_light_count, osh.u32(64)):
            break
        contribution = shadeVolumeAreaLightScattering(
            header, area_lights[osh.u32(light_index)], world_position,
            ray_direction,
        )
        single_scattered = single_scattered + contribution
        isotropic_scattered = isotropic_scattered + contribution
    environment_count = osh.minimum(environment_sample_count, osh.u32(4))
    if environment_count > osh.u32(0):
        direction = osh.normalize(osh.vec3(1.0, 1.0, 1.0))
        contribution = shadeVolumeEnvironmentScattering(
            header, environment_radiance_0, direction, world_position,
            ray_direction, environment_count,
        )
        single_scattered = single_scattered + contribution
        isotropic_scattered = isotropic_scattered + contribution
    if environment_count > osh.u32(1):
        direction = osh.normalize(osh.vec3(-1.0, -1.0, 1.0))
        contribution = shadeVolumeEnvironmentScattering(
            header, environment_radiance_1, direction, world_position,
            ray_direction, environment_count,
        )
        single_scattered = single_scattered + contribution
        isotropic_scattered = isotropic_scattered + contribution
    if environment_count > osh.u32(2):
        direction = osh.normalize(osh.vec3(-1.0, 1.0, -1.0))
        contribution = shadeVolumeEnvironmentScattering(
            header, environment_radiance_2, direction, world_position,
            ray_direction, environment_count,
        )
        single_scattered = single_scattered + contribution
        isotropic_scattered = isotropic_scattered + contribution
    if environment_count > osh.u32(3):
        direction = osh.normalize(osh.vec3(1.0, -1.0, -1.0))
        contribution = shadeVolumeEnvironmentScattering(
            header, environment_radiance_3, direction, world_position,
            ray_direction, environment_count,
        )
        single_scattered = single_scattered + contribution
        isotropic_scattered = isotropic_scattered + contribution
    if not multiple_scattering:
        isotropic_scattered = osh.vec3(0.0)
        optical_depth = 0.0
    return shadeFinalizeVolumeScattering(
        header, single_scattered, isotropic_scattered, optical_depth
    )


@osh.function
def shadeVolumePhase(header: VolumeHeader, cosine: osh.f32) -> osh.f32:
    if header.phase_parameters.y < 0.5:
        return 0.0795774715459
    anisotropy = osh.clamp(header.phase_parameters.x, -0.99, 0.99)
    denominator = osh.maximum(
        1.0 + anisotropy * anisotropy
        - 2.0 * anisotropy * osh.clamp(cosine, -1.0, 1.0),
        1.0e-8,
    )
    return (
        (1.0 - anisotropy * anisotropy)
        / (12.5663706144 * denominator * osh.sqrt(denominator))
    )


@osh.function
def shadeVolumeExtinction(
    header: VolumeHeader, sample_value: osh.vec4,
) -> osh.f32:
    reference_alpha = osh.clamp(
        sample_value.a * header.value_parameters.z, 0.0, 0.999999
    )
    return (
        -osh.logarithm(1.0 - reference_alpha)
        / osh.maximum(header.render_parameters.x, 1.0e-5)
    )


@osh.function
def shadeVolumeStepCount(
    entry: osh.f32, exit_distance: osh.f32, reference_step: osh.f32,
) -> osh.u32:
    distance = osh.maximum(exit_distance - entry, 0.0)
    return osh.minimum(
        osh.u32(osh.ceiling(distance / osh.maximum(reference_step, 1.0e-5))),
        osh.u32(4096),
    )


@osh.function
def shadeCompositeVolumeStep(
    input_state: ShadeVolumeMarchState,
    header: VolumeHeader,
    sample_value: osh.vec4,
    scattering_source: osh.vec3,
    step_size: osh.f32,
) -> ShadeVolumeMarchState:
    state = input_state
    reference_step = osh.maximum(header.render_parameters.x, 1.0e-5)
    reference_alpha = osh.clamp(
        sample_value.a * header.value_parameters.z, 0.0, 0.999999
    )
    alpha = 1.0 - osh.power(
        1.0 - reference_alpha, step_size / reference_step
    )
    source = (
        sample_value.rgb * header.value_parameters.w + scattering_source
    )
    state.integrated = (
        state.integrated + state.transmittance * alpha * source
    )
    state.transmittance = state.transmittance * (1.0 - alpha)
    return state


@osh.function
def shadeApplyVolumeMarch(
    input_path: WavePathState, state: ShadeVolumeMarchState,
) -> WavePathState:
    path = input_path
    path.radiance = osh.vec4(
        path.radiance.rgb + path.throughput.rgb * state.integrated,
        path.radiance.w,
    )
    path.throughput = osh.vec4(
        path.throughput.rgb * state.transmittance, path.throughput.w
    )
    return path


@osh.function
def shadeIntegrateVolumeUntil(
    volume_index: osh.u32,
    origin: osh.vec3,
    direction: osh.vec3,
    maximum_distance: osh.f32,
    empty_space_skipping: osh.boolean,
    scattering_enabled: osh.boolean,
    multiple_scattering: osh.boolean,
    point_light_count: osh.u32,
    area_light_count: osh.u32,
    environment_sample_count: osh.u32,
    environment_radiance_0: osh.vec3,
    environment_radiance_1: osh.vec3,
    environment_radiance_2: osh.vec3,
    environment_radiance_3: osh.vec3,
) -> ShadeVolumeIntegrationResult:
    header = volume_headers[volume_index]
    interval = shadeVolumeInterval(header, origin, direction)
    entry = osh.maximum(interval.x, 0.0)
    exit_distance = osh.minimum(interval.y, maximum_distance)
    state = ShadeVolumeMarchState(osh.vec3(0.0), 1.0)
    if exit_distance <= entry:
        return ShadeVolumeIntegrationResult(state, entry)
    reference_step = osh.maximum(header.render_parameters.x, 1.0e-5)
    steps = shadeVolumeStepCount(entry, exit_distance, reference_step)
    step_size = (
        (exit_distance - entry) / osh.f32(osh.maximum(steps, osh.u32(1)))
    )
    step_index = osh.u32(0)
    voxel_ray = shadeVolumeVoxelRay(header, origin, direction)
    occupied_until = -1.0
    scattering_source = osh.vec3(0.0)
    if scattering_enabled:
        scattering_midpoint = 0.5 * (entry + exit_distance)
        scattering_position = origin + direction * scattering_midpoint
        scattering_sample = shadeVolumeTransferSample(
            header, shadeVolumeScalar(
                volume_index, header, scattering_position
            )
        )
        optical_depth = shadeVolumeExtinction(header, scattering_sample) * (
            exit_distance - entry
        )
        scattering_source = shadeVolumeScatteringSource(
            header, scattering_position, direction, optical_depth,
            point_light_count, area_light_count, environment_sample_count,
            environment_radiance_0, environment_radiance_1,
            environment_radiance_2, environment_radiance_3,
            multiple_scattering,
        )
    while step_index < steps:
        distance = entry + (osh.f32(step_index) + 0.5) * step_size
        if empty_space_skipping and distance + 1.0e-7 >= occupied_until:
            brick_step = shadeVolumeBrickStepAtVoxel(
                header,
                voxel_ray.origin + voxel_ray.direction * distance,
                voxel_ray.direction, distance,
            )
            if not brick_step.occupied:
                jump = osh.u32(osh.clamp(
                    osh.ceiling((brick_step.exit_distance - distance) / step_size),
                    1.0, osh.f32(steps - step_index),
                ))
                step_index = osh.minimum(step_index + jump, steps)
                continue
            occupied_until = brick_step.exit_distance
        scalar = shadeVolumeScalar(
            volume_index, header, origin + direction * distance
        )
        sample_value = shadeVolumeTransferSample(header, scalar)
        state = shadeCompositeVolumeStep(
            state, header, sample_value, scattering_source, step_size
        )
        if state.transmittance < 1.0e-4:
            break
        step_index = step_index + osh.u32(1)
    return ShadeVolumeIntegrationResult(state, exit_distance)


@osh.function
def shadeIntegrateOverlappingVolumes(
    origin: osh.vec3,
    direction: osh.vec3,
    surface_distance: osh.f32,
    empty_space_skipping: osh.boolean,
    scattering_enabled: osh.boolean,
    multiple_scattering: osh.boolean,
    point_light_count: osh.u32,
    area_light_count: osh.u32,
    environment_sample_count: osh.u32,
    environment_radiance_0: osh.vec3,
    environment_radiance_1: osh.vec3,
    environment_radiance_2: osh.vec3,
    environment_radiance_3: osh.vec3,
) -> ShadeVolumeIntegrationResult:
    entries = osh.local_array(osh.f32, 16)
    exits = osh.local_array(osh.f32, 16)
    scattering_sources = osh.local_array(osh.vec3, 16)
    volume_count = osh.minimum(
        osh.u32(volume_headers[osh.u32(0)].render_parameters.z), osh.u32(16)
    )
    bounds = ShadeVolumeUnionBounds(
        surface_distance, 0.0, 1.0e30, False
    )
    for volume_index in range(16):
        entries[volume_index] = 1.0
        exits[volume_index] = 0.0
        scattering_sources[volume_index] = osh.vec3(0.0)
        if osh.u32(volume_index) >= volume_count:
            continue
        header = volume_headers[osh.u32(volume_index)]
        interval = shadeVolumeInterval(header, origin, direction)
        entry = osh.maximum(interval.x, 0.0)
        exit_distance = osh.minimum(interval.y, surface_distance)
        if exit_distance <= entry:
            continue
        entries[volume_index] = entry
        exits[volume_index] = exit_distance
        bounds = shadeExtendVolumeUnion(
            bounds, interval, header, surface_distance
        )
        if scattering_enabled:
            midpoint = 0.5 * (entry + exit_distance)
            position = origin + direction * midpoint
            sample_value = shadeVolumeTransferSample(
                header, shadeVolumeScalar(
                    osh.u32(volume_index), header, position
                )
            )
            optical_depth = shadeVolumeExtinction(
                header, sample_value
            ) * (exit_distance - entry)
            scattering_sources[volume_index] = shadeVolumeScatteringSource(
                header, position, direction, optical_depth,
                point_light_count, area_light_count,
                environment_sample_count, environment_radiance_0,
                environment_radiance_1, environment_radiance_2,
                environment_radiance_3, multiple_scattering,
            )
    state = ShadeVolumeMarchState(osh.vec3(0.0), 1.0)
    if not bounds.valid:
        return ShadeVolumeIntegrationResult(state, 0.0)
    steps = shadeVolumeStepCount(
        bounds.entry, bounds.exit_distance, bounds.reference_step
    )
    step_size = (
        (bounds.exit_distance - bounds.entry)
        / osh.f32(osh.maximum(steps, osh.u32(1)))
    )
    step_index = osh.u32(0)
    while step_index < steps:
        distance = bounds.entry + (osh.f32(step_index) + 0.5) * step_size
        world_position = origin + direction * distance
        if empty_space_skipping:
            any_occupied = False
            empty_exit = 1.0e30
            for volume_index in range(16):
                if osh.u32(volume_index) >= volume_count:
                    break
                if (
                    distance < entries[volume_index]
                    or distance >= exits[volume_index]
                ):
                    continue
                header = volume_headers[osh.u32(volume_index)]
                voxel_ray = shadeVolumeVoxelRay(header, origin, direction)
                brick_step = shadeVolumeBrickStepAtVoxel(
                    header,
                    voxel_ray.origin + voxel_ray.direction * distance,
                    voxel_ray.direction, distance,
                )
                if brick_step.occupied:
                    any_occupied = True
                    break
                empty_exit = osh.minimum(
                    empty_exit, brick_step.exit_distance
                )
            if not any_occupied:
                jump = osh.u32(osh.clamp(
                    osh.ceiling((empty_exit - distance) / step_size),
                    1.0, osh.f32(steps - step_index),
                ))
                step_index = osh.minimum(step_index + jump, steps)
                continue
        medium = ShadeOverlappingMedium(0.0, osh.vec3(0.0))
        for volume_index in range(16):
            if osh.u32(volume_index) >= volume_count:
                break
            if (
                distance < entries[volume_index]
                or distance >= exits[volume_index]
            ):
                continue
            header = volume_headers[osh.u32(volume_index)]
            sample_value = shadeVolumeTransferSample(
                header, shadeVolumeScalar(
                    osh.u32(volume_index), header, world_position
                )
            )
            medium = shadeAccumulateOverlappingMedium(
                medium, header, sample_value,
                scattering_sources[volume_index],
            )
        state = shadeCompositeOverlappingStep(state, medium, step_size)
        if state.transmittance < 1.0e-4:
            break
        step_index = step_index + osh.u32(1)
    return ShadeVolumeIntegrationResult(state, bounds.exit_distance)


@osh.function
def shadeIntegrateVolumesBeforeSurfaceConfigured(
    input_path: WavePathState,
    origin: osh.vec3,
    direction: osh.vec3,
    surface_distance: osh.f32,
    overlapping_enabled: osh.boolean,
    empty_space_skipping: osh.boolean,
    scattering_enabled: osh.boolean,
    multiple_scattering: osh.boolean,
    point_light_count: osh.u32,
    area_light_count: osh.u32,
    environment_sample_count: osh.u32,
    environment_radiance_0: osh.vec3,
    environment_radiance_1: osh.vec3,
    environment_radiance_2: osh.vec3,
    environment_radiance_3: osh.vec3,
) -> WavePathState:
    path = input_path
    if volume_headers[osh.u32(0)].dimensions_offset.x == osh.u32(0):
        return path
    if (
        overlapping_enabled
        and osh.u32(volume_headers[osh.u32(0)].render_parameters.z)
        > osh.u32(1)
    ):
        integrated = shadeIntegrateOverlappingVolumes(
            origin, direction, surface_distance, empty_space_skipping,
            scattering_enabled, multiple_scattering, point_light_count,
            area_light_count, environment_sample_count,
            environment_radiance_0, environment_radiance_1,
            environment_radiance_2, environment_radiance_3,
        )
        return shadeApplyVolumeMarch(path, integrated.state)
    traversal_origin = origin
    traveled = 0.0
    for traversal in range(32):
        remaining = surface_distance - traveled
        if remaining <= 0.001:
            break
        volume_query = osh.ray_query()
        volume_query.initialize(
            scene_tlas, osh.u32(1), osh.u32(0x02), traversal_origin, 0.001,
            direction, remaining,
        )
        while volume_query.proceed():
            pass
        if volume_query.intersection_type(True) != osh.u32(1):
            break
        primitive = (
            volume_query.primitive_index(True)
            + volume_query.instance_custom_index(True)
        )
        volume_index = triangle_volumes[primitive]
        interval = shadeVolumeInterval(
            volume_headers[volume_index], traversal_origin, direction
        )
        segment_end = osh.minimum(interval.y, remaining)
        integrated = shadeIntegrateVolumeUntil(
            volume_index, traversal_origin, direction, segment_end,
            empty_space_skipping, scattering_enabled, multiple_scattering,
            point_light_count, area_light_count, environment_sample_count,
            environment_radiance_0, environment_radiance_1,
            environment_radiance_2, environment_radiance_3,
        )
        path = shadeApplyVolumeMarch(path, integrated.state)
        maximum_throughput = osh.maximum(
            path.throughput.r,
            osh.maximum(path.throughput.g, path.throughput.b),
        )
        if segment_end >= remaining - 0.001 or maximum_throughput < 1.0e-4:
            break
        advance = interval.y + 0.002
        traversal_origin = traversal_origin + direction * advance
        traveled = traveled + advance
    return path


@osh.function
def shadeIntegrateVolumesBeforeSurface(
    input_path: WavePathState,
    origin: osh.vec3,
    direction: osh.vec3,
    surface_distance: osh.f32,
) -> WavePathState:
    return shadeIntegrateVolumesBeforeSurfaceConfigured(
        input_path, origin, direction, surface_distance,
        False, False, False, False,
        osh.u32(0), osh.u32(0), osh.u32(0),
        osh.vec3(0.0), osh.vec3(0.0), osh.vec3(0.0), osh.vec3(0.0),
    )


@osh.function
def shadeVolumeShadowTransmittanceConfigured(
    origin: osh.vec3, direction: osh.vec3, maximum_distance: osh.f32,
    overlapping_enabled: osh.boolean,
    empty_space_skipping: osh.boolean,
    scattering_enabled: osh.boolean,
    multiple_scattering: osh.boolean,
    point_light_count: osh.u32,
    area_light_count: osh.u32,
    environment_sample_count: osh.u32,
    environment_radiance_0: osh.vec3,
    environment_radiance_1: osh.vec3,
    environment_radiance_2: osh.vec3,
    environment_radiance_3: osh.vec3,
) -> osh.f32:
    shadow_path = WavePathState(
        osh.vec4(1.0), osh.vec4(0.0), osh.uvec4(0)
    )
    shadow_path = shadeIntegrateVolumesBeforeSurfaceConfigured(
        shadow_path, origin, direction, maximum_distance,
        overlapping_enabled, empty_space_skipping, scattering_enabled,
        multiple_scattering, point_light_count, area_light_count,
        environment_sample_count, environment_radiance_0,
        environment_radiance_1, environment_radiance_2,
        environment_radiance_3,
    )
    return osh.clamp(shadow_path.throughput.r, 0.0, 1.0)


@osh.function
def shadeVolumeShadowTransmittance(
    origin: osh.vec3, direction: osh.vec3, maximum_distance: osh.f32,
) -> osh.f32:
    return shadeVolumeShadowTransmittanceConfigured(
        origin, direction, maximum_distance,
        False, False, False, False,
        osh.u32(0), osh.u32(0), osh.u32(0),
        osh.vec3(0.0), osh.vec3(0.0), osh.vec3(0.0), osh.vec3(0.0),
    )


@osh.function
def shadePrepareEnvironmentLight(
    hit: osh.vec3,
    normal: osh.vec3,
    initial_random_state: osh.u32,
    sample_count: osh.u32,
    technique_probability: osh.f32,
) -> ShadeEnvironmentSample:
    random_state = rayQueryRandomState(initial_random_state)
    random_u = rayQueryRandomValue(random_state)
    random_state = rayQueryRandomState(random_state)
    random_v = rayQueryRandomValue(random_state)
    direction = waveCosineHemisphere(normal, random_u, random_v)
    cosine = osh.maximum(osh.dot(normal, direction), 0.0)
    pdf = cosine / 3.14159265359
    effective_pdf = pdf * osh.maximum(technique_probability, 0.000001)
    return ShadeEnvironmentSample(
        direction, hit + normal * 0.002, cosine, effective_pdf,
        osh.f32(sample_count), random_state, cosine > 0.0,
    )


@osh.function
def shadeEnvironmentVisible(sample: ShadeEnvironmentSample) -> osh.boolean:
    if not sample.valid:
        return False
    shadow = osh.ray_query()
    shadow.initialize(
        scene_tlas, osh.u32(5), osh.u32(0x01), sample.shadow_origin, 0.001,
        sample.direction, 1.0e30,
    )
    while shadow.proceed():
        pass
    return shadow.intersection_type(True) == osh.u32(0)


@osh.function
def shadeEnvironmentContribution(
    sample: ShadeEnvironmentSample,
    environment_radiance: osh.vec3,
    material: MaterialData,
    normal: osh.vec3,
    incoming: osh.vec3,
    volume_transmittance: osh.vec3,
) -> osh.vec3:
    pdf = sample.cosine / 3.14159265359
    mis = shadePowerHeuristic(
        sample.effective_pdf * sample.sample_count, pdf
    )
    material_weight = osh.mix(
        material.texture_parameters.w, 1.0, material.emission_metallic.a
    )
    return (
        shadeEvaluatePbr(material, normal, -incoming, sample.direction)
        * environment_radiance * sample.cosine * mis * volume_transmittance
        / osh.maximum(sample.effective_pdf, 0.000001) * material_weight
    )


@osh.function
def shadeSelectUnifiedSecondaryDomain(
    area_enabled: osh.boolean,
    environment_enabled: osh.boolean,
    area_light_weight: osh.f32,
    secondary_area_samples: osh.u32,
    environment_samples: osh.u32,
    initial_random_state: osh.u32,
) -> ShadeUnifiedDomainSelection:
    random_state = initial_random_state
    if not area_enabled and not environment_enabled:
        return ShadeUnifiedDomainSelection(random_state, 0.0, False, False)
    area_weight = 0.0
    if area_enabled:
        area_weight = (
            osh.sqrt(osh.maximum(area_light_weight, 0.000001))
            * osh.f32(osh.maximum(secondary_area_samples, osh.u32(1)))
        )
    environment_weight = 0.0
    if environment_enabled:
        environment_weight = osh.f32(
            osh.minimum(environment_samples, osh.u32(4))
        )
    area_probability = area_weight / osh.maximum(
        area_weight + environment_weight, 0.000001
    )
    random_state = rayQueryRandomState(random_state)
    area_selected = rayQueryRandomValue(random_state) < area_probability
    return ShadeUnifiedDomainSelection(
        random_state, area_probability, area_selected, True
    )


@osh.function
def shadeEncodeEnvironmentDirection(direction_input: osh.vec3) -> osh.vec2:
    direction = direction_input / (
        osh.absolute(direction_input.x) + osh.absolute(direction_input.y)
        + osh.absolute(direction_input.z)
    )
    encoded = direction.xy
    if direction.z < 0.0:
        encoded = (osh.vec2(1.0) - osh.absolute(encoded.yx)) * osh.sign(encoded.xy)
    return encoded * 0.5 + osh.vec2(0.5)


@osh.function
def shadeDecodeEnvironmentDirection(encoded: osh.vec2) -> osh.vec3:
    value = encoded * 2.0 - osh.vec2(1.0)
    direction = osh.vec3(
        value, 1.0 - osh.absolute(value.x) - osh.absolute(value.y)
    )
    if direction.z < 0.0:
        direction.xy = (
            (osh.vec2(1.0) - osh.absolute(direction.yx)) * osh.sign(direction.xy)
        )
    return osh.normalize(direction)


@osh.compute(workgroup_size=(1, 1, 1))
def shade_pbr_probe(
    probe_materials: osh.storage_buffer(MaterialData, access="read", binding=0),
    probe_output: osh.storage_buffer(osh.vec4, binding=1),
    probe_stacks: osh.storage_buffer(ShadeMediumStack, binding=2),
):
    sampled = shadeSamplePbr(
        probe_materials[osh.u32(0)], osh.vec3(0.0, 1.0, 0.0),
        osh.vec3(0.0, -1.0, 0.0), osh.u32(1),
    )
    evaluated = evaluateMaterial(
        probe_materials[osh.u32(0)], osh.vec3(0.0, 1.0, 0.0),
        osh.vec2(0.0), osh.vec3(0.0, -1.0, 0.0), True,
        0.5, 0.5, 0.0, 1.0, 1.5,
    )
    applied = shadeApplyMaterialEvaluation(
        probe_materials[osh.u32(0)], evaluated
    )
    stack = shadeSetMediumIor(
        probe_stacks[osh.u32(0)], osh.u32(15), 1.5
    )
    probe_stacks[osh.u32(0)] = stack
    medium = shadeMediumIor(stack, osh.u32(15))
    probe_output[osh.u32(0)] = osh.vec4(
        sampled.weight + applied.base_roughness.rgb * 0.0,
        sampled.pdf + medium * 0.0
    )


@osh.compute(workgroup_size=(1, 1, 1))
def shade_texture_probe(
    texture_words: osh.storage_buffer(osh.u32, access="read", binding=0),
    texture_bindings: osh.storage_buffer(
        TextureBindingData, access="read", binding=1
    ),
    texture_probe_output: osh.storage_buffer(osh.vec4, binding=2),
    materials: osh.storage_buffer(MaterialData, access="read", binding=3),
    vertices: osh.storage_buffer(osh.vec4, access="read", binding=4),
    attributes: osh.storage_buffer(
        VertexAttributeData, access="read", binding=5
    ),
):
    sampled = shadeSampleMaterialTexture(
        -1.0, osh.vec2(0.5), osh.vec2(0.5), False, 0.0, 0.0
    )
    tangent = shadeTriangleTangent(
        osh.vec3(0.0), osh.vec3(1.0, 0.0, 0.0),
        osh.vec3(0.0, 1.0, 0.0), osh.vec2(0.0),
        osh.vec2(1.0, 0.0), osh.vec2(0.0, 1.0), osh.vec3(0.0, 0.0, 1.0),
    )
    material = MaterialData(
        osh.vec4(1.0), osh.vec4(0.0), osh.vec4(1.0, 1.0, 1.0, 0.0),
        osh.vec4(1.5, 0.0, 0.0, 0.0), osh.vec4(-1.0),
        osh.vec4(1.0, -1.0, 0.0, -1.0),
        osh.vec4(0.0), osh.vec4(0.0), osh.vec4(0.0), osh.vec4(0.0),
        osh.vec4(-1.0, 1.0, 0.5, 0.0), osh.vec4(0.0),
    )
    material = shadeApplyMaterialTextures(
        material, osh.vec2(0.5), osh.vec2(0.5), 0.0, 0.0
    )
    normal = shadeApplyNormalTexture(
        material, osh.vec2(0.5), osh.vec2(0.5), 0.0, 0.0,
        osh.vec3(0.0, 0.0, 1.0), tangent,
    )
    surface = shadeResolveSurface(
        osh.u32(0), osh.vec2(0.25), osh.vec3(0.0, 0.0, 1.0),
        osh.vec3(0.0, 0.0, -1.0), 0.0,
    )
    texture_probe_output[osh.u32(0)] = (
        sampled + osh.vec4(normal + surface.normal, material.base_roughness.a)
        * 0.0
    )


@osh.compute(workgroup_size=(1, 1, 1))
def shade_native_texture_probe(
    texture_words: osh.storage_buffer(osh.u32, access="read", binding=0),
    texture_bindings: osh.storage_buffer(
        TextureBindingData, access="read", binding=1
    ),
    native_textures: osh.sampled_texture_2d_array(128, binding=2),
    texture_probe_output: osh.storage_buffer(osh.vec4, binding=3),
):
    texture_probe_output[osh.u32(0)] = shadeSampleNativeMaterialTexture(
        0.0, osh.vec2(0.5), osh.vec2(0.5), True, 0.01, 0.01
    )


@osh.compute(workgroup_size=(1, 1, 1))
def shade_profile_probe(
    work_counters: osh.storage_buffer(osh.u32, binding=14),
):
    shadeProfileWork(osh.u32(0), osh.u32(1), osh.u32(3))
    shadeProfileWork(osh.u32(1), osh.u32(2), osh.u32(4))
    shadeProfileWork(osh.u32(3), osh.u32(5), osh.u32(5))


@osh.compute(workgroup_size=(1, 1, 1), capabilities=("subgroup_ballot",))
def shade_control_probe(
    hit_queue: osh.storage_record(HitQueue, access="read", binding=0),
    input_queue: osh.storage_record(RayQueue, access="read", binding=1),
    paths: osh.storage_buffer(WavePathState, binding=2),
    vertices: osh.storage_buffer(osh.vec4, access="read", binding=4),
    output_queue: osh.storage_record(RayQueue, binding=6),
    stacks: osh.storage_buffer(ShadeMediumStack, binding=7),
    scene_tlas: osh.acceleration_structure(binding=8),
    point_lights: osh.storage_buffer(PointLightData, access="read", binding=9),
    area_lights: osh.storage_buffer(AreaLightData, access="read", binding=10),
    push: osh.push_constants(ShadeConstants),
):
    loaded = shadeLoadHit(osh.global_invocation_id.x)
    if not loaded.valid:
        return
    cone = shadeRayCone(loaded.ray, loaded.hit.position_t.w)
    loaded.ray = shadeSetRayCone(loaded.ray, cone.x, cone.y)
    if loaded.hit.primitive_index == osh.u32(0xFFFFFFFF):
        miss = shadeResolveEnvironmentMiss(
            paths[loaded.hit.path_index], osh.vec3(0.01),
            push.environment_samples, 0.5,
        )
        paths[loaded.hit.path_index] = miss.path
        return
    paths[loaded.hit.path_index] = shadeAdvanceBounceOrTerminate(
        paths[loaded.hit.path_index], push.max_bounces
    )
    material = MaterialData(
        osh.vec4(1.0), osh.vec4(0.0),
        osh.vec4(1.0, 1.0, 1.0, 1.0),
        osh.vec4(1.5, 0.0, 0.0, 0.0), osh.vec4(-1.0), osh.vec4(0.0),
        osh.vec4(0.0), osh.vec4(0.0), osh.vec4(0.0), osh.vec4(0.0),
        osh.vec4(-1.0, 1.0, 0.5, 0.0), osh.vec4(0.0),
    )
    transmitted = shadeTransmitPath(
        paths[loaded.hit.path_index], stacks[loaded.hit.path_index], material,
        osh.vec3(0.0, 1.0, 0.0), loaded.ray.direction_tmax.xyz, True,
        loaded.hit.position_t.w,
    )
    paths[loaded.hit.path_index] = transmitted.path
    stacks[loaded.hit.path_index] = transmitted.stack
    roulette = shadeApplyRussianRoulette(
        transmitted.path, shadePathRng(transmitted.path),
        shadePathBounce(transmitted.path),
        material.attenuation_transmission.a,
        push.russian_roulette_start, push.russian_roulette_min_survival,
    )
    paths[loaded.hit.path_index] = roulette.path
    continuation = shadeBuildContinuation(
        roulette.path, loaded.hit.position_t.xyz, transmitted.direction,
        loaded.hit.path_index, shadePathBounce(roulette.path),
        transmitted.medium_depth, material.attenuation_transmission.a,
        0.5, roulette.random_state, cone.x, cone.y,
        push.unified_secondary_nee != osh.u32(0),
    )
    output_index = shadeReserveOutputIndex(push.subgroup_enqueue != osh.u32(0))
    enqueue = shadeEnqueueContinuation(continuation, output_index)
    paths[loaded.hit.path_index] = enqueue.path
    direct_path = paths[loaded.hit.path_index]
    sample_direct = shadeSelectSecondaryNee(
        osh.clamp(push.secondary_nee_probability, 0.000001, 1.0),
        direct_path.metadata.x, direct_path.metadata.y,
        shadePathBounce(direct_path),
    )
    if sample_direct and push.point_light_count > osh.u32(0):
        light_sample = shadePreparePointLight(
            point_lights[osh.u32(0)], loaded.hit.position_t.xyz,
            osh.vec3(0.0, 1.0, 0.0),
        )
        if shadePointLightVisible(light_sample):
            contribution = shadePointLightContribution(
                light_sample, material, osh.vec3(0.0, 1.0, 0.0),
                loaded.ray.direction_tmax.xyz, osh.vec3(1.0),
            )
            paths[loaded.hit.path_index] = shadeAccumulateDirectContribution(
                paths[loaded.hit.path_index], contribution,
                push.secondary_nee_probability,
            )


@osh.function
def shadeCandidateVolumeEnvironmentRadiance(sample_index: osh.u32) -> osh.vec3:
    direction = osh.vec3(0.577350269)
    if sample_index == osh.u32(1):
        direction = osh.vec3(-0.577350269, -0.577350269, 0.577350269)
    if sample_index == osh.u32(2):
        direction = osh.vec3(-0.577350269, 0.577350269, -0.577350269)
    if sample_index == osh.u32(3):
        direction = osh.vec3(0.577350269, -0.577350269, -0.577350269)
    return shadeEnvironmentRadiance(direction, push.point_light_count)


@osh.function
def shadeCandidateIntegrateVolumes(
    input_path: WavePathState,
    origin: osh.vec3,
    direction: osh.vec3,
    surface_distance: osh.f32,
) -> WavePathState:
    configured_overlapping_volumes = False
    configured_empty_space_skipping = False
    configured_volume_scattering = False
    configured_multiple_scattering = False
    return shadeIntegrateVolumesBeforeSurfaceConfigured(
        input_path, origin, direction, surface_distance,
        configured_overlapping_volumes, configured_empty_space_skipping,
        configured_volume_scattering, configured_multiple_scattering,
        push.point_light_count, push.area_light_count,
        push.environment_samples,
        shadeCandidateVolumeEnvironmentRadiance(osh.u32(0)),
        shadeCandidateVolumeEnvironmentRadiance(osh.u32(1)),
        shadeCandidateVolumeEnvironmentRadiance(osh.u32(2)),
        shadeCandidateVolumeEnvironmentRadiance(osh.u32(3)),
    )


@osh.function
def shadeCandidateVolumeShadowTransmittance(
    origin: osh.vec3,
    direction: osh.vec3,
    maximum_distance: osh.f32,
    bounce: osh.u32,
) -> osh.f32:
    shadeProfileWork(osh.u32(1), osh.u32(1), bounce)
    configured_overlapping_volumes = False
    configured_empty_space_skipping = False
    configured_volume_scattering = False
    configured_multiple_scattering = False
    return shadeVolumeShadowTransmittanceConfigured(
        origin, direction, maximum_distance,
        configured_overlapping_volumes, configured_empty_space_skipping,
        configured_volume_scattering, configured_multiple_scattering,
        push.point_light_count, push.area_light_count,
        push.environment_samples,
        shadeCandidateVolumeEnvironmentRadiance(osh.u32(0)),
        shadeCandidateVolumeEnvironmentRadiance(osh.u32(1)),
        shadeCandidateVolumeEnvironmentRadiance(osh.u32(2)),
        shadeCandidateVolumeEnvironmentRadiance(osh.u32(3)),
    )


@osh.compute(workgroup_size=(64, 1, 1), capabilities=("subgroup_ballot",))
def wavefront_shade_candidate(
    hit_queue: osh.storage_record(HitQueue, access="read", binding=0),
    input_queue: osh.storage_record(RayQueue, access="read", binding=1),
    paths: osh.storage_buffer(WavePathState, binding=2),
    materials: osh.storage_buffer(MaterialData, access="read", binding=3),
    vertices: osh.storage_buffer(osh.vec4, access="read", binding=4),
    attributes: osh.storage_buffer(
        VertexAttributeData, access="read", binding=5
    ),
    output_queue: osh.storage_record(RayQueue, binding=6),
    stacks: osh.storage_buffer(ShadeMediumStack, binding=7),
    scene_tlas: osh.acceleration_structure(binding=8),
    point_lights: osh.storage_buffer(PointLightData, access="read", binding=9),
    area_lights: osh.storage_buffer(AreaLightData, access="read", binding=10),
    texture_words: osh.storage_buffer(osh.u32, access="read", binding=11),
    texture_bindings: osh.storage_buffer(
        TextureBindingData, access="read", binding=12
    ),
    native_textures: osh.sampled_texture_2d_array(128, binding=13),
    work_counters: osh.storage_buffer(osh.u32, binding=14),
    secondary_paths: osh.storage_buffer(SecondaryPathState, binding=15),
    volume_headers: osh.storage_buffer(
        VolumeHeader, access="read", binding=17
    ),
    volume_scalars: osh.storage_buffer(osh.f32, access="read", binding=18),
    volume_transfer: osh.storage_buffer(osh.vec4, access="read", binding=19),
    triangle_volumes: osh.storage_buffer(osh.u32, access="read", binding=20),
    volume_textures: osh.sampled_texture_3d_array(16, binding=21),
    push: osh.push_constants(ShadeConstants),
):
    loaded = shadeLoadHit(osh.global_invocation_id.x)
    if not loaded.valid:
        return
    path_index = loaded.hit.path_index
    path = paths[path_index]
    shadeProfileWork(osh.u32(0), osh.u32(1), shadePathBounce(path))
    incoming = loaded.ray.direction_tmax.xyz
    surface_distance = loaded.ray.direction_tmax.w
    if loaded.hit.primitive_index != osh.u32(0xFFFFFFFF):
        surface_distance = loaded.hit.position_t.w
    path = shadeCandidateIntegrateVolumes(
        path, loaded.ray.origin_tmin.xyz, incoming, surface_distance
    )
    if osh.maximum(
        path.throughput.r,
        osh.maximum(path.throughput.g, path.throughput.b),
    ) < 1.0e-4:
        path.metadata.w = path.metadata.w & ~osh.u32(1)
        paths[path_index] = path
        return
    if loaded.hit.primitive_index == osh.u32(0xFFFFFFFF):
        shadeProfileWork(
            osh.u32(4), osh.u32(1), shadePathBounce(path)
        )
        miss = shadeResolveEnvironmentMiss(
            path, shadeEnvironmentRadiance(incoming, push.point_light_count),
            push.environment_samples,
            shadeUnifiedAreaDomainProbability(
                push.area_light_count, push.environment_samples,
                push.area_light_weight,
            ),
        )
        paths[path_index] = miss.path
        return
    shadeProfileWork(osh.u32(3), osh.u32(1), shadePathBounce(path))
    cone = shadeRayCone(loaded.ray, loaded.hit.position_t.w)
    surface = shadeResolveSurface(
        loaded.hit.primitive_index, loaded.hit.barycentrics,
        loaded.hit.geometric_normal, incoming, cone.x,
    )
    random_state = shadePathRng(path)
    medium_depth = osh.maximum(path.metadata.w >> osh.u32(8), osh.u32(1))
    evaluated = evaluateMaterial(
        surface.material, surface.normal, surface.uv, incoming,
        surface.entering, 0.5, 0.5, osh.f32(shadePathBounce(path)),
        1.0, surface.material.ior_distance.x,
    )
    if evaluated.custom_scattering > 0.5:
        current_ior = shadeMediumIor(
            stacks[path_index], medium_depth - osh.u32(1)
        )
        exterior_ior = osh.maximum(surface.material.ior_distance.x, 1.0001)
        if not surface.entering:
            exterior_ior = 1.0
            if medium_depth > osh.u32(1):
                exterior_ior = shadeMediumIor(
                    stacks[path_index], medium_depth - osh.u32(2)
                )
        random_state = rayQueryRandomState(random_state)
        random_u = rayQueryRandomValue(random_state)
        random_state = rayQueryRandomState(random_state)
        random_v = rayQueryRandomValue(random_state)
        evaluated = evaluateMaterial(
            surface.material, surface.normal, surface.uv, incoming,
            surface.entering, random_u, random_v,
            osh.f32(shadePathBounce(path)), current_ior, exterior_ior,
        )
    surface.material = shadeApplyMaterialEvaluation(
        surface.material, evaluated
    )
    if (
        push.indirect_secondary_capture != osh.u32(0)
        and shadePathBounce(path) == osh.u32(1)
        and secondary_paths[path_index].primary_throughput.w > 0.5
    ):
        secondary = secondary_paths[path_index]
        secondary.position_valid = osh.vec4(loaded.hit.position_t.xyz, 1.0)
        secondary.normal_pdf = osh.vec4(
            surface.normal, secondary.normal_pdf.w
        )
        secondary_paths[path_index] = secondary
    emission = shadeEmissionContribution(
        path, surface.material, surface.vertex_a, surface.vertex_b,
        surface.vertex_c, surface.geometric_normal, incoming,
        loaded.hit.position_t.w, surface.entering,
        push.area_light_weight, push.secondary_area_light_samples,
        push.unified_secondary_nee != osh.u32(0),
        shadeUnifiedAreaDomainProbability(
            push.area_light_count, push.environment_samples,
            push.area_light_weight,
        ),
    )
    path.radiance = osh.vec4(
        path.radiance.rgb + emission.contribution, path.radiance.w
    )
    next_bounce = shadePathBounce(path) + osh.u32(1)
    if next_bounce >= push.max_bounces:
        path = shadeAdvanceBounceOrTerminate(path, push.max_bounces)
        paths[path_index] = path
        return
    transmission = surface.material.attenuation_transmission.a
    next_direction = osh.vec3(0.0)
    bsdf_pdf = 0.0
    cone_spread = cone.y
    sampled_specular = False
    if evaluated.custom_scattering > 0.5:
        event = osh.i32(evaluated.event + 0.5)
        if event == 0:
            path.metadata.w = path.metadata.w & ~osh.u32(1)
            paths[path_index] = path
            return
        next_direction = osh.normalize(evaluated.next_direction)
        bsdf_pdf = osh.maximum(evaluated.pdf, 0.000001)
        path.throughput = osh.vec4(
            path.throughput.rgb * evaluated.weight / bsdf_pdf,
            path.throughput.w,
        )
        sampled_specular = event != 1
        transmission = 0.0
        if event == 3:
            transmission = 1.0
            tracked = shadeTrackCustomTransmission(
                path, stacks[path_index], surface.material, next_direction,
                surface.entering,
            )
            path = tracked.path
            stacks[path_index] = tracked.stack
            medium_depth = tracked.medium_depth
    elif transmission > 0.001:
        transmitted = shadeTransmitPath(
            path, stacks[path_index], surface.material, surface.normal,
            incoming, surface.entering, loaded.hit.position_t.w,
        )
        path = transmitted.path
        stacks[path_index] = transmitted.stack
        next_direction = transmitted.direction
        medium_depth = transmitted.medium_depth
        sampled_specular = True
    else:
        nee_probability = osh.clamp(
            push.secondary_nee_probability, 0.000001, 1.0
        )
        sample_direct = shadeSelectSecondaryNee(
            nee_probability, path.metadata.x, path.metadata.y, next_bounce
        )
        direct = osh.vec3(0.0)
        if sample_direct:
            for light_index in range(64):
                if osh.u32(light_index) >= push.point_light_count:
                    break
                point_sample = shadePreparePointLight(
                    point_lights[osh.u32(light_index)],
                    loaded.hit.position_t.xyz, surface.normal,
                )
                if shadePointLightVisible(point_sample):
                    volume_transmittance = shadeCandidateVolumeShadowTransmittance(
                        point_sample.shadow_origin, point_sample.direction,
                        point_sample.shadow_distance, shadePathBounce(path),
                    )
                    direct = direct + shadePointLightContribution(
                        point_sample, surface.material, surface.normal,
                        incoming, osh.vec3(volume_transmittance),
                    )
            if push.unified_secondary_nee != osh.u32(0):
                domain = shadeSelectUnifiedSecondaryDomain(
                    push.area_light_count > osh.u32(0),
                    push.environment_samples > osh.u32(0),
                    push.area_light_weight, push.secondary_area_light_samples,
                    push.environment_samples, random_state,
                )
                random_state = domain.random_state
                if domain.valid and domain.area_selected:
                    area_sample = shadePrepareAreaLight(
                        loaded.hit.position_t.xyz, surface.normal,
                        random_state, osh.u32(0), osh.u32(1),
                        push.area_light_count, domain.area_probability,
                    )
                    random_state = area_sample.random_state
                    if shadeAreaLightVisible(area_sample):
                        volume_transmittance = shadeCandidateVolumeShadowTransmittance(
                            area_sample.shadow_origin, area_sample.direction,
                            area_sample.shadow_distance, shadePathBounce(path),
                        )
                        direct = direct + shadeAreaLightContribution(
                            area_sample, surface.material, surface.normal,
                            incoming, osh.vec3(volume_transmittance),
                        )
                if domain.valid and not domain.area_selected:
                    environment_sample = shadePrepareEnvironmentLight(
                        loaded.hit.position_t.xyz, surface.normal,
                        random_state, osh.u32(1),
                        1.0 - domain.area_probability,
                    )
                    random_state = environment_sample.random_state
                    if shadeEnvironmentVisible(environment_sample):
                        environment_radiance = shadeEnvironmentRadiance(
                            environment_sample.direction,
                            push.point_light_count,
                        )
                        volume_transmittance = shadeCandidateVolumeShadowTransmittance(
                            environment_sample.shadow_origin,
                            environment_sample.direction, 1.0e30,
                            shadePathBounce(path),
                        )
                        direct = direct + shadeEnvironmentContribution(
                            environment_sample, environment_radiance,
                            surface.material, surface.normal, incoming,
                            osh.vec3(volume_transmittance),
                        )
            else:
                area_sample_count = osh.clamp(
                    push.secondary_area_light_samples, osh.u32(1), osh.u32(16)
                )
                area_direct = osh.vec3(0.0)
                for sample_index in range(16):
                    if osh.u32(sample_index) >= area_sample_count:
                        break
                    area_sample = shadePrepareAreaLight(
                        loaded.hit.position_t.xyz, surface.normal,
                        random_state, osh.u32(sample_index), area_sample_count,
                        push.area_light_count, 1.0,
                    )
                    random_state = area_sample.random_state
                    if shadeAreaLightVisible(area_sample):
                        volume_transmittance = shadeCandidateVolumeShadowTransmittance(
                            area_sample.shadow_origin, area_sample.direction,
                            area_sample.shadow_distance, shadePathBounce(path),
                        )
                        area_direct = area_direct + shadeAreaLightContribution(
                            area_sample, surface.material, surface.normal,
                            incoming, osh.vec3(volume_transmittance),
                        )
                direct = direct + area_direct / osh.f32(area_sample_count)
                environment_count = osh.minimum(
                    push.environment_samples, osh.u32(4)
                )
                environment_direct = osh.vec3(0.0)
                for sample_index in range(4):
                    if osh.u32(sample_index) >= environment_count:
                        break
                    environment_sample = shadePrepareEnvironmentLight(
                        loaded.hit.position_t.xyz, surface.normal,
                        random_state, environment_count, 1.0,
                    )
                    random_state = environment_sample.random_state
                    if shadeEnvironmentVisible(environment_sample):
                        environment_radiance = shadeEnvironmentRadiance(
                            environment_sample.direction,
                            push.point_light_count,
                        )
                        volume_transmittance = shadeCandidateVolumeShadowTransmittance(
                            environment_sample.shadow_origin,
                            environment_sample.direction, 1.0e30,
                            shadePathBounce(path),
                        )
                        environment_direct = (
                            environment_direct + shadeEnvironmentContribution(
                                environment_sample, environment_radiance,
                                surface.material, surface.normal, incoming,
                                osh.vec3(volume_transmittance),
                            )
                        )
                if environment_count > osh.u32(0):
                    direct = direct + environment_direct / osh.f32(
                        environment_count
                    )
            path = shadeAccumulateDirectContribution(
                path, direct, nee_probability
            )
        scattered = shadeScatterOpaquePath(
            path, surface.material, surface.normal, incoming,
            random_state, cone_spread,
        )
        path = scattered.path
        next_direction = scattered.direction
        bsdf_pdf = scattered.pdf
        random_state = scattered.random_state
        cone_spread = scattered.cone_spread
        sampled_specular = scattered.sampled_specular
    if (
        push.indirect_secondary_capture != osh.u32(0)
        and shadePathBounce(path) == osh.u32(0)
    ):
        secondary = secondary_paths[path_index]
        secondary.primary_throughput = osh.vec4(
            path.throughput.rgb, 2.0 if sampled_specular else 1.0,
        )
        secondary.primary_radiance = osh.vec4(
            path.radiance.rgb,
            shadePbrSpecularProbability(surface.material),
        )
        secondary.normal_pdf.w = bsdf_pdf
        secondary.primary_position = osh.vec4(
            loaded.hit.position_t.xyz,
            1.0 + osh.clamp(surface.material.base_roughness.a, 0.0, 1.0),
        )
        secondary_paths[path_index] = secondary
    roulette = shadeApplyRussianRoulette(
        path, random_state, next_bounce, transmission,
        push.russian_roulette_start, push.russian_roulette_min_survival,
    )
    paths[path_index] = roulette.path
    if not roulette.survived:
        return
    continuation = shadeBuildContinuation(
        roulette.path, loaded.hit.position_t.xyz, next_direction, path_index,
        next_bounce, medium_depth, transmission, bsdf_pdf,
        roulette.random_state, cone.x, cone_spread,
        push.unified_secondary_nee != osh.u32(0),
    )
    output_index = shadeReserveOutputIndex(
        push.subgroup_enqueue != osh.u32(0)
    )
    enqueue = shadeEnqueueContinuation(continuation, output_index)
    paths[path_index] = enqueue.path


WAVEFRONT_SHADE_CANDIDATE_HELPERS = (
    waveCosineHemisphere, evaluateMaterial, shadeApplyMaterialEvaluation,
    rayQueryRandomState, rayQueryRandomValue, rayQueryReflect,
    shadeSrgbChannel, shadeWrapTextureCoordinate, shadeWrapTextureIndex,
    shadeDecodeTextureTexel, shadeFetchTextureTexel, shadeTextureMipOffset,
    shadeSampleTextureLevel, shadeSampleMaterialTexture,
    shadeSampleNativeMaterialTexture,
    shadeTriangleUvDensity, shadeMaterialHasTextures,
    shadeTextureBindingUsesUv1, shadeMaterialUsesUv1, shadeTriangleTangent,
    shadeApplyMaterialTextures, shadeApplyNormalTexture, shadeResolveSurface,
    shadePathRng, shadePathBounce, shadePathPreviousPdf,
    shadeProfileWork,
    shadeMediumIor, shadeSetMediumIor, shadeLoadHit, shadeRayCone,
    shadePowerHeuristic, shadeResolveEnvironmentMiss,
    shadeAdvanceBounceOrTerminate, shadeTransmitPath,
    shadeTrackCustomTransmission,
    shadeApplyRussianRoulette, shadeBuildContinuation,
    shadeReserveOutputIndex, shadeEnqueueContinuation, shadeSmoothstep,
    shadePbrFresnel, shadeGgxDistribution, shadeGgxSmithComponent,
    shadePbrSpecularProbability, shadeEvaluatePbrLobes, shadeEvaluatePbr,
    shadePbrPdf,
    shadeSampleGgxHalfVector, shadeSamplePbr, shadeScatterOpaquePath,
    shadePreparePointLight, shadePointLightVisible,
    shadePointLightContribution, shadeSelectAreaLight, shadePrepareAreaLight,
    shadeAreaLightVisible, shadeAreaLightContribution,
    shadeUnifiedAreaDomainProbability, shadeSecondaryNeeHash,
    shadeSelectSecondaryNee, shadeEmissionContribution,
    shadeAnalyticEnvironment, shadeFindEnvironmentLight,
    shadeEnvironmentUv, shadeDecodeEnvironmentRadiance,
    shadeSamplePackedSceneTexture, shadeSampleNativeSceneTexture,
    shadeEnvironmentRadiance,
    shadeAccumulateDirectContribution, shadeSelectUnifiedSecondaryDomain,
    shadePrepareEnvironmentLight, shadeEnvironmentVisible,
    shadeEnvironmentContribution,
    shadeIsVolumePrimitive, shadeVolumeLocalCoordinate, shadeVolumeInterval,
    shadeVolumeTransferSample, shadeVolumeScalar,
    shadeVolumeBrickIndexFromVoxel, shadeVolumeBrickOccupiedAtVoxel,
    shadeVolumeVoxelRay, shadeVolumeAxisExitDelta,
    shadeVolumeBrickStepAtVoxel, shadeExtendVolumeUnion,
    shadeAccumulateOverlappingMedium, shadeCompositeOverlappingStep,
    shadeVolumeOpaqueVisibility, shadeApproximateVolumeLightTransmittance,
    shadeVolumePointLightScattering, shadeVolumeAreaLightScattering,
    shadeVolumeEnvironmentScattering, shadeFinalizeVolumeScattering,
    shadeVolumeScatteringSource, shadeVolumePhase, shadeVolumeExtinction,
    shadeVolumeStepCount, shadeCompositeVolumeStep, shadeApplyVolumeMarch,
    shadeIntegrateVolumeUntil, shadeIntegrateOverlappingVolumes,
    shadeIntegrateVolumesBeforeSurfaceConfigured,
    shadeIntegrateVolumesBeforeSurface,
    shadeVolumeShadowTransmittanceConfigured,
    shadeVolumeShadowTransmittance,
    shadeCandidateVolumeEnvironmentRadiance,
    shadeCandidateIntegrateVolumes,
    shadeCandidateVolumeShadowTransmittance,
)


@osh.compute(workgroup_size=(1, 1, 1))
def shade_volume_probe(
    scene_tlas: osh.acceleration_structure(binding=8),
    point_lights: osh.storage_buffer(
        PointLightData, access="read", binding=9
    ),
    area_lights: osh.storage_buffer(
        AreaLightData, access="read", binding=10
    ),
    volume_headers: osh.storage_buffer(
        VolumeHeader, access="read", binding=17
    ),
    volume_scalars: osh.storage_buffer(osh.f32, access="read", binding=18),
    volume_transfer: osh.storage_buffer(osh.vec4, access="read", binding=19),
    triangle_volumes: osh.storage_buffer(osh.u32, access="read", binding=20),
    volume_textures: osh.sampled_texture_3d_array(16, binding=21),
    volume_probe_output: osh.storage_buffer(osh.vec4, binding=22),
):
    primitive = osh.global_invocation_id.x
    if not shadeIsVolumePrimitive(primitive):
        volume_probe_output[primitive] = osh.vec4(0.0)
        return
    volume_index = triangle_volumes[primitive]
    header = volume_headers[volume_index]
    interval = shadeVolumeInterval(
        header, osh.vec3(0.0), osh.vec3(0.0, 0.0, 1.0)
    )
    local = shadeVolumeLocalCoordinate(header, osh.vec3(0.5))
    sample_value = shadeVolumeTransferSample(header, local.x)
    union_bounds = shadeExtendVolumeUnion(
        ShadeVolumeUnionBounds(0.0, 0.0, 1.0e30, False),
        interval, header, interval.y,
    )
    steps = shadeVolumeStepCount(
        interval.x, interval.y, header.render_parameters.x
    )
    extinction = shadeVolumeExtinction(header, sample_value)
    phase = shadeVolumePhase(header, 0.5)
    point_scattering = shadeVolumePointLightScattering(
        header, point_lights[osh.u32(0)], osh.vec3(0.5),
        osh.vec3(0.0, 0.0, 1.0),
    )
    area_scattering = shadeVolumeAreaLightScattering(
        header, area_lights[osh.u32(0)], osh.vec3(0.5),
        osh.vec3(0.0, 0.0, 1.0),
    )
    environment_scattering = shadeVolumeEnvironmentScattering(
        header, osh.vec3(0.1), osh.normalize(osh.vec3(1.0)),
        osh.vec3(0.5), osh.vec3(0.0, 0.0, 1.0), osh.u32(4),
    )
    scattering = shadeFinalizeVolumeScattering(
        header, point_scattering + area_scattering + environment_scattering,
        osh.vec3(0.0), extinction,
    )
    state = shadeCompositeVolumeStep(
        ShadeVolumeMarchState(osh.vec3(0.0), 1.0), header, sample_value,
        osh.vec3(phase) + scattering,
        (interval.y - interval.x) / osh.f32(osh.maximum(steps, osh.u32(1))),
    )
    overlapping = shadeAccumulateOverlappingMedium(
        ShadeOverlappingMedium(0.0, osh.vec3(0.0)),
        header, sample_value, osh.vec3(0.0),
    )
    state = shadeCompositeOverlappingStep(
        state, overlapping, union_bounds.reference_step
    )
    integrated = shadeIntegrateVolumeUntil(
        volume_index, osh.vec3(0.0), osh.vec3(0.0, 0.0, 1.0), interval.y,
        True, True, True, osh.u32(1), osh.u32(1), osh.u32(4),
        osh.vec3(0.1), osh.vec3(0.1), osh.vec3(0.1), osh.vec3(0.1),
    )
    shadow_transmittance = shadeVolumeShadowTransmittance(
        osh.vec3(0.0), osh.vec3(0.0, 0.0, 1.0), 1000.0
    )
    volume_probe_output[primitive] = osh.vec4(
        state.integrated + integrated.state.integrated
        + osh.vec3(extinction * 0.0),
        integrated.state.transmittance * shadow_transmittance
    )


@osh.function
def rayQuerySamplePointLights(
    hit: osh.vec3, normal: osh.vec3, diffuse_albedo: osh.vec3,
) -> osh.vec3:
    direct = osh.vec3(0.0)
    light_count = osh.clamp(osh.i32(image_camera.previous_up.w), 0, 64)
    for light_index in range(64):
        if light_index >= light_count:
            break
        light = point_lights[light_index]
        light_type = osh.i32(light.position_type.w + 0.5)
        if light_type == 3:
            continue
        light_direction = osh.vec3(0.0)
        distance_to_light = 10000.0
        attenuation = 1.0
        if light_type == 1:
            light_direction = -osh.normalize(light.direction_range.xyz)
        else:
            to_light = light.position_type.xyz - hit
            distance_squared = osh.maximum(osh.dot(to_light, to_light), 0.000001)
            distance_to_light = osh.sqrt(distance_squared)
            light_direction = to_light / distance_to_light
            if (
                light.direction_range.w > 0.0
                and distance_to_light > light.direction_range.w
            ):
                continue
            attenuation = 1.0 / distance_squared
            if light_type == 2:
                cone = osh.dot(
                    osh.normalize(light.direction_range.xyz), -light_direction
                )
                spot = reconstructSmoothstep(
                    light.spot_parameters.y, light.spot_parameters.x, cone
                )
                if spot <= 0.0:
                    continue
                attenuation = attenuation * spot
        cosine = osh.maximum(osh.dot(normal, light_direction), 0.0)
        if cosine <= 0.0:
            continue
        shadow = osh.ray_query()
        maximum_distance = osh.maximum(distance_to_light - 0.004, 0.001)
        if light_type == 1:
            maximum_distance = 10000.0
        shadow.initialize(
            scene_tlas, osh.u32(5), osh.u32(0xFF),
            hit + normal * 0.002, 0.001, light_direction, maximum_distance,
        )
        while shadow.proceed():
            pass
        if shadow.intersection_type(True) != osh.u32(0):
            continue
        incident = (
            light.color_intensity.rgb * light.color_intensity.a * attenuation
        )
        direct = direct + diffuse_albedo * incident * (cosine / 3.14159265359)
    return direct


@osh.function
def rayQuerySampleAreaLight(
    hit: osh.vec3, normal: osh.vec3, diffuse_albedo: osh.vec3,
    initial_random_state: osh.u32, sample_index: osh.i32,
    sample_count: osh.i32,
) -> AreaLightSampleResult:
    random_state = initial_random_state
    light_count = osh.maximum(osh.i32(image_camera.lighting.x), 0)
    if light_count == 0:
        return AreaLightSampleResult(osh.vec3(0.0), random_state)
    random_state = rayQueryRandomState(random_state)
    selection = (
        osh.f32(sample_index) + rayQueryRandomValue(random_state)
    ) / osh.f32(sample_count)
    lower = 0
    upper = light_count - 1
    for search_step in range(32):
        if lower >= upper:
            break
        middle = lower + (upper - lower) / 2
        if selection <= area_lights[middle].distribution.x:
            upper = middle
        else:
            lower = middle + 1
    light = area_lights[lower]
    random_state = rayQueryRandomState(random_state)
    root_u = osh.sqrt(rayQueryRandomValue(random_state))
    random_state = rayQueryRandomState(random_state)
    value_v = rayQueryRandomValue(random_state)
    light_position = (
        (1.0 - root_u) * light.a.xyz
        + root_u * (1.0 - value_v) * light.b.xyz
        + root_u * value_v * light.c.xyz
    )
    to_light = light_position - hit
    distance_squared = osh.dot(to_light, to_light)
    distance_to_light = osh.sqrt(distance_squared)
    light_direction = to_light / osh.maximum(distance_to_light, 0.000001)
    surface_cosine = osh.maximum(osh.dot(normal, light_direction), 0.0)
    light_normal = osh.normalize(osh.cross(
        light.b.xyz - light.a.xyz, light.c.xyz - light.a.xyz
    ))
    raw_light_cosine = osh.dot(light_normal, -light_direction)
    light_cosine = osh.maximum(raw_light_cosine, 0.0)
    if light.distribution.z > 0.5:
        light_cosine = osh.absolute(raw_light_cosine)
    if surface_cosine <= 0.0 or light_cosine <= 0.000001:
        return AreaLightSampleResult(osh.vec3(0.0), random_state)
    shadow = osh.ray_query()
    shadow.initialize(
        scene_tlas, osh.u32(5), osh.u32(0xFF),
        hit + normal * 0.002, 0.001, light_direction,
        osh.maximum(distance_to_light - 0.004, 0.001),
    )
    while shadow.proceed():
        pass
    if shadow.intersection_type(True) != osh.u32(0):
        return AreaLightSampleResult(osh.vec3(0.0), random_state)
    light_pdf = light.distribution.y * distance_squared / osh.maximum(
        light_cosine * light.emission_area.a, 0.000001
    )
    bsdf_pdf = surface_cosine / 3.14159265359
    mis_weight = rayQueryPowerHeuristic(
        light_pdf * osh.f32(sample_count), bsdf_pdf
    )
    result = (
        diffuse_albedo * light.emission_area.rgb
        * (surface_cosine / 3.14159265359) * mis_weight
        / osh.maximum(light_pdf, 0.000001)
    )
    return AreaLightSampleResult(result, random_state)


@osh.function
def rayQueryTracePath(
    origin_value: osh.vec3, direction_value: osh.vec3,
    initial_random_state: osh.u32,
) -> osh.vec3:
    origin = origin_value
    direction = direction_value
    random_state = initial_random_state
    radiance = osh.vec3(0.0)
    throughput = osh.vec3(1.0)
    max_bounces = osh.clamp(osh.i32(camera.overlay.z), 1, 16)
    medium_low = osh.vec4(1.0, 1.0, 1.0, 1.0)
    medium_high = osh.vec4(1.0, 1.0, 1.0, 1.0)
    medium_depth = 1
    for bounce in range(16):
        if bounce >= max_bounces:
            break
        query = osh.ray_query()
        query.initialize(
            scene_tlas, osh.u32(1), osh.u32(0xFF), origin, 0.001,
            direction, 10000.0,
        )
        while query.proceed():
            pass
        if query.intersection_type(True) == osh.u32(0):
            radiance = radiance + throughput * rayQueryEnvironment(direction)
            break
        primitive = query.primitive_index(True) + query.instance_custom_index(True)
        distance = query.intersection_t(True)
        hit = origin + direction * distance
        vertex_a = vertices[primitive * osh.u32(3)].xyz
        vertex_b = vertices[primitive * osh.u32(3) + osh.u32(1)].xyz
        vertex_c = vertices[primitive * osh.u32(3) + osh.u32(2)].xyz
        geometric_normal = osh.normalize(
            osh.cross(vertex_b - vertex_a, vertex_c - vertex_a)
        )
        entering = osh.dot(direction, geometric_normal) < 0.0
        barycentric = query.barycentrics(True)
        weights = osh.vec3(
            1.0 - barycentric.x - barycentric.y,
            barycentric.x, barycentric.y,
        )
        shading_normal = osh.normalize(
            attributes[primitive * osh.u32(3)].normal.xyz * weights.x
            + attributes[primitive * osh.u32(3) + osh.u32(1)].normal.xyz
            * weights.y
            + attributes[primitive * osh.u32(3) + osh.u32(2)].normal.xyz
            * weights.z
        )
        if osh.dot(shading_normal, geometric_normal) < 0.0:
            shading_normal = -shading_normal
        normal = shading_normal if entering else -shading_normal
        uv = (
            attributes[primitive * osh.u32(3)].texcoord.xy * weights.x
            + attributes[primitive * osh.u32(3) + osh.u32(1)].texcoord.xy
            * weights.y
            + attributes[primitive * osh.u32(3) + osh.u32(2)].texcoord.xy
            * weights.z
        )
        material = materials[primitive]
        current_ior = medium_low.x
        if medium_depth == 2:
            current_ior = medium_low.y
        if medium_depth == 3:
            current_ior = medium_low.z
        if medium_depth == 4:
            current_ior = medium_low.w
        if medium_depth == 5:
            current_ior = medium_high.x
        if medium_depth == 6:
            current_ior = medium_high.y
        if medium_depth == 7:
            current_ior = medium_high.z
        if medium_depth == 8:
            current_ior = medium_high.w
        exterior_ior = osh.maximum(material.ior_distance.x, 1.0001)
        if not entering:
            exterior_ior = 1.0
            if medium_depth > 1:
                exterior_ior = medium_low.x
                if medium_depth == 3:
                    exterior_ior = medium_low.y
                if medium_depth == 4:
                    exterior_ior = medium_low.z
                if medium_depth == 5:
                    exterior_ior = medium_low.w
                if medium_depth == 6:
                    exterior_ior = medium_high.x
                if medium_depth == 7:
                    exterior_ior = medium_high.y
                if medium_depth == 8:
                    exterior_ior = medium_high.z
        random_state = rayQueryRandomState(random_state)
        random_u = rayQueryRandomValue(random_state)
        random_state = rayQueryRandomState(random_state)
        random_v = rayQueryRandomValue(random_state)
        evaluated = evaluateMaterial(
            material, normal, uv, direction, entering, random_u, random_v,
            osh.f32(bounce), current_ior, exterior_ior,
        )
        base = evaluated.base_color
        if entering or material.ior_distance.w > 0.5:
            radiance = radiance + throughput * evaluated.emission
        if evaluated.custom_scattering > 0.5:
            event = osh.i32(evaluated.event + 0.5)
            if event == 0:
                break
            direction = osh.normalize(evaluated.next_direction)
            origin = hit + direction * 0.002
            throughput = throughput * evaluated.weight / osh.maximum(
                evaluated.pdf, 0.000001
            )
            if event == 3:
                if entering and medium_depth < 8:
                    value = osh.maximum(material.ior_distance.x, 1.0001)
                    if medium_depth == 1:
                        medium_low.y = value
                    if medium_depth == 2:
                        medium_low.z = value
                    if medium_depth == 3:
                        medium_low.w = value
                    if medium_depth == 4:
                        medium_high.x = value
                    if medium_depth == 5:
                        medium_high.y = value
                    if medium_depth == 6:
                        medium_high.z = value
                    if medium_depth == 7:
                        medium_high.w = value
                    medium_depth = medium_depth + 1
                else:
                    if not entering and medium_depth > 1:
                        medium_depth = medium_depth - 1
            continue
        transmission = evaluated.transmission
        if transmission > 0.001:
            target_ior = exterior_ior
            if entering:
                target_ior = osh.maximum(evaluated.ior, 1.0001)
            eta = current_ior / target_ior
            next_direction = rayQueryRefract(direction, normal, eta)
            if osh.dot(next_direction, next_direction) < 0.01:
                next_direction = rayQueryReflect(direction, normal)
            direction = osh.normalize(next_direction)
            origin = hit + direction * 0.002
            throughput = throughput * osh.mix(osh.vec3(1.0), base, 0.2) * transmission
            if entering and medium_depth < 8:
                if medium_depth == 1:
                    medium_low.y = target_ior
                if medium_depth == 2:
                    medium_low.z = target_ior
                if medium_depth == 3:
                    medium_low.w = target_ior
                if medium_depth == 4:
                    medium_high.x = target_ior
                if medium_depth == 5:
                    medium_high.y = target_ior
                if medium_depth == 6:
                    medium_high.z = target_ior
                if medium_depth == 7:
                    medium_high.w = target_ior
                medium_depth = medium_depth + 1
            else:
                if not entering and medium_depth > 1:
                    medium_depth = medium_depth - 1
            continue
        if evaluated.metallic > 0.5 and bounce + 1 < max_bounces:
            direction = osh.normalize(rayQueryReflect(direction, normal))
            origin = hit + direction * 0.002
            throughput = throughput * base
            continue
        light_direction = osh.normalize(osh.vec3(-0.45, 0.8, 0.35))
        diffuse = osh.maximum(osh.dot(normal, light_direction), 0.0)
        radiance = radiance + throughput * base * (0.10 + 0.90 * diffuse)
        break
    return radiance


@osh.function
def rayQueryImageTracePath(
    origin_value: osh.vec3, direction_value: osh.vec3,
    initial_random_state: osh.u32,
) -> RayQueryTraceResult:
    origin = origin_value
    direction = direction_value
    random_state = initial_random_state
    radiance = osh.vec3(0.0)
    throughput = osh.vec3(1.0)
    max_bounces = osh.clamp(osh.i32(image_camera.overlay.z), 1, 16)
    medium_low = osh.vec4(1.0)
    medium_high = osh.vec4(1.0)
    medium_depth = 1
    primary_depth = -1.0
    primary_normal = osh.vec3(0.0)
    primary_id = -1.0
    previous_bsdf_pdf = 0.0
    previous_was_diffuse = False
    for bounce in range(16):
        if bounce >= max_bounces:
            break
        query = osh.ray_query()
        query.initialize(
            scene_tlas, osh.u32(1), osh.u32(0xFF), origin, 0.001,
            direction, 10000.0,
        )
        while query.proceed():
            pass
        if query.intersection_type(True) == osh.u32(0):
            radiance = radiance + throughput * rayQueryEnvironment(direction)
            break
        primitive = query.primitive_index(True) + query.instance_custom_index(True)
        distance = query.intersection_t(True)
        hit = origin + direction * distance
        vertex_a = vertices[primitive * osh.u32(3)].xyz
        vertex_b = vertices[primitive * osh.u32(3) + osh.u32(1)].xyz
        vertex_c = vertices[primitive * osh.u32(3) + osh.u32(2)].xyz
        geometric_normal = osh.normalize(osh.cross(
            vertex_b - vertex_a, vertex_c - vertex_a
        ))
        entering = osh.dot(direction, geometric_normal) < 0.0
        barycentric = query.barycentrics(True)
        weights = osh.vec3(
            1.0 - barycentric.x - barycentric.y,
            barycentric.x, barycentric.y,
        )
        shading_normal = osh.normalize(
            attributes[primitive * osh.u32(3)].normal.xyz * weights.x
            + attributes[primitive * osh.u32(3) + osh.u32(1)].normal.xyz * weights.y
            + attributes[primitive * osh.u32(3) + osh.u32(2)].normal.xyz * weights.z
        )
        if osh.dot(shading_normal, geometric_normal) < 0.0:
            shading_normal = -shading_normal
        normal = shading_normal if entering else -shading_normal
        uv = (
            attributes[primitive * osh.u32(3)].texcoord.xy * weights.x
            + attributes[primitive * osh.u32(3) + osh.u32(1)].texcoord.xy * weights.y
            + attributes[primitive * osh.u32(3) + osh.u32(2)].texcoord.xy * weights.z
        )
        if bounce == 0:
            primary_depth = distance
            primary_normal = normal
        material = materials[primitive]
        emitted_toward_ray = entering or material.ior_distance.w > 0.5
        current_ior = medium_low.x
        if medium_depth == 2: current_ior = medium_low.y
        if medium_depth == 3: current_ior = medium_low.z
        if medium_depth == 4: current_ior = medium_low.w
        if medium_depth == 5: current_ior = medium_high.x
        if medium_depth == 6: current_ior = medium_high.y
        if medium_depth == 7: current_ior = medium_high.z
        if medium_depth == 8: current_ior = medium_high.w
        exterior_ior = osh.maximum(material.ior_distance.x, 1.0001)
        if not entering:
            exterior_ior = 1.0
            if medium_depth > 1:
                exterior_ior = medium_low.x
                if medium_depth == 3: exterior_ior = medium_low.y
                if medium_depth == 4: exterior_ior = medium_low.z
                if medium_depth == 5: exterior_ior = medium_low.w
                if medium_depth == 6: exterior_ior = medium_high.x
                if medium_depth == 7: exterior_ior = medium_high.y
                if medium_depth == 8: exterior_ior = medium_high.z
        random_state = rayQueryRandomState(random_state)
        random_u = rayQueryRandomValue(random_state)
        random_state = rayQueryRandomState(random_state)
        random_v = rayQueryRandomValue(random_state)
        evaluated = evaluateMaterial(
            material, normal, uv, direction, entering, random_u, random_v,
            osh.f32(bounce), current_ior, exterior_ior,
        )
        if bounce == 0:
            primary_event = osh.i32(evaluated.event + 0.5)
            custom_delta = (
                evaluated.custom_scattering > 0.5 and primary_event != 1
            )
            primary_id = 0.0
            if evaluated.metallic > 0.5: primary_id = 1.0
            if custom_delta or evaluated.transmission > 0.001: primary_id = 2.0
            if (
                emitted_toward_ray
                and osh.dot(evaluated.emission, evaluated.emission) > 0.0
            ): primary_id = 3.0
        base = evaluated.base_color
        emission_weight = 1.0
        if (
            emitted_toward_ray and bounce > 0 and previous_was_diffuse
            and osh.dot(evaluated.emission, evaluated.emission) > 0.0
        ):
            area = 0.5 * osh.length(osh.cross(
                vertex_b - vertex_a, vertex_c - vertex_a
            ))
            raw_light_cosine = osh.dot(geometric_normal, -direction)
            light_cosine = osh.maximum(raw_light_cosine, 0.0)
            if material.ior_distance.w > 0.5:
                light_cosine = osh.absolute(raw_light_cosine)
            emission_luminance = osh.dot(
                material.emission_metallic.rgb, osh.vec3(0.2126, 0.7152, 0.0722)
            )
            selection_probability = area * emission_luminance / osh.maximum(
                image_camera.lighting.y, 0.000001
            )
            light_pdf = selection_probability * distance * distance / osh.maximum(
                light_cosine * area, 0.000001
            )
            emission_weight = rayQueryPowerHeuristic(
                previous_bsdf_pdf,
                light_pdf * osh.maximum(image_camera.lighting.z, 1.0),
            )
        if emitted_toward_ray:
            radiance = radiance + throughput * evaluated.emission * emission_weight
        if evaluated.custom_scattering > 0.5:
            event = osh.i32(evaluated.event + 0.5)
            if event == 0: break
            if event == 1:
                diffuse_albedo = evaluated.weight / osh.maximum(evaluated.pdf, 0.000001)
                radiance = radiance + throughput * rayQuerySamplePointLights(
                    hit, normal, diffuse_albedo
                )
                area_count = osh.clamp(osh.i32(image_camera.lighting.z), 1, 16)
                area_direct = osh.vec3(0.0)
                for light_sample in range(16):
                    if light_sample >= area_count: break
                    sampled = rayQuerySampleAreaLight(
                        hit, normal, diffuse_albedo, random_state,
                        light_sample, area_count,
                    )
                    random_state = sampled.random_state
                    area_direct = area_direct + sampled.radiance
                radiance = radiance + throughput * area_direct / osh.f32(area_count)
            previous_bsdf_pdf = evaluated.pdf
            previous_was_diffuse = event == 1
            direction = osh.normalize(evaluated.next_direction)
            origin = hit + direction * 0.002
            throughput = throughput * evaluated.weight / osh.maximum(evaluated.pdf, 0.000001)
            if event == 3:
                if entering and medium_depth < 8:
                    value = osh.maximum(material.ior_distance.x, 1.0001)
                    if medium_depth == 1: medium_low.y = value
                    if medium_depth == 2: medium_low.z = value
                    if medium_depth == 3: medium_low.w = value
                    if medium_depth == 4: medium_high.x = value
                    if medium_depth == 5: medium_high.y = value
                    if medium_depth == 6: medium_high.z = value
                    if medium_depth == 7: medium_high.w = value
                    medium_depth = medium_depth + 1
                else:
                    if not entering and medium_depth > 1:
                        medium_depth = medium_depth - 1
            continue
        transmission = evaluated.transmission
        if transmission > 0.001:
            target_ior = exterior_ior
            if entering: target_ior = osh.maximum(evaluated.ior, 1.0001)
            next_direction = rayQueryRefract(
                direction, normal, current_ior / target_ior
            )
            if osh.dot(next_direction, next_direction) < 0.01:
                next_direction = rayQueryReflect(direction, normal)
            direction = osh.normalize(next_direction)
            origin = hit + direction * 0.002
            throughput = throughput * osh.mix(osh.vec3(1.0), base, 0.2) * transmission
            if entering and medium_depth < 8:
                if medium_depth == 1: medium_low.y = target_ior
                if medium_depth == 2: medium_low.z = target_ior
                if medium_depth == 3: medium_low.w = target_ior
                if medium_depth == 4: medium_high.x = target_ior
                if medium_depth == 5: medium_high.y = target_ior
                if medium_depth == 6: medium_high.z = target_ior
                if medium_depth == 7: medium_high.w = target_ior
                medium_depth = medium_depth + 1
            else:
                if not entering and medium_depth > 1: medium_depth = medium_depth - 1
            continue
        if evaluated.metallic > 0.5 and bounce + 1 < max_bounces:
            direction = osh.normalize(rayQueryReflect(direction, normal))
            origin = hit + direction * 0.002
            throughput = throughput * base
            continue
        light_direction = osh.normalize(osh.vec3(-0.45, 0.8, 0.35))
        diffuse = osh.maximum(osh.dot(normal, light_direction), 0.0)
        radiance = radiance + throughput * base * (0.10 + 0.90 * diffuse)
        break
    return RayQueryTraceResult(
        radiance, primary_depth, primary_normal, primary_id
    )


@osh.function
def rayQueryImagePrimaryRay(screen: osh.vec2) -> PrimaryRayResult:
    projection = osh.i32(image_camera.up.w + 0.5)
    origin = image_camera.origin.xyz
    direction = image_camera.forward.xyz
    if projection == 1:
        origin = origin + screen.x * image_camera.right.xyz + screen.y * image_camera.up.xyz
        direction = osh.normalize(image_camera.forward.xyz)
    else:
        if projection == 2:
            yaw = screen.x * osh.length(image_camera.right.xyz)
            pitch = screen.y * osh.length(image_camera.up.xyz)
            direction = osh.normalize(
                osh.normalize(image_camera.forward.xyz) * osh.cosine(pitch) * osh.cosine(yaw)
                + osh.normalize(image_camera.right.xyz) * osh.cosine(pitch) * osh.sine(yaw)
                + osh.normalize(image_camera.up.xyz) * osh.sine(pitch)
            )
        else:
            direction = osh.normalize(
                image_camera.forward.xyz + screen.x * image_camera.right.xyz
                + screen.y * image_camera.up.xyz
            )
    return PrimaryRayResult(origin, direction)


@osh.compute(workgroup_size=(8, 8, 1))
def ray_query_image(
    scene_tlas: osh.acceleration_structure(binding=0),
    output_image: osh.storage_image("rgba8", access="write", binding=1),
    materials: osh.storage_buffer(MaterialData, access="read", binding=2),
    vertices: osh.storage_buffer(osh.vec4, access="read", binding=3),
    accumulation_a: osh.storage_image("rgba32f", binding=4),
    gbuffer_a: osh.storage_image("rgba32f", binding=5),
    accumulation_b: osh.storage_image("rgba32f", binding=6),
    gbuffer_b: osh.storage_image("rgba32f", binding=7),
    moment_a: osh.storage_image("r32f", binding=8),
    moment_b: osh.storage_image("r32f", binding=9),
    point_lights: osh.storage_buffer(PointLightData, access="read", binding=10),
    area_lights: osh.storage_buffer(AreaLightData, access="read", binding=11),
    attributes: osh.storage_buffer(VertexAttributeData, access="read", binding=14),
    image_camera: osh.push_constants(RayQueryImageCamera),
):
    pixel = osh.uvec2(osh.global_invocation_id.xy)
    size = osh.uvec2(osh.u32(image_camera.origin.w), osh.u32(image_camera.forward.w))
    if pixel.x >= size.x or pixel.y >= size.y: return
    write_a = (osh.i32(image_camera.accumulation.x) & 1) == 0
    read_a = not write_a
    sample_count = osh.clamp(osh.i32(image_camera.overlay.w), 1, 64)
    previous_accumulation = accumulation_b.load(osh.ivec2(pixel))
    previous_second_moment = moment_b.load(osh.ivec2(pixel)).r
    if read_a:
        previous_accumulation = accumulation_a.load(osh.ivec2(pixel))
        previous_second_moment = moment_a.load(osh.ivec2(pixel)).r
    if image_camera.previous_origin.w > 0.5 and image_camera.accumulation.z > 0.5:
        previous_luminance = osh.dot(previous_accumulation.rgb, osh.vec3(0.2126, 0.7152, 0.0722))
        variance = osh.maximum(previous_second_moment - previous_luminance * previous_luminance, 0.0)
        if previous_accumulation.a >= 2.0 and variance < image_camera.previous_forward.w:
            sample_count = osh.minimum(sample_count, osh.maximum(1, osh.i32(image_camera.previous_right.w)))
    color = osh.vec3(0.0)
    current_second_moment = 0.0
    primary_depth = -1.0
    primary_normal = osh.vec3(0.0)
    primary_id = -1.0
    primary_direction = image_camera.forward.xyz
    for sample_index in range(64):
        if sample_index >= sample_count: break
        random_state = (
            pixel.x * osh.u32(1973) + pixel.y * osh.u32(9277)
            + osh.u32(sample_index) * osh.u32(26699)
            + osh.u32(image_camera.accumulation.x) * osh.u32(104729)
            + osh.u32(911)
        )
        jitter = osh.vec2(0.5)
        if sample_count > 1:
            random_state = rayQueryRandomState(random_state)
            jitter_x = rayQueryRandomValue(random_state)
            random_state = rayQueryRandomState(random_state)
            jitter_y = rayQueryRandomValue(random_state)
            jitter = osh.vec2(jitter_x, jitter_y)
        screen = ((osh.vec2(pixel) + jitter) / osh.vec2(size)) * 2.0 - 1.0
        screen.y = -screen.y
        ray = rayQueryImagePrimaryRay(screen)
        traced = rayQueryImageTracePath(ray.origin, ray.direction, random_state)
        color = color + traced.radiance
        luminance = osh.dot(traced.radiance, osh.vec3(0.2126, 0.7152, 0.0722))
        current_second_moment = current_second_moment + luminance * luminance
        if sample_index == 0:
            primary_depth = traced.primary_depth
            primary_normal = traced.primary_normal
            primary_id = traced.primary_id
            primary_direction = ray.direction
    color = color / osh.f32(sample_count)
    current_second_moment = current_second_moment / osh.f32(sample_count)
    history = osh.vec4(0.0)
    history_pixel = osh.ivec2(pixel)
    if image_camera.accumulation.z > 0.5 and primary_depth >= 0.0:
        valid = True
        if image_camera.accumulation.y > 0.5:
            center_screen = osh.vec2(
                (osh.f32(pixel.x) + 0.5) / osh.f32(size.x) * 2.0 - 1.0,
                1.0 - (osh.f32(pixel.y) + 0.5) / osh.f32(size.y) * 2.0,
            )
            center_ray = rayQueryImagePrimaryRay(center_screen)
            world_position = center_ray.origin + primary_direction * primary_depth
            relative = world_position - image_camera.previous_origin.xyz
            previous_z = osh.dot(relative, image_camera.previous_forward.xyz)
            previous_x = osh.dot(relative, osh.normalize(image_camera.previous_right.xyz)) / osh.maximum(previous_z * osh.length(image_camera.previous_right.xyz), 0.000001)
            previous_y = osh.dot(relative, osh.normalize(image_camera.previous_up.xyz)) / osh.maximum(previous_z * osh.length(image_camera.previous_up.xyz), 0.000001)
            previous_uv = osh.vec2(previous_x, -previous_y) * 0.5 + 0.5
            valid = previous_z > 0.0 and not osh.any_value(previous_uv < osh.vec2(0.0)) and not osh.any_value(previous_uv >= osh.vec2(1.0))
            history_pixel = osh.ivec2(previous_uv * osh.vec2(size))
            if valid:
                old_gbuffer = gbuffer_b.load(history_pixel)
                if read_a: old_gbuffer = gbuffer_a.load(history_pixel)
                old_normal = rayQueryDecodeNormal(old_gbuffer.xy)
                expected_depth = osh.length(relative)
                valid = (
                    osh.absolute(old_gbuffer.z - expected_depth) < osh.maximum(0.03 * expected_depth, 0.02)
                    and osh.absolute(old_gbuffer.w - primary_id) < 0.25
                    and osh.dot(old_normal, primary_normal) > 0.85
                )
        if valid:
            history = accumulation_b.load(history_pixel)
            if read_a: history = accumulation_a.load(history_pixel)
            if image_camera.accumulation.y > 0.5 and image_camera.accumulation.z < 1.5:
                neighborhood_min = osh.vec3(1.0e30)
                neighborhood_max = osh.vec3(-1.0e30)
                compatible_neighbors = 0
                for offset_y in range(-1, 2):
                    for offset_x in range(-1, 2):
                        if offset_x == 0 and offset_y == 0: continue
                        neighbor = history_pixel + osh.ivec2(offset_x, offset_y)
                        if osh.any_value(neighbor < osh.ivec2(0)) or osh.any_value(neighbor >= osh.ivec2(size)): continue
                        neighbor_gbuffer = gbuffer_b.load(neighbor)
                        if read_a: neighbor_gbuffer = gbuffer_a.load(neighbor)
                        neighbor_normal = rayQueryDecodeNormal(neighbor_gbuffer.xy)
                        if osh.absolute(neighbor_gbuffer.w - primary_id) >= 0.25 or osh.dot(neighbor_normal, primary_normal) <= 0.85: continue
                        neighbor_color = accumulation_b.load(neighbor).rgb
                        if read_a: neighbor_color = accumulation_a.load(neighbor).rgb
                        neighborhood_min = osh.minimum(neighborhood_min, neighbor_color)
                        neighborhood_max = osh.maximum(neighborhood_max, neighbor_color)
                        compatible_neighbors = compatible_neighbors + 1
                if compatible_neighbors > 0:
                    extent = osh.maximum(neighborhood_max - neighborhood_min, osh.vec3(0.02))
                    history.rgb = osh.clamp(
                        history.rgb, neighborhood_min - extent * 0.1,
                        neighborhood_max + extent * 0.1,
                    )
            history.a = osh.minimum(history.a, image_camera.accumulation.w)
    history_moment = 0.0
    if history.a > 0.0:
        history_moment = moment_b.load(history_pixel).r
        if read_a: history_moment = moment_a.load(history_pixel).r
    new_samples = osh.f32(sample_count)
    total_samples = history.a + new_samples
    combined_moment = (history_moment * history.a + current_second_moment * new_samples) / osh.maximum(total_samples, 1.0)
    color = (history.rgb * history.a + color * new_samples) / osh.maximum(total_samples, 1.0)
    if write_a:
        accumulation_a.store(osh.ivec2(pixel), osh.vec4(color, total_samples))
    else:
        accumulation_b.store(osh.ivec2(pixel), osh.vec4(color, total_samples))
    if image_camera.previous_origin.w > 0.5 or image_camera.lighting.w > 0.5:
        if write_a: moment_a.store(osh.ivec2(pixel), osh.vec4(combined_moment))
        else: moment_b.store(osh.ivec2(pixel), osh.vec4(combined_moment))
    if image_camera.accumulation.y > 0.5:
        encoded_normal = osh.vec2(0.0)
        if primary_depth >= 0.0: encoded_normal = rayQueryEncodeNormal(primary_normal)
        if write_a: gbuffer_a.store(osh.ivec2(pixel), osh.vec4(encoded_normal, primary_depth, primary_id))
        else: gbuffer_b.store(osh.ivec2(pixel), osh.vec4(encoded_normal, primary_depth, primary_id))


@osh.compute(workgroup_size=(8, 8, 1))
def ray_query(
    scene_tlas: osh.acceleration_structure(binding=0),
    pixels: osh.storage_buffer(osh.u32, binding=1),
    materials: osh.storage_buffer(MaterialData, access="read", binding=2),
    vertices: osh.storage_buffer(osh.vec4, access="read", binding=3),
    point_lights: osh.storage_buffer(PointLightData, access="read", binding=10),
    area_lights: osh.storage_buffer(AreaLightData, access="read", binding=11),
    attributes: osh.storage_buffer(
        VertexAttributeData, access="read", binding=14
    ),
    camera: osh.push_constants(RayQueryCamera),
):
    pixel = osh.uvec2(osh.global_invocation_id.xy)
    size = osh.uvec2(osh.u32(camera.origin.w), osh.u32(camera.forward.w))
    if pixel.x >= size.x or pixel.y >= size.y:
        return
    sample_count = osh.clamp(osh.i32(camera.overlay.w), 1, 64)
    color = osh.vec3(0.0)
    for sample_index in range(64):
        if sample_index >= sample_count:
            break
        random_state = (
            pixel.x * osh.u32(1973) + pixel.y * osh.u32(9277)
            + osh.u32(sample_index) * osh.u32(26699) + osh.u32(911)
        )
        jitter = osh.vec2(0.5)
        if sample_count > 1:
            random_state = rayQueryRandomState(random_state)
            jitter_x = rayQueryRandomValue(random_state)
            random_state = rayQueryRandomState(random_state)
            jitter_y = rayQueryRandomValue(random_state)
            jitter = osh.vec2(jitter_x, jitter_y)
        uv = (osh.vec2(pixel) + jitter) / osh.vec2(size)
        screen = uv * 2.0 - 1.0
        screen.y = -screen.y
        ray = rayQueryPrimaryRay(screen)
        color = color + rayQueryTracePath(
            ray.origin, ray.direction, random_state
        )
    color = color / osh.f32(sample_count)
    color = osh.power(color / (osh.vec3(1.0) + color), osh.vec3(1.0 / 2.2))
    pixels[pixel.y * size.x + pixel.x] = osh.pack_unorm4x8(
        osh.vec4(color, 1.0)
    )


@osh.compute(workgroup_size=(8, 8, 1))
def wavefront_generate(
    ray_queue: osh.storage_record(RayQueue, binding=0),
    paths: osh.storage_buffer(WavePathState, binding=1),
    stacks: osh.storage_buffer(osh.f32, binding=2),
    # The Vulkan ABI binds camera data as VK_DESCRIPTOR_TYPE_STORAGE_BUFFER.
    # Keep this a fixed-layout storage record rather than a uniform block.
    camera: osh.storage_record(CameraData, access="read", binding=3),
    push: osh.push_constants(GenerateConstants),
):
    local_pixel = osh.uvec2(osh.global_invocation_id.xy)
    if local_pixel.x >= push.tile_frame.x:
        return
    if local_pixel.y >= push.tile_frame.y:
        return
    pixel = push.image_tile.zw + local_pixel
    if pixel.x >= push.image_tile.x:
        return
    if pixel.y >= push.image_tile.y:
        return
    path_index = local_pixel.y * push.tile_frame.x + local_pixel.x
    if path_index == osh.u32(0):
        ray_queue.count = osh.minimum(
            push.tile_frame.x * push.tile_frame.y, ray_queue.capacity
        )
    if path_index >= ray_queue.capacity:
        return
    pixel_index = pixel.y * push.image_tile.x + pixel.x
    frame_index = osh.u32(camera.camera_origin.w + 0.5)
    rng = waveHash(
        pixel_index ^ waveHash(frame_index)
        ^ waveHash(push.tile_frame.w + osh.u32(1))
    )
    rng = waveHash(rng)
    jitter_x = waveRandomFloat(rng)
    rng = waveHash(rng)
    jitter_y = waveRandomFloat(rng)
    jitter = osh.vec2(jitter_x, jitter_y)
    ndc = ((osh.vec2(pixel) + jitter) / osh.vec2(push.image_tile.xy)) * 2.0 - 1.0
    aspect = osh.f32(push.image_tile.x) / osh.f32(push.image_tile.y)
    projection = osh.i32(camera.camera_up.w + 0.5)
    ray_origin = camera.camera_origin.xyz
    direction = camera.camera_forward.xyz
    if projection == 1:
        ray_origin = (
            ray_origin + ndc.x * aspect * camera.camera_right.xyz
            - ndc.y * camera.camera_up.xyz
        )
        direction = osh.normalize(camera.camera_forward.xyz)
    else:
        if projection == 2:
            yaw = ndc.x * osh.length(camera.camera_right.xyz)
            pitch = -ndc.y * osh.length(camera.camera_up.xyz)
            direction = osh.normalize(
                osh.normalize(camera.camera_forward.xyz)
                * osh.cosine(pitch) * osh.cosine(yaw)
                + osh.normalize(camera.camera_right.xyz)
                * osh.cosine(pitch) * osh.sine(yaw)
                + osh.normalize(camera.camera_up.xyz) * osh.sine(pitch)
            )
        else:
            direction = osh.normalize(
                camera.camera_forward.xyz
                + ndc.x * aspect * camera.camera_right.xyz
                - ndc.y * camera.camera_up.xyz
            )
    ray_queue.rays[path_index].origin_tmin = osh.vec4(ray_origin, 0.001)
    ray_queue.rays[path_index].direction_tmax = osh.vec4(direction, 1.0e30)
    ray_queue.rays[path_index].path_index = path_index
    ray_queue.rays[path_index].padding_a = osh.u32(0)
    ray_queue.rays[path_index].padding_b = osh.u32(0)
    ray_queue.rays[path_index].padding_c = osh.u32(0)
    paths[path_index].throughput = osh.vec4(1.0)
    paths[path_index].radiance = osh.vec4(0.0)
    capture_secondary = (
        push.tile_frame.z & osh.u32(0x80000000)
    ) != osh.u32(0)
    path_flags = osh.u32(257)
    if capture_secondary:
        path_flags = path_flags | osh.u32(8)
    paths[path_index].metadata = osh.uvec4(
        pixel_index,
        (frame_index << osh.u32(8)) | (push.tile_frame.w & osh.u32(255)),
        rng, path_flags,
    )
    paths[path_index].throughput.w = 0.0
    stacks[path_index * osh.u32(16)] = 1.0


@osh.compute(workgroup_size=(64, 1, 1))
def wavefront_intersect(
    scene_tlas: osh.acceleration_structure(binding=0),
    ray_queue: osh.storage_record(RayQueue, access="read", binding=1),
    hit_queue: osh.storage_record(HitQueue, binding=2),
    vertices: osh.storage_buffer(osh.vec4, access="read", binding=3),
):
    ray_index = osh.global_invocation_id.x
    active_count = osh.minimum(ray_queue.count, ray_queue.capacity)
    if ray_index >= active_count:
        return
    if ray_index == osh.u32(0):
        hit_queue.count = osh.minimum(active_count, hit_queue.capacity)
    query = osh.ray_query()
    query.initialize(
        scene_tlas, osh.u32(1), osh.u32(0x01),
        ray_queue.rays[ray_index].origin_tmin.xyz,
        ray_queue.rays[ray_index].origin_tmin.w,
        ray_queue.rays[ray_index].direction_tmax.xyz,
        ray_queue.rays[ray_index].direction_tmax.w,
    )
    while query.proceed():
        pass
    hit_index = ray_index
    if hit_index >= hit_queue.capacity:
        return
    hit_queue.hits[hit_index].path_index = ray_queue.rays[ray_index].path_index
    hit_queue.hits[hit_index].ray_index = ray_index
    if query.intersection_type(True) == osh.u32(1):
        distance = query.intersection_t(True)
        primitive = (
            query.primitive_index(True) + query.instance_custom_index(True)
        )
        vertex_a = vertices[primitive * osh.u32(3) + osh.u32(0)].xyz
        vertex_b = vertices[primitive * osh.u32(3) + osh.u32(1)].xyz
        vertex_c = vertices[primitive * osh.u32(3) + osh.u32(2)].xyz
        hit_queue.hits[hit_index].position_t = osh.vec4(
            ray_queue.rays[ray_index].origin_tmin.xyz
            + distance * ray_queue.rays[ray_index].direction_tmax.xyz,
            distance,
        )
        hit_queue.hits[hit_index].geometric_normal = osh.normalize(
            osh.cross(vertex_b - vertex_a, vertex_c - vertex_a)
        )
        hit_queue.hits[hit_index].primitive_index = primitive
        hit_queue.hits[hit_index].barycentrics = query.barycentrics(True)
    else:
        hit_queue.hits[hit_index].position_t = osh.vec4(0.0, 0.0, 0.0, -1.0)
        hit_queue.hits[hit_index].geometric_normal = osh.vec3(0.0)
        hit_queue.hits[hit_index].primitive_index = osh.u32(0xFFFFFFFF)
        hit_queue.hits[hit_index].barycentrics = osh.vec2(0.0)


@osh.compute(workgroup_size=(64, 1, 1), capabilities=("subgroup_ballot",))
def wavefront_intersect_bucketed(
    scene_tlas: osh.acceleration_structure(binding=0),
    ray_queue: osh.storage_record(BucketRayQueue, access="read", binding=1),
    plain_queue: osh.storage_record(PlainHitQueue, binding=2),
    textured_queue: osh.storage_record(TexturedHitQueue, binding=3),
    vertices: osh.storage_buffer(osh.vec4, access="read", binding=4),
    materials: osh.storage_buffer(MaterialData, access="read", binding=5),
):
    ray_index = osh.global_invocation_id.x
    if ray_index >= osh.minimum(ray_queue.ray_count, ray_queue.ray_capacity):
        return
    query = osh.ray_query()
    query.initialize(
        scene_tlas, osh.u32(1), osh.u32(0x01),
        ray_queue.rays[ray_index].origin_tmin.xyz,
        ray_queue.rays[ray_index].origin_tmin.w,
        ray_queue.rays[ray_index].direction_tmax.xyz,
        ray_queue.rays[ray_index].direction_tmax.w,
    )
    while query.proceed():
        pass
    hit = WaveHit(
        osh.vec4(0.0, 0.0, 0.0, -1.0), osh.vec3(0.0),
        osh.u32(0xFFFFFFFF), osh.vec2(0.0), ray_index,
        ray_queue.rays[ray_index].path_index,
    )
    textured = False
    if query.intersection_type(True) == osh.u32(1):
        distance = query.intersection_t(True)
        primitive = (
            query.primitive_index(True) + query.instance_custom_index(True)
        )
        vertex_a = vertices[primitive * osh.u32(3)].xyz
        vertex_b = vertices[primitive * osh.u32(3) + osh.u32(1)].xyz
        vertex_c = vertices[primitive * osh.u32(3) + osh.u32(2)].xyz
        hit.position_t = osh.vec4(
            ray_queue.rays[ray_index].origin_tmin.xyz
            + distance * ray_queue.rays[ray_index].direction_tmax.xyz,
            distance,
        )
        hit.geometric_normal = osh.normalize(
            osh.cross(vertex_b - vertex_a, vertex_c - vertex_a)
        )
        hit.primitive_index = primitive
        hit.barycentrics = query.barycentrics(True)
        textured = (
            osh.any_value(materials[primitive].texture_indices >= osh.vec4(0.0))
            or materials[primitive].texture_parameters.y >= 0.0
            or materials[primitive].texture_parameters.w >= 0.0
        )
    for bucket in range(2):
        belongs = textured == (bucket == 1)
        lane_mask = osh.subgroup_ballot(belongs)
        amount = osh.subgroup_ballot_bit_count(lane_mask)
        base = osh.u32(0)
        if osh.subgroup_elect() and amount > osh.u32(0):
            if bucket == 1:
                base = osh.atomic_add(textured_queue.textured_count, amount)
            else:
                base = osh.atomic_add(plain_queue.plain_count, amount)
        base = osh.subgroup_broadcast_first(base)
        if not belongs:
            continue
        output_index = base + osh.subgroup_ballot_exclusive_bit_count(lane_mask)
        capacity = (
            textured_queue.textured_capacity
            if bucket == 1 else plain_queue.plain_capacity
        )
        if output_index >= capacity:
            if bucket == 1:
                osh.atomic_add(textured_queue.textured_overflow, osh.u32(1))
            else:
                osh.atomic_add(plain_queue.plain_overflow, osh.u32(1))
            continue
        if bucket == 1:
            textured_queue.textured_hits[output_index] = hit
        else:
            plain_queue.plain_hits[output_index] = hit


@osh.compute(workgroup_size=(1, 1, 1))
def wavefront_indirect_reuse_probe(
    indirect_reservoir_words: osh.storage_buffer(osh.u32, binding=0),
):
    if osh.global_invocation_id.x != osh.u32(0):
        return
    reservoir = IndirectLightReservoir(
        IndirectLightSample(
            osh.vec3(2.25, -1.5, 0.75), 0.375,
            osh.normalize(osh.vec3(0.2, 0.9, -0.3)), 3.25,
            osh.vec3(8.0, 1.5, 0.125),
        ),
        7.5, osh.u32(9), True, osh.u32(0),
    )
    storeIndirectLightReservoir(osh.u32(0), reservoir, osh.vec3(0.0))
    osh.memory_barrier_buffer()
    decoded = loadIndirectLightReservoir(osh.u32(0), osh.vec3(0.0))
    if not decoded.valid:
        indirect_reservoir_words[osh.u32(5)] = osh.u32(0)


GENERATED = {
    ROOT / "ordinarylight/shaders/ray_query.comp": ray_query,
    ROOT / "ordinarylight/shaders/ray_query_image.comp": ray_query_image,
    ROOT / "ordinarylight/shaders/wavefront_indirect_clear.comp": (
        wavefront_indirect_clear
    ),
    ROOT / "ordinarylight/shaders/wavefront_prepare_indirect.comp": (
        wavefront_prepare_indirect
    ),
    ROOT / "ordinarylight/shaders/wavefront_resolve.comp": wavefront_resolve,
    ROOT / "ordinarylight/shaders/wavefront_path_to_hdr.comp": (
        wavefront_path_to_hdr
    ),
    ROOT / "ordinarylight/shaders/wavefront_indirect_candidates.comp": (
        wavefront_indirect_candidates
    ),
    ROOT / "ordinarylight/shaders/wavefront_indirect_debug.comp": (
        wavefront_indirect_debug
    ),
    ROOT / "ordinarylight/shaders/wavefront_reconstruct.comp": (
        wavefront_reconstruct
    ),
    ROOT / "ordinarylight/shaders/wavefront_tone_map.comp": wavefront_tone_map,
    ROOT / "ordinarylight/shaders/wavefront_tone_map_image.comp": (
        wavefront_tone_map_image
    ),
    ROOT / "ordinarylight/shaders/rgba_to_nv12.comp": rgba_to_nv12,
    ROOT / "ordinarylight/shaders/hdr_to_p010.comp": hdr_to_p010,
    ROOT / "ordinarylight/shaders/denoise_atrous.comp": denoise_atrous,
    ROOT / "ordinarylight/shaders/tone_map.comp": tone_map,
    ROOT / "ordinarylight/shaders/wavefront_generate.comp": wavefront_generate,
    ROOT / "ordinarylight/shaders/wavefront_intersect.comp": wavefront_intersect,
    ROOT / "ordinarylight/shaders/wavefront_intersect_bucketed.comp": (
        wavefront_intersect_bucketed
    ),
    ROOT / "ordinarylight/shaders/wavefront_indirect_reuse_probe.comp": (
        wavefront_indirect_reuse_probe
    ),
    ROOT / "ordinarylight/shaders/wavefront_shade_candidate.glsl": (
        wavefront_shade_candidate
    ),
}


HELPERS = {
    ROOT / "ordinarylight/shaders/ray_query.comp": (
        waveFresnelSchlick, waveCosineHemisphere, evaluateMaterial,
        rayQueryEnvironment, rayQueryPrimaryRay,
        rayQueryRandomState, rayQueryRandomValue, rayQueryReflect,
        rayQueryRefract, rayQueryTracePath,
    ),
    ROOT / "ordinarylight/shaders/ray_query_image.comp": (
        waveFresnelSchlick, waveCosineHemisphere, evaluateMaterial,
        rayQueryEnvironment, rayQueryRandomState, rayQueryRandomValue,
        rayQueryReflect, rayQueryRefract, rayQueryEncodeNormal,
        rayQueryDecodeNormal, rayQueryPowerHeuristic,
        reconstructSmoothstep, rayQuerySamplePointLights,
        rayQuerySampleAreaLight, rayQueryImageTracePath,
        rayQueryImagePrimaryRay,
    ),
    ROOT / "ordinarylight/shaders/wavefront_tone_map.comp": (
        acesApproximation,
        linearToSrgb,
    ),
    ROOT / "ordinarylight/shaders/wavefront_tone_map_image.comp": (
        acesApproximation,
        linearToSrgb,
    ),
    ROOT / "ordinarylight/shaders/rgba_to_nv12.comp": (
        nv12ByteValue, nv12Luma, nv12Chroma, nv12Pixel, pack4Bytes,
    ),
    ROOT / "ordinarylight/shaders/hdr_to_p010.comp": (
        acesApproximation, linearToSrgb, p010TenBitValue, p010Luma,
        p010Chroma, p010SourcePixel, p010Color, pack2x16,
    ),
    ROOT / "ordinarylight/shaders/denoise_atrous.comp": (
        decodeAtrousNormal, atrousKernel,
    ),
    ROOT / "ordinarylight/shaders/tone_map.comp": (
        overlayLetterPixel, overlayDigitMask, overlayDigitPixel, fpsOverlay,
    ),
    ROOT / "ordinarylight/shaders/wavefront_generate.comp": (
        waveHash, waveRandomFloat,
    ),
    ROOT / "ordinarylight/shaders/wavefront_path_to_hdr.comp": (
        indirectEncodeNormal, indirectPackRgb9e5,
        emptyIndirectLightReservoir, storeIndirectLightReservoir,
    ),
    ROOT / "ordinarylight/shaders/wavefront_indirect_candidates.comp": (
        indirectDecodeNormal, indirectUnpackRgb9e5,
        emptyIndirectLightReservoir, indirectEncodeNormal,
        indirectPackRgb9e5, storeIndirectLightReservoir,
        loadIndirectLightReservoir, candidateInstrumentedPixel,
        candidateCountEvent, candidateRejectionDebugFlag,
        candidateLoadPreviousReservoir, candidateRandom,
        candidateMergePrevious, candidateDecodePrimaryNormal,
        candidatePrimaryWorldPosition, candidatePreviousWorldPosition,
        candidateSecondaryVisible, candidateReprojectPrevious,
        candidateSpatialCompatibility,
    ),
    ROOT / "ordinarylight/shaders/wavefront_indirect_debug.comp": (
        indirectDecodeNormal, indirectUnpackRgb9e5,
        emptyIndirectLightReservoir, loadIndirectLightReservoir,
        indirectAcceptanceColor, indirectInvalidColor, indirectCorrection,
    ),
    ROOT / "ordinarylight/shaders/wavefront_reconstruct.comp": (
        acesApproximation, linearToSrgb,
        ordinarylight_tint, ordinarylight_emissive,
        ordinarylight_isolation, ordinarylight_outline,
        reconstructSmoothstep, reconstructDecodeNormal,
        reconstructUnpackNormal, reconstructClampSource,
        reconstructLoadHdr, reconstructEffectSlot,
        reconstructWorldPosition, reconstructCurrentPosition,
        reconstructPreviousPosition, reconstructBilinearHdr,
        reconstructLuminance, reconstructReproject,
        reconstructCurrentNormal, reconstructPreviousNormal,
        reconstructFilterDiffuse, reconstructEffect, reconstructMain,
    ),
    ROOT / "ordinarylight/shaders/wavefront_indirect_reuse_probe.comp": (
        indirectEncodeNormal, indirectDecodeNormal, indirectPackRgb9e5,
        indirectUnpackRgb9e5, emptyIndirectLightReservoir,
        storeIndirectLightReservoir, loadIndirectLightReservoir,
    ),
    ROOT / "ordinarylight/shaders/wavefront_shade_candidate.glsl": (
        WAVEFRONT_SHADE_CANDIDATE_HELPERS
    ),
}


def generated_source(shader, helpers=()):
    source = osh.compile(shader, helpers=helpers).source
    if shader is wavefront_shade_candidate:
        volume_specializations = {
            "bool configured_overlapping_volumes = false;":
                "bool configured_overlapping_volumes = "
                "WAVE_OVERLAPPING_VOLUMES != 0;",
            "bool configured_empty_space_skipping = false;":
                "bool configured_empty_space_skipping = "
                "WAVE_VOLUME_EMPTY_SPACE_SKIPPING != 0;",
            "bool configured_volume_scattering = false;":
                "bool configured_volume_scattering = "
                "WAVE_VOLUME_SCATTERING != 0;",
            "bool configured_multiple_scattering = false;":
                "bool configured_multiple_scattering = "
                "WAVE_VOLUME_MULTIPLE_SCATTERING != 0;",
        }
        for baseline, specialized in volume_specializations.items():
            source = source.replace(baseline, specialized)
        volume_defaults = """\
#ifndef WAVE_OVERLAPPING_VOLUMES
#define WAVE_OVERLAPPING_VOLUMES 0
#endif
#ifndef WAVE_VOLUME_EMPTY_SPACE_SKIPPING
#define WAVE_VOLUME_EMPTY_SPACE_SKIPPING 0
#endif
#ifndef WAVE_VOLUME_SCATTERING
#define WAVE_VOLUME_SCATTERING 0
#endif
#ifndef WAVE_VOLUME_MULTIPLE_SCATTERING
#define WAVE_VOLUME_MULTIPLE_SCATTERING 0
#endif
#ifndef WAVE_NATIVE_TEXTURES
#define WAVE_NATIVE_TEXTURES 0
#endif
#ifndef WAVE_WORK_COUNTERS
#define WAVE_WORK_COUNTERS 0
#endif
"""
        source = source.replace(
            "#version 460\n", "#version 460\n" + volume_defaults, 1
        )
        source = source.replace(
            "#extension GL_EXT_nonuniform_qualifier : require",
            "#if WAVE_NATIVE_TEXTURES\n"
            "#extension GL_EXT_nonuniform_qualifier : require\n"
            "#endif",
            1,
        )
        native_declaration = (
            "layout(set = 0, binding = 13) uniform sampler2D "
            "native_textures[128];"
        )
        source = source.replace(
            native_declaration,
            "#if WAVE_NATIVE_TEXTURES\n" + native_declaration + "\n#endif",
            1,
        )
        work_declaration_start = (
            "layout(std430, set = 0, binding = 14) buffer work_counters_Block"
        )
        work_start = source.index(work_declaration_start)
        work_end = source.index("\n};", work_start) + 3
        source = (
            source[:work_start] + "#if WAVE_WORK_COUNTERS\n"
            + source[work_start:work_end] + "\n#endif"
            + source[work_end:]
        )
        for signature, suffix in (
            (
                "vec4 shadeSampleNativeMaterialTexture(",
                "\n#define shadeSampleMaterialTexture "
                "shadeSampleNativeMaterialTexture",
            ),
            ("vec4 shadeSampleNativeSceneTexture(", ""),
        ):
            start = source.index(signature, source.index(signature) + 1)
            end = source.index("\n}\n", start) + 2
            source = (
                source[:start] + "#if WAVE_NATIVE_TEXTURES\n"
                + source[start:end] + suffix + "\n#endif"
                + source[end:]
            )
        profile_signature = "void shadeProfileWork("
        profile_start = source.index(
            profile_signature, source.index(profile_signature) + 1
        )
        profile_end = source.index("\n}\n", profile_start) + 2
        source = (
            source[:profile_start] + "#if WAVE_WORK_COUNTERS\n"
            + source[profile_start:profile_end] + "\n#endif"
            + source[profile_end:]
        )
        for call in (
            "shadeProfileWork(uint(1), uint(1), bounce);",
            "shadeProfileWork(uint(0), uint(1), shadePathBounce(path));",
            "shadeProfileWork(uint(4), uint(1), shadePathBounce(path));",
            "shadeProfileWork(uint(3), uint(1), shadePathBounce(path));",
        ):
            source = source.replace(
                call,
                "#if WAVE_WORK_COUNTERS\n    " + call + "\n#endif",
                1,
            )
        packed_environment_sample = (
            "encoded = shadeSamplePackedSceneTexture(texture_index, uv).rgb;"
        )
        source = source.replace(
            packed_environment_sample,
            "#if WAVE_NATIVE_TEXTURES\n"
            "        encoded = shadeSampleNativeSceneTexture(texture_index, uv).rgb;\n"
            "#else\n        " + packed_environment_sample + "\n#endif",
            1,
        )
    if (
        shader is ray_query or shader is ray_query_image
        or shader is wavefront_shade_candidate
    ):
        signature = "MaterialEvaluation evaluateMaterial("
        source = "\n".join(
            line for line in source.splitlines()
            if not (line.startswith(signature) and line.endswith(");"))
        ) + "\n"
        # Helper prototypes precede definitions in complete linked modules;
        # material replacement must mark the definition, not its prototype.
        start = source.rindex(signature)
        end = source.index("\n}\n", start) + 2
        source = (
            source[:start] + "// WAVE_RENDER_MATERIAL_BEGIN\n"
            + source[start:end] + "\n// WAVE_RENDER_MATERIAL_END"
            + source[end:]
        )
    if shader is wavefront_reconstruct:
        bgra = osh.compile(
            wavefront_reconstruct_bgra, helpers=helpers
        ).source
        rgba_line = next(
            line for line in source.splitlines()
            if "output_images[8]" in line
        )
        bgra_line = next(
            line for line in bgra.splitlines()
            if "output_images[8]" in line
        )
        source = source.replace(
            rgba_line,
            "#if WAVE_BGRA_OUTPUT\n" + bgra_line
            + "\n#else\n" + rgba_line + "\n#endif",
            1,
        )
    marker = (
        "// Generated by scripts/generate_core_shaders.py using Ordinary Shade.\n"
        "// Edit the typed Python source, not this generated GLSL.\n"
    )
    return source.replace("#version 460\n", f"#version 460\n\n{marker}", 1)


def main():
    check = "--check" in sys.argv
    for output, shader in GENERATED.items():
        source = generated_source(shader, HELPERS.get(output, ()))
        if check:
            if output.read_text() != source:
                raise SystemExit(f"{output} is stale; regenerate it")
            print(f"Verified {output.relative_to(ROOT)}")
        else:
            output.write_text(source)
            print(f"Wrote {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
