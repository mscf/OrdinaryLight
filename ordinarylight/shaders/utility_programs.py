"""Typed OrdinaryShade sources for image preparation and metadata utilities."""
import ordinaryshade as osh


@osh.structure
class ToneConstants:
    exposure: osh.f32


@osh.structure
class FsrPreparationConstants:
    jitter: osh.vec2
    extent: osh.uvec2


@osh.structure
class MetadataConstants:
    count: osh.u32
    triangles: osh.u32


@osh.structure
class SampleAccumulation:
    radiance: osh.vec4
    counts: osh.uvec4
    events: osh.uvec4


@osh.structure
class AccumulationConstants:
    width: osh.u32
    height: osh.u32
    capacity: osh.u32


def _tone_map_program(format):
    """Specialize only the storage format; both inputs use the same display transform."""
    @osh.compute(workgroup_size=(8, 8, 1))
    def external_hdr_tone_map(
        hdr: osh.storage_image(format, access='read', binding=0),
        output_image: osh.storage_image('rgba8', access='write', binding=1),
        push: osh.push_constants(ToneConstants),
    ):
        pixel = osh.ivec2(osh.global_invocation_id.xy)
        extent = output_image.size()
        if pixel.x >= extent.x or pixel.y >= extent.y:
            return
        x = osh.maximum(hdr.load(pixel).rgb, osh.vec3(0.0)) * push.exposure
        color = osh.clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14),
                          osh.vec3(0.0), osh.vec3(1.0))
        color = osh.select(color <= osh.vec3(0.0031308), 12.92 * color,
                           1.055 * osh.power(color, osh.vec3(1.0 / 2.4)) - 0.055)
        output_image.store(pixel, osh.vec4(color, 1.0))
    return external_hdr_tone_map


external_hdr_tone_map = _tone_map_program('rgba32f')
external_hdr_tone_map_16f = _tone_map_program('rgba16f')


@osh.compute(workgroup_size=(8, 8, 1))
def fsr2_prepare(
    view_z: osh.storage_image('r32f', access='read', binding=0),
    motion: osh.storage_image('rgba16f', access='read', binding=1),
    normal_roughness: osh.storage_image('rgba16f', access='read', binding=2),
    depth_out: osh.storage_image('r32f', access='write', binding=3),
    motion_out: osh.storage_image('rg16f', access='write', binding=4),
    reactive_out: osh.storage_image('r8', access='write', binding=5),
    push: osh.push_constants(FsrPreparationConstants),
):
    pixel = osh.ivec2(osh.global_invocation_id.xy)
    if pixel.x >= osh.i32(push.extent.x) or pixel.y >= osh.i32(push.extent.y):
        return
    z = view_z.load(pixel).x
    m = motion.load(pixel)
    depth = 0.0
    if z > 0.1:
        depth = osh.clamp((1000.0 / z - 0.1) / 9999.9, 0.0, 1.0)
    depth_out.store(pixel, osh.vec4(depth))
    motion_out.store(pixel, osh.vec4(m.xy - push.jitter, 0.0, 0.0))
    roughness = normal_roughness.load(pixel).w
    reactive = 0.0
    if z <= 0.0 or m.z <= 0.0:
        reactive = 1.0
    elif roughness < 0.25:
        reactive = 0.9
    reactive_out.store(pixel, osh.vec4(reactive))


@osh.compute(workgroup_size=(64, 1, 1))
def primary_metadata(
    paths: osh.storage_buffer(osh.u32, access='read', binding=0),
    secondary: osh.storage_buffer(osh.u32, binding=1),
    metadata: osh.storage_buffer(osh.uvec4, access='read', binding=2),
    material_image: osh.storage_image('r32ui', access='write', binding=3),
    push: osh.push_constants(MetadataConstants),
):
    i = osh.global_invocation_id.x
    if i >= push.count:
        return
    pixel = paths[i * osh.u32(12) + osh.u32(8)]
    size = material_image.size()
    if pixel >= osh.u32(size.x * size.y):
        return
    p = osh.ivec2(pixel % osh.u32(size.x), pixel / osh.u32(size.x))
    base = i * osh.u32(32)
    primitive = secondary[base + osh.u32(28)]
    if osh.uint_bits_to_float(secondary[base + osh.u32(27)]) <= 0.5 or primitive >= push.triangles:
        material_image.store(p, osh.uvec4(0))
        return
    value = metadata[primitive]
    secondary[base + osh.u32(27)] = osh.float_bits_to_uint(
        1.0 + osh.clamp(osh.uint_bits_to_float(value.z), 0.0, 1.0))
    secondary[base + osh.u32(31)] = value.y
    material_image.store(p, osh.uvec4(value.x))


@osh.compute(workgroup_size=(64, 1, 1))
def accumulation_resolve(
    accumulated: osh.storage_buffer(SampleAccumulation, access='read', binding=0),
    hdr: osh.storage_image('rgba32f', access='write', binding=1),
    pc: osh.push_constants(AccumulationConstants),
):
    i = osh.global_invocation_id.x
    if i >= pc.width * pc.height:
        return
    value = osh.vec3(0.0)
    if i < pc.capacity:
        state = accumulated[i]
        if state.radiance.w > 0.0:
            value = state.radiance.rgb / state.radiance.w
        if state.counts.z != osh.u32(0):
            value = osh.vec3(1.0, 0.0, 1.0)
    hdr.store(osh.ivec2(i % pc.width, i / pc.width), osh.vec4(value, 1.0))


PROGRAMS = {name: globals()[name] for name in (
    'external_hdr_tone_map', 'external_hdr_tone_map_16f', 'fsr2_prepare', 'primary_metadata', 'accumulation_resolve',
)}


@osh.compute(workgroup_size=(4, 4, 4))
def volume_upload(source: osh.storage_buffer(osh.f32, access='read', binding=0),
                  destination: osh.storage_image('rgba16f', access='write', binding=1, dimensions=3)):
    index = osh.uvec3(osh.global_invocation_id.x, osh.global_invocation_id.y, osh.global_invocation_id.z)
    size = osh.uvec3(destination.size())
    if osh.any_value(index >= size):
        return
    offset = (index.z * size.y + index.y) * size.x + index.x
    destination.store(osh.ivec3(index), osh.vec4(source[offset], 0.0, 0.0, 1.0))
