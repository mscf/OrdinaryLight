"""Temporary OrdinaryShade producer/consumer variants for lossless hit planes.

Diagnostic only: does not establish a public buffer ABI or modify packages.
"""
import importlib.util
from pathlib import Path
import sys


def load_module(name, source, directory):
    path = directory / (name.replace('.', '_') + '.py')
    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def variants(directory, *, compact=False):
    import generate_fused_primary
    from ordinarylight.shaders import fused_primary_programs
    from vxl8r_render.backends import sparse_average

    source = Path(fused_primary_programs.__file__).read_text()
    begin = source.index('        primary_hits[hit_output_index] = PrimaryHitOutput(')
    end = source.index("    if osh.specialization('WAVE_CUSTOM_GEOMETRY'):", begin)
    block = source[begin:end].replace('primary_hits[hit_output_index]', 'export_hit')
    block += '        export_plane_size = push.image_tile.x * push.image_tile.y\n'
    block += '        export_base = push.tile_frame.w * export_plane_size * osh.u32(6) + pixel_index\n'
    fields = ('position_distance', 'geometric_normal', 'shading_normal', 'identity', 'ray_origin', 'ray_direction')
    for plane, field in enumerate(fields):
        value = 'export_hit.' + field
        if field == 'identity':
            value = 'osh.uint_bits_to_float(' + value + ')'
        block += f'        primary_hits[export_base + export_plane_size * osh.u32({plane})] = {value}\n'
    source = source[:begin] + block + source[end:]
    source = source.replace('primary_hits[hit_output_index].shading_normal =',
                            'primary_hits[export_base + export_plane_size * osh.u32(2)] =')
    if compact:
        begin = source.index('        export_hit = PrimaryHitOutput(')
        end = source.index("    if osh.specialization('WAVE_CUSTOM_GEOMETRY'):", begin)
        source = source[:begin] + '''        compact_base = hit_output_index * osh.u32(5)
        primary_hits[compact_base] = intersection.identity.x
        primary_hits[compact_base + osh.u32(1)] = intersection.identity.y
        primary_hits[compact_base + osh.u32(2)] = intersection.identity.z
        primary_hits[compact_base + osh.u32(3)] = intersection.identity.w
        primary_hits[compact_base + osh.u32(4)] = osh.u32(surface_hit)
''' + source[end:]
        source = source.replace('        primary_hits[export_base + export_plane_size * osh.u32(2)] = osh.vec4(shading_normal, 0)\n', '')
    program = load_module('ordinarylight.shaders.diagnostic_hit_planes', source, directory)
    generator_source = Path(generate_fused_primary.__file__).read_text().replace(
        'primary_hits=osh.runtime_array(p.PrimaryHitOutput)', 'primary_hits=osh.runtime_array(osh.vec4)')
    generator = load_module('diagnostic_hit_plane_generator', generator_source, directory)
    generator.p = program
    # Mechanical buffer declaration; all shader algorithms remain typed Python.
    generator.LAYOUT = generator.LAYOUT.replace('PrimaryHitOutput primary_hits[];', 'vec4 primary_hits[];')
    if compact:
        generator = load_module('diagnostic_compact_hit_generator', generator_source.replace(
            'primary_hits=osh.runtime_array(osh.vec4)', 'primary_hits=osh.runtime_array(osh.u32)'), directory)
        generator.p = program
        generator.LAYOUT = generator.LAYOUT.replace('PrimaryHitOutput primary_hits[];', 'uint primary_hits[];')

    consumer_source = Path(sparse_average.__file__).read_text().replace(
        "hits: osh.storage_buffer(PrimaryHit,access='read',binding=1)",
        "hits: osh.storage_buffer(osh.vec4,access='read',binding=1)")
    consumer_source = consumer_source.replace('face_key(hits[i],pc.slots)',
        'face_key(PrimaryHit(hits[i], osh.vec4(0.0), osh.vec4(0.0), '
        'osh.float_bits_to_uint(hits[pc.width * pc.height * osh.u32(3) + i]), '
        'osh.vec4(0.0), osh.vec4(0.0)),pc.slots)')
    if compact:
        consumer_source = Path(sparse_average.__file__).read_text().replace(
            "hits: osh.storage_buffer(PrimaryHit,access='read',binding=1)",
            "hits: osh.storage_buffer(osh.u32,access='read',binding=1)")
        consumer_source = consumer_source.replace('face_key(hits[i],pc.slots)',
            'face_key(PrimaryHit(osh.vec4(0.0, 0.0, 0.0, 1.0 if hits[i * osh.u32(5) + osh.u32(4)] != osh.u32(0) else -1.0), '
            'osh.vec4(0.0), osh.vec4(0.0), osh.uvec4(hits[i * osh.u32(5)], '
            'hits[i * osh.u32(5) + osh.u32(1)], hits[i * osh.u32(5) + osh.u32(2)], hits[i * osh.u32(5) + osh.u32(3)]), '
            'osh.vec4(0.0), osh.vec4(0.0)),pc.slots)')
    consumer = load_module('vxl8r_render.backends.diagnostic_hit_planes', consumer_source, directory)
    return generator.generated_source(), consumer.compiled_programs


def unpack_hits(packed):
    """Diagnostic readback conversion only; normal rendering stays on GPU."""
    import numpy as np
    samples, height, width = packed.shape
    words = packed.view(np.uint32).reshape(samples, 6, height * width, 4)
    return words.transpose(0, 2, 1, 3).copy().reshape(-1, 24).view(packed.dtype).reshape(packed.shape)
