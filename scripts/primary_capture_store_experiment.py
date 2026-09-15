"""Diagnostic typed source variant: one complete secondary record per primary."""
from pathlib import Path
from primary_hit_layout_experiment import load_module


def variant(directory, *, clear_only=False):
    from ordinarylight.shaders import fused_primary_programs
    source=Path(fused_primary_programs.__file__).read_text()
    if clear_only:
        source = source.replace(
            '    if indirect_capture_pixel:\n        ordinarylight_clear_secondary_path(path_index)\n',
            '    if indirect_capture_pixel:\n        secondary_paths[path_index] = SecondaryPathState(' + ', '.join(['osh.vec4(0.0)'] * 8) + ')\n', 1)
        return load_module('ordinarylight.shaders.primary_clear_diagnostic', source, directory)
    begin=source.index('    if indirect_capture_pixel:\n        ordinarylight_clear_secondary_path(path_index)\n')
    capture=source.index('    if capture_secondary:',begin)
    early=source[begin:capture]
    early=early.replace('    if indirect_capture_pixel:\n        ordinarylight_clear_secondary_path(path_index)\n','',1)
    # All returns in this region have a valid path index/capture policy, and
    # no secondary consumer has run yet. Miss/termination records stay zero.
    lines=[]
    count=0
    for line in early.splitlines(True):
        if line.strip()=='return':
            indent=line[:len(line)-len(line.lstrip())]
            lines.extend([indent+'if indirect_capture_pixel:\n',indent+'    ordinarylight_clear_secondary_path(path_index)\n'])
            count+=1
        lines.append(line)
    assert count==3
    end=source.index('    ordinarylight_store_path(path_index, path)',capture)
    replacement='''    if capture_secondary:
        geometry = osh.vec4(osh.uint_bits_to_float(primitive), barycentrics,
                            osh.uint_bits_to_float(instance_key))
        if osh.specialization('WAVE_DENOISER_SIGNAL_CAPTURE'):
            if material.attenuation_transmission.a > 0.001:
                geometry.y = osh.uint_bits_to_float(osh.float_bits_to_uint(geometry.y) | osh.u32(2147483648))
        secondary_paths[path_index] = SecondaryPathState(
            osh.vec4(0.0), osh.vec4(0.0, 0.0, 0.0, bsdf_pdf),
            osh.vec4(path.throughput.rgb, 1.0 + sampled_specular),
            osh.vec4(path.radiance.rgb, pbrSpecularProbability(material)),
            osh.vec4(indirect_specular_fraction, -1.0),
            osh.vec4(primary_specular, -1.0),
            osh.vec4(position, 1.0 + osh.clamp(material.base_roughness.a, 0.0, 1.0)),
            geometry)
    elif indirect_capture_pixel:
        ordinarylight_clear_secondary_path(path_index)
'''
    return load_module('ordinarylight.shaders.primary_capture_diagnostic',source[:begin]+''.join(lines)+replacement+source[end:],directory)
