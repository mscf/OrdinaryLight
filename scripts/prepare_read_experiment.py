"""Diagnostic OrdinaryShade variant: access secondary fields at their use sites."""
from pathlib import Path
from primary_hit_layout_experiment import load_module


def compile_variant(directory, *, single_sample=False):
    import ordinaryshade as osh
    from ordinarylight.denoising import kernels
    from ordinarylight.runtime import compile_compute
    source = Path(kernels.__file__).read_text()
    begin = source.index('def prepare_relax_signals(')
    end = source.index('\n\n@osh.structure\nclass TemporalConstants', begin)
    body = source[begin:end]
    assert body.count('    secondary = secondary_paths[path_index]\n') == 1
    if single_sample:
        # Diagnostic contract: sampled-indirect, one sample, no planar-mirror guides.
        for field, value in (('x', 0), ('y', 1), ('z', 1), ('w', 0)):
            body = body.replace('constants.samples.' + field, 'osh.u32(' + str(value) + ')')
    else:
        body = body.replace('    secondary = secondary_paths[path_index]\n', '')
        body = body.replace('secondary.', 'secondary_paths[path_index].')
        body = body.replace('prepare_surface_history(secondary,', 'prepare_surface_history(secondary_paths[path_index],')
    module = load_module('prepare_read_diagnostic', source[:begin]+body+source[end:], directory)
    return compile_compute(osh.compile(module.prepare_relax_signals, helpers=(
        module.prepare_decode_normal, module.prepare_unpack_normal,
        module.prepare_previous_pixel, module.prepare_custom_surface_history,
    )).source)
