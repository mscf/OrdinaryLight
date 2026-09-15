"""Diagnostic OrdinaryShade stripe grouping sweep against production."""
from pathlib import Path
from unittest.mock import patch
from primary_hit_layout_experiment import load_module


def install(stack, directory, selected, group_size):
    if group_size not in (8, 16):
        raise ValueError('Expected an eight- or sixteen-pixel group')
    from vxl8r_render.backends import sparse_average
    source = Path(sparse_average.__file__).read_text()
    original = '(i/osh.u32(4))%pc.stripes'
    assert source.count(original) == 2
    source = source.replace(original, f'(i/osh.u32({group_size}))%pc.stripes')
    module = load_module('vxl8r_render.backends.diagnostic_group_size', source, directory)
    original_programs = sparse_average.compiled_programs
    stack.enter_context(patch.object(sparse_average, 'compiled_programs',
        lambda *a, **kw: (module.compiled_programs if selected() else original_programs)(*a, **kw)))
