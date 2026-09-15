"""Diagnostic OrdinaryShade four-pixel stripe grouping for list locality."""
from pathlib import Path
from unittest.mock import patch
from primary_hit_layout_experiment import load_module


def install(stack, directory, selected):
    from vxl8r_render.backends import sparse_average
    source = Path(sparse_average.__file__).read_text()
    grouped = '(i/osh.u32(4))%pc.stripes'
    promoted = grouped in source
    assert source.count(grouped if promoted else 'i%pc.stripes') == 2
    source = source.replace(grouped, 'i%pc.stripes') if promoted else source.replace('i%pc.stripes', grouped)
    module = load_module('vxl8r_render.backends.diagnostic_stripe_locality', source, directory)
    original = sparse_average.compiled_programs
    stack.enter_context(patch.object(sparse_average, 'compiled_programs',
        lambda *a, **kw: (module.compiled_programs if selected() != promoted else original)(*a, **kw)))
