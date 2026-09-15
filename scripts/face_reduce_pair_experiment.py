"""Diagnostic OrdinaryShade two-node linked-list reduction, preserving sum order."""
from pathlib import Path
from unittest.mock import patch
from primary_hit_layout_experiment import load_module


def install(stack, directory, selected):
    from vxl8r_render.backends import sparse_average
    source = Path(sparse_average.__file__).read_text()
    original = '''        while current!=osh.u32(4294967295):
            color=hdr.load(osh.ivec2(current%pc.width,current/pc.width))
            total=total+osh.vec4(color.rgb,1.0)
            current=links[current]'''
    replacement = '''        while current!=osh.u32(4294967295):
            next_pixel = links[current]
            color = hdr.load(osh.ivec2(current%pc.width,current/pc.width))
            total = total + osh.vec4(color.rgb,1.0)
            current = next_pixel
            if current!=osh.u32(4294967295):
                next_pixel = links[current]
                color = hdr.load(osh.ivec2(current%pc.width,current/pc.width))
                total = total + osh.vec4(color.rgb,1.0)
                current = next_pixel'''
    assert source.count(original) == 1
    source = source.replace(original, replacement)
    module = load_module('vxl8r_render.backends.diagnostic_reduce_pair', source, directory)
    original_programs = sparse_average.compiled_programs
    stack.enter_context(patch.object(sparse_average, 'compiled_programs',
        lambda *a, **kw: (module.compiled_programs if selected() else original_programs)(*a, **kw)))
