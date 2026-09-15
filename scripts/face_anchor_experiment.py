"""Diagnostic OrdinaryShade subgroup sharing of coherent face anchor lookup."""
from pathlib import Path
from unittest.mock import patch
from primary_hit_layout_experiment import load_module


def install(stack, directory, selected):
    from vxl8r_render.backends import sparse_average
    source = Path(sparse_average.__file__).read_text()
    old = '        anchor=insert_key(base,pc.capacity)'
    new = '''        first_base = osh.subgroup_broadcast_first(base)
        active_count = osh.subgroup_ballot_bit_count(osh.subgroup_ballot(True))
        matching = osh.subgroup_ballot_bit_count(osh.subgroup_ballot(base == first_base))
        anchor = osh.u32(4294967295)
        if active_count == matching:
            if osh.subgroup_elect():
                anchor = insert_key(base, pc.capacity)
            anchor = osh.subgroup_broadcast_first(anchor)
        else:
            anchor = insert_key(base, pc.capacity)'''
    assert source.count(old) == 1
    source = source.replace(old, new).replace('@osh.compute(workgroup_size=(64,1,1))',
        "@osh.compute(workgroup_size=(64,1,1), capabilities=('subgroup_ballot',))", 1)
    module = load_module('vxl8r_render.backends.diagnostic_face_anchor', source, directory)
    original = sparse_average.compiled_programs
    stack.enter_context(patch.object(sparse_average, 'compiled_programs',
        lambda *a, **kw: (module.compiled_programs if selected() else original)(*a, **kw)))
