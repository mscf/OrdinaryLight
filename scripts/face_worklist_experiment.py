"""Diagnostic OrdinaryShade subgroup reservation for newly inserted face entries."""
from pathlib import Path
from unittest.mock import patch
from primary_hit_layout_experiment import load_module


def install(stack, directory, selected):
    from vxl8r_render.backends import sparse_average
    source = Path(sparse_average.__file__).read_text()
    assert source.count('osh.atomic_add(worklist[0],osh.u32(1))') == 1
    assert source.count('osh.atomic_add(worklist[0], osh.u32(1))') == 2
    source = source.replace('osh.atomic_add(worklist[0],osh.u32(1))', 'reserve_worklist()')
    source = source.replace('osh.atomic_add(worklist[0], osh.u32(1))', 'reserve_worklist()')
    helper = '''
@osh.function
def reserve_worklist() -> osh.u32:
    lanes = osh.subgroup_ballot(True)
    base = osh.u32(0)
    if osh.subgroup_elect():
        base = osh.atomic_add(worklist[0], osh.subgroup_ballot_bit_count(lanes))
    base = osh.subgroup_broadcast_first(base)
    return base + osh.subgroup_ballot_exclusive_bit_count(lanes)

'''
    source = source.replace('@osh.function\ndef hashed', helper + '@osh.function\ndef hashed')
    source = source.replace('@osh.compute(workgroup_size=(64,1,1))',
        "@osh.compute(workgroup_size=(64,1,1), capabilities=('subgroup_ballot',))", 1)
    source = source.replace('hashed,insert_key,insert_stripe)', 'hashed,insert_key,insert_stripe,reserve_worklist)')
    module = load_module('vxl8r_render.backends.diagnostic_worklist', source, directory)
    original = sparse_average.compiled_programs
    stack.enter_context(patch.object(sparse_average, 'compiled_programs',
        lambda *a, **kw: (module.compiled_programs if selected() else original)(*a, **kw)))
