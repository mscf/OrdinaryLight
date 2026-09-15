"""Diagnostic direct anchor storage with a half-full stripe hash table."""
from pathlib import Path
from unittest.mock import patch
from primary_hit_layout_experiment import load_module


def install(stack, directory, selected):
    from vxl8r_render.backends import sparse_average
    source = Path(sparse_average.__file__).read_text()
    original = '        anchor=insert_key(base,pc.capacity)'
    replacement = '''        anchor = osh.u32(4294967295)
        if osh.array_length(keys) > pc.capacity:
            anchor = pc.capacity + key
            previous = osh.atomic_compare_exchange(keys[anchor], osh.u32(4294967295), base)
            if previous == osh.u32(4294967295):
                offset = osh.atomic_add(worklist[0], osh.u32(1))
                worklist[osh.u32(4) + offset] = anchor
        else:
            anchor = insert_key(base, pc.capacity)'''
    promoted = replacement in source
    assert source.count(replacement if promoted else original) == 1
    source = source.replace(replacement, original) if promoted else source.replace(original, replacement)
    allocation = '        self.capacity=1 << (max(4,4*w*h)-1).bit_length()'
    direct_allocation = '''        stripe_capacity = 1 << (max(4,2*w*h)-1).bit_length()
        self.direct_anchor_count = self.slots*6 if self.slots*6 <= stripe_capacity else 0
        self.capacity = stripe_capacity if self.direct_anchor_count else 1 << (max(4,4*w*h)-1).bit_length()
        storage_capacity = self.capacity + self.direct_anchor_count'''
    assert source.count(direct_allocation if promoted else allocation) == 1
    source = source.replace(direct_allocation, allocation) if promoted else source.replace(allocation, direct_allocation)
    for field, size in (('keys',4),('heads',4),('partials',16)):
        old = f'self.{field}=own(runtime.buffer(self.capacity*{size},'
        new = f'self.{field}=own(runtime.buffer(storage_capacity*{size},'
        assert source.count(new if promoted else old) == 1
        source = source.replace(new, old) if promoted else source.replace(old, new)
    source = source.replace('# At most two keys per pixel (anchor and stripe), so load <= 1/2.',
        '# Direct anchors are outside the hash domain; at most one stripe key per pixel.')
    module = load_module('vxl8r_render.backends.diagnostic_direct_anchors', source, directory)
    original_init = sparse_average.SparseFaceAverageTarget.__init__
    original_programs = sparse_average.compiled_programs
    def initialize(target, *a, **kw):
        (module.SparseFaceAverageTarget.__init__ if selected() != promoted else original_init)(target, *a, **kw)
    stack.enter_context(patch.object(sparse_average.SparseFaceAverageTarget, '__init__', initialize))
    stack.enter_context(patch.object(sparse_average, 'compiled_programs',
        lambda *a, **kw: (module.compiled_programs if selected() != promoted else original_programs)(*a, **kw)))
