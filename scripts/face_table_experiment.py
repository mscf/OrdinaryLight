"""Diagnostic sparse averaging table with two buckets per input pixel."""
from pathlib import Path
from unittest.mock import patch
from primary_hit_layout_experiment import load_module


def install(stack, directory, selected):
    from vxl8r_render.backends import sparse_average
    source = Path(sparse_average.__file__).read_text()
    original = 'self.capacity=1 << (max(4,4*w*h)-1).bit_length()'
    assert source.count(original) == 1
    source = source.replace(original, 'self.capacity=1 << (max(4,2*w*h)-1).bit_length()')
    source = source.replace('# At most two keys per pixel (anchor and stripe), so load <= 1/2.',
        '# Diagnostic: at most two keys per pixel; high occupancy can hurt probing.')
    module = load_module('vxl8r_render.backends.diagnostic_face_table', source, directory)
    original_init = sparse_average.SparseFaceAverageTarget.__init__
    def initialize(target, *a, **kw):
        (module.SparseFaceAverageTarget.__init__ if selected() else original_init)(target, *a, **kw)
    stack.enter_context(patch.object(sparse_average.SparseFaceAverageTarget, '__init__', initialize))
