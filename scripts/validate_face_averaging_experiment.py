"""Run vxl8r's GPU averaging tests against a temporary diagnostic variant.

Run from the Ordinary root with VXL8R_TEST_VULKAN=1 and vxl8r/src on PYTHONPATH.
"""
import argparse
from contextlib import ExitStack
from pathlib import Path
import tempfile
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('anchor', 'stripes32', 'combined', 'table', 'direct', 'worklist', 'locality', 'reduce-pair', 'group'), required=True)
    parser.add_argument('--group-size', type=int, choices=(8,16), default=8)
    parser.add_argument('--tests', type=Path, default=Path('../vxl8r/tests/test_face_average.py'))
    args = parser.parse_args()
    import pytest
    with ExitStack() as stack:
        if args.variant == 'group':
            from face_group_size_experiment import install
            directory = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            install(stack, directory, lambda: True, args.group_size)
            source = args.tests.read_text()
            old = '(np.arange(count)//4)%target.stripes'
            assert source.count(old) == 1
            source = source.replace(old, f'(np.arange(count)//{args.group_size})%target.stripes')
            args.tests = directory / 'test_face_group.py'
            args.tests.write_text(source)
        if args.variant == 'reduce-pair':
            from face_reduce_pair_experiment import install
            directory = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            install(stack, directory, lambda: True)
        if args.variant == 'locality':
            from face_stripe_locality_experiment import install
            directory = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            install(stack, directory, lambda: True)
            source = args.tests.read_text()
            old = '(count+target.stripes-1)//target.stripes'
            assert source.count(old) == 1 or 'np.count_nonzero((np.arange(count)//4)%target.stripes == 0)' in source
            source = source.replace(old, 'np.count_nonzero((np.arange(count)//4)%target.stripes == 0)')
            args.tests = directory / 'test_face_locality.py'
            args.tests.write_text(source)
        if args.variant == 'worklist':
            from face_worklist_experiment import install
            directory = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            install(stack, directory, lambda: True)
        if args.variant == 'direct':
            from face_direct_anchor_experiment import install
            directory = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            install(stack, directory, lambda: True)
        if args.variant == 'table':
            from face_table_experiment import install
            directory = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            install(stack, directory, lambda: True)
        if args.variant in ('anchor', 'combined'):
            from face_anchor_experiment import install
            directory = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            install(stack, directory, lambda: True)
        if args.variant in ('stripes32', 'combined'):
            from vxl8r_render.backends.sparse_average import SparseFaceAverageTarget
            original = SparseFaceAverageTarget.__init__
            def initialize(target, *a, **kw):
                original(target, *a, **kw)
                target.stripes = min(target.stripes, 32)
            stack.enter_context(patch.object(SparseFaceAverageTarget, '__init__', initialize))
        return pytest.main([str(args.tests), '-q'])


if __name__ == '__main__':
    raise SystemExit(main())
