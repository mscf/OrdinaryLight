"""Compare production paired filtering to the retained separate-lobe fallback."""
from unittest.mock import patch


def install(stack, directory, selected):
    from ordinarylight.runtime import relax
    supported = relax._supports_paired_filter
    stack.enter_context(patch.object(relax, '_supports_paired_filter',
                                    lambda runtime: selected() and supported(runtime)))
