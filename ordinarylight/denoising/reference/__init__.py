"""Optional third-party denoisers used only as validation references."""

from .nrd import (
    NrdBenchmarkResult,
    NrdRelaxReference,
    NrdReferenceResult,
    ReferenceDenoiserUnavailable,
)

__all__ = [
    "NrdBenchmarkResult",
    "NrdRelaxReference",
    "NrdReferenceResult",
    "ReferenceDenoiserUnavailable",
]
