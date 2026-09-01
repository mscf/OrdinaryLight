"""Denoising signal contracts and portable implementations.

NRD is deliberately exposed from :mod:`ordinarylight.denoising.reference`:
it is a validation oracle, not a runtime dependency of Ordinary Light.
"""

from .signals import (
    DenoiserFrameInfo,
    DenoiserSignals,
    SignalValidationError,
)
from .portable import (
    PortableDenoiser,
    PortableDenoiserConfig,
    PortableDenoiserResult,
)
from .quality import DenoiserQualityBaseline, DenoiserQualityMetrics
from .evaluation import DenoiserSequenceEvaluation, evaluate_denoiser_sequence

__all__ = [
    "DenoiserFrameInfo",
    "DenoiserSignals",
    "SignalValidationError",
    "PortableDenoiser",
    "PortableDenoiserConfig",
    "PortableDenoiserResult",
    "DenoiserQualityBaseline",
    "DenoiserQualityMetrics",
    "DenoiserSequenceEvaluation",
    "evaluate_denoiser_sequence",
]
