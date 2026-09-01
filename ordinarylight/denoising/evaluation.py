"""Sequence evaluation shared by portable and reference denoisers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .portable import PortableDenoiser, PortableDenoiserConfig
from .quality import DenoiserQualityMetrics
from .signals import DenoiserSignals


@dataclass(frozen=True)
class DenoiserSequenceEvaluation:
    """Quality measurements for one canonical denoiser signal sequence."""

    portable: DenoiserQualityMetrics
    reference: DenoiserQualityMetrics | None = None
    portable_against_reference: DenoiserQualityMetrics | None = None
    reference_implementation: str | None = None


def evaluate_denoiser_sequence(
    signals, ground_truth, *, portable_config=None, reference_denoiser=None,
):
    """Evaluate the portable denoiser and an optional reference implementation.

    ``ground_truth`` is a high-sample linear RGB image. The optional reference
    implements the same ``denoise(signals)`` protocol as ``NrdRelaxReference``.
    """
    sequence = tuple(signals)
    if not sequence:
        raise ValueError("signals must contain at least one frame")
    if not all(isinstance(frame, DenoiserSignals) for frame in sequence):
        raise TypeError("signals must contain only DenoiserSignals")
    extent = sequence[0].extent
    if any(frame.extent != extent for frame in sequence):
        raise ValueError("all denoiser signal frames must share one extent")
    truth = np.ascontiguousarray(ground_truth, dtype=np.float32)
    expected = (extent[1], extent[0], 3)
    if truth.shape != expected:
        raise ValueError(
            f"ground_truth must have shape {expected}, got {truth.shape}"
        )
    if not np.isfinite(truth).all():
        raise ValueError("ground_truth must contain finite values")

    portable = PortableDenoiser(portable_config or PortableDenoiserConfig())
    portable_frames = np.stack([
        portable.denoise(frame).combined for frame in sequence
    ])
    portable_metrics = DenoiserQualityMetrics.measure(portable_frames, truth)
    if reference_denoiser is None:
        return DenoiserSequenceEvaluation(portable=portable_metrics)

    reference_results = [reference_denoiser.denoise(frame) for frame in sequence]
    reference_frames = np.stack([result.combined for result in reference_results])
    reference_metrics = DenoiserQualityMetrics.measure(reference_frames, truth)
    portable_against_reference = DenoiserQualityMetrics.measure(
        portable_frames, np.mean(reference_frames, axis=0),
    )
    versions = {
        str(getattr(result, "implementation_version", "unknown"))
        for result in reference_results
    }
    implementation = versions.pop() if len(versions) == 1 else "mixed"
    return DenoiserSequenceEvaluation(
        portable=portable_metrics,
        reference=reference_metrics,
        portable_against_reference=portable_against_reference,
        reference_implementation=implementation,
    )


__all__ = ["DenoiserSequenceEvaluation", "evaluate_denoiser_sequence"]
