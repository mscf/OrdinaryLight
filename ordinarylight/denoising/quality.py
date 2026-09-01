"""Deterministic quality metrics and regression baselines for denoisers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DenoiserQualityMetrics:
    relative_rmse: float
    relative_bias: float
    temporal_residual: float
    firefly_p999: float
    edge_error: float

    @classmethod
    def measure(cls, frames, reference):
        frames = np.asarray(frames, np.float32)
        reference = np.asarray(reference, np.float32)
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError("frames must have shape (frames, height, width, 3)")
        if reference.shape != frames.shape[1:]:
            raise ValueError("reference extent must match frames")
        scale = max(float(np.sqrt(np.mean(reference * reference))), 1e-6)
        mean = np.mean(frames, axis=0)
        error = frames - reference
        relative_rmse = float(np.sqrt(np.mean(error * error)) / scale)
        relative_bias = float(np.sqrt(np.mean((mean - reference) ** 2)) / scale)
        if len(frames) > 1:
            temporal_residual = float(
                np.sqrt(np.mean(np.diff(error, axis=0) ** 2)) / scale
            )
        else:
            temporal_residual = 0.0
        firefly_p999 = float(np.percentile(np.abs(error), 99.9) / scale)
        candidate_edges = np.concatenate((
            np.diff(mean, axis=0).reshape(-1, 3),
            np.diff(mean, axis=1).reshape(-1, 3),
        ))
        reference_edges = np.concatenate((
            np.diff(reference, axis=0).reshape(-1, 3),
            np.diff(reference, axis=1).reshape(-1, 3),
        ))
        edge_error = float(
            np.sqrt(np.mean((candidate_edges - reference_edges) ** 2)) / scale
        )
        return cls(
            relative_rmse, relative_bias, temporal_residual,
            firefly_p999, edge_error,
        )


@dataclass(frozen=True)
class DenoiserQualityBaseline:
    """Maximum accepted metrics for one fixed scene/pose sequence."""

    scene: str
    metrics: DenoiserQualityMetrics
    tolerance: float = 0.03

    def save(self, path):
        payload = {
            "schema": 1,
            "scene": self.scene,
            "tolerance": self.tolerance,
            "metrics": asdict(self.metrics),
        }
        Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        if payload.get("schema") != 1:
            raise ValueError("unsupported denoiser quality baseline schema")
        return cls(
            payload["scene"], DenoiserQualityMetrics(**payload["metrics"]),
            float(payload["tolerance"]),
        )

    def regressions(self, measured):
        limit_scale = 1.0 + self.tolerance
        return {
            name: (float(getattr(measured, name)), float(value) * limit_scale)
            for name, value in asdict(self.metrics).items()
            if float(getattr(measured, name)) > float(value) * limit_scale
        }

    def require(self, measured, *, override_reason=None):
        failures = self.regressions(measured)
        if failures and not override_reason:
            details = ", ".join(
                f"{name}={actual:.6g} > {limit:.6g}"
                for name, (actual, limit) in failures.items()
            )
            raise AssertionError(
                f"denoiser quality regressed for {self.scene}: {details}; "
                "provide an explicit override reason to accept a new baseline"
            )
        return failures


__all__ = ["DenoiserQualityBaseline", "DenoiserQualityMetrics"]
