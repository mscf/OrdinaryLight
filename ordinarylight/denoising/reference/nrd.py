"""Optional NVIDIA NRD/ReLAX reference adapter.

The native bridge is intentionally a separate install. Ordinary Light and its
portable Ordinary Shade denoiser never import or link NRD during normal use.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import math

import numpy as np

from ..signals import DenoiserSignals


class ReferenceDenoiserUnavailable(RuntimeError):
    """Raised when the optional NRD native bridge is not installed."""


@dataclass(frozen=True)
class NrdBenchmarkResult:
    """GPU-only NRD timing and allocation telemetry.

    Upload and readback are deliberately excluded from the GPU timings. The
    native bridge measures them with Vulkan timestamp queries around NRD's
    dispatch list. ``wall_ms`` exposes the complete offline bridge overhead.
    """

    median_gpu_ms: float
    p95_gpu_ms: float
    wall_ms: float
    persistent_mib: float
    transient_mib: float
    measured_frames: int
    implementation_version: str

    def __post_init__(self):
        for name in (
            "median_gpu_ms", "p95_gpu_ms", "wall_ms", "persistent_mib",
            "transient_mib",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"NRD benchmark {name} must be finite and non-negative"
                )
            object.__setattr__(self, name, value)
        measured_frames = int(self.measured_frames)
        if measured_frames < 1:
            raise ValueError("NRD benchmark measured_frames must be positive")
        object.__setattr__(self, "measured_frames", measured_frames)


@dataclass(frozen=True)
class NrdReferenceResult:
    diffuse: np.ndarray
    specular: np.ndarray
    implementation_version: str

    def __post_init__(self):
        diffuse = np.ascontiguousarray(self.diffuse, dtype=np.float32)
        specular = np.ascontiguousarray(self.specular, dtype=np.float32)
        if diffuse.ndim != 3 or diffuse.shape[-1] != 3:
            raise ValueError("NRD diffuse output must have shape (height, width, 3)")
        if specular.shape != diffuse.shape:
            raise ValueError("NRD specular output must match diffuse output")
        if not np.isfinite(diffuse).all() or not np.isfinite(specular).all():
            raise ValueError("NRD output must contain finite values")
        object.__setattr__(self, "diffuse", diffuse)
        object.__setattr__(self, "specular", specular)

    @property
    def combined(self):
        return self.diffuse + self.specular


class NrdRelaxReference:
    """Run RELAX through the optional ``ordinarylight_nrd`` native bridge.

    A bridge can be injected for tests. The production bridge API is kept
    deliberately small: ``version()`` and ``denoise_relax(signals, settings)``.
    """

    def __init__(self, bridge=None):
        self._bridge = bridge

    @property
    def available(self):
        try:
            self._resolve_bridge()
        except ReferenceDenoiserUnavailable:
            return False
        return True

    def _resolve_bridge(self):
        if self._bridge is not None:
            return self._bridge
        try:
            self._bridge = importlib.import_module("ordinarylight_nrd")
        except ImportError as error:
            raise ReferenceDenoiserUnavailable(
                "NRD reference support is not installed; build the optional "
                "ordinarylight_nrd bridge from tools/nrd_reference"
            ) from error
        return self._bridge

    def denoise(self, signals, **settings):
        if not isinstance(signals, DenoiserSignals):
            raise TypeError("signals must be ordinarylight.DenoiserSignals")
        bridge = self._resolve_bridge()
        raw = bridge.denoise_relax(signals, dict(settings))
        if not isinstance(raw, (tuple, list)) or len(raw) != 2:
            raise RuntimeError("ordinarylight_nrd returned an invalid RELAX result")
        version = str(bridge.version())
        result = NrdReferenceResult(raw[0], raw[1], version)
        expected = (signals.extent[1], signals.extent[0], 3)
        if result.diffuse.shape != expected:
            raise RuntimeError(
                f"ordinarylight_nrd returned {result.diffuse.shape}, expected {expected}"
            )
        return result

    def denoise_sequence(self, signals, **settings):
        sequence = tuple(signals)
        if not sequence or not all(
            isinstance(value, DenoiserSignals) for value in sequence
        ):
            raise TypeError("signals must contain at least one DenoiserSignals frame")
        extent = sequence[0].extent
        if any(value.extent != extent for value in sequence):
            raise ValueError("all NRD signal frames must share one extent")
        bridge = self._resolve_bridge()
        operation = getattr(bridge, "denoise_relax_sequence", None)
        if operation is None:
            return [self.denoise(value, **settings) for value in sequence]
        raw_results = operation(sequence, dict(settings))
        if len(raw_results) != len(sequence):
            raise RuntimeError("ordinarylight_nrd returned the wrong frame count")
        version = str(bridge.version())
        results = [NrdReferenceResult(*raw, version) for raw in raw_results]
        expected = (extent[1], extent[0], 3)
        if any(result.diffuse.shape != expected for result in results):
            raise RuntimeError("ordinarylight_nrd returned an invalid sequence extent")
        return results

    def benchmark(self, signals, *, warmup=8, iterations=32, **settings):
        """Benchmark RELAX without counting signal upload or result readback."""
        sequence = tuple(signals)
        if not sequence or not all(
            isinstance(value, DenoiserSignals) for value in sequence
        ):
            raise TypeError(
                "signals must contain at least one DenoiserSignals frame"
            )
        extent = sequence[0].extent
        if any(value.extent != extent for value in sequence):
            raise ValueError("all benchmark signal frames must share one extent")
        warmup = int(warmup)
        iterations = int(iterations)
        if warmup < 0 or iterations < 1:
            raise ValueError(
                "warmup must be non-negative and iterations must be positive"
            )
        bridge = self._resolve_bridge()
        benchmark_relax = getattr(bridge, "benchmark_relax", None)
        if benchmark_relax is None:
            raise RuntimeError(
                "ordinarylight_nrd does not expose benchmark_relax; rebuild "
                "the reference bridge with benchmark support"
            )
        raw = benchmark_relax(
            sequence, dict(settings), warmup=warmup, iterations=iterations,
        )
        if not isinstance(raw, dict):
            raise RuntimeError(
                "ordinarylight_nrd returned invalid benchmark telemetry"
            )
        required = {
            "median_gpu_ms", "p95_gpu_ms", "wall_ms", "persistent_mib",
            "transient_mib", "measured_frames",
        }
        missing = sorted(required.difference(raw))
        if missing:
            raise RuntimeError(
                "ordinarylight_nrd benchmark telemetry is missing "
                + ", ".join(missing)
            )
        return NrdBenchmarkResult(
            **{name: raw[name] for name in required},
            implementation_version=str(bridge.version()),
        )


__all__ = [
    "NrdBenchmarkResult", "NrdRelaxReference", "NrdReferenceResult",
    "ReferenceDenoiserUnavailable",
]
