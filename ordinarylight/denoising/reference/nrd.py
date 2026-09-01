"""Optional NVIDIA NRD/ReLAX reference adapter.

The native bridge is intentionally a separate install. Ordinary Light and its
portable Ordinary Shade denoiser never import or link NRD during normal use.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib

import numpy as np

from ..signals import DenoiserSignals


class ReferenceDenoiserUnavailable(RuntimeError):
    """Raised when the optional NRD native bridge is not installed."""


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


__all__ = [
    "NrdRelaxReference", "NrdReferenceResult", "ReferenceDenoiserUnavailable",
]
