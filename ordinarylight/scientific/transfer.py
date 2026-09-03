"""Physical scalar-value mapping shared by scientific render primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..scene import Texture1D, VolumeMaterial


@dataclass(frozen=True, slots=True)
class ScalarMapping:
    """Map physical scalar values to the normalized transfer-function domain."""

    mode: str = "linear"
    value_range: tuple[float, float] | None = None
    percentiles: tuple[float, float] = (2.0, 98.0)
    linear_threshold: float = 1.0

    def __post_init__(self):
        if self.mode not in {"linear", "log", "symlog", "percentile", "normalized"}:
            raise ValueError("unsupported scalar mapping mode")
        if self.value_range is not None:
            values = tuple(float(value) for value in self.value_range)
            if len(values) != 2 or not np.isfinite(values).all() or values[1] <= values[0]:
                raise ValueError("value_range must be a finite increasing pair")
            if self.mode == "log" and values[0] <= 0.0:
                raise ValueError("logarithmic value_range must be positive")
            object.__setattr__(self, "value_range", values)
        percentiles = tuple(float(value) for value in self.percentiles)
        if len(percentiles) != 2 or not 0.0 <= percentiles[0] < percentiles[1] <= 100.0:
            raise ValueError("percentiles must be an increasing pair in [0, 100]")
        object.__setattr__(self, "percentiles", percentiles)
        threshold = float(self.linear_threshold)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("linear_threshold must be positive")
        object.__setattr__(self, "linear_threshold", threshold)

    def resolved_range(self, data):
        finite = np.asarray(data)[np.isfinite(data)]
        if self.value_range is not None:
            return self.value_range
        if self.mode == "normalized":
            return (0.0, 1.0)
        if not finite.size:
            raise ValueError("cannot infer a range from data without finite values")
        if self.mode == "log":
            finite = finite[finite > 0.0]
            if not finite.size:
                raise ValueError("logarithmic mapping requires positive values")
        if self.mode == "percentile":
            low, high = np.percentile(finite, self.percentiles)
        else:
            low, high = np.min(finite), np.max(finite)
        low, high = float(low), float(high)
        if low == high:
            high = np.nextafter(low, np.inf)
        return low, high

    def normalize(self, data):
        values = np.asarray(data, dtype=np.float32)
        low, high = self.resolved_range(values)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            if self.mode == "log":
                mapped = (np.log(values) - np.log(low)) / (np.log(high) - np.log(low))
            elif self.mode == "symlog":
                transform = lambda value: np.sign(value) * np.log1p(np.abs(value) / self.linear_threshold)
                mapped = (transform(values) - transform(low)) / (transform(high) - transform(low))
            else:
                mapped = (values - low) / (high - low)
        valid = np.isfinite(values) & np.isfinite(mapped)
        return np.asarray(np.where(valid, np.clip(mapped, 0.0, 1.0), 0.0), np.float32), valid


@dataclass(frozen=True, slots=True)
class TransferFunction:
    """A reproducible physical-value mapping and linear RGBA lookup table."""

    rgba: np.ndarray
    mapping: ScalarMapping = ScalarMapping()
    missing_rgba: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def __post_init__(self):
        if not isinstance(self.mapping, ScalarMapping):
            raise TypeError("mapping must be a ScalarMapping")
        texture = Texture1D(self.rgba)
        missing = tuple(float(value) for value in self.missing_rgba)
        if len(missing) != 4 or not np.isfinite(missing).all():
            raise ValueError("missing_rgba must contain four finite values")
        object.__setattr__(self, "rgba", texture.values)
        object.__setattr__(self, "missing_rgba", missing)

    @classmethod
    def from_colormap(
        cls, name="viridis", *, mapping=None, samples=256, opacity=1.0,
        reverse=False, missing_rgba=(0.0, 0.0, 0.0, 0.0),
    ):
        """Construct a shared transfer function from a built-in color map."""
        from .colormaps import colormap, opacity_curve
        rgb = colormap(name, samples, reverse=reverse)
        alpha = opacity_curve(opacity, samples)[:, None]
        return cls(
            np.concatenate((rgb, alpha), axis=1),
            ScalarMapping() if mapping is None else mapping,
            missing_rgba,
        )

    def map(self, data):
        """Return normalized values and a validity mask without mutating input."""
        return self.mapping.normalize(data)

    def colors(self, data):
        """Map physical values directly to linear RGBA, including missing data."""
        normalized, valid = self.map(data)
        colors = Texture1D(self.rgba).sample(normalized)
        return np.asarray(
            np.where(valid[..., None], colors, self.missing_rgba), np.float32,
        )

    def encode_volume(self, data):
        """Encode values with a dedicated lookup entry for missing samples."""
        normalized, valid = self.map(data)
        count = len(self.rgba)
        # Entry zero is reserved for invalid data. Valid values address the
        # original lookup samples exactly at entries 1..count.
        encoded = (1.0 + normalized * max(count - 1, 0)) / count
        return np.asarray(np.where(valid, encoded, 0.0), np.float32), valid

    def volume_material(self, **kwargs):
        """Build an unlit volume material whose first entry is missing data."""
        lookup = np.vstack((self.missing_rgba, self.rgba))
        return VolumeMaterial(Texture1D(lookup), **kwargs)

    def snapshot(self, data=None):
        value_range = self.mapping.value_range
        if data is not None:
            value_range = self.mapping.resolved_range(data)
        return {
            "mode": self.mapping.mode,
            "value_range": None if value_range is None else list(value_range),
            "percentiles": list(self.mapping.percentiles),
            "linear_threshold": self.mapping.linear_threshold,
            "rgba": self.rgba.tolist(),
            "missing_rgba": list(self.missing_rgba),
        }
