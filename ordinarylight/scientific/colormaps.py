"""Dependency-free, deterministic scientific color-map definitions."""

from __future__ import annotations

import numpy as np


# Canonical display-sRGB control samples. Fixed literals make exports
# independent of optional plotting libraries; ``colormap`` converts its
# resampled result to linear RGB for Ordinary Light's rendering contract.
_MAPS = {
    "viridis": np.asarray((
        (0.267004, 0.004874, 0.329415), (0.278826, 0.175490, 0.483397),
        (0.229739, 0.322361, 0.545706), (0.172719, 0.448791, 0.557885),
        (0.127568, 0.566949, 0.550556), (0.157851, 0.683765, 0.501686),
        (0.369214, 0.788888, 0.382914), (0.678489, 0.863742, 0.189503),
        (0.993248, 0.906157, 0.143936),
    ), np.float32),
    "cividis": np.asarray((
        (0.000000, 0.135112, 0.304751), (0.103401, 0.220406, 0.435790),
        (0.263738, 0.307831, 0.422789), (0.401418, 0.395617, 0.382826),
        (0.516644, 0.482832, 0.350383), (0.621227, 0.569197, 0.319977),
        (0.720438, 0.656756, 0.288912), (0.824940, 0.746572, 0.242316),
        (0.995737, 0.909344, 0.217772),
    ), np.float32),
    "magma": np.asarray((
        (0.001462, 0.000466, 0.013866), (0.078815, 0.054184, 0.211667),
        (0.232077, 0.059889, 0.437695), (0.390384, 0.100379, 0.501864),
        (0.550287, 0.161158, 0.505719), (0.716387, 0.214982, 0.475290),
        (0.868793, 0.287728, 0.409303), (0.967671, 0.439703, 0.359810),
        (0.987053, 0.991438, 0.749504),
    ), np.float32),
    "coolwarm": np.asarray((
        (0.229806, 0.298718, 0.753683), (0.383662, 0.510183, 0.917831),
        (0.554312, 0.690097, 0.995516), (0.724041, 0.814910, 0.975651),
        (0.865395, 0.865411, 0.865396), (0.958279, 0.604336, 0.483298),
        (0.865391, 0.371128, 0.295769), (0.705673, 0.015556, 0.150233),
    ), np.float32),
    "gray": np.asarray(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)), np.float32),
}


def available_colormaps():
    """Return stable built-in map names in recommended-first order."""
    return tuple(_MAPS)


def colormap(name="viridis", samples=256, *, reverse=False):
    """Return a deterministic linear-float32 RGB lookup table."""
    try:
        controls = _MAPS[str(name).lower()]
    except KeyError as error:
        raise ValueError(
            f"unknown color map {name!r}; choose from {', '.join(_MAPS)}"
        ) from error
    if isinstance(samples, bool):
        raise TypeError("samples must be an integer")
    samples = int(samples)
    if samples < 2:
        raise ValueError("samples must be at least 2")
    if not isinstance(reverse, bool):
        raise TypeError("reverse must be a bool")
    coordinates = np.linspace(0.0, len(controls) - 1, samples)
    lower = np.floor(coordinates).astype(np.int64)
    upper = np.minimum(lower + 1, len(controls) - 1)
    weight = (coordinates - lower)[:, None]
    result = controls[lower] * (1.0 - weight) + controls[upper] * weight
    result = np.where(
        result <= 0.04045, result / 12.92,
        ((result + 0.055) / 1.055) ** 2.4,
    )
    if reverse:
        result = result[::-1]
    return np.ascontiguousarray(result, np.float32)


def opacity_curve(opacity, samples=256):
    """Resolve a scalar, endpoint pair, table, or callable opacity curve."""
    coordinates = np.linspace(0.0, 1.0, int(samples), dtype=np.float32)
    if callable(opacity):
        values = np.asarray(opacity(coordinates), np.float32)
    else:
        values = np.asarray(opacity, np.float32)
        if values.ndim == 0:
            values = np.full(len(coordinates), values, np.float32)
        elif values.ndim == 1 and len(values) != len(coordinates):
            if len(values) < 2:
                raise ValueError("opacity table must contain at least two values")
            values = np.interp(
                coordinates, np.linspace(0.0, 1.0, len(values)), values,
            ).astype(np.float32)
    if values.shape != coordinates.shape:
        raise ValueError("opacity must resolve to one value per color sample")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("opacity values must be finite and in [0, 1]")
    return np.ascontiguousarray(values, np.float32)
