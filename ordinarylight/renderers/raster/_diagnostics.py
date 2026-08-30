"""Internal frame diagnostics shared by raster implementations and tools."""

from __future__ import annotations

import numpy as np


def frame_difference(reference, current):
    """Return exact per-frame error statistics for equally shaped images."""
    reference = np.asarray(reference)
    current = np.asarray(current)
    if reference.shape != current.shape:
        raise ValueError(
            f"frame shapes differ: {reference.shape} != {current.shape}"
        )
    if reference.ndim < 2:
        raise ValueError("frames must have at least two dimensions")
    delta = current.astype(np.float64) - reference.astype(np.float64)
    if not np.all(np.isfinite(delta)):
        raise ValueError("frame difference contains non-finite values")
    absolute = np.abs(delta)
    changed = delta != 0.0
    if reference.ndim >= 3:
        changed = np.any(changed, axis=-1)
    locations = np.argwhere(changed)
    bounds = None
    if locations.size:
        minimum = locations.min(axis=0)
        maximum = locations.max(axis=0)
        bounds = {
            "x": [int(minimum[1]), int(maximum[1])],
            "y": [int(minimum[0]), int(maximum[0])],
        }
    return {
        "maximum_absolute_difference": float(absolute.max(initial=0.0)),
        "rmse": float(np.sqrt(np.mean(delta ** 2))) if delta.size else 0.0,
        "changed_pixels": int(np.count_nonzero(changed)),
        "changed_bounds": bounds,
    }


__all__ = ["frame_difference"]
