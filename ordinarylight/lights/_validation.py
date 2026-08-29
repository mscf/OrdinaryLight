"""Shared analytic-light validation."""

import math
import numpy as np


def vec3(value, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result


def validate_color_intensity(color, intensity):
    color = vec3(color, "color")
    if np.any(color < 0.0):
        raise ValueError("light color components cannot be negative")
    if not math.isfinite(intensity) or intensity < 0.0:
        raise ValueError("light intensity must be finite and non-negative")


def validate_direction(direction):
    result = vec3(direction, "direction")
    length = float(np.linalg.norm(result))
    if length < 1e-6:
        raise ValueError("light direction cannot be zero")
    return result / length


def validate_range(value):
    if value is not None and (not math.isfinite(value) or value <= 0.0):
        raise ValueError("light range must be finite and positive, or None")
