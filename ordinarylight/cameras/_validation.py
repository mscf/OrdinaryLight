"""Shared camera validation."""

import numpy as np


def vec3(value, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result


def validate_look_at(position, target, up):
    position = vec3(position, "position")
    target = vec3(target, "target")
    up = vec3(up, "up")
    forward = target - position
    if np.linalg.norm(forward) < 1e-6:
        raise ValueError("camera position and target must differ")
    if np.linalg.norm(up) < 1e-6:
        raise ValueError("camera up vector cannot be zero")
    if np.linalg.norm(np.cross(forward, up)) < 1e-6:
        raise ValueError(
            "camera up vector cannot be parallel to its view direction"
        )
