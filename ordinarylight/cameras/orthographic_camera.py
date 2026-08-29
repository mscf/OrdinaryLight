"""Orthographic camera model."""

from dataclasses import dataclass

import numpy as np

from ._validation import validate_look_at


@dataclass(frozen=True)
class OrthographicCamera:
    """Parallel-projection camera with a world-space vertical view size."""

    position: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    vertical_size: float = 2.0

    def __post_init__(self):
        validate_look_at(self.position, self.target, self.up)
        if not np.isfinite(self.vertical_size) or self.vertical_size <= 0.0:
            raise ValueError("vertical_size must be finite and positive")

    @property
    def vertical_fov_degrees(self):
        """Compatibility scale used only by motion heuristics."""
        return 90.0
