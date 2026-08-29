"""Panoramic camera model."""

from dataclasses import dataclass

from ._validation import validate_look_at


@dataclass(frozen=True)
class PanoramicCamera:
    """Equirectangular camera centered on its look-at orientation."""

    position: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 1.0, 0.0)
    horizontal_fov_degrees: float = 360.0
    vertical_fov_degrees: float = 180.0

    def __post_init__(self):
        validate_look_at(self.position, self.target, self.up)
        if not 1.0 <= self.horizontal_fov_degrees <= 360.0:
            raise ValueError("horizontal_fov_degrees must be in [1, 360]")
        if not 1.0 <= self.vertical_fov_degrees <= 180.0:
            raise ValueError("vertical_fov_degrees must be in [1, 180]")
