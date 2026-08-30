"""Local image-based reflection resource."""

from dataclasses import dataclass, field
import math
import numpy as np


@dataclass(frozen=True)
class ReflectionProbe:
    """A local equirectangular radiance probe used by raster renderers.

    GI renderers intentionally ignore probes and trace the represented scene.
    ``radius`` controls the probe's influence. Render targets may choose the
    nearest containing probe when several probes overlap.
    """

    image: np.ndarray = field(compare=False, repr=False)
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = float("inf")
    intensity: float = 1.0
    rotation: float = 0.0

    def __post_init__(self):
        image = np.asarray(self.image, dtype=np.float32)
        position = np.asarray(self.position, dtype=np.float32)
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError("probe image must have shape (height, width, 3 or 4)")
        if not np.all(np.isfinite(image)) or np.any(image < 0.0):
            raise ValueError("probe radiance must be finite and non-negative")
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("probe position must contain three finite values")
        if self.radius <= 0.0 or math.isnan(self.radius):
            raise ValueError("probe radius must be positive")
        if not math.isfinite(self.intensity) or self.intensity < 0.0:
            raise ValueError("probe intensity must be finite and non-negative")
        if not math.isfinite(self.rotation):
            raise ValueError("probe rotation must be finite")
        image = np.ascontiguousarray(image[..., :3])
        image.flags.writeable = False
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "position", tuple(float(v) for v in position))
