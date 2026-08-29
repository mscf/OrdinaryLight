"""Environment-light resource."""

from dataclasses import dataclass, field
import math
import numpy as np

from ._validation import validate_color_intensity


@dataclass(frozen=True)
class EnvironmentLight:
    image: np.ndarray | None = field(default=None, compare=False, repr=False)
    color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    intensity: float = 1.0
    rotation: float = 0.0

    def __post_init__(self):
        validate_color_intensity(self.color, self.intensity)
        if not math.isfinite(self.rotation):
            raise ValueError("environment rotation must be finite")
        if self.image is None:
            return
        image = np.array(self.image, dtype=np.float32, copy=True, order="C")
        if image.ndim != 3 or image.shape[2] not in (3, 4):
            raise ValueError(
                "environment image must have shape (height, width, 3 or 4)"
            )
        if image.shape[0] < 1 or image.shape[1] < 1:
            raise ValueError("environment image dimensions must be positive")
        image = image[..., :3]
        if not np.all(np.isfinite(image)) or np.any(image < 0.0):
            raise ValueError(
                "environment image radiance must be finite and non-negative"
            )
        image = np.ascontiguousarray(image, dtype=np.float32)
        image.flags.writeable = False
        object.__setattr__(self, "image", image)
