"""Local image-based reflection resource."""

from dataclasses import dataclass, field, replace
import math
import numpy as np


@dataclass(frozen=True)
class ReflectionProbe:
    """A local equirectangular radiance probe used by raster renderers.

    GI renderers intentionally ignore probes and trace the represented scene.
    ``radius`` controls the probe's influence. Render targets may choose the
    nearest containing probe when several probes overlap.
    """

    image: np.ndarray | None = field(default=None, compare=False, repr=False)
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = float("inf")
    intensity: float = 1.0
    rotation: float = 0.0
    projection: str = "sphere"
    box_min: tuple[float, float, float] | None = None
    box_max: tuple[float, float, float] | None = None
    blend_distance: float = 1.0
    priority: int = 0
    refresh_policy: str = "static"
    capture_resolution: int = 256

    def __post_init__(self):
        image = None if self.image is None else np.asarray(
            self.image, dtype=np.float32,
        )
        position = np.asarray(self.position, dtype=np.float32)
        if image is not None and (image.ndim != 3 or image.shape[2] not in (3, 4)):
            raise ValueError("probe image must have shape (height, width, 3 or 4)")
        if image is not None and (
            not np.all(np.isfinite(image)) or np.any(image < 0.0)
        ):
            raise ValueError("probe radiance must be finite and non-negative")
        if position.shape != (3,) or not np.all(np.isfinite(position)):
            raise ValueError("probe position must contain three finite values")
        if self.radius <= 0.0 or math.isnan(self.radius):
            raise ValueError("probe radius must be positive")
        if not math.isfinite(self.intensity) or self.intensity < 0.0:
            raise ValueError("probe intensity must be finite and non-negative")
        if not math.isfinite(self.rotation):
            raise ValueError("probe rotation must be finite")
        if self.projection not in {"sphere", "box"}:
            raise ValueError("probe projection must be 'sphere' or 'box'")
        if self.refresh_policy not in {"static", "on-demand", "scene-change", "always"}:
            raise ValueError(
                "probe refresh_policy must be static, on-demand, scene-change, or always"
            )
        if not math.isfinite(self.blend_distance) or self.blend_distance < 0.0:
            raise ValueError("probe blend_distance must be finite and non-negative")
        if not isinstance(self.priority, int):
            raise TypeError("probe priority must be an integer")
        if not isinstance(self.capture_resolution, int) or self.capture_resolution < 8:
            raise ValueError("probe capture_resolution must be an integer >= 8")
        if self.projection == "box":
            if self.box_min is None or self.box_max is None:
                raise ValueError("box probes require box_min and box_max")
            box_min = np.asarray(self.box_min, dtype=np.float32)
            box_max = np.asarray(self.box_max, dtype=np.float32)
            if (box_min.shape != (3,) or box_max.shape != (3,)
                    or not np.all(np.isfinite(box_min))
                    or not np.all(np.isfinite(box_max))
                    or np.any(box_max <= box_min)):
                raise ValueError("probe box bounds must be finite and box_max > box_min")
            object.__setattr__(self, "box_min", tuple(float(v) for v in box_min))
            object.__setattr__(self, "box_max", tuple(float(v) for v in box_max))
        if image is not None:
            image = np.ascontiguousarray(image[..., :3])
            image.flags.writeable = False
        object.__setattr__(self, "image", image)
        object.__setattr__(self, "position", tuple(float(v) for v in position))

    @property
    def captured(self):
        """Whether this probe currently has radiance available for sampling."""
        return self.image is not None

    def with_image(self, image):
        """Return this immutable probe with newly captured radiance."""
        return replace(self, image=image)
