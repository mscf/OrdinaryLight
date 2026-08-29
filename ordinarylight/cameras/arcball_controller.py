"""Toolkit-neutral arcball camera interaction."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .perspective_camera import PerspectiveCamera


@dataclass
class ArcballCameraController:
    """Mutable orbit, pan, and dolly controller for a perspective camera.

    UI integrations provide angular deltas to :meth:`orbit`, normalized view
    deltas to :meth:`pan`, and scroll steps to :meth:`dolly`.  The controller
    deliberately has no dependency on a particular window toolkit.
    """

    target: tuple[float, float, float] = (0.0, 0.0, 0.0)
    distance: float = 5.0
    azimuth: float = 0.0
    elevation: float = 0.0
    vertical_fov_degrees: float = 45.0
    minimum_distance: float = 0.01
    maximum_distance: float = 1.0e6

    def __post_init__(self):
        self.target = self._vector(self.target, "target")
        for name in (
            "distance", "azimuth", "elevation", "vertical_fov_degrees",
            "minimum_distance", "maximum_distance",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        if self.minimum_distance <= 0.0:
            raise ValueError("minimum_distance must be positive")
        if self.maximum_distance < self.minimum_distance:
            raise ValueError("maximum_distance must not be smaller than minimum_distance")
        if not 1.0 <= self.vertical_fov_degrees < 179.0:
            raise ValueError("vertical_fov_degrees must be in [1, 179)")
        self.distance = float(np.clip(
            self.distance, self.minimum_distance, self.maximum_distance,
        ))
        self._clamp_elevation()

    @staticmethod
    def _vector(value, name):
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain three finite values")
        return tuple(float(component) for component in vector)

    @classmethod
    def from_camera(cls, camera: PerspectiveCamera, **options):
        """Construct a controller preserving an existing camera pose."""
        if not isinstance(camera, PerspectiveCamera):
            raise TypeError("camera must be a PerspectiveCamera")
        target = np.asarray(camera.target, dtype=np.float64)
        offset = np.asarray(camera.position, dtype=np.float64) - target
        distance = float(np.linalg.norm(offset))
        if distance <= 0.0:
            raise ValueError("camera position and target must differ")
        azimuth = math.atan2(float(offset[0]), float(offset[2]))
        elevation = math.asin(float(np.clip(offset[1] / distance, -1.0, 1.0)))
        return cls(
            target=tuple(target), distance=distance, azimuth=azimuth,
            elevation=elevation,
            vertical_fov_degrees=camera.vertical_fov_degrees, **options,
        )

    def _clamp_elevation(self):
        limit = math.pi * 0.5 - 1.0e-4
        self.elevation = float(np.clip(self.elevation, -limit, limit))

    def orbit(self, delta_azimuth: float, delta_elevation: float):
        """Orbit around the target by angular deltas in radians."""
        self.azimuth += float(delta_azimuth)
        self.elevation += float(delta_elevation)
        self._clamp_elevation()
        return self

    def dolly(self, steps: float, *, sensitivity: float = 0.12):
        """Move toward the target for positive scroll ``steps``."""
        factor = math.exp(-float(steps) * float(sensitivity))
        self.distance = float(np.clip(
            self.distance * factor, self.minimum_distance,
            self.maximum_distance,
        ))
        return self

    def pan(self, horizontal: float, vertical: float):
        """Pan by fractions of the current view height."""
        camera = self.camera()
        forward = np.asarray(camera.target) - np.asarray(camera.position)
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray(camera.up, dtype=np.float64))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        view_height = 2.0 * self.distance * math.tan(
            math.radians(self.vertical_fov_degrees) * 0.5
        )
        target = np.asarray(self.target) + view_height * (
            -float(horizontal) * right + float(vertical) * up
        )
        self.target = tuple(float(value) for value in target)
        return self

    def camera(self):
        """Return the current immutable Ordinary Light camera."""
        horizontal = self.distance * math.cos(self.elevation)
        position = (
            self.target[0] + horizontal * math.sin(self.azimuth),
            self.target[1] + self.distance * math.sin(self.elevation),
            self.target[2] + horizontal * math.cos(self.azimuth),
        )
        return PerspectiveCamera(
            position, self.target,
            vertical_fov_degrees=self.vertical_fov_degrees,
        )


__all__ = ["ArcballCameraController"]
