"""Render targets shared by all ordinarylight backends."""

from abc import ABC, abstractmethod

import numpy as np


class RenderSurface(ABC):
    """A fixed-resolution destination for RGBA8 render results."""

    def __init__(self, width, height):
        if width <= 0 or height <= 0:
            raise ValueError("surface width and height must be positive")
        self.width = int(width)
        self.height = int(height)

    def validate(self, rgba):
        pixels = np.asarray(rgba)
        expected = (self.height, self.width, 4)
        if pixels.shape != expected:
            raise ValueError(f"surface expected pixels with shape {expected}, got {pixels.shape}")
        if pixels.dtype != np.uint8:
            raise ValueError("surface pixels must use uint8 RGBA components")
        return pixels

    @abstractmethod
    def present(self, rgba):
        """Make a completed RGBA8 frame visible to this surface."""


class ArraySurface(RenderSurface):
    """In-memory surface useful for image processing and tests."""

    def __init__(self, width, height):
        super().__init__(width, height)
        self.pixels = np.zeros((height, width, 4), dtype=np.uint8)

    def present(self, rgba):
        np.copyto(self.pixels, self.validate(rgba))
        return self.pixels
