"""Shared dispatch parameters for native ReLAX spatial filtering."""

import math
from operator import index
import struct


def spatial_settings(width, height, iterations, color_weight):
    width, height, iterations = index(width), index(height), index(iterations)
    color_weight = float(color_weight)
    if min(width, height) <= 0:
        raise ValueError("Spatial extent must be positive")
    if not 1 <= iterations <= 5:
        raise ValueError("Spatial iterations must be between 1 and 5")
    if not math.isfinite(color_weight) or color_weight <= 0:
        raise ValueError("Spatial color weight must be finite and positive")
    return width, height, iterations, color_weight


def atrous_constants(width, height, iteration, lobe, color_weight):
    """The shared native 32-byte push-constant ABI; specular clamp on first pass."""
    return struct.pack(
        "8f",
        float(width),
        float(height),
        float(1 << iteration),
        0.0,
        32.0,
        0.02,
        float(color_weight),
        float(lobe == 1 and iteration == 0),
    )


def compose_constants(width, height):
    return struct.pack("4f", float(width), float(height), 0.0, 0.0)
