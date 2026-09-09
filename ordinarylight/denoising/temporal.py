"""Shared native temporal denoising policy ABI."""

import math
from operator import index
import struct


def temporal_constants(
    width,
    height,
    history_valid,
    *,
    history_limit=32,
    normal_threshold=0.8,
    depth_threshold=0.02,
    clamp_sigma=2.5,
    reactive_sigma=0.0,
):
    width, height = index(width), index(height)
    if min(width, height) <= 0:
        raise ValueError("Temporal extent must be positive")
    values = tuple(
        map(
            float,
            (
                history_limit,
                normal_threshold,
                depth_threshold,
                clamp_sigma,
                reactive_sigma,
            ),
        )
    )
    if not all(map(math.isfinite, values)):
        raise ValueError("Temporal settings must be finite")
    limit, normal, depth, clamp, reactive = values
    if limit < 1 or not -1 <= normal <= 1 or depth < 0 or clamp < 0 or reactive < 0:
        raise ValueError("Invalid temporal rejection settings")
    return struct.pack(
        "8f",
        width,
        height,
        limit,
        float(bool(history_valid)),
        normal,
        depth,
        clamp,
        reactive,
    )
