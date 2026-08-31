"""Target-neutral reflection-probe influence selection."""

from __future__ import annotations

import math
import numpy as np


def _influence(probe, position):
    position = np.asarray(position, np.float32)
    if probe.projection == "box":
        lower = np.asarray(probe.box_min, np.float32)
        upper = np.asarray(probe.box_max, np.float32)
        center = (lower + upper) * 0.5
        extent = np.maximum((upper - lower) * 0.5, 1e-6)
        interior = max(0.0, 1.0 - float(np.max(np.abs(position - center) / extent)))
        outside = np.maximum(np.maximum(lower - position, position - upper), 0.0)
        distance = float(np.linalg.norm(outside))
        if distance == 0.0:
            return max(interior, 1e-6) * (2.0 ** probe.priority)
    else:
        center_distance = float(
            np.linalg.norm(position - np.asarray(probe.position))
        )
        if math.isfinite(probe.radius):
            if center_distance <= probe.radius:
                return max(
                    1e-6, 1.0 - center_distance / probe.radius,
                ) * (2.0 ** probe.priority)
            distance = center_distance - probe.radius
        else:
            distance = 0.0
    if distance > probe.blend_distance and probe.blend_distance > 0.0:
        return 0.0
    falloff = 1.0 if distance == 0.0 else max(
        0.0, 1.0 - distance / max(probe.blend_distance, 1e-6),
    )
    return falloff * (2.0 ** probe.priority)


def select_reflection_probes(probes, position, *, limit=2):
    """Return the strongest captured probes and normalized influence weights."""
    if limit < 1:
        raise ValueError("probe selection limit must be positive")
    ranked = sorted(
        ((probe, _influence(probe, position)) for probe in probes if probe.captured),
        key=lambda item: item[1], reverse=True,
    )[:limit]
    total = sum(weight for _probe, weight in ranked)
    if total <= 0.0:
        ranked = [(probe, 1.0) for probe in probes if probe.captured][:limit]
        total = float(len(ranked))
    return tuple((probe, weight / total) for probe, weight in ranked)
