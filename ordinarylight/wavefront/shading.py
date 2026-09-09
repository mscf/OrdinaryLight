"""Native split-shading buffer and push-constant ABI (version 1).

Sampled texture/volume arrays and custom material descriptor sets are separate
from this buffer contract. These helpers do not create a shading pipeline.
"""

from dataclasses import dataclass
from operator import index
from types import MappingProxyType
import math
import struct

SHADE_BUFFER_BINDINGS = MappingProxyType(
    {
        "hits": 0,
        "rays": 1,
        "paths": 2,
        "material": 3,
        "vertex": 4,
        "attribute": 5,
        "next_rays": 6,
        "media": 7,
        "light": 9,
        "area_light": 10,
        "texture": 11,
        "texture_binding": 12,
        "work_counters": 14,
        "secondary_paths": 15,
        "custom_attribute": 16,
        "volume_header": 17,
        "volume_scalar": 18,
        "volume_transfer": 19,
        "triangle_volume": 20,
    }
)
SHADE_WRITABLE_BUFFERS = frozenset(
    {"paths", "next_rays", "media", "work_counters", "secondary_paths"}
)
_OPTIONAL = frozenset({"work_counters", "custom_attribute"})


def shade_buffer_bindings(buffers):
    """Map named buffers to shader slots; reject missing/unknown bindings.

    Values are opaque to this backend-neutral helper. Optional absent bindings
    are omitted, so the selected shader variant must agree with their presence.
    """
    unknown = buffers.keys() - SHADE_BUFFER_BINDINGS.keys()
    if unknown:
        raise ValueError(f"Unknown shading buffers: {sorted(unknown)}")
    missing = set(SHADE_BUFFER_BINDINGS) - _OPTIONAL - buffers.keys()
    missing |= {
        name for name in buffers if buffers[name] is None and name not in _OPTIONAL
    }
    if missing:
        raise ValueError(f"Missing shading buffers: {sorted(missing)}")
    return MappingProxyType(
        {
            slot: buffers[name]
            for name, slot in SHADE_BUFFER_BINDINGS.items()
            if buffers.get(name) is not None
        }
    )


@dataclass(frozen=True)
class WavefrontShadeSettings:
    max_bounces: int = 4
    light_count: int = 0
    area_light_count: int = 0
    primary_area_samples: int = 1
    secondary_area_samples: int = 1
    area_light_weight: float = 1.0
    environment_samples: int = 1
    roulette_start: int = 0
    roulette_min_survival: float = 0.05
    fused_intersection: bool = False
    subgroup_enqueue: bool = False
    secondary_nee_probability: float = 1.0
    unified_secondary_nee: bool = False
    capture_secondary: bool = False

    def pack(self):
        integers = tuple(
            index(v)
            for v in (
                self.max_bounces,
                self.light_count,
                self.area_light_count,
                self.primary_area_samples,
                self.secondary_area_samples,
                self.environment_samples,
                self.roulette_start,
            )
        )
        if any(v < 0 or v > 0xFFFFFFFF for v in integers) or integers[0] == 0:
            raise ValueError("Invalid shading count")
        if not math.isfinite(self.area_light_weight) or self.area_light_weight < 0:
            raise ValueError("Shading light weight must be finite and nonnegative")
        for value in (self.roulette_min_survival, self.secondary_nee_probability):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Shading probabilities must be in [0, 1]")
        return struct.pack(
            "5IfIIf2If2I",
            *integers[:5],
            self.area_light_weight,
            *integers[5:],
            self.roulette_min_survival,
            int(self.fused_intersection),
            int(self.subgroup_enqueue),
            self.secondary_nee_probability,
            int(self.unified_secondary_nee),
            int(self.capture_secondary),
        )
