"""Device-independent fused-primary and direct-light ReSTIR push constants."""

from dataclasses import dataclass, fields
import math
from operator import index
import struct


@dataclass(frozen=True, kw_only=True)
class PrimarySettings:
    """Named native ABI tail; tile prefix is eight uint32s, effects four pairs.

    Contains values only. Shader capabilities and semantic policy are validated
    by the caller; packing checks representation without changing native policy.
    """

    max_bounces: int
    light_count: int
    area_light_count: int
    primary_area_samples: int
    secondary_area_samples: int
    area_light_weight: float
    write_guides: int
    environment_samples: int
    subgroup_enqueue: int
    roulette_start: int
    roulette_min_survival: float
    secondary_nee_probability: float
    inline_bounces: int
    restir_enabled: int
    restir_history_valid: int
    restir_history_limit: int
    restir_candidates: int
    restir_spatial_reuse: int
    restir_spatial_neighbors: int
    restir_spatial_radius: int
    restir_pairwise_mis: int
    restir_generalized_mis: int
    restir_generalized_balance_cap: float
    unified_secondary_nee: int
    unified_primary_restir: int
    stratified_primary_restir: int
    capture_secondary: int
    capture_stride: int
    effect_ranges: tuple[int, ...] = (0,) * 8

    def pack(self, tile_constants):
        tile_constants = bytes(tile_constants)
        if len(tile_constants) != 32:
            raise ValueError("Primary tile constants require eight uint32s")
        values = []
        float_fields = {
            "area_light_weight",
            "roulette_min_survival",
            "secondary_nee_probability",
            "restir_generalized_balance_cap",
        }
        for field in fields(self):
            if field.name == "effect_ranges":
                continue
            value = getattr(self, field.name)
            if field.name in float_fields:
                value = float(value)
                if not math.isfinite(value) or abs(value) > 3.402823466e38:
                    raise ValueError(f"{field.name} must fit a finite float32")
            else:
                value = index(value)
                if not 0 <= value <= 0xFFFFFFFF:
                    raise ValueError(f"{field.name} must fit uint32")
            values.append(value)
        if len(self.effect_ranges) != 8:
            raise ValueError("Primary effects require four start/end pairs")
        ranges = tuple(index(value) for value in self.effect_ranges)
        if any(not 0 <= value <= 0xFFFFFFFF for value in ranges):
            raise ValueError("Effect ranges must fit uint32")
        return tile_constants + struct.pack("5If4I2f10If13I", *values, *ranges)
