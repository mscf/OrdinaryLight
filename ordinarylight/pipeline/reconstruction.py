"""Serializable settings for native HDR reconstruction and display conversion."""

from dataclasses import dataclass
from operator import index
import math
import struct

RECONSTRUCT_BASE_FORMAT = "f3If2IfIfIfIIf"
RECONSTRUCT_PUSH_SIZE = 256
FILTERS = {"bilinear": 0, "clamped-cubic": 1, "fsr1": 2, "fsr2": 3, "fsr1-shade": 4}


@dataclass(frozen=True)
class ReconstructionEffect:
    kind: int = 0
    radius: int = 1
    strength: float = 0.0
    object_id: int = 0
    color: tuple = (0.0, 0.0, 0.0, 0.0)
    rect: tuple = (0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ReconstructionSettings:
    exposure: float = 1.0
    temporal_enabled: bool = False
    history_weight: float = 0.9
    diffuse_filter: bool = False
    diffuse_filter_strength: float = 1.0
    variance_confidence: bool = False
    variance_strength: float = 1.0
    material_confidence: bool = False
    transmission_history_scale: float = 1.0
    reprojection_search: bool = False
    outlier_confidence: bool = False
    outlier_strength: float = 1.0
    upscale_filter: str = "bilinear"

    @classmethod
    def from_renderer_config(cls, config):
        return cls(
            config.wavefront_exposure,
            config.wavefront_temporal_reconstruction or config.stationary_accumulation,
            config.wavefront_temporal_weight,
            config.wavefront_diffuse_filter,
            config.wavefront_diffuse_filter_strength,
            config.wavefront_temporal_variance_confidence,
            config.wavefront_temporal_variance_strength,
            config.wavefront_temporal_material_confidence,
            config.wavefront_temporal_transmission_history_scale,
            config.wavefront_temporal_reprojection_search,
            config.wavefront_temporal_outlier_confidence,
            config.wavefront_temporal_outlier_strength,
            config.wavefront_upscale_filter,
        )

    def pack(self, extent, *, history_valid=False, effects=()):
        width, height = map(index, extent)
        if min(width, height) <= 0:
            raise ValueError("Reconstruction extent must be positive")
        if self.upscale_filter not in FILTERS:
            raise ValueError("Unknown reconstruction upscale filter")
        floats = (
            self.exposure,
            self.history_weight,
            self.diffuse_filter_strength,
            self.variance_strength,
            self.transmission_history_scale,
            self.outlier_strength,
        )
        if (
            not all(math.isfinite(v) and v >= 0 for v in floats)
            or self.history_weight > 1
        ):
            raise ValueError("Invalid reconstruction weights")
        if self.upscale_filter in ("fsr1", "fsr1-shade") and (
            self.temporal_enabled or self.diffuse_filter
        ):
            raise ValueError(
                "EASU requires temporal and diffuse reconstruction filters disabled"
            )
        effects = tuple(effects)
        if len(effects) > 4 or not all(
            isinstance(e, ReconstructionEffect) for e in effects
        ):
            raise ValueError("Expected at most four ReconstructionEffect values")
        effects += (ReconstructionEffect(),) * (4 - len(effects))
        for effect in effects:
            if (
                len(effect.color) != 4
                or len(effect.rect) != 4
                or not all(
                    map(math.isfinite, (*effect.color, *effect.rect, effect.strength))
                )
            ):
                raise ValueError(
                    "Effects require finite four-component colors and rectangles"
                )
            if (
                not 0 <= index(effect.kind) <= 6
                or not 0 <= index(effect.radius) <= 0xFFFFFFFF
                or not 0 <= index(effect.object_id) <= 0xFFFFFFFF
            ):
                raise ValueError("Invalid reconstruction effect identifiers")
        result = struct.pack(
            RECONSTRUCT_BASE_FORMAT,
            self.exposure,
            width,
            height,
            int(self.temporal_enabled),
            self.history_weight,
            int(history_valid),
            int(self.diffuse_filter),
            self.diffuse_filter_strength,
            int(self.variance_confidence),
            self.variance_strength,
            int(self.material_confidence),
            self.transmission_history_scale,
            int(self.reprojection_search),
            int(self.outlier_confidence),
            self.outlier_strength,
        )
        result += struct.pack("I", FILTERS[self.upscale_filter])
        result += b"".join(
            struct.pack("IIfI", e.kind, e.radius, e.strength, e.object_id)
            for e in effects
        )
        result += struct.pack("16f", *(v for e in effects for v in e.color))
        result += struct.pack("16f", *(v for e in effects for v in e.rect))
        assert len(result) == RECONSTRUCT_PUSH_SIZE
        return result
