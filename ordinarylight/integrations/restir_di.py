"""Reference weighted reservoirs for direct-light sample reuse.

The Vulkan implementation mirrors this small CPU oracle.  Keeping the
selection math independent from scene traversal makes temporal and spatial
reuse testable without a GPU and keeps ReSTIR-DI composable with the existing
light sampler.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DirectLightSample:
    """One point sampled from an emissive triangle."""

    light_index: int
    barycentric_u: float
    barycentric_v: float
    target: float

    def __post_init__(self):
        if self.light_index < 0:
            raise ValueError("light_index cannot be negative")
        if not math.isfinite(self.target) or self.target < 0.0:
            raise ValueError("target must be finite and non-negative")


@dataclass
class DirectLightReservoir:
    """Streaming weighted reservoir with ReSTIR-compatible merge semantics."""

    sample: DirectLightSample | None = None
    weight_sum: float = 0.0
    sample_count: int = 0

    def clear(self):
        self.sample = None
        self.weight_sum = 0.0
        self.sample_count = 0

    def update(
        self,
        sample: DirectLightSample,
        weight: float,
        random_value: float,
        *,
        represented_samples: int = 1,
    ) -> bool:
        """Stream a candidate and return whether it became representative."""
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError("weight must be finite and non-negative")
        if not 0.0 <= random_value < 1.0:
            raise ValueError("random_value must be in [0, 1)")
        if represented_samples < 1:
            raise ValueError("represented_samples must be positive")
        self.sample_count += represented_samples
        self.weight_sum += weight
        selected = weight > 0.0 and (
            self.sample is None
            or random_value * self.weight_sum < weight
        )
        if selected:
            self.sample = sample
        return selected

    def merge(
        self,
        source: "DirectLightReservoir",
        target_at_current_surface: float,
        random_value: float,
    ) -> bool:
        """Reuse a reservoir after reevaluating its sample at this surface."""
        if source.sample is None or source.sample_count == 0:
            return False
        if not math.isfinite(target_at_current_surface) or target_at_current_surface < 0:
            raise ValueError("target_at_current_surface must be finite and non-negative")
        source_target = source.sample.target
        reuse_weight = 0.0
        if source_target > 0.0:
            reuse_weight = (
                target_at_current_surface * source.weight_sum / source_target
            )
        reused = DirectLightSample(
            source.sample.light_index,
            source.sample.barycentric_u,
            source.sample.barycentric_v,
            target_at_current_surface,
        )
        return self.update(
            reused,
            reuse_weight,
            random_value,
            represented_samples=source.sample_count,
        )

    def merge_canonical(
        self,
        source: "DirectLightReservoir",
        target_at_current_surface: float,
        random_value: float,
    ) -> bool:
        """Reuse one canonical representative from a correlated source."""
        self._validate_reuse(source, target_at_current_surface)
        if source.sample is None or source.sample_count == 0:
            return False
        source_target = source.sample.target
        weight = 0.0
        if source_target > 0.0:
            weight = (
                target_at_current_surface * source.weight_sum
                / (source_target * source.sample_count)
            )
        return self.update(
            self._reused_sample(source, target_at_current_surface),
            weight, random_value,
        )

    def merge_pairwise(
        self,
        source: "DirectLightReservoir",
        target_at_current_surface: float,
        random_value: float,
    ) -> bool:
        """Canonical reuse with a two-proposal balance-heuristic weight."""
        self._validate_reuse(source, target_at_current_surface)
        if source.sample is None or source.sample_count == 0:
            return False
        source_target = source.sample.target
        target_sum = target_at_current_surface + source_target
        balance = 2.0 * source_target / target_sum if target_sum > 0.0 else 0.0
        weight = 0.0
        if source_target > 0.0:
            weight = (
                target_at_current_surface * source.weight_sum
                / (source_target * source.sample_count) * balance
            )
        return self.update(
            self._reused_sample(source, target_at_current_surface),
            weight, random_value,
        )

    @staticmethod
    def _validate_reuse(source, target_at_current_surface):
        if not math.isfinite(target_at_current_surface) or target_at_current_surface < 0:
            raise ValueError("target_at_current_surface must be finite and non-negative")

    @staticmethod
    def _reused_sample(source, target_at_current_surface):
        return DirectLightSample(
            source.sample.light_index,
            source.sample.barycentric_u,
            source.sample.barycentric_v,
            target_at_current_surface,
        )

    @property
    def normalization(self) -> float:
        """Final unbiased multiplier W = weight_sum / (M * selected target)."""
        if self.sample is None or self.sample_count == 0 or self.sample.target <= 0.0:
            return 0.0
        return self.weight_sum / (self.sample_count * self.sample.target)
