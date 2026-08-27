"""GPU-time-driven sample-budget selection for interactive rendering."""

from dataclasses import dataclass
import math


@dataclass
class DynamicSampleController:
    """Choose the highest integer SPP predicted to fit a GPU-time budget."""

    target_ms: float = 16.67
    minimum_samples: int = 1
    maximum_samples: int = 1
    current_samples: int = 1
    hysteresis: float = 0.08
    smoothing: float = 0.2
    recovery_smoothing: float = 0.05
    update_interval: int = 4

    def __post_init__(self):
        if self.target_ms <= 0.0:
            raise ValueError("target_ms must be positive")
        if not 1 <= self.minimum_samples <= self.maximum_samples:
            raise ValueError("samples must satisfy 1 <= minimum <= maximum")
        if not self.minimum_samples <= self.current_samples <= self.maximum_samples:
            raise ValueError("current_samples must be within the configured range")
        if not 0.0 <= self.hysteresis < 1.0:
            raise ValueError("hysteresis must be in [0, 1)")
        if not 0.0 < self.smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        if not 0.0 < self.recovery_smoothing <= 1.0:
            raise ValueError("recovery_smoothing must be in (0, 1]")
        if self.update_interval < 1:
            raise ValueError("update_interval must be positive")
        self.filtered_full_scale_sample_ms = 0.0
        self.filtered_gpu_ms = 0.0
        self.sample_count = 0

    def update(
        self, gpu_ms, observed_samples, observed_scale=1.0, *,
        selected_scale=None, allow_increase=True,
    ):
        """Observe one completed frame and return the next motion SPP."""
        gpu_ms = float(gpu_ms)
        observed_samples = int(observed_samples)
        observed_scale = float(observed_scale)
        selected_scale = (
            observed_scale if selected_scale is None else float(selected_scale)
        )
        if (
            not math.isfinite(gpu_ms) or gpu_ms <= 0.0
            or observed_samples < 1
            or not math.isfinite(observed_scale) or observed_scale <= 0.0
            or not math.isfinite(selected_scale) or selected_scale <= 0.0
        ):
            return self.current_samples
        sample_ms = gpu_ms / (
            observed_samples * observed_scale * observed_scale
        )
        if self.filtered_full_scale_sample_ms <= 0.0:
            self.filtered_full_scale_sample_ms = sample_ms
        else:
            smoothing = (
                self.smoothing
                if sample_ms >= self.filtered_full_scale_sample_ms
                else self.recovery_smoothing
            )
            self.filtered_full_scale_sample_ms += smoothing * (
                sample_ms - self.filtered_full_scale_sample_ms
            )
        if not allow_increase:
            self.current_samples = self.minimum_samples
            self.filtered_gpu_ms = (
                self.filtered_full_scale_sample_ms
                * selected_scale * selected_scale * self.current_samples
            )
            return self.current_samples
        self.sample_count += 1
        predicted_ms = (
            self.filtered_full_scale_sample_ms
            * selected_scale * selected_scale * self.current_samples
        )
        self.filtered_gpu_ms = predicted_ms
        if self.sample_count % self.update_interval:
            return self.current_samples
        target_with_margin = self.target_ms * (1.0 - self.hysteresis)
        ideal = int(
            target_with_margin / max(
                self.filtered_full_scale_sample_ms
                * selected_scale * selected_scale,
                1e-6,
            )
        )
        ideal = max(self.minimum_samples, min(self.maximum_samples, ideal))
        if ideal < self.current_samples:
            self.current_samples = ideal
        elif ideal > self.current_samples:
            self.current_samples += 1
        self.filtered_gpu_ms = (
            self.filtered_full_scale_sample_ms
            * selected_scale * selected_scale * self.current_samples
        )
        return self.current_samples
