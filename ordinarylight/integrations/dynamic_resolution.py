"""GPU-time-driven dynamic resolution policy."""

from dataclasses import dataclass
import math


@dataclass
class DynamicResolutionController:
    """Select a stable render scale for a target GPU frame duration."""

    target_ms: float = 16.67
    minimum_scale: float = 0.5
    maximum_scale: float = 1.0
    current_scale: float = 1.0
    hysteresis: float = 0.08
    smoothing: float = 0.2
    recovery_smoothing: float = 0.04
    maximum_step: float = 0.0625
    quantization: float = 1.0 / 32.0
    update_interval: int = 4
    recovery_updates: int = 12

    def __post_init__(self):
        if self.target_ms <= 0.0:
            raise ValueError("target_ms must be positive")
        if not 0.25 <= self.minimum_scale <= self.maximum_scale <= 1.0:
            raise ValueError("scales must satisfy 0.25 <= minimum <= maximum <= 1")
        if not self.minimum_scale <= self.current_scale <= self.maximum_scale:
            raise ValueError("current_scale must be within the configured range")
        if not 0.0 <= self.hysteresis < 1.0:
            raise ValueError("hysteresis must be in [0, 1)")
        if not 0.0 < self.smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        if not 0.0 < self.recovery_smoothing <= 1.0:
            raise ValueError("recovery_smoothing must be in (0, 1]")
        if self.maximum_step <= 0.0 or self.quantization <= 0.0:
            raise ValueError("maximum_step and quantization must be positive")
        if self.update_interval < 1:
            raise ValueError("update_interval must be positive")
        if self.recovery_updates < 1:
            raise ValueError("recovery_updates must be positive")
        self.filtered_gpu_ms = 0.0
        self.filtered_full_scale_ms = 0.0
        self.sample_count = 0
        self.under_budget_updates = 0

    def update(
        self, gpu_ms, sample_scale=None, *, work_units=1.0,
        target_work_units=1.0,
    ):
        """Observe a completed GPU frame and return the selected scale."""
        gpu_ms = float(gpu_ms)
        if not math.isfinite(gpu_ms) or gpu_ms <= 0.0:
            return self.current_scale
        work_units = float(work_units)
        target_work_units = float(target_work_units)
        if (
            not math.isfinite(work_units) or work_units <= 0.0
            or not math.isfinite(target_work_units)
            or target_work_units <= 0.0
        ):
            return self.current_scale
        sample_scale = (
            self.current_scale if sample_scale is None else float(sample_scale)
        )
        sample_scale = max(self.minimum_scale, min(self.maximum_scale, sample_scale))
        full_scale_ms = gpu_ms / (
            sample_scale * sample_scale * work_units
        )
        if self.filtered_full_scale_ms <= 0.0:
            self.filtered_full_scale_ms = full_scale_ms
        else:
            smoothing = (
                self.smoothing if full_scale_ms >= self.filtered_full_scale_ms
                else self.recovery_smoothing
            )
            self.filtered_full_scale_ms += smoothing * (
                full_scale_ms - self.filtered_full_scale_ms
            )
        self.filtered_gpu_ms = (
            self.filtered_full_scale_ms * self.current_scale
            * self.current_scale * target_work_units
        )
        self.sample_count += 1
        if self.sample_count % self.update_interval:
            return self.current_scale

        raw_current_ms = (
            gpu_ms / (sample_scale * sample_scale * work_units)
            * self.current_scale * self.current_scale * target_work_units
        )
        raw_ratio = raw_current_ms / self.target_ms
        if 1.0 - self.hysteresis <= raw_ratio <= 1.0 + self.hysteresis:
            self.under_budget_updates = 0
            return self.current_scale

        ratio = self.filtered_gpu_ms / self.target_ms
        if 1.0 - self.hysteresis <= ratio <= 1.0 + self.hysteresis:
            self.under_budget_updates = 0
            return self.current_scale
        if ratio < 1.0 - self.hysteresis:
            self.under_budget_updates += 1
            if self.under_budget_updates < self.recovery_updates:
                return self.current_scale
        else:
            self.under_budget_updates = 0
        ideal = math.sqrt(
            self.target_ms / max(
                self.filtered_full_scale_ms * target_work_units, 1e-6
            )
        )
        delta = max(
            -self.maximum_step,
            min(self.maximum_step, ideal - self.current_scale),
        )
        selected = self.current_scale + delta
        selected = round(selected / self.quantization) * self.quantization
        self.current_scale = max(
            self.minimum_scale, min(self.maximum_scale, selected)
        )
        self.under_budget_updates = 0
        self.filtered_gpu_ms = (
            self.filtered_full_scale_ms * self.current_scale
            * self.current_scale * target_work_units
        )
        return self.current_scale
