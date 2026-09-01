"""Portable RELAX-inspired denoiser and its deterministic CPU oracle."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .signals import DenoiserSignals


@dataclass(frozen=True)
class PortableDenoiserConfig:
    """Stable, backend-neutral controls for the portable denoiser."""

    max_history_frames: int = 24
    normal_threshold: float = 0.85
    relative_depth_threshold: float = 0.02
    history_clamp_sigma: float = 2.5
    spatial_iterations: int = 2
    diffuse_phi_color: float = 4.0
    specular_phi_color: float = 2.0

    def __post_init__(self):
        if not 1 <= int(self.max_history_frames) <= 255:
            raise ValueError("max_history_frames must be between 1 and 255")
        if not 0.0 <= float(self.normal_threshold) <= 1.0:
            raise ValueError("normal_threshold must be in [0, 1]")
        if float(self.relative_depth_threshold) < 0.0:
            raise ValueError("relative_depth_threshold cannot be negative")
        if float(self.history_clamp_sigma) < 0.0:
            raise ValueError("history_clamp_sigma cannot be negative")
        if not 0 <= int(self.spatial_iterations) <= 5:
            raise ValueError("spatial_iterations must be between 0 and 5")


@dataclass(frozen=True)
class PortableDenoiserResult:
    diffuse: np.ndarray
    specular: np.ndarray
    history_length: np.ndarray
    temporal_acceptance: float

    @property
    def combined(self):
        return self.diffuse + self.specular


def _shift(value, dx, dy):
    """Shift without wrapping and return shifted values plus valid pixels."""
    height, width = value.shape[:2]
    result = np.zeros_like(value)
    valid = np.zeros((height, width), np.bool_)
    sx0, sx1 = max(0, -dx), min(width, width - dx)
    sy0, sy1 = max(0, -dy), min(height, height - dy)
    tx0, tx1 = sx0 + dx, sx1 + dx
    ty0, ty1 = sy0 + dy, sy1 + dy
    if sx1 > sx0 and sy1 > sy0:
        result[ty0:ty1, tx0:tx1] = value[sy0:sy1, sx0:sx1]
        valid[ty0:ty1, tx0:tx1] = True
    return result, valid


def _nearest_reproject(value, motion):
    height, width = value.shape[:2]
    yy, xx = np.indices((height, width), dtype=np.float32)
    px = np.rint(xx + motion[..., 0]).astype(np.int64)
    py = np.rint(yy + motion[..., 1]).astype(np.int64)
    valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
    return value[np.clip(py, 0, height - 1), np.clip(px, 0, width - 1)], valid


def _neighborhood_bounds(color, sigma):
    samples = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            shifted, valid = _shift(color, dx, dy)
            samples.append(np.where(valid[..., None], shifted, color))
    stack = np.stack(samples)
    mean = np.mean(stack, axis=0)
    deviation = np.sqrt(np.maximum(
        np.mean(stack * stack, axis=0) - mean * mean, 0.0
    ))
    return mean - sigma * deviation, mean + sigma * deviation


def _atrous(color, signals, iterations, phi_color):
    result = np.asarray(color, np.float32)
    center_normal = signals.normal_roughness[..., :3]
    center_depth = signals.view_z
    center_material = signals.material_id
    background = center_depth == 0.0
    luminance_weights = np.asarray((0.2126, 0.7152, 0.0722))
    for iteration in range(iterations):
        step = 1 << iteration
        total = result.copy()
        weight_sum = np.ones(result.shape[:2], np.float32)
        center_luma = np.sum(result * luminance_weights, axis=-1)
        for dy, dx, kernel in (
            (-step, 0, 0.5), (step, 0, 0.5),
            (0, -step, 0.5), (0, step, 0.5),
            (-step, -step, 0.25), (-step, step, 0.25),
            (step, -step, 0.25), (step, step, 0.25),
        ):
            sample, valid = _shift(result, dx, dy)
            normal, _ = _shift(center_normal, dx, dy)
            depth, _ = _shift(center_depth, dx, dy)
            material, _ = _shift(center_material, dx, dy)
            sample_luma = np.sum(sample * luminance_weights, axis=-1)
            normal_weight = np.maximum(
                np.sum(center_normal * normal, axis=-1), 0.0
            ) ** 32
            depth_scale = np.maximum(np.abs(center_depth) * 0.02, 1e-3)
            depth_weight = np.exp(-np.abs(depth - center_depth) / depth_scale)
            color_scale = np.maximum(
                np.abs(center_luma) / max(phi_color, 1e-3), 0.02
            )
            color_weight = np.exp(
                -np.abs(sample_luma - center_luma) / color_scale
            )
            weight = kernel * normal_weight * depth_weight * color_weight
            weight *= (
                valid & (material == center_material)
                & ((depth == 0.0) == background)
            )
            total += sample * weight[..., None]
            weight_sum += weight
        result = total / weight_sum[..., None]
    return np.ascontiguousarray(result, np.float32)


class PortableDenoiser:
    """Stateful CPU oracle for Ordinary Light's Ordinary Shade denoiser."""

    def __init__(self, config=None):
        self.config = config or PortableDenoiserConfig()
        self.reset()

    def reset(self):
        self._diffuse = None
        self._specular = None
        self._history_length = None
        self._normal_roughness = None
        self._view_z = None
        self._material_id = None

    def _temporal(self, current, previous, signals):
        if previous is None or signals.frame.camera_cut:
            return current.copy(), np.ones(signals.view_z.shape, np.float32), 0
        history, in_bounds = _nearest_reproject(previous, signals.motion)
        old_normal, _ = _nearest_reproject(
            self._normal_roughness[..., :3], signals.motion
        )
        old_depth, _ = _nearest_reproject(self._view_z, signals.motion)
        old_material, _ = _nearest_reproject(self._material_id, signals.motion)
        old_length, _ = _nearest_reproject(self._history_length, signals.motion)
        normal = signals.normal_roughness[..., :3]
        depth = signals.view_z
        tolerance = np.maximum(
            np.abs(depth) * self.config.relative_depth_threshold, 1e-3
        )
        accepted = (
            in_bounds & (depth != 0.0) & (old_depth != 0.0)
            & (np.sum(normal * old_normal, axis=-1)
               >= self.config.normal_threshold)
            & (np.abs(depth - old_depth) <= tolerance)
            & (signals.material_id == old_material)
        )
        lower, upper = _neighborhood_bounds(
            current, self.config.history_clamp_sigma
        )
        history = np.clip(history, lower, upper)
        length = np.where(
            accepted,
            np.minimum(
                old_length + 1.0, float(self.config.max_history_frames)
            ),
            1.0,
        ).astype(np.float32)
        alpha = np.where(accepted, 1.0 / length, 1.0).astype(np.float32)
        filtered = history * (1.0 - alpha[..., None]) + current * alpha[..., None]
        return (
            np.ascontiguousarray(filtered, np.float32), length,
            int(np.count_nonzero(accepted)),
        )

    def denoise(self, signals):
        if not isinstance(signals, DenoiserSignals):
            raise TypeError("signals must be a DenoiserSignals")
        diffuse, diffuse_length, diffuse_accepted = self._temporal(
            signals.diffuse_radiance_hit_distance[..., :3],
            self._diffuse, signals,
        )
        specular, specular_length, specular_accepted = self._temporal(
            signals.specular_radiance_hit_distance[..., :3],
            self._specular, signals,
        )
        history_length = np.minimum(diffuse_length, specular_length)
        self._diffuse = diffuse.copy()
        self._specular = specular.copy()
        self._history_length = history_length.copy()
        self._normal_roughness = signals.normal_roughness.copy()
        self._view_z = signals.view_z.copy()
        self._material_id = signals.material_id.copy()
        diffuse = _atrous(
            diffuse, signals, self.config.spatial_iterations,
            self.config.diffuse_phi_color,
        )
        specular = _atrous(
            specular, signals, self.config.spatial_iterations,
            self.config.specular_phi_color,
        )
        acceptance = (
            (diffuse_accepted + specular_accepted)
            / max(signals.view_z.size * 2, 1)
        )
        return PortableDenoiserResult(
            diffuse, specular, history_length, float(acceptance)
        )


__all__ = [
    "PortableDenoiser", "PortableDenoiserConfig", "PortableDenoiserResult",
]
