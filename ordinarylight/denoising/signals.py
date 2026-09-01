"""Renderer-neutral signal contract for spatiotemporal denoisers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class SignalValidationError(ValueError):
    """Raised when a denoiser input violates the canonical contract."""


def _matrix(value, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (4, 4):
        raise SignalValidationError(f"{name} must have shape (4, 4)")
    if not np.isfinite(result).all():
        raise SignalValidationError(f"{name} must contain finite values")
    return np.ascontiguousarray(result)


@dataclass(frozen=True)
class DenoiserFrameInfo:
    """Camera and sequence state shared by portable and reference denoisers.

    Matrices transform world space to clip space. ``jitter`` is expressed in
    output pixels. A camera cut invalidates all temporal history.
    """

    world_to_clip: np.ndarray
    previous_world_to_clip: np.ndarray
    frame_index: int
    jitter: tuple[float, float] = (0.0, 0.0)
    previous_jitter: tuple[float, float] = (0.0, 0.0)
    camera_cut: bool = False

    def __post_init__(self):
        object.__setattr__(self, "world_to_clip", _matrix(
            self.world_to_clip, "world_to_clip"
        ))
        object.__setattr__(self, "previous_world_to_clip", _matrix(
            self.previous_world_to_clip, "previous_world_to_clip"
        ))
        if int(self.frame_index) < 0:
            raise SignalValidationError("frame_index cannot be negative")
        object.__setattr__(self, "frame_index", int(self.frame_index))
        for name in ("jitter", "previous_jitter"):
            value = tuple(float(component) for component in getattr(self, name))
            if len(value) != 2 or not np.isfinite(value).all():
                raise SignalValidationError(f"{name} must contain two finite values")
            object.__setattr__(self, name, value)


def _image(value, name, shape, dtype=np.float32):
    result = np.asarray(value)
    if result.dtype != np.dtype(dtype):
        raise SignalValidationError(f"{name} must use dtype {np.dtype(dtype)}")
    if result.shape != shape:
        raise SignalValidationError(
            f"{name} must have shape {shape}, got {result.shape}"
        )
    if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
        raise SignalValidationError(f"{name} must contain finite values")
    return np.ascontiguousarray(result)


@dataclass(frozen=True)
class DenoiserSignals:
    """Canonical per-frame inputs for diffuse/specular denoising.

    Radiance is linear scene-referred RGB. The fourth component stores the
    in-lobe hit distance for the corresponding diffuse or specular sample.
    ``normal_roughness`` stores a world-space unit normal and perceptual
    roughness. ``view_z`` is linear view-space depth. Motion is measured in
    output pixels from the current pixel to its previous-frame location, with
    +X right and +Y down. ``material_id`` is stable across frames.
    """

    diffuse_radiance_hit_distance: np.ndarray
    specular_radiance_hit_distance: np.ndarray
    normal_roughness: np.ndarray
    view_z: np.ndarray
    motion: np.ndarray
    material_id: np.ndarray
    frame: DenoiserFrameInfo

    def __post_init__(self):
        view_z = np.asarray(self.view_z)
        if view_z.ndim != 2:
            raise SignalValidationError("view_z must have shape (height, width)")
        height, width = view_z.shape
        if width < 1 or height < 1:
            raise SignalValidationError("denoiser extent cannot be empty")
        values = {
            "diffuse_radiance_hit_distance": _image(
                self.diffuse_radiance_hit_distance,
                "diffuse_radiance_hit_distance", (height, width, 4),
            ),
            "specular_radiance_hit_distance": _image(
                self.specular_radiance_hit_distance,
                "specular_radiance_hit_distance", (height, width, 4),
            ),
            "normal_roughness": _image(
                self.normal_roughness, "normal_roughness", (height, width, 4),
            ),
            "view_z": _image(view_z, "view_z", (height, width)),
            "motion": _image(self.motion, "motion", (height, width, 2)),
            "material_id": _image(
                self.material_id, "material_id", (height, width), np.uint32
            ),
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        if not isinstance(self.frame, DenoiserFrameInfo):
            raise SignalValidationError("frame must be a DenoiserFrameInfo")
        if np.any(values["diffuse_radiance_hit_distance"][..., 3] < 0.0):
            raise SignalValidationError("diffuse hit distance cannot be negative")
        if np.any(values["specular_radiance_hit_distance"][..., 3] < 0.0):
            raise SignalValidationError("specular hit distance cannot be negative")
        normals = values["normal_roughness"][..., :3]
        lengths = np.linalg.norm(normals, axis=-1)
        foreground = values["view_z"] != 0.0
        if foreground.any() and np.max(np.abs(lengths[foreground] - 1.0)) > 2e-3:
            raise SignalValidationError("foreground normals must be unit length")
        roughness = values["normal_roughness"][..., 3]
        if np.any((roughness < 0.0) | (roughness > 1.0)):
            raise SignalValidationError("roughness must be in [0, 1]")

    @property
    def extent(self):
        return self.view_z.shape[1], self.view_z.shape[0]

    def save(self, path):
        """Write a deterministic, uncompressed reference capture."""
        path = Path(path)
        np.savez(
            path,
            contract_version=np.asarray(1, np.uint32),
            diffuse_radiance_hit_distance=self.diffuse_radiance_hit_distance,
            specular_radiance_hit_distance=self.specular_radiance_hit_distance,
            normal_roughness=self.normal_roughness,
            view_z=self.view_z,
            motion=self.motion,
            material_id=self.material_id,
            world_to_clip=self.frame.world_to_clip,
            previous_world_to_clip=self.frame.previous_world_to_clip,
            frame_index=np.asarray(self.frame.frame_index, np.uint64),
            jitter=np.asarray(self.frame.jitter, np.float32),
            previous_jitter=np.asarray(self.frame.previous_jitter, np.float32),
            camera_cut=np.asarray(self.frame.camera_cut, np.bool_),
        )

    @classmethod
    def load(cls, path):
        """Load and validate a capture written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as data:
            version = int(data["contract_version"])
            if version != 1:
                raise SignalValidationError(
                    f"unsupported denoiser capture contract version {version}"
                )
            frame = DenoiserFrameInfo(
                data["world_to_clip"], data["previous_world_to_clip"],
                int(data["frame_index"]), tuple(data["jitter"]),
                tuple(data["previous_jitter"]), bool(data["camera_cut"]),
            )
            return cls(
                data["diffuse_radiance_hit_distance"],
                data["specular_radiance_hit_distance"],
                data["normal_roughness"], data["view_z"], data["motion"],
                data["material_id"], frame,
            )


__all__ = ["DenoiserFrameInfo", "DenoiserSignals", "SignalValidationError"]
