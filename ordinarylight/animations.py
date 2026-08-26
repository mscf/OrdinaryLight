"""Backend-neutral keyframe animation resources and playback."""

from dataclasses import dataclass, field
import math

import numpy as np


def decompose_matrix(matrix):
    """Return translation, xyzw rotation, and scale from an affine matrix."""
    matrix = np.asarray(matrix, np.float64)
    translation = matrix[:3, 3].copy()
    scale = np.linalg.norm(matrix[:3, :3], axis=0)
    rotation = matrix[:3, :3] / scale[None, :]
    if np.linalg.det(rotation) < 0.0:
        scale[0] *= -1.0
        rotation[:, 0] *= -1.0
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.asarray((
            (rotation[2, 1] - rotation[1, 2]) / s,
            (rotation[0, 2] - rotation[2, 0]) / s,
            (rotation[1, 0] - rotation[0, 1]) / s,
            0.25 * s,
        ))
    else:
        index = int(np.argmax(np.diag(rotation)))
        if index == 0:
            s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1]
                          - rotation[2, 2]) * 2.0
            quaternion = np.asarray((
                0.25 * s, (rotation[0, 1] + rotation[1, 0]) / s,
                (rotation[0, 2] + rotation[2, 0]) / s,
                (rotation[2, 1] - rotation[1, 2]) / s,
            ))
        elif index == 1:
            s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0]
                          - rotation[2, 2]) * 2.0
            quaternion = np.asarray((
                (rotation[0, 1] + rotation[1, 0]) / s, 0.25 * s,
                (rotation[1, 2] + rotation[2, 1]) / s,
                (rotation[0, 2] - rotation[2, 0]) / s,
            ))
        else:
            s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0]
                          - rotation[1, 1]) * 2.0
            quaternion = np.asarray((
                (rotation[0, 2] + rotation[2, 0]) / s,
                (rotation[1, 2] + rotation[2, 1]) / s, 0.25 * s,
                (rotation[1, 0] - rotation[0, 1]) / s,
            ))
    quaternion /= np.linalg.norm(quaternion)
    return tuple(translation), tuple(quaternion), tuple(scale)


def compose_matrix(translation, rotation, scale):
    """Compose an affine matrix from translation, xyzw rotation, and scale."""
    x, y, z, w = np.asarray(rotation, np.float64)
    quaternion_length = math.sqrt(x*x + y*y + z*z + w*w)
    if quaternion_length < 1e-12:
        raise ValueError("animation rotation quaternion cannot be zero")
    x, y, z, w = (x/quaternion_length, y/quaternion_length,
                  z/quaternion_length, w/quaternion_length)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = np.asarray((
        (1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w),
        (2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w),
        (2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y),
    )) @ np.diag(np.asarray(scale, np.float64))
    matrix[:3, 3] = translation
    return matrix


def _quaternion_slerp(first, second, weight):
    first = np.asarray(first, np.float64)
    second = np.asarray(second, np.float64)
    first /= np.linalg.norm(first)
    second /= np.linalg.norm(second)
    cosine = float(np.dot(first, second))
    if cosine < 0.0:
        second = -second
        cosine = -cosine
    if cosine > 0.9995:
        value = first + weight * (second - first)
        return value / np.linalg.norm(value)
    angle = math.acos(np.clip(cosine, -1.0, 1.0))
    sine = math.sin(angle)
    return (
        math.sin((1.0 - weight) * angle) / sine * first
        + math.sin(weight * angle) / sine * second
    )


@dataclass(frozen=True)
class AnimationTrack:
    """Keyframes for one property on one or more scene resource handles."""

    targets: tuple[object, ...] | object
    property: str
    times: np.ndarray = field(compare=False, repr=False)
    values: np.ndarray = field(compare=False, repr=False)
    interpolation: str = "linear"

    def __post_init__(self):
        targets = self.targets if isinstance(self.targets, tuple) else (self.targets,)
        if not targets:
            raise ValueError("animation track requires at least one target")
        if not isinstance(self.property, str) or not self.property:
            raise ValueError("animation property must be a nonempty string")
        times = np.array(self.times, dtype=np.float32, copy=True)
        values = np.array(self.values, dtype=np.float32, copy=True)
        if times.ndim != 1 or len(times) < 1:
            raise ValueError("animation times must be a nonempty 1D array")
        if not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0.0):
            raise ValueError("animation times must be finite and strictly increasing")
        interpolation = self.interpolation.lower()
        if interpolation not in {"linear", "step", "cubic"}:
            raise ValueError("interpolation must be linear, step, or cubic")
        expected = len(times) * (3 if interpolation == "cubic" else 1)
        if values.ndim < 2 or len(values) != expected:
            raise ValueError(
                f"animation values require {expected} rows for {interpolation} interpolation"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("animation values must be finite")
        times.flags.writeable = False
        values.flags.writeable = False
        object.__setattr__(self, "targets", tuple(targets))
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "interpolation", interpolation)

    @property
    def start_time(self):
        return float(self.times[0])

    @property
    def end_time(self):
        return float(self.times[-1])

    def sample(self, time):
        """Sample this track at absolute clip time."""
        time = float(time)
        if not math.isfinite(time):
            raise ValueError("animation time must be finite")
        if time <= self.times[0]:
            index = 0
            return np.array(
                self.values[index * 3 + 1] if self.interpolation == "cubic"
                else self.values[index], copy=True,
            )
        if time >= self.times[-1]:
            index = len(self.times) - 1
            return np.array(
                self.values[index * 3 + 1] if self.interpolation == "cubic"
                else self.values[index], copy=True,
            )
        upper = int(np.searchsorted(self.times, time, side="right"))
        lower = upper - 1
        duration = float(self.times[upper] - self.times[lower])
        weight = (time - float(self.times[lower])) / duration
        if self.interpolation == "step":
            return np.array(self.values[lower], copy=True)
        if self.interpolation == "cubic":
            first = self.values[lower * 3 + 1]
            first_tangent = self.values[lower * 3 + 2] * duration
            second = self.values[upper * 3 + 1]
            second_tangent = self.values[upper * 3] * duration
            w2 = weight * weight
            w3 = w2 * weight
            return np.asarray(
                (2*w3 - 3*w2 + 1) * first
                + (w3 - 2*w2 + weight) * first_tangent
                + (-2*w3 + 3*w2) * second
                + (w3 - w2) * second_tangent,
                np.float32,
            )
        if self.property == "rotation" and self.values.shape[1:] == (4,):
            return np.asarray(_quaternion_slerp(
                self.values[lower], self.values[upper], weight
            ), np.float32)
        return np.asarray(
            self.values[lower] * (1.0 - weight) + self.values[upper] * weight,
            np.float32,
        )


@dataclass
class AnimationClip:
    """Named collection of property tracks sharing one playback timeline."""

    tracks: tuple[AnimationTrack, ...] | list[AnimationTrack]
    name: str | None = None

    def __post_init__(self):
        self.tracks = tuple(self.tracks)
        if not self.tracks or any(
            not isinstance(track, AnimationTrack) for track in self.tracks
        ):
            raise ValueError("animation clip requires AnimationTrack objects")
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError("animation name must be a string or None")

    @property
    def start_time(self):
        return min(track.start_time for track in self.tracks)

    @property
    def end_time(self):
        return max(track.end_time for track in self.tracks)

    @property
    def duration(self):
        return self.end_time - self.start_time

    def normalized_time(self, time, *, loop=False):
        time = float(time)
        if loop and self.duration > 0.0:
            return self.start_time + (time - self.start_time) % self.duration
        return min(max(time, self.start_time), self.end_time)

    def sample(self, time, *, loop=False):
        """Return ``(target, property, value)`` samples in track order."""
        time = self.normalized_time(time, loop=loop)
        return tuple(
            (target, track.property, track.sample(time))
            for track in self.tracks for target in track.targets
        )


@dataclass
class AnimationPlayer:
    """Stateful convenience controller for a clip attached to a scene."""

    scene: object
    clip: AnimationClip
    time: float = 0.0
    speed: float = 1.0
    loop: bool = True
    playing: bool = True

    def seek(self, time):
        self.time = float(time)
        self.scene.apply_animation(self.clip, self.time, loop=self.loop)
        return self.time

    def update(self, delta_time):
        if self.playing:
            self.time += float(delta_time) * self.speed
            self.scene.apply_animation(self.clip, self.time, loop=self.loop)
        return self.time


@dataclass(frozen=True)
class MorphTarget:
    """Object-space vertex and optional normal deltas for shape animation."""

    position_deltas: np.ndarray = field(compare=False, repr=False)
    normal_deltas: np.ndarray | None = field(
        default=None, compare=False, repr=False
    )
    name: str | None = None

    def __post_init__(self):
        positions = np.array(
            self.position_deltas, dtype=np.float32, copy=True, order="C"
        )
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("position_deltas must have shape (vertex_count, 3)")
        if not np.all(np.isfinite(positions)):
            raise ValueError("position_deltas must be finite")
        normals = self.normal_deltas
        if normals is not None:
            normals = np.array(normals, dtype=np.float32, copy=True, order="C")
            if normals.shape != positions.shape or not np.all(np.isfinite(normals)):
                raise ValueError(
                    "normal_deltas must be finite and match position_deltas"
                )
            normals.flags.writeable = False
        positions.flags.writeable = False
        object.__setattr__(self, "position_deltas", positions)
        object.__setattr__(self, "normal_deltas", normals)


@dataclass(frozen=True)
class Skin:
    """Joint hierarchy and inverse bind matrices shared by skinned instances."""

    joints: tuple[object, ...] | list[object]
    inverse_bind_matrices: np.ndarray = field(compare=False, repr=False)
    name: str | None = None

    def __post_init__(self):
        joints = tuple(self.joints)
        matrices = np.array(
            self.inverse_bind_matrices, dtype=np.float32, copy=True, order="C"
        )
        if not joints:
            raise ValueError("skin requires at least one joint")
        if matrices.shape != (len(joints), 4, 4):
            raise ValueError("inverse_bind_matrices must have shape (joint_count, 4, 4)")
        if not np.all(np.isfinite(matrices)):
            raise ValueError("inverse bind matrices must be finite")
        matrices.flags.writeable = False
        object.__setattr__(self, "joints", joints)
        object.__setattr__(self, "inverse_bind_matrices", matrices)


__all__ = [
    "AnimationClip", "AnimationPlayer", "AnimationTrack", "MorphTarget",
    "Skin",
]
