"""Dielectric boundary semantics and an independent numerical reference."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OpticalMedium:
    ior: float = 1.0
    absorption: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self):
        absorption = np.asarray(self.absorption, dtype=np.float64)
        if not np.isfinite(self.ior) or self.ior <= 0:
            raise ValueError("Medium IOR must be finite and positive")
        if (
            absorption.shape != (3,)
            or not np.isfinite(absorption).all()
            or np.any(absorption < 0)
        ):
            raise ValueError(
                "Absorption must be three finite nonnegative coefficients per world unit"
            )
        object.__setattr__(self, "absorption", tuple(absorption))

    def transmittance(self, distance):
        if not np.isfinite(distance) or distance < 0:
            raise ValueError("Optical path length must be finite and nonnegative")
        return np.exp(-np.asarray(self.absorption) * distance)


@dataclass(frozen=True)
class MediumBoundary:
    identity: int
    outside: int
    inside: int

    def __post_init__(self):
        if any(
            not isinstance(value, int) or not 0 <= value < 0xFFFFFFFF
            for value in (self.identity, self.outside, self.inside)
        ):
            raise ValueError("Boundary and medium identities must be uint32 values")
        if self.outside == self.inside:
            raise ValueError("A medium boundary must separate distinct media")


class MediumStack:
    """Strictly nested boundary stack; non-LIFO overlapping exits are rejected."""

    def __init__(self, *, capacity=8):
        if capacity < 2:
            raise ValueError("Medium stack needs capacity for exterior and interior")
        self.capacity = capacity
        self.media = [0]
        self.boundaries = []

    def target(self, boundary, entering):
        if entering:
            if (
                self.media[-1] != boundary.outside
                or boundary.identity in self.boundaries
            ):
                raise ValueError("Invalid overlapping or repeated medium entry")
            return boundary.inside
        if (
            not self.boundaries
            or self.boundaries[-1] != boundary.identity
            or self.media[-1] != boundary.inside
            or self.media[-2] != boundary.outside
        ):
            raise ValueError("Invalid non-nested medium exit")
        return boundary.outside

    def transmit(self, boundary, entering):
        target = self.target(boundary, entering)
        if entering:
            if len(self.media) >= self.capacity:
                raise ValueError("Medium stack capacity exceeded")
            self.media.append(target)
            self.boundaries.append(boundary.identity)
        else:
            self.media.pop()
            self.boundaries.pop()


@dataclass(frozen=True)
class DielectricEvent:
    direction: object
    reflected: bool
    total_internal_reflection: bool
    fresnel: float
    throughput: float


def dielectric_event(direction, geometric_normal, eta_i, eta_t, random_value):
    """Exact unpolarized Fresnel and radiance-mode eta-squared transmission.

    The normal must point toward the incident medium. The sampled branch's
    Fresnel probability cancels its BSDF weight; reflection has unit throughput.
    Stack changes are separate and occur only for successful transmission.
    """
    direction = np.asarray(direction, dtype=np.float64)
    normal = np.asarray(geometric_normal, dtype=np.float64)
    if (
        direction.shape != (3,)
        or normal.shape != (3,)
        or not np.isfinite(direction).all()
        or not np.isfinite(normal).all()
    ):
        raise ValueError("Direction and normal must be finite vectors")
    if not np.isclose(np.linalg.norm(direction), 1) or not np.isclose(
        np.linalg.norm(normal), 1
    ):
        raise ValueError("Direction and normal must be normalized")
    if (
        not np.isfinite([eta_i, eta_t, random_value]).all()
        or min(eta_i, eta_t) <= 0
        or not 0 <= random_value < 1
    ):
        raise ValueError("Invalid dielectric parameters")
    cosine = float(-np.dot(direction, normal))
    if cosine < -1e-10:
        raise ValueError("Geometric normal must face the incident direction")
    cosine = np.clip(cosine, 0, 1)
    eta = eta_i / eta_t
    sin_t2 = eta**2 * max(0, 1 - cosine**2)
    tir = sin_t2 >= 1
    cosine_t = np.sqrt(max(0, 1 - sin_t2))
    if tir:
        fresnel = 1.0
    elif eta_i == eta_t:
        fresnel = 0.0
    else:
        rs = (eta_i * cosine - eta_t * cosine_t) / max(
            eta_i * cosine + eta_t * cosine_t, 1e-30
        )
        rp = (eta_t * cosine - eta_i * cosine_t) / max(
            eta_t * cosine + eta_i * cosine_t, 1e-30
        )
        fresnel = float((rs * rs + rp * rp) * 0.5)
    reflected = tir or random_value < fresnel
    result = (
        direction + 2 * cosine * normal
        if reflected
        else eta * direction + (eta * cosine - cosine_t) * normal
    )
    result /= np.linalg.norm(result)
    return DielectricEvent(
        result, reflected, bool(tir), fresnel, 1.0 if reflected else eta * eta
    )
