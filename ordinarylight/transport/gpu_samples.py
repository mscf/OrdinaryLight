"""Reusable GPU-resident transport inputs and explicit output grouping."""

from dataclasses import dataclass
from operator import index

import numpy as np

from . import SURFACE_SAMPLE_DTYPE
from ._synchronization import serialized


def validate_samples(samples, scene=None):
    samples = np.array(samples, copy=True)
    if samples.dtype != SURFACE_SAMPLE_DTYPE or samples.ndim != 1 or not len(samples):
        raise ValueError("Use ray_samples() or surface_samples() to supply samples")
    for field in ("position", "incoming"):
        if not np.isfinite(samples[field]).all():
            raise ValueError("Sample positions/directions must be finite")
    if not np.allclose(
        np.linalg.norm(samples["incoming"][:, :3], axis=1), 1, rtol=1e-5
    ):
        raise ValueError("Sample directions must be unit vectors")
    if np.any(samples["identity"][:, 3] > 1):
        raise ValueError("Unknown sample mode")
    surface = samples["identity"][:, 3] == 1
    for field in ("geometric_normal", "shading_normal"):
        values = samples[field][surface, :3]
        if not np.isfinite(values).all() or not np.allclose(
            np.linalg.norm(values, axis=1), 1, rtol=1e-5
        ):
            raise ValueError("Surface normals must be finite unit vectors")
    if np.any(
        np.sum(
            samples["geometric_normal"][surface, :3]
            * samples["shading_normal"][surface, :3],
            axis=1,
        )
        <= 0
    ):
        raise ValueError("Shading normals must lie in the geometric hemisphere")
    if scene is not None:
        for i in np.flatnonzero(surface):
            material = int(samples["identity"][i, 2])
            if material >= len(scene.materials):
                raise ValueError("Surface sample material is out of range")
            boundary = int(samples["media"][i, 2])
            scene._boundary_index(
                None if boundary == 0xFFFFFFFF else boundary, scene.materials[material]
            )
    return samples


@dataclass(frozen=True)
class SampleReduction:
    """Map each input slot to an output ID; sums/counts produce a path mean.

    Duplicate output IDs are intentional. Grouping is deterministic, with one
    reducer per output and no float atomics. Each input has equal weight.
    """

    output_ids: object

    def __post_init__(self):
        ids = np.asarray(self.output_ids)
        if (
            ids.ndim != 1
            or not len(ids)
            or ids.dtype.kind not in "iu"
            or np.any(ids < 0)
            or np.any(ids >= 0xFFFFFFFF)
        ):
            raise ValueError("Reduction requires nonnegative uint32 output IDs")
        ids = np.array(ids, dtype=np.uint32, copy=True)
        ids.flags.writeable = False
        object.__setattr__(self, "output_ids", ids)

    def pack(self, output_capacity):
        if np.any(self.output_ids >= output_capacity):
            raise ValueError("Reduction output ID exceeds accumulator capacity")
        order = np.argsort(self.output_ids, kind="stable").astype(np.uint32)
        outputs, starts, counts = np.unique(
            self.output_ids[order], return_index=True, return_counts=True
        )
        groups = np.zeros((len(outputs), 4), np.uint32)
        groups[:, 0], groups[:, 1], groups[:, 2] = outputs, starts, counts
        return groups, order


class GpuTransportSamples:
    """Mutable SURFACE_SAMPLE_DTYPE storage with stable allocation capacity.

    GPU producers bind buffer as storage and pass their completion to accumulate.
    Boundary IDs retain application identity in media.z, just like host inputs.
    """

    def __init__(self, runtime, capacity, *, samples=None, count=None):
        with runtime.lock:
            self.runtime = runtime
            self.capacity = index(capacity)
            if not 1 <= self.capacity <= 16_777_216:
                raise ValueError("Invalid sample capacity")
            prepared = None if samples is None else validate_samples(samples)
            self.count = (
                index(count)
                if count is not None
                else len(prepared)
                if prepared is not None
                else self.capacity
            )
            if not 1 <= self.count <= self.capacity or (
                prepared is not None and len(prepared) != self.count
            ):
                raise ValueError("Sample count must fit capacity and supplied data")
            self.closed = False
            self._borrowers = set()
            self.buffer = None
            runtime.retain(self)
            try:
                self.buffer = runtime.buffer(
                    self.capacity * SURFACE_SAMPLE_DTYPE.itemsize,
                    data=np.zeros(self.capacity, SURFACE_SAMPLE_DTYPE),
                )
                if prepared is not None:
                    self.buffer.upload(prepared)
            except Exception:
                self.close()
                raise

    @serialized
    def update(self, samples, *, after=()):
        self.require_open()
        prepared = validate_samples(samples)
        if len(prepared) > self.capacity:
            raise ValueError("Sample update exceeds allocation capacity")
        self._wait(after)
        self.buffer.upload(prepared)
        self.count = len(prepared)

    @serialized
    def set_count(self, count, *, after=()):
        self.require_open()
        count = index(count)
        if not 1 <= count <= self.capacity:
            raise ValueError("Sample count must fit capacity")
        self._wait(after)
        self.count = count

    def _wait(self, after):
        from ..runtime.resources import VulkanCompletion

        dependencies = tuple(after)
        if any(
            not isinstance(d, VulkanCompletion) or d.runtime is not self.runtime
            for d in dependencies
        ):
            raise ValueError("Sample dependencies must belong to this runtime")
        for dependency in dependencies:
            dependency.wait()

    def require_open(self):
        self.runtime.require_open()
        if self.closed:
            raise RuntimeError("GPU samples are closed")
        if self.buffer is not None:
            self.buffer.require_open()

    @serialized
    def close(self):
        if self.closed:
            return
        if self._borrowers:
            raise RuntimeError("Close transport integrators before GPU samples")
        if self.buffer is not None:
            self.buffer.close()
        self.closed = True
        self.runtime.release(self)

    def __enter__(self):
        self.require_open()
        return self

    def __exit__(self, *_exc):
        self.close()
