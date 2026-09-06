"""Array-backed declarations for arbitrary custom intersection programs."""

from collections.abc import Sequence
from operator import index

import numpy as np

from .intersection import CustomGeometry, IntersectionProgram


def _snapshot(value):
    """Immutable owned storage, including against setflags(write=True)."""
    return np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)


class CustomGeometryBatch(Sequence):
    """A snapshot of N custom primitives sharing one intersection program.

    Bounds are (N, 2, 3). Parameters broadcast from (4,) to (N, 4).
    Material, boundary and application IDs are uint32 scalars or (N,) arrays;
    boundary 0xFFFFFFFF means no boundary. Materials index custom_materials,
    whereas boundaries are application boundary IDs, not scene indices.
    Indexing lazily constructs a CustomGeometry; scene upload uses the arrays.
    """

    def __init__(
        self,
        bounds,
        program,
        parameters=(0, 0, 0, 0),
        *,
        materials=0,
        boundaries=None,
        identities=0,
    ):
        if not isinstance(program, IntersectionProgram):
            raise TypeError("Expected IntersectionProgram")
        with np.errstate(over="ignore", invalid="ignore"):
            bounds = np.asarray(bounds, dtype=np.float32)
            if (
                bounds.ndim != 3
                or bounds.shape[1:] != (2, 3)
                or not len(bounds)
                or not np.isfinite(bounds).all()
                or np.any(bounds[:, 1] <= bounds[:, 0])
            ):
                raise ValueError(
                    "Bounds must be finite nondegenerate float32 (N, 2, 3)"
                )
            parameters = np.broadcast_to(
                np.asarray(parameters, dtype=np.float32), (len(bounds), 4)
            )
            if not np.isfinite(parameters).all():
                raise ValueError("Parameters must be finite float32 values")
        self._program = program
        self._bounds = _snapshot(bounds)
        self._parameters = _snapshot(parameters)

        def ids(value):
            raw = np.asarray(value)
            if (
                raw.dtype.kind not in "iu"
                or np.any(raw < 0)
                or np.any(raw > 0xFFFFFFFF)
            ):
                raise ValueError("Identifiers must contain uint32 values")
            return _snapshot(np.broadcast_to(raw, (len(bounds),)).astype(np.uint32))

        self._materials = ids(materials)
        self._boundaries = ids(0xFFFFFFFF if boundaries is None else boundaries)
        self._identities = ids(identities)

    program = property(lambda self: self._program)
    bounds = property(lambda self: self._bounds)
    parameters = property(lambda self: self._parameters)
    materials = property(lambda self: self._materials)
    boundaries = property(lambda self: self._boundaries)
    identities = property(lambda self: self._identities)

    def __len__(self):
        return len(self.bounds)

    def __getitem__(self, slot):
        if isinstance(slot, slice):
            return tuple(self[i] for i in range(*slot.indices(len(self))))
        slot = index(slot)
        boundary = int(self.boundaries[slot])
        return CustomGeometry(
            self.bounds[slot],
            self.program,
            self.parameters[slot],
            material=int(self.materials[slot]),
            boundary=None if boundary == 0xFFFFFFFF else boundary,
            identity=int(self.identities[slot]),
        )
