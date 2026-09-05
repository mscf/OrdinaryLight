"""Prepare arbitrary world-space rays or known surfaces, independent of pixels."""

import numpy as np
from . import SURFACE_SAMPLE_DTYPE


def ray_samples(origins, directions, *, identities=None):
    origins = np.asarray(origins, np.float32)
    directions = np.asarray(directions, np.float32)
    if (
        origins.ndim != 2
        or origins.shape[1] != 3
        or origins.shape != directions.shape
        or not len(origins)
        or not np.isfinite(origins).all()
        or not np.isfinite(directions).all()
        or not np.allclose(np.linalg.norm(directions, axis=1), 1, rtol=1e-5)
    ):
        raise ValueError("Supply finite (N,3) positions and unit directions")
    ids = np.arange(len(origins)) if identities is None else np.asarray(identities)
    if (
        ids.shape != (len(origins),)
        or ids.dtype.kind not in "iu"
        or np.any(ids < 0)
        or np.any(ids >= 0xFFFFFFFF)
        or len(np.unique(ids)) != len(ids)
    ):
        raise ValueError("Sample identities must be unique nonnegative uint32 indices")
    samples = np.zeros(len(origins), SURFACE_SAMPLE_DTYPE)
    samples["position"][:, :3] = origins
    samples["incoming"][:, :3] = directions
    samples["identity"][:, 0] = ids
    samples["media"][:, 2] = 0xFFFFFFFF
    return samples


def surface_samples(
    positions,
    geometric_normals,
    *,
    materials,
    identities=None,
    shading_normals=None,
    incoming=None,
    boundaries=None,
):
    normals = np.asarray(geometric_normals, np.float32)
    samples = ray_samples(
        positions, -normals if incoming is None else incoming, identities=identities
    )
    shading = (
        normals if shading_normals is None else np.asarray(shading_normals, np.float32)
    )
    for name, values in [("geometric", normals), ("shading", shading)]:
        if (
            values.shape != (len(samples), 3)
            or not np.isfinite(values).all()
            or not np.allclose(np.linalg.norm(values, axis=1), 1, rtol=1e-5)
        ):
            raise ValueError(f"{name} normals must be finite unit vectors")
    if np.any(np.sum(normals * shading, axis=1) <= 0):
        raise ValueError("Shading normals must lie in the geometric hemisphere")
    materials = np.broadcast_to(np.asarray(materials), (len(samples),))
    if (
        materials.dtype.kind not in "iu"
        or np.any(materials < 0)
        or np.any(materials >= 0xFFFFFFFF)
    ):
        raise ValueError("Material indices must be uint32 values")
    samples["geometric_normal"][:, :3] = normals
    samples["shading_normal"][:, :3] = shading
    samples["identity"][:, 2] = materials
    samples["identity"][:, 3] = 1
    if boundaries is not None:
        ids = np.broadcast_to(np.asarray(boundaries), (len(samples),))
        if ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= 0xFFFFFFFF):
            raise ValueError("Boundary identities must be uint32 values")
        samples["media"][:, 2] = ids
    return samples
