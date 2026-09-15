"""Per-sample primary-hit buffer ABI, written by the radiance camera dispatch."""

import numpy as np

PRIMARY_HIT_DTYPE = np.dtype([
    ("position_distance", "<f4", (4,)),
    ("geometric_normal", "<f4", (4,)),
    ("shading_normal", "<f4", (4,)),
    ("identity", "<u4", (4,)),
    ("ray_origin", "<f4", (4,)),
    ("ray_direction", "<f4", (4,)),
])

PRIMARY_HIT_IDENTITY_DTYPE = np.dtype([("identity", "<u4", (4,)), ("valid", "<u4")])


def primary_hit_dtype(format="full"):
    """Return the public sampled-hit record ABI for an export format."""
    if format == "full":
        return PRIMARY_HIT_DTYPE
    if format == "identity":
        return PRIMARY_HIT_IDENTITY_DTYPE
    raise ValueError("primary hit format must be full or identity")

# Complete NativeIntersection payload; unlike public guide exports this retains
# custom addresses, texture coordinates and previous positions for shading.
PRIMARY_VISIBILITY_DTYPE = np.dtype([
    ("position_distance", "<f4", (4,)),
    ("geometric_normal", "<f4", (4,)),
    ("shading_normal", "<f4", (4,)),
    ("identity", "<u4", (4,)),
    ("address", "<u4", (4,)),
    ("texcoord", "<f4", (4,)),
    ("previous_position", "<f4", (4,)),
])


# Experimental scalar-packed cache. Position XYZ is recomputed from the same
# sampled camera ray; all remaining values retain their original 32-bit words.
PRIMARY_VISIBILITY_DISTANCE_DTYPE = np.dtype(
    [("distance", "<u4")] +
    [(f"{field}_{lane}", "<u4")
     for field in ("geometric_normal", "shading_normal", "identity", "address",
                   "texcoord", "previous_position")
     for lane in "xyzw"]
)


def primary_visibility_dtype(format="full"):
    if format == "full":
        return PRIMARY_VISIBILITY_DTYPE
    if format == "distance":
        return PRIMARY_VISIBILITY_DISTANCE_DTYPE
    raise ValueError("primary visibility format must be full or distance")


def primary_visibility_byte_size(width, height, *, samples=1, format="full"):
    """Required capture/replay cache bytes for a pixel extent and sample count.

    full uses 112-byte records; distance uses 100-byte records. planes stores
    seven vec4 fields in sample/field/pixel order. distance_planes stores six
    vec4 fields (normal, shading normal, identity, address, texcoord, previous
    position) followed by one distance word per pixel. Its distance tail is
    padded to a vec4 boundary separately for each sample. These planar layouts
    do not have a per-pixel NumPy dtype; consumers must use the matching format.
    """
    from operator import index
    width,height,samples=map(index,(width,height,samples))
    if min(width,height,samples)<1:
        raise ValueError("Visibility dimensions and samples must be positive")
    pixels=width*height
    if format=="distance_planes":
        return samples*(pixels*6+(pixels+3)//4)*16
    if format in ("full","planes"):
        return samples*pixels*112
    if format=="distance":
        return samples*pixels*100
    raise ValueError("Unknown visibility format")
