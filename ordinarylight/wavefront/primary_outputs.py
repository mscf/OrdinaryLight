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
