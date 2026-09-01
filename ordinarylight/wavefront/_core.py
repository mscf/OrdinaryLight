"""Backend-neutral data layouts for wavefront path tracing.

The NumPy dtypes in this module deliberately mirror std430 GLSL structures.
They are an ABI shared by scene-independent queue management, Vulkan storage
buffers, and future CPU/debug implementations.  No Vulkan objects live here.
"""

from dataclasses import dataclass

import numpy as np

from ..pipeline import RenderPipeline, RenderStage


MAX_MEDIUM_STACK_DEPTH = 16


RAY_DTYPE = np.dtype({
    "names": ("origin_tmin", "direction_tmax", "path_index", "padding"),
    "formats": ((np.float32, 4), (np.float32, 4), np.uint32, (np.uint32, 3)),
    "offsets": (0, 16, 32, 36),
    "itemsize": 48,
})
"""One queued ray: two vec4s, its persistent path index, and std430 padding."""


HIT_DTYPE = np.dtype({
    "names": (
        "position_t", "geometric_normal", "primitive_index",
        "barycentrics", "ray_index", "path_index",
    ),
    "formats": (
        (np.float32, 4), (np.float32, 3), np.uint32,
        (np.float32, 2), np.uint32, np.uint32,
    ),
    "offsets": (0, 16, 28, 32, 40, 44),
    "itemsize": 48,
})
"""Intersection result retaining geometric data and queue/path identity."""


PATH_STATE_DTYPE = np.dtype({
    "names": ("throughput", "radiance", "metadata", "rng", "ior_stack"),
    "formats": (
        (np.float32, 4), (np.float32, 4), (np.uint32, 4),
        (np.uint32, 4), (np.float32, MAX_MEDIUM_STACK_DEPTH),
    ),
    "offsets": (0, 16, 32, 48, 64),
    "itemsize": 128,
})
"""Persistent path data; metadata is pixel, sample, bounce, and flags."""


HOT_PATH_STATE_DTYPE = np.dtype({
    "names": ("throughput", "radiance", "metadata"),
    "formats": (
        (np.float32, 4), (np.float32, 4), (np.uint32, 4),
    ),
    "offsets": (0, 16, 32),
    "itemsize": 48,
})
"""Hot path state; bounce/PDF occupy vec4 w lanes and RNG uses metadata.z."""


MEDIUM_STACK_DTYPE = np.dtype({
    "names": ("ior",),
    "formats": ((np.float32, MAX_MEDIUM_STACK_DEPTH),),
    "offsets": (0,),
    "itemsize": 64,
})
"""Cold nested-medium state stored separately from common path data."""


SECONDARY_PATH_STATE_DTYPE = np.dtype({
    "names": (
        "position_valid", "normal_pdf", "primary_throughput",
        "primary_radiance", "diffuse_radiance_hit_distance",
        "specular_radiance_hit_distance", "primary_position",
    ),
    "formats": ((np.float32, 4),) * 7,
    "offsets": (0, 16, 32, 48, 64, 80, 96),
    "itemsize": 112,
})
"""Optional cold path state kept out of the hot path ABI.

The final three vectors are dormant unless denoiser-signal capture is enabled.
Their RGB lanes accumulate demodulated diffuse/specular radiance and their W
lanes retain the corresponding first-event hit distance.  Keeping these
signals here preserves the 48-byte hot path state used by every bounce.
"""


QUEUE_HEADER_DTYPE = np.dtype({
    "names": ("count", "capacity", "overflow", "padding"),
    "formats": (np.uint32, np.uint32, np.uint32, np.uint32),
    "offsets": (0, 4, 8, 12),
    "itemsize": 16,
})
"""Atomic queue header followed immediately by queue records."""


RESOLVED_PIXEL_DTYPE = np.dtype({
    "names": ("radiance", "metadata"),
    "formats": ((np.float32, 4), (np.uint32, 4)),
    "offsets": (0, 16),
    "itemsize": 32,
})
"""Resolved HDR radiance plus pixel/sample/bounce/path flags."""


@dataclass(frozen=True)
class WavefrontQueueLayout:
    """Describes the bytes required by a counter-prefixed GPU work queue."""

    record_dtype: np.dtype
    capacity: int

    def __post_init__(self):
        dtype = np.dtype(self.record_dtype)
        if dtype.itemsize <= 0:
            raise ValueError("queue record dtype cannot be empty")
        if self.capacity < 1:
            raise ValueError("queue capacity must be at least one")
        object.__setattr__(self, "record_dtype", dtype)

    @property
    def data_offset(self):
        return QUEUE_HEADER_DTYPE.itemsize

    @property
    def byte_size(self):
        return self.data_offset + self.capacity * self.record_dtype.itemsize

    def empty_host_buffer(self):
        """Create zeroed host storage with an initialized immutable capacity."""
        storage = np.zeros(self.byte_size, dtype=np.uint8)
        header = storage[:self.data_offset].view(QUEUE_HEADER_DTYPE)
        header["capacity"] = self.capacity
        return storage

    def header_view(self, storage):
        storage = np.asarray(storage)
        if not storage.flags.c_contiguous or storage.nbytes < self.byte_size:
            raise ValueError("queue storage is not contiguous or is too small")
        return storage.view(np.uint8)[:self.data_offset].view(QUEUE_HEADER_DTYPE)

    def records_view(self, storage):
        storage = np.asarray(storage)
        if not storage.flags.c_contiguous or storage.nbytes < self.byte_size:
            raise ValueError("queue storage is not contiguous or is too small")
        raw = storage.view(np.uint8)[self.data_offset:self.byte_size]
        return raw.view(self.record_dtype)


def wavefront_glsl_structs():
    """Return GLSL declarations matching the public NumPy wavefront ABI."""
    return f"""\
const uint WAVE_MAX_MEDIUM_STACK_DEPTH = {MAX_MEDIUM_STACK_DEPTH}u;

struct WaveRay {{
    vec4 origin_tmin;
    vec4 direction_tmax;
    uint path_index;
    uint padding_a;
    uint padding_b;
    uint padding_c;
}};

struct WaveHit {{
    vec4 position_t;
    vec3 geometric_normal;
    uint primitive_index;
    vec2 barycentrics;
    uint ray_index;
    uint path_index;
}};

struct WavePathState {{
    vec4 throughput;
    vec4 radiance;
    uvec4 metadata;
    uvec4 rng;
    float ior_stack[WAVE_MAX_MEDIUM_STACK_DEPTH];
}};

struct WaveHotPathState {{
    vec4 throughput;
    vec4 radiance;
    uvec4 metadata;
}};

struct WaveMediumStack {{
    float ior[WAVE_MAX_MEDIUM_STACK_DEPTH];
}};

struct WaveQueueHeader {{
    uint count;
    uint capacity;
    uint overflow;
    uint padding;
}};
"""


def create_wavefront_pipeline(*, include_shading=True):
    """Create the logical stage graph for split wavefront dispatches.

    Generation, intersection, and shading have shader implementations. Shading
    can be excluded for low-level queue and intersection diagnostics.
    """
    stages = [
        RenderStage(
            "wavefront_generate_primary",
            reads={"camera", "tile"},
            writes={"ray_queue", "path_states", "medium_stacks"},
        ),
        RenderStage(
            "wavefront_intersect",
            reads={"scene", "geometry", "ray_queue"},
            writes={"hit_queue"},
        ),
    ]
    if include_shading:
        stages.append(RenderStage(
            "wavefront_shade",
            reads={"scene", "hit_queue", "path_states", "medium_stacks"},
            writes={
                "next_ray_queue", "path_states", "medium_stacks", "radiance"
            },
        ))
    return RenderPipeline(
        stages, initial_resources={"camera", "tile", "scene", "geometry"}
    )
