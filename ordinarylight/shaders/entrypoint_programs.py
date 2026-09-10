"""Typed continuation and shader-execution-reordering probe entry points."""
import ordinaryshade as osh
from .fused_primary_programs import WaveRay, WavePathState


@osh.structure
class ProbeOutput:
    values: osh.runtime_array(osh.u32)


@osh.function
def ser_probe() -> osh.void:
    index = gl_LaunchIDEXT.y * gl_LaunchSizeEXT.x + gl_LaunchIDEXT.x
    if osh.specialization('WAVE_SER'):
        osh.reorder_thread(index & osh.u32(3), osh.u32(2))
    probe_output.values[index] = index ^ osh.u32(0x51e4a9bd)


@osh.function
def empty_miss() -> osh.void:
    pass


@osh.function
def continuation() -> osh.void:
    queue_index = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x
    if queue_index >= output_queue.count:
        return
    ray = output_queue.rays[queue_index]
    path_index = ray.path_index
    path = paths[path_index]
    origin = ray.origin_tmin.xyz
    direction = ray.direction_tmax.xyz
    cone_width = osh.uint_bits_to_float(ray.padding_a)
    cone_spread = osh.uint_bits_to_float(ray.padding_b)
    medium_depth = osh.maximum(path.metadata.w >> osh.u32(8), osh.u32(1))
    rng = pathRng(path)
    traceRemaining(path, origin, direction, path_index, medium_depth, rng, cone_width, cone_spread)
    paths[path_index] = path
