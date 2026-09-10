"""Typed transport helpers. Generated artifacts must be rebuilt from this source."""
import ordinaryshade as osh
from .transport_programs import OrdinaryLightSurfaceSample

@osh.structure
class SampleAccumulation:
    radiance: osh.vec4
    counts: osh.uvec4
    events: osh.uvec4

@osh.function
def main() -> osh.void:
    i = gl_GlobalInvocationID.x
    if pc.mode == osh.u32(0):
        if i != osh.u32(0):
            return
        valid = ((counts.x <= pc.capacity and counts.y <= pc.groups_capacity) and counts.y <= counts.x) and (counts.x == osh.u32(0)) == (counts.y == osh.u32(0))
        state = osh.uvec4(counts.xy if valid else osh.uvec2(0), osh.u32(0) if valid else osh.u32(32), 0)
        commands[0] = osh.uvec4((state.x + osh.u32(63)) / osh.u32(64), 1, 1, 0)
        commands[1] = osh.uvec4(0, 1, 1, 0)
        commands[2] = osh.uvec4(0, 1, 1, 0)
        commands[3] = osh.uvec4(0, 1, 1, 0)
        return
    if pc.mode == osh.u32(2):
        if i != osh.u32(0):
            return
        commands[1] = osh.uvec4((state.x + osh.u32(63)) / osh.u32(64) if state.z == osh.u32(0) else osh.u32(0), 1, 1, 0)
        commands[2] = osh.uvec4((state.y + osh.u32(63)) / osh.u32(64) if state.z == osh.u32(0) else osh.u32(0), 1, 1, 0)
        commands[3] = osh.uvec4((pc.output_capacity + osh.u32(63)) / osh.u32(64) if state.z != osh.u32(0) else osh.u32(0), 1, 1, 0)
        return
    if pc.mode == osh.u32(3):
        if i < pc.output_capacity and state.z != osh.u32(0):
            accumulated[i].counts.z = accumulated[i].counts.z | osh.u32(32)
        return
    if i >= state.x:
        return
    if i < state.y:
        group = groups[i]
        valid = ((group.x < pc.output_capacity and group.y <= state.x) and group.z > osh.u32(0)) and group.z <= state.x - group.y
        if i == osh.u32(0):
            valid = valid and group.y == osh.u32(0)
        else:
            previous = groups[i - osh.u32(1)]
            valid = (((valid and previous.x < group.x) and previous.y <= state.x) and previous.z <= state.x - previous.y) and group.y == previous.y + previous.z
        if i + osh.u32(1) == state.y:
            valid = valid and group.y + group.z == state.x
        if not valid:
            osh.atomic_or(state.z, osh.u32(32))
    lower = osh.u32(0)
    upper = state.y
    while lower < upper:
        middle = lower + (upper - lower) / osh.u32(2)
        if groups[middle].y <= i:
            lower = middle + osh.u32(1)
        else:
            upper = middle
    if lower == osh.u32(0):
        osh.atomic_or(state.z, osh.u32(32))
        return
    group = groups[lower - osh.u32(1)]
    slot = indices[i]
    if ((slot >= state.x or group.y > i) or group.z > state.x) or i - group.y >= group.z:
        osh.atomic_or(state.z, osh.u32(32))
        return
    if (((i > group.y and indices[i - osh.u32(1)] >= slot or samples[slot].identity.x != group.x) or osh.any_value(osh.is_nan(weights[slot]))) or osh.any_value(osh.is_inf(weights[slot]))) or osh.any_value(weights[slot] < osh.vec2(0)):
        osh.atomic_or(state.z, osh.u32(32))

@osh.structure
class PipelineConstants:
    capacity: osh.u32
    groups_capacity: osh.u32
    output_capacity: osh.u32
    mode: osh.u32
