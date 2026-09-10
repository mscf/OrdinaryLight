"""Typed transport helpers. Generated artifacts must be rebuilt from this source."""
import ordinaryshade as osh
from .transport_integrator_programs import SampleAccumulation

@osh.function
def main() -> osh.void:
    i = gl_GlobalInvocationID.x
    if osh.specialization('defined(OL_GPU_REDUCTION)'):
        if gpu_control.z != osh.u32(0) or i >= osh.minimum(pc.group_count, gpu_control.y):
            return
    elif i >= pc.group_count:
        return
    group = groups[i]
    result = accumulated[group.x]
    j = osh.u32(0)
    while j < group.z:
        value = contributions[indices[group.y + j]]
        weight = weights[indices[group.y + j]]
        result.radiance.rgb = result.radiance.rgb + value.radiance.rgb * weight.x
        result.radiance.w = result.radiance.w + osh.f32(value.counts.y) * weight.y
        result.counts.xy = result.counts.xy + value.counts.xy
        result.counts.z = result.counts.z | value.counts.z
        result.counts.w = result.counts.w + value.counts.w
        result.events = result.events + value.events
        j = j + 1
    if osh.any_value(osh.is_nan(result.radiance)) or osh.any_value(osh.is_inf(result.radiance)):
        result.counts.z = result.counts.z | osh.u32(16)
    accumulated[group.x] = result

@osh.structure
class PipelineConstants:
    group_count: osh.u32
