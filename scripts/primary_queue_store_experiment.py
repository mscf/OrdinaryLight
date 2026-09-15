"""Diagnostic typed OrdinaryShade complete continuation-record store."""
from pathlib import Path
from primary_hit_layout_experiment import load_module


def variant(directory):
    from ordinarylight.shaders import fused_primary_programs
    source=Path(fused_primary_programs.__file__).read_text()
    original='''        if not ordinarylight_enqueue_continuation(output_index, queued_origin, queued_direction, path_index, cone_width, cone_spread):
            ordinarylight_deactivate_stored_path(path_index)
            return'''
    replacement='''        if output_index >= output_queue.capacity:
            osh.atomic_add(output_queue.overflow, osh.u32(1))
            ordinarylight_deactivate_stored_path(path_index)
            return
        output_queue.rays[output_index] = WaveRay(
            queued_origin, queued_direction, path_index,
            osh.float_bits_to_uint(cone_width), osh.float_bits_to_uint(cone_spread), osh.u32(0))'''
    assert source.count(original)==1
    return load_module('ordinarylight.shaders.primary_queue_diagnostic',source.replace(original,replacement),directory)
