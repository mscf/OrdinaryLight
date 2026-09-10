"""Typed transport helpers. Generated artifacts must be rebuilt from this source."""
import ordinaryshade as osh

@osh.structure
class Record:
    lower: osh.vec4
    upper: osh.vec4
    parameters: osh.vec4
    metadata: osh.uvec4

@osh.function
def main() -> osh.void:
    i = gl_GlobalInvocationID.x
    if i >= CAPACITY:
        return
    record = input_records[i]
    status = osh.u32(0)
    enabled = record.metadata.x != osh.u32(4294967295)
    if enabled:
        if record.metadata.x >= PROGRAMS or record.metadata.y >= MATERIALS:
            status = status | osh.u32(1)
        if (((((osh.any_value(osh.is_nan(record.lower)) or osh.any_value(osh.is_inf(record.lower))) or osh.any_value(osh.is_nan(record.upper))) or osh.any_value(osh.is_inf(record.upper))) or osh.any_value(osh.is_nan(record.parameters))) or osh.any_value(osh.is_inf(record.parameters))) or osh.any_value(record.upper.xyz <= record.lower.xyz):
            status = status | osh.u32(2)
        boundary = boundaryIndex(record.metadata.z)
        if record.metadata.z != osh.u32(4294967295) and boundary == osh.u32(4294967295) or dielectricMaterial(record.metadata.y) != (boundary != osh.u32(4294967295)):
            status = status | osh.u32(4)
        record.metadata.y = record.metadata.y + TRIANGLES
        record.metadata.z = boundary
    if not enabled or status != osh.u32(0):
        record.lower = osh.vec4(0)
        record.upper = osh.vec4(0.0001)
        record.parameters = osh.vec4(0)
        record.metadata = osh.uvec4(osh.u32(4294967295), 0, osh.u32(4294967295), 0)
    output_records[i] = record
    axis = osh.u32(0)
    while axis < osh.u32(3):
        bounds[i * osh.u32(6) + axis] = record.lower[axis]
        bounds[i * osh.u32(6) + axis + osh.u32(3)] = record.upper[axis]
        axis = axis + 1
    diagnostics[i] = status
