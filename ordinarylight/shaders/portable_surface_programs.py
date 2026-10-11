"""Typed stages for the portable unlit scalar surface renderer."""
import ordinaryshade as osh


@osh.structure
class SurfaceCamera:
    row0: osh.vec4
    row1: osh.vec4
    row2: osh.vec4
    row3: osh.vec4


@osh.structure
class SurfaceVertex:
    position: osh.builtin(osh.vec4, "position")
    scalar: osh.location(osh.f32, 0)


@osh.vertex
def surface_vertex(
    position: osh.location(osh.vec3, 0),
    scalar: osh.location(osh.f32, 1),
    camera: osh.uniform_buffer(SurfaceCamera, binding=0),
) -> SurfaceVertex:
    point = osh.vec4(position, 1.0)
    clip = osh.vec4(osh.dot(camera.row0, point), osh.dot(camera.row1, point),
                    osh.dot(camera.row2, point), osh.dot(camera.row3, point))
    return SurfaceVertex(clip, scalar)


@osh.fragment
def surface_fragment(
    scalar: osh.location(osh.f32, 0),
    transfer: osh.storage_buffer(osh.vec4, access="read", binding=1),
) -> osh.location(osh.vec4, 0):
    fraction = osh.clamp(scalar, 0.0, 1.0)
    color = transfer[0].rgb
    index = osh.u32(1)
    count = osh.array_length(transfer)
    while index < count:
        lower = transfer[index - osh.u32(1)]
        upper = transfer[index]
        amount = osh.clamp((fraction - lower.a) / (upper.a - lower.a), 0.0, 1.0)
        color = osh.mix(lower.rgb, upper.rgb, amount)
        if fraction <= upper.a:
            break
        index = index + osh.u32(1)
    return osh.vec4(color, 1.0)
