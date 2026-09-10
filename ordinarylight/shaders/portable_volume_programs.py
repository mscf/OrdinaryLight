"""Portable volume preview, including slicing and front-to-back compositing."""
import ordinaryshade as osh


@osh.structure
class Settings:
    viewport: osh.vec4
    camera: osh.vec4
    slice: osh.vec4
    dimensions: osh.vec4


@osh.structure
class VolumePreviewVertex:
    position: osh.builtin(osh.vec4, 'position')
    uv: osh.location(osh.vec2, 0)


@osh.vertex
def volume_preview_vertex(index: osh.builtin(osh.u32, 'vertex_index')) -> VolumePreviewVertex:
    position = osh.vec2(-1.0, -1.0)
    if index == osh.u32(1):
        position = osh.vec2(3.0, -1.0)
    if index == osh.u32(2):
        position = osh.vec2(-1.0, 3.0)
    return VolumePreviewVertex(osh.vec4(position, 0.0, 1.0), position)


@osh.function
def color_at(position: osh.vec3) -> osh.vec4:
    dimensions = osh.uvec3(settings.dimensions.xyz)
    cell = osh.minimum(osh.uvec3(osh.clamp(position * 0.5 + 0.5, osh.vec3(0.0), osh.vec3(1.0))
                               * osh.vec3(dimensions)), dimensions - osh.u32(1))
    value = density[(cell.z * dimensions.y + cell.y) * dimensions.x + cell.x]
    fraction = osh.clamp(value / settings.viewport.w, 0.0, 1.0)
    index = osh.u32(fraction * osh.f32(osh.array_length(transfer) - osh.u32(1)))
    return transfer[index]


@osh.fragment
def volume_preview_fragment(
    uv: osh.location(osh.vec2, 0),
    density: osh.storage_buffer(osh.f32, access='read', binding=0),
    settings: osh.uniform_buffer(Settings, binding=1),
    transfer: osh.storage_buffer(osh.vec4, access='read', binding=2),
) -> osh.location(osh.vec4, 0):
    yaw = settings.camera.x
    pitch = settings.camera.y
    eye = settings.camera.z * osh.vec3(osh.cosine(pitch) * osh.sine(yaw),
                                      osh.sine(pitch), osh.cosine(pitch) * osh.cosine(yaw))
    forward = osh.normalize(-eye)
    right = osh.normalize(osh.cross(forward, osh.vec3(0.0, 1.0, 0.0)))
    up = osh.cross(right, forward)
    ray = osh.normalize(forward + right * uv.x * settings.viewport.x / settings.viewport.y * 0.5
                        + up * uv.y * 0.5)
    inverse = 1.0 / osh.select(osh.absolute(ray) > osh.vec3(1e-7), ray, osh.vec3(1e-7))
    a = (-osh.vec3(1.0) - eye) * inverse
    b = (osh.vec3(1.0) - eye) * inverse
    near = osh.maximum(osh.maximum(osh.minimum(a.x, b.x), osh.minimum(a.y, b.y)), osh.minimum(a.z, b.z))
    far = osh.minimum(osh.minimum(osh.maximum(a.x, b.x), osh.maximum(a.y, b.y)), osh.maximum(a.z, b.z))
    result = osh.vec4(0.0)
    if far >= osh.maximum(near, 0.0):
        if settings.slice.x > 0.5:
            distance = (settings.slice.y * 2.0 - 1.0 - eye.z) * inverse.z
            if distance >= osh.maximum(near, 0.0) and distance <= far:
                result = color_at(eye + distance * ray)
                result.a = 1.0
        else:
            distance = osh.maximum(near, 0.0)
            steps = osh.u32(0)
            while True:
                if distance > far or result.a > 0.995 or steps >= osh.u32(1024):
                    break
                color = color_at(eye + distance * ray)
                alpha = 1.0 - osh.exp(-color.a * settings.viewport.z * settings.camera.w * 40.0)
                result = osh.vec4(result.rgb + (1.0 - result.a) * alpha * color.rgb,
                                  result.a + (1.0 - result.a) * alpha)
                distance = distance + settings.camera.w
                steps = steps + osh.u32(1)
    return osh.vec4(result.rgb + (1.0 - result.a) * osh.vec3(0.025, 0.03, 0.04), 1.0)
