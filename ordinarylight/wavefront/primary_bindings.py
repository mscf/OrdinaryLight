"""Device-independent descriptor contract shared by fused primary shaders."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PrimaryBinding:
    name: str
    binding: int
    kind: str
    count: int = 1
    access: str = "read"


_BASE = (
    PrimaryBinding("tlas", 0, "acceleration_structure"),
    PrimaryBinding("paths", 1, "buffer", access="read_write"),
    PrimaryBinding("materials", 2, "buffer"),
    PrimaryBinding("vertices", 3, "buffer"),
    PrimaryBinding("attributes", 4, "buffer"),
    PrimaryBinding("next_rays", 5, "buffer", access="read_write"),
    PrimaryBinding("media", 6, "buffer", access="read_write"),
    PrimaryBinding("camera", 7, "buffer"),
    PrimaryBinding("position", 8, "image", access="write"),
    PrimaryBinding("normal", 9, "image", access="write"),
    PrimaryBinding("lights", 10, "buffer"),
    PrimaryBinding("area_lights", 11, "buffer"),
    PrimaryBinding("textures", 12, "buffer"),
    PrimaryBinding("texture_bindings", 13, "buffer"),
    PrimaryBinding("reservoirs", 16, "buffer", access="read_write"),
    PrimaryBinding("previous_reservoirs", 17, "buffer"),
    PrimaryBinding("previous_camera", 18, "buffer"),
    PrimaryBinding("previous_position", 19, "image"),
    PrimaryBinding("previous_normal", 20, "image"),
    PrimaryBinding("material", 21, "image", access="write"),
    PrimaryBinding("previous_material", 22, "image"),
    PrimaryBinding("secondary_paths", 23, "buffer", access="read_write"),
    PrimaryBinding("custom_attributes", 24, "buffer"),
    PrimaryBinding("volume_headers", 25, "buffer"),
    PrimaryBinding("volume_scalars", 26, "buffer"),
    PrimaryBinding("volume_transfers", 27, "buffer"),
    PrimaryBinding("triangle_volumes", 28, "buffer"),
    PrimaryBinding("sampled_volumes", 29, "combined_image_sampler", 16),
)


def primary_bindings(*, native_textures=False, profiling=False, primary_hits=False, custom_history=False):
    """Return the immutable native-compatible fused layout in binding order.

    Reserved custom attributes remain in the layout even when a shader does not
    use them. Access is conservative across primary/hybrid/megakernel variants;
    optional runtime policies may disable individual writes. No handles required.
    """
    if any(type(flag) is not bool for flag in (native_textures, profiling, primary_hits, custom_history)):
        raise TypeError("Primary layout flags must be bools")
    bindings = list(_BASE)
    if custom_history:
        bindings.extend((PrimaryBinding("primary_history", 31, "buffer", access="write"),
                         PrimaryBinding("previous_vertices", 32, "buffer")))
    if primary_hits:
        bindings.append(PrimaryBinding("primary_hits", 30, "buffer", access="write"))
    if native_textures:
        bindings.append(
            PrimaryBinding("sampled_textures", 14, "combined_image_sampler", 128)
        )
    if profiling:
        bindings.append(
            PrimaryBinding("work_counters", 15, "buffer", access="read_write")
        )
    return tuple(sorted(bindings, key=lambda item: item.binding))
