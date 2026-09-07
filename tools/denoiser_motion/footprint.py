"""Experimental screen-space depth footprint for offline temporal replay."""


def instrument_depth_footprint(source):
    marker = "        let depth_tolerance: f32 = max((abs(expected_old_depth) * constants.rejection.y), 0.001);"
    if source.count(marker) != 1:
        raise ValueError("temporal shader depth hook changed")
    replacement = """
        let base_tolerance = max(abs(expected_old_depth) * constants.rejection.y, 0.001);
        var footprint = 0.0;
        for (var axis = 0; axis < 2; axis += 1) {
            var slope = 1e30;
            var found = false;
            for (var side = -1; side <= 1; side += 2) {
                var offset = vec2<i32>(0);
                offset[axis] = side;
                let q = previous_pixel + offset;
                if (any(q < vec2<i32>(0)) || any(q >= extent)) { continue; }
                if (textureLoad(previous_identity, q).r != old_primitive ||
                    textureLoad(previous_material_id, q).r != old_material) { continue; }
                let z = textureLoad(previous_view_z, q).r;
                let n = textureLoad(previous_normal_roughness, q).xyz;
                if (z <= 0.0 || dot(n, old_normal) < constants.rejection.x) { continue; }
                slope = min(slope, abs(z - old_depth));
                found = true;
            }
            if (found) { footprint += slope; }
        }
        // One pixel of measured depth variation, capped at the tested 2%.
        // Flat surfaces retain the original tolerance.
        let depth_tolerance = max(base_tolerance,
            min(base_tolerance + footprint, abs(expected_old_depth) * 0.02));
"""
    return source.replace(marker, replacement)


def instrument_plane_gate(source):
    """Experimental static-geometry gate; diagnostic camera FOV is 45 degrees."""
    marker = "\n        if (accepted) {\n            history = textureLoad(previous_radiance, previous_pixel);"
    if source.count(marker) != 1:
        raise ValueError("temporal shader plane hook changed")
    declarations = """
@group(0) @binding(16) var current_position: texture_storage_2d<rgba32float, read>;
@group(0) @binding(17) var previous_position: texture_storage_2d<rgba32float, read>;
"""
    gate = """
        let delta = textureLoad(current_position, pixel).xyz -
                    textureLoad(previous_position, previous_pixel).xyz;
        let plane_distance = abs(dot(delta, old_normal));
        let normal_change = length(current_normal - old_normal);
        let tangent = sqrt(max(dot(delta, delta) - plane_distance * plane_distance, 0.0));
        let plane_footprint = 2.0 * old_depth * 0.41421356237 / f32(extent.y);
        let plane_tolerance = old_depth * 0.00001 +
                              normal_change * max(plane_footprint, tangent);
        accepted = accepted && plane_distance <= plane_tolerance;
"""
    return declarations + source.replace(marker, gate + marker)
