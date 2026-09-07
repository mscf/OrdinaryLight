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
