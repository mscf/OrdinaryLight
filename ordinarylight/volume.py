"""Backend-neutral dense-volume sampling and packing helpers."""

from __future__ import annotations

import numpy as np

from .lights import DirectionalLight, SpotLight


VOLUME_HEADER_DTYPE = np.dtype([
    ("world_to_local", np.float32, (4, 4)),
    ("dimensions_offset", np.uint32, (4,)),
    ("value_parameters", np.float32, (4,)),
    ("render_parameters", np.float32, (4,)),
    ("scattering_parameters", np.float32, (4,)),
    ("phase_parameters", np.float32, (4,)),
    ("multiple_scattering_parameters", np.float32, (4,)),
    ("acceleration_parameters", np.uint32, (4,)),
])

VOLUME_BRICK_SIZE = 8


def _transfer_interval_has_opacity(transfer_function, lower, upper):
    """Conservatively classify a normalized scalar interval as nonempty."""
    lower = float(np.clip(lower, 0.0, 1.0))
    upper = float(np.clip(upper, lower, 1.0))
    count = len(transfer_function.values)
    coordinates = [lower, upper]
    if count > 1:
        first = max(0, int(np.ceil(lower * (count - 1))))
        last = min(count - 1, int(np.floor(upper * (count - 1))))
        coordinates.extend(index / (count - 1) for index in range(first, last + 1))
    alpha = transfer_function.sample(np.asarray(coordinates, np.float32))[:, 3]
    return bool(np.any(alpha > 0.0))


def volume_brick_occupancy(volume, brick_size=VOLUME_BRICK_SIZE):
    """Return conservative z/y/x occupancy for transfer-visible bricks.

    Each brick includes a one-voxel interpolation halo on both sides.  The
    normalized-coordinate convention used by sampled 3-D images is centered
    on texels rather than voxel corners, so both neighbours are required to
    prove that every filtered sample in a brick has zero transfer alpha.
    """
    brick_size = int(brick_size)
    if brick_size < 1:
        raise ValueError("brick_size must be positive")
    depth, height, width = volume.shape
    grid_xyz = tuple(
        max(1, int(np.ceil((dimension - 1) / brick_size)))
        for dimension in (width, height, depth)
    )
    occupancy = np.zeros(grid_xyz[::-1], np.float32)
    data = volume.normalized_data
    if volume.material.density_scale <= 0.0:
        return occupancy
    for brick_z in range(grid_xyz[2]):
        z0 = max(brick_z * brick_size - 1, 0)
        z1 = min((brick_z + 1) * brick_size + 1, depth)
        for brick_y in range(grid_xyz[1]):
            y0 = max(brick_y * brick_size - 1, 0)
            y1 = min((brick_y + 1) * brick_size + 1, height)
            for brick_x in range(grid_xyz[0]):
                x0 = max(brick_x * brick_size - 1, 0)
                x1 = min((brick_x + 1) * brick_size + 1, width)
                values = data[z0:z1, y0:y1, x0:x1]
                occupancy[brick_z, brick_y, brick_x] = float(
                    _transfer_interval_has_opacity(
                        volume.material.transfer_function,
                        float(np.min(values)), float(np.max(values)),
                    )
                )
    return occupancy


def volume_empty_space_statistics(volumes, brick_size=VOLUME_BRICK_SIZE):
    """Describe conservative brick occupancy without allocating GPU state."""
    per_volume = []
    total = occupied = 0
    for volume in volumes:
        bricks = volume_brick_occupancy(volume, brick_size)
        brick_count = int(bricks.size)
        occupied_count = int(np.count_nonzero(bricks))
        total += brick_count
        occupied += occupied_count
        per_volume.append({
            "name": volume.name,
            "grid": tuple(map(int, bricks.shape[::-1])),
            "bricks": brick_count,
            "occupied": occupied_count,
            "empty_fraction": 1.0 - occupied_count / max(brick_count, 1),
        })
    return {
        "brick_size": int(brick_size),
        "bricks": total,
        "occupied": occupied,
        "empty_fraction": 1.0 - occupied / max(total, 1),
        "volumes": tuple(per_volume),
    }


def phase_function(cosine, anisotropy=0.0, kind="isotropic"):
    """Evaluate a normalized isotropic or Henyey--Greenstein phase function."""
    cosine = np.clip(np.asarray(cosine, np.float32), -1.0, 1.0)
    if kind == "isotropic":
        return np.full_like(cosine, 1.0 / (4.0 * np.pi), dtype=np.float32)
    if kind != "henyey_greenstein":
        raise ValueError("kind must be 'isotropic' or 'henyey_greenstein'")
    g = float(anisotropy)
    if not np.isfinite(g) or not -0.99 <= g <= 0.99:
        raise ValueError("anisotropy must be between -0.99 and 0.99")
    denominator = np.maximum(1.0 + g * g - 2.0 * g * cosine, 1e-8)
    return np.asarray(
        (1.0 - g * g) / (4.0 * np.pi * denominator ** 1.5), np.float32
    )


def _approximate_light_transmittance(volumes, positions, light_position):
    """Estimate source-path attenuation with one midpoint per crossed medium."""
    if not volumes:
        return np.ones(len(positions), np.float32)
    offset = np.asarray(light_position, np.float32) - positions
    distances = np.linalg.norm(offset, axis=1)
    directions = offset / np.maximum(distances, 1e-8)[:, None]
    entries, exits = intersect_unit_boxes(positions, directions, volumes)
    optical_depth = np.zeros(len(positions), np.float32)
    for volume_index, volume in enumerate(volumes):
        entry = np.maximum(entries[volume_index], 0.0)
        exit_distance = np.minimum(exits[volume_index], distances)
        valid = np.isfinite(entry) & (exit_distance > entry)
        if not np.any(valid):
            continue
        ray_indices = np.flatnonzero(valid)
        midpoint = 0.5 * (entry[ray_indices] + exit_distance[ray_indices])
        samples = sample_trilinear(
            volume,
            positions[ray_indices] + directions[ray_indices] * midpoint[:, None],
        )
        rgba = volume.material.transfer_function.sample(samples)
        alpha = np.clip(
            rgba[:, 3] * volume.material.density_scale, 0.0, 1.0 - 1e-7,
        )
        extinction = -np.log1p(-alpha) / volume.material.step_size
        optical_depth[ray_indices] += extinction * (
            exit_distance[ray_indices] - entry[ray_indices]
        )
    return np.asarray(np.exp(-optical_depth), np.float32)


def _point_light_scattering(
    material, positions, view_directions, lights, volumes=(),
    optical_depth=None,
):
    if material.scattering_scale <= 0.0 or not lights:
        return np.zeros((len(positions), 3), np.float32)
    result = np.zeros((len(positions), 3), np.float32)
    isotropic = np.zeros_like(result)
    outgoing = -np.asarray(view_directions, np.float32)
    for light in lights:
        attenuation = np.ones(len(positions), np.float32)
        if isinstance(light, DirectionalLight):
            incoming_direction = -np.asarray(light.direction, np.float32)
            incoming_direction /= np.linalg.norm(incoming_direction)
            incoming = np.repeat(
                incoming_direction[None, :], len(positions), axis=0
            )
            light_positions = positions + incoming * 10000.0
        else:
            light_position = np.asarray(light.position, np.float32)
            offset = light_position - positions
            distance_squared = np.maximum(
                np.sum(offset * offset, axis=1), 1e-8
            )
            distance = np.sqrt(distance_squared)
            incoming = offset / distance[:, None]
            attenuation /= distance_squared
            if light.range is not None:
                attenuation[distance > light.range] = 0.0
            if isinstance(light, SpotLight):
                direction = np.asarray(light.direction, np.float32)
                direction /= np.linalg.norm(direction)
                cone = np.sum(direction[None, :] * -incoming, axis=1)
                inner = np.cos(light.inner_cone_angle)
                outer = np.cos(light.outer_cone_angle)
                spot = np.clip((cone - outer) / max(inner - outer, 1e-8), 0, 1)
                attenuation *= spot * spot * (3.0 - 2.0 * spot)
            light_positions = np.repeat(
                light_position[None, :], len(positions), axis=0
            )
        phase = phase_function(
            np.sum(-incoming * outgoing, axis=1), material.anisotropy,
            material.phase_function,
        )
        incident = (
            np.asarray(light.color, np.float32)[None, :] * float(light.intensity)
            * attenuation[:, None]
        )
        incident *= _approximate_light_transmittance(
            volumes, positions, light_positions,
        )[:, None]
        result += incident * phase[:, None]
        isotropic += incident * (1.0 / (4.0 * np.pi))
    if material.scattering_orders > 1 and optical_depth is not None:
        trapping = 1.0 - np.exp(-np.maximum(
            np.asarray(optical_depth, np.float32), 0.0,
        ))
        ratio = (
            trapping[:, None]
            * np.asarray(material.scattering_albedo, np.float32)[None, :]
        )
        power = ratio.copy()
        for _order in range(2, material.scattering_orders + 1):
            result += isotropic * power
            power *= ratio
    return result * np.asarray(material.scattering_color, np.float32)[None, :]


def intersect_unit_boxes(origins, directions, volumes):
    """Return entry/exit distances for transformed local unit cubes."""
    origins = np.asarray(origins, np.float32)
    directions = np.asarray(directions, np.float32)
    entries = np.full((len(volumes), len(origins)), np.inf, np.float32)
    exits = np.full_like(entries, -np.inf)
    for index, volume in enumerate(volumes):
        inverse = np.linalg.inv(volume.transform.matrix).astype(np.float32)
        local_origins = origins @ inverse[:3, :3].T + inverse[:3, 3]
        local_directions = directions @ inverse[:3, :3].T
        safe = np.where(
            np.abs(local_directions) > 1e-12,
            local_directions,
            np.copysign(1e-12, local_directions + 1e-30),
        )
        bounds_a = -local_origins / safe
        bounds_b = (1.0 - local_origins) / safe
        entry = np.max(np.minimum(bounds_a, bounds_b), axis=1)
        exit = np.min(np.maximum(bounds_a, bounds_b), axis=1)
        valid = exit >= np.maximum(entry, 0.0)
        entries[index, valid] = np.maximum(entry[valid], 0.0)
        exits[index, valid] = exit[valid]
    return entries, exits


def sample_trilinear(volume, world_positions):
    """Sample normalized volume values at world-space positions."""
    positions = np.asarray(world_positions, np.float32)
    inverse = np.linalg.inv(volume.transform.matrix).astype(np.float32)
    local = positions @ inverse[:3, :3].T + inverse[:3, 3]
    # Data is indexed z, y, x while local coordinates are x, y, z.
    shape_xyz = np.asarray(volume.data.shape[::-1], np.float32)
    coordinates = np.clip(local, 0.0, 1.0) * (shape_xyz - 1.0)
    lower = np.floor(coordinates).astype(np.int64)
    upper = np.minimum(lower + 1, np.asarray(volume.data.shape[::-1]) - 1)
    weight = coordinates - lower
    x0, y0, z0 = lower.T
    x1, y1, z1 = upper.T
    wx, wy, wz = weight.T
    values = volume.normalized_data
    c000 = values[z0, y0, x0]
    c100 = values[z0, y0, x1]
    c010 = values[z0, y1, x0]
    c110 = values[z0, y1, x1]
    c001 = values[z1, y0, x0]
    c101 = values[z1, y0, x1]
    c011 = values[z1, y1, x0]
    c111 = values[z1, y1, x1]
    c00 = c000 * (1.0 - wx) + c100 * wx
    c10 = c010 * (1.0 - wx) + c110 * wx
    c01 = c001 * (1.0 - wx) + c101 * wx
    c11 = c011 * (1.0 - wx) + c111 * wx
    c0 = c00 * (1.0 - wy) + c10 * wy
    c1 = c01 * (1.0 - wy) + c11 * wy
    return np.asarray(c0 * (1.0 - wz) + c1 * wz, np.float32)


def integrate_volume(
    volume, origins, directions, entries, exits, *, max_distance=None, lights=(),
):
    """Front-to-back integrate one volume for a vector of ray intervals."""
    origins = np.asarray(origins, np.float32)
    directions = np.asarray(directions, np.float32)
    entries = np.asarray(entries, np.float32)
    exits = np.asarray(exits, np.float32)
    if max_distance is not None:
        exits = np.minimum(exits, np.asarray(max_distance, np.float32))
    valid = np.isfinite(entries) & (exits > entries)
    radiance = np.zeros((len(origins), 3), np.float32)
    transmittance = np.ones(len(origins), np.float32)
    if not np.any(valid):
        return radiance, transmittance
    material = volume.material
    # Use a world-space step while bounding work for unusually large volumes.
    length = np.maximum(exits - entries, 0.0)
    steps = np.minimum(
        np.ceil(length / material.step_size).astype(np.int32), 4096
    )
    scattering_source = np.zeros((len(origins), 3), np.float32)
    scattering_indices = np.flatnonzero(valid)
    if material.scattering_scale > 0.0 and len(scattering_indices):
        midpoint = 0.5 * (
            entries[scattering_indices] + exits[scattering_indices]
        )
        positions = origins[scattering_indices] + (
            directions[scattering_indices] * midpoint[:, None]
        )
        midpoint_rgba = material.transfer_function.sample(
            sample_trilinear(volume, positions)
        )
        midpoint_alpha = np.clip(
            midpoint_rgba[:, 3] * material.density_scale,
            0.0, 1.0 - 1e-7,
        )
        optical_depth = (
            -np.log1p(-midpoint_alpha) / material.step_size
            * length[scattering_indices]
        )
        scattering_source[scattering_indices] = _point_light_scattering(
            material, positions, directions[scattering_indices], lights,
            (volume,), optical_depth,
        ) * material.scattering_scale
    max_steps = int(steps[valid].max(initial=0))
    for step_index in range(max_steps):
        active = valid & (step_index < steps) & (transmittance > 1e-4)
        if not np.any(active):
            break
        indices = np.flatnonzero(active)
        dt = length[indices] / steps[indices]
        distance = entries[indices] + (step_index + 0.5) * dt
        positions = origins[indices] + directions[indices] * distance[:, None]
        scalar = sample_trilinear(volume, positions)
        rgba = material.transfer_function.sample(scalar)
        reference_alpha = np.clip(
            rgba[:, 3] * material.density_scale, 0.0, 1.0 - 1e-7
        )
        alpha = 1.0 - np.power(
            1.0 - reference_alpha, dt / material.step_size
        )
        contribution = transmittance[indices] * alpha
        radiance[indices] += (
            contribution[:, None] * rgba[:, :3] * material.emission_scale
        )
        radiance[indices] += (
            contribution[:, None] * scattering_source[indices]
        )
        transmittance[indices] *= 1.0 - alpha
    return radiance, transmittance


def integrate_volumes(
    volumes, origins, directions, entries, exits, *, max_distance=None,
    lights=(),
):
    """Integrate possibly overlapping media with order-independent mixing.

    Extinction coefficients add within an overlap. Emitted radiance is weighted
    by each medium's extinction, matching the analytic emission--absorption
    solution over each ray-march step.
    """
    volumes = tuple(volumes)
    origins = np.asarray(origins, np.float32)
    directions = np.asarray(directions, np.float32)
    entries = np.asarray(entries, np.float32)
    exits = np.asarray(exits, np.float32)
    if entries.shape != (len(volumes), len(origins)) or exits.shape != entries.shape:
        raise ValueError("entries and exits must have shape (volume_count, ray_count)")
    if max_distance is not None:
        exits = np.minimum(exits, np.asarray(max_distance, np.float32))

    radiance = np.zeros((len(origins), 3), np.float32)
    transmittance = np.ones(len(origins), np.float32)
    for ray_index in range(len(origins)):
        intervals = []
        for volume_index in range(len(volumes)):
            entry = float(max(entries[volume_index, ray_index], 0.0))
            exit_distance = float(exits[volume_index, ray_index])
            if np.isfinite(entry) and exit_distance > entry:
                intervals.append((entry, exit_distance, volume_index))
        if not intervals:
            continue
        events = sorted({event for interval in intervals for event in interval[:2]})
        ray_radiance = np.zeros(3, np.float64)
        ray_transmittance = 1.0
        scattering_sources = {}
        for entry, exit_distance, volume_index in intervals:
            volume = volumes[volume_index]
            material = volume.material
            distance = 0.5 * (entry + exit_distance)
            position = origins[ray_index] + directions[ray_index] * distance
            midpoint_rgba = material.transfer_function.sample(
                sample_trilinear(volume, position[None, :])
            )[0]
            midpoint_alpha = float(np.clip(
                midpoint_rgba[3] * material.density_scale,
                0.0, 1.0 - 1e-7,
            ))
            optical_depth = np.asarray((
                -np.log1p(-midpoint_alpha) / material.step_size
                * (exit_distance - entry),
            ), np.float32)
            scattering_sources[volume_index] = _point_light_scattering(
                material, position[None, :],
                directions[ray_index][None, :], lights, volumes,
                optical_depth,
            )[0] * material.scattering_scale
        for segment_start, segment_end in zip(events[:-1], events[1:]):
            midpoint = 0.5 * (segment_start + segment_end)
            active = [
                volume_index for entry, exit_distance, volume_index in intervals
                if entry <= midpoint < exit_distance
            ]
            if not active:
                continue
            reference_step = min(volumes[index].material.step_size for index in active)
            steps = min(int(np.ceil((segment_end - segment_start) / reference_step)), 4096)
            step_size = (segment_end - segment_start) / max(steps, 1)
            for step_index in range(steps):
                distance = segment_start + (step_index + 0.5) * step_size
                position = origins[ray_index] + directions[ray_index] * distance
                extinction = 0.0
                emission_extinction = np.zeros(3, np.float64)
                scattering_extinction = np.zeros(3, np.float64)
                for volume_index in active:
                    volume = volumes[volume_index]
                    material = volume.material
                    scalar = sample_trilinear(volume, position[None, :])
                    rgba = material.transfer_function.sample(scalar)[0]
                    reference_alpha = float(np.clip(
                        rgba[3] * material.density_scale, 0.0, 1.0 - 1e-7
                    ))
                    medium_extinction = -np.log1p(-reference_alpha) / material.step_size
                    extinction += medium_extinction
                    emission_extinction += (
                        medium_extinction * rgba[:3] * material.emission_scale
                    )
                    scattering_extinction += medium_extinction * (
                        scattering_sources[volume_index]
                    )
                if extinction > 1e-12:
                    alpha = 1.0 - np.exp(-extinction * step_size)
                    ray_radiance += (
                        ray_transmittance * alpha
                        * (emission_extinction + scattering_extinction)
                        / extinction
                    )
                    ray_transmittance *= 1.0 - alpha
                if ray_transmittance < 1e-4:
                    break
            if ray_transmittance < 1e-4:
                break
        radiance[ray_index] = ray_radiance
        transmittance[ray_index] = ray_transmittance
    return radiance, transmittance


def pack_volumes(volumes, *, empty_space_skipping=True):
    """Pack scalar data, transfer functions, and transforms for GPU upload."""
    volumes = tuple(volumes)
    headers = np.zeros(max(1, len(volumes)), dtype=VOLUME_HEADER_DTYPE)
    scalar_chunks = []
    transfer_chunks = []
    scalar_offset = 0
    transfer_offset = 0
    for index, volume in enumerate(volumes):
        depth, height, width = volume.shape
        transfer = volume.material.transfer_function.values
        # NumPy stores rows while GLSL mat4 values are column-major.
        headers[index]["world_to_local"] = np.linalg.inv(
            volume.transform.matrix
        ).astype(np.float32).T
        headers[index]["dimensions_offset"] = (
            width, height, depth, scalar_offset,
        )
        headers[index]["value_parameters"] = (
            float(transfer_offset), float(len(transfer)),
            volume.material.density_scale, volume.material.emission_scale,
        )
        headers[index]["render_parameters"] = (
            volume.material.step_size, float(volume.id), float(len(volumes)), 0.0,
        )
        headers[index]["scattering_parameters"] = (
            *volume.material.scattering_color, volume.material.scattering_scale,
        )
        headers[index]["phase_parameters"] = (
            volume.material.anisotropy,
            1.0 if volume.material.phase_function == "henyey_greenstein" else 0.0,
            0.0, 0.0,
        )
        headers[index]["multiple_scattering_parameters"] = (
            *volume.material.scattering_albedo,
            float(volume.material.scattering_orders),
        )
        data = volume.normalized_data.reshape(-1)
        scalar_chunks.append(data)
        brick_offset = scalar_offset + len(data)
        if empty_space_skipping:
            bricks = volume_brick_occupancy(volume)
            if np.any(bricks == 0.0):
                brick_depth, brick_height, brick_width = bricks.shape
                headers[index]["acceleration_parameters"] = (
                    brick_offset, brick_width, brick_height, brick_depth,
                )
                scalar_chunks.append(bricks.reshape(-1))
                scalar_offset += bricks.size
        transfer_chunks.append(np.asarray(transfer, np.float32))
        scalar_offset += len(data)
        transfer_offset += len(transfer)
    scalars = (
        np.concatenate(scalar_chunks).astype(np.float32, copy=False)
        if scalar_chunks else np.zeros(1, np.float32)
    )
    transfers = (
        np.concatenate(transfer_chunks).astype(np.float32, copy=False)
        if transfer_chunks else np.zeros((1, 4), np.float32)
    )
    return headers, np.ascontiguousarray(scalars), np.ascontiguousarray(transfers)
