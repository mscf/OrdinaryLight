"""Deterministic scenes and metrics for renderer implementation parity."""

import math

import numpy as np

from ..materials import (
    SCATTER_DIFFUSE,
    SCATTER_REFLECTION,
    SCATTER_TRANSMISSION,
    SurfaceResponse,
    cosine_sample_hemisphere,
    dot,
    fresnel_schlick,
    material,
    maximum,
    reflect,
    refract,
    select,
)
from ..scene import Material, PerspectiveCamera, Scene, Texture, TextureTransform


def performance_gate_result(
    throughput_fps,
    minimum_fps,
    measured_extent,
    requested_extent,
    minimum_pixel_ratio,
    *,
    allow_failure=False,
    override_reason="",
):
    """Evaluate a renderer benchmark without importing a window integration."""
    requested_pixels = max(requested_extent[0] * requested_extent[1], 1)
    measured_pixels = measured_extent[0] * measured_extent[1]
    pixel_ratio = measured_pixels / requested_pixels
    failures = []
    if throughput_fps < minimum_fps:
        failures.append(
            f"throughput {throughput_fps:.2f} FPS < {minimum_fps:.2f} FPS"
        )
    if pixel_ratio < minimum_pixel_ratio:
        failures.append(
            f"pixel ratio {pixel_ratio:.4f} < {minimum_pixel_ratio:.4f} "
            f"({measured_extent[0]}x{measured_extent[1]} versus "
            f"{requested_extent[0]}x{requested_extent[1]})"
        )
    overridden = bool(failures and allow_failure and override_reason.strip())
    return {
        "status": "override" if overridden else ("fail" if failures else "pass"),
        "failures": failures,
        "override_reason": override_reason.strip() if overridden else "",
        "throughput_fps": throughput_fps,
        "minimum_fps": minimum_fps,
        "measured_extent": list(measured_extent),
        "requested_extent": list(requested_extent),
        "pixel_ratio": pixel_ratio,
        "minimum_pixel_ratio": minimum_pixel_ratio,
    }


@material
def _parity_diffuse(ctx):
    direction = cosine_sample_hemisphere(ctx.normal, ctx.random_u, ctx.random_v)
    pdf = maximum(dot(ctx.normal, direction), 0.0) / math.pi
    return SurfaceResponse(
        emission=ctx.emission,
        weight=ctx.base_color * pdf,
        next_direction=direction,
        event=SCATTER_DIFFUSE,
        pdf=pdf,
    )


@material
def _parity_mirror(ctx):
    return SurfaceResponse(
        emission=ctx.emission,
        weight=ctx.base_color,
        next_direction=reflect(ctx.direction, ctx.normal),
        event=SCATTER_REFLECTION,
        pdf=1.0,
    )


@material
def _parity_glass(ctx):
    cosine = -dot(ctx.direction, ctx.normal)
    fresnel = fresnel_schlick(cosine, ctx.current_ior, ctx.exterior_ior)
    transmitted = refract(
        ctx.direction, ctx.normal, ctx.current_ior / ctx.exterior_ior
    )
    total_internal_reflection = dot(transmitted, transmitted) < 0.01
    probability = select(total_internal_reflection, 1.0, fresnel)
    reflect_path = (ctx.random_u < probability) | total_internal_reflection
    selected_pdf = select(reflect_path, probability, 1.0 - probability)
    return SurfaceResponse(
        emission=ctx.emission,
        weight=ctx.base_color * selected_pdf,
        next_direction=select(
            reflect_path, reflect(ctx.direction, ctx.normal), transmitted
        ),
        event=select(reflect_path, SCATTER_REFLECTION, SCATTER_TRANSMISSION),
        pdf=selected_pdf,
    )


def _quad(a, b, c, d):
    return np.asarray((a, b, c, d), np.float32), np.asarray(
        ((0, 1, 2), (0, 2, 3)), np.uint32
    )


def _sphere(center, radius, rings=20, segments=40):
    vertices = []
    for ring in range(rings + 1):
        theta = math.pi * ring / rings
        for segment in range(segments):
            phi = 2.0 * math.pi * segment / segments
            vertices.append((
                center[0] + radius * math.sin(theta) * math.cos(phi),
                center[1] + radius * math.cos(theta),
                center[2] + radius * math.sin(theta) * math.sin(phi),
            ))
    indices = []
    for ring in range(rings):
        for segment in range(segments):
            following = (segment + 1) % segments
            a = ring * segments + segment
            b = ring * segments + following
            c = (ring + 1) * segments + following
            d = (ring + 1) * segments + segment
            if ring > 0:
                indices.append((a, b, d))
            if ring + 1 < rings:
                indices.append((b, c, d))
    return np.asarray(vertices, np.float32), np.asarray(indices, np.uint32)


def build_feature_parity_scene():
    """Return a fixed scene stressing diffuse, metal, emission, and nested glass."""
    scene = Scene()
    checker = Texture(np.asarray((
        ((235, 225, 205, 255), (72, 92, 128, 255)),
        ((72, 92, 128, 255), (235, 225, 205, 255)),
    ), dtype=np.uint8), wrap_s="repeat", wrap_t="repeat", linear_filter=False)
    metallic_roughness_map = Texture(np.asarray((
        ((0, 40, 255, 255), (0, 180, 255, 255)),
        ((0, 220, 255, 255), (0, 80, 255, 255)),
    ), dtype=np.uint8), wrap_s="repeat", wrap_t="repeat", linear_filter=True)
    emissive_map = Texture(np.asarray((
        ((255, 110, 40, 255), (255, 230, 160, 255)),
        ((255, 230, 160, 255), (255, 110, 40, 255)),
    ), dtype=np.uint8), wrap_s="repeat", wrap_t="repeat", linear_filter=True)
    normal_map = Texture(np.asarray((
        ((96, 128, 245, 255), (160, 128, 245, 255)),
        ((128, 96, 245, 255), (128, 160, 245, 255)),
    ), dtype=np.uint8), wrap_s="repeat", wrap_t="repeat", linear_filter=True)
    occlusion_map = Texture(np.asarray((
        ((48, 0, 0, 255), (220, 0, 0, 255)),
        ((220, 0, 0, 255), (96, 0, 0, 255)),
    ), dtype=np.uint8), wrap_s="repeat", wrap_t="repeat", linear_filter=True)
    transmission_map = Texture(np.asarray((
        ((96, 0, 0, 255), (255, 0, 0, 255)),
        ((255, 0, 0, 255), (160, 0, 0, 255)),
    ), dtype=np.uint8), wrap_s="repeat", wrap_t="repeat", linear_filter=True)
    surfaces = (
        (((-5, 0, -4), (5, 0, -4), (5, 0, 4), (-5, 0, 4)), (0.68, 0.70, 0.74)),
        (((-5, 0, -4), (-5, 0, 4), (-5, 5, 4), (-5, 5, -4)), (0.18, 0.28, 0.68)),
        (((5, 0, 4), (5, 0, -4), (5, 5, -4), (5, 5, 4)), (0.68, 0.20, 0.14)),
        (((-5, 0, 4), (5, 0, 4), (5, 5, 4), (-5, 5, 4)), (0.70, 0.70, 0.70)),
    )
    for surface_index, (corners, color) in enumerate(surfaces):
        vertices, indices = _quad(*corners)
        scene.add_mesh(
            vertices, indices, Material(
                base_color=color,
                base_color_texture=checker if surface_index == 0 else None,
                base_color_transform=(
                    TextureTransform(
                        offset=(0.17, -0.11), scale=(1.35, 0.8), rotation=0.31
                    ) if surface_index == 0 else TextureTransform()
                ),
                normal_texture=normal_map if surface_index == 0 else None,
                normal_scale=0.65,
                occlusion_texture=occlusion_map if surface_index == 0 else None,
                occlusion_strength=0.7,
                occlusion_transform=(
                    TextureTransform(
                        offset=(-0.08, 0.13), scale=(0.7, 1.4), rotation=-0.22,
                        texcoord_set=1,
                    ) if surface_index == 0 else TextureTransform()
                ),
                program=_parity_diffuse,
            ),
            texcoords=(
                np.asarray(((0, 0), (4, 0), (4, 4), (0, 4)), np.float32)
                if surface_index == 0 else None
            ),
            texcoords1=(
                np.asarray(((0, 0), (2, 0), (2, 3), (0, 3)), np.float32)
                if surface_index == 0 else None
            ),
        )

    vertices, indices = _quad(
        (-1.8, 4.85, -1.0), (1.8, 4.85, -1.0),
        (1.8, 4.85, 1.0), (-1.8, 4.85, 1.0),
    )
    scene.add_mesh(vertices, indices, Material(
        base_color=(1.0, 0.88, 0.70), emission=(12.0, 8.0, 4.0),
        emission_two_sided=True, emissive_texture=emissive_map,
        program=_parity_diffuse,
    ), texcoords=np.asarray(((0, 0), (2, 0), (2, 2), (0, 2)), np.float32))

    vertices, indices = _sphere((-2.15, 1.15, 0.35), 1.15)
    scene.add_mesh(vertices, indices, Material(
        base_color=(0.94, 0.72, 0.22), metallic=1.0, roughness=0.0,
        metallic_roughness_texture=metallic_roughness_map, program=_parity_mirror,
    ))

    # Concentric media deliberately exercise the complete nested IOR stack.
    for glass_index, (radius, ior, color) in enumerate((
        (1.35, 1.52, (0.88, 0.96, 1.0)),
        (0.78, 1.25, (0.96, 0.84, 1.0)),
        (0.36, 1.62, (0.84, 1.0, 0.90)),
    )):
        vertices, indices = _sphere((1.65, 1.35, 0.0), radius)
        scene.add_mesh(vertices, indices, Material(
            base_color=color, transmission=1.0, ior=ior,
            transmission_texture=transmission_map if glass_index == 0 else None,
            transmission_transform=(
                TextureTransform(scale=(1.7, 0.8), rotation=0.19)
                if glass_index == 0 else TextureTransform()
            ),
            program=_parity_glass,
        ))
    return scene


def feature_parity_camera():
    return PerspectiveCamera(
        position=(0.0, 2.45, -8.8), target=(0.0, 1.65, 0.15),
        vertical_fov_degrees=42.0,
    )


def image_error_metrics(reference, candidate):
    """Return exposure-aware absolute and relative HDR error statistics."""
    reference = np.asarray(reference, dtype=np.float64)[..., :3]
    candidate = np.asarray(candidate, dtype=np.float64)[..., :3]
    if reference.shape != candidate.shape:
        raise ValueError("images must have matching shapes")
    difference = candidate - reference
    rmse = float(np.sqrt(np.mean(difference * difference)))
    scale = float(np.sqrt(np.mean(reference * reference)))
    return {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": rmse,
        "relative_rmse": rmse / max(scale, 1e-8),
        "max_absolute": float(np.max(np.abs(difference))),
    }


def renderer_visual_metrics(
    reference, candidate, *, reference_mask=None, candidate_mask=None,
):
    """Compare rendering techniques without demanding path-identical HDR."""
    reference = np.asarray(reference, np.float64)[..., :3]
    candidate = np.asarray(candidate, np.float64)[..., :3]
    if reference.shape != candidate.shape or reference.ndim != 3:
        raise ValueError("images must be matching HxWxRGB(A) arrays")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(candidate)):
        raise ValueError("images must contain finite values")
    weights = np.asarray((0.2126, 0.7152, 0.0722), np.float64)
    reference_luminance = reference @ weights
    candidate_luminance = candidate @ weights
    denominator = float(np.sum(candidate_luminance * candidate_luminance))
    exposure = (
        float(np.sum(reference_luminance * candidate_luminance)) / denominator
        if denominator > 1e-12 else 1.0
    )
    exposure = float(np.clip(exposure, 1e-4, 1e4))
    matched = np.maximum(candidate * exposure, 0.0)
    matched_luminance = matched @ weights
    log_difference = np.log1p(matched_luminance) - np.log1p(
        np.maximum(reference_luminance, 0.0)
    )
    reference_edges = np.hypot(
        np.diff(reference_luminance, axis=1, append=reference_luminance[:, -1:]),
        np.diff(reference_luminance, axis=0, append=reference_luminance[-1:, :]),
    )
    candidate_edges = np.hypot(
        np.diff(matched_luminance, axis=1, append=matched_luminance[:, -1:]),
        np.diff(matched_luminance, axis=0, append=matched_luminance[-1:, :]),
    )
    reference_centered = reference_edges - reference_edges.mean()
    candidate_centered = candidate_edges - candidate_edges.mean()
    edge_denominator = float(
        np.linalg.norm(reference_centered) * np.linalg.norm(candidate_centered)
    )
    edge_correlation = (
        float(np.sum(reference_centered * candidate_centered) / edge_denominator)
        if edge_denominator > 1e-12 else 1.0
    )
    if reference_mask is None:
        reference_mask = reference_luminance > max(
            float(np.quantile(reference_luminance, 0.05)) * 1.25, 1e-5,
        )
    if candidate_mask is None:
        candidate_mask = matched_luminance > max(
            float(np.quantile(matched_luminance, 0.05)) * 1.25, 1e-5,
        )
    reference_mask = np.asarray(reference_mask, bool)
    candidate_mask = np.asarray(candidate_mask, bool)
    if (
        reference_mask.shape != reference.shape[:2]
        or candidate_mask.shape != candidate.shape[:2]
    ):
        raise ValueError("coverage masks must match image height and width")
    union = np.count_nonzero(reference_mask | candidate_mask)
    coverage_iou = (
        float(np.count_nonzero(reference_mask & candidate_mask) / union)
        if union else 1.0
    )
    color_difference = (
        np.log1p(matched) - np.log1p(np.maximum(reference, 0.0))
    )
    return {
        "exposure_scale": exposure,
        "log_luminance_rmse": float(np.sqrt(np.mean(log_difference ** 2))),
        "log_color_rmse": float(np.sqrt(np.mean(color_difference ** 2))),
        "edge_correlation": edge_correlation,
        "coverage_iou": coverage_iou,
    }
