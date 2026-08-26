"""Reusable point, line, and mesh-glyph showcase scenes."""

import numpy as np

import ordinarylight as ol


def flat_shaded_pyramid():
    """Return a correctly wound square pyramid with per-face normals."""
    apex = np.asarray((0.0, 0.8, 0.0), np.float32)
    bottom_left = np.asarray((-0.45, 0.0, -0.45), np.float32)
    bottom_right = np.asarray((0.45, 0.0, -0.45), np.float32)
    top_right = np.asarray((0.45, 0.0, 0.45), np.float32)
    top_left = np.asarray((-0.45, 0.0, 0.45), np.float32)
    faces = (
        (apex, bottom_right, bottom_left),
        (apex, top_right, bottom_right),
        (apex, top_left, top_right),
        (apex, bottom_left, top_left),
        (bottom_left, bottom_right, top_right),
        (bottom_left, top_right, top_left),
    )
    vertices = np.asarray(faces, np.float32).reshape((-1, 3))
    indices = np.arange(len(vertices), dtype=np.uint32).reshape((-1, 3))
    edge_a = vertices[indices[:, 1]] - vertices[indices[:, 0]]
    edge_b = vertices[indices[:, 2]] - vertices[indices[:, 0]]
    face_normals = np.cross(edge_a, edge_b)
    face_normals /= np.linalg.norm(face_normals, axis=1)[:, None]
    normals = np.repeat(face_normals, 3, axis=0).astype(np.float32)
    return vertices, indices, normals


def build_primitive_showcase():
    scene = ol.Scene()
    floor = np.asarray((
        (-6, 0, -5), (6, 0, -5), (6, 0, 5), (-6, 0, 5),
    ), np.float32)
    scene.add_mesh(
        floor, ((0, 1, 2), (0, 2, 3)),
        ol.Material(base_color=(0.16, 0.19, 0.24), roughness=0.82),
        name="floor",
    )
    emitter = np.asarray((
        (-2.4, 5.5, -0.8), (2.4, 5.5, -0.8),
        (2.4, 5.5, 0.8), (-2.4, 5.5, 0.8),
    ), np.float32)
    scene.add_mesh(
        emitter, ((0, 1, 2), (0, 2, 3)),
        ol.Material(
            base_color=(1.0, 0.88, 0.65), emission=(13.0, 10.0, 7.0),
            emission_two_sided=True,
        ),
        name="area-light",
    )

    count = 32
    phase = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    points = np.column_stack((
        3.4 * np.cos(phase),
        1.7 + 0.8 * np.sin(phase * 2.0),
        3.4 * np.sin(phase),
    )).astype(np.float32)
    colors = [
        ol.Material(
            base_color=(
                0.35 + 0.45 * (0.5 + 0.5 * np.cos(angle)),
                0.30 + 0.45 * (0.5 + 0.5 * np.cos(angle + 2.1)),
                0.35 + 0.45 * (0.5 + 0.5 * np.cos(angle + 4.2)),
            ),
            metallic=0.15, roughness=0.28,
        )
        for angle in phase
    ]
    scene.add_points(
        points, radii=0.16 + 0.055 * (1.0 + np.sin(phase * 3.0)),
        materials=colors, names=[f"point-{index}" for index in range(count)],
    )
    scene.add_lines(
        points, np.roll(points, -1, axis=0), radii=0.035,
        material=ol.Material(
            base_color=(0.62, 0.70, 0.88), metallic=0.7, roughness=0.22,
        ),
        names=[f"edge-{index}" for index in range(count)],
    )

    vertices, indices, normals = flat_shaded_pyramid()
    glyph = scene.create_mesh(
        vertices, indices,
        ol.Material(base_color=(0.92, 0.52, 0.16), roughness=0.38),
        normals=normals, name="pyramid-glyph",
    )
    glyph_phase = phase[::4]
    transforms = np.stack([
        (
            ol.Transform.translation((
                2.1 * np.cos(angle), 0.02, 2.1 * np.sin(angle)
            ))
            @ ol.Transform.rotation((0, 1, 0), -float(angle))
            @ ol.Transform.scale(0.65)
        ).matrix
        for angle in glyph_phase
    ])
    scene.add_glyphs(glyph, transforms, names=[
        f"glyph-{index}" for index in range(len(transforms))
    ])
    return scene
