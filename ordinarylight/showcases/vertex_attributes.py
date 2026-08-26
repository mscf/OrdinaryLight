"""Showcase interpolated vertex normals and texture coordinates.

The two spheres use the same topology and Python-authored UV material.  The
left sphere supplies smooth radial normals; the right sphere duplicates each
triangle and supplies a face normal, making the normal interpolation difference
easy to see without requiring a texture asset.
"""

import math
import numpy as np

import ordinarylight as ol
from .materials import diffuse, quad


@ol.material
def uv_quadrants(ctx):
    """A procedural material that makes both interpolated UV axes visible."""
    right = ctx.uv.x >= 0.5
    upper = ctx.uv.y >= 0.5
    lower_color = ol.select(
        right, ol.vec3(1.0, 0.22, 0.08), ol.vec3(0.08, 0.38, 1.0)
    )
    upper_color = ol.select(
        right, ol.vec3(1.0, 0.82, 0.08), ol.vec3(0.12, 1.0, 0.34)
    )
    color = ol.select(upper, upper_color, lower_color)
    direction = ol.cosine_sample_hemisphere(ctx.normal, ctx.random_u, ctx.random_v)
    pdf = ol.maximum(ol.dot(ctx.normal, direction), 0.0) / math.pi
    return ol.SurfaceResponse(
        emission=ctx.emission,
        weight=color * pdf,
        next_direction=direction,
        event=ol.SCATTER_DIFFUSE,
        pdf=pdf,
    )


def uv_sphere(center, radius, *, rings=16, segments=32, smooth=True):
    """Return a UV sphere with explicit texture coordinates and normals."""
    vertices = []
    normals = []
    texcoords = []
    for ring in range(rings + 1):
        v = ring / rings
        theta = math.pi * v
        for segment in range(segments + 1):
            u = segment / segments
            phi = 2.0 * math.pi * u
            normal = (
                math.sin(theta) * math.cos(phi),
                math.cos(theta),
                math.sin(theta) * math.sin(phi),
            )
            vertices.append(tuple(center[i] + radius * normal[i] for i in range(3)))
            normals.append(normal)
            texcoords.append((u, 1.0 - v))

    indices = []
    stride = segments + 1
    for ring in range(rings):
        for segment in range(segments):
            a = ring * stride + segment
            b = a + 1
            d = (ring + 1) * stride + segment
            c = d + 1
            if ring > 0:
                indices.append((a, b, d))
            if ring + 1 < rings:
                indices.append((b, c, d))

    vertices = np.asarray(vertices, np.float32)
    indices = np.asarray(indices, np.uint32)
    normals = np.asarray(normals, np.float32)
    texcoords = np.asarray(texcoords, np.float32)
    if smooth:
        return vertices, indices, normals, texcoords

    # Duplicating the vertices prevents interpolation across triangle edges.
    flat_vertices = vertices[indices].reshape(-1, 3)
    flat_texcoords = texcoords[indices].reshape(-1, 2)
    edges_a = flat_vertices[1::3] - flat_vertices[0::3]
    edges_b = flat_vertices[2::3] - flat_vertices[0::3]
    face_normals = np.cross(edges_a, edges_b)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)
    flat_normals = np.repeat(face_normals, 3, axis=0).astype(np.float32)
    flat_indices = np.arange(len(flat_vertices), dtype=np.uint32).reshape(-1, 3)
    return flat_vertices, flat_indices, flat_normals, flat_texcoords


def build_vertex_attribute_showcase():
    scene = ol.Scene()
    scene.add_point_light((-4.0, 6.0, 4.0), intensity=105.0)
    scene.add_point_light((4.5, 3.5, -2.0), color=(0.35, 0.55, 1.0), intensity=35.0)

    vertices, indices = quad(
        (-7.0, 0.0, -5.0), (7.0, 0.0, -5.0),
        (7.0, 0.0, 5.0), (-7.0, 0.0, 5.0),
    )
    scene.add_mesh(vertices, indices, ol.Material(
        base_color=(0.58, 0.61, 0.67), roughness=0.7, program=diffuse,
    ))

    material = ol.Material(base_color=(1.0, 1.0, 1.0), roughness=0.55, program=uv_quadrants)
    for center, smooth in (((-2.0, 1.5, 0.0), True), ((2.0, 1.5, 0.0), False)):
        vertices, indices, normals, texcoords = uv_sphere(center, 1.5, smooth=smooth)
        scene.add_mesh(
            vertices, indices, material, normals=normals, texcoords=texcoords
        )
    return scene
