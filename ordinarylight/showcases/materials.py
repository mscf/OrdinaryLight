"""Reusable Python-authored stochastic material showcase scenes."""

import math
import numpy as np

import ordinarylight as ol


@ol.material
def diffuse(ctx):
    direction = ol.cosine_sample_hemisphere(ctx.normal, ctx.random_u, ctx.random_v)
    pdf = ol.maximum(ol.dot(ctx.normal, direction), 0.0) / math.pi
    return ol.SurfaceResponse(
        emission=ctx.emission,
        weight=ctx.base_color * pdf,
        next_direction=direction,
        event=ol.SCATTER_DIFFUSE,
        pdf=pdf,
    )


@ol.material
def mirror(ctx):
    return ol.SurfaceResponse(
        emission=ctx.emission,
        weight=ctx.base_color,
        next_direction=ol.reflect(ctx.direction, ctx.normal),
        event=ol.SCATTER_REFLECTION,
        pdf=1.0,
    )


@ol.material
def fresnel_glass(ctx):
    cosine = -ol.dot(ctx.direction, ctx.normal)
    fresnel = ol.fresnel_schlick(cosine, ctx.current_ior, ctx.exterior_ior)
    eta = ctx.current_ior / ctx.exterior_ior
    transmitted = ol.refract(ctx.direction, ctx.normal, eta)
    total_internal_reflection = ol.dot(transmitted, transmitted) < 0.01
    probability = ol.select(total_internal_reflection, 1.0, fresnel)
    reflect_path = (ctx.random_u < probability) | total_internal_reflection
    selected_pdf = ol.select(reflect_path, probability, 1.0 - probability)
    return ol.SurfaceResponse(
        emission=ctx.emission,
        weight=ctx.base_color * selected_pdf,
        next_direction=ol.select(
            reflect_path,
            ol.reflect(ctx.direction, ctx.normal),
            transmitted,
        ),
        event=ol.select(
            reflect_path,
            ol.SCATTER_REFLECTION,
            ol.SCATTER_TRANSMISSION,
        ),
        pdf=selected_pdf,
    )


def quad(a, b, c, d):
    return np.asarray((a, b, c, d), np.float32), np.asarray(
        ((0, 1, 2), (0, 2, 3)), np.uint32
    )


def sphere(center, radius, rings=24, segments=48):
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


def build_showcase_scene():
    scene = ol.Scene()
    scene.add_point_light(
        (-2.5, 5.5, 3.0), color=(1.0, 0.78, 0.58), intensity=85.0
    )
    scene.add_point_light(
        (3.5, 3.8, -2.5), color=(0.42, 0.62, 1.0), intensity=45.0
    )
    vertices, indices = quad(
        (-7.0, 0.0, -6.0), (7.0, 0.0, -6.0),
        (7.0, 0.0, 6.0), (-7.0, 0.0, 6.0),
    )
    scene.add_mesh(vertices, indices, ol.Material(
        base_color=(0.65, 0.68, 0.72), program=diffuse
    ))

    vertices, indices = sphere((-2.0, 1.25, 0.0), 1.25)
    scene.add_mesh(vertices, indices, ol.Material(
        base_color=(0.92, 0.72, 0.25), metallic=1.0, program=mirror
    ))

    # Concentric transmissive shells exercise the nested-medium IOR stack.
    vertices, indices = sphere((1.7, 1.45, 0.0), 1.45)
    scene.add_mesh(vertices, indices, ol.Material(
        base_color=(0.82, 0.94, 1.0), transmission=1.0, ior=1.52,
        program=fresnel_glass,
    ))
    vertices, indices = sphere((1.7, 1.45, 0.0), 0.72)
    scene.add_mesh(vertices, indices, ol.Material(
        base_color=(0.94, 0.82, 1.0), transmission=1.0, ior=1.25,
        program=fresnel_glass,
    ))
    return scene
