"""Render a small scene through Ordinary Light's reference path tracer."""

import argparse
import time
from pathlib import Path

import numpy as np

import ordinarylight as ol


def quad(a, b, c, d):
    return np.asarray((a, b, c, d), dtype=np.float32), np.asarray(((0, 1, 2), (0, 2, 3)))


def build_scene():
    scene = ol.Scene()
    ground_vertices, ground_indices = quad(
        (-4, 0, -4), (4, 0, -4), (4, 0, 4), (-4, 0, 4)
    )
    scene.add_mesh(
        ground_vertices,
        ground_indices,
        ol.Material(base_color=(0.65, 0.65, 0.65)),
    )

    # A colored tetrahedron exercises multiple triangles and bounces.
    vertices = np.asarray(
        ((0, 1.8, 0), (-1.2, 0, -0.9), (1.2, 0, -0.9), (0, 0, 1.2)),
        dtype=np.float32,
    )
    indices = np.asarray(((0, 2, 1), (0, 3, 2), (0, 1, 3), (1, 2, 3)))
    scene.add_mesh(vertices, indices, ol.Material(base_color=(0.8, 0.18, 0.08)))
    return scene


def write_ppm(path, rgba):
    height, width, _ = rgba.shape
    with open(path, "wb") as output:
        output.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        output.write(rgba[..., :3].tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/ordinarylight_reference.ppm"),
    )
    args = parser.parse_args()
    scene = build_scene()
    camera = ol.PerspectiveCamera(
        position=(4.5, 2.8, 5.5),
        target=(0.0, 0.7, 0.0),
        vertical_fov_degrees=42.0,
    )
    start = time.perf_counter()
    image = ol.ReferencePathTracer(seed=7).render(
        scene, camera, width=320, height=180, samples=8, max_bounces=3
    )
    elapsed = time.perf_counter() - start
    write_ppm(args.output, image)
    print(f"Wrote {args.output} in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
