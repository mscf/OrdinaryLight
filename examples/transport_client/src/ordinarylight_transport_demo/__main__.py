"""Diffuse surfaces, SDF glass and triangle glass through one public transport API."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np
from PIL import Image

import ordinarylight as ol
from ordinarylight.geometry import SdfSphere
from ordinarylight.runtime import VulkanOutput
from ordinarylight.transport import (
    VulkanTransportScene,
    TransportMaterial,
    VulkanTransportIntegrator,
    GpuSampleAccumulator,
    OpticalMedium,
    MediumBoundary,
    ray_samples,
    surface_samples,
)


def glass_box():
    scene = ol.Scene()
    vertices = [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ]
    indices = [
        [4, 5, 6],
        [4, 6, 7],
        [0, 2, 1],
        [0, 3, 2],
        [1, 2, 6],
        [1, 6, 5],
        [0, 4, 7],
        [0, 7, 3],
        [3, 7, 6],
        [3, 6, 2],
        [0, 1, 5],
        [0, 5, 4],
    ]
    mesh = scene.add_mesh(vertices, indices, ol.Material(transmission=1, roughness=0))
    return scene, mesh.id


def run(*, output, frames=4, samples_per_frame=32, columns=64, present=False):
    window = None
    glfw = None
    if present:
        import glfw

        if not glfw.init():
            raise RuntimeError("Cannot initialize GLFW")
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        window = glfw.create_window(
            768, 384, "Diffuse samples / SDF glass / triangle glass", None, None
        )
        if window is None:
            glfw.terminate()
            raise RuntimeError("Cannot create presentation window")
    try:
        with ExitStack() as stack:
            runtime = stack.enter_context(ol.VulkanRuntime(glfw_window=window))
            # Each row owns application IDs; the library does not choose pixels
            # or perform any voxel-specific averaging.
            materials = [
                TransportMaterial("diffuse", (0.25, 0.25, 0.25), (2, 1, 0.5), True)
            ]
            materials += [
                TransportMaterial("diffuse", (float(a),) * 3)
                for a in np.linspace(0.15, 0.8, columns)
            ]
            diffuse = stack.enter_context(
                VulkanTransportScene(
                    runtime,
                    custom_geometry=[SdfSphere().geometry()],
                    custom_materials=materials,
                )
            )
            media = [OpticalMedium(), OpticalMedium(1.5, (0.2, 0.5, 0.8))]
            boundaries = [MediumBoundary(19, 0, 1)]
            sphere = stack.enter_context(
                VulkanTransportScene(
                    runtime,
                    custom_geometry=[SdfSphere().geometry(boundary=19)],
                    custom_materials=[TransportMaterial("dielectric")],
                    media=media,
                    boundaries=boundaries,
                )
            )
            box_scene, box_id = glass_box()
            triangles = stack.enter_context(
                VulkanTransportScene(
                    runtime,
                    box_scene,
                    media=media,
                    boundaries=boundaries,
                    triangle_boundaries={box_id: 19},
                )
            )
            accumulation = stack.enter_context(
                GpuSampleAccumulator(runtime, columns * 3, extent=(columns, 3))
            )
            points = np.zeros((columns, 3))
            normals = np.tile([0, 0, 1], (columns, 1))
            origins = np.column_stack(
                (
                    np.linspace(-1.2, 1.2, columns),
                    np.zeros(columns),
                    np.full(columns, 3),
                )
            )
            directions = np.tile([0, 0, -1], (columns, 1))
            sets = [
                surface_samples(
                    points,
                    normals,
                    materials=np.arange(1, columns + 1),
                    identities=np.arange(columns),
                ),
                ray_samples(
                    origins, directions, identities=np.arange(columns, 2 * columns)
                ),
                ray_samples(
                    origins, directions, identities=np.arange(2 * columns, 3 * columns)
                ),
            ]
            integrators = [
                stack.enter_context(
                    VulkanTransportIntegrator(scene, inputs, accumulation)
                )
                for scene, inputs in zip((diffuse, sphere, triangles), sets)
            ]
            display = stack.enter_context(VulkanOutput(runtime))
            for _ in range(frames):
                for index, integrator in enumerate(integrators):
                    integrator.accumulate(
                        samples_per_element=samples_per_frame,
                        max_bounces=2 if index == 0 else 24,
                        max_steps=2048,
                        environment=(1, 1, 1) if index else (0, 0, 0),
                        seed=73,
                    )
                ready = accumulation.resolve()
                if window is not None:
                    glfw.poll_events()
                    if glfw.window_should_close(window):
                        break
                    display.present(
                        accumulation.hdr,
                        after=ready,
                        surface_size=glfw.get_framebuffer_size(window),
                    )
            records = accumulation.read()
            expected = (
                np.linspace(0.15, 0.8, columns)[:, None] * np.array([2, 1, 0.5]) * 1.25
            )
            np.testing.assert_allclose(
                accumulation.means()[:columns], expected, rtol=3e-5
            )
            with display.tone_map(accumulation.hdr, after=ready) as frame:
                pixels = display.read(
                    frame
                )  # Explicit final file export, not the live presentation path.
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            Image.frombytes("RGBA", (columns, 3), pixels).resize(
                (columns * 8, 240), Image.Resampling.NEAREST
            ).save(output)
            report = {
                "rows": [
                    "two diffuse bounces; exact cavity series",
                    "SDF dielectric sphere",
                    "triangle dielectric box",
                ],
                "samples_per_identity": int(records["counts"][:, 0].min()),
                "invalid_status_mask": int(
                    np.bitwise_or.reduce(records["counts"][:, 2])
                ),
                "truncated_paths_by_row": records["counts"][:, 3]
                .reshape(3, columns)
                .sum(axis=1)
                .tolist(),
                "hdr_row_means": accumulation.means()
                .reshape(3, columns, 3)
                .mean(axis=1)
                .tolist(),
            }
            output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
            np.savez(
                output.with_suffix(".npz"), records=records, hdr=accumulation.means()
            )
            print(json.dumps(report, indent=2))
            return report
    finally:
        if window is not None:
            glfw.destroy_window(window)
        if glfw is not None:
            glfw.terminate()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("transport.png"))
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--columns", type=int, default=64)
    parser.add_argument("--present", action="store_true")
    args = parser.parse_args()
    if min(args.frames, args.samples, args.columns) <= 0:
        parser.error("counts must be positive")
    run(
        output=args.output,
        frames=args.frames,
        samples_per_frame=args.samples,
        columns=args.columns,
        present=args.present,
    )


if __name__ == "__main__":
    main()
