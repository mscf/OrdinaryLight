"""Gate volume single scattering and phase directionality in native HDR."""

import argparse
import json

import numpy as np

import ordinarylight as ol


def _scene(
    scattering_scale, anisotropy, *, point=True, area=False, blocker=False,
):
    scene = ol.Scene()
    density = np.ones((12, 12, 12), np.float32)
    transfer = ol.Texture1D(((0.0, 0.0, 0.0, 0.055),) * 2)
    scene.add_volume(
        density,
        ol.VolumeMaterial(
            transfer, emission_scale=0.0, step_size=0.04,
            scattering_scale=scattering_scale,
            scattering_color=(0.45, 0.7, 1.0),
            phase_function="henyey_greenstein", anisotropy=anisotropy,
        ),
        transform=(
            ol.Transform.translation((-0.6, -0.6, 0.0))
            @ ol.Transform.scale((1.2, 1.2, 1.2))
        ),
        value_range=(0.0, 1.0), name="homogeneous-medium",
    )
    if point:
        # Behind the volume: positive g preferentially scatters toward the camera.
        scene.add_point_light((0.0, 0.0, 2.2), intensity=38.0)
    if area:
        emitter = np.asarray((
            (-1.5, 1.8, -0.2), (1.5, 1.8, -0.2),
            (1.5, 1.8, 1.4), (-1.5, 1.8, 1.4),
        ), np.float32)
        scene.add_mesh(
            emitter, ((0, 1, 2), (0, 2, 3)),
            ol.Material(emission=(4.0, 2.0, 0.7)), name="area-emitter",
        )
    if blocker:
        blocker_vertices = np.asarray((
            (-2.0, -2.0, 1.35), (2.0, -2.0, 1.35),
            (2.0, 2.0, 1.35), (-2.0, 2.0, 1.35),
        ), np.float32)
        scene.add_mesh(
            blocker_vertices, ((0, 1, 2), (0, 2, 3)),
            ol.Material(base_color=(0.01, 0.01, 0.01)), name="opaque-blocker",
        )
    return scene


def _render(renderer, scene, camera, width, height, seed):
    return np.array(renderer.render_wavefront(
        scene, camera, width, height, samples=1, frame_index=seed,
    ), copy=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--min-scattering-effect", type=float, default=0.01)
    parser.add_argument("--min-forward-ratio", type=float, default=1.5)
    parser.add_argument("--max-strategy-relative-rmse", type=float, default=0.01)
    args = parser.parse_args()
    camera = ol.PerspectiveCamera((0.0, 0.0, -2.3), (0.0, 0.0, 0.6))

    def backend(strategy):
        return ol.renderers.gi.VulkanGlobalIlluminationRenderer(config=ol.RendererConfig(
            max_bounces=2, samples_per_pixel=1,
            wavefront_tile_capacity=args.width * args.height,
            wavefront_execution_strategy=strategy,
            wavefront_environment_samples=4,
        ))

    with backend("wavefront") as renderer:
        control = _render(
            renderer, _scene(0.0, 0.65), camera,
            args.width, args.height, args.seed,
        )
        forward = _render(
            renderer, _scene(1.0, 0.65), camera,
            args.width, args.height, args.seed,
        )
        backward = _render(
            renderer, _scene(1.0, -0.65), camera,
            args.width, args.height, args.seed,
        )
        environment_control = _render(
            renderer, _scene(0.0, 0.0, point=False), camera,
            args.width, args.height, args.seed,
        )
        environment_lit = _render(
            renderer, _scene(1.0, 0.0, point=False), camera,
            args.width, args.height, args.seed,
        )
        area_control = _render(
            renderer, _scene(0.0, 0.0, point=False, area=True), camera,
            args.width, args.height, args.seed,
        )
        area_lit = _render(
            renderer, _scene(1.0, 0.0, point=False, area=True), camera,
            args.width, args.height, args.seed,
        )
        unblocked = _render(
            renderer, _scene(1.0, 0.65), camera,
            args.width, args.height, args.seed,
        )
        blocked = _render(
            renderer, _scene(1.0, 0.65, blocker=True), camera,
            args.width, args.height, args.seed,
        )
        blocked_control = _render(
            renderer, _scene(0.0, 0.65, blocker=True), camera,
            args.width, args.height, args.seed,
        )
    with backend("megakernel") as renderer:
        megakernel = _render(
            renderer, _scene(1.0, 0.65), camera,
            args.width, args.height, args.seed,
        )

    center = np.s_[args.height // 4:3 * args.height // 4,
                   args.width // 4:3 * args.width // 4, :3]
    control_luma = float(np.mean(control[center]))
    forward_luma = float(np.mean(forward[center]))
    backward_luma = float(np.mean(backward[center]))
    environment_effect = float(np.mean(
        environment_lit[center] - environment_control[center]
    ))
    area_effect = float(np.mean(area_lit[center] - area_control[center]))
    unblocked_effect = float(np.mean(unblocked[center] - control[center]))
    blocked_effect = float(np.mean(blocked[center] - blocked_control[center]))
    opaque_shadow_ratio = blocked_effect / max(unblocked_effect, 1e-8)
    scattering_effect = forward_luma - control_luma
    forward_ratio = forward_luma / max(backward_luma, 1e-8)
    difference = forward[..., :3] - megakernel[..., :3]
    strategy_relative_rmse = float(
        np.sqrt(np.mean(difference * difference))
        / max(np.sqrt(np.mean(forward[..., :3] ** 2)), 1e-8)
    )
    result = {
        "control_luminance": control_luma,
        "forward_luminance": forward_luma,
        "backward_luminance": backward_luma,
        "scattering_effect": scattering_effect,
        "forward_ratio": forward_ratio,
        "environment_effect": environment_effect,
        "area_light_effect": area_effect,
        "opaque_shadow_ratio": opaque_shadow_ratio,
        "strategy_relative_rmse": strategy_relative_rmse,
    }
    print(json.dumps(result, indent=2))
    if scattering_effect < args.min_scattering_effect:
        raise SystemExit("FAIL: point-light volume scattering is not visible")
    if forward_ratio < args.min_forward_ratio:
        raise SystemExit("FAIL: Henyey--Greenstein forward lobe is too weak")
    if environment_effect < args.min_scattering_effect:
        raise SystemExit("FAIL: environment volume scattering is not visible")
    if area_effect < args.min_scattering_effect:
        raise SystemExit("FAIL: area-light volume scattering is not visible")
    if opaque_shadow_ratio > 0.75:
        raise SystemExit("FAIL: opaque geometry does not shadow volume lighting")
    if strategy_relative_rmse > args.max_strategy_relative_rmse:
        raise SystemExit("FAIL: scattering execution strategies diverge")
    print(
        "PASS: volume scattering, phase directionality, direct-light "
        "coverage, and opaque visibility"
    )


if __name__ == "__main__":
    main()
