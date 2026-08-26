"""Gate bounded higher-order volume scattering in native HDR."""

import argparse
import json

import numpy as np

import ordinarylight as ol


def _scene(scattering_orders, *, opacity=0.08):
    scene = ol.Scene()
    transfer = ol.Texture1D(((0.0, 0.0, 0.0, opacity),) * 2)
    scene.add_volume(
        np.ones((10, 10, 10), np.float32),
        ol.VolumeMaterial(
            transfer, emission_scale=0.0, step_size=0.04,
            scattering_scale=1.0, scattering_color=(0.6, 0.8, 1.0),
            phase_function="isotropic", scattering_albedo=(0.92, 0.94, 0.98),
            scattering_orders=scattering_orders,
        ),
        transform=(
            ol.Transform.translation((-0.6, -0.6, 0.0))
            @ ol.Transform.scale((1.2, 1.2, 1.2))
        ),
        value_range=(0.0, 1.0), name="homogeneous-medium",
    )
    scene.add_point_light((0.0, 0.0, 2.2), intensity=38.0)
    return scene


def _overlap_scene(*, reverse=False):
    scene = ol.Scene()
    definitions = (
        ((-0.72, -0.6, 0.0), (0.9, 0.55, 0.35)),
        ((-0.48, -0.6, 0.08), (0.35, 0.65, 1.0)),
    )
    if reverse:
        definitions = tuple(reversed(definitions))
    transfer = ol.Texture1D(((0.0, 0.0, 0.0, 0.045),) * 2)
    for index, (translation, color) in enumerate(definitions):
        scene.add_volume(
            np.ones((8, 8, 8), np.float32),
            ol.VolumeMaterial(
                transfer, emission_scale=0.0, step_size=0.05,
                scattering_scale=0.75, scattering_color=color,
                scattering_albedo=(0.88, 0.92, 0.96), scattering_orders=3,
            ),
            transform=(
                ol.Transform.translation(translation)
                @ ol.Transform.scale((1.2, 1.2, 1.1))
            ),
            value_range=(0.0, 1.0), name=f"overlap-medium-{index}",
        )
    scene.add_point_light((0.0, 0.0, 2.2), intensity=34.0)
    return scene


def _render(renderer, scene, camera, width, height, seed):
    return np.array(renderer.render_wavefront(
        scene, camera, width, height, samples=1, frame_index=seed,
    ), copy=True)


def _relative_rmse(first, second):
    difference = first[..., :3] - second[..., :3]
    return float(
        np.sqrt(np.mean(difference * difference))
        / max(np.sqrt(np.mean(first[..., :3] ** 2)), 1e-8)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--seed", type=int, default=19)
    parser.add_argument("--min-multiple-ratio", type=float, default=1.25)
    parser.add_argument("--max-strategy-relative-rmse", type=float, default=0.01)
    args = parser.parse_args()
    camera = ol.PerspectiveCamera((0.0, 0.0, -2.3), (0.0, 0.0, 0.6))

    def backend(strategy):
        return ol.VulkanRayTracingBackend(config=ol.RendererConfig(
            max_bounces=2, samples_per_pixel=1,
            wavefront_tile_capacity=args.width * args.height,
            wavefront_execution_strategy=strategy,
            wavefront_environment_samples=4,
        ))

    single_scene = _scene(1)
    multiple_scene = _scene(4)
    with backend("wavefront") as renderer:
        single = _render(
            renderer, single_scene, camera, args.width, args.height, args.seed,
        )
        multiple = _render(
            renderer, multiple_scene, camera, args.width, args.height, args.seed,
        )
        overlap_forward = _render(
            renderer, _overlap_scene(), camera,
            args.width, args.height, args.seed,
        )
        overlap_reverse = _render(
            renderer, _overlap_scene(reverse=True), camera,
            args.width, args.height, args.seed,
        )
    with backend("megakernel") as renderer:
        multiple_megakernel = _render(
            renderer, multiple_scene, camera, args.width, args.height, args.seed,
        )
        overlap_megakernel = _render(
            renderer, _overlap_scene(), camera,
            args.width, args.height, args.seed,
        )

    center = np.s_[args.height // 4:3 * args.height // 4,
                   args.width // 4:3 * args.width // 4, :3]
    single_luminance = float(np.mean(single[center]))
    multiple_luminance = float(np.mean(multiple[center]))
    multiple_ratio = multiple_luminance / max(single_luminance, 1e-8)
    strategy_relative_rmse = _relative_rmse(multiple, multiple_megakernel)
    overlap_order_relative_rmse = _relative_rmse(
        overlap_forward, overlap_reverse,
    )
    overlap_strategy_relative_rmse = _relative_rmse(
        overlap_forward, overlap_megakernel,
    )
    result = {
        "single_scattering_luminance": single_luminance,
        "multiple_scattering_luminance": multiple_luminance,
        "multiple_scattering_ratio": multiple_ratio,
        "strategy_relative_rmse": strategy_relative_rmse,
        "overlap_order_relative_rmse": overlap_order_relative_rmse,
        "overlap_strategy_relative_rmse": overlap_strategy_relative_rmse,
    }
    print(json.dumps(result, indent=2))
    if multiple_ratio < args.min_multiple_ratio:
        raise SystemExit("FAIL: higher scattering orders do not add enough energy")
    if strategy_relative_rmse > args.max_strategy_relative_rmse:
        raise SystemExit("FAIL: multiple-scattering execution strategies diverge")
    if overlap_order_relative_rmse > args.max_strategy_relative_rmse:
        raise SystemExit("FAIL: overlapping multiple scattering is order dependent")
    if overlap_strategy_relative_rmse > args.max_strategy_relative_rmse:
        raise SystemExit(
            "FAIL: overlapping multiple-scattering strategies diverge"
        )
    print("PASS: bounded multiple scattering and execution-strategy parity")


if __name__ == "__main__":
    main()
