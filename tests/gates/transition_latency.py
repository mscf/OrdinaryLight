"""Opt-in resident scene/settings transition and startup latency gate."""

from __future__ import annotations

import argparse
import json
import time

import ordinarylight as ol


def _scene(color, light_x):
    scene = ol.Scene()
    scene.add_mesh(
        ((-2, -1, 0), (2, -1, 0), (0, 2, 0)), ((0, 1, 2),),
        ol.Material(base_color=color),
    )
    scene.add_point_light((light_x, 3, -2), intensity=25.0)
    return scene


def _render(renderer, scene, camera, extent, frame_index):
    started = time.perf_counter()
    with renderer.render_gpu(
        scene, camera, extent, frame_index=frame_index, pixel_format="p010",
    ) as frame:
        frame.wait()
        handle = frame.metadata.buffer_handle
        device = frame.metadata.device_handle
    return (time.perf_counter() - started) * 1000.0, handle, device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--maximum-startup-ms", type=float, default=10_000.0)
    parser.add_argument("--maximum-transition-ms", type=float, default=1_000.0)
    args = parser.parse_args()
    extent = (args.width, args.height)
    camera = ol.PerspectiveCamera((0, 0, -4), (0, 0, 0))
    first = _scene((0.8, 0.2, 0.08), 2.0)
    second = _scene((0.1, 0.55, 0.9), -2.0)

    started = time.perf_counter()
    renderer = ol.Renderer(config=ol.RendererConfig(
        external_image_interop=True,
        wavefront_execution_strategy="wavefront",
    ))
    constructor_ms = (time.perf_counter() - started) * 1000.0
    try:
        first_ms, first_handle, first_device = _render(
            renderer, first, camera, extent, 0
        )
        _warm_ms, second_handle, _device = _render(
            renderer, first, camera, extent, 1
        )
        started = time.perf_counter()
        renderer.replace_scene(second)
        replace_scene_ms = (time.perf_counter() - started) * 1000.0
        switched_ms, switched_handle, switched_device = _render(
            renderer, second, camera, extent, 0
        )
        _switched_ms, switched_second_handle, _device = _render(
            renderer, second, camera, extent, 1
        )
        started = time.perf_counter()
        renderer.reconfigure(
            samples_per_pixel=2, max_bounces=6, wavefront_exposure=1.1,
        )
        reconfigure_ms = (time.perf_counter() - started) * 1000.0
        configured_ms, configured_handle, configured_device = _render(
            renderer, second, camera, extent, 2
        )
    finally:
        renderer.close()

    pool_preserved = (
        (first_handle, second_handle)
        == (switched_handle, switched_second_handle)
        and configured_handle in {first_handle, second_handle}
    )
    report = {
        "extent": list(extent),
        "constructor_ms": constructor_ms,
        "first_render_ms": first_ms,
        "startup_ms": constructor_ms + first_ms,
        "replace_scene_ms": replace_scene_ms,
        "first_switched_render_ms": switched_ms,
        "reconfigure_ms": reconfigure_ms,
        "first_reconfigured_render_ms": configured_ms,
        "device_preserved": (
            first_device == switched_device == configured_device
        ),
        "external_frame_pool_preserved": pool_preserved,
        "maximum_startup_ms": args.maximum_startup_ms,
        "maximum_transition_ms": args.maximum_transition_ms,
    }
    print(json.dumps(report, indent=2))
    if not report["device_preserved"]:
        raise SystemExit("FAIL: Vulkan device changed across transition")
    if not pool_preserved:
        raise SystemExit("FAIL: external video frame pool was recreated")
    if report["startup_ms"] > args.maximum_startup_ms:
        raise SystemExit("FAIL: startup latency exceeds threshold")
    if max(replace_scene_ms, reconfigure_ms) > args.maximum_transition_ms:
        raise SystemExit("FAIL: resident transition latency exceeds threshold")
    print("PASS: resident transitions preserve Vulkan and video resources")


if __name__ == "__main__":
    main()
