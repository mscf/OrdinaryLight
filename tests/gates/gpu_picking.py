"""Live Vulkan gate for compact resident-TLAS picking."""

from __future__ import annotations

import json

import ordinarylight as ol


def main():
    scene = ol.Scene()
    target = scene.add_mesh(
        ((-1, -1, 0), (1, -1, 0), (0, 1, 0)), ((0, 1, 2),),
        name="gpu-pick-target",
    )
    camera = ol.PerspectiveCamera((0, 0, 3), (0, 0, 0))
    with ol.Renderer(config=ol.RendererConfig(max_bounces=1)) as renderer:
        hit = renderer.pick(scene, camera, (101, 101), (50, 50))
        cold_ms = renderer.last_statistics.timings["pick_ms"]
        job = renderer.pick_async(scene, camera, (101, 101), (50, 50))
        warm_hit = job.result()
        statistics = job.statistics
        if hit is None or hit.object_id != target.id:
            raise RuntimeError("GPU picking did not return the center target")
        if "gpu_picking" not in renderer.capabilities.features:
            raise RuntimeError("Vulkan backend did not advertise GPU picking")
        if warm_hit is None or warm_hit.object_id != target.id:
            raise RuntimeError("warm asynchronous GPU pick changed its result")
        print(json.dumps({
            "object_id": hit.object_id,
            "object_name": hit.object.name,
            "distance": hit.distance,
            "barycentric": hit.barycentric,
            "cold_pick_ms": cold_ms,
            "warm_pick_ms": statistics.timings["pick_ms"],
        }, indent=2))


if __name__ == "__main__":
    main()
