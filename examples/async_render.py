"""Await a renderer submission without blocking an asyncio event loop."""

import asyncio

import ordinarylight as ol
from examples.common import scene_and_camera


async def main():
    scene, camera = scene_and_camera()
    with ol.Renderer(
        backend=ol.backends.ReferenceBackend(samples_per_pixel=2, seed=11)
    ) as renderer:
        job = renderer.render_async(scene, camera, (160, 90))
        while not job.done():
            await asyncio.sleep(0)
        hdr = await job
        print(hdr.shape, job.statistics.as_dict())


if __name__ == "__main__":
    asyncio.run(main())
