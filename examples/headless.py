"""Render a linear-HDR NumPy still without a window or Vulkan dependency."""

from pathlib import Path

import numpy as np

import ordinarylight as ol
from examples.common import scene_and_camera


def main():
    scene, camera = scene_and_camera()
    destination = Path("/tmp/ordinarylight-headless.npy")
    with ol.Renderer(
        backend=ol.backends.ReferenceBackend(samples_per_pixel=4, seed=7)
    ) as renderer:
        hdr = renderer.render(scene, camera, (320, 180))
    np.save(destination, hdr)
    print(f"Wrote linear HDR {hdr.dtype} {hdr.shape} to {destination}")


if __name__ == "__main__":
    main()
