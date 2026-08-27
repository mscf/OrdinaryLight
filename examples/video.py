"""Stream rendered frames into an optional FFmpeg video sink."""

import math

import ordinarylight as ol
from examples.common import scene_and_camera


def main():
    scene, base_camera = scene_and_camera()
    size = (320, 180)
    with (
        ol.Renderer(
            backend=ol.backends.ReferenceBackend(samples_per_pixel=1, seed=5)
        ) as renderer,
        ol.outputs.FFmpegVideoWriter(
            "/tmp/ordinarylight.mp4", size, fps=24
        ) as video,
    ):
        for frame in range(48):
            angle = 2.0 * math.pi * frame / 48
            camera = ol.PerspectiveCamera(
                (4.0 * math.sin(angle), 0.5, -4.0 * math.cos(angle)),
                base_camera.target,
            )
            video.write(ol.outputs.to_sdr(renderer.render(scene, camera, size)))
    print("Wrote /tmp/ordinarylight.mp4")


if __name__ == "__main__":
    main()
