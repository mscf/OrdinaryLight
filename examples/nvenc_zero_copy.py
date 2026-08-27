"""Encode an Ordinary Light render without moving pixels through the CPU."""

import ordinarylight as ol

from common import scene_and_camera


def main():
    size = (1280, 720)
    scene, camera = scene_and_camera()
    config = ol.RendererConfig(external_image_interop=True)
    with ol.Renderer(config=config) as renderer, \
            ol.outputs.NvencVideoWriter(
                "ordinarylight-zero-copy.h264", size, fps=30
            ) as video:
        for index in range(120):
            frame = renderer.render_gpu(
                scene, camera, size, frame_index=index,
                pixel_format="nv12",
            )
            video.write(frame)
    print(
        f"encoded {video.frame_count} frames / {video.byte_count} bytes "
        "without host pixel readback"
    )


if __name__ == "__main__":
    main()
