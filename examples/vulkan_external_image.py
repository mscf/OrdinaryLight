"""Export one tone-mapped Vulkan image for CUDA/NVENC interop.

This example validates the Ordinary Light side of the contract. A real CUDA
consumer imports the two opaque FDs instead of closing them directly.
"""

import os

import ordinarylight as ol
from common import scene_and_camera


def main():
    scene, camera = scene_and_camera()
    config = ol.RendererConfig(external_image_interop=True)
    with ol.Renderer(config=config) as renderer:
        frame = renderer.render_gpu(scene, camera, (1280, 720))
        memory_fd = frame.export_memory_fd()
        ready_fd = frame.export_ready_semaphore_fd()
        try:
            print(frame.metadata)
            print(
                "Import memory FD", memory_fd, "and ready semaphore FD",
                ready_fd, "into CUDA"
            )
            # A CUDA implementation waits on ready_fd, maps memory_fd as the
            # described array, converts to NV12, submits NVENC, and synchronizes
            # its stream before leaving this block.
            frame.wait()
        finally:
            # This validation example did not import the descriptors. CUDA
            # assumes ownership after a successful import, so production code
            # must not close successfully imported FDs here.
            os.close(memory_fd)
            os.close(ready_fd)
            frame.close()


if __name__ == "__main__":
    main()
