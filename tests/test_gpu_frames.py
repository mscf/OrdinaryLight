import os
import unittest

import numpy as np

import ordinarylight as ol


class GpuBackend:
    available_outputs = ("color",)
    config = object()
    device = "gpu"
    last_timings = {"gpu_ms": 1.25}

    def __init__(self):
        self.closed = False
        self.released = False
        self.external_release = False

    def render_frame(self, scene, camera, width, height, **_options):
        return np.zeros((height, width, 4), np.float32)

    def render_gpu_frame(
        self, scene, camera, width, height, *, samples=None, frame_index=0,
        pixel_format="rgba8",
    ):
        metadata = ol.VulkanImageMetadata(
            width=width, height=height,
            format="VK_FORMAT_R8G8B8A8_UNORM", format_value=37,
            layout="VK_IMAGE_LAYOUT_GENERAL", memory_size=4096,
            memory_offset=0, dedicated_allocation=True,
            device_uuid="00" * 16, image_handle=1, device_handle=2,
            memory_handle=5, physical_device_handle=3,
            completion_fence_handle=4, queue_family_index=0,
        )
        return ol.GpuFrame(
            api="vulkan", metadata=metadata,
            export_memory_fd=lambda: os.open(os.devnull, os.O_RDONLY),
            export_ready_semaphore_fd=(
                lambda: os.open(os.devnull, os.O_RDONLY)
            ),
            wait=lambda timeout: timeout is None or timeout >= 0.01,
            export_release_semaphore_fd=(
                lambda: os.open(os.devnull, os.O_RDONLY)
            ),
            close=self._release,
            attributes={
                "frame_index": frame_index, "pixel_format": pixel_format,
            },
        )

    def close(self):
        self.closed = True

    def _release(self, external=False):
        self.released = True
        self.external_release = bool(external)


def fixture():
    scene = ol.Scene()
    scene.add_mesh(
        ((-1, 0, 0), (1, 0, 0), (0, 1, 0)), ((0, 1, 2),),
    )
    return scene, ol.PerspectiveCamera((0, 0, -3), (0, 0, 0))


class GpuFrameTests(unittest.TestCase):
    def test_renderer_returns_owned_gpu_frame_without_numpy_conversion(self):
        backend = GpuBackend()
        renderer = ol.Renderer(backend=backend)
        scene, camera = fixture()
        frame = renderer.render_gpu(scene, camera, (64, 32))
        self.assertIsInstance(backend, ol.GpuRenderBackend)
        self.assertIsInstance(frame, ol.GpuFrame)
        self.assertEqual((frame.metadata.width, frame.metadata.height), (64, 32))
        self.assertEqual(renderer.frame_index, 1)
        self.assertEqual(renderer.last_statistics.gpu_ms, 1.25)
        self.assertTrue(frame.wait())
        descriptors = (
            frame.export_memory_fd(), frame.export_ready_semaphore_fd()
        )
        for descriptor in descriptors:
            os.close(descriptor)
        with self.assertRaisesRegex(RuntimeError, "already exported"):
            frame.export_memory_fd()
        frame.close()
        self.assertTrue(backend.released)
        self.assertFalse(backend.external_release)
        renderer.close()

    def test_frame_close_is_idempotent_and_prevents_export(self):
        backend = GpuBackend()
        scene, camera = fixture()
        with ol.Renderer(backend=backend) as renderer:
            frame = renderer.render_gpu(scene, camera, (8, 4))
            frame.close()
            frame.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                frame.export_memory_fd()

    def test_cached_release_semaphore_can_release_without_reexport(self):
        backend = GpuBackend()
        scene, camera = fixture()
        with ol.Renderer(backend=backend) as renderer:
            frame = renderer.render_gpu(scene, camera, (8, 4))
            descriptor = frame.export_release_semaphore_fd()
            os.close(descriptor)
            frame.mark_external_release_scheduled()
            frame.close()
        self.assertTrue(backend.released)
        self.assertTrue(backend.external_release)

    def test_renderer_forwards_gpu_pixel_format(self):
        backend = GpuBackend()
        scene, camera = fixture()
        with ol.Renderer(backend=backend) as renderer:
            frame = renderer.render_gpu(
                scene, camera, (8, 4), pixel_format="nv12"
            )
            self.assertEqual(frame.attributes["pixel_format"], "nv12")
            frame.close()

    def test_backend_without_gpu_output_reports_capability_error(self):
        class CpuBackend(GpuBackend):
            render_gpu_frame = None

        scene, camera = fixture()
        with ol.Renderer(backend=CpuBackend()) as renderer:
            with self.assertRaisesRegex(RuntimeError, "GPU-resident"):
                renderer.render_gpu(scene, camera, (8, 4))


if __name__ == "__main__":
    unittest.main()
