import unittest

import ordinarylight as ol
from ordinarylight.integrations.glfw import NativeViewport


class FakeGlfw:
    def __init__(self):
        self.closed = False
        self.extent = (640, 360)
        self.polls = 0

    def poll_events(self): self.polls += 1
    def window_should_close(self, _window): return self.closed
    def set_window_should_close(self, _window, value): self.closed = bool(value)
    def get_framebuffer_size(self, _window): return self.extent
    def wait_events_timeout(self, _timeout): pass
    def destroy_window(self, _window): pass


class FakePresenter:
    def __init__(self):
        self.swapchain_image_count = 0
        self.effective_samples_per_pixel = 1
        self.last_timings = {"gpu_frame_ms": 4.0, "total_ms": 5.0}
        self.calls = []
        self.resets = 0
        self.closed = False

    def present_wavefront(self, scene, camera, width, height):
        self.calls.append((scene, camera, width, height))
        self.swapchain_image_count = 3

    def reset_accumulation(self): self.resets += 1
    def close(self): self.closed = True


def fixture(**options):
    scene = ol.Scene()
    scene.add_mesh(((-1, 0, 0), (1, 0, 0), (0, 1, 0)), ((0, 1, 2),))
    camera = ol.PerspectiveCamera((0, 0, -3), (0, 0, 0))
    glfw, presenter = FakeGlfw(), FakePresenter()
    viewport = NativeViewport(
        scene, camera, config=ol.RendererConfig(), _glfw=glfw,
        _window=object(), _presenter=presenter,
        resize_settle_seconds=0, **options,
    )
    return viewport, glfw, presenter


class NativeViewportTests(unittest.TestCase):
    def test_step_shares_scene_camera_and_returns_statistics(self):
        viewport, glfw, presenter = fixture()
        statistics = viewport.step()
        self.assertIsInstance(statistics, ol.RenderStatistics)
        self.assertEqual(statistics.gpu_ms, 4.0)
        self.assertEqual(statistics.total_ms, 5.0)
        self.assertEqual(statistics.width, 640)
        self.assertEqual(viewport.frame_index, 1)
        self.assertIs(presenter.calls[0][0], viewport.scene)
        self.assertIs(presenter.calls[0][1], viewport.camera)
        self.assertEqual(glfw.polls, 1)

    def test_controller_and_run_callback(self):
        updates, frames = [], []
        viewport, _glfw, _presenter = fixture(
            controller=lambda view, dt: updates.append((view, dt))
        )
        rendered = viewport.run(
            max_frames=2,
            on_frame=lambda view, stats: frames.append((view, stats)),
        )
        self.assertEqual(rendered, 2)
        self.assertEqual(len(updates), 2)
        self.assertEqual(len(frames), 2)

    def test_scene_camera_reset_close_and_close_request(self):
        viewport, glfw, presenter = fixture()
        viewport.step()
        camera = ol.PerspectiveCamera((1, 0, -3), (0, 0, 0))
        viewport.set_camera(camera)
        self.assertIs(viewport.camera, camera)
        self.assertEqual(viewport.frame_index, 0)
        self.assertEqual(presenter.resets, 1)
        viewport.request_close()
        self.assertTrue(viewport.should_close)
        viewport.close()
        viewport.close()
        self.assertTrue(presenter.closed)
        with self.assertRaises(RuntimeError):
            viewport.step()


if __name__ == "__main__":
    unittest.main()
