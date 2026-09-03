import time
import unittest
from threading import Event, get_ident

from ordinarylight.integrations.presentation import AsyncPresenter


class FakePresenter:
    def __init__(self, config):
        self.config = config
        self.last_timings = {"gpu_ms": 3.0}
        self.effective_samples_per_pixel = 1
        self.closed = False
        self.reset_count = 0
        self.object_effect = None
        self.thread_ids = []

    def present_wavefront(self, scene, camera, width, height):
        self.thread_ids.append(get_ident())

    def reset_accumulation(self):
        self.thread_ids.append(get_ident())
        self.reset_count += 1

    def set_object_effects(self, bindings):
        self.thread_ids.append(get_ident())
        self.object_effect = tuple(bindings)

    def close(self):
        self.thread_ids.append(get_ident())
        self.closed = True


def wait_event(worker, kind, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for event in worker.poll():
            if event.kind == kind:
                return event
        time.sleep(0.001)
    raise AssertionError(f"timed out waiting for {kind!r}")


class AsyncPresenterTests(unittest.TestCase):
    def test_lifecycle_and_frames_stay_on_one_worker(self):
        created = []

        def factory(config):
            presenter = FakePresenter(config)
            created.append(presenter)
            return presenter

        worker = AsyncPresenter(factory)
        caller = get_ident()
        generation = worker.restart("first")
        self.assertEqual(wait_event(worker, "ready").generation, generation)
        self.assertTrue(worker.request_frame(object(), object(), (32, 16)))
        self.assertFalse(worker.request_frame(object(), object(), (32, 16)))
        frame = wait_event(worker, "frame")
        self.assertEqual(
            (frame.statistics.width, frame.statistics.height), (32, 16)
        )
        self.assertEqual(frame.statistics.gpu_ms, 3.0)
        effect = object()
        worker.set_object_effect((3, 7), effect)
        self.assertTrue(worker.request_frame(object(), object(), (32, 16)))
        wait_event(worker, "frame")
        self.assertEqual(created[0].object_effect, (((3, 7), effect),))
        worker.set_object_effects((((8, 10), effect), ((12, 14), effect)))
        self.assertTrue(worker.request_frame(object(), object(), (32, 16)))
        wait_event(worker, "frame")
        self.assertEqual(len(created[0].object_effect), 2)
        worker.reset()
        generation = worker.restart("second")
        self.assertEqual(wait_event(worker, "ready").generation, generation)
        self.assertTrue(worker.close(1.0))
        self.assertEqual(len(created), 2)
        worker_threads = {
            thread for presenter in created for thread in presenter.thread_ids
        }
        self.assertEqual(len(worker_threads), 1)
        self.assertNotIn(caller, worker_threads)
        self.assertTrue(all(presenter.closed for presenter in created))

    def test_factory_error_is_reported_without_cross_thread_raise(self):
        worker = AsyncPresenter(
            lambda _config: (_ for _ in ()).throw(RuntimeError("broken"))
        )
        generation = worker.restart(None)
        event = wait_event(worker, "error")
        self.assertEqual(event.generation, generation)
        self.assertIsInstance(event.error, RuntimeError)
        self.assertFalse(worker.ready)
        self.assertTrue(worker.close(1.0))

    def test_denoiser_configuration_restart_recreates_owned_resources(self):
        created = []

        def factory(config):
            presenter = FakePresenter(config)
            presenter.pipeline_token = object()
            presenter.history_token = object() if config["denoiser"] else None
            created.append(presenter)
            return presenter

        worker = AsyncPresenter(factory)
        first_generation = worker.restart({"denoiser": False})
        self.assertEqual(
            wait_event(worker, "ready").generation, first_generation
        )
        first_pipeline = created[0].pipeline_token
        second_generation = worker.restart({"denoiser": True})
        self.assertEqual(
            wait_event(worker, "ready").generation, second_generation
        )
        self.assertTrue(created[0].closed)
        self.assertIsNot(created[1].pipeline_token, first_pipeline)
        self.assertIsNotNone(created[1].history_token)
        third_generation = worker.restart({"denoiser": False})
        self.assertEqual(
            wait_event(worker, "ready").generation, third_generation
        )
        self.assertTrue(created[1].closed)
        self.assertIsNone(created[2].history_token)
        self.assertTrue(worker.close(1.0))
        self.assertTrue(created[2].closed)


if __name__ == "__main__":
    unittest.main()
