import unittest
import asyncio
from threading import Event, get_ident

import numpy as np

import ordinarylight as ol


class FakeBackend:
    available_outputs = (
        "color", "variance", "depth", "normal", "instance_id", "object_id",
        "material_id", "motion",
    )
    def __init__(self):
        self.config = object()
        self.device = "test-device"
        self.last_timings = {"gpu_ms": 2.5}
        self.calls = []
        self.scene_replacements = []
        self.setting_changes = []
        self.object_effect_changes = []
        self.closed = False
        self.accumulation_state = ol.AccumulationState.ACCUMULATING
        self.accumulated_frames = 7
        self.effective_samples_per_pixel = 3

    def render_frame(
        self, scene, camera, width, height, *, samples=None, frame_index=0,
    ):
        self.calls.append((scene, camera, width, height, samples, frame_index))
        return np.full((height, width, 4), frame_index, np.float32)

    def render_products(
        self, scene, camera, width, height, *, outputs, samples=None,
        frame_index=0,
    ):
        self.calls.append((scene, camera, width, height, samples, frame_index))
        products = {
            "color": np.full((height, width, 4), frame_index, np.float32),
            "variance": np.full((height, width), 0.25, np.float32),
            "depth": np.full((height, width), 2.0, np.float32),
            "normal": np.full((height, width, 3), (0, 0, 1), np.float32),
            "instance_id": np.full((height, width), 3, np.uint32),
            "object_id": np.full((height, width), 3, np.uint32),
            "material_id": np.full((height, width), 5, np.uint32),
            "motion": np.zeros((height, width, 2), np.float32),
        }
        return {name: products[name] for name in outputs}

    def close(self):
        self.closed = True

    def replace_scene(self, scene):
        self.scene_replacements.append(scene)

    def reconfigure(self, **changes):
        self.setting_changes.append(changes)
        return changes

    def apply_object_effect(self, scene, reference, effect):
        self.object_effect_changes.append((scene, reference, effect))
        return effect

    def set_object_effects(self, scene, bindings):
        bindings = tuple(bindings)
        self.object_effect_changes.append((scene, bindings))
        return bindings

    def clear_object_effect(self):
        self.object_effect_changes.append(None)


class BlockingBackend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.started = Event()
        self.release = Event()
        self.render_thread = None

    def render_frame(self, *args, **kwargs):
        self.render_thread = get_ident()
        self.started.set()
        self.release.wait(2.0)
        return super().render_frame(*args, **kwargs)


def fixture():
    scene = ol.Scene()
    scene.add_mesh(
        ((-1, 0, 0), (1, 0, 0), (0, 1, 0)), ((0, 1, 2),),
    )
    camera = ol.PerspectiveCamera((0, 0, -3), (0, 0, 0))
    return scene, camera


class RendererTests(unittest.TestCase):
    def test_pick_async_uses_portable_fallback_and_is_awaitable(self):
        scene = ol.Scene()
        mesh = scene.add_mesh(
            ((-1, -1, 0), (1, -1, 0), (0, 1, 0)), ((0, 1, 2),),
        )
        camera = ol.PerspectiveCamera((0, 0, 3), (0, 0, 0))
        with ol.Renderer(backend=FakeBackend()) as renderer:
            job = renderer.pick_async(scene, camera, (101, 101), (50, 50))
            hit = job.result()
            self.assertIs(hit.object, mesh)
            self.assertIsNotNone(job.statistics)
            self.assertGreaterEqual(job.statistics.timings["pick_ms"], 0.0)

    def test_object_effect_response_is_ordered_and_backend_neutral(self):
        backend = FakeBackend()
        renderer = ol.Renderer(backend=backend)
        self.assertIsInstance(backend, ol.ObjectEffectBackend)
        self.assertIsInstance(backend, ol.MultiObjectEffectBackend)
        scene, _camera = fixture()
        mesh = scene.meshes[0]
        effect = ol.effects.Outline(color=(1, 0, 0), width=3)
        self.assertEqual(
            renderer.apply_object_effect(scene, mesh.id, effect),
            ((mesh.id, effect),),
        )
        tint = ol.effects.Tint(color=(0, 1, 0), strength=0.4)
        renderer.set_object_effects(scene, ((mesh, effect), (mesh.id, tint)))
        renderer.clear_object_effect()
        self.assertEqual(
            backend.object_effect_changes,
            [
                (scene, ((mesh.id, effect),)),
                (scene, ((mesh, effect), (mesh.id, tint))),
                None,
            ],
        )
        with self.assertRaises(TypeError):
            renderer.apply_object_effect(scene, mesh, object())
        renderer.close()

    def test_renderer_exposes_backend_accumulation_state(self):
        renderer = ol.Renderer(backend=FakeBackend())
        self.assertEqual(
            renderer.accumulation_state, ol.AccumulationState.ACCUMULATING
        )
        self.assertEqual(renderer.accumulated_frames, 7)
        self.assertEqual(renderer.effective_samples_per_pixel, 3)
        renderer.close()

    def test_backend_contract_is_structural_and_backend_neutral(self):
        backend = FakeBackend()
        self.assertIsInstance(backend, ol.RenderBackend)
        self.assertIsInstance(backend, ol.ProductRenderBackend)

        class WavefrontOnlyBackend:
            def render_wavefront(self, *args, **kwargs):
                return None

            def close(self):
                pass

        with self.assertRaisesRegex(TypeError, "render_frame"):
            ol.Renderer(backend=WavefrontOnlyBackend())

    def test_render_job_is_awaitable(self):
        renderer = ol.Renderer(backend=FakeBackend())
        scene, camera = fixture()

        async def render_one():
            return await renderer.render_async(scene, camera, (4, 2))

        image = asyncio.run(render_one())
        self.assertEqual(image.shape, (2, 4, 4))
        renderer.close()

    def test_async_render_is_nonblocking_ordered_and_reports_statistics(self):
        backend = BlockingBackend()
        renderer = ol.Renderer(backend=backend)
        scene, camera = fixture()
        caller_thread = get_ident()
        first = renderer.render_async(scene, camera, (5, 3), samples=2)
        self.assertIsInstance(first, ol.RenderJob)
        self.assertTrue(backend.started.wait(1.0))
        self.assertFalse(first.done())
        self.assertNotEqual(backend.render_thread, caller_thread)
        second = renderer.render_async(scene, camera, (5, 3))
        self.assertEqual((first.frame_index, second.frame_index), (0, 1))
        self.assertTrue(second.cancel())
        backend.release.set()
        self.assertTrue(first.wait(1.0))
        np.testing.assert_array_equal(first.result(), 0.0)
        self.assertEqual(first.statistics.frame_index, 0)
        self.assertTrue(second.cancelled())
        renderer.close()

    def test_high_level_renderer_accepts_all_public_camera_models(self):
        scene, _camera = fixture()
        renderer = ol.Renderer(backend=FakeBackend())
        cameras = (
            ol.PerspectiveCamera((0, 0, -3), (0, 0, 0)),
            ol.OrthographicCamera((0, 0, -3), (0, 0, 0)),
            ol.PanoramicCamera((0, 0, -3), (0, 0, 0)),
        )
        for camera in cameras:
            self.assertEqual(renderer.render(scene, camera, (4, 2)).shape, (2, 4, 4))

    def test_motion_product_tracks_rigid_object_transform(self):
        backend = ol.VulkanRayTracingBackend.__new__(
            ol.VulkanRayTracingBackend
        )
        scene, camera = fixture()
        backend._output_history = backend._capture_motion_state(
            scene, camera, (100, 100)
        )
        mesh = scene.meshes[0]
        scene.update_mesh(mesh, transform=ol.Transform.translation((0.2, 0, 0)))
        primitive = np.full((100, 100), np.uint32(0xffffffff), np.uint32)
        position = np.zeros((100, 100, 3), np.float32)
        barycentric = np.zeros((100, 100, 2), np.float32)
        primitive[50, 50] = 0
        barycentric[50, 50] = (1 / 3, 1 / 3)
        position[50, 50] = mesh.world_vertices[mesh.indices[0]].mean(axis=0)
        motion = backend._motion_product(
            scene, camera, 100, 100, primitive, position, barycentric,
        )
        self.assertGreater(abs(float(motion[50, 50, 0])), 0.0)
        self.assertAlmostEqual(float(motion[50, 50, 1]), 0.0, places=5)
        motion[50, 50] = 0
        self.assertFalse(np.any(motion))

    def test_named_outputs_return_structured_frame(self):
        renderer = ol.Renderer(backend=FakeBackend())
        scene, camera = fixture()
        color = np.empty((3, 5, 4), np.float32)
        frame = renderer.render(
            scene, camera, (5, 3), samples=4,
            outputs=(
                "color", "variance", "depth", "normal",
                "instance_id", "object_id", "material_id",
                "motion",
            ),
            out={"color": color},
        )
        self.assertIsInstance(frame, ol.RenderFrame)
        self.assertEqual(
            tuple(frame), (
                "color", "variance", "depth", "normal",
                "instance_id", "object_id", "material_id",
                "motion",
            )
        )
        self.assertIs(frame.color, color)
        self.assertEqual(frame.variance.shape, (3, 5))
        self.assertEqual(frame.depth.shape, (3, 5))
        self.assertEqual(frame.normal.shape, (3, 5, 3))
        self.assertEqual(frame.object_id.dtype, np.uint32)
        self.assertEqual(frame.instance_id.dtype, np.uint32)
        self.assertEqual(frame.material_id.dtype, np.uint32)
        self.assertEqual(frame.motion.shape, (3, 5, 2))
        self.assertEqual(frame.metadata["frame_index"], 0)
        self.assertEqual(frame.metadata["size"], (5, 3))
        self.assertEqual(frame.metadata["samples"], 4)
        statistics = frame.metadata["statistics"]
        self.assertIsInstance(statistics, ol.RenderStatistics)
        self.assertEqual(statistics.frame_index, 0)
        self.assertEqual(statistics.gpu_ms, 2.5)
        with self.assertRaises(TypeError):
            frame["extra"] = color

    def test_named_output_validation_and_capabilities(self):
        renderer = ol.Renderer(backend=FakeBackend())
        scene, camera = fixture()
        capabilities = renderer.capabilities
        self.assertIsInstance(capabilities, ol.RendererCapabilities)
        self.assertEqual(capabilities.backend, "FakeBackend")
        self.assertEqual(capabilities.device, "test-device")
        self.assertTrue(capabilities.supports_output("motion"))
        self.assertFalse(capabilities.supports("hardware_ray_tracing"))
        with self.assertRaises(RuntimeError):
            capabilities.require("hardware_ray_tracing")
        with self.assertRaises(TypeError):
            capabilities.limits["max_bounces"] = 1
        self.assertEqual(
            renderer.available_outputs,
            (
                "color", "variance", "depth", "normal",
                "instance_id", "object_id", "material_id",
                "motion",
            ),
        )
        with self.assertRaises(ValueError):
            renderer.render(scene, camera, (4, 2), outputs=())
        with self.assertRaises(ValueError):
            renderer.render(scene, camera, (4, 2), outputs=("albedo",))
        with self.assertRaises(ValueError):
            renderer.render(
                scene, camera, (4, 2), outputs=("color", "color")
            )
        with self.assertRaises(TypeError):
            renderer.render(
                scene, camera, (4, 2), outputs=("color",),
                out=np.empty((2, 4, 4), np.float32),
            )

    def test_explicit_capability_contract_is_normalized(self):
        backend = FakeBackend()
        backend.capabilities = {
            "backend": "test",
            "features": {"volumes", "offscreen_rendering"},
            "limits": {"max_bounces": 8},
        }
        capabilities = ol.Renderer(backend=backend).capabilities
        self.assertEqual(capabilities.backend, "test")
        self.assertTrue(capabilities.supports("volumes"))
        capabilities.require("volumes", "offscreen_rendering")
        self.assertEqual(capabilities.limits["max_bounces"], 8)
        self.assertEqual(capabilities.outputs, FakeBackend.available_outputs)
        self.assertEqual(capabilities.as_dict()["device"], "test-device")

    def test_returns_hdr_and_advances_deterministic_sequence(self):
        backend = FakeBackend()
        renderer = ol.Renderer(backend=backend)
        scene, camera = fixture()
        first = renderer.render(scene, camera, (5, 3), samples=2)
        second = renderer.render(scene, camera, (5, 3))
        self.assertEqual(first.shape, (3, 5, 4))
        self.assertEqual(first.dtype, np.float32)
        np.testing.assert_array_equal(first, 0.0)
        np.testing.assert_array_equal(second, 1.0)
        self.assertEqual(backend.calls[0][2:], (5, 3, 2, 0))
        self.assertEqual(renderer.frame_index, 2)
        self.assertEqual(renderer.device, "test-device")
        self.assertEqual(renderer.last_timings, {"gpu_ms": 2.5})
        self.assertEqual(renderer.last_statistics.as_dict(), {
            "frame_index": 1,
            "width": 5,
            "height": 3,
            "samples": None,
            "total_ms": None,
            "gpu_ms": 2.5,
        })

    def test_replace_scene_reuses_backend_and_resets_sequence(self):
        backend = FakeBackend()
        renderer = ol.Renderer(backend=backend)
        first, camera = fixture()
        second, _camera = fixture()
        renderer.render(first, camera, (2, 2))
        returned = renderer.replace_scene(second)
        self.assertIs(returned, second)
        self.assertEqual(backend.scene_replacements, [second])
        self.assertEqual(renderer.frame_index, 0)
        self.assertIsNone(renderer.last_statistics)
        renderer.render(second, camera, (2, 2))
        self.assertEqual(backend.calls[-1][-1], 0)

    def test_replace_scene_requires_backend_support(self):
        backend = FakeBackend()
        backend.replace_scene = None
        renderer = ol.Renderer(backend=backend)
        scene, _camera = fixture()
        with self.assertRaisesRegex(RuntimeError, "resident scene replacement"):
            renderer.replace_scene(scene)

    def test_reconfigure_is_ordered_and_does_not_replace_backend(self):
        backend = FakeBackend()
        renderer = ol.Renderer(backend=backend)
        result = renderer.reconfigure(samples_per_pixel=2, max_bounces=8)
        self.assertEqual(result, {"samples_per_pixel": 2, "max_bounces": 8})
        self.assertEqual(backend.setting_changes, [result])
        self.assertIs(renderer._backend, backend)

    def test_statistics_preserve_backend_timings_without_name_collisions(self):
        statistics = ol.RenderStatistics(
            frame_index=8, size=(640, 480), samples=2,
            timings={"gpu_frame_ms": 3.25, "total_ms": 4.5, "width": 1},
        )
        self.assertEqual(statistics.total_ms, 4.5)
        self.assertEqual(statistics.gpu_ms, 3.25)
        self.assertEqual(statistics.timings["width"], 1)
        self.assertEqual(statistics.as_dict()["width"], 640)
        with self.assertRaises(TypeError):
            statistics.timings["gpu_frame_ms"] = 0

    def test_explicit_frame_and_caller_owned_output(self):
        renderer = ol.Renderer(backend=FakeBackend())
        scene, camera = fixture()
        output = np.empty((2, 4, 4), np.float32)
        result = renderer.render(
            scene, camera, (4, 2), frame_index=7, out=output,
        )
        self.assertIs(result, output)
        np.testing.assert_array_equal(output, 7.0)
        self.assertEqual(renderer.frame_index, 8)
        renderer.reset_sequence()
        self.assertEqual(renderer.frame_index, 0)

    def test_validates_inputs_and_owns_backend_lifetime(self):
        backend = FakeBackend()
        renderer = ol.Renderer(backend=backend)
        scene, camera = fixture()
        with self.assertRaises(TypeError):
            renderer.render(object(), camera, (4, 2))
        with self.assertRaises(TypeError):
            renderer.render(scene, object(), (4, 2))
        with self.assertRaises(ValueError):
            renderer.render(scene, camera, (0, 2))
        with self.assertRaises(TypeError):
            renderer.render(scene, camera, (4, 2), out=np.empty((2, 4, 4)))
        renderer.close()
        renderer.close()
        self.assertTrue(backend.closed)


if __name__ == "__main__":
    unittest.main()
