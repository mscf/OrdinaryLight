import unittest
from types import SimpleNamespace
from unittest.mock import patch
from threading import Event

import numpy as np

import ordinarylight as ol
from ordinarylight.integrations.qt_workbench import (
    QUALITY_PRESETS, WorkbenchState, _SceneLoader, _initial_showcase_index,
    load_workbench_extension,
)
from ordinarylight.integrations.workbench import Showcase


def triangle(offset=0.0):
    scene = ol.Scene()
    scene.add_mesh(
        np.asarray(((offset, 0, 0), (offset + 1, 0, 0), (offset, 1, 0)), np.float32),
        np.asarray(((0, 1, 2),), np.uint32),
    )
    return scene


class WorkbenchStateTests(unittest.TestCase):
    def test_application_extension_is_loaded_only_from_explicit_reference(self):
        factory = lambda *_args: None
        with patch.dict("sys.modules", {"example_extension": SimpleNamespace(
            create=factory,
        )}):
            self.assertIs(
                load_workbench_extension("example_extension:create"), factory,
            )
        self.assertIsNone(load_workbench_extension(""))

    def test_invalid_extension_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "module:factory"):
            load_workbench_extension("missing-separator")

    def test_initial_showcase_can_be_selected_by_id_for_smoke_runs(self):
        state = WorkbenchState()
        state.add_showcase(Showcase("first", "First", triangle))
        state.add_showcase(Showcase("scientific", "Scientific", triangle))
        self.assertEqual(_initial_showcase_index(state.scenes), 0)
        self.assertEqual(
            _initial_showcase_index(state.scenes, "scientific"), 1,
        )
        self.assertEqual(_initial_showcase_index(state.scenes, "missing"), 0)

    def test_scene_loader_close_does_not_wait_for_running_user_code(self):
        started = Event()
        release = Event()
        loader = _SceneLoader()

        def slow_build():
            started.set()
            release.wait(1.0)
            return "scene"

        future = loader.submit(slow_build)
        self.assertTrue(started.wait(0.5))
        loader.close()
        self.assertFalse(future.done())
        release.set()
        self.assertEqual(future.result(1.0), "scene")

    def test_balanced_preset_uses_temporal_restir_without_extra_rays(self):
        preset = QUALITY_PRESETS["balanced"]
        self.assertEqual(preset["samples"], 1)
        self.assertEqual(preset["scale"], 1.0)
        self.assertEqual(preset["area_light_samples"], 2)
        self.assertTrue(preset["temporal"])
        self.assertFalse(preset["restir"])
        self.assertFalse(preset["restir_spatial"])
        self.assertFalse(preset["denoiser"])

    def test_clean_preset_spends_samples_instead_of_enabling_inactive_denoiser(self):
        preset = QUALITY_PRESETS["clean"]
        config = ol.RendererConfig(
            samples_per_pixel=preset["samples"],
            denoiser_enabled=preset["denoiser"],
            wavefront_temporal_reconstruction=preset["temporal"],
        )
        self.assertEqual(config.samples_per_pixel, 2)
        self.assertEqual(preset["scale"], 1.0)
        self.assertFalse(config.denoiser_enabled)

    def test_fast_preset_is_explicit_about_reduced_resolution(self):
        preset = QUALITY_PRESETS["fast"]
        self.assertEqual(preset["samples"], 2)
        self.assertEqual(preset["scale"], 0.5)

    def test_scene_lifecycle(self):
        state = WorkbenchState()
        first = state.add("first", triangle())
        state.add("second", triangle(2), activate=False)
        self.assertIs(state.active, first)
        state.activate(1)
        self.assertEqual(state.active.name, "second")
        state.remove(1)
        self.assertEqual(state.active.name, "first")

    def test_camera_fit_is_finite_and_positive(self):
        state = WorkbenchState()
        entry = state.add("triangle", triangle())
        center, radius, height = WorkbenchState.camera_parameters(entry)
        self.assertTrue(np.all(np.isfinite(center)))
        self.assertGreater(radius, 0.0)
        self.assertTrue(np.isfinite(height))

    def test_authored_camera_metadata_is_preserved(self):
        state = WorkbenchState()
        entry = state.add(
            "room", triangle(), camera_target=(0, 1.25, 0),
            orbit_radius=-8.5, camera_height=3.2,
            presentation_arc_radians=0.48,
        )
        self.assertEqual(entry.camera_target, (0.0, 1.25, 0.0))
        self.assertEqual(entry.orbit_radius, -8.5)
        self.assertEqual(entry.presentation_arc_radians, 0.48)
        center, radius, height = WorkbenchState.camera_parameters(entry)
        np.testing.assert_allclose(center, (0.0, 1.25, 0.0))
        self.assertEqual(radius, -8.5)
        self.assertEqual(height, 3.2)


if __name__ == "__main__":
    unittest.main()
