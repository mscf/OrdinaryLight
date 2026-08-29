import unittest
from unittest.mock import patch

import ordinarylight as ol


class _Backend:
    available_outputs = ("color",)
    config = None
    device = "test-device"
    last_timings = {}

    def render_frame(self, *args, **kwargs):
        raise AssertionError("selection tests do not render")

    def close(self):
        pass


class BackendSelectionTests(unittest.TestCase):
    def test_auto_prefers_gi_and_reports_decision(self):
        backend = _Backend()
        with patch(
            "ordinarylight.backend_selection._gi_backend",
            return_value=backend,
        ) as gi, patch(
            "ordinarylight.backend_selection._raster_backend",
        ) as raster:
            selected = ol.select_vulkan_backend("auto")
        self.assertIs(selected, backend)
        gi.assert_called_once()
        raster.assert_not_called()
        self.assertEqual(selected.backend_selection, {
            "requested": "auto", "selected": "vulkan-gi",
            "fallback": False, "reason": None,
        })

    def test_auto_falls_back_only_for_missing_ray_query_adapter(self):
        backend = _Backend()
        unavailable = RuntimeError(
            "No compatible Vulkan ray-tracing adapter: test adapter"
        )
        with patch(
            "ordinarylight.backend_selection._gi_backend",
            side_effect=unavailable,
        ), patch(
            "ordinarylight.backend_selection._raster_backend",
            return_value=backend,
        ):
            selected = ol.select_vulkan_backend("auto")
        self.assertTrue(selected.backend_selection["fallback"])
        self.assertEqual(
            selected.backend_selection["selected"], "vulkan-raster"
        )
        self.assertIn("test adapter", selected.backend_selection["reason"])

    def test_auto_does_not_mask_initialization_failures(self):
        with patch(
            "ordinarylight.backend_selection._gi_backend",
            side_effect=RuntimeError("shader initialization failed"),
        ), patch(
            "ordinarylight.backend_selection._raster_backend",
        ) as raster:
            with self.assertRaisesRegex(RuntimeError, "shader initialization"):
                ol.select_vulkan_backend("auto")
        raster.assert_not_called()

    def test_explicit_preferences_never_switch_backend_class(self):
        gi_backend, raster_backend = _Backend(), _Backend()
        with patch(
            "ordinarylight.backend_selection._gi_backend",
            return_value=gi_backend,
        ), patch(
            "ordinarylight.backend_selection._raster_backend",
            return_value=raster_backend,
        ):
            self.assertIs(ol.select_vulkan_backend("gi"), gi_backend)
            self.assertIs(
                ol.select_vulkan_backend("raster"), raster_backend
            )
        self.assertFalse(gi_backend.backend_selection["fallback"])
        self.assertFalse(raster_backend.backend_selection["fallback"])

    def test_renderer_exposes_selection_through_capabilities(self):
        backend = _Backend()
        backend.backend_selection = ol.BackendSelection(
            "auto", "vulkan-raster", True, "no ray queries",
        ).as_dict()
        renderer = ol.Renderer(backend=backend)
        self.assertEqual(
            renderer.backend_selection["selected"], "vulkan-raster"
        )
        self.assertEqual(
            renderer.capabilities.as_dict()["selection"]["reason"],
            "no ray queries",
        )

    def test_invalid_preference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "auto, gi, or raster"):
            ol.select_vulkan_backend("magic")


if __name__ == "__main__":
    unittest.main()
