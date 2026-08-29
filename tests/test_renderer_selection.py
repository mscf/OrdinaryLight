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


class RendererSelectionTests(unittest.TestCase):
    def test_auto_prefers_gi_and_reports_decision(self):
        backend = _Backend()
        with patch(
            "ordinarylight.renderers.selection._gi_implementation",
            return_value=backend,
        ) as gi, patch(
            "ordinarylight.renderers.selection._raster_implementation",
        ) as raster:
            selected = ol.select_vulkan_renderer("auto")
        self.assertIs(selected, backend)
        gi.assert_called_once()
        raster.assert_not_called()
        self.assertEqual(selected.renderer_selection, {
            "requested": "auto", "selected": "vulkan-gi",
            "fallback": False, "reason": None,
        })

    def test_auto_falls_back_only_for_missing_ray_query_adapter(self):
        backend = _Backend()
        unavailable = RuntimeError(
            "No compatible Vulkan ray-tracing adapter: test adapter"
        )
        with patch(
            "ordinarylight.renderers.selection._gi_implementation",
            side_effect=unavailable,
        ), patch(
            "ordinarylight.renderers.selection._raster_implementation",
            return_value=backend,
        ):
            selected = ol.select_vulkan_renderer("auto")
        self.assertTrue(selected.renderer_selection["fallback"])
        self.assertEqual(
            selected.renderer_selection["selected"], "vulkan-raster"
        )
        self.assertIn("test adapter", selected.renderer_selection["reason"])

    def test_auto_does_not_mask_initialization_failures(self):
        with patch(
            "ordinarylight.renderers.selection._gi_implementation",
            side_effect=RuntimeError("shader initialization failed"),
        ), patch(
            "ordinarylight.renderers.selection._raster_implementation",
        ) as raster:
            with self.assertRaisesRegex(RuntimeError, "shader initialization"):
                ol.select_vulkan_renderer("auto")
        raster.assert_not_called()

    def test_explicit_preferences_never_switch_implementation_class(self):
        gi_implementation, raster_implementation = _Backend(), _Backend()
        with patch(
            "ordinarylight.renderers.selection._gi_implementation",
            return_value=gi_implementation,
        ), patch(
            "ordinarylight.renderers.selection._raster_implementation",
            return_value=raster_implementation,
        ):
            self.assertIs(ol.select_vulkan_renderer("gi"), gi_implementation)
            self.assertIs(
                ol.select_vulkan_renderer("raster"), raster_implementation
            )
        self.assertFalse(gi_implementation.renderer_selection["fallback"])
        self.assertFalse(raster_implementation.renderer_selection["fallback"])

    def test_renderer_exposes_selection_through_capabilities(self):
        backend = _Backend()
        backend.renderer_selection = ol.RendererSelection(
            "auto", "vulkan-raster", True, "no ray queries",
        ).as_dict()
        renderer = ol.Renderer(implementation=backend)
        self.assertEqual(
            renderer.renderer_selection["selected"], "vulkan-raster"
        )
        self.assertEqual(
            renderer.capabilities.as_dict()["selection"]["reason"],
            "no ray queries",
        )

    def test_invalid_preference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "auto, gi, or raster"):
            ol.select_vulkan_renderer("magic")


if __name__ == "__main__":
    unittest.main()
