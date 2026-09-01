import unittest
from unittest.mock import patch

import numpy as np

import ordinarylight as ol
from ordinarylight.showcases.materials import diffuse, fresnel_glass
from ordinarylight import RendererConfig
from ordinarylight.renderers.gi import VulkanGlobalIlluminationRenderer
from ordinarylight.targets.vulkan.api import _render_options, _resolve_execution_strategy
from ordinarylight.targets.vulkan.core import (
    RECONSTRUCT_PUSH_SIZE,
    VulkanRayQueryCore,
    _camera_angular_motion_pixels,
    _effect_screen_rect,
    _motion_adaptive_history_limit,
    _wavefront_history_semaphore_plan,
)


class RendererConfigTests(unittest.TestCase):
    def test_wavefront_history_semaphore_plan_forms_one_ping_pong_chain(self):
        self.assertEqual(
            _wavefront_history_semaphore_plan(
                current_pending=False,
                previous_pending=False,
                history_enabled=True,
            ),
            (False, True),
        )
        self.assertEqual(
            _wavefront_history_semaphore_plan(
                current_pending=False,
                previous_pending=True,
                history_enabled=True,
            ),
            (True, True),
        )

    def test_wavefront_history_chain_consumes_stale_signal_when_disabled(self):
        self.assertEqual(
            _wavefront_history_semaphore_plan(
                current_pending=False,
                previous_pending=True,
                history_enabled=False,
            ),
            (True, False),
        )
        with self.assertRaisesRegex(RuntimeError, "not consumed"):
            _wavefront_history_semaphore_plan(
                current_pending=True,
                previous_pending=False,
                history_enabled=True,
            )

    def test_pipeline_cache_configuration(self):
        default = RendererConfig()
        self.assertTrue(default.vulkan_pipeline_cache)
        self.assertIsNone(default.vulkan_pipeline_cache_path)
        configured = RendererConfig(
            vulkan_pipeline_cache=True,
            vulkan_pipeline_cache_path="/tmp/ordinarylight-cache.bin",
        )
        self.assertEqual(
            configured.vulkan_pipeline_cache_path,
            "/tmp/ordinarylight-cache.bin",
        )
        self.assertFalse(
            RendererConfig(vulkan_pipeline_cache=False).vulkan_pipeline_cache
        )
        with self.assertRaises(TypeError):
            RendererConfig(vulkan_pipeline_cache=1)
        with self.assertRaises(TypeError):
            RendererConfig(vulkan_pipeline_cache_path=object())

    def test_ordinaryshade_shade_is_default_and_configurable(self):
        self.assertTrue(RendererConfig().wavefront_ordinaryshade_shade)
        self.assertTrue(RendererConfig(
            wavefront_ordinaryshade_shade=True
        ).wavefront_ordinaryshade_shade)
        with self.assertRaises(TypeError):
            RendererConfig(wavefront_ordinaryshade_shade=1)
        self.assertTrue(RendererConfig(
            wavefront_ordinaryshade_shade=True,
            wavefront_execution_strategy="megakernel",
        ).wavefront_ordinaryshade_shade)
        self.assertIs(
            RendererConfig(
                wavefront_ordinaryshade_shade=True,
                material_program=diffuse,
            ).material_program,
            diffuse,
        )

    def test_projected_effect_bounds_track_object_and_aspect(self):
        scene = ol.Scene()
        mesh = scene.add_mesh(
            ((-1, -1, 0), (1, -1, 0), (0, 1, 0)), ((0, 1, 2),),
        )
        rect = _effect_screen_rect(
            scene, scene.object_triangle_range(mesh),
            ol.PerspectiveCamera((0, 0, 3), (0, 0, 0)), (1600, 800),
        )
        self.assertLess(rect[0], 0.5)
        self.assertGreater(rect[2], 0.5)
        self.assertLess(rect[1], 0.5)
        self.assertGreater(rect[3], 0.5)

    def test_reconstruct_push_constants_match_shader_layout(self):
        self.assertEqual(RECONSTRUCT_PUSH_SIZE, 256)

    def test_object_effect_storage_is_boolean_and_opt_in(self):
        self.assertFalse(RendererConfig().object_effects)
        self.assertTrue(RendererConfig(object_effects=True).object_effects)
        with self.assertRaises(TypeError):
            RendererConfig(object_effects=1)

    def test_object_effect_is_runtime_state_not_selection_state(self):
        core = object.__new__(VulkanRayQueryCore)
        core.config = RendererConfig(object_effects=True)
        core.object_effect_bindings = ()
        core.window_frames = [{
            "wavefront_command_key": "cached",
            "wavefront_reservoir_valid": True,
        }]
        effect = ol.effects.Outline(color=(0.2, 0.4, 0.8), width=3)
        self.assertEqual(
            core.set_object_effect((4, 9), effect), (((4, 9), effect),)
        )
        self.assertEqual(core.object_effect_triangle_range, (4, 9))
        self.assertIs(core.object_effect, effect)
        self.assertIsNone(core.window_frames[0]["wavefront_command_key"])
        self.assertFalse(core.window_frames[0]["wavefront_reservoir_valid"])
        self.assertEqual(core.set_object_effect(), ())
        second = ol.effects.Tint(color=(1, 0, 0), strength=0.25)
        self.assertEqual(len(core.set_object_effects((
            ((4, 9), effect), ((10, 12), second),
        ))), 2)
        with self.assertRaises(ValueError):
            core.set_object_effects((((index, index + 1), effect)
                                     for index in range(5)))
        with self.assertRaises(TypeError):
            core.set_object_effect((4, 9), object())

    def test_scene_replacement_preserves_output_resources(self):
        core = object.__new__(VulkanRayQueryCore)
        core.device = object()
        core.material_programs = ()
        old = type("OldResources", (), {
            "scene": object(), "scene_revision": 0,
            "custom_attribute_layout": None,
            "closed": False,
            "close": lambda self: setattr(self, "closed", True),
        })()
        core.scene_resources = old
        core._try_update_window_scene = lambda _scene: False
        core._capture_scene_state = lambda: {}
        core._restore_scene_state = lambda _state: None
        activated = []
        invalidated = []
        core._activate_scene_resources = activated.append
        core._invalidate_scene_history = lambda: invalidated.append(True)
        core._destroy_swapchain_resources = lambda: self.fail(
            "resident scene replacement destroyed output resources"
        )
        replacement = object()
        with patch(
            "ordinarylight.targets.vulkan.core.vk.vkDeviceWaitIdle"
        ), patch(
            "ordinarylight.targets.vulkan.core.VulkanSceneResources",
            return_value=replacement,
        ):
            core.upload_window_scene(object())
        self.assertEqual(activated, [replacement])
        self.assertEqual(invalidated, [True])
        self.assertTrue(old.closed)

    def test_hot_reconfigure_updates_core_without_recreation(self):
        backend = object.__new__(VulkanGlobalIlluminationRenderer)
        backend.config = RendererConfig()
        backend._output_history = object()

        class Core:
            config = backend.config
            window_frames = [{"wavefront_command_key": "old"}]
            reset = False

            def reset_accumulation(self):
                self.reset = True

        backend._core = Core()
        updated = backend.reconfigure(
            samples_per_pixel=2, max_bounces=8, wavefront_exposure=1.2,
        )
        self.assertEqual(updated.samples_per_pixel, 2)
        self.assertEqual(backend._core.config, updated)
        self.assertTrue(backend._core.reset)
        self.assertIsNone(backend._output_history)
        self.assertIsNone(
            backend._core.window_frames[0]["wavefront_command_key"]
        )
        with self.assertRaisesRegex(RuntimeError, "recreation required"):
            backend.reconfigure(present_mode="fifo")

    def test_external_image_interop_is_boolean_and_opt_in(self):
        self.assertFalse(RendererConfig().external_image_interop)
        self.assertTrue(
            RendererConfig(external_image_interop=True).external_image_interop
        )
        with self.assertRaises(TypeError):
            RendererConfig(external_image_interop=1)

    def test_stationary_accumulation_is_explicit_and_progressive(self):
        self.assertFalse(RendererConfig().stationary_accumulation)
        config = RendererConfig(
            progressive_accumulation=True,
            stationary_accumulation=True,
        )
        self.assertTrue(config.stationary_accumulation)
        with self.assertRaisesRegex(ValueError, "progressive_accumulation"):
            RendererConfig(stationary_accumulation=True)
        with self.assertRaises(TypeError):
            RendererConfig(
                progressive_accumulation=True,
                stationary_accumulation=1,
            )

    def test_interactive_render_scale_is_optional_and_bounded(self):
        self.assertIsNone(
            RendererConfig().wavefront_interactive_render_scale
        )
        config = RendererConfig(wavefront_interactive_render_scale=0.5)
        self.assertEqual(config.wavefront_interactive_render_scale, 0.5)
        with self.assertRaisesRegex(ValueError, "between 0.25 and 1.0"):
            RendererConfig(wavefront_interactive_render_scale=0.2)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            RendererConfig(
                wavefront_render_scale=0.5,
                wavefront_interactive_render_scale=0.75,
            )

    def test_interactive_target_fps_is_optional_and_bounded(self):
        config = RendererConfig(
            wavefront_interactive_target_fps=60.0,
            wavefront_interactive_min_scale=0.4,
        )
        self.assertEqual(config.wavefront_interactive_target_fps, 60.0)
        self.assertEqual(config.wavefront_interactive_min_scale, 0.4)
        with self.assertRaisesRegex(ValueError, "must be positive"):
            RendererConfig(wavefront_interactive_target_fps=0.0)
        with self.assertRaisesRegex(ValueError, "interactive maximum"):
            RendererConfig(
                wavefront_interactive_render_scale=0.5,
                wavefront_interactive_min_scale=0.75,
                wavefront_interactive_target_fps=60.0,
            )
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            RendererConfig(
                wavefront_interactive_target_fps=60.0,
                wavefront_dynamic_resolution=True,
            )

    def test_interactive_sample_scaling_is_opt_in_and_bounded(self):
        config = RendererConfig(
            samples_per_pixel=8,
            wavefront_interactive_target_fps=60.0,
            wavefront_interactive_sample_scaling=True,
            wavefront_interactive_min_samples=2,
        )
        self.assertTrue(config.wavefront_interactive_sample_scaling)
        self.assertEqual(config.wavefront_interactive_min_samples, 2)
        with self.assertRaisesRegex(ValueError, "requires"):
            RendererConfig(wavefront_interactive_sample_scaling=True)
        with self.assertRaisesRegex(ValueError, "between 1"):
            RendererConfig(
                samples_per_pixel=2,
                wavefront_interactive_min_samples=3,
            )
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            RendererConfig(
                samples_per_pixel=4,
                interactive_samples_per_pixel=1,
                wavefront_interactive_target_fps=60.0,
                wavefront_interactive_sample_scaling=True,
            )

    def test_stationary_accumulation_motion_state_machine(self):
        core = object.__new__(VulkanRayQueryCore)
        core.config = RendererConfig(
            progressive_accumulation=True,
            stationary_accumulation=True,
            stationary_delay_seconds=0.15,
            temporal_history=True,
        )
        core.accumulation_frame = 0
        core.accumulation_key = None
        core.accumulation_history_valid = False
        core.accumulation_camera_signature = None
        core.accumulation_state = ol.AccumulationState.DISABLED
        core.camera_change_time = 0.0
        camera = ol.PerspectiveCamera((0, 0, -3), (0, 0, 0))
        moved = ol.PerspectiveCamera((0.1, 0, -3), (0, 0, 0))

        with patch(
            "ordinarylight.targets.vulkan.core.time.perf_counter",
            side_effect=(0.0, 0.05, 0.20, 0.21, 0.22),
        ):
            signature, active, interactive = core._begin_accumulation_frame(
                (1, 0, 64, 64), camera
            )
            self.assertFalse(active)
            self.assertTrue(interactive)
            self.assertEqual(core.accumulation_state, ol.AccumulationState.MOVING)
            core._finish_accumulation_frame(signature, active)

            signature, active, interactive = core._begin_accumulation_frame(
                (1, 0, 64, 64), camera
            )
            self.assertFalse(active)
            self.assertTrue(interactive)
            self.assertEqual(
                core.accumulation_state, ol.AccumulationState.SETTLING
            )
            core._finish_accumulation_frame(signature, active)

            signature, active, interactive = core._begin_accumulation_frame(
                (1, 0, 64, 64), camera
            )
            self.assertTrue(active)
            self.assertFalse(interactive)
            core._finish_accumulation_frame(signature, active)
            self.assertEqual(core.accumulation_frame, 1)

            signature, active, interactive = core._begin_accumulation_frame(
                (1, 0, 64, 64), moved
            )
            self.assertFalse(active)
            self.assertTrue(interactive)
            self.assertEqual(core.accumulation_frame, 0)
            core._finish_accumulation_frame(signature, active)

            _signature, active, interactive = core._begin_accumulation_frame(
                (1, 1, 64, 64), moved
            )
            self.assertFalse(active)
            self.assertTrue(interactive)
            self.assertEqual(core.accumulation_state, ol.AccumulationState.MOVING)

    def test_interactive_motion_detection_does_not_require_accumulation(self):
        core = object.__new__(VulkanRayQueryCore)
        core.config = RendererConfig(stationary_delay_seconds=0.1)
        core.accumulation_frame = 0
        core.accumulation_key = None
        core.accumulation_history_valid = False
        core.accumulation_camera_signature = None
        core.accumulation_state = ol.AccumulationState.DISABLED
        core.camera_change_time = 0.0
        camera = ol.PerspectiveCamera((0, 0, -3), (0, 0, 0))
        with patch(
            "ordinarylight.targets.vulkan.core.time.perf_counter",
            side_effect=(0.0, 0.2),
        ):
            signature, active, interactive = core._begin_accumulation_frame(
                (1, 0, 64, 64), camera
            )
            self.assertFalse(active)
            self.assertTrue(interactive)
            core._finish_accumulation_frame(signature, active)
            _signature, active, interactive = core._begin_accumulation_frame(
                (1, 0, 64, 64), camera
            )
            self.assertFalse(active)
            self.assertFalse(interactive)

    def test_volume_empty_space_skipping_is_boolean_and_opt_in(self):
        self.assertFalse(RendererConfig().volume_empty_space_skipping)
        self.assertTrue(
            RendererConfig(volume_empty_space_skipping=True)
            .volume_empty_space_skipping
        )
        with self.assertRaises(TypeError):
            RendererConfig(volume_empty_space_skipping=1)

    @staticmethod
    def _strategy_scene(transmission, triangle_count=1, program=None):
        scene = ol.Scene()
        scene.add_mesh(
            np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0)), np.float32),
            np.repeat(
                np.asarray(((0, 1, 2),), np.uint32), triangle_count, axis=0),
            ol.Material(transmission=transmission, program=program),
        )
        return scene

    def test_auto_strategy_selects_kernel_from_transmission_density(self):
        config = RendererConfig(wavefront_execution_strategy="auto")
        self.assertEqual(
            _resolve_execution_strategy(config, self._strategy_scene(0.0)),
            "wavefront",
        )
        self.assertEqual(
            _resolve_execution_strategy(config, self._strategy_scene(1.0)),
            "megakernel",
        )
        self.assertEqual(
            _resolve_execution_strategy(
                config, self._strategy_scene(0.0, 16384)),
            "megakernel",
        )
        explicit = RendererConfig(wavefront_execution_strategy="wavefront")
        self.assertEqual(
            _resolve_execution_strategy(explicit, self._strategy_scene(1.0)),
            "wavefront",
        )
        persistent = RendererConfig(wavefront_execution_strategy="persistent")
        self.assertEqual(
            _resolve_execution_strategy(persistent, self._strategy_scene(1.0)),
            "persistent",
        )

    def test_opaque_specialization_proves_only_literal_nontransmission_events(self):
        core = object.__new__(VulkanRayQueryCore)
        core.config = RendererConfig(wavefront_scene_specialization=True)

        opaque = self._strategy_scene(0.0, program=diffuse)
        self.assertTrue(core._use_opaque_scene_specialization(opaque))

        glass = self._strategy_scene(0.0, program=fresnel_glass)
        self.assertFalse(core._use_opaque_scene_specialization(glass))

    def test_camera_motion_is_measured_in_output_pixels(self):
        previous = ol.PerspectiveCamera(
            position=(0.0, 0.0, -4.0), target=(0.0, 0.0, 0.0),
            vertical_fov_degrees=60.0,
        )
        stationary = ol.PerspectiveCamera(
            position=(0.0, 0.0, -4.0), target=(0.0, 0.0, 0.0),
            vertical_fov_degrees=60.0,
        )
        rotated = ol.PerspectiveCamera(
            position=(0.0, 0.0, -4.0), target=(1.0, 0.0, 0.0),
            vertical_fov_degrees=60.0,
        )
        self.assertEqual(
            _camera_angular_motion_pixels(previous, stationary, 1080), 0.0
        )
        self.assertGreater(
            _camera_angular_motion_pixels(previous, rotated, 1080), 64.0
        )
        self.assertTrue(np.isinf(_camera_angular_motion_pixels(
            previous,
            ol.OrthographicCamera((0, 0, -4), (0, 0, 0)),
            1080,
        )))

    def test_external_surface_requires_instance_and_surface_pair(self):
        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            VulkanRayQueryCore(external_instance=1)

    def test_defaults_and_override(self):
        config = RendererConfig(
            max_bounces=8, swapchain_images=6, samples_per_pixel=4,
            progressive_accumulation=True, interactive_samples_per_pixel=1,
            temporal_history=True,
            temporal_history_limit=48,
            adaptive_sampling=True, adaptive_min_samples=2,
            adaptive_variance_threshold=0.01,
            area_light_samples=4,
            wavefront_secondary_area_light_samples=2,
            wavefront_environment_samples=2,
            wavefront_secondary_nee_probability=0.5,
            wavefront_unified_secondary_nee=True,
            wavefront_stratified_primary_restir=True,
            wavefront_restir_history_limit=32,
            wavefront_restir_history_motion_pixels=24.0,
            wavefront_russian_roulette=True,
            wavefront_russian_roulette_start=4,
            wavefront_russian_roulette_min_survival=0.2,
            wavefront_fused_secondary=True,
            wavefront_subgroup_enqueue=True,
            direct_swapchain_storage=False,
            wavefront_execution_strategy="megakernel",
            denoiser_enabled=True, denoiser_iterations=4,
            denoiser_variance_threshold=0.02,
            wavefront_tile_capacity=65536,
            wavefront_exposure=1.5,
            wavefront_render_scale=0.75,
            wavefront_dynamic_resolution=True,
            wavefront_dynamic_target_ms=18.0,
            wavefront_dynamic_min_scale=0.5,
            wavefront_temporal_reconstruction=True,
            wavefront_temporal_weight=0.8,
            wavefront_temporal_variance_confidence=True,
            wavefront_temporal_variance_strength=0.6,
            wavefront_temporal_material_confidence=True,
            wavefront_temporal_transmission_history_scale=0.55,
            wavefront_temporal_reprojection_search=True,
            wavefront_temporal_outlier_confidence=True,
            wavefront_temporal_outlier_strength=0.7,
            wavefront_temporal_motion_limit_pixels=48.0,
            wavefront_indirect_reuse_storage=True,
            wavefront_indirect_reuse_candidates=True,
            wavefront_indirect_reuse_temporal=True,
            wavefront_indirect_reuse_spatial=True,
            wavefront_indirect_reuse_profiling=True,
            wavefront_indirect_reuse_apply=True,
            wavefront_indirect_reuse_apply_strength=0.4,
            wavefront_indirect_reuse_history_limit=48,
            wavefront_indirect_reuse_history_motion_pixels=24.0,
            wavefront_indirect_reuse_debug_view="acceptance",
            wavefront_indirect_reuse_scale=0.5,
            wavefront_indirect_reuse_budget_mib=160.0,
            wavefront_diffuse_filter=True,
            wavefront_diffuse_filter_strength=0.45,
        )
        self.assertEqual(_render_options(config, 1, None), (1, 8))
        self.assertEqual(_render_options(config, 4, 3), (4, 3))
        self.assertTrue(config.progressive_accumulation)
        self.assertEqual(config.interactive_samples_per_pixel, 1)
        self.assertTrue(config.temporal_history)
        self.assertEqual(config.temporal_history_limit, 48)
        self.assertTrue(config.adaptive_sampling)
        self.assertEqual(config.adaptive_min_samples, 2)
        self.assertEqual(config.area_light_samples, 4)
        self.assertEqual(config.wavefront_secondary_area_light_samples, 2)
        self.assertEqual(config.wavefront_environment_samples, 2)
        self.assertEqual(config.wavefront_secondary_nee_probability, 0.5)
        self.assertTrue(config.wavefront_unified_secondary_nee)
        self.assertFalse(config.wavefront_unified_primary_restir)
        self.assertTrue(config.wavefront_stratified_primary_restir)
        self.assertFalse(config.wavefront_restir_di)
        self.assertEqual(config.wavefront_restir_history_limit, 32)
        self.assertEqual(config.wavefront_restir_history_motion_pixels, 24.0)
        self.assertTrue(config.wavefront_russian_roulette)
        self.assertEqual(config.wavefront_russian_roulette_start, 4)
        self.assertEqual(config.wavefront_russian_roulette_min_survival, 0.2)
        self.assertTrue(config.wavefront_fused_secondary)
        self.assertTrue(config.wavefront_subgroup_enqueue)
        self.assertFalse(config.direct_swapchain_storage)
        self.assertEqual(config.wavefront_execution_strategy, "megakernel")
        self.assertTrue(config.denoiser_enabled)
        self.assertEqual(config.denoiser_iterations, 4)
        self.assertEqual(config.denoiser_variance_threshold, 0.02)
        self.assertEqual(config.wavefront_tile_capacity, 65536)
        self.assertEqual(config.wavefront_exposure, 1.5)
        self.assertEqual(config.wavefront_render_scale, 0.75)
        self.assertTrue(config.wavefront_dynamic_resolution)
        self.assertEqual(config.wavefront_dynamic_target_ms, 18.0)
        self.assertEqual(config.wavefront_dynamic_min_scale, 0.5)
        self.assertTrue(config.wavefront_temporal_reconstruction)
        self.assertEqual(config.wavefront_temporal_weight, 0.8)
        self.assertTrue(config.wavefront_temporal_variance_confidence)
        self.assertEqual(config.wavefront_temporal_variance_strength, 0.6)
        self.assertTrue(config.wavefront_temporal_material_confidence)
        self.assertEqual(
            config.wavefront_temporal_transmission_history_scale, 0.55)
        self.assertTrue(config.wavefront_temporal_reprojection_search)
        self.assertTrue(config.wavefront_temporal_outlier_confidence)
        self.assertEqual(config.wavefront_temporal_outlier_strength, 0.7)
        self.assertEqual(config.wavefront_temporal_motion_limit_pixels, 48.0)
        self.assertTrue(config.wavefront_indirect_reuse_storage)
        self.assertTrue(config.wavefront_indirect_reuse_candidates)
        self.assertTrue(config.wavefront_indirect_reuse_temporal)
        self.assertTrue(config.wavefront_indirect_reuse_spatial)
        self.assertTrue(config.wavefront_indirect_reuse_profiling)
        self.assertTrue(config.wavefront_indirect_reuse_apply)
        self.assertEqual(config.wavefront_indirect_reuse_apply_strength, 0.4)
        self.assertEqual(config.wavefront_indirect_reuse_history_limit, 48)
        self.assertEqual(
            config.wavefront_indirect_reuse_history_motion_pixels, 24.0)
        self.assertEqual(config.wavefront_indirect_reuse_debug_view, "acceptance")
        self.assertEqual(config.wavefront_indirect_reuse_scale, 0.5)
        self.assertEqual(config.wavefront_indirect_reuse_budget_mib, 160.0)
        self.assertTrue(config.wavefront_diffuse_filter)
        self.assertEqual(config.wavefront_diffuse_filter_strength, 0.45)

    def test_rejects_invalid_configuration(self):
        for value in (0, 17):
            with self.subTest(max_bounces=value):
                with self.assertRaises(ValueError):
                    RendererConfig(max_bounces=value)
        with self.assertRaises(ValueError):
            RendererConfig(present_mode="adaptive")
        with self.assertRaises(ValueError):
            RendererConfig(swapchain_images=-1)
        with self.assertRaises(TypeError):
            RendererConfig(material_program=object())
        with self.assertRaises(ValueError):
            RendererConfig(samples_per_pixel=65)
        with self.assertRaises(ValueError):
            RendererConfig(interactive_samples_per_pixel=0)
        with self.assertRaises(ValueError):
            RendererConfig(stationary_delay_seconds=-0.1)
        with self.assertRaises(ValueError):
            RendererConfig(temporal_history=True)
        with self.assertRaises(ValueError):
            RendererConfig(temporal_history_limit=0)
        with self.assertRaises(ValueError):
            RendererConfig(adaptive_sampling=True)
        with self.assertRaises(ValueError):
            RendererConfig(samples_per_pixel=2, adaptive_min_samples=3)
        with self.assertRaises(ValueError):
            RendererConfig(area_light_samples=0)
        with self.assertRaises(ValueError):
            RendererConfig(area_light_samples=17)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_environment_samples=5)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_temporal_motion_limit_pixels=0.0)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_russian_roulette_start=1)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_russian_roulette_min_survival=0.0)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_execution_strategy="recursive")
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_hybrid_inline_bounces=2)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_hybrid_inline_bounces=4)
        self.assertEqual(
            RendererConfig(
                wavefront_execution_strategy="hybrid",
                wavefront_hybrid_inline_bounces=3,
            ).wavefront_hybrid_inline_bounces,
            3,
        )
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_auto_megakernel_transmission_fraction=1.1)
        with self.assertRaises(TypeError):
            RendererConfig(wavefront_device_local_textures=1)
        with self.assertRaises(TypeError):
            RendererConfig(wavefront_native_textures=1)
        with self.assertRaises(TypeError):
            RendererConfig(wavefront_material_bucketing=1)
        self.assertEqual(
            RendererConfig(
                wavefront_material_bucketing_start_bounce=3
            ).wavefront_material_bucketing_start_bounce,
            3,
        )
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_material_bucketing_start_bounce=0)
        with self.assertRaises(TypeError):
            RendererConfig(wavefront_persistent_coarse_tiles=1)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_persistent_coarse_tiles=True)
        self.assertTrue(RendererConfig(
            wavefront_execution_strategy="persistent",
            wavefront_persistent_coarse_tiles=True,
        ).wavefront_persistent_coarse_tiles)
        with self.assertRaises(TypeError):
            RendererConfig(wavefront_persistent_continuations=1)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_persistent_continuations=True)
        self.assertTrue(RendererConfig(
            wavefront_execution_strategy="hybrid",
            wavefront_persistent_continuations=True,
        ).wavefront_persistent_continuations)
        with self.assertRaises(TypeError):
            RendererConfig(wavefront_scene_specialization=1)
        with self.assertRaises(TypeError):
            RendererConfig(wavefront_megakernel_single_warp=1)
        self.assertTrue(RendererConfig(
            wavefront_megakernel_single_warp=True,
        ).wavefront_megakernel_single_warp)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_megakernel_group_swizzle=12)
        self.assertEqual(
            RendererConfig(wavefront_megakernel_group_swizzle=32)
            .wavefront_megakernel_group_swizzle,
            32,
        )
        with self.assertRaises(TypeError):
            RendererConfig(wavefront_ser=1)
        self.assertTrue(RendererConfig(wavefront_ser=True).wavefront_ser)
        with self.assertRaises(TypeError):
            RendererConfig(wavefront_ser_reorder=1)
        with self.assertRaises(TypeError):
            RendererConfig(wavefront_untextured_specialization=1)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_execution_strategy="ser")
        self.assertEqual(
            RendererConfig(
                wavefront_execution_strategy="ser", wavefront_ser=True
            ).wavefront_execution_strategy,
            "ser",
        )
        with self.assertRaises(ValueError):
            RendererConfig(
                wavefront_material_bucketing=True,
                wavefront_execution_strategy="megakernel",
            )
        with self.assertRaises(ValueError):
            RendererConfig(denoiser_iterations=0)
        with self.assertRaises(ValueError):
            RendererConfig(denoiser_iterations=6)
        with self.assertRaises(ValueError):
            RendererConfig(
                progressive_accumulation=True,
                denoiser_enabled=True,
            )
        with self.assertRaises(ValueError):
            RendererConfig(denoiser_variance_threshold=0.0)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_tile_capacity=0)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_tile_capacity=4194305)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_exposure=0.0)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_render_scale=0.24)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_secondary_nee_probability=0.0)
        with self.assertRaises(TypeError):
            RendererConfig(wavefront_unified_secondary_nee=1)
        with self.assertRaises(TypeError):
            RendererConfig(wavefront_unified_primary_restir=1)
        with self.assertRaises(TypeError):
            RendererConfig(wavefront_stratified_primary_restir=1)
        with self.assertRaises(ValueError):
            RendererConfig(
                wavefront_unified_primary_restir=True,
                wavefront_stratified_primary_restir=True,
            )
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_secondary_area_light_samples=17)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_restir_history_limit=0)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_restir_history_motion_pixels=0.0)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_render_scale=1.01)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_temporal_variance_strength=1.01)
        with self.assertRaises(TypeError):
            RendererConfig(wavefront_temporal_variance_confidence=1)
        with self.assertRaises(TypeError):
            RendererConfig(wavefront_temporal_material_confidence=1)
        with self.assertRaises(ValueError):
            RendererConfig(
                wavefront_temporal_transmission_history_scale=1.01)
        with self.assertRaises(TypeError):
            RendererConfig(wavefront_temporal_reprojection_search=1)
        with self.assertRaises(TypeError):
            RendererConfig(wavefront_temporal_outlier_confidence=1)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_temporal_outlier_strength=1.01)
        with self.assertRaises(TypeError):
            RendererConfig(wavefront_indirect_reuse_storage=1)
        with self.assertRaises(TypeError):
            RendererConfig(wavefront_indirect_reuse_candidates=1)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_indirect_reuse_candidates=True)
        with self.assertRaises(TypeError):
            RendererConfig(wavefront_indirect_reuse_temporal=1)
        with self.assertRaises(ValueError):
            RendererConfig(
                wavefront_indirect_reuse_storage=True,
                wavefront_indirect_reuse_temporal=True,
            )
        with self.assertRaises(TypeError):
            RendererConfig(wavefront_indirect_reuse_spatial=1)
        with self.assertRaises(ValueError):
            RendererConfig(
                wavefront_indirect_reuse_storage=True,
                wavefront_indirect_reuse_candidates=True,
                wavefront_indirect_reuse_spatial=True,
            )
        with self.assertRaises(TypeError):
            RendererConfig(wavefront_indirect_reuse_profiling=1)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_indirect_reuse_profiling=True)
        with self.assertRaises(TypeError):
            RendererConfig(wavefront_indirect_reuse_apply=1)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_indirect_reuse_apply=True)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_indirect_reuse_apply_strength=1.01)
        with self.assertRaises(TypeError):
            RendererConfig(wavefront_indirect_reuse_history_limit=32.0)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_indirect_reuse_history_limit=128)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_indirect_reuse_history_motion_pixels=0.0)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_indirect_reuse_debug_view="normals")
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_indirect_reuse_debug_view="history")
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_indirect_reuse_scale=0.1)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_indirect_reuse_budget_mib=0.0)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_dynamic_target_ms=0.0)
        with self.assertRaises(ValueError):
            RendererConfig(
                wavefront_render_scale=0.5,
                wavefront_dynamic_min_scale=0.75,
            )
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_temporal_weight=1.0)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_diffuse_filter_strength=1.1)

    def test_rejects_unsupported_samples_and_invalid_override(self):
        config = RendererConfig()
        with self.assertRaises(ValueError):
            _render_options(config, 0, None)
        with self.assertRaises(ValueError):
            _render_options(config, 1, 20)

    def test_restir_requires_one_sample_per_pixel(self):
        self.assertTrue(RendererConfig().wavefront_restir_pairwise_mis)
        config = RendererConfig(
            samples_per_pixel=1,
            wavefront_restir_di=True,
            wavefront_restir_reservoirs=8,
            wavefront_restir_candidates=2,
            wavefront_restir_history_limit=24,
            wavefront_restir_spatial_reuse=True,
            wavefront_restir_spatial_neighbors=6,
            wavefront_restir_spatial_radius=8,
            wavefront_restir_pairwise_mis=True,
            wavefront_restir_generalized_mis=True,
            wavefront_restir_generalized_balance_cap=2.0,
            wavefront_restir_specialization=False,
        )
        self.assertTrue(config.wavefront_restir_di)
        self.assertEqual(config.wavefront_restir_reservoirs, 8)
        self.assertEqual(config.wavefront_restir_candidates, 2)
        self.assertEqual(config.wavefront_restir_history_limit, 24)
        self.assertTrue(config.wavefront_restir_spatial_reuse)
        self.assertEqual(config.wavefront_restir_spatial_neighbors, 6)
        self.assertEqual(config.wavefront_restir_spatial_radius, 8)
        self.assertTrue(config.wavefront_restir_pairwise_mis)
        self.assertTrue(config.wavefront_restir_generalized_mis)
        self.assertEqual(config.wavefront_restir_generalized_balance_cap, 2.0)
        self.assertFalse(config.wavefront_restir_specialization)
        with self.assertRaises(TypeError):
            RendererConfig(wavefront_restir_specialization=1)
        with self.assertRaises(ValueError):
            RendererConfig(samples_per_pixel=2, wavefront_restir_di=True)
        with self.assertRaises(TypeError):
            RendererConfig(wavefront_restir_reservoirs=1.5)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_restir_reservoirs=0)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_restir_reservoirs=9)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_restir_candidates=5)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_restir_spatial_neighbors=9)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_restir_spatial_radius=0)
        self.assertFalse(RendererConfig(
            wavefront_restir_pairwise_mis=False
        ).wavefront_restir_pairwise_mis)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_restir_generalized_mis=True)
        with self.assertRaises(ValueError):
            RendererConfig(wavefront_restir_generalized_balance_cap=0.5)


if __name__ == "__main__":
    unittest.main()
