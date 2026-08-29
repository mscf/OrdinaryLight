import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).parents[1]


class OrdinaryShadePrimaryTests(unittest.TestCase):
    def test_fused_primary_uses_generated_camera_helpers(self):
        helper = (
            ROOT / "ordinarylight/shaders/ordinaryshade_primary.glsl"
        ).read_text()
        primary = (
            ROOT / "ordinarylight/shaders/wavefront_primary_impl.glsl"
        ).read_text()
        self.assertIn("vec3 ordinarylight_primary_ray_origin", helper)
        self.assertIn("vec3 ordinarylight_primary_ray_direction", helper)
        self.assertIn("uint ordinarylight_primary_rng_seed", helper)
        self.assertIn("uint ordinarylight_primary_path_identity", helper)
        self.assertIn("uint ordinarylight_primary_path_flags", helper)
        self.assertIn("vec3 ordinarylight_primary_hit_position", helper)
        self.assertIn("vec3 ordinarylight_primary_shading_normal", helper)
        self.assertIn("float ordinarylight_primary_surface_class", helper)
        self.assertIn("vec4 ordinarylight_primary_interpolate_vec4", helper)
        self.assertIn("float ordinarylight_primary_uv_density", helper)
        self.assertIn("vec4 ordinarylight_primary_triangle_tangent", helper)
        self.assertIn("vec3 ordinarylight_texture_apply_rgb", helper)
        self.assertIn("vec3 ordinarylight_texture_apply_normal", helper)
        self.assertIn("vec3 ordinarylight_primary_emission", helper)
        self.assertIn("uint ordinarylight_primary_deactivate", helper)
        self.assertIn("vec4 ordinarylight_primary_invalid_position", helper)
        self.assertIn("float ordinarylight_primary_target_ior", helper)
        self.assertIn(
            "vec3 ordinarylight_primary_refracted_direction", helper
        )
        self.assertIn(
            "vec3 ordinarylight_primary_transmission_weight", helper
        )
        self.assertIn(
            "uint ordinarylight_primary_continuation_flags", helper
        )
        self.assertIn(
            "vec3 ordinarylight_primary_continuation_origin", helper
        )
        self.assertIn("vec3 ordinarylight_pbr_evaluate", helper)
        self.assertIn("vec3 ordinarylight_pbr_weight", helper)
        self.assertIn(
            "vec3 ordinarylight_analytic_light_direction", helper
        )
        self.assertIn(
            "vec3 ordinarylight_analytic_light_contribution", helper
        )
        self.assertIn("vec3 ordinarylight_area_light_position", helper)
        self.assertIn("float ordinarylight_area_light_mis", helper)
        self.assertIn("vec2 ordinarylight_environment_uv", helper)
        self.assertIn(
            "vec3 ordinarylight_environment_contribution", helper
        )
        self.assertIn(
            "float ordinarylight_unified_area_probability", helper
        )
        self.assertIn("bool ordinarylight_secondary_nee_select", helper)
        self.assertIn("float ordinarylight_emissive_hit_mis", helper)
        self.assertIn("float ordinarylight_environment_miss_mis", helper)
        self.assertIn(
            "float ordinarylight_secondary_survival_probability", helper
        )
        self.assertIn(
            "vec3 ordinarylight_secondary_direct_contribution", helper
        )
        self.assertIn(
            "vec3 ordinarylight_secondary_refracted_direction", helper
        )
        self.assertIn(
            "uint ordinarylight_secondary_medium_depth", helper
        )
        self.assertIn("float ordinarylight_secondary_cone_width", helper)
        self.assertIn(
            "vec3 ordinarylight_secondary_correct_shading_normal", helper
        )
        self.assertIn(
            "bool ordinarylight_secondary_throughput_visible", helper
        )
        self.assertIn("bool ordinarylight_secondary_capture_hit", helper)
        self.assertIn("uint ordinarylight_secondary_ser_hint", helper)
        self.assertIn("void ordinarylight_secondary_trace_query", helper)
        self.assertIn("float ordinarylight_medium_ior", helper)
        self.assertIn("void ordinarylight_set_medium_ior", helper)
        self.assertIn("bool ordinarylight_enqueue_continuation", helper)
        self.assertIn(
            "void ordinarylight_integrate_secondary_volumes", helper
        )
        self.assertIn("void ordinarylight_profile_work", helper)
        self.assertIn(
            "MaterialEvaluation ordinarylight_apply_material_program", helper
        )
        self.assertIn(
            "void ordinarylight_persistent_coarse_schedule", helper
        )
        self.assertIn("void ordinarylight_trace_remaining", helper)
        self.assertIn(
            "uint ordinarylight_secondary_area_sample_count", helper
        )
        self.assertIn('#include "ordinaryshade_primary.glsl"', primary)
        self.assertIn("#if WAVE_ORDINARYSHADE_PRIMARY_CAMERA", primary)
        self.assertIn("#if WAVE_ORDINARYSHADE_PRIMARY_STATE", primary)
        self.assertIn("#if WAVE_ORDINARYSHADE_PRIMARY_SURFACE", primary)
        self.assertIn(
            "#if WAVE_ORDINARYSHADE_PRIMARY_TEXTURE_STATE", primary
        )
        self.assertIn("#if WAVE_ORDINARYSHADE_PRIMARY_OUTPUT", primary)
        self.assertIn(
            "#if WAVE_ORDINARYSHADE_PRIMARY_TRANSMISSION", primary
        )
        self.assertIn(
            "#if WAVE_ORDINARYSHADE_PRIMARY_CONTINUATION", primary
        )
        self.assertIn("#else\n    vec3 ray_origin = camera.origin.xyz", primary)
        self.assertIn("#else\n    uint rng = hashValue", primary)

    def test_fused_secondary_backend_orchestration_has_generated_default(self):
        primary = (
            ROOT / "ordinarylight/shaders/wavefront_primary_impl.glsl"
        ).read_text()
        start = primary.index("bool ordinarylightSecondaryBounce(")
        end = primary.index("\nvoid main()", start)
        secondary = primary[start:end]

        # Raw backend operations remain only as compile-time fallbacks.
        for primitive in (
            "rayQueryInitializeEXT(",
            "rayQueryProceedEXT(",
            "integrateVolumesBeforeSurface(",
            "profileWork(",
        ):
            self.assertIn(primitive, secondary)

        for generated in (
            "ordinarylight_secondary_trace_query(",
            "WAVE_INTEGRATE_VOLUMES(",
            "WAVE_PROFILE_WORK(",
            "ordinarylight_secondary_material(",
            "ordinarylight_enqueue_continuation(",
            "ordinarylight_trace_remaining(",
        ):
            self.assertIn(generated, secondary)

        # Renderer decisions at the orchestration seams remain typed,
        # generated Ordinary Shade calls with a compile-time fallback.
        for policy in (
            "ordinarylight_secondary_ser_hint(",
            "ordinarylight_secondary_capture_position(",
            "ordinarylight_secondary_emission_visible(",
            "ordinarylight_secondary_nee_probability(",
            "ordinarylight_secondary_area_sample_count(",
            "ordinarylight_secondary_environment_sample_count(",
            "ordinarylight_secondary_average_contribution(",
        ):
            self.assertIn(policy, secondary)
        self.assertIn(
            "#if WAVE_ORDINARYSHADE_SECONDARY_CONTROL", secondary
        )

    @unittest.skipUnless(
        (ROOT.parent / "ordinaryshade/ordinaryshade").is_dir(),
        "sibling Ordinary Shade checkout is unavailable",
    )
    def test_checked_in_helpers_match_ordinaryshade_output(self):
        path = ROOT / "scripts/generate_primary_shaders.py"
        spec = importlib.util.spec_from_file_location(
            "generate_primary_shaders", path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        expected = (
            ROOT / "ordinarylight/shaders/ordinaryshade_primary.glsl"
        ).read_text()
        self.assertEqual(module.generated_source(), expected)

    def test_generated_and_fallback_primary_variants_compile(self):
        from scripts.compile_shaders import find_compiler

        compiler = find_compiler()
        source = ROOT / "ordinarylight/shaders/wavefront_primary.comp"
        with tempfile.TemporaryDirectory() as directory:
            feature_count = 19
            configurations = {(0,) * feature_count, (1,) * feature_count}
            for index in range(feature_count):
                enabled = [0] * feature_count
                enabled[index] = 1
                configurations.add(tuple(enabled))
                disabled = [1] * feature_count
                disabled[index] = 0
                configurations.add(tuple(disabled))
            for (
                camera_enabled, state_enabled,
                surface_enabled, texture_state_enabled,
                texture_application_enabled, output_enabled,
                transmission_enabled, continuation_enabled, pbr_enabled,
                analytic_lights_enabled,
                area_lights_enabled,
                environment_lights_enabled,
                unified_nee_enabled,
                emissive_mis_enabled,
                secondary_transport_enabled,
                secondary_transmission_enabled,
                secondary_surface_enabled,
                secondary_control_enabled,
                secondary_orchestration_enabled,
            ) in sorted(configurations):
                output = Path(directory) / (
                    "primary_"
                    f"{camera_enabled}_{state_enabled}_"
                    f"{surface_enabled}_{texture_state_enabled}_"
                    f"{texture_application_enabled}_{output_enabled}_"
                    f"{transmission_enabled}_{continuation_enabled}_"
                    f"{pbr_enabled}_{analytic_lights_enabled}_"
                    f"{area_lights_enabled}_{environment_lights_enabled}_"
                    f"{unified_nee_enabled}_{emissive_mis_enabled}_"
                    f"{secondary_transport_enabled}_"
                    f"{secondary_transmission_enabled}_"
                    f"{secondary_surface_enabled}_"
                    f"{secondary_control_enabled}_"
                    f"{secondary_orchestration_enabled}.spv"
                )
                subprocess.run([
                    compiler, "-V", "--target-env", "vulkan1.2",
                    "-S", "comp",
                    "-DWAVE_ORDINARYSHADE_PRIMARY_CAMERA="
                    f"{camera_enabled}",
                    "-DWAVE_ORDINARYSHADE_PRIMARY_STATE="
                    f"{state_enabled}",
                    "-DWAVE_ORDINARYSHADE_PRIMARY_SURFACE="
                    f"{surface_enabled}",
                    "-DWAVE_ORDINARYSHADE_PRIMARY_TEXTURE_STATE="
                    f"{texture_state_enabled}",
                    "-DWAVE_ORDINARYSHADE_TEXTURE_APPLICATION="
                    f"{texture_application_enabled}",
                    "-DWAVE_ORDINARYSHADE_PRIMARY_OUTPUT="
                    f"{output_enabled}",
                    "-DWAVE_ORDINARYSHADE_PRIMARY_TRANSMISSION="
                    f"{transmission_enabled}",
                    "-DWAVE_ORDINARYSHADE_PRIMARY_CONTINUATION="
                    f"{continuation_enabled}",
                    f"-DWAVE_ORDINARYSHADE_PBR={pbr_enabled}",
                    "-DWAVE_ORDINARYSHADE_ANALYTIC_LIGHTS="
                    f"{analytic_lights_enabled}",
                    "-DWAVE_ORDINARYSHADE_AREA_LIGHTS="
                    f"{area_lights_enabled}",
                    "-DWAVE_ORDINARYSHADE_ENVIRONMENT_LIGHTS="
                    f"{environment_lights_enabled}",
                    "-DWAVE_ORDINARYSHADE_UNIFIED_NEE="
                    f"{unified_nee_enabled}",
                    "-DWAVE_ORDINARYSHADE_EMISSIVE_MIS="
                    f"{emissive_mis_enabled}",
                    "-DWAVE_ORDINARYSHADE_SECONDARY_TRANSPORT="
                    f"{secondary_transport_enabled}",
                    "-DWAVE_ORDINARYSHADE_SECONDARY_TRANSMISSION="
                    f"{secondary_transmission_enabled}",
                    "-DWAVE_ORDINARYSHADE_SECONDARY_SURFACE="
                    f"{secondary_surface_enabled}",
                    "-DWAVE_ORDINARYSHADE_SECONDARY_CONTROL="
                    f"{secondary_control_enabled}",
                    "-DWAVE_ORDINARYSHADE_SECONDARY_ORCHESTRATION="
                    f"{secondary_orchestration_enabled}",
                    str(source), "-o", str(output),
                ], check=True, capture_output=True)
                self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
