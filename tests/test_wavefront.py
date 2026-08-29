import unittest
from pathlib import Path

import numpy as np

import ordinarylight as ol


class WavefrontLayoutTests(unittest.TestCase):
    def test_custom_attribute_storage_is_opt_in_and_scene_owned(self):
        backend = (Path(ol.__file__).parent / "targets" / "vulkan" / "core.py").read_text()
        self.assertIn("self.scene_custom_attribute_buffer = None", backend)
        self.assertIn("if custom_attribute_layout is not None:", backend)
        self.assertIn("custom_attribute_layout.pack(scene)", backend)
        self.assertIn("resources.custom_attribute_buffer", backend)
        self.assertIn("storage(23), storage(24)", backend)
        self.assertIn("storage(15),\n            storage(16)", backend)
        self.assertIn("24, self.core.scene_custom_attribute_buffer", backend)
        self.assertIn("16, self.core.scene_custom_attribute_buffer", backend)
        self.assertIn("def ensure_custom_material_pipelines(self):", backend)
        self.assertIn("strategy = \"wavefront\"", backend)
        self.assertIn("self.custom_shade_pipeline or self.shade_pipeline", backend)

    def test_scene_blases_are_refittable_for_equal_topology_updates(self):
        backend = (
            Path(ol.__file__).parent / "targets" / "vulkan" / "core.py"
        ).read_text()
        self.assertIn("class SceneBlas:", backend)
        self.assertIn(
            "VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR", backend
        )
        self.assertIn(
            "mode=vk.VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR", backend
        )
        self.assertIn("def _refit_scene_blases(self, entries):", backend)
        self.assertIn("item.indices = item.mesh.indices.copy()", backend)
        self.assertIn("item.vertices = item.mesh.vertices.copy()", backend)

    def test_ray_query_shaders_apply_instance_primitive_offsets(self):
        shader_root = Path(ol.__file__).parent / "shaders"
        for name in (
            "ray_query.comp", "ray_query_image.comp",
            "wavefront_primary_impl.glsl", "wavefront_intersect.comp",
            "wavefront_intersect_bucketed.comp", "wavefront_shade.comp",
        ):
            with self.subTest(shader=name):
                source = (shader_root / name).read_text()
                queries = source.count(
                    "rayQueryGetIntersectionPrimitiveIndexEXT"
                )
                offsets = source.count(
                    "rayQueryGetIntersectionInstanceCustomIndexEXT"
                )
                self.assertGreater(queries, 0)
                self.assertEqual(offsets, queries)

    def test_wavefront_stages_share_lighting_implementation(self):
        shader_dir = Path(__file__).parents[1] / "ordinarylight" / "shaders"
        primary = (shader_dir / "wavefront_primary.comp").read_text()
        megakernel = (shader_dir / "wavefront_megakernel.comp").read_text()
        hybrid = (shader_dir / "wavefront_hybrid.comp").read_text()
        primary_impl = (
            shader_dir / "wavefront_primary_impl.glsl"
        ).read_text()
        shade = (shader_dir / "wavefront_shade.comp").read_text()
        lighting = (shader_dir / "wavefront_lighting.glsl").read_text()
        textures = (shader_dir / "wavefront_textures.glsl").read_text()
        include = '#include "wavefront_lighting.glsl"'
        self.assertIn('#include "wavefront_primary_impl.glsl"', primary)
        self.assertIn('#include "wavefront_primary_impl.glsl"', megakernel)
        self.assertIn('#include "wavefront_primary_impl.glsl"', hybrid)
        self.assertIn("#define WAVE_HYBRID 1", hybrid)
        self.assertIn(include, primary_impl)
        self.assertIn(include, shade)
        self.assertNotIn("vec3 sampleAreaLight(", primary_impl)
        self.assertNotIn("vec3 sampleAreaLight(", shade)
        self.assertIn("vec3 sampleAreaLight(", lighting)
        self.assertIn("vec3 sampleEnvironment(", lighting)
        self.assertIn("float powerHeuristic(", lighting)
        self.assertIn("float ggxDistribution(", lighting)
        self.assertIn("vec3 evaluatePbr(", lighting)
        self.assertIn("void samplePbr(", lighting)
        self.assertNotIn("else if (metallic > 0.5)", primary_impl)
        self.assertNotIn("else if (metallic > 0.5)", shade)
        self.assertIn("vec3 applyNormalTexture(", textures)
        self.assertIn("material.attenuation_transmission.a *= transmission", textures)
        self.assertEqual(primary_impl.count("applyNormalTexture("), 2)
        self.assertEqual(shade.count("applyNormalTexture("), 1)
        self.assertIn("uint textureMipOffset(", textures)
        self.assertIn("float triangleUvDensity(", textures)
        self.assertIn("floatBitsToUint(cone_width)", primary_impl)
        self.assertIn("uintBitsToFloat(input_ray.padding_a)", shade)
        self.assertIn("uint inline_bounces", primary_impl)
        self.assertIn("min(push.inline_bounces, push.max_bounces)", primary_impl)
        self.assertIn("#define WAVE_LOCAL_SIZE_X 8", primary_impl)
        self.assertIn("#define WAVE_LOCAL_SIZE_Y 8", primary_impl)
        self.assertIn("#define WAVE_UNIFIED_PRIMARY_RESTIR 0u", primary_impl)
        self.assertIn("#define WAVE_GENERALIZED_RESTIR 0u", primary_impl)
        continuation = (
            shader_dir / "wavefront_persistent_continuation.comp"
        ).read_text()
        self.assertIn("#define WAVE_LOCAL_SIZE_X 64", continuation)
        self.assertIn("#define WAVE_LOCAL_SIZE_Y 1", continuation)
        indirect = (
            shader_dir / "wavefront_prepare_indirect.comp"
        ).read_text()
        self.assertIn("active_count + uint(63)", indirect)
        self.assertIn("/ uint(64)", indirect)
        coarse = (
            shader_dir / "wavefront_persistent_coarse.comp"
        ).read_text()
        self.assertIn("#define WAVE_LOCAL_SIZE_X 8", coarse)
        self.assertIn("#define WAVE_LOCAL_SIZE_Y 8", coarse)
        self.assertIn("uint pathRng(WavePathState path)", primary_impl)
        self.assertNotIn("uvec4 rng;\n};", primary_impl)

    def test_secondary_nee_uses_temporally_stratified_path_identity(self):
        shader_dir = Path(__file__).parents[1] / "ordinarylight" / "shaders"
        lighting = (shader_dir / "wavefront_lighting.glsl").read_text()
        generate = (shader_dir / "wavefront_generate.comp").read_text()
        primary_impl = (
            shader_dir / "wavefront_primary_impl.glsl"
        ).read_text()
        shade = (shader_dir / "wavefront_shade.comp").read_text()

        self.assertIn("bool selectSecondaryNee(", lighting)
        self.assertIn("bitfieldReverse(frame_index)", lighting)
        self.assertIn("uint sample_index = frame_sample & 255u", lighting)
        identity = "(push.tile_frame.z << 8u) | (push.tile_frame.w & 255u)"
        generated_identity = (
            "(push.tile_frame.z << uint(8)) | "
            "(push.tile_frame.w & uint(255))"
        )
        self.assertTrue(
            identity in generate or generated_identity in generate,
            "generated primary rays must preserve frame/sample path identity",
        )
        self.assertIn(identity, primary_impl)
        self.assertIn("path.metadata.y, next_bounce", shade)
        self.assertIn("path.metadata.y,\n                next_bounce", primary_impl)

    def test_generated_primary_camera_uses_storage_buffer_abi(self):
        generate = (
            Path(__file__).parents[1]
            / "ordinarylight" / "shaders" / "wavefront_generate.comp"
        ).read_text()
        self.assertIn(
            "layout(std430, set = 0, binding = 3) readonly buffer camera_Block",
            generate,
        )
        self.assertNotIn(
            "layout(std140, set = 0, binding = 3) uniform camera_Block",
            generate,
        )

    def test_hot_path_packs_rng_into_unused_vector_lanes(self):
        self.assertEqual(ol.HOT_PATH_STATE_DTYPE.itemsize, 48)
        self.assertEqual(ol.HOT_PATH_STATE_DTYPE.fields["throughput"][1], 0)
        self.assertEqual(ol.HOT_PATH_STATE_DTYPE.fields["radiance"][1], 16)
        self.assertEqual(ol.HOT_PATH_STATE_DTYPE.fields["metadata"][1], 32)
        self.assertEqual(ol.MEDIUM_STACK_DTYPE.itemsize, 64)
        self.assertEqual(
            ol.MEDIUM_STACK_DTYPE.fields["ior"][0].shape,
            (ol.MAX_MEDIUM_STACK_DEPTH,),
        )

    def test_std430_record_sizes_and_offsets(self):
        self.assertEqual(ol.RAY_DTYPE.itemsize, 48)
        self.assertEqual(ol.RAY_DTYPE.fields["path_index"][1], 32)
        self.assertEqual(ol.HIT_DTYPE.itemsize, 48)
        self.assertEqual(ol.HIT_DTYPE.fields["primitive_index"][1], 28)
        self.assertEqual(ol.HIT_DTYPE.fields["path_index"][1], 44)
        self.assertEqual(ol.HIT_DTYPE.fields["ray_index"][1], 40)
        self.assertEqual(ol.PATH_STATE_DTYPE.itemsize, 128)
        self.assertEqual(ol.PATH_STATE_DTYPE.fields["ior_stack"][1], 64)
        self.assertEqual(
            ol.PATH_STATE_DTYPE.fields["ior_stack"][0].shape,
            (ol.MAX_MEDIUM_STACK_DEPTH,),
        )
        self.assertEqual(ol.SECONDARY_PATH_STATE_DTYPE.itemsize, 64)
        self.assertEqual(
            ol.SECONDARY_PATH_STATE_DTYPE.fields["position_valid"][1], 0)
        self.assertEqual(
            ol.SECONDARY_PATH_STATE_DTYPE.fields["normal_pdf"][1], 16)
        self.assertEqual(
            ol.SECONDARY_PATH_STATE_DTYPE.fields["primary_throughput"][1], 32)
        self.assertEqual(
            ol.SECONDARY_PATH_STATE_DTYPE.fields["primary_radiance"][1], 48)
        self.assertEqual(ol.RESOLVED_PIXEL_DTYPE.itemsize, 32)
        self.assertEqual(ol.RESOLVED_PIXEL_DTYPE.fields["metadata"][1], 16)

    def test_queue_storage_views_share_memory(self):
        layout = ol.WavefrontQueueLayout(ol.RAY_DTYPE, 7)
        storage = layout.empty_host_buffer()
        self.assertEqual(storage.nbytes, 16 + 7 * 48)
        header = layout.header_view(storage)
        records = layout.records_view(storage)
        self.assertEqual(int(header["capacity"][0]), 7)
        self.assertEqual(int(header["count"][0]), 0)
        self.assertEqual(records.shape, (7,))
        records[2]["path_index"] = 19
        self.assertEqual(int(layout.records_view(storage)[2]["path_index"]), 19)

    def test_queue_rejects_bad_capacity_and_storage(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            ol.WavefrontQueueLayout(ol.HIT_DTYPE, 0)
        layout = ol.WavefrontQueueLayout(ol.HIT_DTYPE, 2)
        with self.assertRaisesRegex(ValueError, "too small"):
            layout.records_view(np.zeros(layout.byte_size - 1, np.uint8))

    def test_glsl_contract_contains_matching_fields(self):
        source = ol.wavefront_glsl_structs()
        self.assertIn("struct WaveRay", source)
        self.assertIn("uint path_index", source)
        self.assertIn("uint padding_a", source)
        self.assertNotIn("uvec3 padding", source)
        self.assertIn("float ior_stack[WAVE_MAX_MEDIUM_STACK_DEPTH]", source)
        self.assertIn("struct WaveQueueHeader", source)

    def test_initial_pipeline_splits_generation_and_intersection(self):
        pipeline = ol.create_wavefront_pipeline(include_shading=False)
        self.assertEqual(pipeline.stage_names, (
            "wavefront_generate_primary", "wavefront_intersect",
        ))
        self.assertIn("ray_queue", pipeline.output_resources)
        self.assertIn("hit_queue", pipeline.output_resources)
        self.assertIn("geometry", pipeline.initial_resources)
        self.assertNotIn("radiance", pipeline.output_resources)

        complete = ol.create_wavefront_pipeline()
        self.assertEqual(complete.stage_names[-1], "wavefront_shade")
        self.assertIn("radiance", complete.output_resources)

    def test_native_texture_variants_are_packaged(self):
        shader_dir = Path(__file__).parents[1] / "ordinarylight" / "shaders"
        for name in (
            "wavefront_primary", "wavefront_hybrid",
            "wavefront_megakernel", "wavefront_shade",
        ):
            path = shader_dir / f"{name}_native.comp.spv"
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 16)
        for name in (
            "wavefront_primary", "wavefront_hybrid", "wavefront_megakernel",
            "wavefront_persistent",
            "wavefront_persistent_coarse",
            "wavefront_persistent_continuation",
        ):
            for suffix in ("_profile", "_native_profile"):
                path = shader_dir / f"{name}{suffix}.comp.spv"
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 16)
        for suffix in ("_opaque", "_native_opaque", "_opaque_profile",
                       "_native_opaque_profile"):
            path = shader_dir / f"wavefront_megakernel{suffix}.comp.spv"
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 16)
        for suffix in (
            "_opaque_untextured", "_native_opaque_untextured",
            "_opaque_untextured_profile",
            "_native_opaque_untextured_profile",
        ):
            path = shader_dir / f"wavefront_megakernel{suffix}.comp.spv"
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 16)
        bgra_reconstruct = shader_dir / "wavefront_reconstruct_bgra.comp.spv"
        self.assertTrue(bgra_reconstruct.is_file())
        self.assertGreater(bgra_reconstruct.stat().st_size, 16)
        ser_probe = shader_dir / "ser_probe.rgen.spv"
        self.assertTrue(ser_probe.is_file())
        self.assertGreater(ser_probe.stat().st_size, 16)
        ser_reorder_probe = shader_dir / "ser_probe_ser.rgen.spv"
        self.assertTrue(ser_reorder_probe.is_file())
        self.assertGreater(ser_reorder_probe.stat().st_size, 16)
        ser_miss_probe = shader_dir / "ser_probe.rmiss.spv"
        self.assertTrue(ser_miss_probe.is_file())
        self.assertGreater(ser_miss_probe.stat().st_size, 16)
        for name in (
            "wavefront_megakernel.rgen.spv",
            "wavefront_megakernel_ser.rgen.spv",
        ):
            path = shader_dir / name
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 16)
        for suffix in (
            "_opaque_untextured_production",
            "_native_opaque_untextured_production",
        ):
            path = shader_dir / f"wavefront_megakernel{suffix}.comp.spv"
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 16)
            hybrid_path = shader_dir / f"wavefront_hybrid{suffix}.comp.spv"
            self.assertTrue(hybrid_path.is_file())
            self.assertGreater(hybrid_path.stat().st_size, 16)
            for swizzle_width in (8, 16, 32):
                swizzle_path = shader_dir / (
                    f"wavefront_megakernel{suffix}"
                    f"_swizzle{swizzle_width}.comp.spv"
                )
                self.assertTrue(swizzle_path.is_file())
                self.assertGreater(swizzle_path.stat().st_size, 16)
        for suffix in (
            "_opaque_untextured_wg32",
            "_native_opaque_untextured_wg32",
            "_opaque_untextured_wg32_profile",
            "_native_opaque_untextured_wg32_profile",
        ):
            path = shader_dir / f"wavefront_megakernel{suffix}.comp.spv"
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 16)
        backend = (Path(__file__).parents[1] / "ordinarylight"
                   / "targets" / "vulkan" / "core.py").read_text()
        self.assertIn("self.megakernel_opaque_pipeline", backend)
        self.assertIn("self.megakernel_opaque_untextured_pipeline", backend)
        self.assertIn(
            "self.megakernel_opaque_untextured_production_pipeline", backend
        )
        self.assertIn(
            "self.hybrid_opaque_untextured_production_pipeline", backend
        )
        self.assertIn("self.reconstruct_bgra_pipeline", backend)
        self.assertIn("self.swapchain_bgra_storage", backend)
        self.assertIn("self.megakernel_swizzle_pipelines", backend)
        self.assertIn("self.ser_reordering_supported", backend)
        self.assertIn("self.cmd_trace_rays", backend)
        self.assertIn(
            "PIPELINE_BIND_POINT_RAY_TRACING_KHR = 1000165000", backend
        )
        self.assertIn("self.medium_capacity = (", backend)
        self.assertIn("def _ensure_medium_buffer(self, scene):", backend)
        self.assertIn('"wavefront_medium_stack_bytes"', backend)
        self.assertIn(
            'strategy == "megakernel" and opaque_specialization', backend)
        texture_source = (shader_dir / "wavefront_textures.glsl").read_text()
        self.assertIn(
            "native_textures[nonuniformEXT(descriptor_index)]",
            texture_source,
        )
        primary_source = (
            shader_dir / "wavefront_primary_impl.glsl"
        ).read_text()
        self.assertIn("#if !WAVE_UNTEXTURED_SECONDARY", primary_source)
        self.assertIn("#if WAVE_UNTEXTURED_PRIMARY", primary_source)
        self.assertIn("material.texture_parameters.w = 1.0", primary_source)
        opaque_size = (
            shader_dir / "wavefront_megakernel_opaque.comp.spv"
        ).stat().st_size
        untextured_size = (
            shader_dir / "wavefront_megakernel_opaque_untextured.comp.spv"
        ).stat().st_size
        self.assertLess(untextured_size, opaque_size)


if __name__ == "__main__":
    unittest.main()
