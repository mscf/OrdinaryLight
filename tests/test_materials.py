import unittest

import ordinarylight as ol
from ordinarylight.shaders.compiler import (
    compile_material_shader,
    compile_wavefront_material_shader,
    find_glsl_compiler,
    material_shader_source,
    wavefront_material_shader_source,
)


class MaterialProgramTests(unittest.TestCase):
    def test_material_declares_and_specializes_vertex_attributes(self):
        @ol.material
        def vertex_tint(ctx):
            color = ctx.attribute("color", components=3)
            weight = ctx.attribute("weight")
            return ol.MaterialEvaluation(
                base_color=color * weight,
                emission=ctx.emission,
                metallic=ctx.metallic,
                roughness=ctx.roughness,
                transmission=ctx.transmission,
                ior=ctx.ior,
                attenuation_color=ctx.attenuation_color,
                attenuation_distance=ctx.attenuation_distance,
            )

        self.assertEqual(
            vertex_tint.required_attributes, (("color", 3), ("weight", 1))
        )
        symbolic = vertex_tint.glsl()
        self.assertIn("waveVertexAttribute3(WAVE_ATTRIBUTE_color)", symbolic)
        layout = ol.VertexAttributeLayout((("weight", 1), ("color", 3)))
        specialized = vertex_tint.glsl(attribute_slots={
            name: layout.slot(name) for name, _ in layout.channels
        })
        self.assertIn("waveVertexAttribute3(1u)", specialized)
        self.assertIn("waveVertexAttribute1(0u)", specialized)
        with self.assertRaises(ValueError):
            material_shader_source("ray_query.comp", vertex_tint)
        source = material_shader_source(
            "ray_query.comp", vertex_tint, attribute_layout=layout
        )
        self.assertNotIn("WAVE_ATTRIBUTE_", source)
        self.assertIn("binding = 15", source)
        self.assertIn("wave_attribute_primitive = primitive", source)
        if find_glsl_compiler():
            spirv = compile_material_shader(
                "ray_query.comp", vertex_tint, attribute_layout=layout
            )
            self.assertEqual(spirv[:4], b"\x03\x02#\x07")

    def test_material_rejects_conflicting_attribute_declarations(self):
        with self.assertRaises(ValueError):
            @ol.material
            def invalid(ctx):
                ctx.attribute("value", components=2)
                value = ctx.attribute("value", components=3)
                return ol.MaterialEvaluation(
                    value, ctx.emission, ctx.metallic, ctx.roughness,
                    ctx.transmission, ctx.ior, ctx.attenuation_color,
                    ctx.attenuation_distance,
                )

    def test_vulkan_material_layout_is_derived_without_default_allocation(self):
        scene = ol.Scene()
        scene.add_mesh(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0)), ((0, 1, 2),),
            attributes={"color": ((1, 0, 0),) * 3},
        )
        from ordinarylight.targets.vulkan.core import VulkanRayQueryCore
        self.assertIsNone(VulkanRayQueryCore._material_attribute_layout(
            scene, (ol.builtin_material,)
        ))
        empty_layout = VulkanRayQueryCore._material_attribute_layout(
            scene, (ol.unlit_material,)
        )
        self.assertEqual(empty_layout.channels, ())

        @ol.material
        def colored(ctx):
            return ol.MaterialEvaluation(
                ctx.attribute("color", components=3), ctx.emission,
                ctx.metallic, ctx.roughness, ctx.transmission, ctx.ior,
                ctx.attenuation_color, ctx.attenuation_distance,
            )

        layout = VulkanRayQueryCore._material_attribute_layout(
            scene, (ol.builtin_material, colored)
        )
        self.assertEqual(layout.channels, (("color", 3),))
        self.assertEqual(layout.pack(scene).shape, (3, 1, 4))

    def test_staged_attribute_material_shaders_compile(self):
        @ol.material
        def colored(ctx):
            return ol.MaterialEvaluation(
                ctx.attribute("color", components=3), ctx.emission,
                ctx.metallic, ctx.roughness, ctx.transmission, ctx.ior,
                ctx.attenuation_color, ctx.attenuation_distance,
            )

        layout = ol.VertexAttributeLayout((("color", 3),))
        for shader, binding in (
            ("wavefront_primary.comp", 24),
            ("wavefront_shade.comp", 16),
        ):
            with self.subTest(shader=shader):
                source = wavefront_material_shader_source(
                    shader, colored, attribute_layout=layout,
                    attribute_binding=binding,
                )
                self.assertIn(f"binding = {binding}", source)
                self.assertIn("waveApplyMaterialProgram(material", source)
                if shader == "wavefront_primary.comp":
                    self.assertIn(
                        "#define WAVE_CUSTOM_MATERIAL_PROGRAM 1", source
                    )
                    self.assertIn(
                        "ordinarylight_apply_material_program(", source
                    )
                self.assertNotIn("WAVE_ATTRIBUTE_color", source)
                multiple_source = wavefront_material_shader_source(
                    shader, colored, attribute_layout=layout,
                    attribute_binding=binding, scattering_volumes=True,
                    multiple_scattering_volumes=True,
                )
                self.assertIn(
                    "#define WAVE_VOLUME_MULTIPLE_SCATTERING 1",
                    multiple_source,
                )
                skipping_source = wavefront_material_shader_source(
                    shader, colored, attribute_layout=layout,
                    attribute_binding=binding,
                    volume_empty_space_skipping=True,
                )
                self.assertIn(
                    "#define WAVE_VOLUME_EMPTY_SPACE_SKIPPING 1",
                    skipping_source,
                )
                if find_glsl_compiler():
                    spirv = compile_wavefront_material_shader(
                        shader, colored, attribute_layout=layout,
                        attribute_binding=binding,
                    )
                    self.assertEqual(spirv[:4], b"\x03\x02#\x07")
                    multiple_spirv = compile_wavefront_material_shader(
                        shader, colored, attribute_layout=layout,
                        attribute_binding=binding, scattering_volumes=True,
                        multiple_scattering_volumes=True,
                    )
                    self.assertEqual(multiple_spirv[:4], b"\x03\x02#\x07")
                    skipping_spirv = compile_wavefront_material_shader(
                        shader, colored, attribute_layout=layout,
                        attribute_binding=binding,
                        volume_empty_space_skipping=True,
                    )
                    self.assertEqual(skipping_spirv[:4], b"\x03\x02#\x07")

        candidate_source = wavefront_material_shader_source(
            "wavefront_shade_candidate.glsl", colored,
            attribute_layout=layout, attribute_binding=16,
        )
        self.assertIn("binding = 16", candidate_source)
        self.assertIn(
            "wave_attribute_primitive = loaded.hit.primitive_index",
            candidate_source,
        )
        self.assertIn("wave_attribute_weights = surface.weights", candidate_source)
        self.assertNotIn("WAVE_ATTRIBUTE_color", candidate_source)
        self.assertEqual(
            candidate_source.count(
                "MaterialEvaluation evaluateMaterial(MaterialData"
            ),
            1,
        )
        if find_glsl_compiler():
            candidate_spirv = compile_wavefront_material_shader(
                "wavefront_shade_candidate.glsl", colored,
                attribute_layout=layout, attribute_binding=16,
            )
            self.assertEqual(candidate_spirv[:4], b"\x03\x02#\x07")
            candidate_native_profile = compile_wavefront_material_shader(
                "wavefront_shade_candidate.glsl", colored,
                attribute_layout=layout, attribute_binding=16,
                native_textures=True, profiling=True,
            )
            self.assertEqual(
                candidate_native_profile[:4], b"\x03\x02#\x07"
            )

    def test_python_material_generates_typed_glsl(self):
        @ol.material
        def tinted_glass(ctx):
            tint = ol.mix(ctx.base_color, ol.vec3(1.0), 0.25)
            return ol.MaterialEvaluation(
                base_color=tint,
                emission=ctx.emission,
                metallic=0.0 * ctx.metallic,
                roughness=ctx.roughness,
                transmission=ctx.transmission,
                ior=ol.select(ctx.entering, ctx.ior, 1.0),
                attenuation_color=ctx.attenuation_color,
                attenuation_distance=ctx.attenuation_distance,
            )

        source = tinted_glass.glsl()
        self.assertIn("MaterialEvaluation evaluateMaterial", source)
        self.assertIn("mix(material.base_roughness.rgb, vec3(1.0), 0.25)", source)
        self.assertIn("entering ? material.ior_distance.x : 1.0", source)
        complete = material_shader_source("ray_query.comp", tinted_glass)
        self.assertIn("MaterialEvaluation evaluateMaterial_0", complete)
        self.assertIn("result.base_color = mix(material.base_roughness.rgb", complete)
        self.assertEqual(
            complete.count("MaterialEvaluation evaluateMaterial(MaterialData"), 1
        )
        if find_glsl_compiler():
            spirv = compile_material_shader("ray_query.comp", tinted_glass)
            self.assertEqual(spirv[:4], b"\x03\x02#\x07")

    def test_rejects_python_branching_on_symbolic_value(self):
        with self.assertRaises(TypeError):
            @ol.material
            def invalid(ctx):
                color = ctx.base_color if ctx.entering else ol.vec3(0.0)
                return ol.MaterialEvaluation(
                    color, ctx.emission, ctx.metallic, ctx.roughness,
                    ctx.transmission, ctx.ior, ctx.attenuation_color,
                    ctx.attenuation_distance,
                )

    def test_rejects_wrong_result_type(self):
        with self.assertRaises(TypeError):
            @ol.material
            def invalid(ctx):
                return ol.MaterialEvaluation(
                    ctx.roughness, ctx.emission, ctx.metallic, ctx.roughness,
                    ctx.transmission, ctx.ior, ctx.attenuation_color,
                    ctx.attenuation_distance,
                )

    def test_multiple_programs_generate_dispatcher_and_ids(self):
        @ol.material
        def red(ctx):
            return ol.MaterialEvaluation(
                ol.vec3(1.0, 0.0, 0.0), ctx.emission, ctx.metallic,
                ctx.roughness, ctx.transmission, ctx.ior,
                ctx.attenuation_color, ctx.attenuation_distance,
            )

        scene = ol.Scene()
        vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
        indices = [[0, 1, 2]]
        scene.add_mesh(vertices, indices, ol.Material())
        scene.add_mesh(vertices, indices, ol.Material(program=red))
        programs = scene.material_programs(ol.builtin_material)
        packed = scene.triangle_material_data(programs, ol.builtin_material)
        self.assertEqual(tuple(packed[:, 3, 2]), (0.0, 1.0))
        source = material_shader_source("ray_query.comp", programs)
        self.assertIn("if (program_id == 1)", source)
        if find_glsl_compiler():
            spirv = compile_material_shader("ray_query.comp", programs)
            self.assertEqual(spirv[:4], b"\x03\x02#\x07")

    def test_surface_response_controls_scattering(self):
        @ol.material
        def mirror(ctx):
            return ol.SurfaceResponse(
                emission=(0.0, 0.0, 0.0),
                weight=ctx.base_color,
                next_direction=ol.reflect(ctx.direction, ctx.normal),
                event=ol.SCATTER_REFLECTION,
                pdf=1.0,
            )

        generated = mirror.glsl()
        self.assertIn("result.custom_scattering = 1.0", generated)
        self.assertIn("result.next_direction = reflect(direction, normal)", generated)
        self.assertIn("result.event = 2.0", generated)
        if find_glsl_compiler():
            spirv = compile_material_shader("ray_query.comp", mirror)
            self.assertEqual(spirv[:4], b"\x03\x02#\x07")

            wavefront = compile_wavefront_material_shader(
                "wavefront_shade_candidate.glsl", mirror,
                attribute_layout=ol.VertexAttributeLayout(()),
                attribute_binding=16,
            )
            self.assertEqual(wavefront[:4], b"\x03\x02#\x07")
            primary = compile_wavefront_material_shader(
                "wavefront_primary.comp", mirror,
                attribute_layout=ol.VertexAttributeLayout(()),
                attribute_binding=24,
            )
            self.assertEqual(primary[:4], b"\x03\x02#\x07")
            handwritten = compile_wavefront_material_shader(
                "wavefront_shade.comp", mirror,
                attribute_layout=ol.VertexAttributeLayout(()),
                attribute_binding=16,
            )
            self.assertEqual(handwritten[:4], b"\x03\x02#\x07")

        staged = wavefront_material_shader_source(
            "wavefront_shade_candidate.glsl", mirror,
            attribute_layout=ol.VertexAttributeLayout(()),
            attribute_binding=16,
        )
        self.assertIn("if ((evaluated.custom_scattering > 0.5))", staged)
        self.assertIn("next_direction = normalize(evaluated.next_direction)", staged)
        self.assertIn("path.throughput.rgb * evaluated.weight", staged)

    def test_stochastic_fresnel_program_compiles(self):
        @ol.material
        def fresnel_glass(ctx):
            cosine = -ol.dot(ctx.direction, ctx.normal)
            probability = ol.fresnel_schlick(
                cosine, ctx.current_ior, ctx.exterior_ior
            )
            reflect_path = ctx.random_u < probability
            selected_pdf = ol.select(reflect_path, probability, 1.0 - probability)
            eta = ctx.current_ior / ctx.exterior_ior
            next_direction = ol.select(
                reflect_path,
                ol.reflect(ctx.direction, ctx.normal),
                ol.refract(ctx.direction, ctx.normal, eta),
            )
            return ol.SurfaceResponse(
                emission=ctx.emission,
                weight=ctx.base_color * selected_pdf,
                next_direction=next_direction,
                event=ol.select(
                    reflect_path,
                    ol.SCATTER_REFLECTION,
                    ol.SCATTER_TRANSMISSION,
                ),
                pdf=selected_pdf,
            )

        generated = fresnel_glass.glsl()
        self.assertIn("waveFresnelSchlick", generated)
        self.assertIn("random_u <", generated)
        if find_glsl_compiler():
            spirv = compile_material_shader("ray_query_image.comp", fresnel_glass)
            self.assertEqual(spirv[:4], b"\x03\x02#\x07")


if __name__ == "__main__":
    unittest.main()
