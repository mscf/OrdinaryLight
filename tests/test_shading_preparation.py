"""Prepared variants select the existing shaders before any device exists."""

from importlib.resources import files
import itertools
import pytest
import ordinarylight as ol
from ordinarylight.wavefront import ShadeVariant, prepare_shading
from ordinarylight.materials import (
    MaterialGraph,
    MaterialNode,
    MaterialResource,
    MaterialResourceLayout,
)


def test_packaged_variant_matrix_without_runtime(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Preparation created a runtime")

    monkeypatch.setattr(ol.VulkanRuntime, "__init__", forbidden)
    for ordinary, native, profile, overlap, scatter, skip in itertools.product(
        (False, True),
        (False, True),
        (False, True),
        (False, True),
        (0, 1, 2),
        (False, True),
    ):
        variant = ShadeVariant(
            ordinaryshade=ordinary,
            native_textures=native,
            profiling=profile,
            overlapping_volumes=overlap,
            scattering_volumes=scatter > 0,
            multiple_scattering_volumes=scatter == 2,
            volume_empty_space_skipping=skip,
        )
        from types import SimpleNamespace
        from ordinarylight.targets.vulkan.core import VulkanWavefrontExecutor

        executor = SimpleNamespace(
            core=SimpleNamespace(
                config=SimpleNamespace(wavefront_ordinaryshade_shade=ordinary)
            )
        )
        suffix = "".join(
            "_" + name
            for enabled, name in (
                (native, "native"),
                (profile, "profile"),
                (overlap, "overlap"),
                (scatter > 0, "scatter"),
                (scatter == 2, "multi"),
                (skip, "skip"),
            )
            if enabled
        )
        assert (
            VulkanWavefrontExecutor._wavefront_stage_shader(
                executor, "wavefront_shade", suffix
            )
            == variant.shader_name
        )
        prepared = prepare_shading(variant=variant)
        assert (
            prepared.spirv
            == files("ordinarylight.shaders")
            .joinpath(variant.shader_name + ".spv")
            .read_bytes()
        )
        assert prepared.spirv[:4] == b"\x03\x02#\x07"
        assert (14 in prepared.buffer_bindings) == profile
        assert dict(prepared.sampled_arrays) == (
            {13: 128, 21: 16} if native else {21: 16}
        )
    with pytest.raises(ValueError, match="requires scattering"):
        ShadeVariant(multiple_scattering_volumes=True)


def test_custom_preparation_matches_existing_compiler():
    from ordinarylight.shaders.compiler import (
        compile_wavefront_material_shader,
        find_glsl_compiler,
    )

    if find_glsl_compiler() is None:
        pytest.skip("GLSL compiler unavailable")
    program = MaterialGraph(
        {
            "value": MaterialNode("uniform", value="emission", type="vec4"),
            "rgb": MaterialNode("components", ("value",), value="rgb", type="vec3"),
        },
        {"emission": "rgb"},
        resources=(MaterialResource("emission", "uniform"),),
    ).compile()
    layout = MaterialResourceLayout.from_programs([program])
    attributes = ol.VertexAttributeLayout(())
    prepared = prepare_shading(
        programs=[program], attribute_layout=attributes, material_layout=layout
    )
    expected = compile_wavefront_material_shader(
        "wavefront_shade.comp",
        [program],
        attribute_layout=attributes,
        attribute_binding=16,
        material_resources=layout,
    )
    assert prepared.spirv == expected
    assert prepared.material_layout == layout
    with pytest.raises(ValueError, match="explicit programs"):
        prepare_shading(material_layout=layout)


@pytest.mark.parametrize('flags', [
    {'ordinaryshade': False}, {'overlapping_volumes': True},
    {'scattering_volumes': True}, {'volume_empty_space_skipping': True},
])
def test_surface_only_rejects_incompatible_variants(flags):
    with pytest.raises(ValueError, match='Surface-only'):
        ShadeVariant(**({'ordinaryshade': True, 'surface_only': True} | flags))


def test_surface_only_requires_compilation():
    with pytest.raises(ValueError, match='runtime shader compilation'):
        prepare_shading(variant=ShadeVariant(ordinaryshade=True, surface_only=True))


@pytest.mark.parametrize('profiling', [False, True])
def test_surface_only_compiles_with_denoiser_and_optional_counters(profiling):
    from ordinarylight.shaders.compiler import find_glsl_compiler
    if find_glsl_compiler() is None:
        pytest.skip('GLSL compiler unavailable')
    prepared = prepare_shading(
        variant=ShadeVariant(ordinaryshade=True, surface_only=True,
                             profiling=profiling, denoiser_signal_capture=True),
        programs=[ol.builtin_material], attribute_layout=ol.VertexAttributeLayout(()),
    )
    assert prepared.spirv[:4] == b'\x03\x02#\x07'
