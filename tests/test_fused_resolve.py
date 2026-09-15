"""Combined resolve policy, compilation and public misuse checks."""
from dataclasses import replace
import pytest
from ordinarylight.targets.vulkan.api import RendererConfig
from ordinarylight.targets.vulkan.relax_prepare_graph import fused_resolve_enabled

@pytest.mark.parametrize('enabled',(False,True))
@pytest.mark.parametrize('sampled',(False,True))
@pytest.mark.parametrize('reuse',(False,True))
def test_fused_resolve_policy(enabled,sampled,reuse):
    config=replace(RendererConfig(),denoiser_enabled=True,temporal_history=True,progressive_accumulation=True,denoiser_fused_resolve=enabled,
        denoiser_sampled_indirect=sampled,wavefront_indirect_reuse_candidates=reuse,wavefront_indirect_reuse_storage=reuse)
    assert fused_resolve_enabled(config)==(enabled and sampled and not reuse)

@pytest.mark.parametrize('custom',(False,True))
def test_combined_shader_compiles(custom):
    from ordinarylight.runtime.relax_prepare import _hdr_shader
    assert _hdr_shader(custom)[:4]==b'\x03\x02#\x07'

def test_flag_rejects_non_boolean():
    with pytest.raises(TypeError,match='denoiser_fused_resolve'):
        replace(RendererConfig(),denoiser_fused_resolve=1)


def test_disabled_denoiser_retains_separate_resolve():
    assert not fused_resolve_enabled(replace(RendererConfig(),
        denoiser_fused_resolve=True,denoiser_sampled_indirect=True))


def test_environment_early_reject_compiles_and_validates():
    import ordinarylight as ol
    from ordinarylight.shaders.compiler import compile_wavefront_material_shader
    options=dict(attribute_layout=ol.VertexAttributeLayout(()),attribute_binding=24)
    result=compile_wavefront_material_shader('wavefront_primary.comp',[ol.builtin_material],
        environment_early_reject=True,**options)
    assert result[:4]==b'\x03\x02#\x07'
    with pytest.raises(TypeError,match='environment_early_reject'):
        compile_wavefront_material_shader('wavefront_primary.comp',[ol.builtin_material],
            environment_early_reject=1,**options)
    with pytest.raises(TypeError,match='wavefront_environment_early_reject'):
        replace(RendererConfig(),wavefront_environment_early_reject=1)
