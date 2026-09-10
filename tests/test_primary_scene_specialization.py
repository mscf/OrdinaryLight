"""Primary specialization must follow the scene's visible-volume lifetime."""
from types import SimpleNamespace

import pytest
import vulkan as vk
import ordinarylight as ol
from ordinarylight.targets.vulkan.core import VulkanWavefrontExecutor


def test_primary_specialization_tracks_volumes_and_user_opt_out(monkeypatch):
    from ordinarylight.shaders import compiler
    from ordinarylight.wavefront import preparation
    captured = []
    variants = []
    def compile_primary(*args, **kwargs):
        captured.append(kwargs['surface_only'])
        variants.append((kwargs['opaque_primary'], kwargs['production_restir']))
        return b'primary'
    monkeypatch.setattr(compiler, 'compile_wavefront_material_shader', compile_primary)
    monkeypatch.setattr(preparation, 'prepare_shading', lambda **kw: SimpleNamespace(spirv=b'shade'))
    for name in ('vkDeviceWaitIdle', 'vkDestroyPipeline', 'vkDestroyShaderModule'):
        monkeypatch.setattr(vk, name, lambda *args: None)
    scene = SimpleNamespace(visible_volumes=[], material_programs=lambda default: [default])
    executor = VulkanWavefrontExecutor.__new__(VulkanWavefrontExecutor)
    executor.core = SimpleNamespace(
        config=ol.RendererConfig(wavefront_restir_shared_primary=True),
        scene_resources=SimpleNamespace(scene=scene, volume_empty_space_skipping=False),
        scene_custom_attribute_layout=None, native_textures_enabled=False,
        material_resources=None, device='device',
        _use_opaque_scene_specialization=lambda scene: not scene.visible_volumes,
    )
    executor.custom_material_signature = None
    executor.custom_primary_pipeline = executor.custom_shade_pipeline = None
    executor.custom_primary_module = executor.custom_shade_module = None
    executor.primary_pipeline_layout = executor.shade_pipeline_layout = 'layout'
    executor._pipeline_bytes = lambda *args: ('module', 'pipeline')
    executor.ensure_custom_material_pipelines()
    executor.ensure_custom_material_pipelines()  # Unchanged scene reuses pipelines.
    assert captured == [True]
    assert variants == [(True, True)]
    scene.visible_volumes = [SimpleNamespace(material=SimpleNamespace(scattering_scale=0))]
    executor.ensure_custom_material_pipelines()
    assert captured == [True, False]
    assert variants[-1] == (False, True)
    scene.visible_volumes = []
    executor.ensure_custom_material_pipelines()
    assert captured == [True, False, True]
    from dataclasses import replace
    executor.core.config = replace(executor.core.config, wavefront_primary_scene_specialization=False)
    executor.ensure_custom_material_pipelines()
    assert captured == [True, False, True, False]
    assert variants[-1] == (False, True)
    executor.core.config = replace(executor.core.config, wavefront_primary_scene_specialization=True,
                                   wavefront_restir_shared_primary=False,
                                   wavefront_restir_spatial_reuse=True)
    executor.core.scene_custom_attribute_layout = ol.VertexAttributeLayout(())
    for option in ('wavefront_restir_generalized_mis', 'wavefront_unified_primary_restir',
                   'wavefront_stratified_primary_restir', 'wavefront_profiling'):
        original = executor.core.config
        executor.core.config = replace(original, **{option: True})
        executor.ensure_custom_material_pipelines()
        assert variants[-1] == (True, False)
        executor.core.config = original
        executor.ensure_custom_material_pipelines()
        assert variants[-1] == (True, True)
    executor.core.config = replace(executor.core.config, wavefront_restir_specialization=False)
    executor.ensure_custom_material_pipelines()
    assert variants[-1] == (True, False)
    # A modifier or a non-builtin program can synthesize transmission.
    original = executor.core.config
    from dataclasses import fields
    values = {field.name: getattr(original, field.name) for field in fields(original)}
    values['material_modifier'] = object()
    executor.core.config = SimpleNamespace(**values)
    executor.ensure_custom_material_pipelines()
    assert variants[-1][0] is False
    executor.core.config = original
    scene.material_programs = lambda default: [object()]
    executor.ensure_custom_material_pipelines()
    assert variants[-1][0] is False


def test_scene_pipeline_uses_statistics_capable_creation():
    executor = VulkanWavefrontExecutor.__new__(VulkanWavefrontExecutor)
    executor.primary_pipeline_layout = 'primary'
    captured = []
    executor._pipeline = lambda name, layout, **kw: captured.append((name, layout, kw))
    executor._pipeline_bytes(b'primary', 'primary')
    executor._pipeline_bytes(b'shade', 'shade')
    assert captured == [
        ('wavefront_primary.comp (scene)', 'primary', {'code': b'primary'}),
        ('wavefront_shade_candidate.glsl (scene)', 'shade', {'code': b'shade'}),
    ]


@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('opaque,production', [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize('camera_policy', [False, True])
def test_surface_only_primary_compiles_with_shared_reservoirs(native, opaque, production, camera_policy):
    from ordinarylight.shaders.compiler import compile_wavefront_material_shader, find_glsl_compiler
    if not find_glsl_compiler():
        pytest.skip('GLSL compiler unavailable')
    result = compile_wavefront_material_shader(
        'wavefront_primary.comp', [ol.builtin_material],
        attribute_layout=ol.VertexAttributeLayout(()), attribute_binding=24,
        shared_primary_reservoirs=2, denoiser_signal_capture=True,
        native_textures=native, surface_only=True,
        opaque_primary=opaque, production_restir=production,
        camera_restir_policy=camera_policy,
    )
    assert result[:4] == b'\x03\x02#\x07'
