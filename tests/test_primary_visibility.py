"""Capture/replay ABI and compilation; GPU integration lives in the profiler."""
import numpy as np
import pytest
import ordinarylight as ol
from ordinarylight.wavefront import PRIMARY_VISIBILITY_DTYPE, primary_bindings
from ordinarylight.shaders.compiler import compile_wavefront_material_shader, find_glsl_compiler


def test_visibility_record_preserves_native_payload():
    assert PRIMARY_VISIBILITY_DTYPE.itemsize == 112
    for i, name in enumerate(('position_distance', 'geometric_normal', 'shading_normal',
                              'identity', 'address', 'texcoord', 'previous_position')):
        assert PRIMARY_VISIBILITY_DTYPE.fields[name][1] == i * 16
    assert PRIMARY_VISIBILITY_DTYPE.fields['identity'][0].base == np.dtype('<u4')
    assert PRIMARY_VISIBILITY_DTYPE.fields['address'][0].base == np.dtype('<u4')
    assert not any(b.binding == 33 for b in primary_bindings())
    assert primary_bindings(primary_visibility=True)[-1].name == 'primary_visibility'


@pytest.mark.parametrize('mode', ('capture', 'replay'))
@pytest.mark.parametrize('surface_only', (False, True))
@pytest.mark.parametrize('hit_format', ('full', 'identity'))
def test_visibility_variants_compile(mode, surface_only, hit_format):
    if not find_glsl_compiler():
        pytest.skip('GLSL compiler unavailable')
    spirv = compile_wavefront_material_shader(
        'wavefront_primary.comp', [ol.builtin_material],
        attribute_layout=ol.VertexAttributeLayout(()), attribute_binding=24,
        primary_visibility=mode, primary_hits=True, primary_hit_format=hit_format,
        surface_only=surface_only, opaque_primary=surface_only,
        shared_primary_reservoirs=2, denoiser_signal_capture=True,
        camera_restir_policy=True,
    )
    assert spirv[:4] == b'\x03\x02#\x07'


@pytest.mark.parametrize('kwargs', (
    {'primary_visibility': 'invalid'},
    {'primary_visibility': 'capture', 'inline_continuation': True},
))
def test_invalid_visibility_policy(kwargs):
    if not find_glsl_compiler():
        pytest.skip('GLSL compiler unavailable')
    with pytest.raises(ValueError, match='primary'):
        compile_wavefront_material_shader('wavefront_primary.comp', [ol.builtin_material],
            attribute_layout=ol.VertexAttributeLayout(()), attribute_binding=24, **kwargs)


def test_hit_view_cache_rejects_reused_vulkan_handles():
    from types import SimpleNamespace as NS
    from ordinarylight.targets.vulkan.gi_images import native_gi_buffers
    class Core:
        pass
    core = Core()
    core.device = object()
    core.runtime = NS(require_open=lambda: None)
    core.config = NS(wavefront_primary_hit_format='full')
    core.swapchain_generation = 1
    first = NS(buffer=42, size=96)
    core.window_frames = [{'wavefront_primary_hit_buffer': first}]
    old = native_gi_buffers(core, 0)
    assert native_gi_buffers(core, 0) is old
    # Same numeric Vulkan handle, different allocation object.
    core.window_frames[0]['wavefront_primary_hit_buffer'] = NS(buffer=42, size=192)
    new = native_gi_buffers(core, 0)
    assert new is not old
    assert new['primary_hits'].byte_size == 192
    new['primary_hits'].require_open()
    with pytest.raises(RuntimeError, match='retired'):
        old['primary_hits'].require_open()
    core.swapchain_generation += 1
    latest = native_gi_buffers(core, 0)
    assert latest is not new
    latest['primary_hits'].require_open()
    with pytest.raises(RuntimeError, match='retired'):
        new['primary_hits'].require_open()


@pytest.mark.parametrize('enabled', (False, True))
def test_selected_diffuse_compiler_variant(enabled):
    if not find_glsl_compiler():
        pytest.skip('GLSL compiler unavailable')
    spirv = compile_wavefront_material_shader(
        'wavefront_primary.comp', [ol.builtin_material],
        attribute_layout=ol.VertexAttributeLayout(()), attribute_binding=24,
        primary_visibility='replay', surface_only=True,
        denoiser_signal_capture=True, primary_lobe_selection=enabled,
    )
    assert spirv[:4] == b'\x03\x02#\x07'


@pytest.mark.parametrize('change', (
    {'primary_visibility':'capture'}, {'surface_only':False},
    {'denoiser_signal_capture':False}, {'inline_continuation':True},
))
def test_selected_diffuse_rejects_unsupported_compiler_modes(change):
    options=dict(primary_visibility='replay', surface_only=True,
                 denoiser_signal_capture=True,primary_lobe_selection=True)
    options.update(change)
    with pytest.raises(ValueError,match='Selected diffuse'):
        compile_wavefront_material_shader(
            'wavefront_primary.comp',[ol.builtin_material],
            attribute_layout=ol.VertexAttributeLayout(()),attribute_binding=24,**options)


@pytest.mark.parametrize('shape', ((0,64),(3,21),(16,8),[8,8],(True,64)))
def test_invalid_primary_workgroup(shape):
    with pytest.raises(ValueError,match='primary_workgroup'):
        compile_wavefront_material_shader(
            'wavefront_primary.comp',[ol.builtin_material],
            attribute_layout=ol.VertexAttributeLayout(()),attribute_binding=24,
            primary_workgroup=shape)


def test_primary_workgroup_rejects_inline_continuation():
    with pytest.raises(ValueError,match='primary_workgroup'):
        compile_wavefront_material_shader(
            'wavefront_primary.comp',[ol.builtin_material],
            attribute_layout=ol.VertexAttributeLayout(()),attribute_binding=24,
            primary_workgroup=(16,4),inline_continuation=True)


@pytest.mark.parametrize('format', ('bad','distance','planes','distance_planes'))
def test_visibility_format_rejects_invalid_mode(format):
    with pytest.raises(ValueError):
        compile_wavefront_material_shader(
            'wavefront_primary.comp',[ol.builtin_material],
            attribute_layout=ol.VertexAttributeLayout(()),attribute_binding=24,
            primary_visibility_format=format)


def test_distance_visibility_dtype():
    from ordinarylight.wavefront import primary_visibility_dtype
    dtype=primary_visibility_dtype("distance")
    assert dtype.itemsize==100
    assert dtype.fields["distance"][1]==0
    assert dtype.fields["identity_x"][1]==36
    assert dtype.fields["address_w"][1]==64
    assert dtype.fields["previous_position_w"][1]==96
    assert primary_visibility_dtype().itemsize==112
    with pytest.raises(ValueError):
        primary_visibility_dtype("invalid")


@pytest.mark.parametrize("pixels", (1,3,4,5,221))
def test_visibility_plane_sample_alignment(pixels):
    from ordinarylight.wavefront import primary_visibility_byte_size
    assert primary_visibility_byte_size(pixels,1,samples=3,format="distance_planes")==3*(6*pixels+(pixels+3)//4)*16
    for format,stride in (("full",112),("planes",112),("distance",100)):
        assert primary_visibility_byte_size(pixels,1,samples=3,format=format)==3*pixels*stride


@pytest.mark.parametrize("kwargs", ({"width":0},{"height":0},{"samples":0},{"format":"invalid"}))
def test_visibility_size_rejects_invalid_inputs(kwargs):
    from ordinarylight.wavefront import primary_visibility_byte_size
    options=dict(width=1,height=1);options.update(kwargs)
    with pytest.raises(ValueError):primary_visibility_byte_size(**options)


@pytest.mark.parametrize('stage,shape', [('classify',(32,2)),('resume',(64,1))])
def test_compact_continuation_variants_compile(stage,shape):
    if not find_glsl_compiler():
        pytest.skip('GLSL compiler unavailable')
    result=compile_wavefront_material_shader('wavefront_primary.comp',[ol.builtin_material],
        attribute_layout=ol.VertexAttributeLayout(()),attribute_binding=24,
        primary_visibility='replay',primary_lobe_selection=True,surface_only=True,
        denoiser_signal_capture=True,primary_continuation=stage,primary_workgroup=shape)
    assert result[:4] == b'\x03\x02#\x07'


@pytest.mark.parametrize('options', [
    {'primary_continuation':'unknown'},
    {'primary_continuation':'classify'},
    {'primary_continuation':'resume','primary_lobe_selection':True},
])
def test_compact_continuation_rejects_incompatible_variants(options):
    if not find_glsl_compiler():
        pytest.skip('GLSL compiler unavailable')
    with pytest.raises(ValueError):
        compile_wavefront_material_shader('wavefront_primary.comp',[ol.builtin_material],
            attribute_layout=ol.VertexAttributeLayout(()),attribute_binding=24,**options)
