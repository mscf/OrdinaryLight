"""Keep handwritten shader debt explicit and prevent accidental growth."""
from pathlib import Path
import runpy


def test_shader_authorship_inventory():
    root = Path(__file__).resolve().parents[1]
    checker = runpy.run_path(str(root / 'scripts/check_shader_authorship.py'))
    assert checker['check']() == []


def test_sampling_library_matches_typed_source():
    from pathlib import Path
    import runpy
    root = Path(__file__).resolve().parents[1]
    generator = runpy.run_path(str(root / "scripts/generate_sampling_shaders.py"))
    assert (root / "ordinarylight/shaders/transport_v1/sampling.glsl").read_text() == generator["generated_source"]()


def test_scalar_and_struct_shader_bodies_cannot_hide_in_python(tmp_path):
    root = Path(__file__).resolve().parents[1]
    checker = runpy.run_path(str(root / 'scripts/check_shader_authorship.py'))
    package = tmp_path / 'ordinarylight'
    package.mkdir()
    for index, body in enumerate((
        'uint dispatch(uint index) { return index; }',
        'bool valid(float value) { return value > 0.0; }',
        'MaterialEvaluation evaluate(MaterialData value) { return convert(value); }',
        'fn main() { let value = 1u; }',
    )):
        (package / f'bad{index}.py').write_text('SOURCE = ' + repr(body))
    (package / 'abi.py').write_text('LAYOUT = ' + repr('layout(local_size_x=64) in;\nstruct Value { uint index; };'))
    discovered = checker['shader_sources'](tmp_path)
    assert discovered == {f'ordinarylight/bad{index}.py' for index in range(4)}
