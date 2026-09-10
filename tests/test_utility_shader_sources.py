"""Utility shader source and Vulkan ABI remain owned by OrdinaryShade."""
from pathlib import Path
import runpy
import pytest
import ordinaryshade as osh
from ordinarylight.shaders.utility_programs import PROGRAMS

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name,offsets,bindings', [
    ('external_hdr_tone_map', (0,), 2), ('fsr2_prepare', (0, 8), 6),
    ('primary_metadata', (0, 4), 4), ('accumulation_resolve', (0, 4, 8), 2),
])
def test_generated_utility_matches_source_and_abi(name, offsets, bindings):
    generator = runpy.run_path(str(ROOT / 'scripts/generate_utility_shaders.py'))
    result = osh.compile(PROGRAMS[name])
    artifact = ROOT / 'ordinarylight/shaders' / f'{name}.comp'
    assert artifact.read_text() == generator['generated_source'](PROGRAMS[name])
    assert len([r for r in result.reflection.resources if r.binding >= 0]) == bindings
    from ordinarylight.shaders.abi import reflect_spirv_struct
    block = 'pc_Block' if name == 'accumulation_resolve' else 'push_Block'
    actual, _ = reflect_spirv_struct(artifact.with_suffix('.comp.spv'), block)
    assert actual == offsets
