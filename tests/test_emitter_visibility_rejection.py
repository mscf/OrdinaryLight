"""Exact-zero rejection preserves sampling and retains every nonzero query."""
from types import SimpleNamespace
import numpy as np
import pytest
from ordinarylight.shaders import lighting_programs as lighting


def test_lighting_artifact_matches_typed_source():
    from pathlib import Path
    import runpy
    import sys
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'scripts'))
    try:
        generator = runpy.run_path(str(root / 'scripts/generate_lighting_shaders.py'))
        assert (root / 'ordinarylight/shaders/transport_v1/lighting.glsl').read_text() == generator['generated_source']()
    finally:
        sys.path.pop(0)


@pytest.mark.parametrize('contribution,query', [
    ((0., 0., 0.), False), ((-0., 0., -0.), False),
    ((1e-30, 0., 0.), True), ((0., 2., 0.), True),
    ((float('nan'), 0., 0.), True), ((float('inf'), 0., 0.), True),
])
def test_native_emitter_zero_rejection_preserves_sampling(monkeypatch, contribution, query):
    events = []
    draws = iter((.2, .3, .4))
    def random(rng):
        events.append('random')
        rng[0] += 1
        return next(draws)
    def evaluate(*args):
        events.append('evaluate')
        assert args[0] == (7, (.3, .4), 0.)
        return np.array(contribution)
    def visibility(*args):
        events.append('visibility')
        return .5
    monkeypatch.setattr(lighting, 'osh', SimpleNamespace(
        specialization=lambda name: True, u32=int, f32=float,
        vec2=lambda *v: v, vec3=lambda v: np.full(3, v), all_value=np.all))
    for name, value in dict(nativeAreaLightCount=lambda _: 1,
        OL_TRANSPORT_AREA_LIGHT_COUNT=1, randomFloat=random,
        AreaLightCandidate=lambda *v: v, nativeSelectEmitter=lambda v: 7,
        evaluateAreaLightCandidateTechnique=evaluate,
        areaLightCandidateVisibility=visibility).items():
        monkeypatch.setattr(lighting, name, value, raising=False)
    rng = [0]
    result = lighting.sampleAreaLightTechnique.function(None, None, None, None, rng, 0, 1, 1.)
    assert rng == [3]
    assert events == ['random', 'random', 'random', 'evaluate'] + (['visibility'] if query else [])
    np.testing.assert_equal(result, np.array(contribution) * .5)
