"""Invalid-lighting cost ablation preserving primary environment RNG draws."""
from pathlib import Path
from primary_hit_layout_experiment import load_module


def variant(directory):
    from ordinarylight.shaders import fused_primary_programs
    source=Path(fused_primary_programs.__file__).read_text()
    old='                contribution = sampleEnvironment(position, normal, incoming, material, rng, environment_samples)'
    new='''                # sampleEnvironment consumes two draws for its cosine direction.
                discarded_environment_draw = randomFloat(rng)
                discarded_environment_draw = randomFloat(rng)
                contribution = osh.vec3(0.0)'''
    if source.count(old)!=1:raise RuntimeError('Primary environment sampling changed')
    return load_module('ordinarylight.shaders.environment_cost_diagnostic',source.replace(old,new),directory)
