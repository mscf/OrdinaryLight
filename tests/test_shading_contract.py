"""Split shading descriptor and push-constant compatibility."""

from dataclasses import replace
import struct
import pytest
from ordinarylight.wavefront.shading import (
    SHADE_BUFFER_BINDINGS,
    shade_buffer_bindings,
    WavefrontShadeSettings,
)


def test_settings_preserve_shader_push_abi():
    settings = WavefrontShadeSettings(
        max_bounces=7,
        light_count=2,
        area_light_count=3,
        primary_area_samples=4,
        secondary_area_samples=5,
        area_light_weight=1.5,
        environment_samples=6,
        roulette_start=2,
        roulette_min_survival=0.25,
        fused_intersection=True,
        subgroup_enqueue=True,
        secondary_nee_probability=0.5,
        unified_secondary_nee=True,
        capture_secondary=True,
    )
    assert settings.pack() == struct.pack(
        "5IfIIf2If2I", 7, 2, 3, 4, 5, 1.5, 6, 2, 0.25, 1, 1, 0.5, 1, 1
    )
    for key, value in [
        ("max_bounces", 0),
        ("light_count", -1),
        ("area_light_weight", float("nan")),
        ("secondary_nee_probability", 1.1),
        ("roulette_min_survival", -0.1),
    ]:
        with pytest.raises(ValueError):
            replace(settings, **{key: value}).pack()


def test_optional_descriptors_and_invalid_contracts():
    buffers = {name: object() for name in SHADE_BUFFER_BINDINGS}
    buffers["custom_attribute"] = None
    buffers.pop("work_counters")
    bound = shade_buffer_bindings(buffers)
    assert 14 not in bound and 16 not in bound
    assert bound[6] is buffers["next_rays"]
    custom = object()
    buffers["custom_attribute"] = custom
    assert shade_buffer_bindings(buffers)[16] is custom
    with pytest.raises(TypeError):
        bound[0] = custom
    for bad in ({**buffers, "rays": None}, {**buffers, "unexpected": object()}):
        with pytest.raises(ValueError):
            shade_buffer_bindings(bad)
