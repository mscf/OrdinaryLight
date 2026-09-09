"""Fused-primary binary offsets are stable for independent preparation."""

from dataclasses import fields, replace
import struct
import pytest
from ordinarylight.wavefront import PrimarySettings


def settings():
    values = {
        field.name: 0
        for field in fields(PrimarySettings)
        if field.name != "effect_ranges"
    }
    values.update(
        max_bounces=8,
        area_light_weight=1.5,
        restir_enabled=1,
        restir_history_valid=1,
        restir_history_limit=12,
        capture_stride=3,
        roulette_min_survival=0.125,
        secondary_nee_probability=0.5,
        restir_generalized_balance_cap=4.5,
    )
    return PrimarySettings(**values, effect_ranges=(1, 2, 3, 4, 5, 6, 7, 8))


def test_fused_primary_abi_offsets():
    tile = struct.pack("8I", 640, 480, 2, 3, 16, 8, 4, 1)
    packed = settings().pack(tile)
    assert len(packed) == 176 and packed[:32] == tile
    assert struct.unpack_from("I", packed, 32) == (8,)
    assert struct.unpack_from("f", packed, 52) == (1.5,)
    assert struct.unpack_from("3I", packed, 84) == (1, 1, 12)
    assert struct.unpack_from("f", packed, 120) == (4.5,)
    assert struct.unpack_from("I", packed, 140) == (3,)
    assert struct.unpack_from("8I", packed, 144) == tuple(range(1, 9))


def test_primary_settings_reject_unrepresentable_values():
    with pytest.raises(ValueError, match="eight"):
        settings().pack(bytes(28))
    with pytest.raises(ValueError, match="uint32"):
        replace(settings(), restir_history_limit=-1).pack(bytes(32))
    with pytest.raises(ValueError, match="finite"):
        replace(settings(), area_light_weight=float("nan")).pack(bytes(32))
    with pytest.raises(ValueError, match="four"):
        replace(settings(), effect_ranges=(1, 2)).pack(bytes(32))
