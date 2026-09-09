"""GI showcase settings remain usable when switching to raster targets."""

import ordinarylight as ol
from ordinarylight.integrations.raster_workbench import _raster_config
from ordinarylight.integrations.workbench import discover_showcases
from ordinarylight.integrations.qt_workbench import _default_showcase_paths


def test_glass_motion_showcase_raster_configuration():
    showcase = discover_showcases(_default_showcase_paths())["glass-detail-motion"]
    settings = dict(showcase.renderer)
    original = settings.copy()
    config = _raster_config(settings, shadows=False, shadow_map_size=256)
    assert settings == original
    assert config.shadows is False
    assert config.shadow_map_size == 256
    assert not hasattr(config, "denoiser_enabled")


def test_raster_settings_preserve_supported_material_and_display_options():
    program = ol.builtin_material
    settings = dict(
        denoiser_enabled=True,
        max_bounces=8,
        wavefront_restir_di=True,
        required_target="wavefront-gi",
        material_program=program,
        ambient_light="0.2",
        optical_quality="screen-space",
        tone_mapping="aces",
    )
    config = _raster_config(settings, shadows=True, shadow_map_size=512)
    assert config.material_program is program
    assert config.ambient_light == 0.2
    assert config.optical_quality == "screen-space"
    assert config.tone_mapping == "aces"
