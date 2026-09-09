from types import SimpleNamespace
from dataclasses import replace
import pytest
import ordinarylight as ol
from ordinarylight.integrations.raster_workbench import _gi_config
from ordinarylight.targets.vulkan.fsr2 import library


def test_fsr2_configuration_requires_compatible_guides():
    config = _gi_config(
        SimpleNamespace(id="glass-detail-camera", renderer={}),
        render_scale=0.5,
        upscale_filter="fsr2",
    )
    assert config.wavefront_upscale_filter == "fsr2"
    assert config.denoiser_enabled
    for key in (
        "denoiser_planar_mirror_guides",
        "object_effects",
        "wavefront_temporal_reconstruction",
        "wavefront_diffuse_filter",
    ):
        with pytest.raises(ValueError, match="fsr2 requires"):
            replace(config, **{key: True})


def test_missing_optional_bridge_has_actionable_error(monkeypatch, tmp_path):
    monkeypatch.setenv("ORDINARYLIGHT_FSR2_LIBRARY", str(tmp_path / "missing.so"))
    with pytest.raises(RuntimeError, match="scripts/build_fsr2.py"):
        library()
