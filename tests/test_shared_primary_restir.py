"""Configuration and allocation contracts for independent paths and DI streams."""

from types import SimpleNamespace

import pytest
from ordinarylight.targets.vulkan.api import RendererConfig
from ordinarylight.targets.vulkan.core import _restir_reservoir_storage_bytes
from ordinarylight.integrations.raster_workbench import _gi_config


def test_each_path_has_independent_reservoir_storage():
    assert _restir_reservoir_storage_bytes(10, 20, 4, path_samples=2) == 19200
    assert _restir_reservoir_storage_bytes(
        10, 20, 4, path_samples=2, stratified=True,
    ) == 32000


def test_viewer_shared_mode_keeps_spp_separate_and_allows_temporal_history():
    config = _gi_config(
        SimpleNamespace(renderer={}, id="optical-screen-rough-reflection"),
        shared_primary=True, path_spp=2, restir_reservoirs=4,
    )
    assert config.samples_per_pixel == 2
    assert config.wavefront_restir_reservoirs == 4
    assert config.wavefront_restir_history_limit > config.wavefront_restir_candidates
    assert config.wavefront_execution_strategy == "wavefront"


@pytest.mark.parametrize("changes", [
    {"wavefront_execution_strategy": "hybrid"},
    {"interactive_samples_per_pixel": 2},
    {"wavefront_interactive_sample_scaling": True},
    {"wavefront_restir_spatial_reuse": True},
])
def test_unsupported_shared_variants_are_rejected(changes):
    with pytest.raises(ValueError, match="shared primary"):
        RendererConfig(wavefront_restir_shared_primary=True, **changes)
