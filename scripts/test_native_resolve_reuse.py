"""Native resolve with real indirect buffers, samples, history and resize."""
from dataclasses import replace
from test_primary_visibility_gpu import test_triangle_samples_history_and_resize as _check_transport


def test_native_indirect_reuse_keeps_reservoir_output(monkeypatch):
    from ordinarylight.integrations import raster_workbench
    original=raster_workbench._gi_config
    def config(*args,**kwargs):
        return replace(original(*args,**kwargs),wavefront_indirect_reuse_candidates=True,
            wavefront_indirect_reuse_storage=True)
    monkeypatch.setattr(raster_workbench,'_gi_config',config)
    _check_transport('single',(8,8),'full')
