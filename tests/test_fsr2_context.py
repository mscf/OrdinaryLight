"""FSR2 history publication without a GPU or optional native bridge."""

import math
from types import SimpleNamespace
import pytest
from ordinarylight.runtime import _fsr2


@pytest.fixture
def context(monkeypatch):
    calls = []
    lib = SimpleNamespace(
        ol_fsr2_create=lambda *args: 1,
        ol_fsr2_destroy=lambda *args: None,
        ol_fsr2_dispatch=lambda *args: calls.append(args) or 0,
    )
    monkeypatch.setattr(_fsr2, "library", lambda: lib)
    monkeypatch.setattr(_fsr2, "handle", lambda value: value)
    stage = _fsr2.Fsr2Context(1, 2, (32, 24), (64, 48))
    yield stage, calls, lib
    stage.close()


def record(stage, **kwargs):
    return stage.record(
        3,
        range(5),
        range(5),
        jitter=(0.25, -0.25),
        fov_y=math.pi / 3,
        dt_ms=16,
        **kwargs,
    )


def test_history_published_only_after_submission(context):
    stage, calls, _ = context
    token = record(stage)
    assert calls[-1][-1] == 1
    assert not stage.history_valid
    stage.submitted(token)
    assert stage.history_valid
    with pytest.raises(RuntimeError, match="stale"):
        stage.submitted(token)
    stage.submitted(record(stage))
    assert calls[-1][-1] == 0
    stage.submitted(record(stage, reset=True))
    assert calls[-1][-1] == 1


def test_abandoned_recording_resets_and_rejects_old_token(context):
    stage, calls, _ = context
    stage.submitted(record(stage))
    abandoned = record(stage)
    replacement = record(stage)
    assert calls[-1][-1] == 1
    with pytest.raises(RuntimeError, match="stale"):
        stage.submitted(abandoned)
    stage.submitted(replacement)


def test_dispatch_error_invalidates_history(context):
    stage, calls, lib = context
    stage.submitted(record(stage))
    dispatch = lib.ol_fsr2_dispatch
    lib.ol_fsr2_dispatch = lambda *args: 7
    with pytest.raises(RuntimeError, match="dispatch failed: 7"):
        record(stage)
    assert not stage.history_valid
    lib.ol_fsr2_dispatch = dispatch
    stage.submitted(record(stage))
    assert calls[-1][-1] == 1
    stage.close()
    stage.close()
    with pytest.raises(RuntimeError, match="closed"):
        record(stage)


@pytest.mark.parametrize(
    "extent,output", [((0, 2), (4, 4)), ((8, 8), (4, 4)), ((2,), (4, 4))]
)
def test_invalid_extents_rejected_before_loading_bridge(extent, output):
    with pytest.raises(ValueError):
        _fsr2.Fsr2Context(None, None, extent, output)
