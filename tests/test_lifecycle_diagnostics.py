"""Profiling must preserve the wrapped compiler/driver call's behavior."""

import json
import logging

import pytest

from ordinarylight.runtime.lifecycle import timed_call


def test_timed_call_preserves_arguments_and_result(caplog):
    result = object()

    def callback(device, *, capture_output):
        assert device == "device"
        assert capture_output is True
        return result

    with caplog.at_level(logging.INFO, logger="ordinarylight.lifecycle"):
        actual = timed_call(
            "shader_compile", callback, "device", capture_output=True,
            lifecycle_details={"compiler": "test-compiler"},
        )
    assert actual is result
    record = json.loads(caplog.records[-1].message)
    assert record["event"] == "shader_compile"
    assert record["compiler"] == "test-compiler"
    assert record["total_ms"] >= 0


def test_timed_call_logs_failed_driver_call_without_swallowing_error(caplog):
    error = RuntimeError("driver failure")

    def fail():
        raise error

    with caplog.at_level(logging.INFO, logger="ordinarylight.lifecycle"):
        with pytest.raises(RuntimeError) as raised:
            timed_call("compute_pipeline_create", fail)
    assert raised.value is error
    assert json.loads(caplog.records[-1].message)["event"] == "compute_pipeline_create"
