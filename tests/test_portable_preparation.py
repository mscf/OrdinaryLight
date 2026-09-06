"""Cross-project integration gate for the portable RT preparation milestone."""

import base64
import json
from pathlib import Path

import pytest

pytest.importorskip("ordinaryscience.vector_volume")
pytest.importorskip("ordinarylattice.portable")
pytest.importorskip("latticemodel")

from ordinarylattice.portable import export_batched
from ordinaryscience.rt_density import load_rt_program
from ordinaryscience.vector_volume import VectorHistogramPreparation, reprepare_volume


def preparation(**updates):
    options = dict(
        x="drift.v",
        y="response.boundary",
        x_range=(0.1, 2),
        y_range=(0.1, 1.2),
        resolution=2,
        bins=8,
        values=dict(trial_count=8, dt=0.01, simulation_time=1.5, seed=7),
    )
    options.update(updates)
    return VectorHistogramPreparation(
        load_rt_program(
            Path(__file__).parents[1] / "examples/portable_volume/dmc.json"
        ),
        **options,
    )


def test_entire_preparation_and_export_need_no_device(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Preparation tried to create a GPU executor")

    import ordinarylight as ol

    monkeypatch.setattr(ol, "WebGpuComputeSequence", forbidden)
    monkeypatch.setattr(ol, "VulkanComputeSequence", forbidden)
    import wgpu

    monkeypatch.setattr(type(wgpu.gpu), "request_adapter_sync", forbidden)
    monkeypatch.setattr(type(wgpu.gpu), "request_adapter_async", forbidden)
    prepared = preparation()
    package = prepared.export()
    json.dumps(package.manifest, allow_nan=False)
    assert package.manifest["render"]["dimensions"] == [2, 2, 8]
    classes = {f["name"]: f["update"] for f in package.manifest["parameters"]["fields"]}
    assert (
        classes["trial_count"]
        == classes["dt"]
        == classes["simulation_time"]
        == "structural"
    )
    assert classes["seed"] == classes["noise.sigma"] == "live"
    exported = json.loads(
        json.dumps(export_batched(prepared.entry, prepared.parameter_sets))
    )
    assert exported["initial_parameters"]["encoding"] == "base64"
    assert (
        base64.b64decode(exported["initial_parameters"]["data"])
        == prepared.entry.pack_inputs(prepared.parameter_sets).astype("<u4").tobytes()
    )
    assert (
        exported["allocations"]["parameters"]["byte_length"]
        == 4 * len(prepared.entry.graph.inputs) * 4
    )
    for i, kernel in enumerate(exported["kernels"]):
        assert kernel["after"] == ([f"kernel_{i - 1}"] if i else [])
        assert set(kernel["resources"].values()) <= set(exported["allocations"])


def test_selected_vector_output_and_structural_replacement():
    package = preparation().export()
    snapshot = dict(
        schema="ordinarylight/view-state-v1",
        package_id=package.manifest["id"],
        parameters=package.manifest["state"]["parameters"],
        presentation=dict(package.manifest["state"]["presentation"], yaw=1.2),
    )
    replacement = reprepare_volume(
        package,
        snapshot,
        dict(parameters={"trial_count": 12}, preparation={"bins": 10}),
    )
    assert replacement.manifest["id"] != package.manifest["id"]
    assert replacement.manifest["science"]["samples_per_cell"] == 12
    assert replacement.manifest["render"]["dimensions"] == [2, 2, 10]
    assert replacement.manifest["state"]["presentation"]["yaw"] == 1.2
    selected = preparation(output="ndt_per_trial").export()
    assert selected.manifest["science"]["output"] == "ndt_per_trial"
    assert set(selected.manifest["outputs"]) == {"ndt_per_trial", "density", "excluded"}
    with pytest.raises(ValueError, match="floating vector"):
        preparation(output="total")
    with pytest.raises(ValueError, match="Structural parameters"):
        preparation(x="trial_count", x_range=(8, 12))
