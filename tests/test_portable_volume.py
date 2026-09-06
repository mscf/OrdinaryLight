"""Device-free portable contract tests; browser/native GPU probe is a separate gate."""

import copy
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from ordinarylight.compute import ComputeBuffer, ComputeStep
from ordinarylight.portable import PortablePackage, make_package, validate_manifest
from ordinaryshade.compiler import CompiledShader
from ordinaryshade.reflection import ResourceReflection, ShaderReflection


@pytest.fixture
def package():
    reflection = ShaderReflection(
        "compute",
        "main",
        (1, 1, 1),
        (ResourceReflection("density", "storage_buffer", "f32", "write", 0, 0),),
    )
    shader = CompiledShader(
        "wgsl",
        """
@group(0) @binding(0) var<storage, read_write> density: array<f32>;
@compute @workgroup_size(1) fn main() { density[0] = 1.; }
""",
        None,
        reflection,
    )
    return make_package(
        [ComputeStep(shader, (1, 1, 1))],
        {
            "density": ComputeBuffer(shape=(2, 2, 2)),
            "transfer": ComputeBuffer(np.ones((2, 4), dtype=np.float32)),
            "parameters": ComputeBuffer(np.array([[1, 2, 4]] * 4, dtype=np.float32)),
        },
        outputs={"density": "density"},
        parameters=dict(
            resource="parameters",
            rows=4,
            row_stride=12,
            byte_order="little",
            fields=[
                dict(
                    name=n,
                    offset=i * 4,
                    type="f32",
                    update="structural" if n == "n" else "live",
                    range=None,
                )
                for i, n in enumerate(("x", "y", "n"))
            ],
        ),
        science=dict(
            kind="vector-histogram/v1",
            layout="zyx",
            x=dict(name="x", samples=[1, 2]),
            y=dict(name="y", samples=[2, 3]),
            edges=[0, 0.5, 1],
        ),
        state=dict(
            parameters=dict(x=1, y=2, n=4),
            presentation=dict(
                yaw=0.65,
                pitch=0.45,
                radius=4,
                opacity=1,
                maximum=1,
                mode="volume",
                slice_z=0.5,
            ),
        ),
        density="density",
        transfer="transfer",
        dimensions=(2, 2, 2),
    )


def test_export_roundtrip_preserves_bytes_and_resource_types(
    package, tmp_path, monkeypatch
):
    package.write(tmp_path)
    restored = PortablePackage.read(tmp_path)
    assert restored.manifest == json.loads(json.dumps(package.manifest))
    assert restored.payload == package.payload
    captured = {}

    def executor(steps, resources, **options):
        captured.update(resources)

    monkeypatch.setattr("ordinarylight.compute.WebGpuComputeSequence", executor)
    restored.native()
    assert captured["density"].shape == (2, 2, 2)
    assert np.dtype(captured["density"].dtype) == np.dtype("<f4")
    assert captured["transfer"].payload() == np.ones((2, 4), dtype="<f4").tobytes()


def test_corrupt_payload_rejected(package):
    with pytest.raises(ValueError, match="integrity"):
        PortablePackage(package.manifest, package.payload[:-1])


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m.update(schema="future"),
        lambda m: m["passes"][0].update(after=["later"]),
        lambda m: m["passes"][0]["bindings"][0].update(access="read"),
        lambda m: m["passes"][0].update(workgroups=[True, 1, 1]),
        lambda m: m["render"].update(dimensions=[2, 2, 3]),
        lambda m: m["render"]["uniform_layout"].update(camera=32),
        lambda m: m["parameters"]["fields"][0].update(offset=4),
        lambda m: m["science"]["x"].update(samples=[1]),
        lambda m: m["state"]["parameters"].update(unknown=1),
        lambda m: m["resources"][1]["initial"].update(offset=2),
        lambda m: m["shaders"]["kernel_0"].update(source="corrupted"),
    ],
)
def test_incompatible_contract_rejected_before_execution(package, change):
    manifest = copy.deepcopy(package.manifest)
    change(manifest)
    with pytest.raises(ValueError):
        PortablePackage(manifest, package.payload)


def test_missing_contract_fields_raise_clear_value_error():
    with pytest.raises(ValueError, match="Malformed portable package"):
        validate_manifest(
            dict(schema="ordinarylight/portable-volume-v1", byte_order="little")
        )


def test_browser_contract_without_gpu(package, tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is needed for the browser contract test")
    package.write(tmp_path)
    # Load as an ES module independently of a host application's package.json.
    runtime = Path(__file__).parents[1] / "ordinarylight/portable/web/runtime.js"
    shutil.copy(runtime, tmp_path / "runtime.mjs")
    result = subprocess.run(
        [node, str(Path(__file__).with_name("portable_contract.mjs")), str(tmp_path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
