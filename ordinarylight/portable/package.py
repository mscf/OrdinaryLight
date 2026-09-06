"""Small explicit export contract; native GPU handles are never accepted."""

from dataclasses import dataclass
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
import hashlib
import json
import re
import numpy as np
from ordinaryshade.portable import export_shader, validate_shader_export

SCHEMA = "ordinarylight/portable-volume-v1"


def producers():
    result = {}
    for name in (
        "ordinarylight",
        "ordinaryshade",
        "ordinarylattice",
        "ordinaryscience",
        "latticemodel",
    ):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = "source-checkout"
    return result


def validate_manifest(m, payload=None):
    """Reject unsupported or inconsistent v1 descriptions before device creation."""
    try:
        return _validate_manifest(m, payload)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as error:
        raise ValueError(f"Malformed portable package: {error}") from error


def _validate_manifest(m, payload=None):
    if m.get("schema") != SCHEMA:
        raise ValueError("Unsupported portable package schema")
    if m.get("byte_order") != "little":
        raise ValueError("Unsupported byte order")
    if m["payload"]["file"] != "payload.bin":
        raise ValueError("Unsupported payload reference")
    if not m["resources"] or not m["passes"]:
        raise ValueError("Resources and passes are required")
    ids = set()
    for r in m["resources"]:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", r["id"]) or r["id"] in ids:
            raise ValueError("Invalid/duplicate resource identity")
        ids.add(r["id"])
        if r["kind"] != "buffer" or r["dtype"] not in ("f32", "i32", "u32"):
            raise ValueError("Unsupported resource kind/dtype")
        if not r["shape"] or any(type(n) is not int or n < 1 for n in r["shape"]):
            raise ValueError("Invalid resource shape")
        if (
            type(r["byte_length"]) is not int
            or int(np.prod(r["shape"], dtype=object)) * 4 != r["byte_length"]
        ):
            raise ValueError("Resource shape/byte length mismatch")
        if r["byte_length"] > 128 * 1024**2:
            raise ValueError("Resource exceeds v1 storage budget")
        if r["usage"] != ["storage", "copy-src", "copy-dst"]:
            raise ValueError("Unsupported resource usage")
        init = r.get("initial")
        if init is not None and (
            type(init["offset"]) is not int
            or init["offset"] < 0
            or init["offset"] % 4
            or init["length"] != r["byte_length"]
            or init["offset"] + init["length"] > m["payload"]["byte_length"]
        ):
            raise ValueError("Invalid payload range")
    if sum(r["byte_length"] for r in m["resources"]) > 512 * 1024**2:
        raise ValueError("Package exceeds v1 allocation budget")
    pass_ids = set()
    for p in m["passes"]:
        if p["id"] in pass_ids or p["kind"] != "compute":
            raise ValueError("Invalid pass identity/kind")
        if any(d not in pass_ids for d in p["after"]):
            raise ValueError("Pass dependency is not earlier in order")
        pass_ids.add(p["id"])
        shader = m["shaders"][p["shader"]]
        validate_shader_export(shader)
        if shader["reflection"]["stage"] != "compute":
            raise ValueError("Expected compute shader")
        expected = {(r["set"], r["binding"]) for r in shader["reflection"]["resources"]}
        bound = {(r["group"], r["binding"]) for r in p["bindings"]}
        if expected != bound or len(bound) != len(p["bindings"]):
            raise ValueError("Binding reflection mismatch")
        reflected = {
            (r["set"], r["binding"]): r for r in shader["reflection"]["resources"]
        }
        for binding in p["bindings"]:
            r = reflected[(binding["group"], binding["binding"])]
            if (
                binding["group"] != 0
                or r["kind"] != "storage_buffer"
                or r["access"] != binding["access"]
            ):
                raise ValueError("Unsupported binding contract")
        if any(r["resource"] not in ids for r in p["bindings"]):
            raise ValueError("Unknown bound resource")
        if len(p["workgroups"]) != 3 or any(
            type(n) is not int or n < 1 for n in p["workgroups"]
        ):
            raise ValueError("Invalid dispatch")
        workgroup = shader["reflection"]["workgroup_size"]
        if len(workgroup) != 3 or any(type(n) is not int or n < 1 for n in workgroup):
            raise ValueError("Invalid workgroup size")
    if not set(m["outputs"].values()) <= ids:
        raise ValueError("Unknown output resource")
    render = m["render"]
    if render["kind"] != "volume-buffer-v1" or any(
        render[k] not in ids for k in ("density", "transfer")
    ):
        raise ValueError("Unsupported volume description")
    if hashlib.sha256(render["source"].encode()).hexdigest() != render["sha256"]:
        raise ValueError("Volume shader digest mismatch")
    resources = {r["id"]: r for r in m["resources"]}
    density, transfer = resources[render["density"]], resources[render["transfer"]]
    dims = render["dimensions"]
    if (
        len(dims) != 3
        or any(type(n) is not int or n < 1 for n in dims)
        or int(np.prod(dims, dtype=object)) * 4 != density["byte_length"]
    ):
        raise ValueError("Volume dimensions mismatch")
    if (
        density["dtype"] != "f32"
        or transfer["dtype"] != "f32"
        or transfer["byte_length"] % 16
    ):
        raise ValueError("Invalid volume/transfer type")
    if (
        render["vertex_entry"] != "vertex_main"
        or render["fragment_entry"] != "fragment_main"
        or render["topology"] != "triangle-list"
        or render["interpolation"] != "nearest"
        or render["uniform_layout"]
        != dict(byte_length=64, viewport=0, camera=16, slice=32, dimensions=48)
    ):
        raise ValueError("Unsupported volume pipeline/layout")
    params = m["parameters"]
    names = set()
    offsets = set()
    if (
        type(params["rows"]) is not int
        or params["rows"] < 1
        or type(params["row_stride"]) is not int
        or params["row_stride"] < 4
        or params["row_stride"] % 4
        or params["byte_order"] != "little"
        or resources[params["resource"]]["byte_length"]
        != params["rows"] * params["row_stride"]
    ):
        raise ValueError("Invalid parameter buffer")
    for f in params["fields"]:
        if (
            f["name"] in names
            or f["offset"] in offsets
            or type(f["offset"]) is not int
            or f["offset"] < 0
            or f["offset"] % 4
            or f["offset"] + 4 > params["row_stride"]
            or f["type"] not in ("f32", "i32", "u32", "bool")
            or f["update"] not in ("live", "structural")
        ):
            raise ValueError("Invalid parameter field")
        names.add(f["name"])
        offsets.add(f["offset"])
    science = m["science"]
    if (
        science["kind"] != "vector-histogram/v1"
        or science["layout"] != "zyx"
        or len(science["x"]["samples"]) != dims[0]
        or len(science["y"]["samples"]) != dims[1]
        or dims[0] * dims[1] != params["rows"]
        or len(science["edges"]) != dims[2] + 1
    ):
        raise ValueError("Scientific grid mismatch")
    if (
        science["x"]["name"] not in names
        or science["y"]["name"] not in names
        or science["x"]["name"] == science["y"]["name"]
    ):
        raise ValueError("Invalid parameter axes")
    if set(m["state"]["parameters"]) != names:
        raise ValueError("State parameter set mismatch")
    for key, value in m["required_limits"].items():
        if type(value) is not int or value < 1:
            raise ValueError(f"Invalid GPU limit {key}")
    if not isinstance(m["required_features"], list) or any(
        not isinstance(f, str) for f in m["required_features"]
    ):
        raise ValueError("Invalid GPU features")
    if payload is not None and (
        len(payload) != m["payload"]["byte_length"]
        or hashlib.sha256(payload).hexdigest() != m["payload"]["sha256"]
    ):
        raise ValueError("Payload integrity failure")
    return m


@dataclass(frozen=True)
class PortablePackage:
    manifest: dict
    payload: bytes

    def __post_init__(self):
        validate_manifest(self.manifest, self.payload)

    def write(self, directory):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, allow_nan=False) + "\n"
        )
        (root / "payload.bin").write_bytes(self.payload)
        return root / "manifest.json"

    @classmethod
    def read(cls, directory):
        root = Path(directory)
        return cls(
            json.loads((root / "manifest.json").read_text()),
            (root / "payload.bin").read_bytes(),
        )

    def native(self, *, device=None):
        """Execute this exact export through the native WebGPU backend."""
        from ordinarylight.compute import (
            ComputeBuffer,
            ComputeStep,
            WebGpuComputeSequence,
        )
        from ordinaryshade.compiler import CompiledShader
        from ordinaryshade.reflection import ShaderReflection, ResourceReflection

        resources = {}
        for r in self.manifest["resources"]:
            init = r["initial"]
            resources[r["id"]] = ComputeBuffer(
                self.payload[init["offset"] : init["offset"] + init["length"]]
                if init
                else None,
                nbytes=r["byte_length"],
                shape=tuple(r["shape"]),
                dtype={"f32": "<f4", "i32": "<i4", "u32": "<u4"}[r["dtype"]],
            )
        steps = []
        for p in self.manifest["passes"]:
            s = self.manifest["shaders"][p["shader"]]
            refl = s["reflection"]
            reflection = ShaderReflection(
                refl["stage"],
                refl["entry_point"],
                tuple(refl["workgroup_size"]),
                tuple(ResourceReflection(**r) for r in refl["resources"]),
            )
            mapping = {(b["group"], b["binding"]): b["resource"] for b in p["bindings"]}
            steps.append(
                ComputeStep(
                    CompiledShader("wgsl", s["source"], None, reflection),
                    tuple(p["workgroups"]),
                    {r.name: mapping[(r.set, r.binding)] for r in reflection.resources},
                )
            )
        return WebGpuComputeSequence(
            steps, resources, **({"device": device} if device else {})
        )


def make_package(
    steps,
    resources,
    *,
    outputs,
    parameters,
    science,
    state,
    density,
    transfer,
    dimensions,
):
    blobs = bytearray()
    descriptions = []
    for name, resource in resources.items():
        dtype = np.dtype(resource.dtype)
        types = {"f": "f32", "i": "i32", "u": "u32"}
        if dtype.itemsize != 4 or dtype.kind not in types:
            raise ValueError("Portable buffers require 32-bit scalars")
        shape = list(resource.shape or np.asarray(resource.data).shape)
        if not shape:
            shape = [resource.byte_size // 4]
        initial = None
        if resource.data is not None:
            raw = (
                np.frombuffer(resource.payload(), dtype=dtype)
                .astype(dtype.newbyteorder("<"), copy=False)
                .tobytes()
            )
            initial = dict(offset=len(blobs), length=len(raw))
            blobs.extend(raw)
        descriptions.append(
            dict(
                id=name,
                kind="buffer",
                shape=shape,
                dtype=types[dtype.kind],
                byte_length=resource.byte_size,
                usage=["storage", "copy-src", "copy-dst"],
                initial=initial,
            )
        )
    shaders = {}
    passes = []
    for i, step in enumerate(steps):
        name = f"kernel_{i}"
        shaders[name] = export_shader(step.program)
        bindings = []
        for r in step.program.reflection.resources:
            if r.kind != "storage_buffer" or r.set != 0:
                raise ValueError("v1 compute supports set-0 storage buffers only")
            bindings.append(
                dict(
                    group=r.set,
                    binding=r.binding,
                    resource=(step.resources or {}).get(r.name, r.name),
                    access=r.access,
                )
            )
        passes.append(
            dict(
                id=name,
                kind="compute",
                shader=name,
                bindings=bindings,
                workgroups=list(step.workgroups),
                after=[passes[-1]["id"]] if passes else [],
            )
        )
    source = Path(__file__).with_name("volume.wgsl").read_text()
    payload = bytes(blobs)
    limits = dict(
        maxStorageBufferBindingSize=max(r["byte_length"] for r in descriptions),
        maxBufferSize=max(r["byte_length"] for r in descriptions),
        maxStorageBuffersPerShaderStage=max(2, max(len(p["bindings"]) for p in passes)),
        maxComputeWorkgroupsPerDimension=max(
            n for p in passes for n in p["workgroups"]
        ),
    )
    m = dict(
        schema=SCHEMA,
        byte_order="little",
        producers=producers(),
        required_features=[],
        required_limits=limits,
        resources=descriptions,
        shaders=shaders,
        passes=passes,
        outputs=outputs,
        parameters=parameters,
        science=science,
        state=state,
        payload=dict(
            file="payload.bin",
            byte_length=len(payload),
            sha256=hashlib.sha256(payload).hexdigest(),
        ),
        render=dict(
            kind="volume-buffer-v1",
            density=density,
            transfer=transfer,
            dimensions=list(dimensions),
            source=source,
            sha256=hashlib.sha256(source.encode()).hexdigest(),
            vertex_entry="vertex_main",
            fragment_entry="fragment_main",
            topology="triangle-list",
            interpolation="nearest",
            uniform_layout=dict(
                byte_length=64, viewport=0, camera=16, slice=32, dimensions=48
            ),
        ),
    )
    m["id"] = hashlib.sha256(
        json.dumps(m, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return PortablePackage(m, payload)
