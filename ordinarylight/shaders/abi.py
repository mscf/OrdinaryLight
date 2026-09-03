"""Canonical CPU and shader storage-buffer ABI contracts."""

from __future__ import annotations

from dataclasses import dataclass
import re
import struct
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class AbiField:
    name: str
    kind: str
    offset: int
    lanes: int = 4
    count: int = 1

    @property
    def shader_type(self):
        if self.lanes == 16:
            return "mat4"
        prefix = "u" if self.kind == "uint" else ""
        return f"{prefix}vec{self.lanes}"

    @property
    def numpy_format(self):
        scalar = np.uint32 if self.kind == "uint" else np.float32
        shape = (4, 4) if self.lanes == 16 else (self.lanes,)
        if self.count > 1:
            shape = (self.count, *shape)
        return scalar, shape


@dataclass(frozen=True)
class ShaderAbi:
    name: str
    fields: tuple[AbiField, ...]
    size: int

    def numpy_dtype(self):
        return np.dtype({
            "names": tuple(field.name for field in self.fields),
            "formats": tuple(field.numpy_format for field in self.fields),
            "offsets": tuple(field.offset for field in self.fields),
            "itemsize": self.size,
        })

    def validate_dtype(self, dtype):
        dtype = np.dtype(dtype)
        actual_names = tuple(dtype.names or ())
        expected_names = tuple(field.name for field in self.fields)
        if actual_names != expected_names:
            raise AssertionError(
                f"{self.name} fields {actual_names!r} != {expected_names!r}"
            )
        for field in self.fields:
            actual = dtype.fields[field.name][1]
            if actual != field.offset:
                raise AssertionError(
                    f"{self.name}.{field.name} offset {actual} != {field.offset}"
                )
        if dtype.itemsize != self.size:
            raise AssertionError(
                f"{self.name} size {dtype.itemsize} != {self.size}"
            )

    def expected_shader_members(self, *, expand_arrays=False):
        result = []
        for field in self.fields:
            if field.count == 1:
                result.append((field.shader_type, field.name))
            elif expand_arrays:
                stem = field.name[:-1] if field.name.endswith("s") else field.name
                result.extend(
                    (field.shader_type, f"{stem}_{index}")
                    for index in range(field.count)
                )
            else:
                result.append((field.shader_type, f"{field.name}[{field.count}]"))
        return tuple(result)

    def validate_shader_source(self, source, *, expand_arrays=False):
        match = re.search(
            rf"\bstruct\s+{re.escape(self.name)}\s*\{{(?P<body>.*?)\}}\s*;?",
            source, re.DOTALL,
        )
        if match is None:
            raise AssertionError(f"shader does not declare {self.name}")
        body = re.sub(r"//.*", "", match.group("body"))
        members = []
        for declaration in re.split(r"[,;]", body):
            declaration = declaration.strip()
            if not declaration:
                continue
            # WGSL uses ``name: type``; GLSL uses ``type name``.
            wgsl = re.fullmatch(
                r"(\w+)\s*:\s*(\w+)(?:<(\w+)>)?", declaration
            )
            if wgsl:
                name, shader_type, scalar = wgsl.groups()
                if shader_type == "mat4x4":
                    shader_type = "mat4"
                elif scalar == "u32":
                    shader_type = "u" + shader_type
            else:
                glsl = re.fullmatch(r"(\w+)\s+(\w+(?:\[\d+\])?)", declaration)
                if glsl is None:
                    raise AssertionError(
                        f"cannot parse {self.name} member {declaration!r}"
                    )
                shader_type, name = glsl.groups()
            members.append((shader_type, name))
        expected = self.expected_shader_members(expand_arrays=expand_arrays)
        if tuple(members) != expected:
            raise AssertionError(
                f"{self.name} shader members {tuple(members)!r} != {expected!r}"
            )


SECONDARY_PATH_STATE_ABI = ShaderAbi("SecondaryPathState", (
    AbiField("position_valid", "float", 0),
    AbiField("normal_pdf", "float", 16),
    AbiField("primary_throughput", "float", 32),
    AbiField("primary_radiance", "float", 48),
    AbiField("diffuse_radiance_hit_distance", "float", 64),
    AbiField("specular_radiance_hit_distance", "float", 80),
    AbiField("primary_position", "float", 96),
    AbiField("primary_geometry", "float", 112),
), 128)

VOLUME_HEADER_ABI = ShaderAbi("VolumeHeader", (
    AbiField("world_to_local", "float", 0, lanes=16),
    AbiField("dimensions_offset", "uint", 64),
    AbiField("value_parameters", "float", 80),
    AbiField("render_parameters", "float", 96),
    AbiField("scattering_parameters", "float", 112),
    AbiField("phase_parameters", "float", 128),
    AbiField("multiple_scattering_parameters", "float", 144),
    AbiField("acceleration_parameters", "uint", 160),
    AbiField("clip_parameters", "uint", 176),
    AbiField("clip_planes", "float", 192, count=8),
), 320)


def reflect_spirv_struct(path, name):
    """Return member offsets and containing array stride from SPIR-V words."""
    data = Path(path).read_bytes()
    if len(data) < 20 or len(data) % 4:
        raise ValueError("invalid SPIR-V byte length")
    words = struct.unpack(f"<{len(data) // 4}I", data)
    if words[0] != 0x07230203:
        raise ValueError("invalid SPIR-V magic")
    names = {}
    member_offsets = {}
    runtime_arrays = {}
    array_strides = {}
    index = 5
    while index < len(words):
        instruction = words[index]
        word_count, opcode = instruction >> 16, instruction & 0xffff
        if word_count < 1 or index + word_count > len(words):
            raise ValueError("malformed SPIR-V instruction")
        operands = words[index + 1:index + word_count]
        if opcode == 5:  # OpName
            raw = struct.pack(f"<{len(operands) - 1}I", *operands[1:])
            names[operands[0]] = raw.split(b"\0", 1)[0].decode("utf8")
        elif opcode == 29:  # OpTypeRuntimeArray
            runtime_arrays[operands[0]] = operands[1]
        elif opcode == 71 and len(operands) >= 3:  # OpDecorate
            if operands[1] == 6:  # ArrayStride
                array_strides[operands[0]] = operands[2]
        elif opcode == 72 and len(operands) >= 4:  # OpMemberDecorate
            if operands[2] == 35:  # Offset
                member_offsets.setdefault(operands[0], {})[operands[1]] = operands[3]
        index += word_count
    matches = [identifier for identifier, value in names.items() if value == name]
    if not matches:
        raise AssertionError(f"SPIR-V does not name {name}")
    struct_id = max(matches, key=lambda value: len(member_offsets.get(value, {})))
    offsets = member_offsets.get(struct_id, {})
    ordered = tuple(offsets[index] for index in sorted(offsets))
    strides = {
        array_strides[array_id] for array_id, element in runtime_arrays.items()
        if element == struct_id and array_id in array_strides
    }
    return ordered, (next(iter(strides)) if len(strides) == 1 else None)


__all__ = [
    "AbiField", "ShaderAbi", "SECONDARY_PATH_STATE_ABI",
    "VOLUME_HEADER_ABI", "reflect_spirv_struct",
]
