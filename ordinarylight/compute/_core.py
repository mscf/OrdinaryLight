"""Generic persistent compute resources for Ordinary Shade programs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class ComputeBuffer:
    """Host description of one uniform or storage-buffer allocation."""

    data: Any = None
    shape: tuple[int, ...] = ()
    dtype: Any = np.float32
    nbytes: int | None = None

    def payload(self) -> bytes:
        if self.data is None:
            size = self.byte_size
            return bytes(size)
        if isinstance(self.data, bytes):
            return self.data
        return np.ascontiguousarray(self.data, dtype=self.dtype).tobytes()

    @property
    def byte_size(self) -> int:
        if self.data is not None:
            return len(self.payload())
        if self.nbytes is not None:
            return int(self.nbytes)
        if not self.shape:
            raise ValueError("an uninitialized compute buffer requires shape or nbytes")
        return int(np.prod(self.shape)) * np.dtype(self.dtype).itemsize


@dataclass(frozen=True, slots=True)
class ComputeStep:
    """One program dispatch in an ordered compute sequence.

    ``resources`` maps each reflected shader resource name to the name of a
    shared sequence allocation. An empty mapping uses identical names.
    """

    program: Any
    workgroups: tuple[int, int, int]
    resources: Mapping[str, str] | None = None


@dataclass(frozen=True, slots=True)
class WebGpuBufferView:
    """A non-owning typed view of a resident WebGPU compute allocation."""

    owner: Any = field(repr=False)
    name: str
    shape: tuple[int, ...]
    dtype: Any

    @property
    def device(self):
        return self.owner.device

    @property
    def buffer(self):
        """The native buffer handle for same-device backend integrations."""
        self.owner._require_open()
        return self.owner._buffers[self.name]

    @property
    def byte_size(self):
        return self.owner._descriptions[self.name].byte_size

    def copy_to_texture(self, encoder, texture) -> None:
        """Encode a tightly packed 3-D buffer-to-texture copy."""
        self.owner._require_open()
        dtype = np.dtype(self.dtype)
        if len(self.shape) != 3:
            raise ValueError("a volume buffer view requires a three-dimensional shape")
        depth, height, width = self.shape
        bytes_per_row = width * dtype.itemsize
        if bytes_per_row % 256:
            raise ValueError("WebGPU volume buffer rows must be 256-byte aligned")
        encoder.copy_buffer_to_texture(
            {"buffer": self.owner._buffers[self.name], "bytes_per_row": bytes_per_row,
             "rows_per_image": height},
            {"texture": texture}, (width, height, depth),
        )


def _validate_program(program):
    if program.target != "wgsl" or program.reflection.stage != "compute":
        raise ValueError("WebGPU compute requires a WGSL compute shader")
    reflected = tuple(program.reflection.resources)
    unsupported = [
        item.name for item in reflected
        if item.kind not in {"uniform_buffer", "storage_buffer"}
    ]
    if unsupported:
        raise ValueError(
            "generic buffer compute does not support resources: "
            + ", ".join(unsupported)
        )
    return reflected


def _workgroups(value):
    groups = tuple(int(item) for item in value)
    if len(groups) != 3 or any(item < 1 for item in groups):
        raise ValueError("workgroups must contain three positive integers")
    return groups


def _description(value):
    return value if isinstance(value, ComputeBuffer) else ComputeBuffer(value)


def _buffer_usage(wgpu, *, uniform=False):
    usage = wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC
    return usage | (wgpu.BufferUsage.UNIFORM if uniform else wgpu.BufferUsage.STORAGE)


class WebGpuComputeSession:
    """Persistent WebGPU compute pipeline with independently updateable buffers."""

    def __init__(
        self, program, resources: Mapping[str, ComputeBuffer | Any], *,
        power_preference="high-performance", device=None, _wgpu=None,
    ):
        reflected = _validate_program(program)
        if _wgpu is None:
            try:
                import wgpu as _wgpu
            except ImportError as error:
                raise RuntimeError(
                    "WebGPU compute requires: pip install 'ordinarylight[webgpu]'"
                ) from error
        self._wgpu = _wgpu
        self.program = program
        self.device = device
        if self.device is None:
            adapter = _wgpu.gpu.request_adapter_sync(
                power_preference=power_preference,
            )
            self.device = adapter.request_device_sync()
        expected = {item.name for item in reflected}
        supplied = set(resources)
        if supplied != expected:
            missing = sorted(expected - supplied)
            extra = sorted(supplied - expected)
            raise ValueError(f"compute resource mismatch: missing={missing}, extra={extra}")

        module = self.device.create_shader_module(code=program.source)
        self.pipeline = self.device.create_compute_pipeline(
            layout="auto",
            compute={"module": module, "entry_point": program.reflection.entry_point},
        )
        self._descriptions = {}
        self._buffers = {}
        entries = []
        for reflection in reflected:
            description = resources[reflection.name]
            description = _description(description)
            payload = description.payload()
            usage = _buffer_usage(
                _wgpu, uniform=reflection.kind == "uniform_buffer",
            )
            buffer = self.device.create_buffer_with_data(data=payload, usage=usage)
            self._descriptions[reflection.name] = description
            self._buffers[reflection.name] = buffer
            entries.append({
                "binding": reflection.binding,
                "resource": {"buffer": buffer, "offset": 0, "size": len(payload)},
            })
        self.bind_group = self.device.create_bind_group(
            layout=self.pipeline.get_bind_group_layout(0), entries=entries,
        )
        self.closed = False

    def update(self, name: str, data: Any) -> None:
        """Update one resident allocation without rebuilding the pipeline."""
        self._require_open()
        if name not in self._buffers:
            raise KeyError(name)
        description = self._descriptions[name]
        payload = ComputeBuffer(data, dtype=description.dtype).payload()
        if len(payload) != description.byte_size:
            raise ValueError(
                f"update for {name!r} has {len(payload)} bytes; "
                f"expected {description.byte_size}"
            )
        self.device.queue.write_buffer(self._buffers[name], 0, payload)

    def dispatch(self, workgroups: tuple[int, int, int]) -> None:
        """Submit one compute dispatch using the current resident resources."""
        self._require_open()
        groups = _workgroups(workgroups)
        encoder = self.device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        compute_pass.set_pipeline(self.pipeline)
        compute_pass.set_bind_group(0, self.bind_group, (), 0, 0)
        compute_pass.dispatch_workgroups(*groups)
        compute_pass.end()
        self.device.queue.submit((encoder.finish(),))

    def read(self, name: str, *, dtype=None, shape=None) -> np.ndarray:
        """Synchronously read one resident buffer into a detached NumPy array."""
        self._require_open()
        if name not in self._buffers:
            raise KeyError(name)
        description = self._descriptions[name]
        resolved_dtype = np.dtype(description.dtype if dtype is None else dtype)
        resolved_shape = description.shape if shape is None else tuple(shape)
        raw = self.device.queue.read_buffer(self._buffers[name])
        result = np.frombuffer(raw, dtype=resolved_dtype).copy()
        return result.reshape(resolved_shape) if resolved_shape else result

    def _require_open(self):
        if self.closed:
            raise RuntimeError("compute session is closed")

    def close(self) -> None:
        self._buffers.clear()
        self.bind_group = None
        self.pipeline = None
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


class WebGpuComputeSequence:
    """Persistent shared buffers and pipelines for ordered GPU-only passes."""

    def __init__(
        self, steps, resources: Mapping[str, ComputeBuffer | Any], *,
        power_preference="high-performance", device=None, _wgpu=None,
    ):
        if _wgpu is None:
            try:
                import wgpu as _wgpu
            except ImportError as error:
                raise RuntimeError(
                    "WebGPU compute requires: pip install 'ordinarylight[webgpu]'"
                ) from error
        self._wgpu = _wgpu
        self.device = device
        if self.device is None:
            adapter = _wgpu.gpu.request_adapter_sync(
                power_preference=power_preference,
            )
            self.device = adapter.request_device_sync()
        self.steps = tuple(steps)
        if not self.steps:
            raise ValueError("compute sequence requires at least one step")
        if not all(isinstance(step, ComputeStep) for step in self.steps):
            raise TypeError("compute sequence steps must be ComputeStep instances")

        self._descriptions = {
            name: _description(value) for name, value in resources.items()
        }
        self._buffers = {}
        for name, description in self._descriptions.items():
            payload = description.payload()
            # A shared allocation may be uniform in one shader and storage in
            # another, so sequence buffers conservatively carry both usages.
            usage = (
                _wgpu.BufferUsage.COPY_DST | _wgpu.BufferUsage.COPY_SRC
                | _wgpu.BufferUsage.UNIFORM | _wgpu.BufferUsage.STORAGE
            )
            self._buffers[name] = self.device.create_buffer_with_data(
                data=payload, usage=usage,
            )

        prepared = []
        for step in self.steps:
            reflected = _validate_program(step.program)
            groups = _workgroups(step.workgroups)
            aliases = dict(step.resources or {
                item.name: item.name for item in reflected
            })
            expected = {item.name for item in reflected}
            if set(aliases) != expected:
                raise ValueError(
                    "compute step resource mismatch: "
                    f"missing={sorted(expected - set(aliases))}, "
                    f"extra={sorted(set(aliases) - expected)}"
                )
            missing_allocations = sorted(set(aliases.values()) - set(self._buffers))
            if missing_allocations:
                raise ValueError(
                    "compute step references missing allocations: "
                    + ", ".join(missing_allocations)
                )
            module = self.device.create_shader_module(code=step.program.source)
            pipeline = self.device.create_compute_pipeline(
                layout="auto", compute={
                    "module": module,
                    "entry_point": step.program.reflection.entry_point,
                },
            )
            entries = [{
                "binding": item.binding,
                "resource": {
                    "buffer": self._buffers[aliases[item.name]], "offset": 0,
                    "size": self._descriptions[aliases[item.name]].byte_size,
                },
            } for item in reflected]
            bind_group = self.device.create_bind_group(
                layout=pipeline.get_bind_group_layout(0), entries=entries,
            )
            prepared.append((pipeline, bind_group, groups))
        self._prepared = tuple(prepared)
        self.closed = False

    def update(self, name: str, data: Any) -> None:
        """Update one shared allocation without rebuilding any pipeline."""
        self._require_open()
        if name not in self._buffers:
            raise KeyError(name)
        description = self._descriptions[name]
        payload = ComputeBuffer(data, dtype=description.dtype).payload()
        if len(payload) != description.byte_size:
            raise ValueError(
                f"update for {name!r} has {len(payload)} bytes; "
                f"expected {description.byte_size}"
            )
        self.device.queue.write_buffer(self._buffers[name], 0, payload)

    def dispatch(self) -> None:
        """Encode every step in order and submit one command buffer."""
        self._require_open()
        encoder = self.device.create_command_encoder()
        compute_pass = encoder.begin_compute_pass()
        for pipeline, bind_group, groups in self._prepared:
            compute_pass.set_pipeline(pipeline)
            compute_pass.set_bind_group(0, bind_group, (), 0, 0)
            compute_pass.dispatch_workgroups(*groups)
        compute_pass.end()
        self.device.queue.submit((encoder.finish(),))

    def read(self, name: str, *, dtype=None, shape=None) -> np.ndarray:
        """Synchronously read one shared allocation after submitted work."""
        self._require_open()
        if name not in self._buffers:
            raise KeyError(name)
        description = self._descriptions[name]
        resolved_dtype = np.dtype(description.dtype if dtype is None else dtype)
        resolved_shape = description.shape if shape is None else tuple(shape)
        raw = self.device.queue.read_buffer(self._buffers[name])
        result = np.frombuffer(raw, dtype=resolved_dtype).copy()
        return result.reshape(resolved_shape) if resolved_shape else result

    def buffer_view(self, name: str) -> WebGpuBufferView:
        """Return a non-owning view for same-device GPU consumers."""
        self._require_open()
        if name not in self._buffers:
            raise KeyError(name)
        description = self._descriptions[name]
        return WebGpuBufferView(
            self, name, tuple(description.shape), np.dtype(description.dtype),
        )

    def _require_open(self):
        if self.closed:
            raise RuntimeError("compute sequence is closed")

    def close(self) -> None:
        self._buffers.clear()
        self._prepared = ()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


__all__ = [
    "ComputeBuffer", "ComputeStep", "WebGpuBufferView", "WebGpuComputeSequence",
    "WebGpuComputeSession",
]
