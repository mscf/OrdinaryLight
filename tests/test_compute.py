from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

import ordinarylight as ol


@dataclass
class Resource:
    name: str
    kind: str
    binding: int


class FakeQueue:
    def __init__(self):
        self.writes = []
        self.submissions = []

    def write_buffer(self, buffer, offset, payload):
        buffer["data"] = bytes(payload)
        self.writes.append((buffer, offset, bytes(payload)))

    def read_buffer(self, buffer):
        return buffer["data"]

    def submit(self, commands):
        self.submissions.append(commands)


class FakePass:
    def __init__(self):
        self.groups = None
        self.dispatches = []

    def set_pipeline(self, pipeline): self.pipeline = pipeline
    def set_bind_group(self, *args): self.bind = args
    def dispatch_workgroups(self, *groups):
        self.groups = groups
        self.dispatches.append((self.pipeline, self.bind, groups))
    def end(self): self.ended = True


class FakeEncoder:
    def __init__(self): self.compute_pass = FakePass()
    def begin_compute_pass(self): return self.compute_pass
    def finish(self): return self


class FakePipeline:
    def get_bind_group_layout(self, index): return ("layout", index)


class FakeDevice:
    def __init__(self):
        self.queue = FakeQueue()
        self.encoder = None

    def create_shader_module(self, **kwargs): return kwargs
    def create_compute_pipeline(self, **kwargs): return FakePipeline()
    def create_buffer_with_data(self, *, data, usage):
        return {"data": bytes(data), "usage": usage}
    def create_bind_group(self, **kwargs): return kwargs
    def create_command_encoder(self):
        self.encoder = FakeEncoder()
        return self.encoder


FAKE_WGPU = SimpleNamespace(BufferUsage=SimpleNamespace(
    COPY_DST=1, COPY_SRC=2, UNIFORM=4, STORAGE=8,
))


def program(resources=None):
    return SimpleNamespace(
        target="wgsl", source="shader", reflection=SimpleNamespace(
            stage="compute", entry_point="main", resources=tuple(resources or ()),
        ),
    )


def test_compute_buffer_sizes_initialized_and_host_data():
    assert ol.ComputeBuffer(shape=(4,), dtype=np.float32).byte_size == 16
    assert ol.ComputeBuffer(np.arange(3, dtype=np.float32)).byte_size == 12
    with pytest.raises(ValueError, match="requires shape or nbytes"):
        ol.ComputeBuffer().byte_size


def test_webgpu_session_persists_updates_dispatches_and_reads():
    device = FakeDevice()
    shader = program((
        Resource("parameters", "uniform_buffer", 0),
        Resource("values", "storage_buffer", 1),
        Resource("output", "storage_buffer", 2),
    ))
    session = ol.WebGpuComputeSession(shader, {
        "parameters": ol.ComputeBuffer(np.array((4,), np.uint32), dtype=np.uint32),
        "values": ol.ComputeBuffer(np.arange(4, dtype=np.float32), shape=(4,)),
        "output": ol.ComputeBuffer(shape=(4,), dtype=np.float32),
    }, device=device, _wgpu=FAKE_WGPU)
    session.update("values", np.full(4, 3.0, np.float32))
    session.dispatch((1, 1, 1))
    np.testing.assert_array_equal(session.read("values"), np.full(4, 3.0))
    assert device.encoder.compute_pass.groups == (1, 1, 1)
    assert len(device.queue.submissions) == 1
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.dispatch((1, 1, 1))


def test_compute_session_validates_program_resources_and_update_sizes():
    shader = program((Resource("values", "storage_buffer", 0),))
    device = FakeDevice()
    with pytest.raises(ValueError, match="resource mismatch"):
        ol.WebGpuComputeSession(shader, {}, device=device, _wgpu=FAKE_WGPU)
    session = ol.WebGpuComputeSession(
        shader, {"values": ol.ComputeBuffer(shape=(4,))},
        device=device, _wgpu=FAKE_WGPU,
    )
    with pytest.raises(ValueError, match="expected 16"):
        session.update("values", np.ones(3, np.float32))


def test_compute_sequence_reuses_allocations_and_submits_steps_in_order():
    device = FakeDevice()
    first = program((
        Resource("parameters", "uniform_buffer", 0),
        Resource("source", "storage_buffer", 1),
        Resource("destination", "storage_buffer", 3),
    ))
    second = program((
        Resource("parameters", "uniform_buffer", 0),
        Resource("source", "storage_buffer", 3),
        Resource("destination", "storage_buffer", 2),
    ))
    steps = (
        ol.ComputeStep(first, (2, 1, 1), {
            "parameters": "parameters_0", "source": "values",
            "destination": "scratch",
        }),
        ol.ComputeStep(second, (1, 1, 1), {
            "parameters": "parameters_1", "source": "scratch",
            "destination": "result",
        }),
    )
    sequence = ol.WebGpuComputeSequence(steps, {
        "parameters_0": ol.ComputeBuffer(bytes(16)),
        "parameters_1": ol.ComputeBuffer(bytes(16)),
        "values": ol.ComputeBuffer(np.arange(8, dtype=np.float32), shape=(8,)),
        "scratch": ol.ComputeBuffer(shape=(2,), dtype=np.float32),
        "result": ol.ComputeBuffer(shape=(1,), dtype=np.float32),
    }, device=device, _wgpu=FAKE_WGPU)
    sequence.dispatch()
    assert [item[2] for item in device.encoder.compute_pass.dispatches] == [
        (2, 1, 1), (1, 1, 1),
    ]
    assert len(device.queue.submissions) == 1
    sequence.update("values", np.full(8, 2.0, np.float32))
    np.testing.assert_array_equal(sequence.read("values"), np.full(8, 2.0))


def test_compute_sequence_validates_aliases_and_allocations():
    shader = program((Resource("values", "storage_buffer", 0),))
    with pytest.raises(ValueError, match="missing allocations"):
        ol.WebGpuComputeSequence(
            (ol.ComputeStep(shader, (1, 1, 1), {"values": "absent"}),),
            {}, device=FakeDevice(), _wgpu=FAKE_WGPU,
        )
