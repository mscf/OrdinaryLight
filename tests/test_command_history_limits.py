"""Inactive history settings must not invalidate reusable GPU commands."""
import pytest
from ordinarylight.targets.vulkan.core import _command_history_limits


@pytest.mark.parametrize('indirect,restir,temporal,spatial,expected', [
    (False, False, False, False, (None, None)),
    (False, True, False, False, (None, None)),
    (False, True, True, False, (None, 20)),
    (False, True, False, True, (None, 20)),
    (True, True, True, False, (32, 20)),
    (True, False, False, False, (32, None)),
])
def test_only_consumed_limits_participate(indirect, restir, temporal, spatial, expected):
    assert _command_history_limits(
        32, 20, indirect_enabled=indirect, restir_enabled=restir,
        restir_history_valid=temporal, restir_spatial_reuse=spatial,
    ) == expected


def test_next_upload_does_not_overwrite_inflight_previous_camera(monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    from ordinarylight.targets.vulkan import core

    executor = object.__new__(core.VulkanWavefrontExecutor)
    executor.core = SimpleNamespace(device=None)
    executor.camera_buffers = [SimpleNamespace(memory=bytearray(64)) for _ in range(2)]
    executor.previous_camera_buffers = [SimpleNamespace(memory=bytearray(64)) for _ in range(2)]
    executor._camera_payloads = [bytes(64), bytes(64)]
    monkeypatch.setattr(core.vk, 'vkMapMemory', lambda device, memory, *args: memory)
    monkeypatch.setattr(core.vk, 'vkUnmapMemory', lambda *args: None)
    vectors = [np.ones(3, dtype=np.float32) * i for i in range(4)]
    executor.update_camera(0, vectors, frame_sequence=10)
    camera0 = bytes(executor.camera_buffers[0].memory)
    executor.update_camera(1, vectors, frame_sequence=11)
    assert bytes(executor.previous_camera_buffers[1].memory) == camera0
    executor.update_camera(0, vectors, frame_sequence=12)
    assert bytes(executor.camera_buffers[0].memory) != camera0
    assert bytes(executor.previous_camera_buffers[1].memory) == camera0
    assert bytes(executor.previous_camera_buffers[0].memory) == bytes(executor.camera_buffers[1].memory)
