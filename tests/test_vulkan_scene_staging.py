"""Synchronous scene uploads reuse storage without retaining stale payloads."""
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from ordinarylight.targets.vulkan.core import VulkanRayQueryCore


def test_staging_reuse_growth_and_payload_offsets():
    core = object.__new__(VulkanRayQueryCore)
    core.device = object()
    core._buffers = []
    allocations = []
    mapped_bytes = []
    copies = []

    def allocate(size, *_args):
        buffer = SimpleNamespace(size=size, buffer=object(), memory=object())
        allocations.append(buffer)
        core._buffers.append(buffer)
        return buffer

    def mapped(_device, _memory, _offset, size, _flags):
        value = bytearray(size)
        mapped_bytes.append(value)
        return value

    core._create_buffer = allocate
    core._single_use = lambda record: record(object())
    first = SimpleNamespace(size=8, buffer=object())
    second = SimpleNamespace(size=4, buffer=object())
    with patch('ordinarylight.targets.vulkan.scene.vk.vkMapMemory', mapped), \
         patch('ordinarylight.targets.vulkan.scene.vk.vkUnmapMemory'), \
         patch('ordinarylight.targets.vulkan.scene.vk.vkCmdCopyBuffer',
               side_effect=lambda *args: copies.append(args[-1][0])), \
         patch('ordinarylight.targets.vulkan.scene.vk.vkDestroyBuffer') as destroy, \
         patch('ordinarylight.targets.vulkan.scene.vk.vkFreeMemory') as free:
        core._update_device_buffers([(first, np.array([1, 2], np.uint32)),
                                     (second, np.array([3], np.uint32))])
        core._update_device_buffers([(first, np.array([4, 5], np.uint32))])
        assert len(allocations) == 1
        assert mapped_bytes[0] == np.array([1, 2, 3], np.uint32).tobytes()
        assert mapped_bytes[1] == np.array([4, 5], np.uint32).tobytes()
        assert [(x.srcOffset, x.size) for x in copies] == [(0, 8), (8, 4), (0, 8)]
        larger = SimpleNamespace(size=16, buffer=object())
        core._update_device_buffers([(larger, np.zeros(4, np.uint32))])
        assert len(allocations) == 2
        assert core._buffers == [allocations[1]]
        destroy.assert_called_once()
        free.assert_called_once()
        with pytest.raises(ValueError, match='byte size'):
            core._update_device_buffers([(first, np.zeros(3, np.uint32))])
        core._update_device_buffers([])
        assert len(allocations) == 2
