"""Queue dispatch counts, graph consumption, and allocation ownership."""

from contextlib import ExitStack
import os
import struct
import pytest
import vulkan as vk
from ordinarylight.runtime import VulkanRuntime, VulkanQueueDispatch


@pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_GRAPH") != "1", reason="opt-in GPU"
)
def test_queue_counts_and_leases():
    with VulkanRuntime() as runtime, ExitStack() as stack:
        queue = stack.enter_context(runtime.buffer(16))
        arguments = stack.enter_context(
            runtime.buffer(
                12,
                usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                | vk.VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
                | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            )
        )
        with pytest.raises(ValueError, match="alias"):
            VulkanQueueDispatch(runtime, queue=queue, arguments=queue)
        with pytest.raises(ValueError, match="indirect"):
            VulkanQueueDispatch(
                runtime, queue=queue, arguments=stack.enter_context(runtime.buffer(16))
            )
        stage = stack.enter_context(
            VulkanQueueDispatch(runtime, queue=queue, arguments=arguments)
        )
        for count, capacity, expected in [
            (0, 128, 0),
            (1, 128, 1),
            (64, 128, 1),
            (65, 128, 2),
            (200, 128, 2),
            (0, 0, 0),
        ]:
            queue.upload(struct.pack("4I", count, capacity, 0, 0))
            stage.operation().execute(runtime).wait()
            assert struct.unpack("3I", arguments.read()) == (expected, 1, 1)
        with pytest.raises(RuntimeError, match="borrowers"):
            arguments.close()
