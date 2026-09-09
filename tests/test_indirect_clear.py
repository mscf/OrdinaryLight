import os
import pytest
from ordinarylight.runtime import VulkanRuntime, clear_indirect_reservoirs


@pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_GRAPH") != "1", reason="opt-in GPU"
)
def test_reservoir_clear_preserves_tail_and_handles_empty_range():
    with (
        VulkanRuntime() as runtime,
        runtime.buffer(80, data=bytes([37]) * 80) as buffer,
    ):
        clear_indirect_reservoirs(buffer, count=0).execute(runtime).wait()
        assert buffer.read() == bytes([37]) * 80
        clear_indirect_reservoirs(buffer, count=2).execute(runtime).wait()
        assert buffer.read() == bytes(48) + bytes([37]) * 32
        with pytest.raises(ValueError, match="count"):
            clear_indirect_reservoirs(buffer, count=4)
