"""Explicit device-local allocation with synchronized staging boundaries."""

import os
import numpy as np
import pytest
import vulkan as vk
import ordinarylight as ol
from ordinarylight.runtime import VulkanKernel, compile_compute
from ordinarylight.pipeline.graph import VulkanGraph
from ordinarylight.pipeline.vulkan import VulkanResource, VulkanResourceUse, VulkanPass

pytestmark = pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_RUNTIME") != "1",
    reason="opt-in device-local buffer GPU validation",
)


def test_device_upload_partial_gpu_write_and_readback():
    with ol.VulkanRuntime() as runtime:
        with runtime.buffer(32, data=bytes(range(32)), memory="device") as buffer:
            assert buffer.memory_flags & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
            buffer.upload(b"abc", offset=1)
            buffer.upload(b"", offset=32)
            assert buffer.read() == bytes([0]) + b"abc" + bytes(range(4, 32))
            with pytest.raises(ValueError, match="bounds"):
                buffer.upload(b"abc", offset=31)
            buffer.upload(np.arange(8, dtype=np.uint32))
            bindings = {0: VulkanResource.buffer(buffer)}
            with VulkanKernel(
                runtime,
                compile_compute("""#version 460
layout(local_size_x=8) in;
layout(set=0,binding=0,std430) buffer Data { uint values[]; };
void main() { uint i=gl_GlobalInvocationID.x; values[i]=values[i]*3u+7u; }
"""),
                bindings,
            ) as kernel:
                stage = VulkanPass(
                    "write",
                    (
                        VulkanResourceUse(
                            bindings[0],
                            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            vk.VK_ACCESS_SHADER_READ_BIT
                            | vk.VK_ACCESS_SHADER_WRITE_BIT,
                        ),
                    ),
                    lambda command: kernel.bind(command),
                    (1, 1, 1),
                )
                VulkanGraph().add("write", stage).compile().execute(runtime).wait()
                np.testing.assert_array_equal(
                    np.frombuffer(buffer.read(), np.uint32), np.arange(8) * 3 + 7
                )
                with pytest.raises(RuntimeError, match="borrowers"):
                    buffer.close()
        with pytest.raises(ValueError, match="memory"):
            runtime.buffer(16, memory="unknown")
        with runtime.buffer(16) as host:
            assert host.memory_kind == "host"
            host.upload(bytes(range(16)))
            assert host.read() == bytes(range(16))
