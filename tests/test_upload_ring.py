"""GPU staging ownership, backpressure and same-queue ordering contracts."""
import os
from unittest.mock import patch
import pytest

pytestmark = pytest.mark.skipif(os.environ.get('ORDINARYLIGHT_TEST_VULKAN_RUNTIME') != '1',
                                reason='opt-in Vulkan GPU test')


def test_uploads_snapshot_ranges_and_submit_without_idle_waits_or_allocations():
    import vulkan as vk
    from ordinarylight.runtime import VulkanRuntime,VulkanUploadRing,VulkanCompletion
    from ordinarylight.pipeline.vulkan import VulkanResource
    with VulkanRuntime() as runtime, runtime.buffer(32,memory='device') as target, \
            VulkanUploadRing(runtime,32) as ring:
        data = bytearray(b'abcd')
        def forbidden(*args,**kwargs):
            raise AssertionError('Unexpected allocation or CPU wait')
        with patch.object(vk,'vkDeviceWaitIdle',forbidden), \
                patch.object(vk,'vkQueueWaitIdle',forbidden), \
                patch.object(vk,'vkWaitForFences',forbidden), \
                patch.object(vk,'vkAllocateMemory',forbidden), \
                patch.object(vk,'vkCreateBuffer',forbidden):
            first = ring.prepare([(target,0,data),(target,16,b'keep')],wait=False)
            data[:] = b'xxxx'
            first.operation.execute(runtime)
            second = ring.prepare([(VulkanResource.buffer(target).byte_range(4,12),4,b'next')],wait=False)
            second.operation.execute(runtime)
        second.completion.wait()
        result = target.read()
        assert result[:4] == b'abcd' and result[8:12] == b'next' and result[16:20] == b'keep'
        with pytest.raises(RuntimeError,match='no longer'):
            first.operation.execute(runtime)


def test_upload_backpressure_cancellation_validation_and_borrowing():
    from ordinarylight.runtime import VulkanRuntime,VulkanUploadRing,VulkanUploadBusy
    with VulkanRuntime() as runtime, runtime.buffer(16) as target, \
            VulkanUploadRing(runtime,16,slots=1) as ring:
        for writes in ([],[(target,0,b'123')],[(target,16,b'abcd')],
                       [(target,0,b'abcdefgh'),(target,4,b'xxxx')]):
            with pytest.raises(ValueError):
                ring.prepare(writes)
        packet = ring.prepare([(target,0,b'old!')])
        with pytest.raises(RuntimeError):
            target.close()
        for wait in (False,True):
            with pytest.raises(VulkanUploadBusy):
                ring.prepare([(target,0,b'new!')],wait=wait)
        packet.cancel()
        replacement = ring.prepare([(target,0,b'new!')])
        with pytest.raises(RuntimeError):
            packet.operation.execute(runtime)
        replacement.operation.execute(runtime)
        completion = replacement.completion
        # Force the busy decision deterministically even on a fast GPU, then
        # verify backpressure waits on this slot's fence, not the whole device.
        with patch.object(completion,'poll',return_value=False), \
                patch.object(completion,'wait',wraps=completion.wait) as waited:
            with pytest.raises(VulkanUploadBusy):
                ring.prepare([(target,0,b'last')],wait=False)
            last = ring.prepare([(target,0,b'last')])
            assert waited.call_count == 1
        last.operation.execute(runtime).wait()
        assert target.read()[:4] == b'last'
        pending = ring.prepare([(target,0,b'drop')])
        ring.close()
        assert pending.state == 'cancelled'
