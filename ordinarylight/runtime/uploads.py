"""Bounded, persistently mapped staging for single-queue graph uploads."""
from operator import index
import vulkan as vk
from .resources import VulkanBuffer
from .acceleration import buffer_copy_operation
from ..pipeline.graph import VulkanOperation
from ..pipeline.vulkan import VulkanResource, VulkanPass


class VulkanUploadBusy(RuntimeError):
    """No staging slot is available without waiting or cancelling a packet."""


class VulkanUploadPacket:
    """One-shot graph operation. Cancel unsubmitted packets after abandoning a frame.

    Payload bytes are snapshotted by prepare(). Operations must not be replayed
    from cached command buffers. Destination buffers are borrowed until submission
    or cancellation; the submission then retains GPU resources until completion.
    """
    def __init__(self, ring, slot, passes, owners):
        self._ring, self._slot = ring, slot
        self.state = 'prepared'
        self.completion = None
        self._owners = []
        try:
            for owner in owners:
                owner.retain(self)
                self._owners.append(owner)
        except BaseException:
            self._release()
            raise
        guarded = []
        for stage in passes:
            def record(command, stage=stage):
                self._validate()
                stage.record(command)
            guarded.append(VulkanPass(stage.name, stage.uses, record))
        self.operation = VulkanOperation(guarded, validate=self._validate,
                                         submitted=self._submitted)

    def _validate(self):
        self._ring.require_open()
        if self.state != 'prepared' or self._slot['packet'] is not self:
            raise RuntimeError('Upload packet is no longer available for submission')

    def _release(self):
        for owner in self._owners:
            owner.release(self)
        self._owners.clear()

    def _submitted(self, completion):
        self.completion = completion
        self.state = 'submitted'
        self._slot['completion'] = completion
        self._slot['packet'] = None
        self._release()

    def cancel(self):
        with self._ring.runtime.lock:
            if self.state == 'prepared':
                self.state = 'cancelled'
                self._slot['packet'] = None
                self._release()


class VulkanUploadRing:
    """Fixed staging capacity; no GPU allocations or idle waits during prepare.

    prepare(writes) accepts (buffer-or-view, byte_offset, contiguous_data) tuples.
    Writes must be nonempty, four-byte aligned and disjoint. A free slot is polled
    first. If all submitted slots are busy, wait=True waits for one completion;
    wait=False raises VulkanUploadBusy. Unsubmitted packets must be cancelled by
    the caller, never overwritten. Graph barriers order destination reuse against
    earlier work on the same runtime queue. Close is a synchronization boundary.
    """
    def __init__(self, runtime, capacity, *, slots=3):
        self.runtime = runtime
        self.capacity, count = index(capacity), index(slots)
        if self.capacity <= 0 or count <= 0:
            raise ValueError('Upload capacity and slot count must be positive')
        self.closed = False
        self._slots = []
        with runtime.lock:
            runtime.require_open()
            try:
                for _ in range(count):
                    buffer = VulkanBuffer(runtime,self.capacity,
                                          usage=vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT)
                    try:
                        mapped = vk.vkMapMemory(runtime.device,buffer.memory,0,buffer.size,0)
                    except BaseException:
                        buffer.close()
                        raise
                    self._slots.append(dict(buffer=buffer,mapped=mapped,packet=None,completion=None))
            except BaseException:
                self.close()
                raise
            runtime.retain(self)

    def require_open(self):
        self.runtime.require_open()
        if self.closed:
            raise RuntimeError('Upload ring is closed')

    def prepare(self, writes, *, wait=True):
        with self.runtime.lock:
            self.require_open()
            packed, total = [], 0
            for destination, offset, data in writes:
                view = destination if isinstance(destination,VulkanResource) else VulkanResource.buffer(destination)
                offset = index(offset)
                payload = memoryview(data).cast('B').tobytes()
                if view.kind != 'buffer' or view.owner.runtime is not self.runtime:
                    raise ValueError('Upload destinations must be same-runtime buffers')
                view.owner.require_open()
                if (view.offset+view.size > view.owner.byte_size
                        or offset < 0 or not payload or offset+len(payload) > view.size
                        or (view.offset+offset)%4 or len(payload)%4):
                    raise ValueError('Upload ranges must be bounded and four-byte aligned')
                if not view.owner.usage & vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT:
                    raise ValueError('Upload destination requires transfer-destination usage')
                target = view.byte_range(offset,len(payload))
                if any(target.handle == old.handle and max(target.offset,old.offset) <
                       min(target.offset+target.size,old.offset+old.size) for old,_ in packed):
                    raise ValueError('Upload destination ranges overlap')
                packed.append((target,payload))
                total += len(payload)
            if not packed or total > self.capacity:
                raise ValueError('Upload payload is empty or exceeds staging capacity')
            slot = None
            for candidate in self._slots:
                completion = candidate['completion']
                if candidate['packet'] is None and (completion is None or completion.poll()):
                    slot = candidate
                    break
            if slot is None:
                submitted = [s for s in self._slots if s['packet'] is None]
                if not wait or not submitted:
                    raise VulkanUploadBusy('Staging ring is busy; submit/cancel packets or wait for a completion')
                slot = submitted[0]
                slot['completion'].wait()
            passes, offset = [], 0
            for target,payload in packed:
                slot['mapped'][offset:offset+len(payload)] = payload
                passes.extend(buffer_copy_operation(slot['buffer'],target,[(offset,0,len(payload))]).passes)
                offset += len(payload)
            packet = VulkanUploadPacket(self,slot,passes,{target.owner for target,_ in packed})
            slot['packet'] = packet
            return packet

    def close(self):
        with self.runtime.lock:
            if self.closed:
                return
            for slot in self._slots:
                if slot['packet'] is not None:
                    slot['packet'].cancel()
                if slot['completion'] is not None:
                    slot['completion'].wait()
                vk.vkUnmapMemory(self.runtime.device,slot['buffer'].memory)
                slot['buffer'].close()
            self._slots.clear()
            self.closed = True
            self.runtime.release(self)

    def __enter__(self):
        self.require_open()
        return self

    def __exit__(self, *_exc):
        self.close()
