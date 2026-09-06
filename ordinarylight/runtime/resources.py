"""Explicitly owned Vulkan allocations and fence-backed submissions."""

from operator import index

import vulkan as vk


class VulkanCompletion:
    """A submission and its retained resources; wait releases command storage."""

    def __init__(self, runtime, command, fence, resources):
        self.runtime, self.command, self.fence = runtime, command, fence
        self.resources = tuple(resources)
        self.complete = False
        runtime.retain(self)

    def poll(self):
        with self.runtime.lock:
            if self.complete:
                return True
            try:
                vk.vkGetFenceStatus(self.runtime.device, self.fence)
            except vk.VkNotReady:
                return False
            self._retire()
            return True

    def wait(self):
        with self.runtime.lock:
            if not self.complete:
                vk.vkWaitForFences(
                    self.runtime.device, 1, [self.fence], vk.VK_TRUE, (1 << 64) - 1
                )
                self._retire()
        return self

    def _retire(self):
        vk.vkDestroyFence(self.runtime.device, self.fence, None)
        vk.vkFreeCommandBuffers(
            self.runtime.device, self.runtime.command_pool, 1, [self.command]
        )
        self.complete = True
        self.resources = ()
        self.runtime.release(self)

    close = wait

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.wait()


def submit(
    runtime,
    recorder,
    *,
    resources=(),
    after=(),
    wait_semaphores=(),
    signal_semaphores=(),
):
    """Submit on one queue. Dependencies use queue order and GPU barriers.

    All completion tokens originate on this queue; no cross-queue synchronization
    is implied. Fence polling retires finished command storage without waiting.
    """
    with runtime.lock:
        runtime.require_open()
        waits, signals = tuple(wait_semaphores), tuple(signal_semaphores)
        for semaphore, stage in waits:
            if (
                not isinstance(semaphore, VulkanSemaphore)
                or semaphore.runtime is not runtime
                or not stage
            ):
                raise ValueError(
                    "Semaphore waits require same-runtime semaphores and stage masks"
                )
        for semaphore in signals:
            if (
                not isinstance(semaphore, VulkanSemaphore)
                or semaphore.runtime is not runtime
            ):
                raise ValueError("Semaphore signals require same-runtime semaphores")
        if (
            len({s for s, _ in waits}) != len(waits)
            or len(set(signals)) != len(signals)
            or {s for s, _ in waits} & set(signals)
        ):
            raise ValueError("Binary semaphore uses must be unique within a submission")
        resources = tuple(resources) + tuple(s for s, _ in waits) + signals
        for dependency in after:
            if (
                not isinstance(dependency, VulkanCompletion)
                or dependency.runtime is not runtime
            ):
                raise ValueError("Completion dependencies must belong to this runtime")
        for consumer in tuple(runtime._consumers):
            if isinstance(consumer, VulkanCompletion):
                consumer.poll()
        for resource in resources:
            if resource.runtime is not runtime:
                raise ValueError("Submitted resources must belong to this runtime")
            resource.require_open()
        command = vk.vkAllocateCommandBuffers(
            runtime.device,
            vk.VkCommandBufferAllocateInfo(
                commandPool=runtime.command_pool,
                level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1,
            ),
        )[0]
        fence = None
        try:
            vk.vkBeginCommandBuffer(
                command,
                vk.VkCommandBufferBeginInfo(
                    flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT
                ),
            )
            vk.vkCmdPipelineBarrier(
                command,
                vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | vk.VK_PIPELINE_STAGE_HOST_BIT,
                vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                0,
                1,
                [
                    vk.VkMemoryBarrier(
                        srcAccessMask=vk.VK_ACCESS_MEMORY_READ_BIT
                        | vk.VK_ACCESS_MEMORY_WRITE_BIT
                        | vk.VK_ACCESS_HOST_WRITE_BIT,
                        dstAccessMask=vk.VK_ACCESS_MEMORY_READ_BIT
                        | vk.VK_ACCESS_MEMORY_WRITE_BIT,
                    )
                ],
                0,
                None,
                0,
                None,
            )
            recorder(command)
            vk.vkEndCommandBuffer(command)
            fence = vk.vkCreateFence(runtime.device, vk.VkFenceCreateInfo(), None)
            vk.vkQueueSubmit(
                runtime.queue,
                1,
                [
                    vk.VkSubmitInfo(
                        commandBufferCount=1,
                        pCommandBuffers=[command],
                        waitSemaphoreCount=len(waits),
                        pWaitSemaphores=[s.handle for s, _ in waits] or None,
                        pWaitDstStageMask=[stage for _, stage in waits] or None,
                        signalSemaphoreCount=len(signals),
                        pSignalSemaphores=[s.handle for s in signals] or None,
                    )
                ],
                fence,
            )
        except Exception:
            if fence is not None:
                vk.vkDestroyFence(runtime.device, fence, None)
            vk.vkFreeCommandBuffers(runtime.device, runtime.command_pool, 1, [command])
            raise
        return VulkanCompletion(runtime, command, fence, resources)


class _Allocation:
    def retain(self, consumer):
        with self.runtime.lock:
            self.require_open()
            self._borrowers.add(consumer)

    def release(self, consumer):
        with self.runtime.lock:
            self._borrowers.discard(consumer)

    def require_open(self):
        self.runtime.require_open()
        if self.closed:
            raise RuntimeError("Vulkan allocation is closed")

    def __enter__(self):
        self.require_open()
        return self

    def __exit__(self, *_exc):
        self.close()


class VulkanBuffer(_Allocation):
    """Persistent storage with explicit host or device-local allocation policy.

    Host memory is the compatibility default. Device-local upload/read use
    synchronized staging; GPU consumers access the allocation directly.
    """

    def __init__(
        self,
        runtime,
        size,
        *,
        usage=None,
        data=None,
        device_address=False,
        memory="host",
    ):
        runtime.require_open()
        if memory not in ("host", "device"):
            raise ValueError("Buffer memory must be host or device")
        self.memory_kind = memory
        self.runtime = runtime
        self.size = index(size)
        if self.size <= 0:
            raise ValueError("buffer size must be positive")
        self.usage = (
            usage
            if usage is not None
            else (
                vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                | vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
            )
        )
        if memory == "device":
            self.usage |= (
                vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
            )
        if device_address:
            self.usage |= 0x00020000
        payload = None if data is None else memoryview(data).cast("B")
        if payload is not None and len(payload) > self.size:
            raise ValueError("payload exceeds buffer size")
        self.buffer = vk.vkCreateBuffer(
            runtime.device,
            vk.VkBufferCreateInfo(
                size=self.size,
                usage=self.usage,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            ),
            None,
        )
        self.memory = None
        try:
            requirements = vk.vkGetBufferMemoryRequirements(runtime.device, self.buffer)
            required = (
                vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
                if memory == "device"
                else vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            )
            memory_index = runtime.memory_type(requirements.memoryTypeBits, required)
            self.memory_flags = int(
                runtime.memory_properties.memoryTypes[memory_index].propertyFlags
            )
            flags = vk.VkMemoryAllocateFlagsInfo(flags=2) if device_address else None
            self.memory = vk.vkAllocateMemory(
                runtime.device,
                vk.VkMemoryAllocateInfo(
                    pNext=flags,
                    allocationSize=requirements.size,
                    memoryTypeIndex=memory_index,
                ),
                None,
            )
            vk.vkBindBufferMemory(runtime.device, self.buffer, self.memory, 0)
        except Exception:
            vk.vkDestroyBuffer(runtime.device, self.buffer, None)
            if self.memory is not None:
                vk.vkFreeMemory(runtime.device, self.memory, None)
            raise
        self.closed = False
        self._borrowers = set()
        runtime.retain(self)
        if payload is not None:
            try:
                self.upload(payload)
            except Exception:
                self.close()
                raise

    @property
    def byte_size(self):
        return self.size

    @property
    def device(self):
        return self.runtime.device

    def upload(self, data, *, offset=0):
        payload = memoryview(data).cast("B")
        offset = index(offset)
        if offset < 0 or offset + len(payload) > self.size:
            raise ValueError("upload exceeds buffer bounds")
        with self.runtime.lock:
            self.require_open()
            if self.memory_kind == "device":
                if len(payload):
                    with VulkanBuffer(
                        self.runtime, len(payload), data=payload
                    ) as staging:
                        self._copy(
                            staging, self, len(payload), destination_offset=offset
                        )
                return
            vk.vkDeviceWaitIdle(self.device)
            mapped = vk.vkMapMemory(self.device, self.memory, 0, self.size, 0)
            try:
                mapped[offset : offset + len(payload)] = payload
            finally:
                vk.vkUnmapMemory(self.device, self.memory)

    def read(self):
        with self.runtime.lock:
            self.require_open()
            if self.memory_kind == "device":
                with VulkanBuffer(self.runtime, self.size) as staging:
                    self._copy(self, staging, self.size)
                    return staging.read()
            vk.vkDeviceWaitIdle(self.device)
            mapped = vk.vkMapMemory(self.device, self.memory, 0, self.size, 0)
            try:
                return bytes(mapped[: self.size])
            finally:
                vk.vkUnmapMemory(self.device, self.memory)

    def _copy(self, source, destination, size, *, destination_offset=0):
        def record(command):
            vk.vkCmdCopyBuffer(
                command,
                source.buffer,
                destination.buffer,
                1,
                [vk.VkBufferCopy(dstOffset=destination_offset, size=size)],
            )
            vk.vkCmdPipelineBarrier(
                command,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                vk.VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | vk.VK_PIPELINE_STAGE_HOST_BIT,
                0,
                1,
                [
                    vk.VkMemoryBarrier(
                        srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                        dstAccessMask=vk.VK_ACCESS_MEMORY_READ_BIT
                        | vk.VK_ACCESS_HOST_READ_BIT,
                    )
                ],
                0,
                None,
                0,
                None,
            )

        submit(self.runtime, record, resources=(source, destination)).wait()

    def close(self):
        with self.runtime.lock:
            if self.closed:
                return
            if self._borrowers:
                raise RuntimeError("Close allocation borrowers before the allocation")
            vk.vkDeviceWaitIdle(self.device)
            vk.vkDestroyBuffer(self.device, self.buffer, None)
            vk.vkFreeMemory(self.device, self.memory, None)
            self.closed = True
            self.runtime.release(self)


class VulkanImage(_Allocation):
    """Persistent single-mip 2D color storage image with tracked queue layout."""

    def __init__(
        self, runtime, width, height, *, format=None, usage=None, exportable=False
    ):
        runtime.require_open()
        self.runtime = runtime
        self.width, self.height = index(width), index(height)
        if min(self.width, self.height) <= 0:
            raise ValueError("image extent must be positive")
        self.format = vk.VK_FORMAT_R32G32B32A32_SFLOAT if format is None else format
        self.usage = (
            usage
            if usage is not None
            else (
                vk.VK_IMAGE_USAGE_STORAGE_BIT
                | vk.VK_IMAGE_USAGE_SAMPLED_BIT
                | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                | vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT
            )
        )
        self.exportable = bool(exportable)
        if exportable and not runtime.capabilities.external_memory:
            raise ValueError("Runtime did not enable external memory")
        self.layout = vk.VK_IMAGE_LAYOUT_UNDEFINED
        self.image = vk.vkCreateImage(
            runtime.device,
            vk.VkImageCreateInfo(
                pNext=(
                    vk.VkExternalMemoryImageCreateInfo(
                        handleTypes=vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
                    )
                    if exportable
                    else None
                ),
                imageType=vk.VK_IMAGE_TYPE_2D,
                format=self.format,
                extent=vk.VkExtent3D(self.width, self.height, 1),
                mipLevels=1,
                arrayLayers=1,
                samples=vk.VK_SAMPLE_COUNT_1_BIT,
                tiling=vk.VK_IMAGE_TILING_OPTIMAL,
                usage=self.usage,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
                initialLayout=self.layout,
            ),
            None,
        )
        self.memory = self.view = None
        try:
            requirements = vk.vkGetImageMemoryRequirements(runtime.device, self.image)
            self.memory_size = int(requirements.size)
            external = (
                vk.VkMemoryDedicatedAllocateInfo(
                    pNext=vk.VkExportMemoryAllocateInfo(
                        handleTypes=vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
                    ),
                    image=self.image,
                    buffer=vk.VK_NULL_HANDLE,
                )
                if exportable
                else None
            )
            self.memory = vk.vkAllocateMemory(
                runtime.device,
                vk.VkMemoryAllocateInfo(
                    pNext=external,
                    allocationSize=requirements.size,
                    memoryTypeIndex=runtime.memory_type(
                        requirements.memoryTypeBits,
                        vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    ),
                ),
                None,
            )
            vk.vkBindImageMemory(runtime.device, self.image, self.memory, 0)
            self.view = vk.vkCreateImageView(
                runtime.device,
                vk.VkImageViewCreateInfo(
                    image=self.image,
                    viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                    format=self.format,
                    subresourceRange=vk.VkImageSubresourceRange(
                        aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                        levelCount=1,
                        layerCount=1,
                    ),
                ),
                None,
            )
        except Exception:
            vk.vkDestroyImage(runtime.device, self.image, None)
            if self.memory is not None:
                vk.vkFreeMemory(runtime.device, self.memory, None)
            raise
        self.closed = False
        self._borrowers = set()
        runtime.retain(self)

    def close(self):
        with self.runtime.lock:
            if self.closed:
                return
            if self._borrowers:
                raise RuntimeError("Close allocation borrowers before the allocation")
            vk.vkDeviceWaitIdle(self.runtime.device)
            vk.vkDestroyImageView(self.runtime.device, self.view, None)
            vk.vkDestroyImage(self.runtime.device, self.image, None)
            vk.vkFreeMemory(self.runtime.device, self.memory, None)
            self.closed = True
            self.runtime.release(self)


class VulkanSemaphore:
    """Owned binary semaphore for external acquisition/presentation boundaries.

    The application must pair each signal with exactly one wait before reuse.
    Ordinary single-queue graph dependencies do not require these objects.
    """

    def __init__(self, runtime):
        with runtime.lock:
            runtime.require_open()
            self.runtime = runtime
            self.handle = vk.vkCreateSemaphore(
                runtime.device, vk.VkSemaphoreCreateInfo(), None
            )
            self.closed = False
            runtime.retain(self)

    def require_open(self):
        self.runtime.require_open()
        if self.closed:
            raise RuntimeError("Semaphore is closed")

    def close(self):
        with self.runtime.lock:
            if not self.closed:
                vk.vkDeviceWaitIdle(self.runtime.device)
                vk.vkDestroySemaphore(self.runtime.device, self.handle, None)
                self.closed = True
                self.runtime.release(self)

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


class VulkanSampler(_Allocation):
    """Owned single-mip color sampler; kernels borrow it like image allocations."""

    def __init__(self, runtime, *, filter="nearest", address="clamp"):
        filters = {"nearest": vk.VK_FILTER_NEAREST, "linear": vk.VK_FILTER_LINEAR}
        addresses = {
            "clamp": vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            "repeat": vk.VK_SAMPLER_ADDRESS_MODE_REPEAT,
        }
        if filter not in filters or address not in addresses:
            raise ValueError("Unsupported sampler filter or address mode")
        with runtime.lock:
            runtime.require_open()
            self.runtime = runtime
            self.handle = vk.vkCreateSampler(
                runtime.device,
                vk.VkSamplerCreateInfo(
                    magFilter=filters[filter],
                    minFilter=filters[filter],
                    mipmapMode=vk.VK_SAMPLER_MIPMAP_MODE_NEAREST,
                    addressModeU=addresses[address],
                    addressModeV=addresses[address],
                    addressModeW=addresses[address],
                    maxLod=0.0,
                ),
                None,
            )
            self.closed = False
            self._borrowers = set()
            runtime.retain(self)

    def close(self):
        with self.runtime.lock:
            if self.closed:
                return
            if self._borrowers:
                raise RuntimeError("Close allocation borrowers before the allocation")
            vk.vkDeviceWaitIdle(self.runtime.device)
            vk.vkDestroySampler(self.runtime.device, self.handle, None)
            self.closed = True
            self.runtime.release(self)
