"""Persistent reflected SPIR-V compute execution on an existing Vulkan device."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ._core import ComputeBuffer, ComputeStep, _description, _workgroups


@dataclass(frozen=True, slots=True)
class VulkanBufferView:
    """A non-owning typed view of a resident Vulkan compute allocation."""

    owner: Any
    name: str
    shape: tuple[int, ...]
    dtype: Any

    @property
    def device(self):
        return self.owner.device

    @property
    def buffer(self):
        self.owner._require_open()
        return self.owner._buffers[self.name][0]

    @property
    def byte_size(self):
        return self.owner._descriptions[self.name].byte_size

    @property
    def revision(self):
        """Monotonic dispatch generation of the underlying allocation."""
        self.owner._require_open()
        return self.owner.revision


class VulkanComputeSequence:
    """Execute ordered reflected SPIR-V kernels on a renderer's Vulkan queue.

    The supplied context owns the Vulkan instance, device, queue, and command
    pool. This sequence owns only its buffers, descriptor resources, shader
    modules, and compute pipelines.
    """

    def __init__(self, steps, resources: Mapping[str, ComputeBuffer | Any], *, context):
        try:
            import vulkan as vk
        except ImportError as error:
            raise RuntimeError(
                "Vulkan compute requires: pip install 'ordinarylight[vulkan]'"
            ) from error
        self.vk = vk
        self.context = context
        self.device = context.device
        self.physical_device = context.physical_device
        self.queue = context.queue
        self.command_pool = context.command_pool
        self.steps = tuple(steps)
        self.revision = 0
        if not self.steps or not all(isinstance(step, ComputeStep) for step in self.steps):
            raise TypeError("Vulkan compute sequence requires ComputeStep values")
        self._descriptions = {
            name: _description(value) for name, value in resources.items()
        }
        self._buffers = {}
        host = (
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        )
        usage = (
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
            | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
            | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
        )
        for name, description in self._descriptions.items():
            self._buffers[name] = self._allocate_buffer(
                description.byte_size, usage, host, description.payload(),
            )
        self._prepared = []
        for step in self.steps:
            program = step.program
            if program.target != "spirv" or program.reflection.stage != "compute":
                raise ValueError("Vulkan compute requires SPIR-V compute programs")
            reflected = tuple(program.reflection.resources)
            aliases = dict(step.resources or {item.name: item.name for item in reflected})
            expected = {item.name for item in reflected}
            if set(aliases) != expected:
                raise ValueError(
                    f"compute step resource mismatch: missing={sorted(expected-set(aliases))}, "
                    f"extra={sorted(set(aliases)-expected)}"
                )
            missing = set(aliases.values()) - set(self._buffers)
            if missing:
                raise ValueError(f"compute step references missing allocations: {sorted(missing)}")
            prepared = self._prepare_step(program, reflected, aliases)
            self._prepared.append((*prepared, _workgroups(step.workgroups)))
        self._prepared = tuple(self._prepared)
        self.closed = False

    def _memory_type(self, bits, flags):
        properties = self.vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device)
        for index in range(properties.memoryTypeCount):
            item = properties.memoryTypes[index]
            if bits & (1 << index) and item.propertyFlags & flags == flags:
                return index
        raise RuntimeError("no compatible Vulkan memory type")

    def _allocate_buffer(self, size, usage, flags, payload):
        vk = self.vk
        buffer = vk.vkCreateBuffer(self.device, vk.VkBufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            size=size, usage=usage, sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        ), None)
        requirements = vk.vkGetBufferMemoryRequirements(self.device, buffer)
        memory = vk.vkAllocateMemory(self.device, vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=requirements.size,
            memoryTypeIndex=self._memory_type(requirements.memoryTypeBits, flags),
        ), None)
        vk.vkBindBufferMemory(self.device, buffer, memory, 0)
        mapped = vk.vkMapMemory(self.device, memory, 0, size, 0)
        vk.ffi.memmove(mapped, payload, size)
        vk.vkUnmapMemory(self.device, memory)
        return buffer, memory

    def _prepare_step(self, program, reflected, aliases):
        vk = self.vk
        bindings = [vk.VkDescriptorSetLayoutBinding(
            binding=item.binding,
            descriptorType=(
                vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                if item.kind == "uniform_buffer"
                else vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
            ),
            descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
        ) for item in reflected]
        layout = vk.vkCreateDescriptorSetLayout(
            self.device, vk.VkDescriptorSetLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                bindingCount=len(bindings), pBindings=bindings,
            ), None,
        )
        pipeline_layout = vk.vkCreatePipelineLayout(
            self.device, vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=1 if bindings else 0,
                pSetLayouts=[layout] if bindings else None,
            ), None,
        )
        module = vk.vkCreateShaderModule(self.device, vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(program.binary), pCode=program.binary,
        ), None)
        # Keep the stage object alive explicitly. Some CFFI Vulkan bindings do
        # not retain the nested entry-point string when it is constructed
        # inline with VkComputePipelineCreateInfo.
        stage = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT, module=module, pName="main",
        )
        create_info = vk.VkComputePipelineCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            stage=stage, layout=pipeline_layout,
        )
        pipeline = vk.vkCreateComputePipelines(
            self.device, vk.VK_NULL_HANDLE, 1, [create_info], None,
        )[0]
        counts = {}
        for item in reflected:
            descriptor_type = (
                vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                if item.kind == "uniform_buffer"
                else vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
            )
            counts[descriptor_type] = counts.get(descriptor_type, 0) + 1
        pool = vk.vkCreateDescriptorPool(self.device, vk.VkDescriptorPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            maxSets=1, poolSizeCount=len(counts),
            pPoolSizes=[vk.VkDescriptorPoolSize(type=kind, descriptorCount=count) for kind, count in counts.items()],
        ), None)
        descriptor = vk.vkAllocateDescriptorSets(
            self.device, vk.VkDescriptorSetAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptorPool=pool, descriptorSetCount=1,
                pSetLayouts=[layout],
            ),
        )[0]
        writes = []
        for item in reflected:
            name = aliases[item.name]
            buffer, _memory = self._buffers[name]
            kind = (
                vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                if item.kind == "uniform_buffer"
                else vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
            )
            writes.append(vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor, dstBinding=item.binding,
                descriptorCount=1, descriptorType=kind,
                pBufferInfo=[vk.VkDescriptorBufferInfo(
                    buffer=buffer, offset=0,
                    range=self._descriptions[name].byte_size,
                )],
            ))
        vk.vkUpdateDescriptorSets(self.device, len(writes), writes, 0, None)
        return module, layout, pipeline_layout, pipeline, pool, descriptor

    def update(self, name, data):
        self._require_open()
        if name not in self._buffers:
            raise KeyError(name)
        description = self._descriptions[name]
        payload = ComputeBuffer(data, dtype=description.dtype).payload()
        if len(payload) != description.byte_size:
            raise ValueError(f"update for {name!r} has the wrong byte size")
        memory = self._buffers[name][1]
        mapped = self.vk.vkMapMemory(self.device, memory, 0, len(payload), 0)
        self.vk.ffi.memmove(mapped, payload, len(payload))
        self.vk.vkUnmapMemory(self.device, memory)

    def dispatch(self):
        self._require_open()
        vk = self.vk
        command = vk.vkAllocateCommandBuffers(
            self.device, vk.VkCommandBufferAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                commandPool=self.command_pool,
                level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1,
            ),
        )[0]
        vk.vkBeginCommandBuffer(command, vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        ))
        barriers = [vk.VkBufferMemoryBarrier(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
            dstAccessMask=(vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT),
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            buffer=buffer, offset=0,
            size=self._descriptions[name].byte_size,
        ) for name, (buffer, _memory) in self._buffers.items()]
        for index, (_module, _layout, pipeline_layout, pipeline, _pool, descriptor, groups) in enumerate(self._prepared):
            vk.vkCmdBindPipeline(command, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
            vk.vkCmdBindDescriptorSets(
                command, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout,
                0, 1, [descriptor], 0, None,
            )
            vk.vkCmdDispatch(command, *groups)
            if index + 1 < len(self._prepared):
                vk.vkCmdPipelineBarrier(
                    command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                    0, None, len(barriers), barriers, 0, None,
                )
        vk.vkEndCommandBuffer(command)
        fence = vk.vkCreateFence(self.device, vk.VkFenceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        ), None)
        try:
            vk.vkQueueSubmit(self.queue, 1, [vk.VkSubmitInfo(
                sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                commandBufferCount=1, pCommandBuffers=[command],
            )], fence)
            vk.vkWaitForFences(self.device, 1, [fence], vk.VK_TRUE, (1 << 64) - 1)
            self.revision += 1
        finally:
            vk.vkDestroyFence(self.device, fence, None)
            vk.vkFreeCommandBuffers(self.device, self.command_pool, 1, [command])

    def read(self, name, *, dtype=None, shape=None):
        self._require_open()
        if name not in self._buffers:
            raise KeyError(name)
        description = self._descriptions[name]
        resolved_dtype = np.dtype(description.dtype if dtype is None else dtype)
        resolved_shape = description.shape if shape is None else tuple(shape)
        memory = self._buffers[name][1]
        mapped = self.vk.vkMapMemory(self.device, memory, 0, description.byte_size, 0)
        try:
            raw = bytes(mapped[:description.byte_size])
        finally:
            self.vk.vkUnmapMemory(self.device, memory)
        result = np.frombuffer(raw, dtype=resolved_dtype).copy()
        return result.reshape(resolved_shape) if resolved_shape else result

    def buffer_view(self, name):
        self._require_open()
        if name not in self._buffers:
            raise KeyError(name)
        description = self._descriptions[name]
        return VulkanBufferView(self, name, tuple(description.shape), np.dtype(description.dtype))

    def _require_open(self):
        if self.closed:
            raise RuntimeError("Vulkan compute sequence is closed")

    def close(self):
        if self.closed:
            return
        vk = self.vk
        vk.vkDeviceWaitIdle(self.device)
        for module, layout, pipeline_layout, pipeline, pool, _descriptor, _groups in self._prepared:
            vk.vkDestroyPipeline(self.device, pipeline, None)
            vk.vkDestroyShaderModule(self.device, module, None)
            vk.vkDestroyDescriptorPool(self.device, pool, None)
            vk.vkDestroyPipelineLayout(self.device, pipeline_layout, None)
            vk.vkDestroyDescriptorSetLayout(self.device, layout, None)
        for buffer, memory in self._buffers.values():
            vk.vkDestroyBuffer(self.device, buffer, None)
            vk.vkFreeMemory(self.device, memory, None)
        self._prepared = ()
        self._buffers.clear()
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


__all__ = ["VulkanBufferView", "VulkanComputeSequence"]
