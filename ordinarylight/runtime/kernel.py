"""Small explicit compute pipeline for application-owned bindings."""

from collections import Counter
from functools import lru_cache

import vulkan as vk


@lru_cache(maxsize=64)
def compile_compute(source):
    """Compile complete GLSL to Vulkan 1.2 SPIR-V; cache by exact source."""
    from ..shaders.compiler import find_glsl_compiler, _compile_source

    compiler = find_glsl_compiler()
    if compiler is None:
        raise RuntimeError(
            "Install glslangValidator or glslc to compile application shaders"
        )
    return _compile_source(source, compiler)


class VulkanKernel:
    """Immutable set-0 buffer/image/sampler/AS descriptors and compute pipeline.

    Bindings borrow resources; those must remain open until kernel close.
    Pipeline execution is recorded through VulkanPass, including non-image
    workgroup sizes. Push constants are explicitly supplied bytes.
    """

    def __init__(self, runtime, spirv, bindings, *, push_constant_size=0):
        with runtime.lock:
            self._initialize(
                runtime, spirv, bindings, push_constant_size=push_constant_size
            )

    def _initialize(self, runtime, spirv, bindings, *, push_constant_size=0):
        runtime.require_open()
        self.runtime = runtime
        self.bindings = dict(bindings)
        self.push_constant_size = int(push_constant_size)
        if self.push_constant_size < 0 or self.push_constant_size % 4:
            raise ValueError(
                "push constant size must be a nonnegative multiple of four"
            )
        limits = vk.vkGetPhysicalDeviceProperties(runtime.physical_device).limits
        if self.push_constant_size > limits.maxPushConstantsSize:
            raise ValueError("push constants exceed runtime device limit")
        for binding, resource in self.bindings.items():
            if not isinstance(binding, int) or binding < 0:
                raise ValueError("binding indices must be nonnegative integers")
            if resource.owner.runtime is not runtime:
                raise ValueError("Kernel resources must belong to this runtime")
            resource.owner.require_open()
            descriptor = resource.descriptor or resource.kind
            expected = {
                "buffer": "buffer",
                "uniform_buffer": "buffer",
                "image": "image",
                "sampled_texture_2d": "image",
                "sampler": "sampler",
                "acceleration_structure": "acceleration_structure",
            }
            if descriptor not in expected or expected[descriptor] != resource.kind:
                raise ValueError("Invalid kernel descriptor kind")
            usage = {
                "buffer": vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                "uniform_buffer": vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                "image": vk.VK_IMAGE_USAGE_STORAGE_BIT,
                "sampled_texture_2d": vk.VK_IMAGE_USAGE_SAMPLED_BIT,
            }.get(descriptor)
            if usage is not None and not (
                getattr(resource.owner, "usage", usage) & usage
            ):
                raise ValueError("Allocation usage does not support descriptor")
            if resource.kind == "buffer":
                alignment = (
                    limits.minUniformBufferOffsetAlignment
                    if descriptor == "uniform_buffer"
                    else limits.minStorageBufferOffsetAlignment
                )
                if resource.offset % alignment:
                    raise ValueError(
                        "Buffer descriptor offset violates device alignment"
                    )
            if (
                descriptor == "uniform_buffer"
                and resource.size > limits.maxUniformBufferRange
            ):
                raise ValueError("Uniform buffer exceeds device descriptor range")
        self.closed = False
        self.module = self.layout = self.pipeline_layout = self.pipeline = self.pool = (
            None
        )
        self._retained_allocations = []
        runtime.retain(self)
        kinds = dict(
            buffer=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            image=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            uniform_buffer=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            sampled_texture_2d=vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
            sampler=vk.VK_DESCRIPTOR_TYPE_SAMPLER,
            acceleration_structure=vk.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
        )
        try:
            from .resources import VulkanBuffer, VulkanImage, VulkanSampler

            for owner in dict.fromkeys(r.owner for r in self.bindings.values()):
                if isinstance(owner, (VulkanBuffer, VulkanImage, VulkanSampler)):
                    owner.retain(self)
                    self._retained_allocations.append(owner)
            self.module = vk.vkCreateShaderModule(
                runtime.device,
                vk.VkShaderModuleCreateInfo(codeSize=len(spirv), pCode=spirv),
                None,
            )
            descriptors = [
                vk.VkDescriptorSetLayoutBinding(
                    binding=binding,
                    descriptorType=kinds[resource.descriptor or resource.kind],
                    descriptorCount=1,
                    stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                )
                for binding, resource in self.bindings.items()
            ]
            self.layout = vk.vkCreateDescriptorSetLayout(
                runtime.device,
                vk.VkDescriptorSetLayoutCreateInfo(
                    bindingCount=len(descriptors), pBindings=descriptors or None
                ),
                None,
            )
            ranges = (
                [
                    vk.VkPushConstantRange(
                        stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                        offset=0,
                        size=self.push_constant_size,
                    )
                ]
                if self.push_constant_size
                else []
            )
            self.pipeline_layout = vk.vkCreatePipelineLayout(
                runtime.device,
                vk.VkPipelineLayoutCreateInfo(
                    setLayoutCount=1,
                    pSetLayouts=[self.layout],
                    pushConstantRangeCount=len(ranges),
                    pPushConstantRanges=ranges or None,
                ),
                None,
            )
            # Keep the CFFI entry-point storage alive through pipeline creation.
            stage = vk.VkPipelineShaderStageCreateInfo(
                stage=vk.VK_SHADER_STAGE_COMPUTE_BIT, module=self.module, pName="main"
            )
            info = vk.VkComputePipelineCreateInfo(
                stage=stage, layout=self.pipeline_layout
            )
            self.pipeline = vk.vkCreateComputePipelines(
                runtime.device,
                runtime.pipeline_cache or vk.VK_NULL_HANDLE,
                1,
                [info],
                None,
            )[0]
            counts = Counter(
                kinds[resource.descriptor or resource.kind]
                for resource in self.bindings.values()
            )
            sizes = [
                vk.VkDescriptorPoolSize(type=kind, descriptorCount=count)
                for kind, count in counts.items()
            ]
            self.pool = vk.vkCreateDescriptorPool(
                runtime.device,
                vk.VkDescriptorPoolCreateInfo(
                    maxSets=1, poolSizeCount=len(sizes), pPoolSizes=sizes or None
                ),
                None,
            )
            self.descriptor = vk.vkAllocateDescriptorSets(
                runtime.device,
                vk.VkDescriptorSetAllocateInfo(
                    descriptorPool=self.pool,
                    descriptorSetCount=1,
                    pSetLayouts=[self.layout],
                ),
            )[0]
            writes = []
            for binding, resource in self.bindings.items():
                options = {}
                if resource.kind == "buffer":
                    options["pBufferInfo"] = [
                        vk.VkDescriptorBufferInfo(
                            buffer=resource.handle,
                            offset=resource.offset,
                            range=resource.size,
                        )
                    ]
                elif resource.kind == "sampler":
                    options["pImageInfo"] = [
                        vk.VkDescriptorImageInfo(sampler=resource.handle)
                    ]
                elif resource.kind == "image":
                    options["pImageInfo"] = [
                        vk.VkDescriptorImageInfo(
                            imageView=resource.owner.view,
                            imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
                        )
                    ]
                else:
                    options["pNext"] = vk.VkWriteDescriptorSetAccelerationStructureKHR(
                        accelerationStructureCount=1,
                        pAccelerationStructures=[resource.handle],
                    )
                writes.append(
                    vk.VkWriteDescriptorSet(
                        dstSet=self.descriptor,
                        dstBinding=binding,
                        descriptorCount=1,
                        descriptorType=kinds[resource.descriptor or resource.kind],
                        **options,
                    )
                )
            vk.vkUpdateDescriptorSets(
                runtime.device, len(writes), writes or None, 0, None
            )
        except Exception:
            self.close()
            raise

    def require_open(self):
        self.runtime.require_open()
        if self.closed:
            raise RuntimeError("Vulkan kernel is closed")
        for resource in self.bindings.values():
            resource.owner.require_open()

    def bind(self, command, push_constants=b""):
        self.require_open()
        if len(push_constants) != self.push_constant_size:
            raise ValueError("push constants have the wrong byte size")
        vk.vkCmdBindPipeline(command, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline)
        vk.vkCmdBindDescriptorSets(
            command,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.pipeline_layout,
            0,
            1,
            [self.descriptor],
            0,
            None,
        )
        if push_constants:
            raw = vk.ffi.new("uint8_t[]", push_constants)
            vk.vkCmdPushConstants(
                command,
                self.pipeline_layout,
                vk.VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                len(push_constants),
                raw,
            )

    def close(self):
        with self.runtime.lock:
            if self.closed:
                return
            vk.vkDeviceWaitIdle(self.runtime.device)
            for resource, destroy in (
                (self.pipeline, vk.vkDestroyPipeline),
                (self.pool, vk.vkDestroyDescriptorPool),
                (self.pipeline_layout, vk.vkDestroyPipelineLayout),
                (self.layout, vk.vkDestroyDescriptorSetLayout),
                (self.module, vk.vkDestroyShaderModule),
            ):
                if resource is not None:
                    destroy(self.runtime.device, resource, None)
            for owner in self._retained_allocations:
                owner.release(self)
            self._retained_allocations.clear()
            self.closed = True
            self.runtime.release(self)

    def __enter__(self):
        self.require_open()
        return self

    def __exit__(self, *_exc):
        self.close()
