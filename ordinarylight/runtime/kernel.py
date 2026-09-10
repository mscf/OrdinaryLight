"""Small explicit compute pipeline for application-owned bindings."""

from collections import Counter
from functools import lru_cache

import vulkan as vk
from ordinarylight.runtime.lifecycle import timed_call


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
    Optional image_arrays maps additional bindings to nonempty tuples of storage
    image resources. sampled_image_arrays maps bindings to nonempty tuples of
    (sampled-image resource, sampler resource) pairs for combined samplers.
    The shader sampler dimension must match the supplied image view. Both images
    and samplers are validated and retained like scalar resources; descriptors
    use GENERAL layout by default; sampled_image_layouts can select read-only
    layout per combined-array binding. Graph image uses must match.
    Pipeline execution is recorded through VulkanPass, including non-image
    workgroup sizes. Push constants are explicitly supplied bytes. An optional
    material_resources bundle supplies and retains descriptor set 1; graph passes
    must include its uses (shade_operation does this automatically).
    """

    def __init__(
        self,
        runtime,
        spirv,
        bindings,
        *,
        push_constant_size=0,
        image_arrays=None,
        sampled_image_arrays=None,
        sampled_image_layouts=None,
        material_resources=None,
    ):
        if spirv is None:
            raise TypeError("VulkanKernel requires SPIR-V")
        with runtime.lock:
            self._initialize(
                runtime,
                spirv,
                bindings,
                push_constant_size=push_constant_size,
                image_arrays=image_arrays,
                sampled_image_arrays=sampled_image_arrays,
                sampled_image_layouts=sampled_image_layouts,
                material_resources=material_resources,
            )

    def _initialize(
        self,
        runtime,
        spirv,
        bindings,
        *,
        push_constant_size=0,
        image_arrays=None,
        sampled_image_arrays=None,
        sampled_image_layouts=None,
        material_resources=None,
    ):
        runtime.require_open()
        self.runtime = runtime
        self.material_resources = material_resources
        if material_resources is not None:
            material_resources.require_open()
            if material_resources.runtime is not runtime:
                raise ValueError("Material resources must belong to kernel runtime")
        self.bindings = dict(bindings)
        self.image_arrays = {
            key: tuple(values) for key, values in (image_arrays or {}).items()
        }
        self.sampled_image_arrays = {
            key: tuple(tuple(pair) for pair in values)
            for key, values in (sampled_image_arrays or {}).items()
        }
        if self.sampled_image_arrays.keys() & (
            self.bindings.keys() | self.image_arrays.keys()
        ):
            raise ValueError("Scalar and array bindings must not overlap")
        self.sampled_image_layouts = dict(sampled_image_layouts or {})
        if self.sampled_image_layouts.keys() - self.sampled_image_arrays.keys():
            raise ValueError("Sampled layouts require matching array bindings")
        if any(
            layout
            not in (
                vk.VK_IMAGE_LAYOUT_GENERAL,
                vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            )
            for layout in self.sampled_image_layouts.values()
        ):
            raise ValueError("Invalid sampled array layout")
        sampled_bindings = []
        for binding, values in self.sampled_image_arrays.items():
            if not values:
                raise ValueError("Sampled arrays must not be empty")
            for pair in values:
                if (
                    len(pair) != 2
                    or pair[0].kind != "image"
                    or pair[0].descriptor != "sampled_texture_2d"
                    or pair[1].kind != "sampler"
                ):
                    raise ValueError(
                        "Sampled arrays require (sampled image, sampler) resource pairs"
                    )
                sampled_bindings.extend((binding, resource) for resource in pair)
        if self.bindings.keys() & self.image_arrays.keys():
            raise ValueError("Scalar and array bindings must not overlap")
        for values in self.image_arrays.values():
            if not values or any(
                r.kind != "image" or r.descriptor not in (None, "image") for r in values
            ):
                raise ValueError(
                    "Image arrays require nonempty storage image resources"
                )
        all_bindings = list(self.bindings.items()) + [
            (binding, resource)
            for binding, values in self.image_arrays.items()
            for resource in values
        ]
        self.push_constant_size = int(push_constant_size)
        if self.push_constant_size < 0 or self.push_constant_size % 4:
            raise ValueError(
                "push constant size must be a nonnegative multiple of four"
            )
        limits = vk.vkGetPhysicalDeviceProperties(runtime.physical_device).limits
        if self.push_constant_size > limits.maxPushConstantsSize:
            raise ValueError("push constants exceed runtime device limit")
        for binding, resource in all_bindings + sampled_bindings:
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
        sampler_count = len(sampled_bindings) // 2 + sum(
            resource.kind == "sampler" for _, resource in all_bindings
        )
        sampled_count = len(sampled_bindings) // 2 + sum(
            resource.descriptor == "sampled_texture_2d" for _, resource in all_bindings
        )
        if sampler_count > min(
            limits.maxPerStageDescriptorSamplers, limits.maxDescriptorSetSamplers
        ):
            raise ValueError("Kernel exceeds device sampler descriptor limits")
        if sampled_count > min(
            limits.maxPerStageDescriptorSampledImages,
            limits.maxDescriptorSetSampledImages,
        ):
            raise ValueError("Kernel exceeds device sampled-image descriptor limits")
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
            for owner in dict.fromkeys(
                r.owner for _, r in all_bindings + sampled_bindings
            ):
                if callable(getattr(owner, "retain", None)) and callable(
                    getattr(owner, "release", None)
                ):
                    owner.retain(self)
                    self._retained_allocations.append(owner)
            if material_resources is not None:
                material_resources.retain(self)
                self._retained_allocations.append(material_resources)
            if spirv is not None:
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
            ] + [
                vk.VkDescriptorSetLayoutBinding(
                    binding=binding,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    descriptorCount=len(values),
                    stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                )
                for binding, values in self.image_arrays.items()
            ]
            descriptors += [
                vk.VkDescriptorSetLayoutBinding(
                    binding=binding,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    descriptorCount=len(values),
                    stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                )
                for binding, values in self.sampled_image_arrays.items()
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
                    setLayoutCount=1 + int(material_resources is not None),
                    pSetLayouts=[self.layout]
                    + (
                        [material_resources.layout]
                        if material_resources is not None
                        else []
                    ),
                    pushConstantRangeCount=len(ranges),
                    pPushConstantRanges=ranges or None,
                ),
                None,
            )
            if spirv is not None:
                # Keep the CFFI entry-point storage alive through pipeline creation.
                stage = vk.VkPipelineShaderStageCreateInfo(
                    stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                    module=self.module,
                    pName="main",
                )
                info = vk.VkComputePipelineCreateInfo(
                    stage=stage, layout=self.pipeline_layout
                )
                self.pipeline = timed_call(
                    "compute_pipeline_create", vk.vkCreateComputePipelines,
                    runtime.device,
                    runtime.pipeline_cache or vk.VK_NULL_HANDLE,
                    1,
                    [info],
                    None,
                )[0]
            counts = Counter(
                kinds[resource.descriptor or resource.kind]
                for _, resource in all_bindings
            )
            if self.sampled_image_arrays:
                counts[vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER] += sum(
                    len(values) for values in self.sampled_image_arrays.values()
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
            for binding, values in self.image_arrays.items():
                writes.append(
                    vk.VkWriteDescriptorSet(
                        dstSet=self.descriptor,
                        dstBinding=binding,
                        descriptorCount=len(values),
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                        pImageInfo=[
                            vk.VkDescriptorImageInfo(
                                imageView=r.owner.view,
                                imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
                            )
                            for r in values
                        ],
                    )
                )
            for binding, values in self.sampled_image_arrays.items():
                writes.append(
                    vk.VkWriteDescriptorSet(
                        dstSet=self.descriptor,
                        dstBinding=binding,
                        descriptorCount=len(values),
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                        pImageInfo=[
                            vk.VkDescriptorImageInfo(
                                imageView=image.owner.view,
                                sampler=sampler.handle,
                                imageLayout=self.sampled_image_layouts.get(
                                    binding, vk.VK_IMAGE_LAYOUT_GENERAL
                                ),
                            )
                            for image, sampler in values
                        ],
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
        if self.material_resources is not None:
            self.material_resources.require_open()
        for resource in (
            *self.bindings.values(),
            *(r for values in self.image_arrays.values() for r in values),
            *(
                r
                for values in self.sampled_image_arrays.values()
                for pair in values
                for r in pair
            ),
        ):
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

        if self.material_resources is not None:
            self.material_resources.bind_graph(command, self.pipeline_layout)

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


class VulkanDescriptorSet(VulkanKernel):
    """Internal immutable descriptors using the kernel's resource validation."""

    def __init__(
        self,
        runtime,
        bindings,
        *,
        image_arrays=None,
        sampled_image_arrays=None,
        sampled_image_layouts=None,
    ):
        with runtime.lock:
            self._initialize(
                runtime,
                None,
                bindings,
                image_arrays=image_arrays,
                sampled_image_arrays=sampled_image_arrays,
                sampled_image_layouts=sampled_image_layouts,
            )

    def bind(self, command, pipeline_layout, *, set_index=0):
        self.require_open()
        vk.vkCmdBindDescriptorSets(
            command,
            vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline_layout,
            set_index,
            1,
            [self.descriptor],
            0,
            None,
        )
