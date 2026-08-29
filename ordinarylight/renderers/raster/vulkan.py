"""Native Vulkan offscreen raster renderer."""

from __future__ import annotations

import numpy as np

from ...capabilities import RendererCapabilities
from ...raster import (
    RasterConfig, RasterMesh, RasterPostProcessor, RasterState,
    CAMERA_DTYPE, camera_matrix, create_raster_pipeline,
    rasterize_geometry_products, scene_mesh,
)
from ..base import RendererImplementation, RendererImplementationInfo


class VulkanRasterRenderer(RendererImplementation):
    """Draw Ordinary Shade SPIR-V programs using a native Vulkan graphics queue.

    This implementation supports one vec2 vertex stream, an
    optional uint32 index stream, and an offscreen linear RGBA16F attachment.
    It establishes the native graphics pipeline/lifetime architecture without
    coupling it to the path tracer.
    """

    implementation = RendererImplementationInfo(
        name="vulkan-raster", family="raster", graphics_api="vulkan",
    )

    def __init__(
        self, program, *, config=None, state=None, device_name=None,
        instance=None, surface=None,
    ):
        try:
            import vulkan as vk
        except ImportError as error:
            raise RuntimeError(
                "Vulkan rasterization requires: pip install 'ordinarylight[vulkan]'"
            ) from error
        if program.vertex.target != "spirv" or program.fragment.target != "spirv":
            raise ValueError("VulkanRasterRenderer requires a SPIR-V RasterProgram")
        self.vk = vk
        self.program = program
        if config is not None and state is not None:
            raise TypeError("pass config or state, not both")
        self.config = config or RasterConfig(state=state or RasterState())
        self.state = self.config.state
        self.pipeline_graph = create_raster_pipeline(self.config)
        self._post = RasterPostProcessor(self.config)
        self.available_outputs = ("color", "depth", "normal", "object_id")
        self.last_timings = {}
        app = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="Ordinary Light Raster", applicationVersion=1,
            pEngineName="Ordinary Light", engineVersion=1,
            apiVersion=vk.VK_MAKE_VERSION(1, 1, 0),
        )
        if (instance is None) != (surface is None):
            raise ValueError("instance and surface must be provided together")
        self._owns_instance = instance is None
        self.surface = surface
        self.instance = instance or vk.vkCreateInstance(vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app,
        ), None)
        self._get_surface_support = (
            vk.vkGetInstanceProcAddr(
                self.instance, "vkGetPhysicalDeviceSurfaceSupportKHR",
            ) if self.surface is not None else None
        )
        candidates = []
        for physical in vk.vkEnumeratePhysicalDevices(self.instance):
            properties = vk.vkGetPhysicalDeviceProperties(physical)
            raw_name = properties.deviceName
            name = (
                raw_name.split("\0", 1)[0] if isinstance(raw_name, str)
                else bytes(raw_name).split(b"\0", 1)[0].decode()
            )
            for index, props in enumerate(vk.vkGetPhysicalDeviceQueueFamilyProperties(physical)):
                if (
                    props.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT
                    and (
                        self.surface is None
                        or self._get_surface_support(physical, index, self.surface)
                    )
                ):
                    if device_name is None or device_name.lower() in name.lower():
                        priority = 0 if properties.deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU else 1
                        candidates.append((priority, physical, index, name))
                    break
        if not candidates:
            vk.vkDestroyInstance(self.instance, None)
            raise RuntimeError("no Vulkan graphics adapter matched the request")
        _priority, self.physical_device, self.queue_family, name = sorted(
            candidates, key=lambda item: item[0],
        )[0]
        queue_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=self.queue_family, queueCount=1,
            pQueuePriorities=[1.0],
        )
        device_extensions = ["VK_KHR_swapchain"] if self.surface is not None else None
        self.device = vk.vkCreateDevice(self.physical_device, vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1, pQueueCreateInfos=[queue_info],
            enabledExtensionCount=len(device_extensions or ()),
            ppEnabledExtensionNames=device_extensions,
        ), None)
        self.queue = vk.vkGetDeviceQueue(self.device, self.queue_family, 0)
        self.command_pool = vk.vkCreateCommandPool(self.device, vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            flags=vk.VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            queueFamilyIndex=self.queue_family,
        ), None)
        self._uses_scene_textures = bool(
            getattr(self.program.fragment.reflection, "resources", ())
        )
        self._descriptor_set_layout = None
        if self._uses_scene_textures:
            self._descriptor_set_layout = vk.vkCreateDescriptorSetLayout(
                self.device, vk.VkDescriptorSetLayoutCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                    bindingCount=5,
                    pBindings=[
                        vk.VkDescriptorSetLayoutBinding(
                            binding=0,
                            descriptorType=vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                            descriptorCount=1,
                            stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                        ),
                        vk.VkDescriptorSetLayoutBinding(
                            binding=3,
                            descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                            descriptorCount=1,
                            stageFlags=vk.VK_SHADER_STAGE_VERTEX_BIT,
                        ),
                        vk.VkDescriptorSetLayoutBinding(
                            binding=1,
                            descriptorType=vk.VK_DESCRIPTOR_TYPE_SAMPLER,
                            descriptorCount=1,
                            stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                        ),
                        vk.VkDescriptorSetLayoutBinding(
                            binding=2,
                            descriptorType=vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                            descriptorCount=1,
                            stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                        ),
                        vk.VkDescriptorSetLayoutBinding(
                            binding=4,
                            descriptorType=vk.VK_DESCRIPTOR_TYPE_SAMPLER,
                            descriptorCount=1,
                            stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                        ),
                    ],
                ), None,
            )
        self._pipeline_layout = vk.vkCreatePipelineLayout(
            self.device, vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=1 if self._descriptor_set_layout else 0,
                pSetLayouts=(
                    [self._descriptor_set_layout]
                    if self._descriptor_set_layout else None
                ),
            ), None,
        )
        self._pipelines = {}
        shadow_program = type(program).shadow(target="spirv")
        self._shadow_vertex_module = self._shader(shadow_program.vertex.binary)
        self._shadow_fragment_module = self._shader(shadow_program.fragment.binary)
        self._shadow_pipeline_layout = vk.vkCreatePipelineLayout(
            self.device, vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=0,
            ), None,
        )
        self._shadow_pipelines = {}
        self.capabilities = RendererCapabilities(
            renderer="vulkan-raster", features=frozenset(
                {"raster", "offscreen", "depth"}
                | ({"direct-presentation", "resident-scene"}
                   if self.surface is not None else set())
            ),
            outputs=self.available_outputs,
            device=name,
        )
        self._closed = False
        self._swapchain = None
        self._swapchain_images = []
        self._swapchain_extent = None
        self._swapchain_format = None
        self._present_mode = None
        self._present_cache = {}
        self._present_scene_token = None
        self._prepared_scene_resources = None
        self._prepared_scene_resources_key = None
        self._prepared_present_mesh = None
        self._present_frames = []
        self._present_frame_index = 0
        if self.surface is not None:
            self._create_swapchain = vk.vkGetDeviceProcAddr(
                self.device, "vkCreateSwapchainKHR",
            )
            self._destroy_swapchain = vk.vkGetDeviceProcAddr(
                self.device, "vkDestroySwapchainKHR",
            )
            self._get_swapchain_images = vk.vkGetDeviceProcAddr(
                self.device, "vkGetSwapchainImagesKHR",
            )
            self._acquire_next_image = vk.vkGetDeviceProcAddr(
                self.device, "vkAcquireNextImageKHR",
            )
            self._queue_present = vk.vkGetDeviceProcAddr(
                self.device, "vkQueuePresentKHR",
            )
            self._get_surface_capabilities = vk.vkGetInstanceProcAddr(
                self.instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR",
            )
            self._get_surface_formats = vk.vkGetInstanceProcAddr(
                self.instance, "vkGetPhysicalDeviceSurfaceFormatsKHR",
            )
            self._get_surface_present_modes = vk.vkGetInstanceProcAddr(
                self.instance, "vkGetPhysicalDeviceSurfacePresentModesKHR",
            )
            for _index in range(2):
                self._present_frames.append({
                    "image_available": vk.vkCreateSemaphore(
                        self.device, vk.VkSemaphoreCreateInfo(
                            sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                        ), None,
                    ),
                    "render_finished": vk.vkCreateSemaphore(
                        self.device, vk.VkSemaphoreCreateInfo(
                            sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                        ), None,
                    ),
                    "fence": vk.vkCreateFence(
                        self.device, vk.VkFenceCreateInfo(
                            sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                            flags=vk.VK_FENCE_CREATE_SIGNALED_BIT,
                        ), None,
                    ),
                })

    @property
    def direct_presentation(self):
        """Whether this renderer owns a swapchain for an external surface."""
        return self.surface is not None

    def _destroy_swapchain_resources(self):
        self._clear_present_cache()
        if self._swapchain is not None:
            self._destroy_swapchain(self.device, self._swapchain, None)
        self._swapchain = None
        self._swapchain_images = []
        self._swapchain_extent = None

    def _destroy_frame_resources(self, resources):
        vk = self.vk
        destroy = {
            "pipeline": vk.vkDestroyPipeline,
            "pipeline_layout": vk.vkDestroyPipelineLayout,
            "shader": vk.vkDestroyShaderModule,
            "framebuffer": vk.vkDestroyFramebuffer,
            "render_pass": vk.vkDestroyRenderPass,
            "image_view": vk.vkDestroyImageView,
            "image": vk.vkDestroyImage,
            "buffer": vk.vkDestroyBuffer,
            "memory": vk.vkFreeMemory,
            "sampler": vk.vkDestroySampler,
            "descriptor_pool": vk.vkDestroyDescriptorPool,
            "descriptor_set_layout": vk.vkDestroyDescriptorSetLayout,
        }
        order = {
            "pipeline": 0, "framebuffer": 1, "pipeline_layout": 2,
            "shader": 3, "render_pass": 4, "descriptor_pool": 5,
            "descriptor_set_layout": 6, "image_view": 7, "sampler": 8,
            "image": 9, "buffer": 10, "memory": 11,
        }
        for kind, handle in sorted(resources, key=lambda item: order[item[0]]):
            destroy[kind](self.device, handle, None)

    def _clear_present_cache(self):
        if not getattr(self, "_present_cache", None):
            return
        for entry in self._present_cache.values():
            commands = [entry["command"]]
            if entry.get("setup_command") is not None:
                commands.append(entry["setup_command"])
            self.vk.vkFreeCommandBuffers(
                self.device, self.command_pool, len(commands), commands,
            )
            self._destroy_frame_resources(entry["resources"])
        self._present_cache.clear()

    def _write_memory(self, memory, payload):
        mapped = self.vk.vkMapMemory(
            self.device, memory, 0, len(payload), 0,
        )
        self.vk.ffi.memmove(mapped, payload, len(payload))
        self.vk.vkUnmapMemory(self.device, memory)

    @staticmethod
    def _present_cache_key(
        mesh, image_index, width, height, cache_token=None,
    ):
        atlas = mesh.resources.get("base_color_atlas")
        shadow_vertices = mesh.resources.get("shadow_vertices")
        shadow_indices = mesh.resources.get("shadow_indices")
        return (
            cache_token, image_index, width, height, mesh.layout,
            mesh.vertices.shape,
            None if mesh.indices is None else mesh.indices.shape,
            None if atlas is None else atlas.shape,
            None if shadow_vertices is None else shadow_vertices.shape,
            None if shadow_indices is None else shadow_indices.shape,
        )

    def _ensure_swapchain(self, width, height):
        if self.surface is None:
            raise RuntimeError("direct presentation requires an external surface")
        requested = (int(width), int(height))
        if self._swapchain is not None and self._swapchain_extent == requested:
            return
        vk = self.vk
        vk.vkDeviceWaitIdle(self.device)
        self._destroy_swapchain_resources()
        capabilities = self._get_surface_capabilities(
            self.physical_device, self.surface,
        )
        formats = self._get_surface_formats(
            self.physical_device, self.surface,
        )
        preferred = next((
            item for item in formats
            if item.format in (
                vk.VK_FORMAT_B8G8R8A8_SRGB,
                vk.VK_FORMAT_R8G8B8A8_SRGB,
            )
        ), formats[0])
        modes = self._get_surface_present_modes(
            self.physical_device, self.surface,
        )
        present_mode = (
            vk.VK_PRESENT_MODE_MAILBOX_KHR
            if vk.VK_PRESENT_MODE_MAILBOX_KHR in modes
            else vk.VK_PRESENT_MODE_FIFO_KHR
        )
        self._present_mode = (
            "mailbox"
            if present_mode == vk.VK_PRESENT_MODE_MAILBOX_KHR else "fifo"
        )
        if not (
            capabilities.supportedUsageFlags
            & vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT
        ):
            raise RuntimeError(
                "Qt Vulkan surface does not support transfer-destination images"
            )
        if capabilities.currentExtent.width != 0xFFFFFFFF:
            extent = capabilities.currentExtent
        else:
            extent = vk.VkExtent2D(
                width=max(capabilities.minImageExtent.width, min(
                    requested[0], capabilities.maxImageExtent.width,
                )),
                height=max(capabilities.minImageExtent.height, min(
                    requested[1], capabilities.maxImageExtent.height,
                )),
            )
        image_count = min(
            capabilities.minImageCount + 1,
            capabilities.maxImageCount or capabilities.minImageCount + 1,
        )
        composite_alpha = next((
            candidate for candidate in (
                vk.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                vk.VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
                vk.VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
                vk.VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
            )
            if capabilities.supportedCompositeAlpha & candidate
        ), None)
        if composite_alpha is None:
            raise RuntimeError("Qt Vulkan surface has no supported composite-alpha mode")
        self._swapchain = self._create_swapchain(
            self.device, vk.VkSwapchainCreateInfoKHR(
                sType=vk.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                surface=self.surface, minImageCount=image_count,
                imageFormat=preferred.format,
                imageColorSpace=preferred.colorSpace,
                imageExtent=extent, imageArrayLayers=1,
                imageUsage=vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                imageSharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
                preTransform=capabilities.currentTransform,
                compositeAlpha=composite_alpha,
                presentMode=present_mode, clipped=vk.VK_TRUE,
            ), None,
        )
        self._swapchain_images = list(
            self._get_swapchain_images(self.device, self._swapchain)
        )
        self._swapchain_extent = (extent.width, extent.height)
        self._swapchain_format = preferred.format

    def _memory_type(self, bits, flags):
        properties = self.vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device)
        for index in range(properties.memoryTypeCount):
            if bits & (1 << index) and properties.memoryTypes[index].propertyFlags & flags == flags:
                return index
        raise RuntimeError("no compatible Vulkan memory type")

    def _buffer(self, size, usage, flags, payload=None):
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
        if payload is not None:
            mapped = vk.vkMapMemory(self.device, memory, 0, len(payload), 0)
            vk.ffi.memmove(mapped, payload, len(payload))
            vk.vkUnmapMemory(self.device, memory)
        return buffer, memory

    def _shader(self, binary):
        vk = self.vk
        return vk.vkCreateShaderModule(self.device, vk.VkShaderModuleCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            codeSize=len(binary), pCode=binary,
        ), None)

    def _shadow_pass(self, mesh, atlas_view, atlas_width, atlas_height, remember):
        """Create per-frame shadow attachments and a compatible cached pipeline."""
        vk = self.vk
        rectangle = mesh.resources.get("shadow_rectangle")
        vertices = mesh.resources.get("shadow_vertices")
        indices = mesh.resources.get("shadow_indices")
        if rectangle is None or vertices is None or not len(vertices) or not len(indices):
            return None
        sx, sy, sw, sh, _aw, _ah = rectangle
        atlas_width, atlas_height = int(sw), int(sh)
        depth_image = remember("image", vk.vkCreateImage(
            self.device, vk.VkImageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                imageType=vk.VK_IMAGE_TYPE_2D, format=vk.VK_FORMAT_D32_SFLOAT,
                extent=vk.VkExtent3D(width=atlas_width, height=atlas_height, depth=1),
                mipLevels=1, arrayLayers=1, samples=vk.VK_SAMPLE_COUNT_1_BIT,
                tiling=vk.VK_IMAGE_TILING_OPTIMAL,
                usage=(vk.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                       | vk.VK_IMAGE_USAGE_SAMPLED_BIT),
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
                initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            ), None,
        ))
        requirements = vk.vkGetImageMemoryRequirements(self.device, depth_image)
        depth_memory = remember("memory", vk.vkAllocateMemory(
            self.device, vk.VkMemoryAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize=requirements.size,
                memoryTypeIndex=self._memory_type(
                    requirements.memoryTypeBits,
                    vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                ),
            ), None,
        ))
        vk.vkBindImageMemory(self.device, depth_image, depth_memory, 0)
        depth_view = remember("image_view", vk.vkCreateImageView(
            self.device, vk.VkImageViewCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                image=depth_image, viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                format=vk.VK_FORMAT_D32_SFLOAT,
                subresourceRange=vk.VkImageSubresourceRange(
                    aspectMask=vk.VK_IMAGE_ASPECT_DEPTH_BIT,
                    baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1,
                ),
            ), None,
        ))
        attachments = [
            vk.VkAttachmentDescription(
                format=vk.VK_FORMAT_D32_SFLOAT,
                samples=vk.VK_SAMPLE_COUNT_1_BIT,
                loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
                storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
                stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
                initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
                finalLayout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            ),
        ]
        depth_ref = vk.VkAttachmentReference(
            attachment=0,
            layout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        )
        render_pass = remember("render_pass", vk.vkCreateRenderPass(
            self.device, vk.VkRenderPassCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                attachmentCount=1, pAttachments=attachments,
                subpassCount=1, pSubpasses=[vk.VkSubpassDescription(
                    pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                    colorAttachmentCount=0, pColorAttachments=None,
                    pDepthStencilAttachment=depth_ref,
                )],
            ), None,
        ))
        framebuffer = remember("framebuffer", vk.vkCreateFramebuffer(
            self.device, vk.VkFramebufferCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                renderPass=render_pass, attachmentCount=1,
                pAttachments=[depth_view],
                width=atlas_width, height=atlas_height, layers=1,
            ), None,
        ))
        sx = sy = 0
        key = (atlas_width, atlas_height)
        pipeline = self._shadow_pipelines.get(key)
        if pipeline is None:
            stages = [
                vk.VkPipelineShaderStageCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
                    module=self._shadow_vertex_module, pName="main",
                ),
                vk.VkPipelineShaderStageCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                    stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                    module=self._shadow_fragment_module, pName="main",
                ),
            ]
            pipeline = vk.vkCreateGraphicsPipelines(
                self.device, vk.VK_NULL_HANDLE, 1,
                [vk.VkGraphicsPipelineCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                    stageCount=2, pStages=stages,
                    pVertexInputState=vk.VkPipelineVertexInputStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                        vertexBindingDescriptionCount=1,
                        pVertexBindingDescriptions=[vk.VkVertexInputBindingDescription(
                            binding=0, stride=16,
                            inputRate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
                        )],
                        vertexAttributeDescriptionCount=1,
                        pVertexAttributeDescriptions=[vk.VkVertexInputAttributeDescription(
                            location=0, binding=0,
                            format=vk.VK_FORMAT_R32G32B32A32_SFLOAT, offset=0,
                        )],
                    ),
                    pInputAssemblyState=vk.VkPipelineInputAssemblyStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                        topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                    ),
                    pViewportState=vk.VkPipelineViewportStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                        viewportCount=1, pViewports=[vk.VkViewport(
                            x=float(sx), y=float(sy + sh), width=float(sw),
                            height=-float(sh), minDepth=0.0, maxDepth=1.0,
                        )],
                        scissorCount=1, pScissors=[vk.VkRect2D(
                            offset=vk.VkOffset2D(x=sx, y=sy),
                            extent=vk.VkExtent2D(width=sw, height=sh),
                        )],
                    ),
                    pRasterizationState=vk.VkPipelineRasterizationStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                        polygonMode=vk.VK_POLYGON_MODE_FILL,
                        cullMode={
                            "none": vk.VK_CULL_MODE_NONE,
                            "front": vk.VK_CULL_MODE_FRONT_BIT,
                            "back": vk.VK_CULL_MODE_BACK_BIT,
                        }[self.config.shadow_cull_mode],
                        frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
                        depthBiasEnable=vk.VK_TRUE,
                        depthBiasConstantFactor=1.25,
                        depthBiasSlopeFactor=1.75,
                        depthBiasClamp=0.0,
                        lineWidth=1.0,
                    ),
                    pMultisampleState=vk.VkPipelineMultisampleStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                        rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT,
                    ),
                    pDepthStencilState=vk.VkPipelineDepthStencilStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                        depthTestEnable=vk.VK_TRUE, depthWriteEnable=vk.VK_TRUE,
                        depthCompareOp=vk.VK_COMPARE_OP_LESS,
                    ),
                    pColorBlendState=vk.VkPipelineColorBlendStateCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                        attachmentCount=0, pAttachments=None,
                    ),
                    layout=self._shadow_pipeline_layout,
                    renderPass=render_pass, subpass=0,
                )], None,
            )[0]
            self._shadow_pipelines[key] = pipeline
        host = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        vertex_buffer, vertex_memory = self._buffer(
            vertices.nbytes, vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            host, vertices.tobytes(),
        )
        index_buffer, index_memory = self._buffer(
            indices.nbytes, vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            host, indices.tobytes(),
        )
        for kind, handle in (("buffer", vertex_buffer), ("memory", vertex_memory),
                             ("buffer", index_buffer), ("memory", index_memory)):
            remember(kind, handle)
        return render_pass, framebuffer, pipeline, vertex_buffer, index_buffer, depth_view

    def render(
        self, mesh: RasterMesh, width: int, height: int, *, present=False,
        surface_size=None, cache_token=None,
    ) -> np.ndarray | None:
        vk = self.vk
        width, height = int(width), int(height)
        if width < 1 or height < 1:
            raise ValueError("raster target dimensions must be positive")
        if present:
            self._ensure_swapchain(*(surface_size or (width, height)))
            present_frame = self._present_frames[self._present_frame_index]
            vk.vkWaitForFences(
                self.device, 1, [present_frame["fence"]],
                vk.VK_TRUE, (1 << 64) - 1,
            )
            image_available = present_frame["image_available"]
            render_finished = present_frame["render_finished"]
            image_index = self._acquire_next_image(
                self.device, self._swapchain, (1 << 64) - 1,
                image_available, vk.VK_NULL_HANDLE,
            )
            vk.vkResetFences(self.device, 1, [present_frame["fence"]])
            cache_key = self._present_cache_key(
                mesh, image_index, width, height, cache_token,
            )
            cached = self._present_cache.get(cache_key)
            if cached is not None:
                camera_payload = mesh.resources.get("camera_uniform")
                if camera_payload is not None:
                    self._write_memory(cached["camera_memory"], camera_payload)
                else:
                    self._write_memory(
                        cached["vertex_memory"], mesh.vertices.tobytes(),
                    )
                started = __import__("time").perf_counter()
                fence = present_frame["fence"]
                vk.vkQueueSubmit(self.queue, 1, [vk.VkSubmitInfo(
                    sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    waitSemaphoreCount=1,
                    pWaitSemaphores=[image_available],
                    pWaitDstStageMask=[vk.VK_PIPELINE_STAGE_TRANSFER_BIT],
                    commandBufferCount=1,
                    pCommandBuffers=[cached["command"]],
                    signalSemaphoreCount=1,
                    pSignalSemaphores=[render_finished],
                )], fence)
                self._queue_present(self.queue, vk.VkPresentInfoKHR(
                    sType=vk.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                    waitSemaphoreCount=1,
                    pWaitSemaphores=[render_finished],
                    swapchainCount=1,
                    pSwapchains=[self._swapchain],
                    pImageIndices=[image_index],
                ))
                self._present_frame_index = (
                    self._present_frame_index + 1
                ) % len(self._present_frames)
                self.last_timings = {
                    "resident_submit_ms": (
                        __import__("time").perf_counter() - started
                    ) * 1000.0,
                    "resident_cache_hit": True,
                }
                return None
            self.last_timings = {"resident_cache_hit": False}
        else:
            image_available = render_finished = None
            image_index = cache_key = None
        if not len(mesh.vertices) and not present:
            clear = np.array((0.04, 0.06, 0.1, 1.0), np.float32)
            return np.broadcast_to(clear, (height, width, 4)).copy()
        resources = []
        def remember(kind, handle):
            resources.append((kind, handle)); return handle
        descriptor_set = None
        camera_buffer = camera_memory = None
        camera_payload = mesh.resources.get("camera_uniform")
        if camera_payload is not None:
            host_flags = (
                vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            )
            camera_buffer, camera_memory = self._buffer(
                len(camera_payload), vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                host_flags, camera_payload,
            )
            resources.extend((("buffer", camera_buffer), ("memory", camera_memory)))
        atlas_image = None
        atlas_view = None
        atlas_staging = None
        if self._uses_scene_textures:
            atlas = np.ascontiguousarray(
                mesh.resources.get("base_color_atlas", np.full((1, 1, 4), 255, np.uint8)),
                dtype=np.uint8,
            )
            atlas_height, atlas_width = atlas.shape[:2]
            host_flags = (
                vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
            )
            atlas_staging, atlas_staging_memory = self._buffer(
                atlas.nbytes, vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                host_flags, atlas.tobytes(),
            )
            resources.extend((
                ("buffer", atlas_staging), ("memory", atlas_staging_memory),
            ))
            atlas_image = remember("image", vk.vkCreateImage(
                self.device, vk.VkImageCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                    imageType=vk.VK_IMAGE_TYPE_2D,
                    format=vk.VK_FORMAT_R8G8B8A8_SRGB,
                    extent=vk.VkExtent3D(
                        width=atlas_width, height=atlas_height, depth=1,
                    ),
                    mipLevels=1, arrayLayers=1,
                    samples=vk.VK_SAMPLE_COUNT_1_BIT,
                    tiling=vk.VK_IMAGE_TILING_OPTIMAL,
                    usage=(vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT
                           | vk.VK_IMAGE_USAGE_SAMPLED_BIT
                           | vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT),
                    sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
                    initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
                ), None,
            ))
            atlas_req = vk.vkGetImageMemoryRequirements(self.device, atlas_image)
            atlas_memory = remember("memory", vk.vkAllocateMemory(
                self.device, vk.VkMemoryAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    allocationSize=atlas_req.size,
                    memoryTypeIndex=self._memory_type(
                        atlas_req.memoryTypeBits,
                        vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    ),
                ), None,
            ))
            vk.vkBindImageMemory(self.device, atlas_image, atlas_memory, 0)
            atlas_view = remember("image_view", vk.vkCreateImageView(
                self.device, vk.VkImageViewCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    image=atlas_image, viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                    format=vk.VK_FORMAT_R8G8B8A8_SRGB,
                    subresourceRange=vk.VkImageSubresourceRange(
                        aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                        baseMipLevel=0, levelCount=1,
                        baseArrayLayer=0, layerCount=1,
                    ),
                ), None,
            ))
            atlas_sampler = remember("sampler", vk.vkCreateSampler(
                self.device, vk.VkSamplerCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                    magFilter=vk.VK_FILTER_LINEAR, minFilter=vk.VK_FILTER_LINEAR,
                    mipmapMode=vk.VK_SAMPLER_MIPMAP_MODE_LINEAR,
                    addressModeU=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    addressModeV=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    addressModeW=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    maxLod=0.0,
                ), None,
            ))
            shadow_sampler = remember("sampler", vk.vkCreateSampler(
                self.device, vk.VkSamplerCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                    magFilter=vk.VK_FILTER_LINEAR,
                    minFilter=vk.VK_FILTER_LINEAR,
                    mipmapMode=vk.VK_SAMPLER_MIPMAP_MODE_LINEAR,
                    addressModeU=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    addressModeV=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    addressModeW=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    compareEnable=vk.VK_TRUE,
                    compareOp=vk.VK_COMPARE_OP_LESS_OR_EQUAL,
                    maxLod=0.0,
                ), None,
            ))
            descriptor_pool = remember("descriptor_pool", vk.vkCreateDescriptorPool(
                self.device, vk.VkDescriptorPoolCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                    maxSets=1, poolSizeCount=3,
                    pPoolSizes=[
                        vk.VkDescriptorPoolSize(
                            type=vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                            descriptorCount=2,
                        ),
                        vk.VkDescriptorPoolSize(
                            type=vk.VK_DESCRIPTOR_TYPE_SAMPLER,
                            descriptorCount=2,
                        ),
                        vk.VkDescriptorPoolSize(
                            type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                            descriptorCount=1,
                        ),
                    ],
                ), None,
            ))
            descriptor_set = vk.vkAllocateDescriptorSets(
                self.device, vk.VkDescriptorSetAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    descriptorPool=descriptor_pool, descriptorSetCount=1,
                    pSetLayouts=[self._descriptor_set_layout],
                ),
            )[0]
            vk.vkUpdateDescriptorSets(self.device, 3, [
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set, dstBinding=0, descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    pImageInfo=[vk.VkDescriptorImageInfo(
                        imageView=atlas_view,
                        imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    )],
                ),
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set, dstBinding=1, descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_SAMPLER,
                    pImageInfo=[vk.VkDescriptorImageInfo(sampler=atlas_sampler)],
                ),
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set, dstBinding=4, descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_SAMPLER,
                    pImageInfo=[vk.VkDescriptorImageInfo(sampler=shadow_sampler)],
                ),
            ], 0, None)
        shadow_bundle = (
            self._shadow_pass(
                mesh, atlas_view, atlas_width, atlas_height, remember,
            ) if atlas_view is not None else None
        )
        if descriptor_set is not None:
            shadow_view = (
                shadow_bundle[5] if shadow_bundle is not None else atlas_view
            )
            shadow_layout = (
                vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                if shadow_bundle is not None else
                vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            )
            vk.vkUpdateDescriptorSets(self.device, 1, [
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set, dstBinding=2, descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    pImageInfo=[vk.VkDescriptorImageInfo(
                        imageView=shadow_view, imageLayout=shadow_layout,
                    )],
                ),
            ], 0, None)
            if camera_buffer is not None:
                vk.vkUpdateDescriptorSets(self.device, 1, [
                    vk.VkWriteDescriptorSet(
                        sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        dstSet=descriptor_set, dstBinding=3,
                        descriptorCount=1,
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        pBufferInfo=[vk.VkDescriptorBufferInfo(
                            buffer=camera_buffer, offset=0,
                            range=len(camera_payload),
                        )],
                    ),
                ], 0, None)
        image = remember("image", vk.vkCreateImage(self.device, vk.VkImageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType=vk.VK_IMAGE_TYPE_2D, format=vk.VK_FORMAT_R16G16B16A16_SFLOAT,
            extent=vk.VkExtent3D(width=width, height=height, depth=1),
            mipLevels=1, arrayLayers=1, samples=vk.VK_SAMPLE_COUNT_1_BIT,
            tiling=vk.VK_IMAGE_TILING_OPTIMAL,
            usage=vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
        ), None))
        req = vk.vkGetImageMemoryRequirements(self.device, image)
        image_memory = remember("memory", vk.vkAllocateMemory(self.device, vk.VkMemoryAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            allocationSize=req.size,
            memoryTypeIndex=self._memory_type(req.memoryTypeBits, vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
        ), None))
        vk.vkBindImageMemory(self.device, image, image_memory, 0)
        view = remember("image_view", vk.vkCreateImageView(self.device, vk.VkImageViewCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, image=image,
            viewType=vk.VK_IMAGE_VIEW_TYPE_2D, format=vk.VK_FORMAT_R16G16B16A16_SFLOAT,
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, baseMipLevel=0, levelCount=1,
                baseArrayLayer=0, layerCount=1,
            ),
        ), None))
        depth_view = None
        if self.state.depth_test:
            depth_image = remember("image", vk.vkCreateImage(self.device, vk.VkImageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                imageType=vk.VK_IMAGE_TYPE_2D, format=vk.VK_FORMAT_D32_SFLOAT,
                extent=vk.VkExtent3D(width=width, height=height, depth=1),
                mipLevels=1, arrayLayers=1, samples=vk.VK_SAMPLE_COUNT_1_BIT,
                tiling=vk.VK_IMAGE_TILING_OPTIMAL,
                usage=vk.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
                initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            ), None))
            depth_req = vk.vkGetImageMemoryRequirements(self.device, depth_image)
            depth_memory = remember("memory", vk.vkAllocateMemory(
                self.device, vk.VkMemoryAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    allocationSize=depth_req.size,
                    memoryTypeIndex=self._memory_type(
                        depth_req.memoryTypeBits,
                        vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    ),
                ), None,
            ))
            vk.vkBindImageMemory(self.device, depth_image, depth_memory, 0)
            depth_view = remember("image_view", vk.vkCreateImageView(
                self.device, vk.VkImageViewCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    image=depth_image, viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                    format=vk.VK_FORMAT_D32_SFLOAT,
                    subresourceRange=vk.VkImageSubresourceRange(
                        aspectMask=vk.VK_IMAGE_ASPECT_DEPTH_BIT,
                        baseMipLevel=0, levelCount=1,
                        baseArrayLayer=0, layerCount=1,
                    ),
                ), None,
            ))
        attachment = vk.VkAttachmentDescription(
            format=vk.VK_FORMAT_R16G16B16A16_SFLOAT, samples=vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR, storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        )
        color_ref = vk.VkAttachmentReference(attachment=0, layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)
        attachments = [attachment]
        depth_ref = None
        if depth_view is not None:
            attachments.append(vk.VkAttachmentDescription(
                format=vk.VK_FORMAT_D32_SFLOAT,
                samples=vk.VK_SAMPLE_COUNT_1_BIT,
                loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
                storeOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
                stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
                initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
                finalLayout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ))
            depth_ref = vk.VkAttachmentReference(
                attachment=1,
                layout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            )
        render_pass = remember("render_pass", vk.vkCreateRenderPass(self.device, vk.VkRenderPassCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
            attachmentCount=len(attachments), pAttachments=attachments, subpassCount=1,
            pSubpasses=[vk.VkSubpassDescription(
                pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                colorAttachmentCount=1, pColorAttachments=[color_ref],
                pDepthStencilAttachment=depth_ref,
            )],
            dependencyCount=1, pDependencies=[vk.VkSubpassDependency(
                srcSubpass=0, dstSubpass=vk.VK_SUBPASS_EXTERNAL,
                srcStageMask=vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                dstStageMask=vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                srcAccessMask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                dstAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
            )],
        ), None))
        framebuffer = remember("framebuffer", vk.vkCreateFramebuffer(self.device, vk.VkFramebufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            renderPass=render_pass,
            attachmentCount=2 if depth_view is not None else 1,
            pAttachments=[view, depth_view] if depth_view is not None else [view],
            width=width, height=height, layers=1,
        ), None))
        vertex_module = remember("shader", self._shader(self.program.vertex.binary))
        fragment_module = remember("shader", self._shader(self.program.fragment.binary))
        layout = self._pipeline_layout
        stages = [
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_VERTEX_BIT, module=vertex_module, pName="main"),
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT, module=fragment_module, pName="main"),
        ]
        formats = {
            "float32": vk.VK_FORMAT_R32_SFLOAT,
            "float32x2": vk.VK_FORMAT_R32G32_SFLOAT,
            "float32x3": vk.VK_FORMAT_R32G32B32_SFLOAT,
            "float32x4": vk.VK_FORMAT_R32G32B32A32_SFLOAT,
        }
        topologies = {"triangle-list": vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, "triangle-strip": vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP, "line-list": vk.VK_PRIMITIVE_TOPOLOGY_LINE_LIST}
        cull_modes = {"none": vk.VK_CULL_MODE_NONE, "front": vk.VK_CULL_MODE_FRONT_BIT, "back": vk.VK_CULL_MODE_BACK_BIT}
        front_faces = {"cw": vk.VK_FRONT_FACE_CLOCKWISE, "ccw": vk.VK_FRONT_FACE_COUNTER_CLOCKWISE}
        compares = {"never": vk.VK_COMPARE_OP_NEVER, "less": vk.VK_COMPARE_OP_LESS, "less-equal": vk.VK_COMPARE_OP_LESS_OR_EQUAL, "always": vk.VK_COMPARE_OP_ALWAYS}
        pipeline_key = (mesh.layout, self.state, width, height)
        pipeline = self._pipelines.get(pipeline_key)
        if pipeline is None:
            pipeline = vk.vkCreateGraphicsPipelines(
            self.device, vk.VK_NULL_HANDLE, 1, [vk.VkGraphicsPipelineCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                stageCount=2, pStages=stages,
                pVertexInputState=vk.VkPipelineVertexInputStateCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                    vertexBindingDescriptionCount=1,
                    pVertexBindingDescriptions=[vk.VkVertexInputBindingDescription(binding=0, stride=mesh.layout.stride, inputRate=vk.VK_VERTEX_INPUT_RATE_VERTEX)],
                    vertexAttributeDescriptionCount=len(mesh.layout.attributes),
                    pVertexAttributeDescriptions=[vk.VkVertexInputAttributeDescription(location=item.location, binding=0, format=formats[item.format], offset=item.offset) for item in mesh.layout.attributes],
                ),
                pInputAssemblyState=vk.VkPipelineInputAssemblyStateCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                    topology=topologies[self.state.topology]),
                pViewportState=vk.VkPipelineViewportStateCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                    # Vulkan's framebuffer Y convention differs from WebGPU's.
                    # A negative viewport height (core since Vulkan 1.1) gives
                    # both target implementations the same image orientation.
                    viewportCount=1, pViewports=[vk.VkViewport(x=0, y=height, width=width, height=-height, minDepth=0, maxDepth=1)],
                    scissorCount=1, pScissors=[vk.VkRect2D(offset=vk.VkOffset2D(x=0, y=0), extent=vk.VkExtent2D(width=width, height=height))]),
                pRasterizationState=vk.VkPipelineRasterizationStateCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                    polygonMode=vk.VK_POLYGON_MODE_FILL, cullMode=cull_modes[self.state.cull_mode],
                    frontFace=front_faces[self.state.front_face], lineWidth=1.0),
                pMultisampleState=vk.VkPipelineMultisampleStateCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                    rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT),
                pDepthStencilState=vk.VkPipelineDepthStencilStateCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                    depthTestEnable=vk.VK_TRUE if self.state.depth_test else vk.VK_FALSE,
                    depthWriteEnable=vk.VK_TRUE if self.state.depth_write else vk.VK_FALSE,
                    depthCompareOp=compares[self.state.depth_compare],
                ),
                pColorBlendState=vk.VkPipelineColorBlendStateCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                    attachmentCount=1, pAttachments=[vk.VkPipelineColorBlendAttachmentState(
                        blendEnable=vk.VK_FALSE if self.state.blend_mode == "opaque" else vk.VK_TRUE,
                        srcColorBlendFactor=(vk.VK_BLEND_FACTOR_SRC_ALPHA if self.state.blend_mode == "alpha" else vk.VK_BLEND_FACTOR_ONE),
                        dstColorBlendFactor=(vk.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA if self.state.blend_mode == "alpha" else vk.VK_BLEND_FACTOR_ONE),
                        colorBlendOp=vk.VK_BLEND_OP_ADD,
                        srcAlphaBlendFactor=vk.VK_BLEND_FACTOR_ONE,
                        dstAlphaBlendFactor=(vk.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA if self.state.blend_mode == "alpha" else vk.VK_BLEND_FACTOR_ONE),
                        alphaBlendOp=vk.VK_BLEND_OP_ADD,
                        colorWriteMask=0xF)]),
                layout=layout, renderPass=render_pass, subpass=0,
            )], None)[0]
            self._pipelines[pipeline_key] = pipeline
        host = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        vertex_buffer, vertex_memory = self._buffer(mesh.vertices.nbytes, vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, host, mesh.vertices.tobytes())
        resources.extend((("buffer", vertex_buffer), ("memory", vertex_memory)))
        index_buffer = None
        if mesh.indices is not None:
            index_buffer, index_memory = self._buffer(mesh.indices.nbytes, vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT, host, mesh.indices.tobytes())
            resources.extend((("buffer", index_buffer), ("memory", index_memory)))
        readback_size = width * height * 4 * np.dtype(np.float16).itemsize
        readback = readback_memory = None
        if not present:
            readback, readback_memory = self._buffer(
                readback_size, vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT, host,
            )
            resources.extend((("buffer", readback), ("memory", readback_memory)))
        separate_setup = bool(present and atlas_image is not None)
        allocated_commands = list(vk.vkAllocateCommandBuffers(self.device, vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool, level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=2 if separate_setup else 1,
        )))
        setup_command = allocated_commands[0] if separate_setup else None
        main_command = allocated_commands[-1]
        command = setup_command or main_command
        vk.vkBeginCommandBuffer(command, vk.VkCommandBufferBeginInfo(sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO))
        if atlas_image is not None:
            atlas_range = vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0, levelCount=1,
                baseArrayLayer=0, layerCount=1,
            )
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                0, None, 0, None, 1, [vk.VkImageMemoryBarrier(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=0,
                    dstAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    oldLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
                    newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    image=atlas_image, subresourceRange=atlas_range,
                )],
            )
            vk.vkCmdCopyBufferToImage(
                command, atlas_staging, atlas_image,
                vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                [vk.VkBufferImageCopy(
                    bufferOffset=0, bufferRowLength=0, bufferImageHeight=0,
                    imageSubresource=vk.VkImageSubresourceLayers(
                        aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                        mipLevel=0, baseArrayLayer=0, layerCount=1,
                    ),
                    imageExtent=vk.VkExtent3D(
                        width=atlas_width, height=atlas_height, depth=1,
                    ),
                )],
            )
            next_layout = vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
            next_stage = vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
            next_access = vk.VK_ACCESS_SHADER_READ_BIT
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                next_stage, 0,
                0, None, 0, None, 1, [vk.VkImageMemoryBarrier(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    dstAccessMask=next_access,
                    oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    newLayout=next_layout,
                    srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    image=atlas_image, subresourceRange=atlas_range,
                )],
            )
        if shadow_bundle is not None:
            shadow_render_pass, shadow_framebuffer, shadow_pipeline, shadow_vb, shadow_ib, _shadow_view = shadow_bundle
            vk.vkCmdBeginRenderPass(command, vk.VkRenderPassBeginInfo(
                sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                renderPass=shadow_render_pass, framebuffer=shadow_framebuffer,
                renderArea=vk.VkRect2D(
                    offset=vk.VkOffset2D(x=0, y=0),
                    extent=vk.VkExtent2D(
                        width=self.config.shadow_map_size,
                        height=self.config.shadow_map_size,
                    ),
                ),
                clearValueCount=1,
                pClearValues=[
                    vk.VkClearValue(depthStencil=vk.VkClearDepthStencilValue(depth=1.0, stencil=0)),
                ],
            ), vk.VK_SUBPASS_CONTENTS_INLINE)
            vk.vkCmdBindPipeline(
                command, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, shadow_pipeline,
            )
            vk.vkCmdBindVertexBuffers(command, 0, 1, [shadow_vb], [0])
            vk.vkCmdBindIndexBuffer(command, shadow_ib, 0, vk.VK_INDEX_TYPE_UINT32)
            vk.vkCmdDrawIndexed(
                command, mesh.resources["shadow_indices"].size, 1, 0, 0, 0,
            )
            vk.vkCmdEndRenderPass(command)
        if separate_setup:
            vk.vkEndCommandBuffer(command)
            command = main_command
            vk.vkBeginCommandBuffer(
                command, vk.VkCommandBufferBeginInfo(
                    sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                ),
            )
        vk.vkCmdBeginRenderPass(command, vk.VkRenderPassBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            renderPass=render_pass, framebuffer=framebuffer,
            renderArea=vk.VkRect2D(offset=vk.VkOffset2D(x=0, y=0), extent=vk.VkExtent2D(width=width, height=height)),
            clearValueCount=2 if depth_view is not None else 1,
            pClearValues=(
                [vk.VkClearValue(color=vk.VkClearColorValue(float32=[0.04, 0.06, 0.1, 1.0])),
                 vk.VkClearValue(depthStencil=vk.VkClearDepthStencilValue(depth=1.0, stencil=0))]
                if depth_view is not None else
                [vk.VkClearValue(color=vk.VkClearColorValue(float32=[0.04, 0.06, 0.1, 1.0]))]
            ),
        ), vk.VK_SUBPASS_CONTENTS_INLINE)
        vk.vkCmdBindPipeline(command, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline)
        if descriptor_set is not None:
            vk.vkCmdBindDescriptorSets(
                command, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, layout,
                0, 1, [descriptor_set], 0, None,
            )
        vk.vkCmdBindVertexBuffers(command, 0, 1, [vertex_buffer], [0])
        if index_buffer is None:
            vk.vkCmdDraw(command, mesh.vertices.shape[0], 1, 0, 0)
        else:
            vk.vkCmdBindIndexBuffer(command, index_buffer, 0, vk.VK_INDEX_TYPE_UINT32)
            vk.vkCmdDrawIndexed(command, mesh.indices.size, 1, 0, 0, 0)
        vk.vkCmdEndRenderPass(command)
        vk.vkCmdPipelineBarrier(
            command,
            vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
            0, None, 0, None, 1,
            [vk.VkImageMemoryBarrier(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                dstAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
                oldLayout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                image=image,
                subresourceRange=vk.VkImageSubresourceRange(
                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0, levelCount=1,
                    baseArrayLayer=0, layerCount=1,
                ),
            )],
        )
        if present:
            swapchain_image = self._swapchain_images[image_index]
            color_range = vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0, levelCount=1,
                baseArrayLayer=0, layerCount=1,
            )
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                0, None, 0, None, 1, [vk.VkImageMemoryBarrier(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=0,
                    dstAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    oldLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
                    newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    image=swapchain_image, subresourceRange=color_range,
                )],
            )
            layers = vk.VkImageSubresourceLayers(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                mipLevel=0, baseArrayLayer=0, layerCount=1,
            )
            vk.vkCmdBlitImage(
                command, image, vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                swapchain_image, vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, [vk.VkImageBlit(
                    srcSubresource=layers,
                    srcOffsets=[
                        vk.VkOffset3D(x=0, y=0, z=0),
                        vk.VkOffset3D(x=width, y=height, z=1),
                    ],
                    dstSubresource=layers,
                    dstOffsets=[
                        vk.VkOffset3D(x=0, y=0, z=0),
                        vk.VkOffset3D(
                            x=self._swapchain_extent[0],
                            y=self._swapchain_extent[1], z=1,
                        ),
                    ],
                )], vk.VK_FILTER_NEAREST,
            )
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
                0, None, 0, None, 1, [vk.VkImageMemoryBarrier(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    dstAccessMask=0,
                    oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    newLayout=vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    image=swapchain_image, subresourceRange=color_range,
                )],
            )
        else:
            vk.vkCmdCopyImageToBuffer(command, image, vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1, [vk.VkBufferImageCopy(
                bufferOffset=0, bufferRowLength=0, bufferImageHeight=0,
                imageSubresource=vk.VkImageSubresourceLayers(aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, mipLevel=0, baseArrayLayer=0, layerCount=1),
                imageOffset=vk.VkOffset3D(x=0, y=0, z=0), imageExtent=vk.VkExtent3D(width=width, height=height, depth=1),
            )])
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                vk.VK_PIPELINE_STAGE_HOST_BIT, 0,
                0, None, 1, [vk.VkBufferMemoryBarrier(
                    sType=vk.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                    srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    dstAccessMask=vk.VK_ACCESS_HOST_READ_BIT,
                    srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    buffer=readback, offset=0, size=readback_size,
                )], 0, None,
            )
        vk.vkEndCommandBuffer(command)
        fence = (
            present_frame["fence"] if present else
            vk.vkCreateFence(
                self.device, vk.VkFenceCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                ), None,
            )
        )
        vk.vkQueueSubmit(self.queue, 1, [vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            waitSemaphoreCount=1 if present else 0,
            pWaitSemaphores=[image_available] if present else None,
            pWaitDstStageMask=(
                [vk.VK_PIPELINE_STAGE_TRANSFER_BIT] if present else None
            ),
            commandBufferCount=(2 if setup_command is not None else 1),
            pCommandBuffers=(
                [setup_command, command]
                if setup_command is not None else [command]
            ),
            signalSemaphoreCount=1 if present else 0,
            pSignalSemaphores=[render_finished] if present else None,
        )], fence)
        if present:
            self._queue_present(self.queue, vk.VkPresentInfoKHR(
                sType=vk.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                waitSemaphoreCount=1,
                pWaitSemaphores=[render_finished],
                swapchainCount=1,
                pSwapchains=[self._swapchain],
                pImageIndices=[image_index],
            ))
        result = None
        if present:
            self._present_frame_index = (
                self._present_frame_index + 1
            ) % len(self._present_frames)
        else:
            vk.vkWaitForFences(
                self.device, 1, [fence], vk.VK_TRUE, 10_000_000_000,
            )
            mapped = vk.vkMapMemory(
                self.device, readback_memory, 0, readback_size, 0,
            )
            result = np.frombuffer(
                bytes(mapped[:readback_size]), np.float16,
            ).astype(np.float32).reshape(height, width, 4)
            vk.vkUnmapMemory(self.device, readback_memory)
            vk.vkDestroyFence(self.device, fence, None)
        if present:
            self._present_cache[cache_key] = {
                "command": command,
                "setup_command": setup_command,
                "resources": resources,
                "vertex_memory": vertex_memory,
                "camera_memory": camera_memory,
            }
        else:
            vk.vkFreeCommandBuffers(
                self.device, self.command_pool, 1, [command],
            )
            self._destroy_frame_resources(resources)
        return result

    def present_frame(
        self, scene, camera, width, height, *, frame_index=0,
        surface_size=None,
    ):
        """Render directly to the external Vulkan surface without readback."""
        import time
        started = time.perf_counter()
        scene_token = (id(scene), scene.revision)
        if self._present_scene_token != scene_token:
            self.vk.vkDeviceWaitIdle(self.device)
            self._clear_present_cache()
            self._present_scene_token = scene_token
        prepare_started = time.perf_counter()
        resources_prepared = False
        prepared_key = (scene_token, self.config)
        if self._prepared_scene_resources_key != prepared_key:
            from ...raster._core import prepare_scene_mesh_resources
            self._prepared_scene_resources = prepare_scene_mesh_resources(
                scene, self.config, native_shadow_maps=True,
            )
            self._prepared_scene_resources_key = prepared_key
            resources_prepared = True
            self._prepared_present_mesh = scene_mesh(
                scene, camera, width, height, self.config,
                native_shadow_maps=True,
                prepared_resources=self._prepared_scene_resources,
                gpu_camera=True,
            )
        prepare_ms = (time.perf_counter() - prepare_started) * 1000.0
        pack_started = time.perf_counter()
        mesh = self._prepared_present_mesh
        camera_data = np.zeros(1, CAMERA_DTYPE)
        camera_data["view_projection"][0] = camera_matrix(
            camera, width, height,
        ).T
        camera_data["position_exposure"][0] = (*camera.position, 1.0)
        mesh.resources["camera_uniform"] = camera_data.tobytes()
        pack_ms = (time.perf_counter() - pack_started) * 1000.0
        self.render(
            mesh, width, height, present=True, surface_size=surface_size,
            cache_token=scene_token,
        )
        details = dict(self.last_timings)
        self.last_timings = details | {
            "total_ms": (time.perf_counter() - started) * 1000.0,
            "scene_pack_ms": pack_ms,
            "scene_prepare_ms": prepare_ms,
            "scene_resources_prepared": resources_prepared,
            "resident_cache_hit": bool(details.get("resident_cache_hit")),
            "direct_swapchain": True,
            "present_mode": self._present_mode,
        }

    def render_frame(self, scene, camera, width, height, *, samples=None, frame_index=0):
        import time
        started = time.perf_counter()
        mesh = scene_mesh(
            scene, camera, width, height, self.config,
            native_shadow_maps=True, gpu_camera=True,
        )
        camera_data = np.zeros(1, CAMERA_DTYPE)
        camera_data["view_projection"][0] = camera_matrix(camera, width, height).T
        camera_data["position_exposure"][0] = (*camera.position, 1.0)
        mesh.resources["camera_uniform"] = camera_data.tobytes()
        image = self.render(mesh, width, height)
        self.last_timings = {"total_ms": (time.perf_counter() - started) * 1000.0}
        return self._post.process(image, scene, camera)

    def render_products(self, scene, camera, width, height, *, outputs, samples=None, frame_index=0):
        mesh = scene_mesh(
            scene, camera, width, height, self.config,
            native_shadow_maps=True,
        )
        products = rasterize_geometry_products(mesh, width, height)
        if "color" in outputs:
            color_mesh = scene_mesh(
                scene, camera, width, height, self.config,
                native_shadow_maps=True, gpu_camera=True,
            )
            camera_data = np.zeros(1, CAMERA_DTYPE)
            camera_data["view_projection"][0] = camera_matrix(
                camera, width, height,
            ).T
            camera_data["position_exposure"][0] = (*camera.position, 1.0)
            color_mesh.resources["camera_uniform"] = camera_data.tobytes()
            image = self.render(color_mesh, width, height)
            products["color"] = self._post.process(image, scene, camera)
        return {name: products[name] for name in outputs}

    @property
    def accumulated_frames(self):
        return self._post.accumulated_frames

    def reset_output_history(self):
        self._post.reset()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.vk.vkDeviceWaitIdle(self.device)
        self._destroy_swapchain_resources()
        for frame in self._present_frames:
            self.vk.vkDestroyFence(self.device, frame["fence"], None)
            self.vk.vkDestroySemaphore(
                self.device, frame["render_finished"], None,
            )
            self.vk.vkDestroySemaphore(
                self.device, frame["image_available"], None,
            )
        self._present_frames.clear()
        for pipeline in self._pipelines.values():
            self.vk.vkDestroyPipeline(self.device, pipeline, None)
        self._pipelines.clear()
        for pipeline in self._shadow_pipelines.values():
            self.vk.vkDestroyPipeline(self.device, pipeline, None)
        self._shadow_pipelines.clear()
        self.vk.vkDestroyPipelineLayout(
            self.device, self._shadow_pipeline_layout, None,
        )
        self.vk.vkDestroyShaderModule(
            self.device, self._shadow_fragment_module, None,
        )
        self.vk.vkDestroyShaderModule(
            self.device, self._shadow_vertex_module, None,
        )
        self.vk.vkDestroyPipelineLayout(
            self.device, self._pipeline_layout, None,
        )
        if self._descriptor_set_layout is not None:
            self.vk.vkDestroyDescriptorSetLayout(
                self.device, self._descriptor_set_layout, None,
            )
        self.vk.vkDestroyCommandPool(self.device, self.command_pool, None)
        self.vk.vkDestroyDevice(self.device, None)
        if self._owns_instance:
            self.vk.vkDestroyInstance(self.instance, None)


__all__ = ["VulkanRasterRenderer"]
