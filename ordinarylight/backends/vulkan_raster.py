"""Minimal native Vulkan offscreen raster backend."""

from __future__ import annotations

import numpy as np

from ..capabilities import RendererCapabilities
from ..raster import (
    RasterConfig, RasterMesh, RasterPostProcessor, RasterState,
    create_raster_pipeline, rasterize_geometry_products, scene_mesh,
)


class VulkanRasterBackend:
    """Draw Ordinary Shade SPIR-V programs using a native Vulkan graphics queue.

    This intentionally small first backend supports one vec2 vertex stream, an
    optional uint32 index stream, and an offscreen RGBA8 color attachment.
    It establishes the native graphics pipeline/lifetime architecture without
    coupling it to the path tracer.
    """

    def __init__(self, program, *, config=None, state=None, device_name=None):
        try:
            import vulkan as vk
        except ImportError as error:
            raise RuntimeError(
                "Vulkan rasterization requires: pip install 'ordinarylight[vulkan]'"
            ) from error
        if program.vertex.target != "spirv" or program.fragment.target != "spirv":
            raise ValueError("VulkanRasterBackend requires a SPIR-V RasterProgram")
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
        self.instance = vk.vkCreateInstance(vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app,
        ), None)
        candidates = []
        for physical in vk.vkEnumeratePhysicalDevices(self.instance):
            properties = vk.vkGetPhysicalDeviceProperties(physical)
            raw_name = properties.deviceName
            name = (
                raw_name.split("\0", 1)[0] if isinstance(raw_name, str)
                else bytes(raw_name).split(b"\0", 1)[0].decode()
            )
            for index, props in enumerate(vk.vkGetPhysicalDeviceQueueFamilyProperties(physical)):
                if props.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT:
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
        self.device = vk.vkCreateDevice(self.physical_device, vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            queueCreateInfoCount=1, pQueueCreateInfos=[queue_info],
        ), None)
        self.queue = vk.vkGetDeviceQueue(self.device, self.queue_family, 0)
        self.command_pool = vk.vkCreateCommandPool(self.device, vk.VkCommandPoolCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            flags=vk.VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
            queueFamilyIndex=self.queue_family,
        ), None)
        self.capabilities = RendererCapabilities(
            backend="vulkan-raster", features=frozenset({"raster", "offscreen", "depth"}),
            outputs=self.available_outputs,
            device=name,
        )
        self._closed = False

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

    def render(self, mesh: RasterMesh, width: int, height: int) -> np.ndarray:
        vk = self.vk
        width, height = int(width), int(height)
        if width < 1 or height < 1:
            raise ValueError("raster target dimensions must be positive")
        if not len(mesh.vertices):
            clear = np.array((0.04, 0.06, 0.1, 1.0), np.float32)
            return np.broadcast_to(np.rint(clear * 255).astype(np.uint8), (height, width, 4)).copy()
        resources = []
        def remember(kind, handle):
            resources.append((kind, handle)); return handle
        image = remember("image", vk.vkCreateImage(self.device, vk.VkImageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType=vk.VK_IMAGE_TYPE_2D, format=vk.VK_FORMAT_R8G8B8A8_UNORM,
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
            viewType=vk.VK_IMAGE_VIEW_TYPE_2D, format=vk.VK_FORMAT_R8G8B8A8_UNORM,
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
            format=vk.VK_FORMAT_R8G8B8A8_UNORM, samples=vk.VK_SAMPLE_COUNT_1_BIT,
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
        layout = remember("pipeline_layout", vk.vkCreatePipelineLayout(self.device, vk.VkPipelineLayoutCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        ), None))
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
        pipeline = remember("pipeline", vk.vkCreateGraphicsPipelines(
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
                    # both backends the same Ordinary Light image orientation.
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
            )], None)[0])
        host = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        vertex_buffer, vertex_memory = self._buffer(mesh.vertices.nbytes, vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, host, mesh.vertices.tobytes())
        resources.extend((("buffer", vertex_buffer), ("memory", vertex_memory)))
        index_buffer = None
        if mesh.indices is not None:
            index_buffer, index_memory = self._buffer(mesh.indices.nbytes, vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT, host, mesh.indices.tobytes())
            resources.extend((("buffer", index_buffer), ("memory", index_memory)))
        readback, readback_memory = self._buffer(width * height * 4, vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT, host)
        resources.extend((("buffer", readback), ("memory", readback_memory)))
        command = vk.vkAllocateCommandBuffers(self.device, vk.VkCommandBufferAllocateInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            commandPool=self.command_pool, level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            commandBufferCount=1,
        ))[0]
        vk.vkBeginCommandBuffer(command, vk.VkCommandBufferBeginInfo(sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO))
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
                buffer=readback, offset=0, size=width * height * 4,
            )], 0, None,
        )
        vk.vkEndCommandBuffer(command)
        fence = vk.vkCreateFence(self.device, vk.VkFenceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO), None)
        vk.vkQueueSubmit(self.queue, 1, [vk.VkSubmitInfo(
            sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO, commandBufferCount=1, pCommandBuffers=[command],
        )], fence)
        vk.vkWaitForFences(self.device, 1, [fence], vk.VK_TRUE, 10_000_000_000)
        vk.vkQueueWaitIdle(self.queue)
        mapped = vk.vkMapMemory(self.device, readback_memory, 0, width * height * 4, 0)
        result = np.frombuffer(
            bytes(mapped[:width * height * 4]), np.uint8,
        ).copy().reshape(height, width, 4)
        vk.vkUnmapMemory(self.device, readback_memory)
        vk.vkDestroyFence(self.device, fence, None)
        vk.vkFreeCommandBuffers(self.device, self.command_pool, 1, [command])
        destroy = {
            "pipeline": vk.vkDestroyPipeline, "pipeline_layout": vk.vkDestroyPipelineLayout,
            "shader": vk.vkDestroyShaderModule, "framebuffer": vk.vkDestroyFramebuffer,
            "render_pass": vk.vkDestroyRenderPass, "image_view": vk.vkDestroyImageView,
            "image": vk.vkDestroyImage, "buffer": vk.vkDestroyBuffer,
            "memory": vk.vkFreeMemory,
        }
        for kind, handle in reversed(resources):
            destroy[kind](self.device, handle, None)
        return result

    def render_frame(self, scene, camera, width, height, *, samples=None, frame_index=0):
        import time
        started = time.perf_counter()
        image = self.render(scene_mesh(scene, camera, width, height, self.config), width, height)
        self.last_timings = {"total_ms": (time.perf_counter() - started) * 1000.0}
        return self._post.process(image.astype(np.float32) / 255.0, scene, camera)

    def render_products(self, scene, camera, width, height, *, outputs, samples=None, frame_index=0):
        mesh = scene_mesh(scene, camera, width, height, self.config)
        products = rasterize_geometry_products(mesh, width, height)
        if "color" in outputs:
            image = self.render(mesh, width, height).astype(np.float32) / 255.0
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
        self.vk.vkDestroyCommandPool(self.device, self.command_pool, None)
        self.vk.vkDestroyDevice(self.device, None)
        self.vk.vkDestroyInstance(self.instance, None)


__all__ = ["VulkanRasterBackend"]
