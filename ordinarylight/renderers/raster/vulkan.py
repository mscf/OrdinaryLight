"""Native Vulkan offscreen raster renderer."""

from __future__ import annotations

import hashlib

import numpy as np

_OPTICAL_DEBUG_MODES = {
    "off": 0.0, "hit": 1.0, "uv": 2.0, "depth-delta": 3.0,
    "confidence": 4.0, "object-id": 5.0, "depth-trace": 6.0,
    "refraction-hit": 7.0, "refraction-uv": 8.0,
    "refraction-source": 9.0,
}

from ...capabilities import RendererCapabilities
from ...raster import (
    RasterConfig, RasterMesh, RasterPostProcessor, RasterState,
    CAMERA_DTYPE, LIGHT_DTYPE, MATERIAL_DTYPE, SHADOW_DTYPE, camera_matrix, create_raster_pipeline,
    geometry_product_mesh, scene_mesh,
)
from ..base import RendererImplementation, RendererImplementationInfo
from ._diagnostics import frame_difference


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

    def request_probe_refresh(self, probe):
        """Request recapture of an ``on-demand`` reflection probe."""
        self.probe_capture.request(probe)

    def refresh_reflection_probes(self, scene, *, force=False):
        """Capture due probes immediately and return their replacements."""
        return self.probe_capture.refresh(self, scene, force=force)

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
        self.available_outputs = ("color", "depth", "normal", "object_id", "motion")
        self._output_history = None
        self.last_timings = {}
        from ...probes import ProbeCaptureManager
        self.probe_capture = ProbeCaptureManager()
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
                    bindingCount=12,
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
                            stageFlags=(vk.VK_SHADER_STAGE_VERTEX_BIT
                                        | vk.VK_SHADER_STAGE_FRAGMENT_BIT),
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
                        vk.VkDescriptorSetLayoutBinding(
                            binding=5,
                            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            descriptorCount=1,
                            stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                        ),
                        vk.VkDescriptorSetLayoutBinding(
                            binding=6,
                            descriptorType=vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                            descriptorCount=1,
                            stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                        ),
                        vk.VkDescriptorSetLayoutBinding(
                            binding=7,
                            descriptorType=vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                            descriptorCount=1,
                            stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                        ),
                        vk.VkDescriptorSetLayoutBinding(
                            binding=8,
                            descriptorType=vk.VK_DESCRIPTOR_TYPE_SAMPLER,
                            descriptorCount=1,
                            stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                        ),
                        vk.VkDescriptorSetLayoutBinding(
                            binding=9,
                            descriptorType=vk.VK_DESCRIPTOR_TYPE_SAMPLER,
                            descriptorCount=1,
                            stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                        ),
                        vk.VkDescriptorSetLayoutBinding(
                            binding=10,
                            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            descriptorCount=1,
                            stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                        ),
                        vk.VkDescriptorSetLayoutBinding(
                            binding=11,
                            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
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
        product_program = type(program).geometry_products(target="spirv")
        self._product_vertex_module = self._shader(product_program.vertex.binary)
        self._product_fragment_module = self._shader(product_program.fragment.binary)
        self._product_descriptor_layout = vk.vkCreateDescriptorSetLayout(
            self.device, vk.VkDescriptorSetLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                bindingCount=1,
                pBindings=[vk.VkDescriptorSetLayoutBinding(
                    binding=0,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    descriptorCount=1,
                    stageFlags=(vk.VK_SHADER_STAGE_VERTEX_BIT
                                | vk.VK_SHADER_STAGE_FRAGMENT_BIT),
                )],
            ), None,
        )
        self._product_pipeline_layout = vk.vkCreatePipelineLayout(
            self.device, vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=1, pSetLayouts=[self._product_descriptor_layout],
            ), None,
        )
        self._product_pipelines = {}
        volume_program = type(program).volume(target="spirv")
        self._volume_vertex_module = self._shader(volume_program.vertex.binary)
        self._volume_fragment_module = self._shader(volume_program.fragment.binary)
        volume_bindings = []
        for binding, descriptor_type in (
            (0, vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER),
            (1, vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
            (2, vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
            (3, vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE),
            (4, vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE),
            (5, vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE),
            (6, vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE),
            (7, vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE),
            (8, vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE),
            (9, vk.VK_DESCRIPTOR_TYPE_SAMPLER),
            (10, vk.VK_DESCRIPTOR_TYPE_SAMPLER),
            (11, vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
            (12, vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE),
            (13, vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE),
            (14, vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE),
            (15, vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE),
            (16, vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE),
            (17, vk.VK_DESCRIPTOR_TYPE_SAMPLER),
            (18, vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER),
        ):
            volume_bindings.append(vk.VkDescriptorSetLayoutBinding(
                binding=binding, descriptorType=descriptor_type,
                descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
            ))
        self._volume_descriptor_layout = vk.vkCreateDescriptorSetLayout(
            self.device, vk.VkDescriptorSetLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                bindingCount=len(volume_bindings), pBindings=volume_bindings,
            ), None,
        )
        self._volume_pipeline_layout = vk.vkCreatePipelineLayout(
            self.device, vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=1, pSetLayouts=[self._volume_descriptor_layout],
            ), None,
        )
        self._volume_pipelines = {}
        self.capabilities = RendererCapabilities(
            renderer="vulkan-raster", features=frozenset(
                {"raster", "offscreen", "depth", "volumes",
                 "volume_scattering", "volume-shadowing",
                 "overlapping-volume-extinction",
                 "volume-empty-space-skipping",
                 "native-volume-ray-march"}
                | ({"direct-presentation", "resident-scene"}
                   if self.surface is not None else set())
            ),
            outputs=self.available_outputs,
            limits={"max_volume_slices": 1024},
            device=name,
        )
        self._closed = False
        self._swapchain = None
        self._swapchain_images = []
        self._present_render_finished = []
        self._swapchain_extent = None
        self._swapchain_format = None
        self._present_mode = None
        self._present_cache = {}
        self._present_cache_serial = 0
        self._present_submission_sequence = 0
        self._present_readback_reference = None
        self._present_opaque_readback_reference = None
        self._present_depth_readback_reference = None
        self._present_scene_token = None
        self._present_cache_generation = None
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
        for semaphore in self._present_render_finished:
            self.vk.vkDestroySemaphore(self.device, semaphore, None)
        self._present_render_finished.clear()
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
        self._present_readback_reference = None
        self._present_opaque_readback_reference = None
        self._present_depth_readback_reference = None

    @staticmethod
    def _diagnostic_hash(value):
        if isinstance(value, bytes):
            payload = value
        else:
            payload = repr(value).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def _activate_present_cache_generation(self, generation):
        """Keep only the resident commands for the active draw ordering.

        Transparent and screen-space optical draws are sorted relative to the
        camera.  A different ordering needs different resident commands, but
        retaining every ordering also retains a complete set of framebuffer
        attachments for every swapchain image.  Bound that storage by retiring
        the previous generation before recording the new one.
        """
        generation = tuple(generation)
        if generation == self._present_cache_generation:
            return False
        if self._present_cache_generation is not None:
            self.vk.vkDeviceWaitIdle(self.device)
            self._clear_present_cache()
        self._present_cache_generation = generation
        return True

    def _write_memory(self, memory, payload):
        mapped = self.vk.vkMapMemory(
            self.device, memory, 0, len(payload), 0,
        )
        self.vk.ffi.memmove(mapped, payload, len(payload))
        self.vk.vkUnmapMemory(self.device, memory)

    @staticmethod
    def _opaque_camera_payload(payload):
        camera = np.frombuffer(payload, dtype=CAMERA_DTYPE).copy()
        mode = camera["viewport_optics"][0, 2]
        camera["viewport_optics"][0, 2] = -2.0 if mode < 0.0 else 0.0
        camera["optical_diagnostic"][0] = 0.0
        return camera.tobytes()

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
        # A render-complete binary semaphore cannot safely be reused merely
        # because its submission fence signalled: that fence does not prove
        # the presentation engine consumed the semaphore wait.  Associate the
        # semaphore with an acquired swapchain image instead. Reacquisition of
        # that image guarantees its previous presentation operation is done
        # using the semaphore.
        self._present_render_finished = [
            vk.vkCreateSemaphore(
                self.device, vk.VkSemaphoreCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                ), None,
            )
            for _image in self._swapchain_images
        ]
        self._swapchain_extent = (extent.width, extent.height)
        self._swapchain_format = preferred.format

    def _render_finished_for_image(self, image_index):
        """Return the presentation semaphore owned by an acquired image."""
        image_index = int(image_index)
        if not 0 <= image_index < len(self._present_render_finished):
            raise RuntimeError("acquired swapchain image has no completion semaphore")
        return self._present_render_finished[image_index]

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

    def _volume_texture(self, field, remember):
        """Upload one immutable scalar field and return its sampled 3-D view."""
        vk = self.vk
        field = np.ascontiguousarray(field, dtype=np.float32)
        depth, height, width = field.shape
        image = remember("image", vk.vkCreateImage(
            self.device, vk.VkImageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                imageType=vk.VK_IMAGE_TYPE_3D,
                format=vk.VK_FORMAT_R32_SFLOAT,
                extent=vk.VkExtent3D(width=width, height=height, depth=depth),
                mipLevels=1, arrayLayers=1,
                samples=vk.VK_SAMPLE_COUNT_1_BIT,
                tiling=vk.VK_IMAGE_TILING_OPTIMAL,
                usage=(vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT
                       | vk.VK_IMAGE_USAGE_SAMPLED_BIT),
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
                initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            ), None,
        ))
        requirements = vk.vkGetImageMemoryRequirements(self.device, image)
        memory = remember("memory", vk.vkAllocateMemory(
            self.device, vk.VkMemoryAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize=requirements.size,
                memoryTypeIndex=self._memory_type(
                    requirements.memoryTypeBits,
                    vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                ),
            ), None,
        ))
        vk.vkBindImageMemory(self.device, image, memory, 0)
        host = (vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        staging, staging_memory = self._buffer(
            field.nbytes, vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            host, field.tobytes(),
        )
        command = vk.vkAllocateCommandBuffers(
            self.device, vk.VkCommandBufferAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                commandPool=self.command_pool,
                level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1,
            ),
        )[0]
        vk.vkBeginCommandBuffer(command, vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        ))
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
                image=image, subresourceRange=color_range,
            )],
        )
        vk.vkCmdCopyBufferToImage(
            command, staging, image, vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, [vk.VkBufferImageCopy(
                bufferOffset=0, bufferRowLength=0, bufferImageHeight=0,
                imageSubresource=vk.VkImageSubresourceLayers(
                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    mipLevel=0, baseArrayLayer=0, layerCount=1,
                ),
                imageExtent=vk.VkExtent3D(
                    width=width, height=height, depth=depth,
                ),
            )],
        )
        vk.vkCmdPipelineBarrier(
            command, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
            0, None, 0, None, 1, [vk.VkImageMemoryBarrier(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                newLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                image=image, subresourceRange=color_range,
            )],
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
            vk.vkWaitForFences(
                self.device, 1, [fence], vk.VK_TRUE, (1 << 64) - 1,
            )
        finally:
            vk.vkDestroyFence(self.device, fence, None)
            vk.vkFreeCommandBuffers(
                self.device, self.command_pool, 1, [command],
            )
            vk.vkDestroyBuffer(self.device, staging, None)
            vk.vkFreeMemory(self.device, staging_memory, None)
        return remember("image_view", vk.vkCreateImageView(
            self.device, vk.VkImageViewCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                image=image, viewType=vk.VK_IMAGE_VIEW_TYPE_3D,
                format=vk.VK_FORMAT_R32_SFLOAT,
                subresourceRange=color_range,
            ), None,
        ))

    def _volume_pipeline(self, render_pass, width, height, remember):
        vk = self.vk
        stages = [
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
                module=self._volume_vertex_module, pName="main",
            ),
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                module=self._volume_fragment_module, pName="main",
            ),
        ]
        return remember("pipeline", vk.vkCreateGraphicsPipelines(
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
                    vertexAttributeDescriptionCount=2,
                    pVertexAttributeDescriptions=[
                        vk.VkVertexInputAttributeDescription(
                            location=0, binding=0,
                            format=vk.VK_FORMAT_R32G32_SFLOAT, offset=0,
                        ),
                        vk.VkVertexInputAttributeDescription(
                            location=1, binding=0,
                            format=vk.VK_FORMAT_R32G32_SFLOAT, offset=8,
                        ),
                    ],
                ),
                pInputAssemblyState=vk.VkPipelineInputAssemblyStateCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                    topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                ),
                pViewportState=vk.VkPipelineViewportStateCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                    viewportCount=1, pViewports=[vk.VkViewport(
                        # The scene image already uses the renderer's
                        # top-left-oriented attachment convention.  This
                        # fullscreen composite must sample it without the
                        # negative-height flip used by geometry pipelines.
                        x=0.0, y=0.0, width=float(width),
                        height=float(height), minDepth=0.0, maxDepth=1.0,
                    )],
                    scissorCount=1, pScissors=[vk.VkRect2D(
                        offset=vk.VkOffset2D(x=0, y=0),
                        extent=vk.VkExtent2D(width=width, height=height),
                    )],
                ),
                pRasterizationState=vk.VkPipelineRasterizationStateCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                    polygonMode=vk.VK_POLYGON_MODE_FILL,
                    cullMode=vk.VK_CULL_MODE_NONE,
                    frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
                    lineWidth=1.0,
                ),
                pMultisampleState=vk.VkPipelineMultisampleStateCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                    rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT,
                ),
                pColorBlendState=vk.VkPipelineColorBlendStateCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                    attachmentCount=1,
                    pAttachments=[vk.VkPipelineColorBlendAttachmentState(
                        blendEnable=vk.VK_FALSE,
                        colorWriteMask=(vk.VK_COLOR_COMPONENT_R_BIT
                                        | vk.VK_COLOR_COMPONENT_G_BIT
                                        | vk.VK_COLOR_COMPONENT_B_BIT
                                        | vk.VK_COLOR_COMPONENT_A_BIT),
                    )],
                ),
                layout=self._volume_pipeline_layout,
                renderPass=render_pass, subpass=0,
            )], None,
        )[0])

    @staticmethod
    def _volume_camera_payload(mesh, width, height):
        dtype = np.dtype([
            ("inverse_view_projection", np.float32, (4, 4)),
            ("camera_position", np.float32, (4,)),
            ("viewport_steps", np.float32, (4,)),
            ("volume_count", np.uint32, (4,)),
        ], align=True)
        camera = np.zeros(1, dtype)
        camera["inverse_view_projection"][0] = np.asarray(
            mesh.resources["volume_inverse_view_projection"], np.float32,
        ).T
        camera["camera_position"][0] = mesh.resources["volume_camera_position"]
        camera["viewport_steps"][0] = (
            width, height, mesh.resources["volume_step_scale"],
            mesh.resources["volume_max_steps"],
        )
        resources = mesh.resources["volume_resources"]
        camera["volume_count"][0, 0] = min(len(resources.scalar_fields), 4)
        camera["volume_count"][0, 1] = min(mesh.resources.get("light_count", 0), 8)
        camera["volume_count"][0, 2] = int(mesh.resources.get(
            "volume_empty_space_skipping", False,
        ))
        camera["volume_count"][0, 3] = min(mesh.resources.get("shadow_count", 0), 24)
        return camera.tobytes()

    def _record_volume_composite(
        self, command, mesh, width, height, source_image, source_view,
        depth_image, depth_view, shadow_view, shadow_sampler, remember,
    ):
        """Record the native volume pass and return color plus camera memory."""
        resources = mesh.resources.get("volume_resources")
        if resources is None or not resources.scalar_fields or depth_view is None:
            return source_image, source_view, None
        vk = self.vk
        color_range = vk.VkImageSubresourceRange(
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1,
        )
        depth_range = vk.VkImageSubresourceRange(
            aspectMask=vk.VK_IMAGE_ASPECT_DEPTH_BIT,
            baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1,
        )
        destination = remember("image", vk.vkCreateImage(
            self.device, vk.VkImageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                imageType=vk.VK_IMAGE_TYPE_2D,
                format=vk.VK_FORMAT_R16G16B16A16_SFLOAT,
                extent=vk.VkExtent3D(width=width, height=height, depth=1),
                mipLevels=1, arrayLayers=1,
                samples=vk.VK_SAMPLE_COUNT_1_BIT,
                tiling=vk.VK_IMAGE_TILING_OPTIMAL,
                usage=(vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                       | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                       | vk.VK_IMAGE_USAGE_SAMPLED_BIT),
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
                initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            ), None,
        ))
        requirements = vk.vkGetImageMemoryRequirements(self.device, destination)
        memory = remember("memory", vk.vkAllocateMemory(
            self.device, vk.VkMemoryAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize=requirements.size,
                memoryTypeIndex=self._memory_type(
                    requirements.memoryTypeBits,
                    vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                ),
            ), None,
        ))
        vk.vkBindImageMemory(self.device, destination, memory, 0)
        destination_view = remember("image_view", vk.vkCreateImageView(
            self.device, vk.VkImageViewCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                image=destination, viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                format=vk.VK_FORMAT_R16G16B16A16_SFLOAT,
                subresourceRange=color_range,
            ), None,
        ))
        render_pass = remember("render_pass", vk.vkCreateRenderPass(
            self.device, vk.VkRenderPassCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                attachmentCount=1,
                pAttachments=[vk.VkAttachmentDescription(
                    format=vk.VK_FORMAT_R16G16B16A16_SFLOAT,
                    samples=vk.VK_SAMPLE_COUNT_1_BIT,
                    loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
                    storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
                    stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                    stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
                    initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
                    finalLayout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                )],
                subpassCount=1,
                pSubpasses=[vk.VkSubpassDescription(
                    pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                    colorAttachmentCount=1,
                    pColorAttachments=[vk.VkAttachmentReference(
                        attachment=0,
                        layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    )],
                )],
            ), None,
        ))
        framebuffer = remember("framebuffer", vk.vkCreateFramebuffer(
            self.device, vk.VkFramebufferCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                renderPass=render_pass, attachmentCount=1,
                pAttachments=[destination_view],
                width=width, height=height, layers=1,
            ), None,
        ))
        pipeline = self._volume_pipeline(render_pass, width, height, remember)
        camera_payload = self._volume_camera_payload(mesh, width, height)
        host = (vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
        buffers = []
        for payload, usage in (
            (camera_payload, vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT),
            (resources.headers.tobytes(), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
            (resources.transfers.tobytes(), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
            (mesh.resources.get("light_buffer", b""), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
            (mesh.resources.get("shadow_buffer", b""), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT),
        ):
            buffer, buffer_memory = self._buffer(
                max(len(payload), 16), usage, host,
                payload if payload else bytes(16),
            )
            remember("buffer", buffer); remember("memory", buffer_memory)
            buffers.append((buffer, max(len(payload), 16), buffer_memory))
        fields = list(resources.scalar_fields[:4])
        while len(fields) < 4:
            fields.append(np.zeros((1, 1, 1), np.float32))
        volume_views = [self._volume_texture(field, remember) for field in fields]
        occupancy_fields = list(resources.occupancy_fields[:4])
        while len(occupancy_fields) < 4:
            occupancy_fields.append(np.ones((1, 1, 1), np.float32))
        occupancy_views = [
            self._volume_texture(field, remember) for field in occupancy_fields
        ]
        linear = remember("sampler", vk.vkCreateSampler(
            self.device, vk.VkSamplerCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                magFilter=vk.VK_FILTER_LINEAR, minFilter=vk.VK_FILTER_LINEAR,
                mipmapMode=vk.VK_SAMPLER_MIPMAP_MODE_NEAREST,
                addressModeU=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                addressModeV=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                addressModeW=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                maxLod=0.0,
            ), None,
        ))
        nearest = remember("sampler", vk.vkCreateSampler(
            self.device, vk.VkSamplerCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                magFilter=vk.VK_FILTER_NEAREST, minFilter=vk.VK_FILTER_NEAREST,
                mipmapMode=vk.VK_SAMPLER_MIPMAP_MODE_NEAREST,
                addressModeU=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                addressModeV=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                addressModeW=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                maxLod=0.0,
            ), None,
        ))
        pool = remember("descriptor_pool", vk.vkCreateDescriptorPool(
            self.device, vk.VkDescriptorPoolCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                maxSets=1, poolSizeCount=4,
                pPoolSizes=[
                    vk.VkDescriptorPoolSize(
                        type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        descriptorCount=1,
                    ),
                    vk.VkDescriptorPoolSize(
                        type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        descriptorCount=4,
                    ),
                    vk.VkDescriptorPoolSize(
                        type=vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                        descriptorCount=11,
                    ),
                    vk.VkDescriptorPoolSize(
                        type=vk.VK_DESCRIPTOR_TYPE_SAMPLER,
                        descriptorCount=3,
                    ),
                ],
            ), None,
        ))
        descriptor_set = vk.vkAllocateDescriptorSets(
            self.device, vk.VkDescriptorSetAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptorPool=pool, descriptorSetCount=1,
                pSetLayouts=[self._volume_descriptor_layout],
            ),
        )[0]
        writes = []
        for binding, descriptor_type, (buffer, size, _memory) in (
            (0, vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, buffers[0]),
            (1, vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, buffers[1]),
            (2, vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, buffers[2]),
            (11, vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, buffers[3]),
            (18, vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, buffers[4]),
        ):
            writes.append(vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set, dstBinding=binding,
                descriptorCount=1, descriptorType=descriptor_type,
                pBufferInfo=[vk.VkDescriptorBufferInfo(
                    buffer=buffer, offset=0, range=size,
                )],
            ))
        for binding, sampled_view, layout in (
            (3, source_view, vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL),
            (4, depth_view, vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL),
            *((5 + index, volume_view, vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
              for index, volume_view in enumerate(volume_views)),
            *((12 + index, volume_view, vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
              for index, volume_view in enumerate(occupancy_views)),
            (16, shadow_view, vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL),
        ):
            writes.append(vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set, dstBinding=binding,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                pImageInfo=[vk.VkDescriptorImageInfo(
                    imageView=sampled_view, imageLayout=layout,
                )],
            ))
        for binding, sampler in ((9, linear), (10, nearest), (17, shadow_sampler)):
            writes.append(vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set, dstBinding=binding,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_SAMPLER,
                pImageInfo=[vk.VkDescriptorImageInfo(sampler=sampler)],
            ))
        vk.vkUpdateDescriptorSets(self.device, len(writes), writes, 0, None)
        # position.xy, top-left-oriented texture coordinate.xy. Vulkan's
        # positive-height volume viewport maps clip and texture Y directly.
        fullscreen = np.asarray((
            (-1, -1, 0, 0), (3, -1, 2, 0), (-1, 3, 0, 2),
        ), np.float32)
        vertex_buffer, vertex_memory = self._buffer(
            fullscreen.nbytes, vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            host, fullscreen.tobytes(),
        )
        remember("buffer", vertex_buffer); remember("memory", vertex_memory)
        vk.vkCmdPipelineBarrier(
            command,
            (vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
             | vk.VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT),
            vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0,
            0, None, 0, None, 2, [
                vk.VkImageMemoryBarrier(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                    oldLayout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    newLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    image=source_image, subresourceRange=color_range,
                ),
                vk.VkImageMemoryBarrier(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=vk.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                    dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                    oldLayout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    newLayout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                    srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    image=depth_image, subresourceRange=depth_range,
                ),
            ],
        )
        vk.vkCmdBeginRenderPass(command, vk.VkRenderPassBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            renderPass=render_pass, framebuffer=framebuffer,
            renderArea=vk.VkRect2D(
                offset=vk.VkOffset2D(x=0, y=0),
                extent=vk.VkExtent2D(width=width, height=height),
            ),
            clearValueCount=1,
            pClearValues=[vk.VkClearValue(
                color=vk.VkClearColorValue(float32=[0.0, 0.0, 0.0, 1.0]),
            )],
        ), vk.VK_SUBPASS_CONTENTS_INLINE)
        vk.vkCmdBindPipeline(
            command, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline,
        )
        vk.vkCmdBindDescriptorSets(
            command, vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            self._volume_pipeline_layout, 0, 1, [descriptor_set], 0, None,
        )
        vk.vkCmdBindVertexBuffers(command, 0, 1, [vertex_buffer], [0])
        vk.vkCmdDraw(command, 3, 1, 0, 0)
        vk.vkCmdEndRenderPass(command)
        return destination, destination_view, buffers[0][2]

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
                       | vk.VK_IMAGE_USAGE_SAMPLED_BIT
                       | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT),
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
        surface_size=None, cache_token=None, diagnostic_readback=False,
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
            image_index = self._acquire_next_image(
                self.device, self._swapchain, (1 << 64) - 1,
                image_available, vk.VK_NULL_HANDLE,
            )
            render_finished = self._render_finished_for_image(image_index)
            cache_key = self._present_cache_key(
                mesh, image_index, width, height, cache_token,
            )
            cached = self._present_cache.get(cache_key)
            camera_payload = mesh.resources.get("camera_uniform")
            self._present_submission_sequence += 1
            diagnostic = {
                "present_submission": self._present_submission_sequence,
                "present_frame_slot": int(self._present_frame_index),
                "swapchain_image_index": int(image_index),
                "render_finished_slot": int(image_index),
                "camera_uniform_hash": self._diagnostic_hash(
                    camera_payload or b""
                ),
                "resident_cache_key_hash": self._diagnostic_hash(cache_key),
                "resident_generation_hash": self._diagnostic_hash(
                    self._present_cache_generation or ()
                ),
                "resident_cache_slot": (
                    int(cached["diagnostic_slot"])
                    if cached is not None else None
                ),
            }
            # Frame-slot fences and swapchain image indices rotate
            # independently. A resident command/resource set is keyed by the
            # acquired image, so its previous submission may belong to the
            # other frame slot. Wait for that exact submission before writing
            # its mapped uniforms or resubmitting its command buffer.
            if cached is not None:
                cached_fence = cached.get("last_fence")
                if (
                    cached_fence is not None
                    and cached_fence != present_frame["fence"]
                ):
                    vk.vkWaitForFences(
                        self.device, 1, [cached_fence],
                        vk.VK_TRUE, (1 << 64) - 1,
                    )
                if cached.get("diagnostic_readback_memory") is not None:
                    readback_size = int(cached["diagnostic_readback_size"])
                    mapped = vk.vkMapMemory(
                        self.device, cached["diagnostic_readback_memory"],
                        0, readback_size, 0,
                    )
                    raw = bytes(mapped[:readback_size])
                    vk.vkUnmapMemory(
                        self.device, cached["diagnostic_readback_memory"],
                    )
                    pixels = np.frombuffer(raw, np.float16).astype(np.float32)
                    diagnostic["present_hdr_hash"] = hashlib.sha256(
                        raw
                    ).hexdigest()[:16]
                    diagnostic["present_captured_submission"] = int(
                        cached.get("last_submission", 0)
                    )
                    if self._present_readback_reference is None:
                        self._present_readback_reference = pixels.copy()
                        diagnostic.update(
                            present_hdr_max_difference=0.0,
                            present_hdr_rmse=0.0,
                            present_hdr_changed_pixels=0,
                        )
                    else:
                        difference = frame_difference(
                            self._present_readback_reference.reshape(
                                height, width, 4,
                            ),
                            pixels.reshape(height, width, 4),
                        )
                        diagnostic.update(
                            present_hdr_max_difference=difference[
                                "maximum_absolute_difference"
                            ],
                            present_hdr_rmse=difference["rmse"],
                            present_hdr_changed_pixels=difference[
                                "changed_pixels"
                            ],
                            present_hdr_changed_bounds=difference[
                                "changed_bounds"
                            ],
                        )
                    opaque_memory = cached.get(
                        "diagnostic_opaque_readback_memory"
                    )
                    if opaque_memory is not None:
                        mapped = vk.vkMapMemory(
                            self.device, opaque_memory, 0, readback_size, 0,
                        )
                        opaque_raw = bytes(mapped[:readback_size])
                        vk.vkUnmapMemory(self.device, opaque_memory)
                        opaque_pixels = np.frombuffer(
                            opaque_raw, np.float16,
                        ).astype(np.float32).reshape(height, width, 4)
                        diagnostic["present_opaque_hdr_hash"] = (
                            hashlib.sha256(opaque_raw).hexdigest()[:16]
                        )
                        if self._present_opaque_readback_reference is None:
                            self._present_opaque_readback_reference = (
                                opaque_pixels.copy()
                            )
                            opaque_difference = frame_difference(
                                opaque_pixels, opaque_pixels,
                            )
                        else:
                            opaque_difference = frame_difference(
                                self._present_opaque_readback_reference,
                                opaque_pixels,
                            )
                        diagnostic.update(
                            present_opaque_hdr_max_difference=(
                                opaque_difference["maximum_absolute_difference"]
                            ),
                            present_opaque_hdr_rmse=opaque_difference["rmse"],
                            present_opaque_hdr_changed_pixels=(
                                opaque_difference["changed_pixels"]
                            ),
                            present_opaque_hdr_changed_bounds=(
                                opaque_difference["changed_bounds"]
                            ),
                        )
                    depth_memory = cached.get(
                        "diagnostic_depth_readback_memory"
                    )
                    if depth_memory is not None:
                        depth_size = int(cached["diagnostic_depth_readback_size"])
                        mapped = vk.vkMapMemory(
                            self.device, depth_memory, 0, depth_size, 0,
                        )
                        depth_raw = bytes(mapped[:depth_size])
                        vk.vkUnmapMemory(self.device, depth_memory)
                        depth_pixels = np.frombuffer(
                            depth_raw, np.float32,
                        ).reshape(height, width)
                        diagnostic["present_depth_hash"] = hashlib.sha256(
                            depth_raw,
                        ).hexdigest()[:16]
                        if self._present_depth_readback_reference is None:
                            self._present_depth_readback_reference = (
                                depth_pixels.copy()
                            )
                            depth_difference = frame_difference(
                                depth_pixels, depth_pixels,
                            )
                        else:
                            depth_difference = frame_difference(
                                self._present_depth_readback_reference,
                                depth_pixels,
                            )
                        diagnostic.update(
                            present_depth_max_difference=depth_difference[
                                "maximum_absolute_difference"
                            ],
                            present_depth_rmse=depth_difference["rmse"],
                            present_depth_changed_pixels=depth_difference[
                                "changed_pixels"
                            ],
                            present_depth_changed_bounds=depth_difference[
                                "changed_bounds"
                            ],
                        )
            vk.vkResetFences(self.device, 1, [present_frame["fence"]])
            if cached is not None:
                if camera_payload is not None:
                    self._write_memory(cached["camera_memory"], camera_payload)
                    if cached.get("opaque_camera_memory") is not None:
                        self._write_memory(
                            cached["opaque_camera_memory"],
                            self._opaque_camera_payload(camera_payload),
                        )
                else:
                    self._write_memory(
                        cached["vertex_memory"], mesh.vertices.tobytes(),
                    )
                if cached.get("volume_camera_memory") is not None:
                    self._write_memory(
                        cached["volume_camera_memory"],
                        self._volume_camera_payload(mesh, width, height),
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
                cached["last_fence"] = fence
                cached["last_submission"] = self._present_submission_sequence
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
                    **diagnostic,
                }
                return None
            self.last_timings = {"resident_cache_hit": False, **diagnostic}
        else:
            image_available = render_finished = None
            image_index = cache_key = None
        if (
            not len(mesh.vertices)
            and mesh.resources.get("volume_resources") is None
            and not present
        ):
            clear = np.array((0.04, 0.06, 0.1, 1.0), np.float32)
            return np.broadcast_to(clear, (height, width, 4)).copy()
        resources = []
        def remember(kind, handle):
            resources.append((kind, handle)); return handle
        descriptor_set = None
        optical_descriptor_set = optical_ping_descriptor_set = None
        optical_immutable_descriptor_set = None
        screen_space_optics = bool(
            self.config.optical_quality == "screen-space"
            and mesh.resources.get("optical_index_count", 0)
            and self.state.depth_test
        )
        host_flags = (
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        )
        camera_buffer = camera_memory = None
        opaque_camera_buffer = opaque_camera_memory = None
        camera_payload = mesh.resources.get("camera_uniform")
        if camera_payload is not None:
            camera_buffer, camera_memory = self._buffer(
                len(camera_payload), vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                host_flags, camera_payload,
            )
            resources.extend((("buffer", camera_buffer), ("memory", camera_memory)))
            if screen_space_optics:
                opaque_camera_buffer, opaque_camera_memory = self._buffer(
                    len(camera_payload), vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                    host_flags, self._opaque_camera_payload(camera_payload),
                )
                resources.extend((
                    ("buffer", opaque_camera_buffer),
                    ("memory", opaque_camera_memory),
                ))
        material_payload = mesh.resources.get("material_buffer")
        material_buffer = material_memory = None
        if material_payload is not None:
            if not material_payload:
                material_payload = bytes(MATERIAL_DTYPE.itemsize)
            material_buffer, material_memory = self._buffer(
                len(material_payload), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                host_flags, material_payload,
            )
            resources.extend((
                ("buffer", material_buffer), ("memory", material_memory),
            ))
        light_payload = mesh.resources.get("light_buffer", b"")
        if not light_payload:
            light_payload = bytes(LIGHT_DTYPE.itemsize)
        light_buffer, light_memory = self._buffer(
            len(light_payload), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            host_flags, light_payload,
        )
        resources.extend((
            ("buffer", light_buffer), ("memory", light_memory),
        ))
        shadow_payload = mesh.resources.get("shadow_buffer", b"")
        if not shadow_payload:
            shadow_payload = bytes(SHADOW_DTYPE.itemsize)
        shadow_record_buffer, shadow_record_memory = self._buffer(
            len(shadow_payload), vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            host_flags, shadow_payload,
        )
        resources.extend((
            ("buffer", shadow_record_buffer),
            ("memory", shadow_record_memory),
        ))
        atlas_image = None
        atlas_view = None
        atlas_staging = None
        shadow_view = None
        shadow_sampler = None
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
            scene_depth_sampler = remember("sampler", vk.vkCreateSampler(
                self.device, vk.VkSamplerCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                    magFilter=vk.VK_FILTER_NEAREST,
                    minFilter=vk.VK_FILTER_NEAREST,
                    mipmapMode=vk.VK_SAMPLER_MIPMAP_MODE_NEAREST,
                    addressModeU=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    addressModeV=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    addressModeW=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                    maxLod=0.0,
                ), None,
            ))
            descriptor_pool = remember("descriptor_pool", vk.vkCreateDescriptorPool(
                self.device, vk.VkDescriptorPoolCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                    maxSets=4, poolSizeCount=4,
                    pPoolSizes=[
                        vk.VkDescriptorPoolSize(
                            type=vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                            descriptorCount=16,
                        ),
                        vk.VkDescriptorPoolSize(
                            type=vk.VK_DESCRIPTOR_TYPE_SAMPLER,
                            descriptorCount=16,
                        ),
                        vk.VkDescriptorPoolSize(
                            type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                            descriptorCount=4,
                        ),
                        vk.VkDescriptorPoolSize(
                            type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            descriptorCount=12,
                        ),
                    ],
                ), None,
            ))
            allocated_descriptor_sets = vk.vkAllocateDescriptorSets(
                self.device, vk.VkDescriptorSetAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    descriptorPool=descriptor_pool, descriptorSetCount=4,
                    pSetLayouts=[self._descriptor_set_layout] * 4,
                ),
            )
            descriptor_set = allocated_descriptor_sets[0]
            optical_descriptor_set = allocated_descriptor_sets[1]
            optical_ping_descriptor_set = allocated_descriptor_sets[2]
            optical_immutable_descriptor_set = allocated_descriptor_sets[3]
            vk.vkUpdateDescriptorSets(self.device, 6, [
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
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set, dstBinding=6, descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    pImageInfo=[vk.VkDescriptorImageInfo(
                        imageView=atlas_view,
                        imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    )],
                ),
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set, dstBinding=8, descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_SAMPLER,
                    pImageInfo=[vk.VkDescriptorImageInfo(sampler=atlas_sampler)],
                ),
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set, dstBinding=9, descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_SAMPLER,
                    pImageInfo=[vk.VkDescriptorImageInfo(
                        sampler=scene_depth_sampler,
                    )],
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
            vk.vkUpdateDescriptorSets(self.device, 2, [
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set, dstBinding=2, descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    pImageInfo=[vk.VkDescriptorImageInfo(
                        imageView=shadow_view, imageLayout=shadow_layout,
                    )],
                ),
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set, dstBinding=7, descriptorCount=1,
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
                            buffer=(opaque_camera_buffer or camera_buffer), offset=0,
                            range=len(camera_payload),
                        )],
                    ),
                ], 0, None)
            if material_buffer is not None:
                vk.vkUpdateDescriptorSets(self.device, 1, [
                    vk.VkWriteDescriptorSet(
                        sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        dstSet=descriptor_set, dstBinding=5,
                        descriptorCount=1,
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        pBufferInfo=[vk.VkDescriptorBufferInfo(
                            buffer=material_buffer, offset=0,
                            range=len(material_payload),
                        )],
                    ),
                ], 0, None)
            vk.vkUpdateDescriptorSets(self.device, 1, [
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set, dstBinding=10,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[vk.VkDescriptorBufferInfo(
                        buffer=light_buffer, offset=0,
                        range=len(light_payload),
                    )],
                ),
            ], 0, None)
            vk.vkUpdateDescriptorSets(self.device, 1, [
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set, dstBinding=11,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[vk.VkDescriptorBufferInfo(
                        buffer=shadow_record_buffer, offset=0,
                        range=len(shadow_payload),
                    )],
                ),
            ], 0, None)
            # Keep an immutable fallback set for the opaque prepass and a
            # second set whose scene-color binding can safely reference the
            # completed opaque attachment during the optical pass.
            copies = [vk.VkCopyDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_COPY_DESCRIPTOR_SET,
                srcSet=descriptor_set, srcBinding=binding, srcArrayElement=0,
                dstSet=target, dstBinding=binding,
                dstArrayElement=0, descriptorCount=1,
            ) for target in (
                optical_descriptor_set, optical_ping_descriptor_set,
                optical_immutable_descriptor_set,
            ) for binding in range(12)]
            vk.vkUpdateDescriptorSets(
                self.device, 0, None, len(copies), copies,
            )
            if camera_buffer is not None and screen_space_optics:
                camera_writes = [
                    vk.VkWriteDescriptorSet(
                        sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        dstSet=target, dstBinding=3,
                        descriptorCount=1,
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        pBufferInfo=[vk.VkDescriptorBufferInfo(
                            buffer=camera_buffer, offset=0,
                            range=len(camera_payload),
                        )],
                    ) for target in (
                        optical_descriptor_set, optical_ping_descriptor_set,
                        optical_immutable_descriptor_set,
                    )
                ]
                vk.vkUpdateDescriptorSets(
                    self.device, len(camera_writes), camera_writes, 0, None,
                )
        image = remember("image", vk.vkCreateImage(self.device, vk.VkImageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType=vk.VK_IMAGE_TYPE_2D, format=vk.VK_FORMAT_R16G16B16A16_SFLOAT,
            extent=vk.VkExtent3D(width=width, height=height, depth=1),
            mipLevels=1, arrayLayers=1, samples=vk.VK_SAMPLE_COUNT_1_BIT,
            tiling=vk.VK_IMAGE_TILING_OPTIMAL,
            usage=(vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                   | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                   | vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT
                   | vk.VK_IMAGE_USAGE_SAMPLED_BIT),
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
                usage=(vk.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                       | vk.VK_IMAGE_USAGE_SAMPLED_BIT
                       | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT),
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
                storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
                stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
                initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
                finalLayout=(
                    vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                    if screen_space_optics else
                    vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                ),
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
                srcStageMask=(vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                              | vk.VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT),
                dstStageMask=(vk.VK_PIPELINE_STAGE_TRANSFER_BIT
                              | vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT),
                srcAccessMask=(vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                               | vk.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT),
                dstAccessMask=(vk.VK_ACCESS_TRANSFER_READ_BIT
                               | vk.VK_ACCESS_SHADER_READ_BIT),
            )],
        ), None))
        framebuffer = remember("framebuffer", vk.vkCreateFramebuffer(self.device, vk.VkFramebufferCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            renderPass=render_pass,
            attachmentCount=2 if depth_view is not None else 1,
            pAttachments=[view, depth_view] if depth_view is not None else [view],
            width=width, height=height, layers=1,
        ), None))
        screen_space_optics = bool(screen_space_optics and depth_view is not None)
        optical_image = optical_view = optical_render_pass = optical_framebuffer = None
        optical_immutable_image = optical_immutable_view = None
        optical_ping_framebuffer = None
        optical_depth_image = optical_depth_view = None
        optical_ping_depth_image = optical_ping_depth_view = None
        if screen_space_optics:
            optical_image = remember("image", vk.vkCreateImage(
                self.device, vk.VkImageCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                    imageType=vk.VK_IMAGE_TYPE_2D,
                    format=vk.VK_FORMAT_R16G16B16A16_SFLOAT,
                    extent=vk.VkExtent3D(width=width, height=height, depth=1),
                    mipLevels=1, arrayLayers=1,
                    samples=vk.VK_SAMPLE_COUNT_1_BIT,
                    tiling=vk.VK_IMAGE_TILING_OPTIMAL,
                    usage=(vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT
                           | vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT
                           | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                           | vk.VK_IMAGE_USAGE_SAMPLED_BIT),
                    sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
                    initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
                ), None,
            ))
            optical_req = vk.vkGetImageMemoryRequirements(
                self.device, optical_image,
            )
            optical_memory = remember("memory", vk.vkAllocateMemory(
                self.device, vk.VkMemoryAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    allocationSize=optical_req.size,
                    memoryTypeIndex=self._memory_type(
                        optical_req.memoryTypeBits,
                        vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    ),
                ), None,
            ))
            vk.vkBindImageMemory(self.device, optical_image, optical_memory, 0)
            optical_view = remember("image_view", vk.vkCreateImageView(
                self.device, vk.VkImageViewCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    image=optical_image,
                    viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                    format=vk.VK_FORMAT_R16G16B16A16_SFLOAT,
                    subresourceRange=vk.VkImageSubresourceRange(
                        aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                        baseMipLevel=0, levelCount=1,
                        baseArrayLayer=0, layerCount=1,
                    ),
                ), None,
            ))
            optical_immutable_image = remember("image", vk.vkCreateImage(
                self.device, vk.VkImageCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                    imageType=vk.VK_IMAGE_TYPE_2D,
                    format=vk.VK_FORMAT_R16G16B16A16_SFLOAT,
                    extent=vk.VkExtent3D(width=width, height=height, depth=1),
                    mipLevels=1, arrayLayers=1,
                    samples=vk.VK_SAMPLE_COUNT_1_BIT,
                    tiling=vk.VK_IMAGE_TILING_OPTIMAL,
                    usage=(vk.VK_IMAGE_USAGE_SAMPLED_BIT
                           | vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT),
                    sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
                    initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
                ), None,
            ))
            optical_immutable_req = vk.vkGetImageMemoryRequirements(
                self.device, optical_immutable_image,
            )
            optical_immutable_memory = remember("memory", vk.vkAllocateMemory(
                self.device, vk.VkMemoryAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    allocationSize=optical_immutable_req.size,
                    memoryTypeIndex=self._memory_type(
                        optical_immutable_req.memoryTypeBits,
                        vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    ),
                ), None,
            ))
            vk.vkBindImageMemory(
                self.device, optical_immutable_image,
                optical_immutable_memory, 0,
            )
            optical_immutable_view = remember(
                "image_view", vk.vkCreateImageView(
                    self.device, vk.VkImageViewCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                        image=optical_immutable_image,
                        viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                        format=vk.VK_FORMAT_R16G16B16A16_SFLOAT,
                        subresourceRange=vk.VkImageSubresourceRange(
                            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                            baseMipLevel=0, levelCount=1,
                            baseArrayLayer=0, layerCount=1,
                        ),
                    ), None,
                ),
            )
            optical_depth_image = remember("image", vk.vkCreateImage(
                self.device, vk.VkImageCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                    imageType=vk.VK_IMAGE_TYPE_2D,
                    format=vk.VK_FORMAT_D32_SFLOAT,
                    extent=vk.VkExtent3D(width=width, height=height, depth=1),
                    mipLevels=1, arrayLayers=1,
                    samples=vk.VK_SAMPLE_COUNT_1_BIT,
                    tiling=vk.VK_IMAGE_TILING_OPTIMAL,
                    usage=(vk.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                           | vk.VK_IMAGE_USAGE_SAMPLED_BIT
                           | vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT
                           | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT),
                    sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
                    initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
                ), None,
            ))
            optical_depth_req = vk.vkGetImageMemoryRequirements(
                self.device, optical_depth_image,
            )
            optical_depth_memory = remember("memory", vk.vkAllocateMemory(
                self.device, vk.VkMemoryAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    allocationSize=optical_depth_req.size,
                    memoryTypeIndex=self._memory_type(
                        optical_depth_req.memoryTypeBits,
                        vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    ),
                ), None,
            ))
            vk.vkBindImageMemory(
                self.device, optical_depth_image, optical_depth_memory, 0,
            )
            optical_depth_view = remember("image_view", vk.vkCreateImageView(
                self.device, vk.VkImageViewCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    image=optical_depth_image,
                    viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                    format=vk.VK_FORMAT_D32_SFLOAT,
                    subresourceRange=vk.VkImageSubresourceRange(
                        aspectMask=vk.VK_IMAGE_ASPECT_DEPTH_BIT,
                        baseMipLevel=0, levelCount=1,
                        baseArrayLayer=0, layerCount=1,
                    ),
                ), None,
            ))
            optical_ping_depth_image = remember("image", vk.vkCreateImage(
                self.device, vk.VkImageCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                    imageType=vk.VK_IMAGE_TYPE_2D,
                    format=vk.VK_FORMAT_D32_SFLOAT,
                    extent=vk.VkExtent3D(width=width, height=height, depth=1),
                    mipLevels=1, arrayLayers=1,
                    samples=vk.VK_SAMPLE_COUNT_1_BIT,
                    tiling=vk.VK_IMAGE_TILING_OPTIMAL,
                    usage=(vk.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                           | vk.VK_IMAGE_USAGE_SAMPLED_BIT
                           | vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT
                           | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT),
                    sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
                    initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
                ), None,
            ))
            optical_ping_depth_req = vk.vkGetImageMemoryRequirements(
                self.device, optical_ping_depth_image,
            )
            optical_ping_depth_memory = remember("memory", vk.vkAllocateMemory(
                self.device, vk.VkMemoryAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    allocationSize=optical_ping_depth_req.size,
                    memoryTypeIndex=self._memory_type(
                        optical_ping_depth_req.memoryTypeBits,
                        vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    ),
                ), None,
            ))
            vk.vkBindImageMemory(
                self.device, optical_ping_depth_image,
                optical_ping_depth_memory, 0,
            )
            optical_ping_depth_view = remember(
                "image_view", vk.vkCreateImageView(
                    self.device, vk.VkImageViewCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                        image=optical_ping_depth_image,
                        viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                        format=vk.VK_FORMAT_D32_SFLOAT,
                        subresourceRange=vk.VkImageSubresourceRange(
                            aspectMask=vk.VK_IMAGE_ASPECT_DEPTH_BIT,
                            baseMipLevel=0, levelCount=1,
                            baseArrayLayer=0, layerCount=1,
                        ),
                    ), None,
                ),
            )
            optical_attachments = [
                vk.VkAttachmentDescription(
                    format=vk.VK_FORMAT_R16G16B16A16_SFLOAT,
                    samples=vk.VK_SAMPLE_COUNT_1_BIT,
                    loadOp=vk.VK_ATTACHMENT_LOAD_OP_LOAD,
                    storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
                    stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                    stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
                    initialLayout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                    finalLayout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                ),
                vk.VkAttachmentDescription(
                    format=vk.VK_FORMAT_D32_SFLOAT,
                    samples=vk.VK_SAMPLE_COUNT_1_BIT,
                    loadOp=vk.VK_ATTACHMENT_LOAD_OP_LOAD,
                    # Layered optical passes share the immutable opaque-depth
                    # seed.  Discarding it after the first pass makes later
                    # refractors fail depth testing nondeterministically.
                    storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
                    stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                    stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
                    initialLayout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    finalLayout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                ),
            ]
            optical_color_ref = vk.VkAttachmentReference(
                attachment=0,
                layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            )
            optical_depth_ref = vk.VkAttachmentReference(
                attachment=1,
                layout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            )
            optical_render_pass = remember("render_pass", vk.vkCreateRenderPass(
                self.device, vk.VkRenderPassCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                    attachmentCount=2, pAttachments=optical_attachments,
                    subpassCount=1, pSubpasses=[vk.VkSubpassDescription(
                        pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                        colorAttachmentCount=1,
                        pColorAttachments=[optical_color_ref],
                        pDepthStencilAttachment=optical_depth_ref,
                    )],
                ), None,
            ))
            optical_framebuffer = remember("framebuffer", vk.vkCreateFramebuffer(
                self.device, vk.VkFramebufferCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                    renderPass=optical_render_pass,
                    attachmentCount=2,
                    pAttachments=[optical_view, optical_depth_view],
                    width=width, height=height, layers=1,
                ), None,
            ))
            optical_ping_framebuffer = remember(
                "framebuffer", vk.vkCreateFramebuffer(
                    self.device, vk.VkFramebufferCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                        renderPass=optical_render_pass,
                        attachmentCount=2,
                        pAttachments=[view, optical_ping_depth_view],
                        width=width, height=height, layers=1,
                    ), None,
                ),
            )
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
        def graphics_pipeline(pass_kind="opaque", target_render_pass=None):
            target_render_pass = target_render_pass or render_pass
            optical_pass = pass_kind in {
                "optical-opaque", "transmissive", "transparent",
            }
            transparent_pass = pass_kind in {"transmissive", "transparent"}
            pipeline_key = (
                mesh.layout, self.state, width, height,
                pass_kind, target_render_pass,
            )
            cached = self._pipelines.get(pipeline_key)
            if cached is not None:
                return cached
            alpha_blend = bool(
                transparent_pass or self.state.blend_mode == "alpha"
            )
            additive_blend = self.state.blend_mode == "additive"
            created = vk.vkCreateGraphicsPipelines(
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
                    polygonMode=vk.VK_POLYGON_MODE_FILL,
                    # Screen-space transmission shades the camera-facing
                    # surface and traces through the opaque buffers.  Letting
                    # both sphere hemispheres reach the same pixel made their
                    # fragments race; writing depth fixed that race but
                    # incorrectly hid farther transparent objects.  Back-face
                    # culling resolves the surface locally while preserving
                    # back-to-front transparency composition.
                    cullMode=(
                        vk.VK_CULL_MODE_BACK_BIT
                        if pass_kind == "transmissive"
                        else cull_modes[self.state.cull_mode]
                    ),
                    frontFace=front_faces[self.state.front_face], lineWidth=1.0),
                pMultisampleState=vk.VkPipelineMultisampleStateCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                    rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT),
                pDepthStencilState=vk.VkPipelineDepthStencilStateCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                    depthTestEnable=vk.VK_TRUE if self.state.depth_test else vk.VK_FALSE,
                    depthWriteEnable=(
                        vk.VK_TRUE
                        if self.state.depth_write
                        and pass_kind in {"opaque", "transmissive"}
                        else vk.VK_FALSE
                    ),
                    depthCompareOp=(
                        vk.VK_COMPARE_OP_LESS_OR_EQUAL
                        if optical_pass and screen_space_optics else
                        compares[self.state.depth_compare]
                    ),
                ),
                pColorBlendState=vk.VkPipelineColorBlendStateCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                    attachmentCount=1, pAttachments=[vk.VkPipelineColorBlendAttachmentState(
                        blendEnable=(
                            vk.VK_TRUE if alpha_blend or additive_blend
                            else vk.VK_FALSE
                        ),
                        srcColorBlendFactor=(
                            vk.VK_BLEND_FACTOR_SRC_ALPHA
                            if alpha_blend
                            else vk.VK_BLEND_FACTOR_ONE
                        ),
                        dstColorBlendFactor=(
                            vk.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA
                            if alpha_blend
                            else vk.VK_BLEND_FACTOR_ONE
                        ),
                        colorBlendOp=vk.VK_BLEND_OP_ADD,
                        srcAlphaBlendFactor=vk.VK_BLEND_FACTOR_ONE,
                        dstAlphaBlendFactor=(
                            vk.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA
                            if alpha_blend
                            else vk.VK_BLEND_FACTOR_ONE
                        ),
                        alphaBlendOp=vk.VK_BLEND_OP_ADD,
                        colorWriteMask=0xF)]),
                layout=layout, renderPass=target_render_pass, subpass=0,
                )], None)[0]
            self._pipelines[pipeline_key] = created
            return created

        pipeline = graphics_pipeline("opaque")
        optical_opaque_pipeline = (
            graphics_pipeline("optical-opaque", optical_render_pass)
            if screen_space_optics
            and mesh.resources.get("optical_opaque_index_count", 0)
            else None
        )
        transmissive_pipeline = (
            graphics_pipeline("transmissive", optical_render_pass)
            if screen_space_optics
            and mesh.resources.get("optical_transmissive_index_count", 0)
            else None
        )
        transparent_pipeline = (
            graphics_pipeline(
                "transparent",
                optical_render_pass if screen_space_optics else render_pass,
            )
            if (mesh.resources.get("transparent")
                or mesh.resources.get("optical_index_count", 0)) else None
        )
        if screen_space_optics and descriptor_set is not None:
            vk.vkUpdateDescriptorSets(self.device, 6, [
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=optical_descriptor_set, dstBinding=6,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    pImageInfo=[vk.VkDescriptorImageInfo(
                        imageView=view,
                        imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    )],
                ),
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=optical_descriptor_set, dstBinding=7,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    pImageInfo=[vk.VkDescriptorImageInfo(
                        imageView=optical_ping_depth_view,
                        imageLayout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                    )],
                ),
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=optical_ping_descriptor_set, dstBinding=6,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    pImageInfo=[vk.VkDescriptorImageInfo(
                        imageView=optical_view,
                        imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    )],
                ),
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=optical_ping_descriptor_set, dstBinding=7,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    pImageInfo=[vk.VkDescriptorImageInfo(
                        imageView=optical_depth_view,
                        imageLayout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                    )],
                ),
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=optical_immutable_descriptor_set, dstBinding=6,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    pImageInfo=[vk.VkDescriptorImageInfo(
                        imageView=optical_immutable_view,
                        imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    )],
                ),
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=optical_immutable_descriptor_set, dstBinding=7,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE,
                    pImageInfo=[vk.VkDescriptorImageInfo(
                        imageView=depth_view,
                        imageLayout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL,
                    )],
                ),
            ], 0, None)
        host = vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        vertex_payload = (
            mesh.vertices.tobytes()
            if mesh.vertices.nbytes else bytes(max(mesh.layout.stride, 8))
        )
        vertex_buffer, vertex_memory = self._buffer(
            len(vertex_payload), vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            host, vertex_payload,
        )
        resources.extend((("buffer", vertex_buffer), ("memory", vertex_memory)))
        index_buffer = None
        if mesh.indices is not None and mesh.indices.nbytes:
            index_buffer, index_memory = self._buffer(mesh.indices.nbytes, vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT, host, mesh.indices.tobytes())
            resources.extend((("buffer", index_buffer), ("memory", index_memory)))
        readback_size = width * height * 4 * np.dtype(np.float16).itemsize
        readback = readback_memory = None
        opaque_readback = opaque_readback_memory = None
        depth_readback = depth_readback_memory = None
        if not present or diagnostic_readback:
            readback, readback_memory = self._buffer(
                readback_size, vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT, host,
            )
            resources.extend((("buffer", readback), ("memory", readback_memory)))
            if diagnostic_readback and screen_space_optics:
                opaque_readback, opaque_readback_memory = self._buffer(
                    readback_size, vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT, host,
                )
                resources.extend((
                    ("buffer", opaque_readback),
                    ("memory", opaque_readback_memory),
                ))
                depth_readback, depth_readback_memory = self._buffer(
                    width * height * np.dtype(np.float32).itemsize,
                    vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT, host,
                )
                resources.extend((
                    ("buffer", depth_readback),
                    ("memory", depth_readback_memory),
                ))
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
            _sx, _sy, shadow_width, shadow_height, _aw, _ah = (
                mesh.resources["shadow_rectangle"]
            )
            vk.vkCmdBeginRenderPass(command, vk.VkRenderPassBeginInfo(
                sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                renderPass=shadow_render_pass, framebuffer=shadow_framebuffer,
                renderArea=vk.VkRect2D(
                    offset=vk.VkOffset2D(x=0, y=0),
                    extent=vk.VkExtent2D(
                        width=int(shadow_width),
                        height=int(shadow_height),
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
            authored_opaque_count = int(mesh.resources.get(
                "opaque_index_count", mesh.indices.size,
            ))
            opaque_count = (
                int(mesh.resources.get("opaque_prepass_index_count", 0))
                if screen_space_optics else authored_opaque_count
            )
            if opaque_count:
                vk.vkCmdDrawIndexed(command, opaque_count, 1, 0, 0, 0)
            transparent_count = mesh.indices.size - opaque_count
            if transparent_count and not screen_space_optics:
                vk.vkCmdBindPipeline(
                    command, vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                    transparent_pipeline,
                )
                vk.vkCmdDrawIndexed(
                    command, transparent_count, 1, opaque_count, 0, 0,
                )
        vk.vkCmdEndRenderPass(command)
        final_image = image
        final_view = view
        final_depth = depth_image
        final_depth_view = depth_view
        if screen_space_optics:
            color_range = vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0, levelCount=1,
                baseArrayLayer=0, layerCount=1,
            )
            depth_range = vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_DEPTH_BIT,
                baseMipLevel=0, levelCount=1,
                baseArrayLayer=0, layerCount=1,
            )
            if diagnostic_readback:
                vk.vkCmdPipelineBarrier(
                    command, vk.VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT,
                    vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                    0, None, 0, None, 1, [vk.VkImageMemoryBarrier(
                        sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        srcAccessMask=(
                            vk.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
                        ),
                        dstAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
                        oldLayout=(
                            vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                        ),
                        newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        image=depth_image, subresourceRange=depth_range,
                    )],
                )
                vk.vkCmdCopyImageToBuffer(
                    command, depth_image,
                    vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    depth_readback, 1, [vk.VkBufferImageCopy(
                        bufferOffset=0, bufferRowLength=0, bufferImageHeight=0,
                        imageSubresource=vk.VkImageSubresourceLayers(
                            aspectMask=vk.VK_IMAGE_ASPECT_DEPTH_BIT,
                            mipLevel=0, baseArrayLayer=0, layerCount=1,
                        ),
                        imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
                        imageExtent=vk.VkExtent3D(
                            width=width, height=height, depth=1,
                        ),
                    )],
                )
                depth_size = width * height * np.dtype(np.float32).itemsize
                vk.vkCmdPipelineBarrier(
                    command, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                    (vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                     | vk.VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT), 0,
                    0, None, 1, [vk.VkBufferMemoryBarrier(
                        sType=vk.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                        srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                        dstAccessMask=vk.VK_ACCESS_HOST_READ_BIT,
                        srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        buffer=depth_readback, offset=0, size=depth_size,
                    )], 1, [vk.VkImageMemoryBarrier(
                        sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        srcAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
                        dstAccessMask=(
                            vk.VK_ACCESS_SHADER_READ_BIT
                            | vk.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                        ),
                        oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        newLayout=(
                            vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                        ),
                        srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        image=depth_image, subresourceRange=depth_range,
                    )],
                )
            # Preserve the opaque depth texture for screen-space sampling, but
            # seed a separate writable depth attachment from it.  Otherwise
            # front and rear faces of transmissive meshes race through a
            # read-only depth pass and the selected surface changes between
            # frames (most visibly at low resolution).
            vk.vkCmdPipelineBarrier(
                command,
                (vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                 | vk.VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT),
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                0, None, 0, None, 3, [
                    vk.VkImageMemoryBarrier(
                        sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        srcAccessMask=(
                            vk.VK_ACCESS_SHADER_READ_BIT
                            | vk.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
                        ),
                        dstAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
                        oldLayout=(
                            vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                        ),
                        newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        image=depth_image, subresourceRange=depth_range,
                    ),
                    vk.VkImageMemoryBarrier(
                        sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        srcAccessMask=0,
                        dstAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                        oldLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
                        newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        image=optical_depth_image,
                        subresourceRange=depth_range,
                    ),
                    vk.VkImageMemoryBarrier(
                        sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        srcAccessMask=0,
                        dstAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                        oldLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
                        newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        image=optical_ping_depth_image,
                        subresourceRange=depth_range,
                    ),
                ],
            )
            depth_layers = vk.VkImageSubresourceLayers(
                aspectMask=vk.VK_IMAGE_ASPECT_DEPTH_BIT,
                mipLevel=0, baseArrayLayer=0, layerCount=1,
            )
            vk.vkCmdCopyImage(
                command,
                depth_image, vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                optical_depth_image, vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, [vk.VkImageCopy(
                    srcSubresource=depth_layers,
                    srcOffset=vk.VkOffset3D(x=0, y=0, z=0),
                    dstSubresource=depth_layers,
                    dstOffset=vk.VkOffset3D(x=0, y=0, z=0),
                    extent=vk.VkExtent3D(
                        width=width, height=height, depth=1,
                    ),
                )],
            )
            vk.vkCmdCopyImage(
                command,
                depth_image, vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                optical_ping_depth_image,
                vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, [vk.VkImageCopy(
                    srcSubresource=depth_layers,
                    srcOffset=vk.VkOffset3D(x=0, y=0, z=0),
                    dstSubresource=depth_layers,
                    dstOffset=vk.VkOffset3D(x=0, y=0, z=0),
                    extent=vk.VkExtent3D(
                        width=width, height=height, depth=1,
                    ),
                )],
            )
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                (vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                 | vk.VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT), 0,
                0, None, 0, None, 3, [
                    vk.VkImageMemoryBarrier(
                        sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        srcAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
                        dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                        oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        newLayout=(
                            vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                        ),
                        srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        image=depth_image, subresourceRange=depth_range,
                    ),
                    vk.VkImageMemoryBarrier(
                        sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                        dstAccessMask=(
                            vk.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                            | vk.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
                        ),
                        oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        newLayout=(
                            vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                        ),
                        srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        image=optical_depth_image,
                        subresourceRange=depth_range,
                    ),
                    vk.VkImageMemoryBarrier(
                        sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                        dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                        oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        newLayout=(
                            vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                        ),
                        srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        image=optical_ping_depth_image,
                        subresourceRange=depth_range,
                    ),
                ],
            )
            vk.vkCmdPipelineBarrier(
                command,
                vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                0, None, 0, None, 3, [
                    vk.VkImageMemoryBarrier(
                        sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        srcAccessMask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                        dstAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
                        oldLayout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        image=image, subresourceRange=color_range,
                    ),
                    vk.VkImageMemoryBarrier(
                        sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        srcAccessMask=0,
                        dstAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                        oldLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
                        newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        image=optical_image, subresourceRange=color_range,
                    ),
                    vk.VkImageMemoryBarrier(
                        sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        srcAccessMask=0,
                        dstAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                        oldLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
                        newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        image=optical_immutable_image,
                        subresourceRange=color_range,
                    ),
                ],
            )
            layers = vk.VkImageSubresourceLayers(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                mipLevel=0, baseArrayLayer=0, layerCount=1,
            )
            vk.vkCmdCopyImage(
                command,
                image, vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                optical_image, vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, [vk.VkImageCopy(
                    srcSubresource=layers,
                    srcOffset=vk.VkOffset3D(x=0, y=0, z=0),
                    dstSubresource=layers,
                    dstOffset=vk.VkOffset3D(x=0, y=0, z=0),
                    extent=vk.VkExtent3D(width=width, height=height, depth=1),
                )],
            )
            vk.vkCmdCopyImage(
                command,
                image, vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                optical_immutable_image,
                vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, [vk.VkImageCopy(
                    srcSubresource=layers,
                    srcOffset=vk.VkOffset3D(x=0, y=0, z=0),
                    dstSubresource=layers,
                    dstOffset=vk.VkOffset3D(x=0, y=0, z=0),
                    extent=vk.VkExtent3D(width=width, height=height, depth=1),
                )],
            )
            if diagnostic_readback:
                vk.vkCmdCopyImageToBuffer(
                    command, image, vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    opaque_readback, 1, [vk.VkBufferImageCopy(
                        bufferOffset=0, bufferRowLength=0, bufferImageHeight=0,
                        imageSubresource=vk.VkImageSubresourceLayers(
                            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                            mipLevel=0, baseArrayLayer=0, layerCount=1,
                        ),
                        imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
                        imageExtent=vk.VkExtent3D(
                            width=width, height=height, depth=1,
                        ),
                    )],
                )
                vk.vkCmdPipelineBarrier(
                    command, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                    vk.VK_PIPELINE_STAGE_HOST_BIT, 0,
                    0, None, 1, [vk.VkBufferMemoryBarrier(
                        sType=vk.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                        srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                        dstAccessMask=vk.VK_ACCESS_HOST_READ_BIT,
                        srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        buffer=opaque_readback, offset=0, size=readback_size,
                    )], 0, None,
                )
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                (vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                 | vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT), 0,
                0, None, 0, None, 3, [
                    vk.VkImageMemoryBarrier(
                        sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        srcAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
                        dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                        oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                        newLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        image=image, subresourceRange=color_range,
                    ),
                    vk.VkImageMemoryBarrier(
                        sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                        dstAccessMask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                        oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        newLayout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                        srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        image=optical_image, subresourceRange=color_range,
                    ),
                    vk.VkImageMemoryBarrier(
                        sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                        srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                        dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                        oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                        newLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                        image=optical_immutable_image,
                        subresourceRange=color_range,
                    ),
                ],
            )
            vk.vkCmdBeginRenderPass(command, vk.VkRenderPassBeginInfo(
                sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                renderPass=optical_render_pass,
                framebuffer=optical_framebuffer,
                renderArea=vk.VkRect2D(
                    offset=vk.VkOffset2D(x=0, y=0),
                    extent=vk.VkExtent2D(width=width, height=height),
                ),
                clearValueCount=0, pClearValues=None,
            ), vk.VK_SUBPASS_CONTENTS_INLINE)
            vk.vkCmdBindDescriptorSets(
                command, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, layout,
                0, 1, [optical_descriptor_set], 0, None,
            )
            vk.vkCmdBindVertexBuffers(
                command, 0, 1, [vertex_buffer], [0],
            )
            vk.vkCmdBindIndexBuffer(
                command, index_buffer, 0, vk.VK_INDEX_TYPE_UINT32,
            )
            optical_opaque_count = int(mesh.resources.get(
                "optical_opaque_index_count", 0,
            ))
            alpha_transparent_count = int(mesh.resources.get(
                "transparent_index_count", 0,
            ))
            transmissive_count = int(mesh.resources.get(
                "optical_transmissive_index_count", 0,
            ))
            authored_transparent_count = (
                alpha_transparent_count - transmissive_count
            )
            if optical_opaque_count:
                vk.vkCmdBindPipeline(
                    command, vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                    optical_opaque_pipeline,
                )
                vk.vkCmdDrawIndexed(
                    command, optical_opaque_count, 1, opaque_count, 0, 0,
                )
            vk.vkCmdEndRenderPass(command)
            final_image = optical_image
            final_view = optical_view
            final_framebuffer = optical_framebuffer
            final_depth = optical_depth_image
            final_depth_view = optical_depth_view

            def composite_optical_layer(
                source_image, destination_image, destination_framebuffer,
                source_depth, destination_depth, source_descriptors,
                draw_pipeline, draw_count, first_index,
            ):
                vk.vkCmdPipelineBarrier(
                    command,
                    (vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                     | vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT),
                    vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                    0, None, 0, None, 4, [
                        vk.VkImageMemoryBarrier(
                            sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                            srcAccessMask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                            dstAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
                            oldLayout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                            image=source_image, subresourceRange=color_range,
                        ),
                        vk.VkImageMemoryBarrier(
                            sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                            srcAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                            dstAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                            oldLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                            image=destination_image,
                            subresourceRange=color_range,
                        ),
                        vk.VkImageMemoryBarrier(
                            sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                            srcAccessMask=(
                                vk.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
                            ),
                            dstAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
                            oldLayout=(
                                vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                            ),
                            newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                            image=source_depth,
                            subresourceRange=depth_range,
                        ),
                        vk.VkImageMemoryBarrier(
                            sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                            srcAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                            dstAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                            oldLayout=(
                                vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                            ),
                            newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                            image=destination_depth,
                            subresourceRange=depth_range,
                        ),
                    ],
                )
                vk.vkCmdCopyImage(
                    command,
                    source_image, vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    destination_image, vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1, [vk.VkImageCopy(
                        srcSubresource=layers,
                        srcOffset=vk.VkOffset3D(x=0, y=0, z=0),
                        dstSubresource=layers,
                        dstOffset=vk.VkOffset3D(x=0, y=0, z=0),
                        extent=vk.VkExtent3D(
                            width=width, height=height, depth=1,
                        ),
                    )],
                )
                vk.vkCmdCopyImage(
                    command,
                    source_depth, vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    destination_depth, vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    1, [vk.VkImageCopy(
                        srcSubresource=depth_layers,
                        srcOffset=vk.VkOffset3D(x=0, y=0, z=0),
                        dstSubresource=depth_layers,
                        dstOffset=vk.VkOffset3D(x=0, y=0, z=0),
                        extent=vk.VkExtent3D(
                            width=width, height=height, depth=1,
                        ),
                    )],
                )
                vk.vkCmdPipelineBarrier(
                    command, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                    (vk.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
                     | vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT), 0,
                    0, None, 0, None, 4, [
                        vk.VkImageMemoryBarrier(
                            sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                            srcAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
                            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                            oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            newLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                            image=source_image, subresourceRange=color_range,
                        ),
                        vk.VkImageMemoryBarrier(
                            sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                            srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                            dstAccessMask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                            oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            newLayout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                            image=destination_image,
                            subresourceRange=color_range,
                        ),
                        vk.VkImageMemoryBarrier(
                            sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                            srcAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
                            dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                            oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            newLayout=(
                                vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL
                            ),
                            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                            image=source_depth,
                            subresourceRange=depth_range,
                        ),
                        vk.VkImageMemoryBarrier(
                            sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                            srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                            dstAccessMask=(
                                vk.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT
                                | vk.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT
                            ),
                            oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            newLayout=(
                                vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
                            ),
                            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                            image=destination_depth,
                            subresourceRange=depth_range,
                        ),
                    ],
                )
                vk.vkCmdBeginRenderPass(
                    command, vk.VkRenderPassBeginInfo(
                        sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
                        renderPass=optical_render_pass,
                        framebuffer=destination_framebuffer,
                        renderArea=vk.VkRect2D(
                            offset=vk.VkOffset2D(x=0, y=0),
                            extent=vk.VkExtent2D(width=width, height=height),
                        ),
                        clearValueCount=0, pClearValues=None,
                    ), vk.VK_SUBPASS_CONTENTS_INLINE,
                )
                vk.vkCmdBindDescriptorSets(
                    command, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, layout,
                    0, 1, [source_descriptors], 0, None,
                )
                vk.vkCmdBindPipeline(
                    command, vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                    draw_pipeline,
                )
                vk.vkCmdBindVertexBuffers(
                    command, 0, 1, [vertex_buffer], [0],
                )
                vk.vkCmdBindIndexBuffer(
                    command, index_buffer, 0, vk.VK_INDEX_TYPE_UINT32,
                )
                vk.vkCmdDrawIndexed(
                    command, draw_count, 1, first_index, 0, 0,
                )
                vk.vkCmdEndRenderPass(command)

            transmissive_counts = tuple(mesh.resources.get(
                "optical_transmissive_index_counts", (),
            ))
            retained_counts = transmissive_counts[
                -int(self.config.screen_space_optical_layers):
            ]
            transmissive_layers_overlap = bool(mesh.resources.get(
                "optical_transmissive_layers_overlap", True,
            ))
            next_index = (
                opaque_count + optical_opaque_count
                + sum(transmissive_counts[:-len(retained_counts)])
                if retained_counts else
                opaque_count + optical_opaque_count
            )
            for draw_count in retained_counts:
                if final_image == optical_image:
                    destination_image = image
                    destination_framebuffer = optical_ping_framebuffer
                    destination_depth = optical_ping_depth_image
                    accumulated_descriptors = optical_ping_descriptor_set
                else:
                    destination_image = optical_image
                    destination_framebuffer = optical_framebuffer
                    destination_depth = optical_depth_image
                    accumulated_descriptors = optical_descriptor_set
                if not transmissive_layers_overlap:
                    source_descriptors = optical_immutable_descriptor_set
                else:
                    source_descriptors = accumulated_descriptors
                composite_optical_layer(
                    final_image, destination_image, destination_framebuffer,
                    final_depth, destination_depth, source_descriptors,
                    transmissive_pipeline,
                    int(draw_count), int(next_index),
                )
                final_image = destination_image
                final_view = (
                    view if destination_image == image else optical_view
                )
                final_framebuffer = destination_framebuffer
                final_depth = destination_depth
                final_depth_view = (
                    optical_ping_depth_view
                    if destination_depth == optical_ping_depth_image
                    else optical_depth_view
                )
                next_index += int(draw_count)
            if authored_transparent_count:
                if final_image == optical_image:
                    destination_image = image
                    destination_framebuffer = optical_ping_framebuffer
                    destination_depth = optical_ping_depth_image
                    source_descriptors = optical_ping_descriptor_set
                else:
                    destination_image = optical_image
                    destination_framebuffer = optical_framebuffer
                    destination_depth = optical_depth_image
                    source_descriptors = optical_descriptor_set
                composite_optical_layer(
                    final_image, destination_image, destination_framebuffer,
                    final_depth, destination_depth, source_descriptors,
                    transparent_pipeline,
                    authored_transparent_count,
                    opaque_count + optical_opaque_count + transmissive_count,
                )
                final_image = destination_image
                final_view = (
                    view if destination_image == image else optical_view
                )
                final_depth = destination_depth
                final_depth_view = (
                    optical_ping_depth_view
                    if destination_depth == optical_ping_depth_image
                    else optical_depth_view
                )
        final_image, final_view, volume_camera_memory = self._record_volume_composite(
            command, mesh, width, height, final_image, final_view,
            final_depth, final_depth_view, shadow_view, shadow_sampler, remember,
        )
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
                image=final_image,
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
                command, final_image, vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
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
            if diagnostic_readback:
                vk.vkCmdCopyImageToBuffer(
                    command, final_image,
                    vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    readback, 1, [vk.VkBufferImageCopy(
                        bufferOffset=0, bufferRowLength=0, bufferImageHeight=0,
                        imageSubresource=vk.VkImageSubresourceLayers(
                            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                            mipLevel=0, baseArrayLayer=0, layerCount=1,
                        ),
                        imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
                        imageExtent=vk.VkExtent3D(
                            width=width, height=height, depth=1,
                        ),
                    )],
                )
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
            vk.vkCmdCopyImageToBuffer(command, final_image, vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback, 1, [vk.VkBufferImageCopy(
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
            self._present_cache_serial += 1
            self._present_cache[cache_key] = {
                "command": command,
                "setup_command": setup_command,
                "resources": resources,
                "vertex_memory": vertex_memory,
                "camera_memory": camera_memory,
                "opaque_camera_memory": opaque_camera_memory,
                "volume_camera_memory": volume_camera_memory,
                "last_fence": fence,
                "diagnostic_slot": self._present_cache_serial,
                "diagnostic_readback_memory": (
                    readback_memory if diagnostic_readback else None
                ),
                "diagnostic_readback_size": (
                    readback_size if diagnostic_readback else 0
                ),
                "diagnostic_opaque_readback_memory": (
                    opaque_readback_memory if diagnostic_readback else None
                ),
                "diagnostic_depth_readback_memory": (
                    depth_readback_memory if diagnostic_readback else None
                ),
                "diagnostic_depth_readback_size": (
                    width * height * np.dtype(np.float32).itemsize
                    if diagnostic_readback else 0
                ),
                "last_submission": self._present_submission_sequence,
            }
            self.last_timings.update(
                swapchain_image_index=int(image_index),
                resident_cache_slot=self._present_cache_serial,
            )
        else:
            vk.vkFreeCommandBuffers(
                self.device, self.command_pool, 1, [command],
            )
            self._destroy_frame_resources(resources)
        return result

    def present_frame(
        self, scene, camera, width, height, *, frame_index=0,
        surface_size=None, diagnostic_readback=False,
    ):
        """Render directly to the external Vulkan surface without readback."""
        import time
        started = time.perf_counter()
        self.probe_capture.refresh(self, scene)
        # Diagnostic captures intentionally serialize the whole device. This
        # distinguishes resource-lifetime/synchronization faults from
        # deterministic shader or input-data faults without affecting normal
        # presentation performance.
        if diagnostic_readback:
            self.vk.vkDeviceWaitIdle(self.device)
        scene_token = (id(scene), scene.revision)
        if self._present_scene_token != scene_token:
            self.vk.vkDeviceWaitIdle(self.device)
            self._clear_present_cache()
            self._present_scene_token = scene_token
            self._present_cache_generation = None
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
        camera_ordered = bool(
            self.config.optical_quality == "screen-space"
            or any(
                mesh.material.alpha_mode == "blend"
                for mesh in scene.visible_meshes
            )
        )
        if resources_prepared or camera_ordered or self._prepared_present_mesh is None:
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
        camera_data["viewport_optics"][0] = (width, height, -1.0 if self.config.optical_quality == "screen-space" else 0.0, self.config.screen_space_ray_steps)
        camera_data["optical_diagnostic"][0, 0] = _OPTICAL_DEBUG_MODES[
            self.config.optical_debug_view
        ]
        camera_data["optical_diagnostic"][0, 1] = mesh.resources.get(
            "light_count", 0,
        )
        camera_data["optical_diagnostic"][0, 2] = mesh.resources.get(
            "shadow_count", 0,
        )
        camera_data["optical_diagnostic"][0, 3] = float(mesh.resources.get(
            "optical_transmissive_layers_nested", False,
        ))
        mesh.resources["camera_uniform"] = camera_data.tobytes()
        mesh.resources["volume_inverse_view_projection"] = np.linalg.inv(
            camera_matrix(camera, width, height),
        )
        mesh.resources["volume_camera_position"] = (*camera.position, 1.0)
        pack_ms = (time.perf_counter() - pack_started) * 1000.0
        camera_order_token = mesh.resources.get("camera_order_token", ())
        self._activate_present_cache_generation(camera_order_token)
        self.render(
            mesh, width, height, present=True, surface_size=surface_size,
            cache_token=(
                scene_token, camera_order_token, bool(diagnostic_readback),
            ),
            diagnostic_readback=diagnostic_readback,
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
        self.probe_capture.refresh(self, scene)
        mesh = scene_mesh(
            scene, camera, width, height, self.config,
            native_shadow_maps=True, gpu_camera=True,
        )
        camera_data = np.zeros(1, CAMERA_DTYPE)
        camera_data["view_projection"][0] = camera_matrix(camera, width, height).T
        camera_data["position_exposure"][0] = (*camera.position, 1.0)
        camera_data["viewport_optics"][0] = (width, height, -1.0 if self.config.optical_quality == "screen-space" else 0.0, self.config.screen_space_ray_steps)
        camera_data["optical_diagnostic"][0, 0] = _OPTICAL_DEBUG_MODES[
            self.config.optical_debug_view
        ]
        camera_data["optical_diagnostic"][0, 1] = mesh.resources.get(
            "light_count", 0,
        )
        camera_data["optical_diagnostic"][0, 2] = mesh.resources.get(
            "shadow_count", 0,
        )
        camera_data["optical_diagnostic"][0, 3] = float(mesh.resources.get(
            "optical_transmissive_layers_nested", False,
        ))
        mesh.resources["camera_uniform"] = camera_data.tobytes()
        mesh.resources["volume_inverse_view_projection"] = np.linalg.inv(
            camera_matrix(camera, width, height),
        )
        mesh.resources["volume_camera_position"] = (*camera.position, 1.0)
        image = self.render(mesh, width, height)
        self.last_timings = {"total_ms": (time.perf_counter() - started) * 1000.0}
        return self._post.process(image, scene, camera)

    def _product_pipeline(self, render_pass, layout, width, height):
        vk = self.vk
        key = (layout, int(width), int(height))
        pipeline = self._product_pipelines.get(key)
        if pipeline is not None:
            return pipeline
        format_map = {
            "float32": vk.VK_FORMAT_R32_SFLOAT,
            "float32x2": vk.VK_FORMAT_R32G32_SFLOAT,
            "float32x3": vk.VK_FORMAT_R32G32B32_SFLOAT,
            "float32x4": vk.VK_FORMAT_R32G32B32A32_SFLOAT,
        }
        stages = [
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_VERTEX_BIT,
                module=self._product_vertex_module, pName="main",
            ),
            vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                module=self._product_fragment_module, pName="main",
            ),
        ]
        pipeline = vk.vkCreateGraphicsPipelines(
            self.device, vk.VK_NULL_HANDLE, 1,
            [vk.VkGraphicsPipelineCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
                stageCount=2,
                pStages=stages,
                pVertexInputState=vk.VkPipelineVertexInputStateCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                    vertexBindingDescriptionCount=1,
                    pVertexBindingDescriptions=[vk.VkVertexInputBindingDescription(
                        binding=0, stride=layout.stride,
                        inputRate=vk.VK_VERTEX_INPUT_RATE_VERTEX,
                    )],
                    vertexAttributeDescriptionCount=len(layout.attributes),
                    pVertexAttributeDescriptions=[
                        vk.VkVertexInputAttributeDescription(
                            location=item.location, binding=0,
                            format=format_map[item.format], offset=item.offset,
                        ) for item in layout.attributes
                    ],
                ),
                pInputAssemblyState=vk.VkPipelineInputAssemblyStateCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                    topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
                ),
                pViewportState=vk.VkPipelineViewportStateCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                    viewportCount=1, pViewports=[vk.VkViewport(
                        x=0.0, y=float(height), width=float(width),
                        height=-float(height), minDepth=0.0, maxDepth=1.0,
                    )],
                    scissorCount=1, pScissors=[vk.VkRect2D(
                        offset=vk.VkOffset2D(x=0, y=0),
                        extent=vk.VkExtent2D(width=width, height=height),
                    )],
                ),
                pRasterizationState=vk.VkPipelineRasterizationStateCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                    polygonMode=vk.VK_POLYGON_MODE_FILL,
                    cullMode={
                        "none": vk.VK_CULL_MODE_NONE,
                        "front": vk.VK_CULL_MODE_FRONT_BIT,
                        "back": vk.VK_CULL_MODE_BACK_BIT,
                    }[self.state.cull_mode],
                    frontFace=(
                        vk.VK_FRONT_FACE_COUNTER_CLOCKWISE
                        if self.state.front_face == "ccw"
                        else vk.VK_FRONT_FACE_CLOCKWISE
                    ), lineWidth=1.0,
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
                    attachmentCount=2,
                    pAttachments=[vk.VkPipelineColorBlendAttachmentState(
                        blendEnable=vk.VK_FALSE,
                        srcColorBlendFactor=vk.VK_BLEND_FACTOR_ONE,
                        dstColorBlendFactor=vk.VK_BLEND_FACTOR_ZERO,
                        colorBlendOp=vk.VK_BLEND_OP_ADD,
                        srcAlphaBlendFactor=vk.VK_BLEND_FACTOR_ONE,
                        dstAlphaBlendFactor=vk.VK_BLEND_FACTOR_ZERO,
                        alphaBlendOp=vk.VK_BLEND_OP_ADD,
                        colorWriteMask=(vk.VK_COLOR_COMPONENT_R_BIT
                                        | vk.VK_COLOR_COMPONENT_G_BIT
                                        | vk.VK_COLOR_COMPONENT_B_BIT
                                        | vk.VK_COLOR_COMPONENT_A_BIT),
                    ) for _ in range(2)],
                ),
                layout=self._product_pipeline_layout, renderPass=render_pass,
                subpass=0,
            )], None,
        )[0]
        self._product_pipelines[key] = pipeline
        return pipeline

    def _render_native_products(self, mesh, width, height):
        """Rasterize typed geometry products in one Vulkan MRT pass."""
        if not len(mesh.vertices):
            return {
                "depth": np.ones((height, width), np.float32),
                "normal": np.zeros((height, width, 3), np.float32),
                "object_id": np.zeros((height, width), np.uint32),
                "motion": np.zeros((height, width, 2), np.float32),
            }
        vk = self.vk
        width, height = int(width), int(height)
        resources = []
        def remember(kind, handle):
            resources.append((kind, handle))
            return handle

        def image(format_name, aspect):
            handle = remember("image", vk.vkCreateImage(
                self.device, vk.VkImageCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                    imageType=vk.VK_IMAGE_TYPE_2D, format=format_name,
                    extent=vk.VkExtent3D(width=width, height=height, depth=1),
                    mipLevels=1, arrayLayers=1,
                    samples=vk.VK_SAMPLE_COUNT_1_BIT,
                    tiling=vk.VK_IMAGE_TILING_OPTIMAL,
                    usage=(
                        (vk.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                         if aspect == vk.VK_IMAGE_ASPECT_DEPTH_BIT
                         else vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)
                        | (0 if aspect == vk.VK_IMAGE_ASPECT_DEPTH_BIT
                           else vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT)
                    ),
                    sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
                    initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
                ), None,
            ))
            requirements = vk.vkGetImageMemoryRequirements(self.device, handle)
            memory = remember("memory", vk.vkAllocateMemory(
                self.device, vk.VkMemoryAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    allocationSize=requirements.size,
                    memoryTypeIndex=self._memory_type(
                        requirements.memoryTypeBits,
                        vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    ),
                ), None,
            ))
            vk.vkBindImageMemory(self.device, handle, memory, 0)
            view = remember("image_view", vk.vkCreateImageView(
                self.device, vk.VkImageViewCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    image=handle, viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                    format=format_name,
                    subresourceRange=vk.VkImageSubresourceRange(
                        aspectMask=aspect, baseMipLevel=0, levelCount=1,
                        baseArrayLayer=0, layerCount=1,
                    ),
                ), None,
            ))
            return handle, view

        formats = (
            vk.VK_FORMAT_R32G32B32A32_SFLOAT,
            vk.VK_FORMAT_R32G32B32A32_SFLOAT,
        )
        color_images = [image(item, vk.VK_IMAGE_ASPECT_COLOR_BIT) for item in formats]
        depth_image, depth_view = image(
            vk.VK_FORMAT_D32_SFLOAT, vk.VK_IMAGE_ASPECT_DEPTH_BIT,
        )
        attachments = [vk.VkAttachmentDescription(
            format=item, samples=vk.VK_SAMPLE_COUNT_1_BIT,
            loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
            storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
            stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            finalLayout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        ) for item in formats]
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
        color_references = [vk.VkAttachmentReference(
            attachment=index,
            layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        ) for index in range(2)]
        depth_reference = vk.VkAttachmentReference(
            attachment=2,
            layout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        )
        subpass = vk.VkSubpassDescription(
            pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            colorAttachmentCount=2,
            pColorAttachments=color_references,
            pDepthStencilAttachment=depth_reference,
        )
        render_pass = remember("render_pass", vk.vkCreateRenderPass(
            self.device, vk.VkRenderPassCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
                attachmentCount=3, pAttachments=attachments,
                subpassCount=1, pSubpasses=[subpass],
                dependencyCount=0, pDependencies=None,
            ), None,
        ))
        framebuffer = remember("framebuffer", vk.vkCreateFramebuffer(
            self.device, vk.VkFramebufferCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
                renderPass=render_pass, attachmentCount=3,
                pAttachments=[item[1] for item in color_images] + [depth_view],
                width=width, height=height, layers=1,
            ), None,
        ))
        vertex, vertex_memory = self._buffer(
            mesh.vertices.nbytes, vk.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            mesh.vertices.tobytes(),
        )
        resources.extend((("buffer", vertex), ("memory", vertex_memory)))
        index = None
        if mesh.indices is not None and mesh.indices.size:
            index, index_memory = self._buffer(
                mesh.indices.nbytes, vk.VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                mesh.indices.tobytes(),
            )
            resources.extend((("buffer", index), ("memory", index_memory)))
        uniform_payload = mesh.resources["geometry_product_camera"]
        uniform, uniform_memory = self._buffer(
            len(uniform_payload), vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            uniform_payload,
        )
        resources.extend((("buffer", uniform), ("memory", uniform_memory)))
        descriptor_pool = remember("descriptor_pool", vk.vkCreateDescriptorPool(
            self.device, vk.VkDescriptorPoolCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                maxSets=1, poolSizeCount=1,
                pPoolSizes=[vk.VkDescriptorPoolSize(
                    type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                    descriptorCount=1,
                )],
            ), None,
        ))
        descriptor = vk.vkAllocateDescriptorSets(
            self.device, vk.VkDescriptorSetAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptorPool=descriptor_pool, descriptorSetCount=1,
                pSetLayouts=[self._product_descriptor_layout],
            ),
        )[0]
        vk.vkUpdateDescriptorSets(self.device, 1, [vk.VkWriteDescriptorSet(
            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=descriptor, dstBinding=0, descriptorCount=1,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            pBufferInfo=[vk.VkDescriptorBufferInfo(
                buffer=uniform, offset=0, range=len(uniform_payload),
            )],
        )], 0, None)
        specs = (
            (color_images[0][0], vk.VK_IMAGE_ASPECT_COLOR_BIT, 16, np.float32, 4),
            (color_images[1][0], vk.VK_IMAGE_ASPECT_COLOR_BIT, 16, np.float32, 4),
        )
        readbacks = []
        for _image, _aspect, bytes_per_pixel, dtype, components in specs:
            size = width * height * bytes_per_pixel
            buffer, memory = self._buffer(
                size, vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            )
            resources.extend((("buffer", buffer), ("memory", memory)))
            readbacks.append((buffer, memory, size, dtype, components))
        command = vk.vkAllocateCommandBuffers(
            self.device, vk.VkCommandBufferAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                commandPool=self.command_pool,
                level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1,
            ),
        )[0]
        vk.vkBeginCommandBuffer(command, vk.VkCommandBufferBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        ))
        vk.vkCmdBeginRenderPass(command, vk.VkRenderPassBeginInfo(
            sType=vk.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            renderPass=render_pass, framebuffer=framebuffer,
            renderArea=vk.VkRect2D(
                offset=vk.VkOffset2D(x=0, y=0),
                extent=vk.VkExtent2D(width=width, height=height),
            ), clearValueCount=3,
            pClearValues=[
                vk.VkClearValue(color=vk.VkClearColorValue(float32=[0, 0, 0, 0])),
                vk.VkClearValue(color=vk.VkClearColorValue(float32=[0, 0, 0, 0])),
                vk.VkClearValue(depthStencil=vk.VkClearDepthStencilValue(depth=1.0, stencil=0)),
            ],
        ), vk.VK_SUBPASS_CONTENTS_INLINE)
        vk.vkCmdBindPipeline(
            command, vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            self._product_pipeline(render_pass, mesh.layout, width, height),
        )
        vk.vkCmdBindDescriptorSets(
            command, vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            self._product_pipeline_layout, 0, 1, [descriptor], 0, None,
        )
        vk.vkCmdBindVertexBuffers(command, 0, 1, [vertex], [0])
        if index is not None:
            vk.vkCmdBindIndexBuffer(command, index, 0, vk.VK_INDEX_TYPE_UINT32)
            vk.vkCmdDrawIndexed(command, mesh.indices.size, 1, 0, 0, 0)
        else:
            vk.vkCmdDraw(command, len(mesh.vertices), 1, 0, 0)
        vk.vkCmdEndRenderPass(command)
        vk.vkCmdPipelineBarrier(
            command, vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
            0, None, 0, None, len(color_images), [vk.VkImageMemoryBarrier(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                srcAccessMask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                dstAccessMask=vk.VK_ACCESS_TRANSFER_READ_BIT,
                oldLayout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                image=source,
                subresourceRange=vk.VkImageSubresourceRange(
                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                    baseMipLevel=0, levelCount=1,
                    baseArrayLayer=0, layerCount=1,
                ),
            ) for source, _view in color_images],
        )
        for (source, aspect, _bpp, _dtype, _components), (buffer, *_rest) in zip(specs, readbacks):
            vk.vkCmdCopyImageToBuffer(
                command, source, vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                buffer, 1, [vk.VkBufferImageCopy(
                    bufferOffset=0, bufferRowLength=0, bufferImageHeight=0,
                    imageSubresource=vk.VkImageSubresourceLayers(
                        aspectMask=aspect, mipLevel=0,
                        baseArrayLayer=0, layerCount=1,
                    ), imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
                    imageExtent=vk.VkExtent3D(width=width, height=height, depth=1),
                )],
            )
        vk.vkCmdPipelineBarrier(
            command, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_PIPELINE_STAGE_HOST_BIT, 0,
            0, None, len(readbacks), [vk.VkBufferMemoryBarrier(
                sType=vk.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                dstAccessMask=vk.VK_ACCESS_HOST_READ_BIT,
                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                buffer=buffer, offset=0, size=size,
            ) for buffer, _memory, size, _dtype, _components in readbacks],
            0, None,
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
            vk.vkWaitForFences(
                self.device, 1, [fence], vk.VK_TRUE, 10_000_000_000,
            )
            arrays = []
            for _buffer, memory, size, dtype, components in readbacks:
                mapped = vk.vkMapMemory(self.device, memory, 0, size, 0)
                values = np.frombuffer(bytes(mapped[:size]), dtype)
                vk.vkUnmapMemory(self.device, memory)
                values = values.reshape(
                    (height, width, components) if components > 1
                    else (height, width)
                )
                arrays.append(values)
        finally:
            vk.vkDestroyFence(self.device, fence, None)
            vk.vkFreeCommandBuffers(self.device, self.command_pool, 1, [command])
            self._destroy_frame_resources(resources)
        normal_depth, motion_object = arrays
        object_ids = np.rint(motion_object[..., 2]).astype(np.uint32)
        depth_output = normal_depth[..., 3].astype(np.float32)
        depth_output[object_ids == 0] = np.inf
        motion = motion_object[..., :2].astype(np.float32)
        motion[np.abs(motion) < 1e-6] = 0.0
        return {
            "depth": depth_output,
            "normal": normal_depth[..., :3].astype(np.float32),
            "object_id": object_ids,
            "motion": motion,
        }

    def render_products(self, scene, camera, width, height, *, outputs, samples=None, frame_index=0):
        mesh, next_history = geometry_product_mesh(
            scene, camera, width, height, self._output_history,
        )
        products = self._render_native_products(mesh, width, height)
        self._output_history = next_history
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
            camera_data["viewport_optics"][0] = (width, height, -1.0 if self.config.optical_quality == "screen-space" else 0.0, self.config.screen_space_ray_steps)
            camera_data["optical_diagnostic"][0, 0] = _OPTICAL_DEBUG_MODES[
                self.config.optical_debug_view
            ]
            camera_data["optical_diagnostic"][0, 1] = color_mesh.resources.get(
                "light_count", 0,
            )
            camera_data["optical_diagnostic"][0, 2] = color_mesh.resources.get(
                "shadow_count", 0,
            )
            camera_data["optical_diagnostic"][0, 3] = float(
                color_mesh.resources.get(
                    "optical_transmissive_layers_nested", False,
                )
            )
            color_mesh.resources["camera_uniform"] = camera_data.tobytes()
            color_mesh.resources["volume_inverse_view_projection"] = np.linalg.inv(
                camera_matrix(camera, width, height),
            )
            color_mesh.resources["volume_camera_position"] = (
                *camera.position, 1.0,
            )
            image = self.render(color_mesh, width, height)
            products["color"] = self._post.process(image, scene, camera)
        return {name: products[name] for name in outputs}

    @property
    def accumulated_frames(self):
        return self._post.accumulated_frames

    def reset_output_history(self):
        self._post.reset()
        self._output_history = None

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.vk.vkDeviceWaitIdle(self.device)
        except self.vk.VkErrorDeviceLost:
            # Preserve the original rendering error and continue releasing
            # host-side ownership after a lost device.
            pass
        self._destroy_swapchain_resources()
        for frame in self._present_frames:
            self.vk.vkDestroyFence(self.device, frame["fence"], None)
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
        for pipeline in self._product_pipelines.values():
            self.vk.vkDestroyPipeline(self.device, pipeline, None)
        self._product_pipelines.clear()
        for pipeline in self._volume_pipelines.values():
            self.vk.vkDestroyPipeline(self.device, pipeline, None)
        self._volume_pipelines.clear()
        self.vk.vkDestroyPipelineLayout(
            self.device, self._volume_pipeline_layout, None,
        )
        self.vk.vkDestroyDescriptorSetLayout(
            self.device, self._volume_descriptor_layout, None,
        )
        self.vk.vkDestroyShaderModule(
            self.device, self._volume_fragment_module, None,
        )
        self.vk.vkDestroyShaderModule(
            self.device, self._volume_vertex_module, None,
        )
        self.vk.vkDestroyPipelineLayout(
            self.device, self._product_pipeline_layout, None,
        )
        self.vk.vkDestroyDescriptorSetLayout(
            self.device, self._product_descriptor_layout, None,
        )
        self.vk.vkDestroyShaderModule(
            self.device, self._product_fragment_module, None,
        )
        self.vk.vkDestroyShaderModule(
            self.device, self._product_vertex_module, None,
        )
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
