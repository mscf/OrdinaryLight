"""Device services shared by built-in and application Vulkan renderers.

This module deliberately has no dependency on the GI core or its pipelines.
All use of a runtime's queue and command pool must be serialized by callers.
"""

from dataclasses import dataclass
import os
from pathlib import Path
from threading import RLock

import vulkan as vk
from ..targets._vulkan_version import vulkan_api_version

DEVICE_EXTENSIONS = (
    "VK_KHR_acceleration_structure",
    "VK_KHR_deferred_host_operations",
    "VK_KHR_ray_query",
)
EXTERNAL_INTEROP_DEVICE_EXTENSIONS = (
    "VK_KHR_external_memory",
    "VK_KHR_external_memory_fd",
    "VK_KHR_external_semaphore",
    "VK_KHR_external_semaphore_fd",
)
EXTERNAL_INTEROP_INSTANCE_EXTENSIONS = (
    "VK_KHR_external_memory_capabilities",
    "VK_KHR_external_semaphore_capabilities",
)
MAX_NATIVE_TEXTURES = 64


@dataclass(frozen=True)
class VulkanCapabilities:
    ray_query: bool
    ray_pipeline: bool
    native_textures: bool
    external_memory: bool
    presentation: bool
    # Built-in scene upload currently accepts triangle geometry only.
    custom_intersections: bool = False


class VulkanRuntime:
    """Own a Vulkan device independently of any rendering algorithm.

    The first runtime profile requires hardware ray queries. ``config`` accepts
    RendererConfig to preserve existing feature selection. No shaders are
    compiled during construction. A borrowed runtime outlives all consumers;
    close rejects live consumers rather than destroying their device.
    """

    def __init__(
        self,
        device_name=None,
        *,
        config=None,
        glfw_window=None,
        external_instance=None,
        external_surface=None,
        headless_surface=False,
    ):
        from ..targets.vulkan.api import RendererConfig

        if config is not None and device_name is not None:
            raise ValueError("Pass device_name or config, not both")
        if (external_instance is None) != (external_surface is None):
            raise ValueError(
                "external_instance and external_surface must be supplied together"
            )
        if glfw_window is not None and external_instance is not None:
            raise ValueError("Pass a GLFW window or an external surface, not both")
        if headless_surface and (
            glfw_window is not None or external_instance is not None
        ):
            raise ValueError("headless_surface cannot be combined with a surface")
        if external_instance is not None:
            if isinstance(external_instance, int):
                external_instance = vk.ffi.cast("VkInstance", external_instance)
            if isinstance(external_surface, int):
                external_surface = vk.ffi.cast("VkSurfaceKHR", external_surface)
        self.config = config or RendererConfig(device_name=device_name)
        self.lock = RLock()
        self._consumers = set()
        self._closed = False
        self.glfw_window = glfw_window
        self._external_instance = external_instance
        self._external_surface = external_surface
        self._headless_surface = bool(headless_surface)
        self._owns_instance = external_instance is None
        self._owns_surface = external_surface is None
        self.instance = self.device = self.surface = None
        self.command_pool = self.pipeline_cache = self.pipeline_cache_path = None
        self.native_textures_supported = False
        self.pipeline_statistics_supported = False
        self.ray_pipeline_supported = self.ray_pipeline_enabled = False
        self.ser_supported = self.ser_reordering_supported = False
        self.present_wait_supported = False
        self.ray_tracing_shader_group_handle_size = 0
        self.ray_tracing_shader_group_handle_alignment = 0
        self.ray_tracing_shader_group_base_alignment = 0
        try:
            self._create_instance_and_device(self.config.device_name)
            self._load_extension_functions()
            self._create_command_pool()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _name(properties):
        return str(properties.deviceName).split("\0", 1)[0]

    @property
    def capabilities(self):
        self.require_open()
        return VulkanCapabilities(
            True,
            self.ray_pipeline_enabled,
            self.native_textures_supported and self.config.wavefront_native_textures,
            self._headless_surface,
            self.surface is not None,
        )

    def require_open(self):
        if self._closed:
            raise RuntimeError("Vulkan runtime is closed")

    def retain(self, consumer):
        with self.lock:
            self.require_open()
            self._consumers.add(consumer)

    def release(self, consumer):
        with self.lock:
            self._consumers.discard(consumer)

    def memory_type(self, bits, flags):
        self.require_open()
        for index in range(self.memory_properties.memoryTypeCount):
            if (
                bits & (1 << index)
                and self.memory_properties.memoryTypes[index].propertyFlags & flags
                == flags
            ):
                return index
        raise RuntimeError(f"No Vulkan memory type satisfies {flags:#x}")

    def buffer(self, size, **options):
        from .resources import VulkanBuffer

        with self.lock:
            return VulkanBuffer(self, size, **options)

    def image(self, width, height, **options):
        from .resources import VulkanImage

        with self.lock:
            return VulkanImage(self, width, height, **options)

    def submit(self, recorder, *, resources=(), after=()):
        from .resources import submit

        return submit(self, recorder, resources=resources, after=after)

    def upload_scene(self, scene, *, config=None):
        from ..targets.vulkan.scene import VulkanSceneResources

        with self.lock:
            return VulkanSceneResources(self, scene, config=config)

    def close(self):
        with self.lock:
            if self._closed:
                return
            from .resources import VulkanCompletion

            for consumer in tuple(self._consumers):
                if isinstance(consumer, VulkanCompletion):
                    consumer.wait()
            if self._consumers:
                raise RuntimeError(
                    "Close Vulkan runtime consumers before their runtime"
                )
            if self.device is not None:
                try:
                    vk.vkDeviceWaitIdle(self.device)
                    self._save_pipeline_cache()
                except vk.VkErrorDeviceLost:
                    pass
                if self.command_pool is not None:
                    vk.vkDestroyCommandPool(self.device, self.command_pool, None)
                if self.pipeline_cache is not None:
                    vk.vkDestroyPipelineCache(self.device, self.pipeline_cache, None)
                vk.vkDestroyDevice(self.device, None)
                self.device = None
            if self.instance is not None:
                if self.surface is not None and self._owns_surface:
                    destroy = vk.vkGetInstanceProcAddr(
                        self.instance, "vkDestroySurfaceKHR"
                    )
                    destroy(self.instance, self.surface, None)
                if self._owns_instance:
                    vk.vkDestroyInstance(self.instance, None)
                self.instance = None
            self._closed = True

    def __enter__(self):
        self.require_open()
        return self

    def __exit__(self, *_exc):
        self.close()

    def _create_instance_and_device(self, requested_name):
        instance_extensions = []
        if self.glfw_window is not None:
            import glfw

            instance_extensions = glfw.get_required_instance_extensions()
        elif self._headless_surface:
            instance_extensions = [
                *EXTERNAL_INTEROP_INSTANCE_EXTENSIONS,
            ]
            available = {
                str(item.extensionName).split("\0", 1)[0]
                for item in vk.vkEnumerateInstanceExtensionProperties(None)
            }
            missing = set(instance_extensions) - available
            if missing:
                raise RuntimeError(
                    "Vulkan GPU interop requires unavailable instance "
                    f"extensions: {sorted(missing)}"
                )
        app = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="Ordinary Light",
            applicationVersion=vk.VK_MAKE_VERSION(0, 1, 0),
            pEngineName="Ordinary Light",
            engineVersion=vk.VK_MAKE_VERSION(0, 1, 0),
            apiVersion=vulkan_api_version(vk),
        )
        if self._external_instance is None:
            self.instance = vk.vkCreateInstance(
                vk.VkInstanceCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
                    pApplicationInfo=app,
                    enabledExtensionCount=len(instance_extensions),
                    ppEnabledExtensionNames=instance_extensions,
                ),
                None,
            )
        else:
            self.instance = self._external_instance

        if self._external_surface is not None:
            self.surface = self._external_surface
        elif self.glfw_window is not None:
            import glfw

            surface_pointer = vk.ffi.new("VkSurfaceKHR*")
            result = glfw.create_window_surface(
                self.instance, self.glfw_window, None, surface_pointer
            )
            if result != vk.VK_SUCCESS:
                raise RuntimeError(f"GLFW Vulkan surface creation failed: {result}")
            self.surface = surface_pointer[0]
        if self.surface is not None:
            self.get_surface_support = vk.vkGetInstanceProcAddr(
                self.instance, "vkGetPhysicalDeviceSurfaceSupportKHR"
            )

        candidates = []
        for physical in vk.vkEnumeratePhysicalDevices(self.instance):
            properties = vk.vkGetPhysicalDeviceProperties(physical)
            name = self._name(properties)
            extensions = {
                str(item.extensionName).split("\0", 1)[0]
                for item in vk.vkEnumerateDeviceExtensionProperties(physical, None)
            }
            required_extensions = set(DEVICE_EXTENSIONS)
            if self._headless_surface:
                required_extensions.update(EXTERNAL_INTEROP_DEVICE_EXTENSIONS)
            if self.surface is not None:
                required_extensions.add("VK_KHR_swapchain")
            if not required_extensions.issubset(extensions):
                continue
            if properties.deviceType not in (
                vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU,
                vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
            ):
                continue
            if requested_name and requested_name.lower() not in name.lower():
                continue
            queue_families = vk.vkGetPhysicalDeviceQueueFamilyProperties(physical)
            queue_index = next(
                (
                    index
                    for index, family in enumerate(queue_families)
                    if family.queueFlags & vk.VK_QUEUE_COMPUTE_BIT
                    and (
                        self.surface is None
                        or self.get_surface_support(physical, index, self.surface)
                    )
                ),
                None,
            )
            if queue_index is None:
                continue
            candidates.append(
                (properties.deviceType, physical, name, queue_index, extensions)
            )
        if not candidates:
            raise RuntimeError(
                "No hardware Vulkan adapter supports acceleration structures and ray queries"
            )
        candidates.sort(
            key=lambda item: item[0] != vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU
        )
        (
            _,
            self.physical_device,
            self.device_name,
            self.queue_family,
            device_extensions,
        ) = candidates[0]
        selected_properties = vk.vkGetPhysicalDeviceProperties(self.physical_device)
        id_properties = vk.VkPhysicalDeviceIDProperties()
        vk.vkGetPhysicalDeviceProperties2(
            self.physical_device,
            vk.VkPhysicalDeviceProperties2(pNext=id_properties),
        )
        self.device_uuid = bytes(id_properties.deviceUUID).hex()
        self.timestamp_period = selected_properties.limits.timestampPeriod
        if (
            self.surface is not None
            and selected_properties.limits.maxPushConstantsSize < 160
        ):
            raise RuntimeError(
                "Temporal window rendering requires 160 bytes of Vulkan push constants; "
                f"this adapter supports {selected_properties.limits.maxPushConstantsSize}"
            )
        queue_info = vk.VkDeviceQueueCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            queueFamilyIndex=self.queue_family,
            queueCount=1,
            pQueuePriorities=[1.0],
        )
        ray_query = vk.VkPhysicalDeviceRayQueryFeaturesKHR(
            sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
            rayQuery=vk.VK_TRUE,
        )
        acceleration = vk.VkPhysicalDeviceAccelerationStructureFeaturesKHR(
            sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
            pNext=ray_query,
            accelerationStructure=vk.VK_TRUE,
        )
        feature_chain = acceleration
        ray_pipeline_extension = "VK_KHR_ray_tracing_pipeline"
        ser_extension = "VK_NV_ray_tracing_invocation_reorder"
        if ray_pipeline_extension in device_extensions:
            queried_ray_pipeline = vk.VkPhysicalDeviceRayTracingPipelineFeaturesKHR()
            vk.vkGetPhysicalDeviceFeatures2(
                self.physical_device,
                vk.VkPhysicalDeviceFeatures2(pNext=queried_ray_pipeline),
            )
            self.ray_pipeline_supported = bool(queried_ray_pipeline.rayTracingPipeline)
        if self.ray_pipeline_supported and ser_extension in device_extensions:
            queried_ser = vk.VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV()
            vk.vkGetPhysicalDeviceFeatures2(
                self.physical_device,
                vk.VkPhysicalDeviceFeatures2(pNext=queried_ser),
            )
            self.ser_supported = bool(queried_ser.rayTracingInvocationReorder)
            if self.ser_supported:
                ser_properties = (
                    vk.VkPhysicalDeviceRayTracingInvocationReorderPropertiesNV()
                )
                vk.vkGetPhysicalDeviceProperties2(
                    self.physical_device,
                    vk.VkPhysicalDeviceProperties2(pNext=ser_properties),
                )
                self.ser_reordering_supported = bool(
                    ser_properties.rayTracingInvocationReorderReorderingHint
                    == vk.VK_RAY_TRACING_INVOCATION_REORDER_MODE_REORDER_NV
                )
        self.ray_pipeline_enabled = bool(
            self.ray_pipeline_supported and self.config.wavefront_ser
        )
        if self.ray_pipeline_enabled:
            ray_pipeline_properties = (
                vk.VkPhysicalDeviceRayTracingPipelinePropertiesKHR()
            )
            vk.vkGetPhysicalDeviceProperties2(
                self.physical_device,
                vk.VkPhysicalDeviceProperties2(pNext=ray_pipeline_properties),
            )
            self.ray_tracing_shader_group_handle_size = int(
                ray_pipeline_properties.shaderGroupHandleSize
            )
            self.ray_tracing_shader_group_handle_alignment = int(
                ray_pipeline_properties.shaderGroupHandleAlignment
            )
            self.ray_tracing_shader_group_base_alignment = int(
                ray_pipeline_properties.shaderGroupBaseAlignment
            )
            ray_pipeline = vk.VkPhysicalDeviceRayTracingPipelineFeaturesKHR(
                pNext=feature_chain,
                rayTracingPipeline=vk.VK_TRUE,
            )
            feature_chain = ray_pipeline
        if self.ser_supported and self.config.wavefront_ser:
            ser_features = vk.VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV(
                pNext=feature_chain,
                rayTracingInvocationReorder=vk.VK_TRUE,
            )
            feature_chain = ser_features
        descriptor_features = vk.VkPhysicalDeviceVulkan12Features(
            sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES
        )
        vk.vkGetPhysicalDeviceFeatures2(
            self.physical_device,
            vk.VkPhysicalDeviceFeatures2(pNext=descriptor_features),
        )
        self.native_textures_supported = bool(
            descriptor_features.shaderSampledImageArrayNonUniformIndexing
            and selected_properties.limits.maxPerStageDescriptorSampledImages
            >= MAX_NATIVE_TEXTURES * 2
            and selected_properties.limits.maxDescriptorSetSampledImages
            >= MAX_NATIVE_TEXTURES * 2
        )
        if self.config.wavefront_native_textures and self.native_textures_supported:
            descriptor_indexing = vk.VkPhysicalDeviceVulkan12Features(
                sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                pNext=feature_chain,
                shaderSampledImageArrayNonUniformIndexing=vk.VK_TRUE,
            )
            feature_chain = descriptor_indexing
        pipeline_statistics_extension = "VK_KHR_pipeline_executable_properties"
        if self.config.wavefront_pipeline_statistics:
            if pipeline_statistics_extension not in device_extensions:
                raise RuntimeError(
                    "Vulkan device does not support pipeline executable statistics"
                )
            queried_pipeline_statistics = (
                vk.VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR()
            )
            vk.vkGetPhysicalDeviceFeatures2(
                self.physical_device,
                vk.VkPhysicalDeviceFeatures2(pNext=queried_pipeline_statistics),
            )
            if not queried_pipeline_statistics.pipelineExecutableInfo:
                raise RuntimeError(
                    "Vulkan device reports pipeline executable statistics disabled"
                )
            pipeline_statistics = (
                vk.VkPhysicalDevicePipelineExecutablePropertiesFeaturesKHR(
                    pNext=feature_chain, pipelineExecutableInfo=vk.VK_TRUE
                )
            )
            feature_chain = pipeline_statistics
            self.pipeline_statistics_supported = True
        present_extensions = {"VK_KHR_present_id", "VK_KHR_present_wait"}
        if self.surface is not None and present_extensions.issubset(device_extensions):
            queried_wait = vk.VkPhysicalDevicePresentWaitFeaturesKHR()
            queried_id = vk.VkPhysicalDevicePresentIdFeaturesKHR(pNext=queried_wait)
            vk.vkGetPhysicalDeviceFeatures2(
                self.physical_device,
                vk.VkPhysicalDeviceFeatures2(pNext=queried_id),
            )
            self.present_wait_supported = bool(
                queried_id.presentId and queried_wait.presentWait
            )
            if self.present_wait_supported:
                present_wait = vk.VkPhysicalDevicePresentWaitFeaturesKHR(
                    pNext=feature_chain,
                    presentWait=vk.VK_TRUE,
                )
                feature_chain = vk.VkPhysicalDevicePresentIdFeaturesKHR(
                    pNext=present_wait,
                    presentId=vk.VK_TRUE,
                )
        buffer_address = vk.VkPhysicalDeviceBufferDeviceAddressFeatures(
            sType=vk.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES,
            pNext=feature_chain,
            bufferDeviceAddress=vk.VK_TRUE,
        )
        enabled_device_extensions = list(DEVICE_EXTENSIONS)
        if self._headless_surface:
            enabled_device_extensions.extend(EXTERNAL_INTEROP_DEVICE_EXTENSIONS)
        if self.ray_pipeline_enabled:
            enabled_device_extensions.append(ray_pipeline_extension)
        if self.ser_supported and self.config.wavefront_ser:
            enabled_device_extensions.append(ser_extension)
        if self.pipeline_statistics_supported:
            enabled_device_extensions.append(pipeline_statistics_extension)
        if self.surface is not None:
            enabled_device_extensions.append("VK_KHR_swapchain")
            if self.present_wait_supported:
                enabled_device_extensions.extend(sorted(present_extensions))
        self.present_pacing_enabled = (
            self.present_wait_supported and self.config.present_pacing
        )
        physical_features = vk.vkGetPhysicalDeviceFeatures(self.physical_device)
        self.formatless_storage_write_supported = bool(
            physical_features.shaderStorageImageWriteWithoutFormat
        )
        enabled_features = vk.VkPhysicalDeviceFeatures(
            shaderStorageImageWriteWithoutFormat=(
                vk.VK_TRUE if self.formatless_storage_write_supported else vk.VK_FALSE
            ),
        )
        device_info = vk.VkDeviceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            pNext=buffer_address,
            pEnabledFeatures=enabled_features,
            queueCreateInfoCount=1,
            pQueueCreateInfos=[queue_info],
            enabledExtensionCount=len(enabled_device_extensions),
            ppEnabledExtensionNames=enabled_device_extensions,
        )
        self.device = vk.vkCreateDevice(self.physical_device, device_info, None)
        self.queue = vk.vkGetDeviceQueue(self.device, self.queue_family, 0)
        self.memory_properties = vk.vkGetPhysicalDeviceMemoryProperties(
            self.physical_device
        )
        self._create_pipeline_cache()

    def _resolved_pipeline_cache_path(self):
        configured = self.config.vulkan_pipeline_cache_path
        if configured is not None:
            return Path(configured).expanduser()
        cache_root = os.environ.get("XDG_CACHE_HOME")
        if cache_root:
            root = Path(cache_root)
        else:
            root = Path.home() / ".cache"
        return root / "ordinarylight" / "vulkan" / f"{self.device_uuid}.bin"

    def _create_pipeline_cache(self):
        if not self.config.vulkan_pipeline_cache:
            return
        path = self._resolved_pipeline_cache_path()
        initial = b""
        try:
            initial = path.read_bytes()
        except (FileNotFoundError, OSError):
            pass
        initial_buffer = vk.ffi.new("uint8_t[]", initial) if initial else None
        create_info = vk.VkPipelineCacheCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            initialDataSize=len(initial),
            pInitialData=(
                vk.ffi.cast("void *", initial_buffer)
                if initial_buffer is not None
                else None
            ),
        )
        try:
            self.pipeline_cache = vk.vkCreatePipelineCache(
                self.device, create_info, None
            )
        except Exception:
            # Driver updates may invalidate opaque cache data. Rebuild from an
            # empty cache rather than making renderer construction fail.
            self.pipeline_cache = vk.vkCreatePipelineCache(
                self.device,
                vk.VkPipelineCacheCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
                ),
                None,
            )
        self.pipeline_cache_path = path

    def _save_pipeline_cache(self):
        if self.pipeline_cache is None or self.pipeline_cache_path is None:
            return
        try:
            size = vk.ffi.new("size_t *")
            result = vk.lib.vkGetPipelineCacheData(
                self.device, self.pipeline_cache, size, vk.ffi.NULL
            )
            if result != vk.VK_SUCCESS or size[0] == 0:
                return
            data = vk.ffi.new("uint8_t[]", size[0])
            result = vk.lib.vkGetPipelineCacheData(
                self.device, self.pipeline_cache, size, data
            )
            if result != vk.VK_SUCCESS:
                return
            path = self.pipeline_cache_path
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
            temporary.write_bytes(bytes(vk.ffi.buffer(data, size[0])))
            os.replace(temporary, path)
        except Exception:
            # Caching is an optimization; read-only homes and containers must
            # remain fully supported, and cache persistence must never make
            # renderer shutdown fail.
            return

    def _load_extension_functions(self):
        self.create_as = vk.vkGetDeviceProcAddr(
            self.device, "vkCreateAccelerationStructureKHR"
        )
        self.destroy_as = vk.vkGetDeviceProcAddr(
            self.device, "vkDestroyAccelerationStructureKHR"
        )
        self.build_as = vk.vkGetDeviceProcAddr(
            self.device, "vkCmdBuildAccelerationStructuresKHR"
        )
        self.get_as_sizes = vk.vkGetDeviceProcAddr(
            self.device, "vkGetAccelerationStructureBuildSizesKHR"
        )
        self.get_pipeline_executables = None
        self.get_pipeline_statistics = None
        self.get_memory_fd = None
        self.get_semaphore_fd = None
        if self._headless_surface:
            self.get_memory_fd = vk.ffi.cast(
                "PFN_vkGetMemoryFdKHR",
                vk.lib.vkGetDeviceProcAddr(self.device, b"vkGetMemoryFdKHR"),
            )
            self.get_semaphore_fd = vk.ffi.cast(
                "PFN_vkGetSemaphoreFdKHR",
                vk.lib.vkGetDeviceProcAddr(self.device, b"vkGetSemaphoreFdKHR"),
            )
        self.create_ray_tracing_pipelines = None
        self.cmd_trace_rays = None
        self.get_ray_tracing_shader_group_handles = None
        if self.ray_pipeline_enabled:
            raw_trace_rays = vk.lib.vkGetDeviceProcAddr(
                self.device, b"vkCmdTraceRaysKHR"
            )
            self.cmd_trace_rays = vk.ffi.cast("PFN_vkCmdTraceRaysKHR", raw_trace_rays)
            for attribute, function_name in (
                ("create_ray_tracing_pipelines", "vkCreateRayTracingPipelinesKHR"),
                (
                    "get_ray_tracing_shader_group_handles",
                    "vkGetRayTracingShaderGroupHandlesKHR",
                ),
            ):
                setattr(
                    self,
                    attribute,
                    vk.vkGetDeviceProcAddr(self.device, function_name),
                )
        if self.pipeline_statistics_supported:
            raw_properties = vk.lib.vkGetDeviceProcAddr(
                self.device, b"vkGetPipelineExecutablePropertiesKHR"
            )
            raw_statistics = vk.lib.vkGetDeviceProcAddr(
                self.device, b"vkGetPipelineExecutableStatisticsKHR"
            )
            self.get_pipeline_executables = vk.ffi.cast(
                "PFN_vkGetPipelineExecutablePropertiesKHR", raw_properties
            )
            self.get_pipeline_statistics = vk.ffi.cast(
                "PFN_vkGetPipelineExecutableStatisticsKHR", raw_statistics
            )

        raw_buffer_address = vk.lib.vkGetDeviceProcAddr(
            self.device, b"vkGetBufferDeviceAddress"
        )
        self._raw_buffer_address = vk.ffi.cast(
            "PFN_vkGetBufferDeviceAddress", raw_buffer_address
        )
        raw_as_address = vk.lib.vkGetDeviceProcAddr(
            self.device, b"vkGetAccelerationStructureDeviceAddressKHR"
        )
        self._raw_as_address = vk.ffi.cast(
            "PFN_vkGetAccelerationStructureDeviceAddressKHR", raw_as_address
        )
        if self.surface is not None:
            self.get_surface_capabilities = vk.vkGetInstanceProcAddr(
                self.instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR"
            )
            self.get_surface_formats = vk.vkGetInstanceProcAddr(
                self.instance, "vkGetPhysicalDeviceSurfaceFormatsKHR"
            )
            self.get_surface_present_modes = vk.vkGetInstanceProcAddr(
                self.instance, "vkGetPhysicalDeviceSurfacePresentModesKHR"
            )
            self.create_swapchain = vk.vkGetDeviceProcAddr(
                self.device, "vkCreateSwapchainKHR"
            )
            self.destroy_swapchain = vk.vkGetDeviceProcAddr(
                self.device, "vkDestroySwapchainKHR"
            )
            self.get_swapchain_images = vk.vkGetDeviceProcAddr(
                self.device, "vkGetSwapchainImagesKHR"
            )
            self.acquire_next_image = vk.vkGetDeviceProcAddr(
                self.device, "vkAcquireNextImageKHR"
            )
            self.queue_present = vk.vkGetDeviceProcAddr(
                self.device, "vkQueuePresentKHR"
            )
            if self.present_wait_supported:
                raw_wait_for_present = vk.lib.vkGetDeviceProcAddr(
                    self.device, b"vkWaitForPresentKHR"
                )
                self.wait_for_present = vk.ffi.cast(
                    "VkResult(*)(VkDevice, VkSwapchainKHR, uint64_t, uint64_t)",
                    raw_wait_for_present,
                )

    def query_pipeline_statistics(self, pipeline):
        """Return executable statistics captured for one diagnostic pipeline."""
        if not self.pipeline_statistics_supported:
            return []
        pipeline_info = vk.VkPipelineInfoKHR(pipeline=pipeline)
        executable_count = vk.ffi.new("uint32_t *")
        self.get_pipeline_executables(
            self.device,
            vk.ffi.addressof(pipeline_info),
            executable_count,
            vk.ffi.NULL,
        )
        properties = vk.ffi.new(
            "VkPipelineExecutablePropertiesKHR[]", executable_count[0]
        )
        for item in properties:
            item.sType = vk.VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR
        self.get_pipeline_executables(
            self.device,
            vk.ffi.addressof(pipeline_info),
            executable_count,
            properties,
        )
        results = []
        for index in range(executable_count[0]):
            executable_info = vk.VkPipelineExecutableInfoKHR(
                pipeline=pipeline, executableIndex=index
            )
            statistic_count = vk.ffi.new("uint32_t *")
            self.get_pipeline_statistics(
                self.device,
                vk.ffi.addressof(executable_info),
                statistic_count,
                vk.ffi.NULL,
            )
            statistics = vk.ffi.new(
                "VkPipelineExecutableStatisticKHR[]", statistic_count[0]
            )
            for statistic in statistics:
                statistic.sType = vk.VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR
            self.get_pipeline_statistics(
                self.device,
                vk.ffi.addressof(executable_info),
                statistic_count,
                statistics,
            )
            values = {}
            for statistic in statistics:
                if (
                    statistic.format
                    == vk.VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR
                ):
                    value = bool(statistic.value.b32)
                elif (
                    statistic.format
                    == vk.VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR
                ):
                    value = int(statistic.value.i64)
                elif (
                    statistic.format
                    == vk.VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR
                ):
                    value = int(statistic.value.u64)
                else:
                    value = float(statistic.value.f64)
                name = vk.ffi.string(statistic.name).decode("utf-8")
                values[name] = value
            results.append(
                {
                    "name": vk.ffi.string(properties[index].name).decode("utf-8"),
                    "description": vk.ffi.string(properties[index].description).decode(
                        "utf-8"
                    ),
                    "statistics": values,
                }
            )
        return results

    def _create_command_pool(self):
        self.command_pool = vk.vkCreateCommandPool(
            self.device,
            vk.VkCommandPoolCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                queueFamilyIndex=self.queue_family,
            ),
            None,
        )
