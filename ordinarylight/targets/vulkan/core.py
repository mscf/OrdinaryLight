"""Minimal Vulkan ray-query backend using KHR acceleration structures."""

from importlib.resources import files
import math
import os
from pathlib import Path
import struct
import time
from types import SimpleNamespace

import numpy as np

from ...cameras import OrthographicCamera, PanoramicCamera, PerspectiveCamera
from ...effects import (
    BoundingBox, EmissiveHighlight, Isolation, ObjectEffect, Outline, Tint,
    XRay,
)
from ...integrations.indirect_reuse import IndirectReservoirPlan
from ...state import AccumulationState
import vulkan as vk


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

# vulkan-python 1.3.275 exposes these promoted aliases as ``None`` and omits
# the Vulkan 1.2 core spellings.  Keep the spec values here until the binding
# is updated.
BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT = 0x00020000
MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT = 0x00000002
# vulkan-python 1.3.275.1 incorrectly aliases this enum to the ray-pipeline
# feature structure type (1000347000).  Vulkan defines the bind point as
# 1000165000.
PIPELINE_BIND_POINT_RAY_TRACING_KHR = 1000165000
WINDOW_FRAMES_IN_FLIGHT = 2
MAX_NATIVE_TEXTURES = 64
MAX_NATIVE_VOLUMES = 16
RECONSTRUCT_BASE_FORMAT = "f3If2IfIfIfIIf"
RECONSTRUCT_PUSH_SIZE = 256


def _restir_reservoir_storage_bytes(
    width, height, reservoir_count, *, stratified=False,
):
    """Return storage for independent direct-light reservoir streams."""
    bytes_per_reservoir = 12 + (8 if stratified else 0)
    return (
        int(width) * int(height) * int(reservoir_count)
        * bytes_per_reservoir
    )


def _wavefront_history_semaphore_plan(
    *, current_pending, previous_pending, history_enabled,
):
    """Describe one binary-semaphore history hand-off.

    A pending signal from the preceding slot must always be consumed, even
    when history was disabled between frames.  The current slot is signaled
    only while a temporal consumer can use it on the next submission.
    """
    if current_pending:
        raise RuntimeError(
            "wavefront history semaphore was not consumed before its frame "
            "slot was reused"
        )
    return bool(previous_pending), bool(history_enabled)


def _camera_projection(camera):
    if isinstance(camera, OrthographicCamera):
        return 1
    if isinstance(camera, PanoramicCamera):
        return 2
    return 0


def _camera_vectors(camera):
    position = np.asarray(camera.position, dtype=np.float32)
    forward = np.asarray(camera.target, dtype=np.float32) - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(camera.up, dtype=np.float32))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    if isinstance(camera, OrthographicCamera):
        scale = float(camera.vertical_size) * 0.5
        right *= scale
        up *= scale
    elif isinstance(camera, PanoramicCamera):
        right *= math.radians(camera.horizontal_fov_degrees) * 0.5
        up *= math.radians(camera.vertical_fov_degrees) * 0.5
    else:
        scale = math.tan(math.radians(camera.vertical_fov_degrees) * 0.5)
        right *= scale
        up *= scale
    return position, forward, right, up


def _camera_signature(camera):
    projection_parameter = (
        camera.vertical_size if isinstance(camera, OrthographicCamera)
        else (
            (camera.horizontal_fov_degrees, camera.vertical_fov_degrees)
            if isinstance(camera, PanoramicCamera)
            else camera.vertical_fov_degrees
        )
    )
    return (
        type(camera), tuple(camera.position), tuple(camera.target),
        tuple(camera.up), projection_parameter,
    )


def _effect_screen_rect(scene, triangle_range, camera, output_size):
    """Return normalized projected bounds for one packed triangle range."""
    start, end = triangle_range
    points = scene.render_triangles()[start:end].reshape((-1, 3)).astype(np.float64)
    if not len(points):
        return (0.0, 0.0, 0.0, 0.0)
    origin = np.asarray(camera.position, np.float64)
    forward = np.asarray(camera.target, np.float64) - origin
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray(camera.up, np.float64))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    relative = points - origin
    x = relative @ right
    y = relative @ up
    z = relative @ forward
    aspect = float(output_size[0]) / float(output_size[1])
    if isinstance(camera, PerspectiveCamera):
        visible = z > 1e-5
        if not np.any(visible):
            return (0.0, 0.0, 0.0, 0.0)
        scale = math.tan(math.radians(camera.vertical_fov_degrees) * 0.5)
        ndc_x = x[visible] / (z[visible] * scale * aspect)
        ndc_y = y[visible] / (z[visible] * scale)
    elif isinstance(camera, OrthographicCamera):
        scale = camera.vertical_size * 0.5
        ndc_x, ndc_y = x / (scale * aspect), y / scale
    else:
        yaw = np.arctan2(x, z)
        pitch = np.arctan2(y, np.sqrt(x * x + z * z))
        ndc_x = yaw / max(math.radians(camera.horizontal_fov_degrees) * 0.5, 1e-6)
        ndc_y = pitch / max(math.radians(camera.vertical_fov_degrees) * 0.5, 1e-6)
    screen_x = np.clip(ndc_x * 0.5 + 0.5, 0.0, 1.0)
    screen_y = np.clip(0.5 - ndc_y * 0.5, 0.0, 1.0)
    return (
        float(np.min(screen_x)), float(np.min(screen_y)),
        float(np.max(screen_x)), float(np.max(screen_y)),
    )


def _align_up(value, alignment):
    return (value + alignment - 1) // alignment * alignment


def _camera_angular_motion_pixels(previous_camera, camera, image_height):
    """Conservatively estimate rotational reprojection displacement."""
    if previous_camera is None or image_height <= 0:
        return 0.0
    # Reconstruction and reservoir reprojection currently encode perspective
    # projection. Other camera models therefore invalidate history safely.
    if not isinstance(previous_camera, PerspectiveCamera) or not isinstance(
        camera, PerspectiveCamera
    ):
        return math.inf
    previous_position = np.asarray(previous_camera.position, dtype=np.float64)
    current_position = np.asarray(camera.position, dtype=np.float64)
    previous_forward = (
        np.asarray(previous_camera.target, dtype=np.float64)
        - previous_position
    )
    current_forward = (
        np.asarray(camera.target, dtype=np.float64) - current_position
    )
    previous_length = np.linalg.norm(previous_forward)
    current_length = np.linalg.norm(current_forward)
    if previous_length <= 1e-12 or current_length <= 1e-12:
        return math.inf
    cosine = float(np.dot(
        previous_forward / previous_length,
        current_forward / current_length,
    ))
    angle = math.acos(max(-1.0, min(1.0, cosine)))
    vertical_fov = math.radians(max(
        float(camera.vertical_fov_degrees), 1e-6
    ))
    return angle * float(image_height) / vertical_fov


def _camera_motion_pixels(previous_camera, camera, image_height):
    """Estimate rotational and translational screen-space camera motion."""
    angular = _camera_angular_motion_pixels(
        previous_camera, camera, image_height
    )
    if not math.isfinite(angular) or previous_camera is None:
        return angular
    previous_position = np.asarray(previous_camera.position, dtype=np.float64)
    current_position = np.asarray(camera.position, dtype=np.float64)
    previous_target = np.asarray(previous_camera.target, dtype=np.float64)
    current_target = np.asarray(camera.target, dtype=np.float64)
    focus_distance = min(
        np.linalg.norm(previous_target - previous_position),
        np.linalg.norm(current_target - current_position),
    )
    if focus_distance <= 1e-12:
        return math.inf
    focal_pixels = float(image_height) / (
        2.0 * math.tan(math.radians(camera.vertical_fov_degrees) * 0.5)
    )
    translation = np.linalg.norm(current_position - previous_position)
    translational = translation * focal_pixels / focus_distance
    previous_tangent = math.tan(
        math.radians(previous_camera.vertical_fov_degrees) * 0.5
    )
    current_tangent = math.tan(
        math.radians(camera.vertical_fov_degrees) * 0.5
    )
    zoom = (
        abs(math.log(current_tangent / previous_tangent))
        * float(image_height) * 0.5
        if previous_tangent > 1e-12 and current_tangent > 1e-12
        else math.inf
    )
    return max(angular, float(translational), zoom)


def _motion_adaptive_history_limit(configured_limit, motion_pixels,
                                   history_footprint_pixels):
    """Bound represented history by its approximate screen-space footprint."""
    if not math.isfinite(motion_pixels) or motion_pixels < 0.0:
        return 1
    if motion_pixels <= 1e-6:
        return configured_limit
    motion_limit = max(1, int(math.floor(
        history_footprint_pixels / motion_pixels
    )))
    return min(configured_limit, motion_limit)


def _relax_temporal_policy(config, motion_pixels):
    """Resolve stationary or motion-safe ReLAX temporal constants."""
    moving = not math.isfinite(motion_pixels) or motion_pixels > 0.25
    return {
        "history_limit": _motion_adaptive_history_limit(
            config.denoiser_history_limit,
            motion_pixels,
            config.denoiser_history_motion_pixels,
        ),
        "normal_threshold": (
            config.denoiser_motion_normal_threshold if moving else 0.8
        ),
        "depth_threshold": (
            config.denoiser_motion_depth_threshold if moving else 0.02
        ),
        "clamp_sigma": (
            config.denoiser_motion_clamp_sigma if moving else 2.5
        ),
        "reactive_sigma": (
            config.denoiser_motion_reactive_sigma if moving else 0.0
        ),
    }


class Buffer:
    def __init__(self, buffer, memory, size):
        self.buffer = buffer
        self.memory = memory
        self.size = size


class AccelerationStructure:
    def __init__(self, handle, storage, scratch=None):
        self.handle = handle
        self.storage = storage
        self.scratch = scratch


class SceneBlas:
    """One refittable BLAS shared by instances of the same mesh resource."""

    def __init__(self, structure, mesh, vertex_buffer):
        self.structure = structure
        self.mesh = mesh
        self.vertex_buffer = vertex_buffer
        self.indices = mesh.indices.copy()
        self.vertices = mesh.vertices.copy()


class SceneTlasInstance:
    """One TLAS placement and its global packed-triangle offset."""

    def __init__(self, mesh, blas, triangle_offset):
        self.mesh = mesh
        self.blas = blas
        self.triangle_offset = triangle_offset
        # Volume bounds are ray-entry proxies, not opaque surfaces.  Keep them
        # in a separate visibility group so path rays can enter volumes while
        # binary surface-shadow queries do not mistake the proxy box for an
        # occluder.
        self.visibility_mask = (
            0x02 if mesh.metadata.get("volume_index") is not None else 0x01
        )


class SampledTexture:
    def __init__(self, image, memory, view, sampler):
        self.image = image
        self.memory = memory
        self.view = view
        self.sampler = sampler


class VulkanWavefrontExecutor:
    """Experimental bounded generate/intersect wavefront dispatch resources."""

    def __init__(self, core, capacity):
        from ...wavefront import (
            HIT_DTYPE, HOT_PATH_STATE_DTYPE, MEDIUM_STACK_DTYPE, RAY_DTYPE,
            RESOLVED_PIXEL_DTYPE, SECONDARY_PATH_STATE_DTYPE,
            WavefrontQueueLayout,
        )

        self.core = core
        # Auto mode compiles only the strategy selected for the resident scene.
        # The core replaces this executor if a later scene resolves differently.
        self.strategy = core.resolved_execution_strategy
        self.scene_pipeline_signature = core._wavefront_pipeline_signature()
        self.capacity = capacity
        self.hit_dtype = HIT_DTYPE
        self.descriptor_pool = None
        self.generate_layout = None
        self.primary_layout = None
        self.intersect_layout = None
        self.shade_layout = None
        self.resolve_layout = None
        self.indirect_layout = None
        self.wavefront_tone_layout = None
        self.wavefront_image_layout = None
        self.reconstruct_layout = None
        self.relax_prepare_layout = None
        self.relax_temporal_layout = None
        self.relax_atrous_layout = None
        self.relax_compose_layout = None
        self.indirect_reuse_clear_layout = None
        self.indirect_reuse_candidate_layout = None
        self.indirect_reuse_debug_layout = None
        self.bucket_intersect_layout = None
        self.generate_pipeline_layout = None
        self.primary_pipeline_layout = None
        self.intersect_pipeline_layout = None
        self.shade_pipeline_layout = None
        self.resolve_pipeline_layout = None
        self.indirect_pipeline_layout = None
        self.wavefront_tone_pipeline_layout = None
        self.wavefront_image_pipeline_layout = None
        self.reconstruct_pipeline_layout = None
        self.relax_prepare_pipeline_layout = None
        self.relax_temporal_pipeline_layout = None
        self.relax_atrous_pipeline_layout = None
        self.relax_compose_pipeline_layout = None
        self.indirect_reuse_clear_pipeline_layout = None
        self.indirect_reuse_candidate_pipeline_layout = None
        self.indirect_reuse_debug_pipeline_layout = None
        self.bucket_intersect_pipeline_layout = None
        self.generate_module = None
        self.primary_module = None
        self.hybrid_module = None
        self.hybrid_opaque_module = None
        self.hybrid_opaque_untextured_production_module = None
        self.megakernel_module = None
        self.megakernel_untextured_module = None
        self.megakernel_untextured_swizzle_modules = {}
        self.megakernel_opaque_module = None
        self.megakernel_opaque_untextured_module = None
        self.megakernel_untextured_primary_module = None
        self.megakernel_untextured_secondary_module = None
        self.megakernel_opaque_untextured_production_module = None
        self.megakernel_opaque_untextured_wg32_module = None
        self.megakernel_swizzle_modules = {}
        self.ser_megakernel_module = None
        self.ser_miss_module = None
        self.persistent_module = None
        self.persistent_coarse_module = None
        self.persistent_continuation_module = None
        self.persistent_continuation_opaque_module = None
        self.intersect_module = None
        self.shade_module = None
        self.custom_primary_module = None
        self.custom_shade_module = None
        self.overlap_modules = {}
        self.scattering_modules = {}
        self.volume_skipping_modules = {}
        self.resolve_module = None
        self.indirect_module = None
        self.wavefront_tone_module = None
        self.wavefront_image_module = None
        self.reconstruct_module = None
        self.relax_prepare_module = None
        self.relax_temporal_module = None
        self.relax_atrous_module = None
        self.relax_compose_module = None
        self.reconstruct_bgra_module = None
        self.indirect_reuse_clear_module = None
        self.indirect_reuse_candidate_module = None
        self.indirect_reuse_debug_module = None
        self.bucket_intersect_module = None
        self.generate_pipeline = None
        self.primary_pipeline = None
        self.hybrid_pipeline = None
        self.hybrid_opaque_pipeline = None
        self.hybrid_opaque_untextured_production_pipeline = None
        self.megakernel_pipeline = None
        self.megakernel_untextured_pipeline = None
        self.megakernel_untextured_swizzle_pipelines = {}
        self.megakernel_opaque_pipeline = None
        self.megakernel_opaque_untextured_pipeline = None
        self.megakernel_untextured_primary_pipeline = None
        self.megakernel_untextured_secondary_pipeline = None
        self.megakernel_opaque_untextured_production_pipeline = None
        self.megakernel_opaque_untextured_wg32_pipeline = None
        self.megakernel_swizzle_pipelines = {}
        self.ser_megakernel_pipeline = None
        self.ser_megakernel_sbt = None
        self.ser_raygen_region = None
        self.ser_miss_region = None
        self.ser_empty_region = None
        self.persistent_pipeline = None
        self.persistent_coarse_pipeline = None
        self.persistent_continuation_pipeline = None
        self.persistent_continuation_opaque_pipeline = None
        self.intersect_pipeline = None
        self.shade_pipeline = None
        self.custom_primary_pipeline = None
        self.custom_shade_pipeline = None
        self.overlap_pipelines = {}
        self.scattering_pipelines = {}
        self.volume_skipping_pipelines = {}
        self.custom_material_signature = None
        self.resolve_pipeline = None
        self.indirect_pipeline = None
        self.wavefront_tone_pipeline = None
        self.wavefront_image_pipeline = None
        self.reconstruct_pipeline = None
        self.relax_prepare_pipeline = None
        self.relax_temporal_pipeline = None
        self.relax_atrous_pipeline = None
        self.relax_compose_pipeline = None
        self.reconstruct_bgra_pipeline = None
        self.indirect_reuse_clear_pipeline = None
        self.indirect_reuse_candidate_pipeline = None
        self.indirect_reuse_debug_pipeline = None
        self.bucket_intersect_pipeline = None
        self.generate_sets = []
        self.primary_sets = []
        self.intersect_sets = []
        self.shade_sets = []
        self.resolve_set = None
        self.indirect_sets = []
        self.wavefront_tone_set = None
        self.wavefront_image_sets = []
        self.reconstruct_sets = []
        self.relax_prepare_sets = []
        self.relax_temporal_sets = []
        self.relax_atrous_sets = []
        self.relax_compose_sets = []
        self.indirect_reuse_clear_sets = []
        self.indirect_reuse_candidate_sets = []
        self.indirect_reuse_debug_sets = []
        self.coherent_shade_sets = []
        self.coherent_indirect_sets = []
        self.bucket_intersect_sets = []
        self._bound_scene_key = None
        self.production_restir = False
        device_flags = vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        usage = (
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
            | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
        )
        ray_layout = WavefrontQueueLayout(RAY_DTYPE, capacity)
        hit_layout = WavefrontQueueLayout(HIT_DTYPE, capacity)
        self.ray_buffer = core._create_buffer(
            ray_layout.byte_size, usage, device_flags
        )
        self.hit_buffer = core._create_buffer(
            hit_layout.byte_size, usage, device_flags
        )
        self.coherent_hit_buffers = [core._create_buffer(
            hit_layout.byte_size, usage, device_flags
        ) for _ in range(2)] if core.config.wavefront_material_bucketing else []
        self.next_ray_buffer = core._create_buffer(
            ray_layout.byte_size, usage, device_flags
        )
        self.path_buffer = core._create_buffer(
            capacity * HOT_PATH_STATE_DTYPE.itemsize, usage, device_flags
        )
        self.medium_stack_itemsize = MEDIUM_STACK_DTYPE.itemsize
        initial_scene = (
            core.scene_resources.scene
            if core.scene_resources is not None else None
        )
        self.medium_capacity = (
            1 if initial_scene is not None
            and core._use_opaque_scene_specialization(initial_scene)
            else capacity
        )
        self.medium_buffer = core._create_buffer(
            self.medium_capacity * self.medium_stack_itemsize,
            usage, device_flags,
        )
        self.secondary_path_buffer = core._create_buffer(
            (capacity if (
                core.config.wavefront_indirect_reuse_candidates
                or core.config.denoiser_signal_capture
                or core.config.denoiser_enabled
            ) else 1)
            * SECONDARY_PATH_STATE_DTYPE.itemsize,
            usage, device_flags,
        )
        self.secondary_path_itemsize = SECONDARY_PATH_STATE_DTYPE.itemsize
        self.resolve_buffer = core._create_buffer(
            capacity * RESOLVED_PIXEL_DTYPE.itemsize, usage, device_flags
        )
        self.indirect_buffer = core._create_buffer(
            12,
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | vk.VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
            | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            device_flags,
        )
        self.packed_buffer = core._create_buffer(
            capacity * 4, usage, device_flags
        )
        staging_flags = (
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
        )
        staging_usage = vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
        camera_usage = vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        self.camera_buffers = [core._create_buffer(
            64, camera_usage, staging_flags
        ) for _ in range(WINDOW_FRAMES_IN_FLIGHT)]
        self.relax_temporal_constant_buffers = [core._create_buffer(
            32, vk.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, staging_flags
        ) for _ in range(WINDOW_FRAMES_IN_FLIGHT)] if (
            core.config.denoiser_enabled
        ) else []
        self.work_counter_buffers = [core._create_buffer(
            160,
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            staging_flags,
        ) for _ in range(WINDOW_FRAMES_IN_FLIGHT)] if (
            core.config.wavefront_profiling
        ) else []
        self.indirect_reuse_counter_buffers = [core._create_buffer(
            72,
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            staging_flags,
        ) for _ in range(WINDOW_FRAMES_IN_FLIGHT)] if (
            core.config.wavefront_indirect_reuse_candidates
        ) else []
        self.ray_readback = core._create_buffer(16, staging_usage, staging_flags)
        self.hit_readback = core._create_buffer(16, staging_usage, staging_flags)
        self.next_ray_readback = core._create_buffer(16, staging_usage, staging_flags)
        self.resolve_readback = core._create_buffer(
            capacity * RESOLVED_PIXEL_DTYPE.itemsize,
            staging_usage, staging_flags,
        )
        self.packed_readback = core._create_buffer(
            capacity * 4, staging_usage, staging_flags
        )
        self.secondary_path_readback = (
            core._create_buffer(
                capacity * SECONDARY_PATH_STATE_DTYPE.itemsize,
                staging_usage, staging_flags,
            )
            if core.config.denoiser_signal_capture else None
        )
        self.primary_hit_snapshot = None
        self.primary_hit_readback = None
        self.primary_hit_capacity = 0
        self.pipeline_statistics = {}
        self._create_pipelines()

    def _ensure_primary_hit_capture(self, count):
        if count <= self.primary_hit_capacity:
            return
        previous = tuple(
            buffer for buffer in (
                self.primary_hit_snapshot, self.primary_hit_readback,
            ) if buffer is not None
        )
        if previous:
            self.core._release_resources([], previous)
            for buffer in previous:
                self.core._buffers.remove(buffer)
        size = 16 + count * self.hit_dtype.itemsize
        usage = (
            vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
            | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
        )
        self.primary_hit_snapshot = self.core._create_buffer(
            size, usage, vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )
        self.primary_hit_readback = self.core._create_buffer(
            size, vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        self.primary_hit_capacity = count

    def _layout(self, bindings):
        return vk.vkCreateDescriptorSetLayout(
            self.core.device,
            vk.VkDescriptorSetLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                bindingCount=len(bindings), pBindings=bindings,
            ), None,
        )

    def _pipeline_layout(self, descriptor_layout, push_size=0):
        push_range = None
        if push_size:
            stages = vk.VK_SHADER_STAGE_COMPUTE_BIT
            if self.core.config.wavefront_ser:
                stages |= vk.VK_SHADER_STAGE_RAYGEN_BIT_KHR
            push_range = vk.VkPushConstantRange(
                stageFlags=stages, offset=0, size=push_size
            )
        return vk.vkCreatePipelineLayout(
            self.core.device,
            vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=1, pSetLayouts=[descriptor_layout],
                pushConstantRangeCount=1 if push_range else 0,
                pPushConstantRanges=[push_range] if push_range else None,
            ), None,
        )

    def _pipeline(self, shader_name, layout):
        code = files("ordinarylight").joinpath(f"shaders/{shader_name}.spv").read_bytes()
        module = vk.vkCreateShaderModule(
            self.core.device,
            vk.VkShaderModuleCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize=len(code), pCode=code,
            ), None,
        )
        stage = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT, module=module, pName="main",
        )
        flags = (
            vk.VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR
            if self.core.config.wavefront_pipeline_statistics else 0
        )
        pipeline = vk.vkCreateComputePipelines(
            self.core.device,
            self.core.pipeline_cache or vk.VK_NULL_HANDLE,
            1,
            [vk.VkComputePipelineCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                flags=flags, stage=stage, layout=layout,
            )], None,
        )[0]
        if self.core.config.wavefront_pipeline_statistics:
            self.pipeline_statistics[shader_name] = (
                self.core.query_pipeline_statistics(pipeline)
            )
        return module, pipeline

    def _pipeline_bytes(self, code, layout):
        module = vk.vkCreateShaderModule(
            self.core.device,
            vk.VkShaderModuleCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize=len(code), pCode=code,
            ), None,
        )
        stage = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT, module=module, pName="main",
        )
        pipeline = vk.vkCreateComputePipelines(
            self.core.device,
            self.core.pipeline_cache or vk.VK_NULL_HANDLE,
            1,
            [vk.VkComputePipelineCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                stage=stage, layout=layout,
            )], None,
        )[0]
        return module, pipeline

    def ensure_custom_material_pipelines(self):
        """Install staged evaluators for the resident scene's custom attributes."""
        scene = self.core.scene_resources.scene
        from ...materials import builtin_material
        programs = scene.material_programs(
            self.core.config.material_program or builtin_material
        )
        layout = self.core.scene_custom_attribute_layout
        overlapping_volumes = len(scene.visible_volumes) > 1
        scattering_volumes = any(
            volume.material.scattering_scale > 0.0
            for volume in scene.visible_volumes
        )
        multiple_scattering_volumes = any(
            volume.material.scattering_scale > 0.0
            and volume.material.scattering_orders > 1
            for volume in scene.visible_volumes
        )
        volume_empty_space_skipping = bool(
            self.core.scene_resources.volume_empty_space_skipping
        )
        signature = (
            programs, layout, overlapping_volumes, scattering_volumes,
            multiple_scattering_volumes, volume_empty_space_skipping,
            self.core.config.wavefront_ordinaryshade_shade,
            self.core.native_textures_enabled,
            self.core.config.wavefront_profiling,
            self._denoiser_signals_active(),
            self.core.config.material_modifier,
        )
        if signature == self.custom_material_signature:
            return
        vk.vkDeviceWaitIdle(self.core.device)
        for pipeline in (
            self.custom_primary_pipeline, self.custom_shade_pipeline,
        ):
            if pipeline:
                vk.vkDestroyPipeline(self.core.device, pipeline, None)
        for module in (self.custom_primary_module, self.custom_shade_module):
            if module:
                vk.vkDestroyShaderModule(self.core.device, module, None)
        self.custom_primary_module = self.custom_primary_pipeline = None
        self.custom_shade_module = self.custom_shade_pipeline = None
        if layout is not None:
            from ...shaders.compiler import compile_wavefront_material_shader
            primary = compile_wavefront_material_shader(
                "wavefront_primary.comp", programs,
                attribute_layout=layout, attribute_binding=24,
                overlapping_volumes=overlapping_volumes,
                scattering_volumes=scattering_volumes,
                multiple_scattering_volumes=multiple_scattering_volumes,
                volume_empty_space_skipping=volume_empty_space_skipping,
                native_textures=self.core.native_textures_enabled,
                profiling=self.core.config.wavefront_profiling,
                denoiser_signal_capture=self._denoiser_signals_active(),
                material_modifier=self.core.config.material_modifier,
            )
            shade = compile_wavefront_material_shader(
                (
                    "wavefront_shade_candidate.glsl"
                    if self.core.config.wavefront_ordinaryshade_shade
                    else "wavefront_shade.comp"
                ), programs,
                attribute_layout=layout, attribute_binding=16,
                overlapping_volumes=overlapping_volumes,
                scattering_volumes=scattering_volumes,
                multiple_scattering_volumes=multiple_scattering_volumes,
                volume_empty_space_skipping=volume_empty_space_skipping,
                native_textures=self.core.native_textures_enabled,
                profiling=self.core.config.wavefront_profiling,
                denoiser_signal_capture=self._denoiser_signals_active(),
                material_modifier=self.core.config.material_modifier,
            )
            self.custom_primary_module, self.custom_primary_pipeline = (
                self._pipeline_bytes(primary, self.primary_pipeline_layout)
            )
            self.custom_shade_module, self.custom_shade_pipeline = (
                self._pipeline_bytes(shade, self.shade_pipeline_layout)
            )
        self.custom_material_signature = signature

    def _wavefront_stage_shader(self, stem, suffix=""):
        if (
            stem == "wavefront_shade"
            and self.core.config.wavefront_ordinaryshade_shade
        ):
            stem = "wavefront_shade_ordinaryshade"
        return f"{stem}{suffix}.comp"

    def ensure_overlap_pipelines(self):
        """Lazily create the heavier kernels used by overlapping media."""
        scene = self.core.scene_resources.scene
        if len(scene.visible_volumes) <= 1:
            return
        if self.custom_primary_pipeline is not None:
            # Custom staged shaders are compiled with the overlap definition
            # as part of their scene-dependent signature.
            return
        strategy = self.core.resolved_execution_strategy
        if strategy == "ser":
            raise RuntimeError(
                "overlapping volumes are not yet supported by the SER strategy"
            )
        required = {"wavefront_primary", "wavefront_shade"}
        required.add({
            "wavefront": "wavefront_primary",
            "hybrid": "wavefront_hybrid",
            "megakernel": "wavefront_megakernel",
            "persistent": "wavefront_persistent",
        }[strategy])
        if strategy == "persistent" and self.core.config.wavefront_persistent_coarse_tiles:
            required.add("wavefront_persistent_coarse")
        if strategy == "hybrid" and self.core.config.wavefront_persistent_continuations:
            required.add("wavefront_persistent_continuation")
        native_suffix = "_native" if self.core.native_textures_enabled else ""
        profile_suffix = "_profile" if self.core.config.wavefront_profiling else ""
        for stem in required:
            if stem in self.overlap_pipelines:
                continue
            module, pipeline = self._pipeline(
                self._wavefront_stage_shader(
                    stem, native_suffix + profile_suffix + "_overlap"
                ),
                (self.shade_pipeline_layout
                 if stem == "wavefront_shade"
                 else self.primary_pipeline_layout),
            )
            self.overlap_modules[stem] = module
            self.overlap_pipelines[stem] = pipeline

    @staticmethod
    def _scene_has_volume_scattering(scene):
        return any(
            volume.material.scattering_scale > 0.0
            for volume in scene.visible_volumes
        )

    @staticmethod
    def _scene_has_multiple_volume_scattering(scene):
        return any(
            volume.material.scattering_scale > 0.0
            and volume.material.scattering_orders > 1
            for volume in scene.visible_volumes
        )

    def ensure_scattering_pipelines(self):
        """Lazily create kernels containing participating-media lighting."""
        scene = self.core.scene_resources.scene
        if not self._scene_has_volume_scattering(scene):
            return
        if self.custom_primary_pipeline is not None:
            return
        strategy = self.core.resolved_execution_strategy
        if strategy == "ser":
            raise RuntimeError(
                "volume scattering is not yet supported by the SER strategy"
            )
        required = {"wavefront_primary", "wavefront_shade"}
        required.add({
            "wavefront": "wavefront_primary",
            "hybrid": "wavefront_hybrid",
            "megakernel": "wavefront_megakernel",
            "persistent": "wavefront_persistent",
        }[strategy])
        if strategy == "persistent" and self.core.config.wavefront_persistent_coarse_tiles:
            required.add("wavefront_persistent_coarse")
        if strategy == "hybrid" and self.core.config.wavefront_persistent_continuations:
            required.add("wavefront_persistent_continuation")
        overlap = len(scene.visible_volumes) > 1
        multiple = self._scene_has_multiple_volume_scattering(scene)
        native_suffix = "_native" if self.core.native_textures_enabled else ""
        profile_suffix = "_profile" if self.core.config.wavefront_profiling else ""
        for stem in required:
            key = (stem, overlap, multiple)
            if key in self.scattering_pipelines:
                continue
            suffix = ("_overlap" if overlap else "") + "_scatter" \
                + ("_multi" if multiple else "")
            module, pipeline = self._pipeline(
                self._wavefront_stage_shader(
                    stem, native_suffix + profile_suffix + suffix
                ),
                (self.shade_pipeline_layout
                 if stem == "wavefront_shade"
                 else self.primary_pipeline_layout),
            )
            self.scattering_modules[key] = module
            self.scattering_pipelines[key] = pipeline

    def ensure_volume_skipping_pipelines(self):
        """Lazily create volume kernels with conservative brick traversal."""
        resources = self.core.scene_resources
        if not resources.volume_empty_space_skipping:
            return
        if self.custom_primary_pipeline is not None:
            return
        scene = resources.scene
        strategy = self.core.resolved_execution_strategy
        if strategy == "ser":
            raise RuntimeError(
                "volume empty-space skipping is not yet supported by SER; "
                "set volume_empty_space_skipping=False"
            )
        required = {"wavefront_primary", "wavefront_shade"}
        required.add({
            "wavefront": "wavefront_primary",
            "hybrid": "wavefront_hybrid",
            "megakernel": "wavefront_megakernel",
            "persistent": "wavefront_persistent",
        }[strategy])
        if (
            strategy == "persistent"
            and self.core.config.wavefront_persistent_coarse_tiles
        ):
            required.add("wavefront_persistent_coarse")
        if (
            strategy == "hybrid"
            and self.core.config.wavefront_persistent_continuations
        ):
            required.add("wavefront_persistent_continuation")
        overlap = len(scene.visible_volumes) > 1
        scattering = self._scene_has_volume_scattering(scene)
        multiple = self._scene_has_multiple_volume_scattering(scene)
        native_suffix = "_native" if self.core.native_textures_enabled else ""
        profile_suffix = (
            "_profile" if self.core.config.wavefront_profiling else ""
        )
        for stem in required:
            key = (stem, overlap, scattering, multiple)
            if key in self.volume_skipping_pipelines:
                continue
            suffix = ("_overlap" if overlap else "") \
                + ("_scatter" if scattering else "") \
                + ("_multi" if multiple else "") + "_skip"
            module, pipeline = self._pipeline(
                self._wavefront_stage_shader(
                    stem, native_suffix + profile_suffix + suffix
                ),
                (self.shade_pipeline_layout
                 if stem == "wavefront_shade"
                 else self.primary_pipeline_layout),
            )
            self.volume_skipping_modules[key] = module
            self.volume_skipping_pipelines[key] = pipeline

    def _volume_pipeline(self, stem, scene):
        # Scene-dependent custom pipelines are compiled with the active volume
        # feature definitions already applied. No parallel precompiled volume
        # variant exists (or is needed) for that specialization.
        if self.custom_primary_pipeline is not None:
            return None
        overlap = len(scene.visible_volumes) > 1
        if self.core.scene_resources.volume_empty_space_skipping:
            scattering = self._scene_has_volume_scattering(scene)
            multiple = self._scene_has_multiple_volume_scattering(scene)
            return self.volume_skipping_pipelines[
                (stem, overlap, scattering, multiple)
            ]
        if self._scene_has_volume_scattering(scene):
            multiple = self._scene_has_multiple_volume_scattering(scene)
            return self.scattering_pipelines[(stem, overlap, multiple)]
        if overlap:
            return self.overlap_pipelines[stem]
        return None

    def _raygen_pipeline(self, shader_name, layout):
        """Create a raygen/miss pipeline and its device-local two-record SBT."""
        modules = []
        stages = []
        for name, stage_flag in (
            (shader_name, vk.VK_SHADER_STAGE_RAYGEN_BIT_KHR),
            ("ser_probe.rmiss", vk.VK_SHADER_STAGE_MISS_BIT_KHR),
        ):
            code = files("ordinarylight").joinpath(
                f"shaders/{name}.spv"
            ).read_bytes()
            module = vk.vkCreateShaderModule(
                self.core.device,
                vk.VkShaderModuleCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                    codeSize=len(code), pCode=code,
                ), None,
            )
            modules.append(module)
            stages.append(vk.VkPipelineShaderStageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                stage=stage_flag, module=module, pName="main",
            ))
        groups = [vk.VkRayTracingShaderGroupCreateInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
            type=vk.VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
            generalShader=index,
            closestHitShader=vk.VK_SHADER_UNUSED_KHR,
            anyHitShader=vk.VK_SHADER_UNUSED_KHR,
            intersectionShader=vk.VK_SHADER_UNUSED_KHR,
        ) for index in range(2)]
        pipeline = self.core.create_ray_tracing_pipelines(
            self.core.device, vk.ffi.NULL, vk.ffi.NULL, 1,
            [vk.VkRayTracingPipelineCreateInfoKHR(
                sType=vk.VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
                stageCount=2, pStages=stages,
                groupCount=2, pGroups=groups,
                maxPipelineRayRecursionDepth=1, layout=layout,
            )], None,
        )[0]
        handle_size = self.core.ray_tracing_shader_group_handle_size
        stride = _align_up(
            handle_size,
            self.core.ray_tracing_shader_group_handle_alignment,
        )
        miss_offset = _align_up(
            stride, self.core.ray_tracing_shader_group_base_alignment
        )
        handles = vk.ffi.new("uint8_t[]", handle_size * 2)
        self.core.get_ray_tracing_shader_group_handles(
            self.core.device, pipeline, 0, 2, handle_size * 2, handles,
        )
        handle_bytes = bytes(vk.ffi.buffer(handles, handle_size * 2))
        payload = np.zeros(miss_offset + stride, dtype=np.uint8)
        payload[:handle_size] = np.frombuffer(
            handle_bytes[:handle_size], dtype=np.uint8
        )
        payload[miss_offset:miss_offset + handle_size] = np.frombuffer(
            handle_bytes[handle_size:], dtype=np.uint8
        )
        sbt = self.core._create_uploaded_device_buffer(
            payload,
            vk.VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR
            | BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            device_address=True,
        )
        address = self.core._buffer_address(sbt)
        return modules[0], modules[1], pipeline, sbt, (
            vk.VkStridedDeviceAddressRegionKHR(
                deviceAddress=address, stride=stride, size=stride
            ),
            vk.VkStridedDeviceAddressRegionKHR(
                deviceAddress=address + miss_offset,
                stride=stride, size=stride,
            ),
            vk.VkStridedDeviceAddressRegionKHR(),
        )

    def _indirect_capture_stride(self):
        """Return an exact integer reservoir-to-pixel stride, or zero.

        Exact 1x, 1/2x, and 1/4x plans can avoid writing secondary path state
        for pixels that cannot seed a reservoir. Arbitrary scales retain the
        conservative all-pixel capture path.
        """
        if not self.core.config.wavefront_indirect_reuse_candidates:
            return 0
        inverse_scale = 1.0 / self.core.config.wavefront_indirect_reuse_scale
        stride = int(round(inverse_scale))
        return stride if stride in (1, 2, 4) and abs(
            inverse_scale - stride) < 1e-6 else 0

    def _denoiser_signals_active(self):
        return bool(
            self.core.config.denoiser_signal_capture
            or self.core.config.denoiser_enabled
        )

    def _create_pipelines(self):
        shader_stages = vk.VK_SHADER_STAGE_COMPUTE_BIT
        if self.core.config.wavefront_ser:
            shader_stages |= vk.VK_SHADER_STAGE_RAYGEN_BIT_KHR
        storage = lambda binding: vk.VkDescriptorSetLayoutBinding(
            binding=binding, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            descriptorCount=1, stageFlags=shader_stages,
        )
        storage_image = lambda binding: vk.VkDescriptorSetLayoutBinding(
            binding=binding, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            descriptorCount=1, stageFlags=shader_stages,
        )
        uniform = lambda binding: vk.VkDescriptorSetLayoutBinding(
            binding=binding, descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            descriptorCount=1, stageFlags=shader_stages,
        )
        acceleration = vk.VkDescriptorSetLayoutBinding(
            binding=0,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
            descriptorCount=1, stageFlags=shader_stages,
        )
        sampled_textures = lambda binding: vk.VkDescriptorSetLayoutBinding(
            binding=binding,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            descriptorCount=MAX_NATIVE_TEXTURES * 2,
            stageFlags=shader_stages,
        )
        sampled_volumes = lambda binding: vk.VkDescriptorSetLayoutBinding(
            binding=binding,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            descriptorCount=MAX_NATIVE_VOLUMES,
            stageFlags=shader_stages,
        )
        self.generate_layout = self._layout([
            storage(0), storage(1), storage(2), storage(3)
        ])
        primary_bindings = (
            [acceleration] + [storage(binding) for binding in range(1, 8)]
            + [storage_image(8), storage_image(9), storage(10), storage(11),
               storage(12), storage(13), storage(16), storage(17), storage(18),
               storage_image(19), storage_image(20), storage_image(21),
               storage_image(22), storage(23), storage(24), storage(25),
               storage(26), storage(27), storage(28), sampled_volumes(29)]
        )
        if self.core.native_textures_enabled:
            primary_bindings.append(sampled_textures(14))
        if self.core.config.wavefront_profiling:
            primary_bindings.append(storage(15))
        self.primary_layout = self._layout(primary_bindings)
        self.intersect_layout = self._layout(
            [acceleration, storage(1), storage(2), storage(3)]
        )
        shade_bindings = (
            [storage(binding) for binding in range(8)]
            + [vk.VkDescriptorSetLayoutBinding(
                binding=8,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ), storage(9), storage(10), storage(11), storage(12), storage(15),
            storage(16), storage(17), storage(18), storage(19), storage(20),
            sampled_volumes(21)]
        )
        if self.core.native_textures_enabled:
            shade_bindings.append(sampled_textures(13))
        if self.core.config.wavefront_profiling:
            shade_bindings.append(storage(14))
        self.shade_layout = self._layout(shade_bindings)
        self.resolve_layout = self._layout([storage(0), storage(1)])
        self.indirect_layout = self._layout([storage(0), storage(1)])
        self.wavefront_tone_layout = self._layout([storage(0), storage(1)])
        self.wavefront_image_layout = self._layout([
            storage(0),
            vk.VkDescriptorSetLayoutBinding(
                binding=1, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                descriptorCount=1, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ), storage(2), storage(3), storage(4), storage(5),
        ])
        reconstruct_bindings = [storage_image(binding) for binding in range(7)]
        reconstruct_bindings.append(vk.VkDescriptorSetLayoutBinding(
            binding=7, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            descriptorCount=8, stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
        ))
        self.reconstruct_layout = self._layout(
            reconstruct_bindings + [storage(8), storage(9), storage_image(10)]
        )
        if self.core.config.denoiser_enabled:
            self.relax_prepare_layout = self._layout(
                [storage(0), storage(1)]
                + [storage_image(binding) for binding in range(2, 9)]
                + [storage(9), storage(10), storage(11)]
                + [storage_image(12)]
            )
            self.relax_temporal_layout = self._layout(
                [storage_image(binding) for binding in range(12)]
                + [uniform(12), storage_image(13), storage_image(14)]
            )
            self.relax_atrous_layout = self._layout(
                [storage_image(binding) for binding in range(5)]
            )
            self.relax_compose_layout = self._layout(
                [storage_image(binding) for binding in range(4)]
            )
        if self.core.config.wavefront_indirect_reuse_storage:
            self.indirect_reuse_clear_layout = self._layout([storage(0)])
        if self.core.config.wavefront_indirect_reuse_candidates:
            indirect_acceleration = vk.VkDescriptorSetLayoutBinding(
                binding=12,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            )
            self.indirect_reuse_candidate_layout = self._layout([
                storage(0), storage_image(1), storage_image(2),
                storage_image(3), storage(4), storage(5),
                storage_image(6), storage_image(7), storage(8),
                storage_image(9), storage_image(10), storage(11),
                indirect_acceleration,
            ])
        if (self.core.config.wavefront_indirect_reuse_debug_view != "off"
                or self.core.config.wavefront_indirect_reuse_apply):
            self.indirect_reuse_debug_layout = self._layout([
                storage(0), storage_image(1), storage(2), storage_image(3),
            ])
        if self.core.config.wavefront_material_bucketing:
            self.bucket_intersect_layout = self._layout([
                acceleration, storage(1), storage(2), storage(3),
                storage(4), storage(5),
            ])
        self.generate_pipeline_layout = self._pipeline_layout(self.generate_layout, 32)
        self.primary_pipeline_layout = self._pipeline_layout(self.primary_layout, 176)
        self.intersect_pipeline_layout = self._pipeline_layout(self.intersect_layout)
        self.shade_pipeline_layout = self._pipeline_layout(self.shade_layout, 56)
        self.resolve_pipeline_layout = self._pipeline_layout(self.resolve_layout, 16)
        self.indirect_pipeline_layout = self._pipeline_layout(self.indirect_layout)
        self.wavefront_tone_pipeline_layout = self._pipeline_layout(
            self.wavefront_tone_layout, 16
        )
        self.wavefront_image_pipeline_layout = self._pipeline_layout(
            self.wavefront_image_layout, 32
        )
        self.reconstruct_pipeline_layout = self._pipeline_layout(
            self.reconstruct_layout, RECONSTRUCT_PUSH_SIZE
        )
        if self.relax_prepare_layout:
            self.relax_prepare_pipeline_layout = self._pipeline_layout(
                self.relax_prepare_layout, 16
            )
        if self.relax_temporal_layout:
            self.relax_temporal_pipeline_layout = self._pipeline_layout(
                self.relax_temporal_layout
            )
        if self.relax_atrous_layout:
            self.relax_atrous_pipeline_layout = self._pipeline_layout(
                self.relax_atrous_layout, 32
            )
        if self.relax_compose_layout:
            self.relax_compose_pipeline_layout = self._pipeline_layout(
                self.relax_compose_layout, 16
            )
        if self.indirect_reuse_clear_layout:
            self.indirect_reuse_clear_pipeline_layout = self._pipeline_layout(
                self.indirect_reuse_clear_layout, 4
            )
        if self.indirect_reuse_candidate_layout:
            self.indirect_reuse_candidate_pipeline_layout = self._pipeline_layout(
                self.indirect_reuse_candidate_layout, 36
            )
        if self.indirect_reuse_debug_layout:
            self.indirect_reuse_debug_pipeline_layout = self._pipeline_layout(
                self.indirect_reuse_debug_layout, 28
            )
        if self.bucket_intersect_layout:
            self.bucket_intersect_pipeline_layout = self._pipeline_layout(
                self.bucket_intersect_layout
            )
        self.generate_module, self.generate_pipeline = self._pipeline(
            "wavefront_generate.comp", self.generate_pipeline_layout
        )
        native_suffix = "_native" if self.core.native_textures_enabled else ""
        profile_suffix = "_profile" if self.core.config.wavefront_profiling else ""
        production_restir = (
            self.core.config.wavefront_restir_specialization
            and not self.core.config.wavefront_profiling
            and not self.core.config.wavefront_restir_generalized_mis
            and not self.core.config.wavefront_unified_primary_restir
            and not self.core.config.wavefront_stratified_primary_restir
        )
        self.production_restir = production_restir
        production_suffix = "_production" if production_restir else ""
        self.primary_module, self.primary_pipeline = self._pipeline(
            f"wavefront_primary{native_suffix}{profile_suffix}.comp",
            self.primary_pipeline_layout,
        )
        strategy = self.strategy
        eager_strategies = False
        scene = self.core.scene_resources.scene
        opaque_scene = self.core._use_opaque_scene_specialization(scene)
        untextured_scene = not scene.textures
        if strategy == "hybrid" and not opaque_scene:
            self.hybrid_module, self.hybrid_pipeline = self._pipeline(
                f"wavefront_hybrid{native_suffix}{production_suffix}"
                f"{profile_suffix}.comp",
                self.primary_pipeline_layout,
            )
        if (
            strategy == "hybrid" and opaque_scene
            and not (
                self.core.config.wavefront_untextured_specialization
                and untextured_scene and production_restir
            )
        ):
            self.hybrid_opaque_module, self.hybrid_opaque_pipeline = self._pipeline(
                f"wavefront_hybrid{native_suffix}_opaque{profile_suffix}.comp",
                self.primary_pipeline_layout,
            )
        if (
            strategy == "hybrid" and opaque_scene and untextured_scene
            and self.core.config.wavefront_untextured_specialization
            and production_restir
        ):
            (
                self.hybrid_opaque_untextured_production_module,
                self.hybrid_opaque_untextured_production_pipeline,
            ) = self._pipeline(
                f"wavefront_hybrid{native_suffix}"
                "_opaque_untextured_production.comp",
                self.primary_pipeline_layout,
            )
        megakernel_untextured = bool(
            strategy == "megakernel"
            and self.core.config.wavefront_untextured_specialization
            and untextured_scene
            and (
                opaque_scene
                or (not native_suffix and not profile_suffix)
            )
        )
        if strategy == "megakernel" and not opaque_scene and not megakernel_untextured:
            self.megakernel_module, self.megakernel_pipeline = self._pipeline(
                f"wavefront_megakernel{native_suffix}{profile_suffix}.comp",
                self.primary_pipeline_layout,
            )
        if (megakernel_untextured and not opaque_scene
                and not native_suffix and not profile_suffix):
            (
                self.megakernel_untextured_module,
                self.megakernel_untextured_pipeline,
            ) = self._pipeline(
                "wavefront_megakernel_untextured.comp",
                self.primary_pipeline_layout,
            )
            swizzle_width = self.core.config.wavefront_megakernel_group_swizzle
            if swizzle_width:
                module, pipeline = self._pipeline(
                    "wavefront_megakernel_untextured"
                    f"_swizzle{swizzle_width}.comp",
                    self.primary_pipeline_layout,
                )
                self.megakernel_untextured_swizzle_modules[
                    swizzle_width
                ] = module
                self.megakernel_untextured_swizzle_pipelines[
                    swizzle_width
                ] = pipeline
        if strategy == "megakernel" and opaque_scene and not megakernel_untextured:
            (
                self.megakernel_opaque_module,
                self.megakernel_opaque_pipeline,
            ) = self._pipeline(
                f"wavefront_megakernel{native_suffix}_opaque"
                f"{profile_suffix}.comp",
                self.primary_pipeline_layout,
            )
        if (
            strategy == "megakernel" and opaque_scene
            and megakernel_untextured
            and self.core.config.wavefront_untextured_specialization_part == "full"
        ):
            (
                self.megakernel_opaque_untextured_module,
                self.megakernel_opaque_untextured_pipeline,
            ) = self._pipeline(
                f"wavefront_megakernel{native_suffix}_opaque_untextured"
                f"{profile_suffix}.comp",
                self.primary_pipeline_layout,
            )
        if (
            strategy == "megakernel" and opaque_scene
            and megakernel_untextured and not native_suffix and not profile_suffix
            and self.core.config.wavefront_untextured_specialization_part == "primary"
        ):
            (
                self.megakernel_untextured_primary_module,
                self.megakernel_untextured_primary_pipeline,
            ) = self._pipeline(
                "wavefront_megakernel_opaque_untextured_primary.comp",
                self.primary_pipeline_layout,
            )
        if (
            strategy == "megakernel" and opaque_scene
            and megakernel_untextured and not native_suffix and not profile_suffix
            and self.core.config.wavefront_untextured_specialization_part == "secondary"
        ):
            (
                self.megakernel_untextured_secondary_module,
                self.megakernel_untextured_secondary_pipeline,
            ) = self._pipeline(
                "wavefront_megakernel_opaque_untextured_secondary.comp",
                self.primary_pipeline_layout,
            )
        if eager_strategies or strategy == "persistent":
            self.persistent_module, self.persistent_pipeline = self._pipeline(
                f"wavefront_persistent{native_suffix}{profile_suffix}.comp",
                self.primary_pipeline_layout,
            )
        if ((eager_strategies or strategy == "persistent")
                and self.core.config.wavefront_persistent_coarse_tiles):
            (
                self.persistent_coarse_module,
                self.persistent_coarse_pipeline,
            ) = self._pipeline(
                f"wavefront_persistent_coarse{native_suffix}{profile_suffix}.comp",
                self.primary_pipeline_layout,
            )
        if ((eager_strategies or strategy == "hybrid")
                and self.core.config.wavefront_persistent_continuations):
            (
                self.persistent_continuation_opaque_module,
                self.persistent_continuation_opaque_pipeline,
            ) = self._pipeline(
                f"wavefront_persistent_continuation{native_suffix}_opaque"
                f"{profile_suffix}.comp",
                self.primary_pipeline_layout,
            )
            (
                self.persistent_continuation_module,
                self.persistent_continuation_pipeline,
            ) = self._pipeline(
                f"wavefront_persistent_continuation{native_suffix}"
                f"{production_suffix}{profile_suffix}.comp",
                self.primary_pipeline_layout,
            )
        self.intersect_module, self.intersect_pipeline = self._pipeline(
            "wavefront_intersect.comp", self.intersect_pipeline_layout
        )
        shade_shader = self._wavefront_stage_shader(
            "wavefront_shade", native_suffix + profile_suffix
        )
        self.shade_module, self.shade_pipeline = self._pipeline(
            shade_shader,
            self.shade_pipeline_layout,
        )
        self.resolve_module, self.resolve_pipeline = self._pipeline(
            "wavefront_resolve.comp", self.resolve_pipeline_layout
        )
        self.indirect_module, self.indirect_pipeline = self._pipeline(
            "wavefront_prepare_indirect.comp", self.indirect_pipeline_layout
        )
        self.wavefront_tone_module, self.wavefront_tone_pipeline = self._pipeline(
            "wavefront_tone_map.comp", self.wavefront_tone_pipeline_layout
        )
        self.wavefront_image_module, self.wavefront_image_pipeline = self._pipeline(
            "wavefront_path_to_hdr.comp", self.wavefront_image_pipeline_layout
        )
        self.reconstruct_module, self.reconstruct_pipeline = self._pipeline(
            "wavefront_reconstruct.comp", self.reconstruct_pipeline_layout
        )
        if self.relax_prepare_pipeline_layout:
            self.relax_prepare_module, self.relax_prepare_pipeline = self._pipeline(
                "denoiser_relax_prepare.comp",
                self.relax_prepare_pipeline_layout,
            )
        if self.relax_temporal_pipeline_layout:
            self.relax_temporal_module, self.relax_temporal_pipeline = self._pipeline(
                "denoiser_relax_temporal.comp",
                self.relax_temporal_pipeline_layout,
            )
        if self.relax_atrous_pipeline_layout:
            self.relax_atrous_module, self.relax_atrous_pipeline = self._pipeline(
                "denoiser_relax_atrous.comp",
                self.relax_atrous_pipeline_layout,
            )
        if self.relax_compose_pipeline_layout:
            self.relax_compose_module, self.relax_compose_pipeline = self._pipeline(
                "denoiser_relax_compose.comp",
                self.relax_compose_pipeline_layout,
            )
        if self.core.formatless_storage_write_supported:
            (
                self.reconstruct_bgra_module,
                self.reconstruct_bgra_pipeline,
            ) = self._pipeline(
                "wavefront_reconstruct_bgra.comp",
                self.reconstruct_pipeline_layout,
            )
        if self.indirect_reuse_clear_pipeline_layout:
            (
                self.indirect_reuse_clear_module,
                self.indirect_reuse_clear_pipeline,
            ) = self._pipeline(
                "wavefront_indirect_clear.comp",
                self.indirect_reuse_clear_pipeline_layout,
            )
        if self.indirect_reuse_candidate_pipeline_layout:
            (
                self.indirect_reuse_candidate_module,
                self.indirect_reuse_candidate_pipeline,
            ) = self._pipeline(
                "wavefront_indirect_candidates.comp",
                self.indirect_reuse_candidate_pipeline_layout,
            )
        if self.indirect_reuse_debug_pipeline_layout:
            self.indirect_reuse_debug_module, self.indirect_reuse_debug_pipeline = (
                self._pipeline(
                    "wavefront_indirect_debug.comp",
                    self.indirect_reuse_debug_pipeline_layout,
                )
            )
        if self.bucket_intersect_layout:
            self.bucket_intersect_module, self.bucket_intersect_pipeline = (
                self._pipeline(
                    "wavefront_intersect_bucketed.comp",
                    self.bucket_intersect_pipeline_layout,
                )
            )
        if self.core.config.wavefront_execution_strategy == "ser":
            (
                self.ser_megakernel_module,
                self.ser_miss_module,
                self.ser_megakernel_pipeline,
                self.ser_megakernel_sbt,
                ser_regions,
            ) = self._raygen_pipeline(
                ("wavefront_megakernel_ser.rgen"
                 if self.core.config.wavefront_ser_reorder
                 else "wavefront_megakernel.rgen"),
                self.primary_pipeline_layout,
            )
            (
                self.ser_raygen_region,
                self.ser_miss_region,
                self.ser_empty_region,
            ) = ser_regions
        self.descriptor_pool = vk.vkCreateDescriptorPool(
            self.core.device,
            vk.VkDescriptorPoolCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                maxSets=66,
                poolSizeCount=5,
                pPoolSizes=[
                    vk.VkDescriptorPoolSize(
                        type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=240
                    ),
                    vk.VkDescriptorPoolSize(
                        type=vk.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                        descriptorCount=8,
                    ),
                    vk.VkDescriptorPoolSize(
                        type=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                        descriptorCount=240,
                    ),
                    vk.VkDescriptorPoolSize(
                        type=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                        descriptorCount=8,
                    ),
                ] + [vk.VkDescriptorPoolSize(
                    type=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                    descriptorCount=(MAX_NATIVE_VOLUMES * 6
                        + (MAX_NATIVE_TEXTURES * 2 * 4
                           if self.core.native_textures_enabled else 0)),
                )],
            ), None,
        )
        shade_set_count = 4 if self.core.config.wavefront_profiling else 2
        fixed_layouts = [
            self.generate_layout, self.generate_layout,
            self.primary_layout, self.primary_layout,
            self.intersect_layout, self.intersect_layout,
        ] + [self.shade_layout] * shade_set_count + [
            self.resolve_layout,
            self.indirect_layout, self.indirect_layout,
            self.wavefront_tone_layout,
            self.wavefront_image_layout, self.wavefront_image_layout,
            self.reconstruct_layout, self.reconstruct_layout,
        ]
        if self.indirect_reuse_clear_layout:
            fixed_layouts += [self.indirect_reuse_clear_layout] * 2
        if self.indirect_reuse_candidate_layout:
            fixed_layouts += [self.indirect_reuse_candidate_layout] * 2
        if self.indirect_reuse_debug_layout:
            fixed_layouts += [self.indirect_reuse_debug_layout] * 2
        if self.relax_prepare_layout:
            fixed_layouts += [self.relax_prepare_layout] * 2
        if self.relax_temporal_layout:
            # Two frame slots, with independent diffuse and specular histories.
            fixed_layouts += [self.relax_temporal_layout] * 4
        if self.relax_atrous_layout:
            # Two frame slots, two lobes, and first/reverse/forward paths.
            fixed_layouts += [self.relax_atrous_layout] * 12
        if self.relax_compose_layout:
            # Compose from either side of the A-trous ping-pong pair.
            fixed_layouts += [self.relax_compose_layout] * 4
        allocated = vk.vkAllocateDescriptorSets(
            self.core.device,
            vk.VkDescriptorSetAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptorPool=self.descriptor_pool,
                descriptorSetCount=len(fixed_layouts),
                pSetLayouts=fixed_layouts,
            ),
        )
        self.generate_sets = list(allocated[0:2])
        self.primary_sets = list(allocated[2:4])
        self.intersect_sets = list(allocated[4:6])
        shade_end = 6 + shade_set_count
        self.shade_sets = list(allocated[6:shade_end])
        self.resolve_set = allocated[shade_end]
        self.indirect_sets = list(allocated[shade_end + 1:shade_end + 3])
        self.wavefront_tone_set = allocated[shade_end + 3]
        self.wavefront_image_sets = list(allocated[shade_end + 4:shade_end + 6])
        self.reconstruct_sets = list(allocated[shade_end + 6:shade_end + 8])
        cursor = shade_end + 8
        if self.indirect_reuse_clear_layout:
            self.indirect_reuse_clear_sets = list(allocated[cursor:cursor + 2])
            cursor += 2
        if self.indirect_reuse_candidate_layout:
            self.indirect_reuse_candidate_sets = list(
                allocated[cursor:cursor + 2]
            )
            cursor += 2
        if self.indirect_reuse_debug_layout:
            self.indirect_reuse_debug_sets = list(
                allocated[cursor:cursor + 2]
            )
            cursor += 2
        if self.relax_prepare_layout:
            self.relax_prepare_sets = list(allocated[cursor:cursor + 2])
            cursor += 2
        if self.relax_temporal_layout:
            self.relax_temporal_sets = list(allocated[cursor:cursor + 4])
            cursor += 4
        if self.relax_atrous_layout:
            self.relax_atrous_sets = list(allocated[cursor:cursor + 12])
            cursor += 12
        if self.relax_compose_layout:
            self.relax_compose_sets = list(allocated[cursor:cursor + 4])
        if self.bucket_intersect_layout:
            coherent_layouts = (
                [self.shade_layout] * (
                    8 if self.core.config.wavefront_profiling else 4
                )
                + [self.indirect_layout] * 2
                + [self.bucket_intersect_layout] * 2
            )
            coherent = vk.vkAllocateDescriptorSets(
                self.core.device,
                vk.VkDescriptorSetAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    descriptorPool=self.descriptor_pool,
                    descriptorSetCount=len(coherent_layouts),
                    pSetLayouts=coherent_layouts,
                ),
            )
            coherent_shade_count = (
                8 if self.core.config.wavefront_profiling else 4
            )
            coherent = list(coherent)
            self.coherent_shade_sets = coherent[0:coherent_shade_count]
            self.coherent_indirect_sets = coherent[
                coherent_shade_count:coherent_shade_count + 2
            ]
            self.bucket_intersect_sets = coherent[
                coherent_shade_count + 2:coherent_shade_count + 4
            ]
            for descriptor_set, hit_buffer in zip(
                self.coherent_indirect_sets, self.coherent_hit_buffers
            ):
                self._write_storage_set(descriptor_set, (
                    (0, hit_buffer), (1, self.indirect_buffer)
                ))
        for descriptor_set, camera_buffer in zip(
            self.generate_sets, self.camera_buffers
        ):
            self._write_storage_set(descriptor_set, (
                (0, self.ray_buffer), (1, self.path_buffer),
                (2, self.medium_buffer), (3, camera_buffer),
            ))
        for slot, (descriptor_set, camera_buffer) in enumerate(zip(
            self.primary_sets, self.camera_buffers
        )):
            bindings = [
                (1, self.path_buffer), (2, self.core.scene_material_buffer),
                (3, self.core.scene_vertex_buffer),
                (4, self.core.scene_attribute_buffer),
                (5, self.next_ray_buffer), (6, self.medium_buffer),
                (7, camera_buffer), (12, self.core.scene_texture_buffer),
                (13, self.core.scene_texture_binding_buffer),
                (23, self.secondary_path_buffer),
                (25, self.core.scene_volume_header_buffer),
                (26, self.core.scene_volume_scalar_buffer),
                (27, self.core.scene_volume_transfer_buffer),
                (28, self.core.scene_triangle_volume_buffer),
            ]
            if self.core.scene_custom_attribute_buffer is not None:
                bindings.append((24, self.core.scene_custom_attribute_buffer))
            if self.core.config.wavefront_profiling:
                bindings.append((15, self.work_counter_buffers[slot]))
            self._write_storage_set(descriptor_set, bindings)
        self._write_storage_set(
            self.resolve_set,
            ((0, self.path_buffer), (1, self.resolve_buffer)),
        )
        for descriptor_set, ray_buffer in zip(
            self.indirect_sets, (self.ray_buffer, self.next_ray_buffer)
        ):
            self._write_storage_set(
                descriptor_set, ((0, ray_buffer), (1, self.indirect_buffer))
            )
        self._write_storage_set(
            self.wavefront_tone_set,
            ((0, self.resolve_buffer), (1, self.packed_buffer)),
        )
        for slot, descriptor_set in enumerate(self.wavefront_image_sets):
            self._write_storage_set(
                descriptor_set, (
                    (0, self.path_buffer), (2, self.secondary_path_buffer),
                    (3, self.secondary_path_buffer),
                    (4, self.camera_buffers[slot]),
                    (5, self.secondary_path_buffer),
                )
            )
        for slot, descriptor_set in enumerate(self.relax_prepare_sets):
            self._write_storage_set(
                descriptor_set, (
                    (0, self.path_buffer), (1, self.secondary_path_buffer),
                    (9, self.camera_buffers[slot]),
                    (10, self.camera_buffers[1 - slot]),
                    (11, self.core.scene_previous_vertex_buffer),
                )
            )
        for index, descriptor_set in enumerate(self.relax_temporal_sets):
            slot = index // 2
            info = self._buffer_info(
                self.relax_temporal_constant_buffers[slot]
            )
            write = vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set, dstBinding=12,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
                pBufferInfo=[info],
            )
            vk.vkUpdateDescriptorSets(
                self.core.device, 1, [write], 0, None
            )
    def bind_output_image(
        self, slot, hdr_view, position_view, normal_view, material_view,
        output_view,
    ):
        """Bind frame-local reconstruction inputs and full-resolution output."""
        if not 0 <= slot < len(self.wavefront_image_sets):
            raise ValueError("wavefront output image slot is out of range")
        hdr_info = vk.VkDescriptorImageInfo(
            imageView=hdr_view, imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL
        )
        write = vk.VkWriteDescriptorSet(
            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=self.wavefront_image_sets[slot], dstBinding=1,
            descriptorCount=1, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            pImageInfo=[hdr_info],
        )
        vk.vkUpdateDescriptorSets(self.core.device, 1, [write], 0, None)
        current = self.core.window_frames[slot]
        previous = self.core.window_frames[1 - slot]
        image_views = (
            hdr_view, position_view, normal_view,
            previous["wavefront_history_color_view"],
            previous["wavefront_position_view"],
            previous["wavefront_normal_view"],
            current["wavefront_history_color_view"],
        )
        image_infos = [vk.VkDescriptorImageInfo(
            imageView=view, imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL
        ) for view in image_views]
        writes = [vk.VkWriteDescriptorSet(
            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=self.reconstruct_sets[slot], dstBinding=binding,
            descriptorCount=1, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            pImageInfo=[info],
        ) for binding, info in enumerate(image_infos)]
        output_infos = [vk.VkDescriptorImageInfo(
            imageView=output_view, imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL
        ) for _ in range(8)]
        writes.append(vk.VkWriteDescriptorSet(
            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=self.reconstruct_sets[slot], dstBinding=7,
            descriptorCount=8,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            pImageInfo=output_infos,
        ))
        camera_infos = [self._buffer_info(self.camera_buffers[index])
                        for index in (1 - slot, slot)]
        writes.extend(vk.VkWriteDescriptorSet(
            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=self.reconstruct_sets[slot], dstBinding=8 + index,
            descriptorCount=1, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[info],
        ) for index, info in enumerate(camera_infos))
        material_info = vk.VkDescriptorImageInfo(
            imageView=material_view, imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL
        )
        writes.append(vk.VkWriteDescriptorSet(
            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=self.reconstruct_sets[slot], dstBinding=10,
            descriptorCount=1,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            pImageInfo=[material_info],
        ))
        vk.vkUpdateDescriptorSets(self.core.device, len(writes), writes, 0, None)

        if self.relax_prepare_sets:
            relax_views = (
                normal_view, material_view,
                current["wavefront_relax_diffuse_view"],
                current["wavefront_relax_specular_view"],
                current["wavefront_relax_normal_roughness_view"],
                current["wavefront_relax_view_z_view"],
                current["wavefront_relax_motion_view"],
                current["wavefront_relax_identity_view"],
            )
            relax_infos = [vk.VkDescriptorImageInfo(
                imageView=view, imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL
            ) for view in relax_views]
            relax_bindings = (2, 3, 4, 5, 6, 7, 8, 12)
            relax_writes = [vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=self.relax_prepare_sets[slot], dstBinding=binding,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                pImageInfo=[info],
            ) for binding, info in zip(relax_bindings, relax_infos)]
            vk.vkUpdateDescriptorSets(
                self.core.device, len(relax_writes), relax_writes, 0, None
            )

        if self.relax_temporal_sets:
            for lobe, name in enumerate(("diffuse", "specular")):
                descriptor_set = self.relax_temporal_sets[slot * 2 + lobe]
                temporal_views = (
                    current[f"wavefront_relax_{name}_view"],
                    current["wavefront_relax_normal_roughness_view"],
                    current["wavefront_relax_view_z_view"],
                    current["wavefront_relax_motion_view"],
                    material_view,
                    previous[f"wavefront_relax_temporal_{name}_view"],
                    previous["wavefront_relax_normal_roughness_view"],
                    previous["wavefront_relax_view_z_view"],
                    previous["wavefront_material_view"],
                    previous[f"wavefront_relax_{name}_history_view"],
                    current[f"wavefront_relax_temporal_{name}_view"],
                    current[f"wavefront_relax_{name}_history_view"],
                )
                infos = [vk.VkDescriptorImageInfo(
                    imageView=view, imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL
                ) for view in temporal_views]
                writes = [vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set, dstBinding=binding,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    pImageInfo=[info],
                ) for binding, info in enumerate(infos)]
                identity_infos = [vk.VkDescriptorImageInfo(
                    imageView=frame["wavefront_relax_identity_view"],
                    imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
                ) for frame in (current, previous)]
                writes.extend(vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=descriptor_set, dstBinding=13 + index,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    pImageInfo=[info],
                ) for index, info in enumerate(identity_infos))
                vk.vkUpdateDescriptorSets(
                    self.core.device, len(writes), writes, 0, None
                )

        if self.relax_atrous_sets:
            for lobe, name in enumerate(("diffuse", "specular")):
                for direction, (source, target) in enumerate((
                    (current[f"wavefront_relax_temporal_{name}_view"],
                     current[f"wavefront_relax_atrous_{name}_view"]),
                    (current[f"wavefront_relax_atrous_{name}_view"],
                     current[f"wavefront_relax_{name}_view"]),
                    (current[f"wavefront_relax_{name}_view"],
                     current[f"wavefront_relax_atrous_{name}_view"]),
                )):
                    descriptor_set = self.relax_atrous_sets[
                        slot * 6 + lobe * 3 + direction
                    ]
                    views = (
                        source,
                        current["wavefront_relax_normal_roughness_view"],
                        current["wavefront_relax_view_z_view"], material_view,
                        target,
                    )
                    infos = [vk.VkDescriptorImageInfo(
                        imageView=view, imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL
                    ) for view in views]
                    writes = [vk.VkWriteDescriptorSet(
                        sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        dstSet=descriptor_set, dstBinding=binding,
                        descriptorCount=1,
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                        pImageInfo=[info],
                    ) for binding, info in enumerate(infos)]
                    vk.vkUpdateDescriptorSets(
                        self.core.device, len(writes), writes, 0, None
                    )

        if self.relax_compose_sets:
            for side, prefix in enumerate((
                "wavefront_relax_atrous", "wavefront_relax",
            )):
                views = (
                    current[f"{prefix}_diffuse_view"],
                    current[f"{prefix}_specular_view"],
                    current["wavefront_relax_view_z_view"], hdr_view,
                )
                infos = [vk.VkDescriptorImageInfo(
                    imageView=view, imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL
                ) for view in views]
                writes = [vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=self.relax_compose_sets[slot * 2 + side],
                    dstBinding=binding, descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    pImageInfo=[info],
                ) for binding, info in enumerate(infos)]
                vk.vkUpdateDescriptorSets(
                    self.core.device, len(writes), writes, 0, None
                )

        primary_infos = [vk.VkDescriptorImageInfo(
            imageView=view, imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL
        ) for view in (position_view, normal_view)]
        primary_writes = [vk.VkWriteDescriptorSet(
            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=self.primary_sets[slot], dstBinding=8 + index,
            descriptorCount=1, descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            pImageInfo=[info],
        ) for index, info in enumerate(primary_infos)]
        reservoir_infos = [
            self._buffer_info(frame["wavefront_reservoir_buffer"])
            for frame in (current, previous)
        ]
        primary_writes.extend(vk.VkWriteDescriptorSet(
            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=self.primary_sets[slot], dstBinding=16 + index,
            descriptorCount=1,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[info],
        ) for index, info in enumerate(reservoir_infos))
        previous_camera_info = self._buffer_info(self.camera_buffers[1 - slot])
        primary_writes.append(vk.VkWriteDescriptorSet(
            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=self.primary_sets[slot], dstBinding=18,
            descriptorCount=1,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[previous_camera_info],
        ))
        previous_gbuffer_infos = [vk.VkDescriptorImageInfo(
            imageView=view, imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL
        ) for view in (
            previous["wavefront_position_view"],
            previous["wavefront_normal_view"],
        )]
        primary_writes.extend(vk.VkWriteDescriptorSet(
            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=self.primary_sets[slot], dstBinding=19 + index,
            descriptorCount=1,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            pImageInfo=[info],
        ) for index, info in enumerate(previous_gbuffer_infos))
        vk.vkUpdateDescriptorSets(
            self.core.device, len(primary_writes), primary_writes, 0, None
        )
        material_infos = [vk.VkDescriptorImageInfo(
            imageView=view, imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL
        ) for view in (
            material_view, previous["wavefront_material_view"],
        )]
        material_writes = [vk.VkWriteDescriptorSet(
            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=self.primary_sets[slot], dstBinding=21 + index,
            descriptorCount=1,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            pImageInfo=[info],
        ) for index, info in enumerate(material_infos)]
        vk.vkUpdateDescriptorSets(
            self.core.device, len(material_writes), material_writes, 0, None
        )

    def bind_indirect_reuse_buffer(self, slot, buffer, seed_buffer):
        """Bind one frame-local compact indirect-reservoir buffer."""
        if not self.indirect_reuse_clear_sets:
            return
        if not 0 <= slot < len(self.indirect_reuse_clear_sets):
            raise ValueError("indirect reuse buffer slot is out of range")
        self._write_storage_set(
            self.indirect_reuse_clear_sets[slot], ((0, buffer),)
        )
        if self.indirect_reuse_candidate_sets:
            previous = self.core.window_frames[1 - slot]
            self._write_storage_set(
                self.indirect_reuse_candidate_sets[slot],
                (
                    (0, buffer),
                    (4, self.camera_buffers[slot]),
                    (5, previous["wavefront_indirect_reservoir_buffer"]),
                    (8, self.camera_buffers[1 - slot]),
                    (11, self.indirect_reuse_counter_buffers[slot]),
                ),
            )
        if self.indirect_reuse_debug_sets:
            self._write_storage_set(
                self.indirect_reuse_debug_sets[slot], (
                    (0, buffer), (2, seed_buffer),
                )
            )
        if self.wavefront_image_sets:
            self._write_storage_set(
                self.wavefront_image_sets[slot], (
                    (3, buffer), (5, seed_buffer),
                )
            )

    def bind_indirect_reuse_inputs(
        self, slot, hdr_view, position_view, normal_view, material_view
    ):
        """Bind frame-local screen-space candidate source images."""
        if not self.indirect_reuse_candidate_sets:
            return
        previous = self.core.window_frames[1 - slot]
        formats = (
            (1, hdr_view), (2, position_view), (3, normal_view),
            (6, previous["wavefront_position_view"]),
            (7, previous["wavefront_normal_view"]),
            (9, material_view),
            (10, previous["wavefront_material_view"]),
        )
        infos = [vk.VkDescriptorImageInfo(
            imageView=view, imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL
        ) for _binding, view in formats]
        writes = [vk.VkWriteDescriptorSet(
            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=self.indirect_reuse_candidate_sets[slot],
            dstBinding=binding, descriptorCount=1,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            pImageInfo=[info],
        ) for (binding, _view), info in zip(formats, infos)]
        vk.vkUpdateDescriptorSets(
            self.core.device, len(writes), writes, 0, None
        )
        if self.indirect_reuse_debug_sets:
            debug_infos = [vk.VkDescriptorImageInfo(
                imageView=view, imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL
            ) for view in (hdr_view, material_view)]
            debug_writes = [vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=self.indirect_reuse_debug_sets[slot],
                dstBinding=binding, descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                pImageInfo=[info],
            ) for binding, info in zip((1, 3), debug_infos)]
            vk.vkUpdateDescriptorSets(
                self.core.device, len(debug_writes), debug_writes, 0, None,
            )

    def record_indirect_reuse_clear(self, command, slot, reservoir_count):
        """Zero a newly allocated compact reservoir buffer exactly once."""
        if not self.indirect_reuse_clear_pipeline:
            return
        vk.vkCmdBindPipeline(
            command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.indirect_reuse_clear_pipeline,
        )
        vk.vkCmdBindDescriptorSets(
            command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.indirect_reuse_clear_pipeline_layout, 0, 1,
            [self.indirect_reuse_clear_sets[slot]], 0, None,
        )
        constants = bytearray(struct.pack("I", reservoir_count))
        vk.vkCmdPushConstants(
            command, self.indirect_reuse_clear_pipeline_layout,
            vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, len(constants),
            vk.ffi.from_buffer(constants),
        )
        vk.vkCmdDispatch(command, (reservoir_count + 255) // 256, 1, 1)

    def record_indirect_reuse_candidates(
        self, command, slot, source_width, source_height,
        reservoir_width, reservoir_height, history_valid, frame_index,
        history_limit,
    ):
        """Generate independent screen-space indirect candidate reservoirs."""
        if not self.indirect_reuse_candidate_pipeline:
            return
        frame = self.core.window_frames[slot]
        previous = self.core.window_frames[1 - slot]
        image_barriers = [self.core._image_barrier(
            owner[name], vk.VK_IMAGE_LAYOUT_GENERAL,
            vk.VK_IMAGE_LAYOUT_GENERAL, vk.VK_ACCESS_SHADER_WRITE_BIT,
            vk.VK_ACCESS_SHADER_READ_BIT,
        ) for owner, name in (
            (frame, "wavefront_hdr_image"),
            (frame, "wavefront_position_image"),
            (frame, "wavefront_normal_image"),
            (frame, "wavefront_material_image"),
            (previous, "wavefront_position_image"),
            (previous, "wavefront_normal_image"),
            (previous, "wavefront_material_image"),
        )]
        reservoir_barriers = [self._buffer_barrier(
            owner["wavefront_indirect_reservoir_buffer"],
            vk.VK_ACCESS_SHADER_WRITE_BIT,
            destination_access,
        ) for owner, destination_access in (
            (frame, vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT),
            (previous, vk.VK_ACCESS_SHADER_READ_BIT),
        )]
        vk.vkCmdPipelineBarrier(
            command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
            0, None, len(reservoir_barriers), reservoir_barriers,
            len(image_barriers), image_barriers,
        )
        vk.vkCmdBindPipeline(
            command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.indirect_reuse_candidate_pipeline,
        )
        vk.vkCmdBindDescriptorSets(
            command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.indirect_reuse_candidate_pipeline_layout, 0, 1,
            [self.indirect_reuse_candidate_sets[slot]], 0, None,
        )
        counter_buffer = self.indirect_reuse_counter_buffers[slot]
        if self.core.config.wavefront_indirect_reuse_profiling:
            vk.vkCmdFillBuffer(
                command, counter_buffer.buffer, 0, counter_buffer.size, 0
            )
            counter_ready = self._buffer_barrier(
                counter_buffer, vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
            )
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                0, None, 1, [counter_ready], 0, None,
            )

        constants = bytearray(struct.pack(
            "9I", source_width, source_height,
            reservoir_width, reservoir_height,
            int(history_valid), frame_index,
            int(self.core.config.wavefront_indirect_reuse_spatial),
            int(self.core.config.wavefront_indirect_reuse_profiling),
            history_limit,
        ))
        vk.vkCmdPushConstants(
            command, self.indirect_reuse_candidate_pipeline_layout,
            vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, len(constants),
            vk.ffi.from_buffer(constants),
        )
        vk.vkCmdDispatch(
            command, (reservoir_width + 7) // 8,
            (reservoir_height + 7) // 8, 1,
        )
        if self.core.config.wavefront_indirect_reuse_profiling:
            host_ready = self._buffer_barrier(
                counter_buffer, vk.VK_ACCESS_SHADER_WRITE_BIT,
                vk.VK_ACCESS_HOST_READ_BIT,
            )
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_HOST_BIT, 0,
                0, None, 1, [host_ready], 0, None,
            )

    def record_indirect_reuse_debug(
        self, command, slot, output_width, output_height,
        reservoir_width, reservoir_height,
    ):
        """Visualize compact indirect reservoirs without affecting lighting."""
        if not self.indirect_reuse_debug_pipeline:
            return
        frame = self.core.window_frames[slot]
        barriers = [self._buffer_barrier(
            buffer, vk.VK_ACCESS_SHADER_WRITE_BIT,
            vk.VK_ACCESS_SHADER_READ_BIT,
        ) for buffer in (
            frame["wavefront_indirect_reservoir_buffer"],
            frame["wavefront_indirect_seed_buffer"],
        )]
        image_barrier = self.core._image_barrier(
            frame["wavefront_hdr_image"], vk.VK_IMAGE_LAYOUT_GENERAL,
            vk.VK_IMAGE_LAYOUT_GENERAL, vk.VK_ACCESS_SHADER_READ_BIT,
            vk.VK_ACCESS_SHADER_WRITE_BIT,
        )
        vk.vkCmdPipelineBarrier(
            command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
            0, None, len(barriers), barriers, 1, [image_barrier],
        )
        vk.vkCmdBindPipeline(
            command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.indirect_reuse_debug_pipeline,
        )
        vk.vkCmdBindDescriptorSets(
            command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.indirect_reuse_debug_pipeline_layout, 0, 1,
            [self.indirect_reuse_debug_sets[slot]], 0, None,
        )
        modes = {
            "radiance": 1, "history": 2, "validity": 3, "acceptance": 4,
        }
        debug_view = self.core.config.wavefront_indirect_reuse_debug_view
        mode = (modes[debug_view] if debug_view != "off" else 5)
        constants = bytearray(struct.pack(
            "6If", output_width, output_height,
            reservoir_width, reservoir_height,
            mode,
            self.core.config.wavefront_indirect_reuse_history_limit,
            self.core.config.wavefront_indirect_reuse_apply_strength,
        ))
        vk.vkCmdPushConstants(
            command, self.indirect_reuse_debug_pipeline_layout,
            vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, len(constants),
            vk.ffi.from_buffer(constants),
        )
        vk.vkCmdDispatch(
            command, (output_width + 7) // 8,
            (output_height + 7) // 8, 1,
        )

    def bind_reconstruction_outputs(self, output_views):
        """Bind an immutable swapchain-image array to both frame slots."""
        if not output_views or len(output_views) > 8:
            raise ValueError("direct swapchain storage supports 1-8 images")
        padded = list(output_views) + [output_views[-1]] * (8 - len(output_views))
        infos = [vk.VkDescriptorImageInfo(
            imageView=view, imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL
        ) for view in padded]
        writes = [vk.VkWriteDescriptorSet(
            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=descriptor_set, dstBinding=7,
            descriptorCount=8,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
            pImageInfo=infos,
        ) for descriptor_set in self.reconstruct_sets]
        vk.vkUpdateDescriptorSets(
            self.core.device, len(writes), writes, 0, None
        )

    def update_camera(
        self, slot, camera_vectors, frame_sequence=0, output_image_index=0,
        projection=0,
    ):
        """Upload one frame slot's camera basis to coherent host memory."""
        values = np.concatenate([
            np.append(
                np.asarray(vector, dtype=np.float32),
                (float(frame_sequence) if index == 0 else
                 float(output_image_index) if index == 1 else
                 float(projection) if index == 3 else 0.0),
            )
            for index, vector in enumerate(camera_vectors)
        ]).astype(np.float32, copy=False)
        memory = self.camera_buffers[slot].memory
        mapped = vk.vkMapMemory(self.core.device, memory, 0, values.nbytes, 0)
        mapped[:] = values.tobytes()
        vk.vkUnmapMemory(self.core.device, memory)

    def record_path_to_hdr(
        self, command, slot, path_count, image_width, image_height,
        sample_index=0, sample_count=1,
    ):
        """Resolve one completed path tile into the internal HDR image."""
        paths_ready = [self._buffer_barrier(
            buffer, vk.VK_ACCESS_SHADER_WRITE_BIT,
            vk.VK_ACCESS_SHADER_READ_BIT,
        ) for buffer in (self.path_buffer, self.secondary_path_buffer)]
        hdr_ready = self.core._image_barrier(
            self.core.window_frames[slot]["wavefront_hdr_image"],
            vk.VK_IMAGE_LAYOUT_GENERAL, vk.VK_IMAGE_LAYOUT_GENERAL,
            vk.VK_ACCESS_SHADER_WRITE_BIT,
            vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
        )
        vk.vkCmdPipelineBarrier(
            command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
            0, None, len(paths_ready), paths_ready, 1, [hdr_ready],
        )
        vk.vkCmdBindPipeline(
            command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.wavefront_image_pipeline,
        )
        vk.vkCmdBindDescriptorSets(
            command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.wavefront_image_pipeline_layout, 0, 1,
            [self.wavefront_image_sets[slot]], 0, None,
        )
        reservoir_extent = self.core.window_frames[slot].get(
            "wavefront_indirect_reservoir_extent", (1, 1)
        ) or (1, 1)
        constants = bytearray(struct.pack(
            "8I", path_count, image_width, image_height,
            sample_index, sample_count,
            int(
                self.core.config.wavefront_indirect_reuse_candidates
                or self._denoiser_signals_active()
            ),
            reservoir_extent[0], reservoir_extent[1],
        ))
        vk.vkCmdPushConstants(
            command, self.wavefront_image_pipeline_layout,
            vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, len(constants),
            vk.ffi.from_buffer(constants),
        )
        vk.vkCmdDispatch(command, (path_count + 63) // 64, 1, 1)

    def record_relax_prepare(
        self, command, slot, path_count, image_width, image_height,
    ):
        """Scatter a completed path tile into full-frame ReLAX signal images."""
        if not self.relax_prepare_pipeline:
            return
        current = self.core.window_frames[slot]
        buffer_barriers = [self._buffer_barrier(
            buffer, vk.VK_ACCESS_SHADER_WRITE_BIT,
            vk.VK_ACCESS_SHADER_READ_BIT,
        ) for buffer in (self.path_buffer, self.secondary_path_buffer)]
        image_barriers = [self.core._image_barrier(
            current[name], vk.VK_IMAGE_LAYOUT_GENERAL,
            vk.VK_IMAGE_LAYOUT_GENERAL,
            vk.VK_ACCESS_SHADER_WRITE_BIT,
            vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
        ) for name in (
            "wavefront_normal_image", "wavefront_material_image",
            "wavefront_relax_diffuse_image",
            "wavefront_relax_specular_image",
            "wavefront_relax_normal_roughness_image",
            "wavefront_relax_view_z_image",
            "wavefront_relax_motion_image",
            "wavefront_relax_identity_image",
        )]
        vk.vkCmdPipelineBarrier(
            command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
            0, None, len(buffer_barriers), buffer_barriers,
            len(image_barriers), image_barriers,
        )
        vk.vkCmdBindPipeline(
            command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.relax_prepare_pipeline,
        )
        vk.vkCmdBindDescriptorSets(
            command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.relax_prepare_pipeline_layout, 0, 1,
            [self.relax_prepare_sets[slot]], 0, None,
        )
        constants = bytearray(struct.pack(
            "4I", image_width, image_height, path_count, 0
        ))
        vk.vkCmdPushConstants(
            command, self.relax_prepare_pipeline_layout,
            vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, len(constants),
            vk.ffi.from_buffer(constants),
        )
        vk.vkCmdDispatch(command, (path_count + 63) // 64, 1, 1)

    def update_relax_temporal_constants(
        self, slot, width, height, history_valid, *, history_limit=32,
        normal_threshold=0.8, depth_threshold=0.02, clamp_sigma=2.5,
        reactive_sigma=0.0,
    ):
        """Upload temporal history and rejection policy for one frame slot."""
        if not self.relax_temporal_constant_buffers:
            return
        values = np.asarray((
            float(width), float(height), float(history_limit),
            1.0 if history_valid else 0.0,
            float(normal_threshold), float(depth_threshold),
            float(clamp_sigma), float(reactive_sigma),
        ), dtype=np.float32)
        memory = self.relax_temporal_constant_buffers[slot].memory
        mapped = vk.vkMapMemory(
            self.core.device, memory, 0, values.nbytes, 0
        )
        mapped[:] = values.tobytes()
        vk.vkUnmapMemory(self.core.device, memory)

    def record_relax_temporal(self, command, slot, width, height):
        """Temporally accumulate prepared diffuse and specular signals."""
        if not self.relax_temporal_pipeline:
            return
        current = self.core.window_frames[slot]
        previous = self.core.window_frames[1 - slot]
        names = (
            "wavefront_relax_diffuse_image",
            "wavefront_relax_specular_image",
            "wavefront_relax_normal_roughness_image",
            "wavefront_relax_view_z_image",
            "wavefront_relax_motion_image",
            "wavefront_relax_identity_image",
            "wavefront_material_image",
            "wavefront_relax_temporal_diffuse_image",
            "wavefront_relax_temporal_specular_image",
            "wavefront_relax_diffuse_history_image",
            "wavefront_relax_specular_history_image",
        )
        barriers = [self.core._image_barrier(
            current[name], vk.VK_IMAGE_LAYOUT_GENERAL,
            vk.VK_IMAGE_LAYOUT_GENERAL, vk.VK_ACCESS_SHADER_WRITE_BIT,
            vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
        ) for name in names]
        barriers.extend(self.core._image_barrier(
            previous[name], vk.VK_IMAGE_LAYOUT_GENERAL,
            vk.VK_IMAGE_LAYOUT_GENERAL, vk.VK_ACCESS_SHADER_WRITE_BIT,
            vk.VK_ACCESS_SHADER_READ_BIT,
        ) for name in (
            "wavefront_relax_normal_roughness_image",
            "wavefront_relax_view_z_image", "wavefront_relax_identity_image",
            "wavefront_material_image",
            "wavefront_relax_temporal_diffuse_image",
            "wavefront_relax_temporal_specular_image",
            "wavefront_relax_diffuse_history_image",
            "wavefront_relax_specular_history_image",
        ))
        vk.vkCmdPipelineBarrier(
            command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
            0, None, 0, None, len(barriers), barriers,
        )
        vk.vkCmdBindPipeline(
            command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.relax_temporal_pipeline,
        )
        for lobe in range(2):
            vk.vkCmdBindDescriptorSets(
                command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                self.relax_temporal_pipeline_layout, 0, 1,
                [self.relax_temporal_sets[slot * 2 + lobe]], 0, None,
            )
            vk.vkCmdDispatch(
                command, (width + 7) // 8, (height + 7) // 8, 1
            )
        atrous_barriers = [self.core._image_barrier(
            current[name], vk.VK_IMAGE_LAYOUT_GENERAL,
            vk.VK_IMAGE_LAYOUT_GENERAL, vk.VK_ACCESS_SHADER_WRITE_BIT,
            vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
        ) for name in (
            "wavefront_relax_temporal_diffuse_image",
            "wavefront_relax_temporal_specular_image",
            "wavefront_relax_normal_roughness_image",
            "wavefront_relax_view_z_image", "wavefront_material_image",
            "wavefront_relax_atrous_diffuse_image",
            "wavefront_relax_atrous_specular_image",
        )]
        vk.vkCmdPipelineBarrier(
            command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
            0, None, 0, None, len(atrous_barriers), atrous_barriers,
        )
        vk.vkCmdBindPipeline(
            command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.relax_atrous_pipeline,
        )
        iteration_count = self.core.config.denoiser_iterations
        for iteration in range(iteration_count):
            direction = 0 if iteration == 0 else (1 if iteration & 1 else 2)
            for lobe in range(2):
                constants = bytearray(struct.pack(
                    "8f", float(width), float(height),
                    float(1 << iteration), 0.0, 32.0, 0.02, 4.0,
                    float(lobe == 1 and iteration == 0),
                ))
                vk.vkCmdPushConstants(
                    command, self.relax_atrous_pipeline_layout,
                    vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, len(constants),
                    vk.ffi.from_buffer(constants),
                )
                vk.vkCmdBindDescriptorSets(
                    command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                    self.relax_atrous_pipeline_layout, 0, 1,
                    [self.relax_atrous_sets[
                        slot * 6 + lobe * 3 + direction
                    ]], 0, None,
                )
                vk.vkCmdDispatch(
                    command, (width + 7) // 8, (height + 7) // 8, 1
                )
            if iteration + 1 < iteration_count:
                vk.vkCmdPipelineBarrier(
                    command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                    0, None, 0, None, len(atrous_barriers), atrous_barriers,
                )
        final_side = 0 if iteration_count & 1 else 1
        final_prefix = (
            "wavefront_relax_atrous" if final_side == 0
            else "wavefront_relax"
        )
        compose_barriers = [self.core._image_barrier(
            current[f"{final_prefix}_{name}_image"],
            vk.VK_IMAGE_LAYOUT_GENERAL, vk.VK_IMAGE_LAYOUT_GENERAL,
            vk.VK_ACCESS_SHADER_WRITE_BIT, vk.VK_ACCESS_SHADER_READ_BIT,
        ) for name in ("diffuse", "specular")]
        compose_barriers.append(self.core._image_barrier(
            current["wavefront_relax_view_z_image"],
            vk.VK_IMAGE_LAYOUT_GENERAL, vk.VK_IMAGE_LAYOUT_GENERAL,
            vk.VK_ACCESS_SHADER_WRITE_BIT, vk.VK_ACCESS_SHADER_READ_BIT,
        ))
        compose_barriers.append(self.core._image_barrier(
            current["wavefront_hdr_image"], vk.VK_IMAGE_LAYOUT_GENERAL,
            vk.VK_IMAGE_LAYOUT_GENERAL, vk.VK_ACCESS_SHADER_READ_BIT,
            vk.VK_ACCESS_SHADER_WRITE_BIT,
        ))
        vk.vkCmdPipelineBarrier(
            command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
            0, None, 0, None, len(compose_barriers), compose_barriers,
        )
        vk.vkCmdBindPipeline(
            command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.relax_compose_pipeline,
        )
        vk.vkCmdBindDescriptorSets(
            command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.relax_compose_pipeline_layout, 0, 1,
            [self.relax_compose_sets[slot * 2 + final_side]], 0, None,
        )
        constants = bytearray(struct.pack(
            "4f", float(width), float(height), 0.0, 0.0
        ))
        vk.vkCmdPushConstants(
            command, self.relax_compose_pipeline_layout,
            vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, len(constants),
            vk.ffi.from_buffer(constants),
        )
        vk.vkCmdDispatch(
            command, (width + 7) // 8, (height + 7) // 8, 1
        )

    def record_reconstruction(
        self, command, slot, source_width, source_height,
        output_width, output_height, history_valid=False, *, scene, camera,
    ):
        """Reconstruct internal HDR, optionally reprojecting valid history."""
        current = self.core.window_frames[slot]
        previous = self.core.window_frames[1 - slot]
        barriers = [self.core._image_barrier(
            current[name], vk.VK_IMAGE_LAYOUT_GENERAL, vk.VK_IMAGE_LAYOUT_GENERAL,
            vk.VK_ACCESS_SHADER_WRITE_BIT,
            vk.VK_ACCESS_SHADER_READ_BIT,
        ) for name in (
            "wavefront_hdr_image", "wavefront_position_image",
            "wavefront_normal_image", "wavefront_material_image",
        )]
        barriers.extend(self.core._image_barrier(
            frame[name], vk.VK_IMAGE_LAYOUT_GENERAL, vk.VK_IMAGE_LAYOUT_GENERAL,
            vk.VK_ACCESS_SHADER_WRITE_BIT,
            vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
        ) for frame, name in (
            (previous, "wavefront_history_color_image"),
            (current, "wavefront_history_color_image"),
            (previous, "wavefront_position_image"),
            (previous, "wavefront_normal_image"),
        ))
        vk.vkCmdPipelineBarrier(
            command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
            0, None, 0, None, len(barriers), barriers,
        )
        vk.vkCmdBindPipeline(
            command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            (
                self.reconstruct_bgra_pipeline
                if self.core.swapchain_bgra_storage
                else self.reconstruct_pipeline
            ),
        )
        vk.vkCmdBindDescriptorSets(
            command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
            self.reconstruct_pipeline_layout, 0, 1,
            [self.reconstruct_sets[slot]], 0, None,
        )
        effect_headers = []
        effect_colors = []
        effect_rects = []
        for triangle_range, effect in self.core.object_effect_bindings:
            kind = (
                1 if isinstance(effect, Outline)
                else 2 if isinstance(effect, Tint)
                else 3 if isinstance(effect, EmissiveHighlight)
                else 4 if isinstance(effect, Isolation)
                else 5 if isinstance(effect, BoundingBox)
                else 6 if isinstance(effect, XRay)
                else 0
            )
            width = effect.width if isinstance(
                effect, (Outline, BoundingBox, XRay)
            ) else 1
            strength = (
                effect.strength
                if isinstance(effect, (Tint, EmissiveHighlight, XRay))
                else effect.dimming if isinstance(effect, Isolation)
                else 0.0
            )
            effect_headers.append((kind, width, strength, 0))
            effect_colors.extend((*getattr(effect, "color", (0.0,) * 3), 0.0))
            effect_rects.extend(_effect_screen_rect(
                scene, triangle_range, camera, (output_width, output_height)
            ))
        effect_headers.extend(
            [(0, 1, 0.0, 0)] * (4 - len(effect_headers))
        )
        effect_colors.extend(
            (0.0, 0.0, 0.0, 0.0) * (4 - len(self.core.object_effect_bindings))
        )
        effect_rects.extend(
            (0.0, 0.0, 0.0, 0.0) * (4 - len(self.core.object_effect_bindings))
        )
        constants = bytearray(struct.pack(
            RECONSTRUCT_BASE_FORMAT, self.core.config.wavefront_exposure,
            source_width, source_height,
            int(
                self.core.config.wavefront_temporal_reconstruction
                or self.core.config.stationary_accumulation
            ),
            self.core.config.wavefront_temporal_weight,
            int(history_valid),
            int(self.core.config.wavefront_diffuse_filter),
            self.core.config.wavefront_diffuse_filter_strength,
            int(self.core.config.wavefront_temporal_variance_confidence),
            self.core.config.wavefront_temporal_variance_strength,
            int(self.core.config.wavefront_temporal_material_confidence),
            self.core.config.wavefront_temporal_transmission_history_scale,
            int(self.core.config.wavefront_temporal_reprojection_search),
            int(self.core.config.wavefront_temporal_outlier_confidence),
            self.core.config.wavefront_temporal_outlier_strength,
        ))
        constants.extend(struct.pack("I", 0))
        for header in effect_headers:
            constants.extend(struct.pack("IIfI", *header))
        constants.extend(struct.pack("16f", *effect_colors))
        constants.extend(struct.pack("16f", *effect_rects))
        if len(constants) != RECONSTRUCT_PUSH_SIZE:
            raise AssertionError("reconstruction push-constant ABI mismatch")
        vk.vkCmdPushConstants(
            command, self.reconstruct_pipeline_layout,
            vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, len(constants),
            vk.ffi.from_buffer(constants),
        )
        vk.vkCmdDispatch(
            command, (output_width + 7) // 8,
            (output_height + 7) // 8, 1,
        )

    def _buffer_info(self, buffer):
        return vk.VkDescriptorBufferInfo(
            buffer=buffer.buffer, offset=0, range=buffer.size
        )

    def _write_storage_set(self, descriptor_set, bindings):
        infos = [self._buffer_info(buffer) for _, buffer in bindings]
        writes = [vk.VkWriteDescriptorSet(
            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            dstSet=descriptor_set, dstBinding=binding, descriptorCount=1,
            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            pBufferInfo=[info],
        ) for (binding, _), info in zip(bindings, infos)]
        vk.vkUpdateDescriptorSets(self.core.device, len(writes), writes, 0, None)

    def _write_sampled_textures(self, descriptor_set, binding):
        textures = list(self.core.scene_sampled_textures)
        if not textures:
            return
        textures.extend(
            [textures[0]] * (MAX_NATIVE_TEXTURES * 2 - len(textures))
        )
        infos = [vk.VkDescriptorImageInfo(
            sampler=texture.sampler,
            imageView=texture.view,
            imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        ) for texture in textures]
        vk.vkUpdateDescriptorSets(
            self.core.device, 1, [vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set, dstBinding=binding,
                descriptorCount=len(infos),
                descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                pImageInfo=infos,
            )], 0, None,
        )

    def _write_sampled_volumes(self, descriptor_set, binding):
        textures = list(self.core.scene_sampled_volumes)
        if not textures:
            return
        textures.extend([textures[0]] * (MAX_NATIVE_VOLUMES - len(textures)))
        infos = [vk.VkDescriptorImageInfo(
            sampler=texture.sampler, imageView=texture.view,
            imageLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        ) for texture in textures]
        vk.vkUpdateDescriptorSets(
            self.core.device, 1, [vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set, dstBinding=binding,
                descriptorCount=len(infos),
                descriptorType=vk.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                pImageInfo=infos,
            )], 0, None,
        )

    def bind_scene(self):
        scene = self.core.scene_resources.scene
        self._ensure_medium_buffer(scene)
        scene_key = (
            id(self.core.scene_tlas),
            id(self.core.scene_vertex_buffer),
            id(self.core.scene_previous_vertex_buffer),
            id(self.core.scene_material_buffer),
            id(self.core.scene_attribute_buffer),
            id(self.core.scene_custom_attribute_buffer),
            id(self.core.scene_light_buffer),
            id(self.core.scene_area_light_buffer),
            id(self.core.scene_texture_buffer),
            id(self.core.scene_texture_binding_buffer),
            id(self.core.scene_volume_header_buffer),
            id(self.core.scene_volume_scalar_buffer),
            id(self.core.scene_volume_transfer_buffer),
            id(self.core.scene_triangle_volume_buffer),
            tuple(id(item) for item in self.core.scene_sampled_textures),
            tuple(id(item) for item in self.core.scene_sampled_volumes),
        )
        if scene_key == self._bound_scene_key:
            return
        as_info = vk.VkWriteDescriptorSetAccelerationStructureKHR(
            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
            accelerationStructureCount=1,
            pAccelerationStructures=[self.core.scene_tlas.handle],
        )
        for descriptor_set in self.primary_sets:
            self._write_storage_set(descriptor_set, (
                (6, self.medium_buffer),
            ))
            primary_buffers = (
                (2, self.core.scene_material_buffer),
                (3, self.core.scene_vertex_buffer),
                (4, self.core.scene_attribute_buffer),
                (10, self.core.scene_light_buffer),
                (11, self.core.scene_area_light_buffer),
                (12, self.core.scene_texture_buffer),
                (13, self.core.scene_texture_binding_buffer),
                (25, self.core.scene_volume_header_buffer),
                (26, self.core.scene_volume_scalar_buffer),
                (27, self.core.scene_volume_transfer_buffer),
                (28, self.core.scene_triangle_volume_buffer),
            )
            if self.core.scene_custom_attribute_buffer is not None:
                primary_buffers += ((
                    24, self.core.scene_custom_attribute_buffer,
                ),)
            primary_infos = [
                self._buffer_info(buffer) for _binding, buffer in primary_buffers
            ]
            primary_writes = [vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, pNext=as_info,
                dstSet=descriptor_set, dstBinding=0, descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
            )]
            primary_writes.extend(vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set, dstBinding=binding, descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[info],
            ) for (binding, _buffer), info in zip(primary_buffers, primary_infos))
            vk.vkUpdateDescriptorSets(
                self.core.device, len(primary_writes), primary_writes, 0, None
            )
            if self.core.native_textures_enabled:
                self._write_sampled_textures(descriptor_set, 14)
            self._write_sampled_volumes(descriptor_set, 29)
        for descriptor_set in self.indirect_reuse_candidate_sets:
            candidate_as_write = vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                pNext=as_info, dstSet=descriptor_set, dstBinding=12,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
            )
            vk.vkUpdateDescriptorSets(
                self.core.device, 1, [candidate_as_write], 0, None
            )
        ray_buffers = (self.ray_buffer, self.next_ray_buffer)
        shade_slots = (
            WINDOW_FRAMES_IN_FLIGHT
            if self.core.config.wavefront_profiling else 1
        )
        for slot in range(shade_slots):
            for index, ray_buffer in enumerate(ray_buffers):
                descriptor_set = self.shade_sets[slot * 2 + index]
                shade_bindings = [
                    (0, self.hit_buffer), (1, ray_buffer),
                    (2, self.path_buffer),
                    (3, self.core.scene_material_buffer),
                    (4, self.core.scene_vertex_buffer),
                    (5, self.core.scene_attribute_buffer),
                    (6, ray_buffers[1 - index]),
                    (7, self.medium_buffer),
                    (9, self.core.scene_light_buffer),
                    (10, self.core.scene_area_light_buffer),
                    (11, self.core.scene_texture_buffer),
                    (12, self.core.scene_texture_binding_buffer),
                    (15, self.secondary_path_buffer),
                    (17, self.core.scene_volume_header_buffer),
                    (18, self.core.scene_volume_scalar_buffer),
                    (19, self.core.scene_volume_transfer_buffer),
                    (20, self.core.scene_triangle_volume_buffer),
                ]
                if self.core.scene_custom_attribute_buffer is not None:
                    shade_bindings.append((
                        16, self.core.scene_custom_attribute_buffer,
                    ))
                if self.core.config.wavefront_profiling:
                    shade_bindings.append((14, self.work_counter_buffers[slot]))
                self._write_storage_set(descriptor_set, shade_bindings)
                shade_as_write = vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    pNext=as_info, dstSet=descriptor_set, dstBinding=8,
                    descriptorCount=1,
                    descriptorType=(
                        vk.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR
                    ),
                )
                vk.vkUpdateDescriptorSets(
                    self.core.device, 1, [shade_as_write], 0, None
                )
                if self.core.native_textures_enabled:
                    self._write_sampled_textures(descriptor_set, 13)
                self._write_sampled_volumes(descriptor_set, 21)

        for index, ray_buffer in enumerate(ray_buffers):
            buffer_bindings = (
                (1, ray_buffer), (2, self.hit_buffer),
                (3, self.core.scene_vertex_buffer),
            )
            infos = [self._buffer_info(buffer) for _, buffer in buffer_bindings]
            writes = [vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, pNext=as_info,
                dstSet=self.intersect_sets[index], dstBinding=0, descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
            )]
            writes.extend(vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=self.intersect_sets[index], dstBinding=binding,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[info],
            ) for (binding, _), info in zip(buffer_bindings, infos))
            vk.vkUpdateDescriptorSets(self.core.device, len(writes), writes, 0, None)
        if self.bucket_intersect_sets:
            for ray_index, ray_buffer in enumerate(ray_buffers):
                bucket_buffers = (
                    (1, ray_buffer), (2, self.coherent_hit_buffers[0]),
                    (3, self.coherent_hit_buffers[1]),
                    (4, self.core.scene_vertex_buffer),
                    (5, self.core.scene_material_buffer),
                )
                bucket_infos = [
                    self._buffer_info(buffer) for _binding, buffer in bucket_buffers
                ]
                bucket_writes = [vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    pNext=as_info,
                    dstSet=self.bucket_intersect_sets[ray_index],
                    dstBinding=0, descriptorCount=1,
                    descriptorType=(
                        vk.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR
                    ),
                )]
                bucket_writes.extend(vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=self.bucket_intersect_sets[ray_index],
                    dstBinding=binding, descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[info],
                ) for (binding, _buffer), info in zip(
                    bucket_buffers, bucket_infos
                ))
                vk.vkUpdateDescriptorSets(
                    self.core.device, len(bucket_writes), bucket_writes, 0, None
                )
                for slot in range(shade_slots):
                    for bucket, hit_buffer in enumerate(
                        self.coherent_hit_buffers
                    ):
                        descriptor_set = self.coherent_shade_sets[
                            slot * 4 + ray_index * 2 + bucket
                        ]
                        coherent_bindings = [
                            (0, hit_buffer), (1, ray_buffer),
                            (2, self.path_buffer),
                            (3, self.core.scene_material_buffer),
                            (4, self.core.scene_vertex_buffer),
                            (5, self.core.scene_attribute_buffer),
                            (6, ray_buffers[1 - ray_index]),
                            (7, self.medium_buffer),
                            (9, self.core.scene_light_buffer),
                            (10, self.core.scene_area_light_buffer),
                            (11, self.core.scene_texture_buffer),
                            (12, self.core.scene_texture_binding_buffer),
                            (15, self.secondary_path_buffer),
                        ]
                        if self.core.config.wavefront_profiling:
                            coherent_bindings.append((
                                14, self.work_counter_buffers[slot]
                            ))
                        self._write_storage_set(
                            descriptor_set, coherent_bindings
                        )
                        coherent_as_write = vk.VkWriteDescriptorSet(
                        sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                        pNext=as_info, dstSet=descriptor_set, dstBinding=8,
                        descriptorCount=1,
                        descriptorType=(
                            vk.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR
                        ),
                    )
                        vk.vkUpdateDescriptorSets(
                            self.core.device, 1, [coherent_as_write], 0, None
                        )
                        if self.core.native_textures_enabled:
                            self._write_sampled_textures(descriptor_set, 13)
        for descriptor_set in self.relax_prepare_sets:
            self._write_storage_set(descriptor_set, (
                (11, self.core.scene_previous_vertex_buffer),
            ))
        self._bound_scene_key = scene_key

    def _ensure_medium_buffer(self, scene):
        """Size medium storage for the scene at a synchronized bind point."""
        desired_capacity = (
            1 if self.core._use_opaque_scene_specialization(scene)
            else self.capacity
        )
        if self.medium_capacity == desired_capacity:
            return
        vk.vkDeviceWaitIdle(self.core.device)
        usage = (
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT
            | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT
        )
        replacement = self.core._create_buffer(
            desired_capacity * self.medium_stack_itemsize,
            usage, vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )
        old = self.medium_buffer
        self.medium_buffer = replacement
        self.medium_capacity = desired_capacity
        for descriptor_set in self.generate_sets:
            self._write_storage_set(descriptor_set, (
                (2, self.medium_buffer),
            ))
        if old in self.core._buffers:
            self.core._buffers.remove(old)
        vk.vkDestroyBuffer(self.core.device, old.buffer, None)
        vk.vkFreeMemory(self.core.device, old.memory, None)

    def _buffer_barrier(self, buffer, src_access, dst_access):
        return vk.VkBufferMemoryBarrier(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
            srcAccessMask=src_access, dstAccessMask=dst_access,
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            buffer=buffer.buffer, offset=0, size=buffer.size,
        )

    def record_work_counter_reset(self, command, slot):
        if not self.work_counter_buffers:
            return
        buffer = self.work_counter_buffers[slot]
        vk.vkCmdFillBuffer(command, buffer.buffer, 0, buffer.size, 0)
        barrier = self._buffer_barrier(
            buffer, vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
        )
        vk.vkCmdPipelineBarrier(
            command, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0, 0, None, 1, [barrier], 0, None,
        )

    def read_work_counters(self, slot):
        if not self.work_counter_buffers:
            return {}
        buffer = self.work_counter_buffers[slot]
        mapped = vk.vkMapMemory(
            self.core.device, buffer.memory, 0, buffer.size, 0
        )
        values = np.frombuffer(mapped, dtype=np.uint32, count=40).copy()
        vk.vkUnmapMemory(self.core.device, buffer.memory)
        names = (
            "path_rays", "shadow_rays", "texture_samples",
            "surface_hits", "environment_misses",
            "texture_transmission", "texture_base_color",
            "texture_metallic_roughness", "texture_emissive",
            "texture_occlusion", "texture_normal", "restir_history_accepted",
            "restir_history_selected", "restir_history_rejected",
            "restir_geometry_rejected", "restir_empty_history",
        ) + tuple(f"path_rays_bounce_{bounce}" for bounce in range(8)) + tuple(
            f"shadow_rays_bounce_{bounce}" for bounce in range(8)
        ) + tuple(f"surface_hits_bounce_{bounce}" for bounce in range(8))
        return {name: int(values[index]) for index, name in enumerate(names)}

    def read_indirect_reuse_counters(self, slot):
        if (not self.indirect_reuse_counter_buffers
                or not self.core.config.wavefront_indirect_reuse_profiling):
            return {}
        buffer = self.indirect_reuse_counter_buffers[slot]
        mapped = vk.vkMapMemory(
            self.core.device, buffer.memory, 0, buffer.size, 0
        )
        values = np.frombuffer(mapped, dtype=np.uint32, count=18).copy()
        vk.vkUnmapMemory(self.core.device, buffer.memory)
        names = (
            "sampled_pixels", "generated", "empty",
            "temporal_attempts", "temporal_accepted",
            "temporal_position_rejected", "temporal_normal_rejected",
            "temporal_material_rejected", "temporal_empty",
            "spatial_attempts", "spatial_accepted",
            "spatial_position_rejected", "spatial_normal_rejected",
            "spatial_material_rejected", "spatial_empty",
            "represented_samples",
            "history_clamped", "weight_saturated",
        )
        return {name: int(values[index]) for index, name in enumerate(names)}

    def dispatch(
        self, constants, tile_width, tile_height, *, output_image_slot=None,
        camera_slot=0,
        image_width=None, image_height=None, sample_index=0, sample_count=1,
        readback=True, command=None, timestamp=None,
        primary_hit_readback=False,
        restir_history_valid_override=None, restir_history_limit=None,
    ):
        self.bind_scene()
        self.ensure_overlap_pipelines()
        self.ensure_scattering_pipelines()
        self.ensure_volume_skipping_pipelines()
        if primary_hit_readback:
            if not readback or command is not None:
                raise ValueError("primary-hit capture requires immediate readback")
            self._ensure_primary_hit_capture(tile_width * tile_height)
        restir_enabled = bool(
            self.core.wavefront_restir_runtime_enabled
            and output_image_slot is not None
            and self.core.window_frames
        )
        previous_frame = (
            self.core.window_frames[1 - camera_slot]
            if restir_enabled else None
        )
        restir_history_valid = bool(
            previous_frame is not None
            and previous_frame.get("wavefront_reservoir_valid")
            and previous_frame.get("wavefront_render_extent")
                == (image_width, image_height)
        )
        if restir_history_valid_override is not None:
            restir_history_valid = bool(restir_history_valid_override)
        if restir_history_limit is None:
            restir_history_limit = self.core.config.wavefront_restir_history_limit

        def record(command):
            shade_slot = (
                camera_slot if self.core.config.wavefront_profiling else 0
            )
            queue_buffers = (
                self.ray_buffer, self.hit_buffer, self.next_ray_buffer,
                *self.coherent_hit_buffers,
            )
            reusable = [self._buffer_barrier(
                buffer,
                vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
                vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            ) for buffer in queue_buffers]
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                0, None, len(reusable), reusable, 0, None,
            )
            for buffer in queue_buffers:
                vk.vkCmdFillBuffer(command, buffer.buffer, 0, 4, 0)
                vk.vkCmdFillBuffer(
                    command, buffer.buffer, 4, 4, self.capacity
                )
                vk.vkCmdFillBuffer(command, buffer.buffer, 8, 4, 0)
            if self._denoiser_signals_active():
                vk.vkCmdFillBuffer(
                    command, self.secondary_path_buffer.buffer, 0,
                    tile_width * tile_height
                    * self.secondary_path_itemsize, 0,
                )
            reset_barriers = [self._buffer_barrier(
                buffer, vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
            ) for buffer in queue_buffers]
            reset_barriers.extend(self._buffer_barrier(
                buffer,
                vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
                vk.VK_ACCESS_SHADER_WRITE_BIT,
            ) for buffer in (
                self.path_buffer, self.medium_buffer,
            ))
            reset_barriers.append(self._buffer_barrier(
                self.secondary_path_buffer,
                (vk.VK_ACCESS_TRANSFER_WRITE_BIT
                 if self._denoiser_signals_active() else
                 vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT),
                vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
            ))
            reset_barriers.extend(self._buffer_barrier(
                buffer,
                vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT
                | vk.VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
                vk.VK_ACCESS_SHADER_WRITE_BIT,
            ) for buffer in (self.resolve_buffer, self.indirect_buffer))
            restir_image_barriers = []
            if restir_enabled:
                current_reservoir = self.core.window_frames[
                    camera_slot
                ]["wavefront_reservoir_buffer"]
                previous_reservoir = previous_frame[
                    "wavefront_reservoir_buffer"
                ]
                reset_barriers.extend((
                    self._buffer_barrier(
                        current_reservoir, vk.VK_ACCESS_SHADER_WRITE_BIT,
                        vk.VK_ACCESS_SHADER_WRITE_BIT,
                    ),
                    self._buffer_barrier(
                        previous_reservoir, vk.VK_ACCESS_SHADER_WRITE_BIT,
                        vk.VK_ACCESS_SHADER_READ_BIT,
                    ),
                ))
                restir_image_barriers = [self.core._image_barrier(
                    previous_frame[name],
                    vk.VK_IMAGE_LAYOUT_GENERAL, vk.VK_IMAGE_LAYOUT_GENERAL,
                    vk.VK_ACCESS_SHADER_WRITE_BIT,
                    vk.VK_ACCESS_SHADER_READ_BIT,
                ) for name in (
                    "wavefront_position_image", "wavefront_normal_image",
                    "wavefront_material_image",
                )]
            primary_stage = vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
            if self.core.resolved_execution_strategy == "ser":
                primary_stage |= vk.VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_TRANSFER_BIT
                | vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                | vk.VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT,
                primary_stage, 0,
                0, None, len(reset_barriers), reset_barriers,
                len(restir_image_barriers), restir_image_barriers,
            )
            coherent_shading = self.core.config.wavefront_material_bucketing
            fused_primary = output_image_slot is not None
            primary_constants = None
            hybrid = False
            persistent = False
            if fused_primary:
                strategy = self.core.resolved_execution_strategy
                if self.custom_primary_pipeline is not None:
                    strategy = "wavefront"
                persistent = strategy == "persistent"
                ser = strategy == "ser"
                megakernel = strategy == "megakernel" or persistent or ser
                hybrid = strategy == "hybrid"
                scene = self.core.scene_resources.scene
                opaque_specialization = (
                    self.core._use_opaque_scene_specialization(scene)
                )
                megakernel_single_warp = bool(
                    strategy == "megakernel"
                    and opaque_specialization
                    and not scene.textures
                    and self.core.config.wavefront_megakernel_single_warp
                )
                primary_pipeline = {
                    "wavefront": (
                        self.custom_primary_pipeline or self.primary_pipeline
                    ),
                    "hybrid": self.hybrid_pipeline,
                    "megakernel": self.megakernel_pipeline,
                    "persistent": self.persistent_pipeline,
                    "ser": self.ser_megakernel_pipeline,
                }[strategy]
                if hybrid and opaque_specialization:
                    primary_pipeline = self.hybrid_opaque_pipeline
                    if (
                        self.core.config.wavefront_untextured_specialization
                        and not scene.textures and self.production_restir
                    ):
                        primary_pipeline = (
                            self.hybrid_opaque_untextured_production_pipeline
                        )
                if strategy == "megakernel" and opaque_specialization:
                    primary_pipeline = self.megakernel_opaque_pipeline
                    if (
                        self.core.config.wavefront_untextured_specialization
                        and not scene.textures
                    ):
                        part = (
                            self.core.config
                            .wavefront_untextured_specialization_part
                        )
                        if part == "primary":
                            primary_pipeline = (
                                self.megakernel_untextured_primary_pipeline
                            )
                        elif part == "secondary":
                            primary_pipeline = (
                                self.megakernel_untextured_secondary_pipeline
                            )
                        else:
                            primary_pipeline = (
                                self.megakernel_opaque_untextured_pipeline
                            )
                elif (
                    strategy == "megakernel"
                    and self.core.config.wavefront_untextured_specialization
                    and not scene.textures
                    and not self.core.config.wavefront_native_textures
                    and not self.core.config.wavefront_profiling
                ):
                    primary_pipeline = self.megakernel_untextured_pipeline
                    swizzle_width = (
                        self.core.config.wavefront_megakernel_group_swizzle
                    )
                    if swizzle_width:
                        primary_pipeline = (
                            self.megakernel_untextured_swizzle_pipelines[
                                swizzle_width
                            ]
                        )
                if persistent and self.core.config.wavefront_persistent_coarse_tiles:
                    primary_pipeline = self._volume_pipeline(
                        "wavefront_persistent_coarse", scene,
                    ) or self.persistent_coarse_pipeline
                volume_pipeline = self._volume_pipeline({
                    "wavefront": "wavefront_primary",
                    "hybrid": "wavefront_hybrid",
                    "megakernel": "wavefront_megakernel",
                    "persistent": "wavefront_persistent",
                }[strategy], scene)
                if volume_pipeline is not None:
                    if self.custom_primary_pipeline is not None:
                        primary_pipeline = self.custom_primary_pipeline
                    elif not (
                        persistent
                        and self.core.config.wavefront_persistent_coarse_tiles
                    ):
                        primary_pipeline = volume_pipeline
                primary_bind_point = (
                    PIPELINE_BIND_POINT_RAY_TRACING_KHR
                    if ser else vk.VK_PIPELINE_BIND_POINT_COMPUTE
                )
                vk.vkCmdBindPipeline(
                    command, primary_bind_point, primary_pipeline,
                )
                vk.vkCmdBindDescriptorSets(
                    command, primary_bind_point,
                    self.primary_pipeline_layout, 0, 1,
                    [self.primary_sets[camera_slot]], 0, None,
                )
                inline_bounces = (
                    self.core.config.wavefront_hybrid_inline_bounces
                    if hybrid else self.core.config.max_bounces
                )
                secondary_area_samples = (
                    self.core.config.wavefront_secondary_area_light_samples
                    or self.core.config.area_light_samples
                )
                effect_ranges = [
                    value
                    for binding in self.core.object_effect_bindings
                    for value in binding[0]
                ]
                effect_ranges.extend([0] * (8 - len(effect_ranges)))
                primary_constants = constants + bytearray(struct.pack(
                    "5If4I2f10If13I", self.core.config.max_bounces,
                    scene.analytic_light_count,
                    scene.emissive_triangle_count,
                    self.core.config.area_light_samples,
                    secondary_area_samples,
                    scene.emissive_light_weight,
                    int(
                        self.core.config.wavefront_temporal_reconstruction
                        or self.core.config.stationary_accumulation
                        or self.core.config.wavefront_diffuse_filter
                        or restir_enabled
                        or self.core.config.wavefront_indirect_reuse_candidates
                        or self.core.config.object_effects
                    ),
                    self.core.config.wavefront_environment_samples,
                    int(self.core.config.wavefront_subgroup_enqueue),
                    (self.core.config.wavefront_russian_roulette_start
                     if self.core.config.wavefront_russian_roulette else 0),
                    self.core.config.wavefront_russian_roulette_min_survival,
                    self.core.config.wavefront_secondary_nee_probability,
                    inline_bounces,
                    int(restir_enabled),
                    int(restir_history_valid),
                    restir_history_limit,
                    self.core.config.wavefront_restir_candidates,
                    int(self.core.config.wavefront_restir_spatial_reuse),
                    self.core.config.wavefront_restir_spatial_neighbors,
                    self.core.config.wavefront_restir_spatial_radius,
                    int(self.core.config.wavefront_restir_pairwise_mis),
                    int(self.core.config.wavefront_restir_generalized_mis),
                    self.core.config.wavefront_restir_generalized_balance_cap,
                    int(self.core.config.wavefront_unified_secondary_nee),
                    int(self.core.config.wavefront_unified_primary_restir),
                    int(self.core.config.wavefront_stratified_primary_restir),
                    int(
                        self.core.config.wavefront_indirect_reuse_candidates
                        or self._denoiser_signals_active()
                    ),
                    self._indirect_capture_stride(),
                    *effect_ranges,
                ))
                vk.vkCmdPushConstants(
                    command, self.primary_pipeline_layout,
                    (vk.VK_SHADER_STAGE_RAYGEN_BIT_KHR
                     if ser else vk.VK_SHADER_STAGE_COMPUTE_BIT),
                    0, len(primary_constants),
                    vk.ffi.from_buffer(primary_constants),
                )
                if ser:
                    self.core.cmd_trace_rays(
                        command,
                        vk.ffi.addressof(self.ser_raygen_region),
                        vk.ffi.addressof(self.ser_miss_region),
                        vk.ffi.addressof(self.ser_empty_region),
                        vk.ffi.addressof(self.ser_empty_region),
                        tile_width, tile_height, 1,
                    )
                elif persistent and self.core.config.wavefront_persistent_coarse_tiles:
                    tile_groups = ((tile_width + 7) // 8) * (
                        (tile_height + 7) // 8
                    )
                    vk.vkCmdDispatch(command, min(tile_groups, 512), 1, 1)
                else:
                    vk.vkCmdDispatch(
                        command, (tile_width + 7) // 8,
                        ((tile_height + 3) // 4
                         if megakernel_single_warp
                         else (tile_height + 7) // 8), 1,
                    )
                if timestamp:
                    timestamp(command, (
                        strategy if strategy != "wavefront" else "primary"
                    ))
                generated_buffers = (
                    self.next_ray_buffer, self.path_buffer, self.medium_buffer,
                    self.secondary_path_buffer,
                )
                first_bounce = (
                    self.core.config.max_bounces if megakernel
                    else inline_bounces if hybrid else 1
                )
            else:
                vk.vkCmdBindPipeline(
                    command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                    self.generate_pipeline,
                )
                vk.vkCmdBindDescriptorSets(
                    command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                    self.generate_pipeline_layout, 0, 1,
                    [self.generate_sets[camera_slot]], 0, None,
                )
                generate_constants = constants
                if self._denoiser_signals_active():
                    # GenerateConstants.tile_frame.z is unused by the split
                    # primary-ray stage. Reserve its high bit for the opt-in
                    # cold-path signal capture flag without changing the
                    # long-lived Vulkan push-constant ABI.
                    generate_constants = bytearray(constants)
                    encoded_sample_count = struct.unpack_from(
                        "I", generate_constants, 24
                    )[0] | 0x80000000
                    struct.pack_into(
                        "I", generate_constants, 24, encoded_sample_count
                    )
                vk.vkCmdPushConstants(
                    command, self.generate_pipeline_layout,
                    vk.VK_SHADER_STAGE_COMPUTE_BIT, 0,
                    len(generate_constants),
                    vk.ffi.from_buffer(generate_constants),
                )
                vk.vkCmdDispatch(
                    command, (tile_width + 7) // 8,
                    (tile_height + 7) // 8, 1,
                )
                if timestamp:
                    timestamp(command, "generate")
                generated_buffers = (
                    self.ray_buffer, self.path_buffer, self.medium_buffer,
                    self.secondary_path_buffer,
                )
                first_bounce = 0
            generated = [self._buffer_barrier(
                buffer, vk.VK_ACCESS_SHADER_WRITE_BIT,
                vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
            ) for buffer in generated_buffers]
            vk.vkCmdPipelineBarrier(
                command,
                (vk.VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR
                 if fused_primary and ser
                 else vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT),
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                0, None, len(generated), generated, 0, None,
            )
            scene = self.core.scene_resources.scene
            secondary_area_samples = (
                self.core.config.wavefront_secondary_area_light_samples
                or self.core.config.area_light_samples
            )
            def make_shade_constants(fused_intersection):
                return bytearray(struct.pack(
                    "5IfIIf2If2I", self.core.config.max_bounces,
                    scene.analytic_light_count, scene.emissive_triangle_count,
                    self.core.config.area_light_samples,
                    secondary_area_samples, scene.emissive_light_weight,
                    self.core.config.wavefront_environment_samples,
                    (self.core.config.wavefront_russian_roulette_start
                     if self.core.config.wavefront_russian_roulette else 0),
                    self.core.config.wavefront_russian_roulette_min_survival,
                    int(fused_intersection),
                    int(self.core.config.wavefront_subgroup_enqueue),
                    self.core.config.wavefront_secondary_nee_probability,
                    int(self.core.config.wavefront_unified_secondary_nee),
                    int(
                        self.core.config.wavefront_indirect_reuse_candidates
                        or self._denoiser_signals_active()
                    ),
                ))
            shade_constants_fused = make_shade_constants(
                self.core.config.wavefront_fused_secondary
            )
            shade_constants_split = make_shade_constants(False)
            selected_shade_pipeline = (
                self.custom_shade_pipeline or self.shade_pipeline
            )
            volume_shade_pipeline = self._volume_pipeline(
                "wavefront_shade", scene,
            )
            if (volume_shade_pipeline is not None
                    and self.custom_shade_pipeline is None):
                selected_shade_pipeline = volume_shade_pipeline
            ray_buffers = (self.ray_buffer, self.next_ray_buffer)
            if hybrid and self.core.config.wavefront_persistent_continuations:
                continuation_queue = first_bounce & 1
                vk.vkCmdBindPipeline(
                    command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                    self.indirect_pipeline,
                )
                vk.vkCmdBindDescriptorSets(
                    command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                    self.indirect_pipeline_layout, 0, 1,
                    [self.indirect_sets[continuation_queue]], 0, None,
                )
                vk.vkCmdDispatch(command, 1, 1, 1)
                continuation_indirect_ready = self._buffer_barrier(
                    self.indirect_buffer, vk.VK_ACCESS_SHADER_WRITE_BIT,
                    vk.VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
                )
                vk.vkCmdPipelineBarrier(
                    command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    vk.VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0,
                    0, None, 1, [continuation_indirect_ready], 0, None,
                )
                vk.vkCmdBindPipeline(
                    command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                    (self._volume_pipeline(
                        "wavefront_persistent_continuation", scene,
                    ) or (self.persistent_continuation_opaque_pipeline
                          if opaque_specialization
                          else self.persistent_continuation_pipeline)),
                )
                vk.vkCmdBindDescriptorSets(
                    command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                    self.primary_pipeline_layout, 0, 1,
                    [self.primary_sets[camera_slot]], 0, None,
                )
                vk.vkCmdPushConstants(
                    command, self.primary_pipeline_layout,
                    vk.VK_SHADER_STAGE_COMPUTE_BIT, 0,
                    len(primary_constants),
                    vk.ffi.from_buffer(primary_constants),
                )
                vk.vkCmdDispatchIndirect(
                    command, self.indirect_buffer.buffer, 0
                )
                if timestamp:
                    timestamp(command, "persistent_continuation")
                first_bounce = self.core.config.max_bounces
            for bounce in range(first_bounce, self.core.config.max_bounces):
                bucket_this_bounce = (
                    coherent_shading
                    and bounce
                        >= self.core.config.wavefront_material_bucketing_start_bounce
                )
                capture_primary = primary_hit_readback and bounce == 0
                shade_constants = (
                    shade_constants_split
                    if bucket_this_bounce or capture_primary
                    else shade_constants_fused
                )
                current = bounce & 1
                following = 1 - current
                if bounce:
                    reusable = [
                        self._buffer_barrier(
                            self.indirect_buffer,
                            vk.VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
                            vk.VK_ACCESS_SHADER_WRITE_BIT,
                        ),
                        self._buffer_barrier(
                            ray_buffers[current], vk.VK_ACCESS_SHADER_WRITE_BIT,
                            vk.VK_ACCESS_SHADER_READ_BIT,
                        ),
                    ]
                    vk.vkCmdPipelineBarrier(
                        command, vk.VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT
                        | vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                        0, None, len(reusable), reusable, 0, None,
                    )
                vk.vkCmdBindPipeline(
                    command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                    self.indirect_pipeline,
                )
                vk.vkCmdBindDescriptorSets(
                    command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                    self.indirect_pipeline_layout, 0, 1,
                    [self.indirect_sets[current]], 0, None,
                )
                vk.vkCmdDispatch(command, 1, 1, 1)
                indirect_ready = self._buffer_barrier(
                    self.indirect_buffer, vk.VK_ACCESS_SHADER_WRITE_BIT,
                    vk.VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
                )
                vk.vkCmdPipelineBarrier(
                    command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    vk.VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0,
                    0, None, 1, [indirect_ready], 0, None,
                )
                if (not self.core.config.wavefront_fused_secondary
                        or bucket_this_bounce or capture_primary):
                    vk.vkCmdBindPipeline(
                        command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                        (self.bucket_intersect_pipeline if bucket_this_bounce
                         else self.intersect_pipeline),
                    )
                    vk.vkCmdBindDescriptorSets(
                        command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                        (self.bucket_intersect_pipeline_layout if bucket_this_bounce
                         else self.intersect_pipeline_layout), 0, 1,
                        ([self.bucket_intersect_sets[current]] if bucket_this_bounce
                         else [self.intersect_sets[current]]), 0, None,
                    )
                    vk.vkCmdDispatchIndirect(
                        command, self.indirect_buffer.buffer, 0
                    )
                    if timestamp:
                        timestamp(command, (
                            f"intersect_bucketed.{bounce}" if bucket_this_bounce
                            else f"intersect.{bounce}"
                        ))
                    intersected = [self._buffer_barrier(
                        buffer, vk.VK_ACCESS_SHADER_WRITE_BIT,
                        vk.VK_ACCESS_SHADER_READ_BIT
                        | vk.VK_ACCESS_SHADER_WRITE_BIT,
                    ) for buffer in ((
                        *self.coherent_hit_buffers,
                        self.path_buffer, self.medium_buffer,
                    ) if bucket_this_bounce else (
                        self.hit_buffer, self.path_buffer, self.medium_buffer,
                    ))]
                    vk.vkCmdPipelineBarrier(
                        command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                        0, None, len(intersected), intersected, 0, None,
                    )
                    if capture_primary:
                        capture_ready = self._buffer_barrier(
                            self.hit_buffer,
                            vk.VK_ACCESS_SHADER_READ_BIT
                            | vk.VK_ACCESS_SHADER_WRITE_BIT,
                            vk.VK_ACCESS_TRANSFER_READ_BIT,
                        )
                        vk.vkCmdPipelineBarrier(
                            command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                            0, None, 1, [capture_ready], 0, None,
                        )
                        capture_size = (
                            16 + tile_width * tile_height * self.hit_dtype.itemsize
                        )
                        vk.vkCmdCopyBuffer(
                            command, self.hit_buffer.buffer,
                            self.primary_hit_snapshot.buffer, 1,
                            [vk.VkBufferCopy(
                                srcOffset=0, dstOffset=0, size=capture_size,
                            )],
                        )
                        capture_complete = self._buffer_barrier(
                            self.hit_buffer, vk.VK_ACCESS_TRANSFER_READ_BIT,
                            vk.VK_ACCESS_SHADER_READ_BIT,
                        )
                        vk.vkCmdPipelineBarrier(
                            command, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                            0, None, 1, [capture_complete], 0, None,
                        )
                if bucket_this_bounce:
                    for bucket in range(2):
                        bucket_ready = [
                            self._buffer_barrier(
                                self.indirect_buffer,
                                vk.VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
                                vk.VK_ACCESS_SHADER_WRITE_BIT,
                            ),
                            self._buffer_barrier(
                                self.coherent_hit_buffers[bucket],
                                vk.VK_ACCESS_SHADER_WRITE_BIT,
                                vk.VK_ACCESS_SHADER_READ_BIT,
                            ),
                        ]
                        if bucket:
                            bucket_ready.extend(self._buffer_barrier(
                                buffer, vk.VK_ACCESS_SHADER_WRITE_BIT,
                                vk.VK_ACCESS_SHADER_READ_BIT
                                | vk.VK_ACCESS_SHADER_WRITE_BIT,
                            ) for buffer in (
                                ray_buffers[following], self.path_buffer,
                                self.medium_buffer,
                            ))
                        vk.vkCmdPipelineBarrier(
                            command,
                            vk.VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT
                            | vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                            0, None, len(bucket_ready), bucket_ready, 0, None,
                        )
                        vk.vkCmdBindPipeline(
                            command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                            self.indirect_pipeline,
                        )
                        vk.vkCmdBindDescriptorSets(
                            command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                            self.indirect_pipeline_layout, 0, 1,
                            [self.coherent_indirect_sets[bucket]], 0, None,
                        )
                        vk.vkCmdDispatch(command, 1, 1, 1)
                        bucket_indirect_ready = self._buffer_barrier(
                            self.indirect_buffer, vk.VK_ACCESS_SHADER_WRITE_BIT,
                            vk.VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
                        )
                        vk.vkCmdPipelineBarrier(
                            command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            vk.VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0,
                            0, None, 1, [bucket_indirect_ready], 0, None,
                        )
                        vk.vkCmdBindPipeline(
                            command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                            selected_shade_pipeline,
                        )
                        vk.vkCmdBindDescriptorSets(
                            command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                            self.shade_pipeline_layout, 0, 1,
                            [self.coherent_shade_sets[
                                shade_slot * 4 + current * 2 + bucket
                            ]],
                            0, None,
                        )
                        vk.vkCmdPushConstants(
                            command, self.shade_pipeline_layout,
                            vk.VK_SHADER_STAGE_COMPUTE_BIT, 0,
                            len(shade_constants),
                            vk.ffi.from_buffer(shade_constants),
                        )
                        vk.vkCmdDispatchIndirect(
                            command, self.indirect_buffer.buffer, 0
                        )
                        if timestamp:
                            timestamp(command, (
                                f"shade.{'plain' if bucket == 0 else 'textured'}."
                                f"{bounce}"
                            ))
                else:
                    vk.vkCmdBindPipeline(
                        command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                        selected_shade_pipeline,
                    )
                    vk.vkCmdBindDescriptorSets(
                        command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                        self.shade_pipeline_layout, 0, 1,
                        [self.shade_sets[shade_slot * 2 + current]], 0, None,
                    )
                    vk.vkCmdPushConstants(
                        command, self.shade_pipeline_layout,
                        vk.VK_SHADER_STAGE_COMPUTE_BIT, 0,
                        len(shade_constants),
                        vk.ffi.from_buffer(shade_constants),
                    )
                    vk.vkCmdDispatchIndirect(
                        command, self.indirect_buffer.buffer, 0
                    )
                    if timestamp:
                        timestamp(command, (
                            f"intersect_shade.{bounce}"
                            if self.core.config.wavefront_fused_secondary
                            else f"shade.{bounce}"
                        ))
                if bounce + 1 < self.core.config.max_bounces:
                    to_reset = (
                        (self.hit_buffer, ray_buffers[current],
                         *self.coherent_hit_buffers)
                        if bucket_this_bounce
                        else ((ray_buffers[current],)
                            if self.core.config.wavefront_fused_secondary
                            else (self.hit_buffer, ray_buffers[current]))
                    )
                    reset_ready = [self._buffer_barrier(
                        buffer, vk.VK_ACCESS_SHADER_READ_BIT
                        | vk.VK_ACCESS_SHADER_WRITE_BIT,
                        vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    ) for buffer in to_reset]
                    vk.vkCmdPipelineBarrier(
                        command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                        0, None, len(reset_ready), reset_ready, 0, None,
                    )
                    for buffer in to_reset:
                        vk.vkCmdFillBuffer(command, buffer.buffer, 0, 4, 0)
                        vk.vkCmdFillBuffer(command, buffer.buffer, 8, 4, 0)
                    next_ready = [self._buffer_barrier(
                        buffer, vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                        vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
                    ) for buffer in to_reset]
                    next_ready.append(self._buffer_barrier(
                        ray_buffers[following], vk.VK_ACCESS_SHADER_WRITE_BIT,
                        vk.VK_ACCESS_SHADER_READ_BIT,
                    ))
                    vk.vkCmdPipelineBarrier(
                        command, vk.VK_PIPELINE_STAGE_TRANSFER_BIT
                        | vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                        0, None, len(next_ready), next_ready, 0, None,
                    )
            if output_image_slot is None:
                path_ready = self._buffer_barrier(
                    self.path_buffer, vk.VK_ACCESS_SHADER_WRITE_BIT,
                    vk.VK_ACCESS_SHADER_READ_BIT,
                )
                vk.vkCmdPipelineBarrier(
                    command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                    0, None, 1, [path_ready], 0, None,
                )
                vk.vkCmdBindPipeline(
                    command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                    self.resolve_pipeline,
                )
                vk.vkCmdBindDescriptorSets(
                    command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                    self.resolve_pipeline_layout, 0, 1, [self.resolve_set], 0, None,
                )
                resolve_constants = bytearray(struct.pack(
                    "4I", tile_width * tile_height, 0, 0, 0
                ))
                vk.vkCmdPushConstants(
                    command, self.resolve_pipeline_layout,
                    vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, len(resolve_constants),
                    vk.ffi.from_buffer(resolve_constants),
                )
                vk.vkCmdDispatch(
                    command, (tile_width * tile_height + 63) // 64, 1, 1
                )
                if timestamp:
                    timestamp(command, "resolve")
                resolved_ready = self._buffer_barrier(
                    self.resolve_buffer, vk.VK_ACCESS_SHADER_WRITE_BIT,
                    vk.VK_ACCESS_SHADER_READ_BIT,
                )
                vk.vkCmdPipelineBarrier(
                    command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                    0, None, 1, [resolved_ready], 0, None,
                )
                vk.vkCmdBindPipeline(
                    command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                    self.wavefront_tone_pipeline,
                )
                vk.vkCmdBindDescriptorSets(
                    command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                    self.wavefront_tone_pipeline_layout, 0, 1,
                    [self.wavefront_tone_set], 0, None,
                )
                tone_constants = bytearray(struct.pack(
                    "IfII", tile_width * tile_height,
                    self.core.config.wavefront_exposure, 0, 0,
                ))
                vk.vkCmdPushConstants(
                    command, self.wavefront_tone_pipeline_layout,
                    vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, len(tone_constants),
                    vk.ffi.from_buffer(tone_constants),
                )
                vk.vkCmdDispatch(
                    command, (tile_width * tile_height + 63) // 64, 1, 1
                )
                if timestamp:
                    timestamp(command, "tone")
            else:
                self.record_path_to_hdr(
                    command, output_image_slot, tile_width * tile_height,
                    image_width, image_height, sample_index, sample_count,
                )
                self.record_relax_prepare(
                    command, output_image_slot, tile_width * tile_height,
                    image_width, image_height,
                )
                if timestamp:
                    timestamp(command, "resolve_hdr")
            if readback:
                copies = [
                    (self.ray_buffer, self.ray_readback, 16),
                    (self.hit_buffer, self.hit_readback, 16),
                    (self.next_ray_buffer, self.next_ray_readback, 16),
                    (self.resolve_buffer, self.resolve_readback,
                     tile_width * tile_height * 32),
                    (self.packed_buffer, self.packed_readback,
                     tile_width * tile_height * 4),
                ]
                if primary_hit_readback:
                    copies.append((
                        self.primary_hit_snapshot,
                        self.primary_hit_readback,
                        16 + tile_width * tile_height * self.hit_dtype.itemsize,
                    ))
                if self.secondary_path_readback is not None:
                    copies.append((
                        self.secondary_path_buffer,
                        self.secondary_path_readback,
                        tile_width * tile_height
                        * self.secondary_path_itemsize,
                    ))
                copy_ready = [self._buffer_barrier(
                    source,
                    (vk.VK_ACCESS_TRANSFER_WRITE_BIT
                     if source is self.primary_hit_snapshot
                     else vk.VK_ACCESS_SHADER_READ_BIT
                     | vk.VK_ACCESS_SHADER_WRITE_BIT
                     | vk.VK_ACCESS_INDIRECT_COMMAND_READ_BIT),
                    vk.VK_ACCESS_TRANSFER_READ_BIT,
                ) for source, _target, _size in copies]
                vk.vkCmdPipelineBarrier(
                    command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                    | vk.VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT
                    | vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                    vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                    0, None, len(copy_ready), copy_ready, 0, None,
                )
                for source, target, size in copies:
                    vk.vkCmdCopyBuffer(
                        command, source.buffer, target.buffer, 1,
                        [vk.VkBufferCopy(srcOffset=0, dstOffset=0, size=size)],
                    )
                host_ready = [self._buffer_barrier(
                    target, vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    vk.VK_ACCESS_HOST_READ_BIT,
                ) for _source, target, _size in copies]
                vk.vkCmdPipelineBarrier(
                    command, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                    vk.VK_PIPELINE_STAGE_HOST_BIT, 0,
                    0, None, len(host_ready), host_ready, 0, None,
                )

        if command is None:
            self.core._single_use(record)
        else:
            if readback:
                raise ValueError("recorded wavefront dispatch cannot request readback")
            record(command)
        if not readback:
            return None
        final_queue = (
            self.next_ray_buffer
            if self.core.config.max_bounces & 1
            else self.ray_buffer
        )
        primary_count = tile_width * tile_height
        result = (
            {"count": primary_count, "capacity": self.capacity, "overflow": 0},
            self._read_counts(self.hit_readback),
            self._read_counts(
                self.next_ray_readback
                if final_queue is self.next_ray_buffer
                else self.ray_readback
            ),
            self._read_resolved(primary_count),
            self._read_packed(primary_count),
        )
        if primary_hit_readback:
            result += (self._read_primary_hits(primary_count),)
        if self.secondary_path_readback is not None:
            result += (self._read_secondary_paths(primary_count),)
        return result

    def _read_secondary_paths(self, count):
        from ...wavefront import SECONDARY_PATH_STATE_DTYPE

        size = count * SECONDARY_PATH_STATE_DTYPE.itemsize
        mapped = vk.vkMapMemory(
            self.core.device, self.secondary_path_readback.memory, 0, size, 0
        )
        result = np.frombuffer(
            bytes(mapped[:size]), dtype=SECONDARY_PATH_STATE_DTYPE, count=count,
        ).copy()
        vk.vkUnmapMemory(
            self.core.device, self.secondary_path_readback.memory
        )
        return result

    def _read_primary_hits(self, count):
        size = 16 + count * self.hit_dtype.itemsize
        mapped = vk.vkMapMemory(
            self.core.device, self.primary_hit_readback.memory, 0, size, 0
        )
        result = np.frombuffer(
            bytes(mapped[16:size]), dtype=self.hit_dtype, count=count,
        ).copy()
        vk.vkUnmapMemory(self.core.device, self.primary_hit_readback.memory)
        return result

    def _read_resolved(self, count):
        from ...wavefront import RESOLVED_PIXEL_DTYPE

        size = count * RESOLVED_PIXEL_DTYPE.itemsize
        mapped = vk.vkMapMemory(self.core.device, self.resolve_readback.memory, 0, size, 0)
        result = np.frombuffer(bytes(mapped[:size]), dtype=RESOLVED_PIXEL_DTYPE).copy()
        vk.vkUnmapMemory(self.core.device, self.resolve_readback.memory)
        return result

    def _read_packed(self, count):
        size = count * 4
        mapped = vk.vkMapMemory(self.core.device, self.packed_readback.memory, 0, size, 0)
        result = np.frombuffer(bytes(mapped[:size]), dtype=np.uint32).copy()
        vk.vkUnmapMemory(self.core.device, self.packed_readback.memory)
        return result

    def _read_counts(self, buffer):
        mapped = vk.vkMapMemory(self.core.device, buffer.memory, 0, 16, 0)
        values = struct.unpack("4I", bytes(mapped[:16]))
        vk.vkUnmapMemory(self.core.device, buffer.memory)
        return {"count": values[0], "capacity": values[1], "overflow": values[2]}

    def close(self):
        device = self.core.device
        for pipeline in (
            self.generate_pipeline, self.primary_pipeline,
            self.hybrid_pipeline,
            self.hybrid_opaque_pipeline,
            self.hybrid_opaque_untextured_production_pipeline,
            self.megakernel_pipeline,
            self.megakernel_untextured_pipeline,
            *self.megakernel_untextured_swizzle_pipelines.values(),
            self.megakernel_opaque_pipeline,
            self.megakernel_opaque_untextured_pipeline,
            self.megakernel_untextured_primary_pipeline,
            self.megakernel_untextured_secondary_pipeline,
            self.megakernel_opaque_untextured_production_pipeline,
            self.megakernel_opaque_untextured_wg32_pipeline,
            *self.megakernel_swizzle_pipelines.values(),
            self.persistent_pipeline,
            self.persistent_coarse_pipeline,
            self.persistent_continuation_pipeline,
            self.persistent_continuation_opaque_pipeline,
            self.intersect_pipeline, self.shade_pipeline,
            self.custom_primary_pipeline, self.custom_shade_pipeline,
            self.resolve_pipeline,
            self.indirect_pipeline,
            self.wavefront_tone_pipeline,
            self.wavefront_image_pipeline,
            self.reconstruct_pipeline,
            self.reconstruct_bgra_pipeline,
            self.relax_prepare_pipeline,
            self.relax_temporal_pipeline,
            self.relax_atrous_pipeline,
            self.relax_compose_pipeline,
            self.indirect_reuse_clear_pipeline,
            self.indirect_reuse_candidate_pipeline,
            self.indirect_reuse_debug_pipeline,
            self.bucket_intersect_pipeline,
            self.ser_megakernel_pipeline,
            *self.overlap_pipelines.values(),
            *self.scattering_pipelines.values(),
            *self.volume_skipping_pipelines.values(),
        ):
            if pipeline:
                vk.vkDestroyPipeline(device, pipeline, None)
        for module in (
            self.generate_module, self.primary_module,
            self.hybrid_module,
            self.hybrid_opaque_module,
            self.hybrid_opaque_untextured_production_module,
            self.megakernel_module,
            self.megakernel_untextured_module,
            *self.megakernel_untextured_swizzle_modules.values(),
            self.megakernel_opaque_module,
            self.megakernel_opaque_untextured_module,
            self.megakernel_untextured_primary_module,
            self.megakernel_untextured_secondary_module,
            self.megakernel_opaque_untextured_production_module,
            self.megakernel_opaque_untextured_wg32_module,
            *self.megakernel_swizzle_modules.values(),
            self.persistent_module,
            self.persistent_coarse_module,
            self.persistent_continuation_module,
            self.persistent_continuation_opaque_module,
            self.intersect_module, self.shade_module,
            self.custom_primary_module, self.custom_shade_module,
            self.resolve_module,
            self.indirect_module,
            self.wavefront_tone_module,
            self.wavefront_image_module,
            self.reconstruct_module,
            self.reconstruct_bgra_module,
            self.relax_prepare_module,
            self.relax_temporal_module,
            self.relax_atrous_module,
            self.relax_compose_module,
            self.indirect_reuse_clear_module,
            self.indirect_reuse_candidate_module,
            self.indirect_reuse_debug_module,
            self.bucket_intersect_module,
            self.ser_megakernel_module,
            self.ser_miss_module,
            *self.overlap_modules.values(),
            *self.scattering_modules.values(),
            *self.volume_skipping_modules.values(),
        ):
            if module:
                vk.vkDestroyShaderModule(device, module, None)
        for layout in (
            self.generate_pipeline_layout, self.primary_pipeline_layout,
            self.intersect_pipeline_layout,
            self.shade_pipeline_layout,
            self.resolve_pipeline_layout,
            self.indirect_pipeline_layout,
            self.wavefront_tone_pipeline_layout,
            self.wavefront_image_pipeline_layout,
            self.reconstruct_pipeline_layout,
            self.relax_prepare_pipeline_layout,
            self.relax_temporal_pipeline_layout,
            self.relax_atrous_pipeline_layout,
            self.relax_compose_pipeline_layout,
            self.indirect_reuse_clear_pipeline_layout,
            self.indirect_reuse_candidate_pipeline_layout,
            self.indirect_reuse_debug_pipeline_layout,
            self.bucket_intersect_pipeline_layout,
        ):
            if layout:
                vk.vkDestroyPipelineLayout(device, layout, None)
        if self.descriptor_pool:
            vk.vkDestroyDescriptorPool(device, self.descriptor_pool, None)
        for layout in (
            self.generate_layout, self.primary_layout,
            self.intersect_layout, self.shade_layout,
            self.resolve_layout,
            self.indirect_layout,
            self.wavefront_tone_layout,
            self.wavefront_image_layout,
            self.reconstruct_layout,
            self.relax_prepare_layout,
            self.relax_temporal_layout,
            self.relax_atrous_layout,
            self.relax_compose_layout,
            self.indirect_reuse_clear_layout,
            self.indirect_reuse_candidate_layout,
            self.indirect_reuse_debug_layout,
            self.bucket_intersect_layout,
        ):
            if layout:
                vk.vkDestroyDescriptorSetLayout(device, layout, None)


class VulkanSceneResources:
    """Owns one uploaded scene's buffers and acceleration structures."""

    def __init__(self, core, scene):
        self._core = core
        self.scene = scene
        self.scene_revision = scene.revision
        self.geometry_revision = scene.geometry_revision
        self.shading_revision = scene.shading_revision
        self.transform_revision = scene.transform_revision
        self.previous_transform_revision = scene.transform_revision
        self.texture_signature = tuple(id(item) for item in scene.textures)
        buffer_start = len(core._buffers)
        structure_start = len(core._structures)
        texture_start = len(core._sampled_textures)
        try:
            self.tlas = core._build_scene(scene)
        except Exception:
            core._release_resources(
                core._structures[structure_start:], core._buffers[buffer_start:]
            )
            core._release_sampled_textures(core._sampled_textures[texture_start:])
            del core._structures[structure_start:]
            del core._buffers[buffer_start:]
            del core._sampled_textures[texture_start:]
            raise
        self.vertex_buffer = core.scene_vertex_buffer
        self.previous_vertex_buffer = core.scene_previous_vertex_buffer
        positions = scene.render_triangles().reshape((-1, 3))
        self.vertex_data = np.ascontiguousarray(
            np.column_stack((positions, np.ones(len(positions), np.float32))),
            dtype=np.float32,
        )
        self.material_buffer = core.scene_material_buffer
        self.light_buffer = core.scene_light_buffer
        self.area_light_buffer = core.scene_area_light_buffer
        self.attribute_buffer = core.scene_attribute_buffer
        self.custom_attribute_buffer = core.scene_custom_attribute_buffer
        self.custom_attribute_layout = core.scene_custom_attribute_layout
        self.texture_buffer = core.scene_texture_buffer
        self.texture_binding_buffer = core.scene_texture_binding_buffer
        self.volume_header_buffer = core.scene_volume_header_buffer
        self.volume_scalar_buffer = core.scene_volume_scalar_buffer
        self.volume_transfer_buffer = core.scene_volume_transfer_buffer
        self.triangle_volume_buffer = core.scene_triangle_volume_buffer
        self.volume_empty_space_skipping = (
            core.scene_volume_empty_space_skipping
        )
        self.sampled_textures = tuple(core._sampled_textures[texture_start:])
        self.scene_sampled_textures = tuple(core.scene_sampled_textures)
        self.scene_sampled_volumes = tuple(core.scene_sampled_volumes)
        self.volume_signature = tuple(
            (id(volume), volume.shape, id(volume.material), volume.visible)
            for volume in scene.volumes
        )
        self.volume_data_revisions = tuple(
            volume.data_revision for volume in scene.visible_volumes
        )
        self.volume_dirty_counts = tuple(
            len(volume.dirty_regions) for volume in scene.visible_volumes
        )
        self.blases = tuple(core.scene_blases)
        self.instances = tuple(core.scene_instances)
        self.instance_buffer = core.scene_instance_buffer
        self._structures = core._structures[structure_start:]
        self._buffers = core._buffers[buffer_start:]
        self._sampled_textures = core._sampled_textures[texture_start:]
        del core._structures[structure_start:]
        del core._buffers[buffer_start:]
        del core._sampled_textures[texture_start:]

    def close(self):
        if self._core is None:
            return
        self._core._release_resources(self._structures, self._buffers)
        self._core._release_sampled_textures(self._sampled_textures)
        self._structures.clear()
        self._buffers.clear()
        self._sampled_textures.clear()
        self._core = None


class VulkanRayQueryCore:
    """Owns Vulkan resources for a minimal triangle ray-query renderer."""

    def _use_opaque_scene_specialization(self, scene):
        """Return whether the exact opaque-only shader variant is safe."""
        if scene.visible_volumes:
            return False
        if not self.config.wavefront_scene_specialization:
            return False
        from ...materials import (
            MaterialEvaluation, SCATTER_DIFFUSE, SCATTER_REFLECTION,
            SurfaceResponse, builtin_material,
        )
        default_program = self.config.material_program or builtin_material
        programs = scene.material_programs(default_program)
        # SurfaceResponse programs can synthesize transmission independently
        # of Material.transmission. Only accept a literal diffuse/reflection
        # event; dynamic expressions (including glass selects) stay generic.
        for program in programs:
            evaluation = program.evaluation
            if isinstance(evaluation, MaterialEvaluation):
                continue
            if not isinstance(evaluation, SurfaceResponse):
                return False
            try:
                event = float(evaluation.event.code)
            except (TypeError, ValueError):
                return False
            if event not in (SCATTER_DIFFUSE, SCATTER_REFLECTION):
                return False
        return all(
            mesh.material.transmission <= 0.001
            and mesh.material.transmission_texture is None
            for mesh in scene.visible_meshes
        )

    def __init__(
        self, device_name=None, glfw_window=None, config=None, *,
        external_instance=None, external_surface=None, headless_surface=False,
    ):
        if config is None:
            from .api import RendererConfig
            config = RendererConfig(device_name=device_name)
        elif device_name is not None:
            raise ValueError("Pass device_name or config, not both")
        self.config = config
        # Resources follow the immutable config; this gate permits matched
        # runtime A/B tests without reallocating them.
        self.wavefront_restir_runtime_enabled = bool(config.wavefront_restir_di)
        self.dynamic_resolution = None
        if config.wavefront_dynamic_resolution:
            from ...integrations.dynamic_resolution import (
                DynamicResolutionController,
            )
            self.dynamic_resolution = DynamicResolutionController(
                target_ms=config.wavefront_dynamic_target_ms,
                minimum_scale=config.wavefront_dynamic_min_scale,
                maximum_scale=config.wavefront_render_scale,
                current_scale=config.wavefront_render_scale,
            )
        self.interactive_dynamic_resolution = None
        if config.wavefront_interactive_target_fps is not None:
            from ...integrations.dynamic_resolution import (
                DynamicResolutionController,
            )
            interactive_max_scale = (
                config.wavefront_interactive_render_scale
                if config.wavefront_interactive_render_scale is not None
                else config.wavefront_render_scale
            )
            self.interactive_dynamic_resolution = DynamicResolutionController(
                target_ms=1000.0 / config.wavefront_interactive_target_fps,
                minimum_scale=config.wavefront_interactive_min_scale,
                maximum_scale=interactive_max_scale,
                current_scale=interactive_max_scale,
            )
        self.interactive_dynamic_samples = None
        if config.wavefront_interactive_sample_scaling:
            from ...integrations.dynamic_sampling import DynamicSampleController
            self.interactive_dynamic_samples = DynamicSampleController(
                target_ms=1000.0 / config.wavefront_interactive_target_fps,
                minimum_samples=config.wavefront_interactive_min_samples,
                maximum_samples=config.samples_per_pixel,
                current_samples=config.wavefront_interactive_min_samples,
            )
        self.glfw_window = glfw_window
        self._headless_surface = bool(headless_surface)
        if (external_instance is None) != (external_surface is None):
            raise ValueError(
                "external_instance and external_surface must be supplied together"
            )
        if glfw_window is not None and external_instance is not None:
            raise ValueError("Pass a GLFW window or an external Vulkan surface, not both")
        if self._headless_surface and (
            glfw_window is not None or external_instance is not None
        ):
            raise ValueError(
                "headless_surface cannot be combined with a window or external surface"
            )
        self._external_instance = external_instance
        self._external_surface = external_surface
        self._owns_instance = external_instance is None
        self._owns_surface = external_surface is None
        self.surface = None
        self.swapchain = None
        self.swapchain_images = []
        self.swapchain_image_views = []
        self.swapchain_direct_storage = False
        self.swapchain_bgra_storage = False
        self.formatless_storage_write_supported = False
        self.swapchain_extent = None
        self.swapchain_wavefront_only = None
        self.window_frames = []
        self.window_frame_index = 0
        self.wavefront_frame_sequence = 0
        self.timestamp_query_pool = None
        self.wavefront_timestamp_query_pool = None
        self.timestamp_period = 1.0
        self.present_mode_name = "FIFO"
        self.present_wait_supported = False
        self.present_pacing_enabled = False
        self.present_id = 0
        self.last_present_id = 0
        self.swapchain_image_count = 0
        self.swapchain_generation = 0
        self.accumulation_images = []
        self.accumulation_memories = []
        self.accumulation_views = []
        self.gbuffer_images = []
        self.gbuffer_memories = []
        self.gbuffer_views = []
        self.moment_images = []
        self.moment_memories = []
        self.moment_views = []
        self.denoise_images = []
        self.denoise_memories = []
        self.denoise_views = []
        self.accumulation_frame = 0
        self.accumulation_key = None
        self.accumulation_history_valid = False
        self.accumulation_state = AccumulationState.DISABLED
        self.accumulation_camera_signature = None
        self.accumulation_render_extent = None
        self.previous_camera = None
        self.wavefront_previous_present_camera = None
        self.camera_change_time = time.perf_counter()
        self.effective_samples_per_pixel = 1
        self.object_effect_bindings = ()
        self.scene_tlas = None
        self.scene_material_buffer = None
        self.scene_vertex_buffer = None
        self.scene_previous_vertex_buffer = None
        self.scene_light_buffer = None
        self.scene_area_light_buffer = None
        self.scene_attribute_buffer = None
        self.scene_custom_attribute_buffer = None
        self.scene_custom_attribute_layout = None
        self.scene_texture_buffer = None
        self.scene_texture_binding_buffer = None
        self.scene_volume_header_buffer = None
        self.scene_volume_scalar_buffer = None
        self.scene_volume_transfer_buffer = None
        self.scene_triangle_volume_buffer = None
        self.scene_volume_empty_space_skipping = False
        self.scene_sampled_textures = []
        self.scene_sampled_volumes = []
        self.scene_resources = None
        self.wavefront_executor = None
        self.instance = None
        self.device = None
        self.pipeline_cache = None
        self.pipeline_cache_path = None
        self.command_pool = None
        self.descriptor_pool = None
        self.descriptor_layout = None
        self.pipeline_layout = None
        self.pipeline = None
        self.shader_module = None
        self.tone_pipeline = None
        self.tone_shader_module = None
        self.denoise_pipeline = None
        self.denoise_shader_module = None
        self.nv12_descriptor_pool = None
        self.nv12_descriptor_layout = None
        self.nv12_pipeline_layout = None
        self.nv12_pipeline = None
        self.nv12_shader_module = None
        self.p010_pipeline = None
        self.p010_shader_module = None
        self.denoiser_output_enabled = config.denoiser_enabled
        self.material_programs = ()
        from ...pipeline import RenderPipeline, RenderStage
        stages = [RenderStage(
            "trace_temporal", reads={"scene", "history"}, writes={"history"},
            recorder=lambda context: context["record_trace"](context),
        )]
        tone_input = "history"
        if config.denoiser_enabled:
            stages.append(RenderStage(
                "denoise", reads={"history"}, writes={"filtered_hdr"},
                recorder=lambda context: context["record_denoise"](context),
            ))
            tone_input = "filtered_hdr"
        stages.extend((
            RenderStage(
                "tone_map", reads={tone_input}, writes={"output"},
                recorder=lambda context: context["record_tone"](context),
            ),
            RenderStage(
                "present", reads={"output", "swapchain"}, writes={"swapchain"},
                recorder=lambda context: context["record_present"](context),
            ),
        ))
        self.window_pipeline = RenderPipeline(
            stages, initial_resources={"scene", "history", "swapchain"}
        )
        self._buffers = []
        self._structures = []
        self._sampled_textures = []
        self.native_textures_supported = False
        self.pipeline_statistics_supported = False
        self.ray_pipeline_supported = False
        self.ray_pipeline_enabled = False
        self.ser_supported = False
        self.ser_reordering_supported = False
        self.ray_tracing_shader_group_handle_size = 0
        self.ray_tracing_shader_group_handle_alignment = 0
        self.ray_tracing_shader_group_base_alignment = 0
        self._closed = False
        self.last_timings = {}
        self.wavefront_last_frame_start = None
        self.wavefront_cadence_ms = 0.0
        try:
            self._create_instance_and_device(config.device_name)
            self._load_extension_functions()
            self._create_command_pool()
            self._create_pipeline()
        except Exception:
            self.close()
            raise

    @staticmethod
    def _name(properties):
        return str(properties.deviceName).split("\0", 1)[0]

    @property
    def resolved_execution_strategy(self):
        from .api import _resolve_execution_strategy
        if self.scene_resources is None:
            return (
                "wavefront" if self.config.wavefront_execution_strategy == "auto"
                else self.config.wavefront_execution_strategy
            )
        return _resolve_execution_strategy(
            self.config, self.scene_resources.scene
        )

    @property
    def native_textures_enabled(self):
        return bool(
            self.config.wavefront_native_textures
            and self.native_textures_supported
        )

    def _wavefront_pipeline_signature(self):
        if self.scene_resources is None:
            return (self.resolved_execution_strategy, False, False)
        scene = self.scene_resources.scene
        return (
            self.resolved_execution_strategy,
            self._use_opaque_scene_specialization(scene),
            bool(scene.textures),
        )

    def _replace_wavefront_executor_if_strategy_changed(self):
        executor = self.wavefront_executor
        if executor is None or (
            executor.strategy == self.resolved_execution_strategy
            and executor.scene_pipeline_signature
            == self._wavefront_pipeline_signature()
        ):
            return False
        # Submitted command buffers and external encoder consumers may still
        # reference executor-owned descriptors and pipelines.
        self._wait_external_releases()
        vk.vkDeviceWaitIdle(self.device)
        executor.close()
        self.wavefront_executor = None
        for frame in self.window_frames:
            frame["wavefront_command_key"] = None
        return True

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
            apiVersion=vk.VK_MAKE_VERSION(1, 2, 0),
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
                    index for index, family in enumerate(queue_families)
                    if family.queueFlags & vk.VK_QUEUE_COMPUTE_BIT
                    and (self.surface is None or self.get_surface_support(physical, index, self.surface))
                ),
                None,
            )
            if queue_index is None:
                continue
            candidates.append((properties.deviceType, physical, name, queue_index, extensions))
        if not candidates:
            raise RuntimeError("No hardware Vulkan adapter supports acceleration structures and ray queries")
        candidates.sort(key=lambda item: item[0] != vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        (
            _, self.physical_device, self.device_name, self.queue_family,
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
        if self.surface is not None and selected_properties.limits.maxPushConstantsSize < 160:
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
            self.ray_pipeline_supported = bool(
                queried_ray_pipeline.rayTracingPipeline
            )
        if self.ray_pipeline_supported and ser_extension in device_extensions:
            queried_ser = (
                vk.VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV()
            )
            vk.vkGetPhysicalDeviceFeatures2(
                self.physical_device,
                vk.VkPhysicalDeviceFeatures2(pNext=queried_ser),
            )
            self.ser_supported = bool(
                queried_ser.rayTracingInvocationReorder
            )
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
            ray_pipeline_properties = vk.VkPhysicalDeviceRayTracingPipelinePropertiesKHR()
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
            ser_features = (
                vk.VkPhysicalDeviceRayTracingInvocationReorderFeaturesNV(
                    pNext=feature_chain,
                    rayTracingInvocationReorder=vk.VK_TRUE,
                )
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
        pipeline_statistics_extension = (
            "VK_KHR_pipeline_executable_properties"
        )
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
                vk.VkPhysicalDeviceFeatures2(
                    pNext=queried_pipeline_statistics
                ),
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
            self.present_wait_supported
            and self.config.present_pacing
        )
        physical_features = vk.vkGetPhysicalDeviceFeatures(self.physical_device)
        self.formatless_storage_write_supported = bool(
            physical_features.shaderStorageImageWriteWithoutFormat
        )
        enabled_features = vk.VkPhysicalDeviceFeatures(
            shaderStorageImageWriteWithoutFormat=(
                vk.VK_TRUE
                if self.formatless_storage_write_supported
                else vk.VK_FALSE
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
        self.memory_properties = vk.vkGetPhysicalDeviceMemoryProperties(self.physical_device)
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
        initial_buffer = (
            vk.ffi.new("uint8_t[]", initial) if initial else None
        )
        create_info = vk.VkPipelineCacheCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
            initialDataSize=len(initial),
            pInitialData=(
                vk.ffi.cast("void *", initial_buffer)
                if initial_buffer is not None else None
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
        self.create_as = vk.vkGetDeviceProcAddr(self.device, "vkCreateAccelerationStructureKHR")
        self.destroy_as = vk.vkGetDeviceProcAddr(self.device, "vkDestroyAccelerationStructureKHR")
        self.build_as = vk.vkGetDeviceProcAddr(self.device, "vkCmdBuildAccelerationStructuresKHR")
        self.get_as_sizes = vk.vkGetDeviceProcAddr(self.device, "vkGetAccelerationStructureBuildSizesKHR")
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
            self.cmd_trace_rays = vk.ffi.cast(
                "PFN_vkCmdTraceRaysKHR", raw_trace_rays
            )
            for attribute, function_name in (
                ("create_ray_tracing_pipelines",
                 "vkCreateRayTracingPipelinesKHR"),
                ("get_ray_tracing_shader_group_handles",
                 "vkGetRayTracingShaderGroupHandlesKHR"),
            ):
                setattr(
                    self, attribute,
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

        raw_buffer_address = vk.lib.vkGetDeviceProcAddr(self.device, b"vkGetBufferDeviceAddress")
        self._raw_buffer_address = vk.ffi.cast("PFN_vkGetBufferDeviceAddress", raw_buffer_address)
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
            self.device, vk.ffi.addressof(pipeline_info), executable_count,
            vk.ffi.NULL,
        )
        properties = vk.ffi.new(
            "VkPipelineExecutablePropertiesKHR[]", executable_count[0]
        )
        for item in properties:
            item.sType = vk.VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR
        self.get_pipeline_executables(
            self.device, vk.ffi.addressof(pipeline_info), executable_count,
            properties,
        )
        results = []
        for index in range(executable_count[0]):
            executable_info = vk.VkPipelineExecutableInfoKHR(
                pipeline=pipeline, executableIndex=index
            )
            statistic_count = vk.ffi.new("uint32_t *")
            self.get_pipeline_statistics(
                self.device, vk.ffi.addressof(executable_info),
                statistic_count, vk.ffi.NULL,
            )
            statistics = vk.ffi.new(
                "VkPipelineExecutableStatisticKHR[]", statistic_count[0]
            )
            for statistic in statistics:
                statistic.sType = (
                    vk.VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR
                )
            self.get_pipeline_statistics(
                self.device, vk.ffi.addressof(executable_info),
                statistic_count, statistics,
            )
            values = {}
            for statistic in statistics:
                if statistic.format == vk.VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR:
                    value = bool(statistic.value.b32)
                elif statistic.format == vk.VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR:
                    value = int(statistic.value.i64)
                elif statistic.format == vk.VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR:
                    value = int(statistic.value.u64)
                else:
                    value = float(statistic.value.f64)
                name = vk.ffi.string(statistic.name).decode("utf-8")
                values[name] = value
            results.append({
                "name": vk.ffi.string(properties[index].name).decode("utf-8"),
                "description": vk.ffi.string(
                    properties[index].description
                ).decode("utf-8"),
                "statistics": values,
            })
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

    def _create_pipeline(self):
        image_output = self.surface is not None or self._headless_surface
        bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=0,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            vk.VkDescriptorSetLayoutBinding(
                binding=1,
                descriptorType=(
                    vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE
                    if image_output
                    else vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                ),
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            vk.VkDescriptorSetLayoutBinding(
                binding=2,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            vk.VkDescriptorSetLayoutBinding(
                binding=3,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
        ]
        if image_output:
            bindings.extend(
                vk.VkDescriptorSetLayoutBinding(
                    binding=binding,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    descriptorCount=1,
                    stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                )
                for binding in (4, 5, 6, 7, 8, 9)
            )
            bindings.extend(
                vk.VkDescriptorSetLayoutBinding(
                    binding=binding,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    descriptorCount=1,
                    stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                )
                for binding in (12, 13)
            )
        bindings.extend(
            vk.VkDescriptorSetLayoutBinding(
                binding=binding,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            )
            for binding in (10, 11, 14, 15)
        )
        self.descriptor_layout = vk.vkCreateDescriptorSetLayout(
            self.device,
            vk.VkDescriptorSetLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                bindingCount=len(bindings),
                pBindings=bindings,
            ),
            None,
        )
        push_range = vk.VkPushConstantRange(
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            offset=0,
            size=176 if image_output else 96,
        )
        self.pipeline_layout = vk.vkCreatePipelineLayout(
            self.device,
            vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=1,
                pSetLayouts=[self.descriptor_layout],
                pushConstantRangeCount=1,
                pPushConstantRanges=[push_range],
            ),
            None,
        )
        from ...materials import builtin_material
        initial_program = self.config.material_program or builtin_material
        if initial_program.required_attributes:
            # Attribute layouts are scene-derived. Install a compatible
            # placeholder until the first scene provides the declared channels.
            initial_program = builtin_material
        self._replace_compute_pipeline((initial_program,))
        if image_output:
            self.tone_shader_module, self.tone_pipeline = self._create_fixed_pipeline(
                "tone_map.comp"
            )
            if self.config.denoiser_enabled:
                self.denoise_shader_module, self.denoise_pipeline = (
                    self._create_fixed_pipeline("denoise_atrous.comp")
                )
            if self._headless_surface:
                self._create_nv12_pipeline()
        descriptor_count = WINDOW_FRAMES_IN_FLIGHT if image_output else 1
        pool_sizes = [
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                descriptorCount=descriptor_count,
            ),
            vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=descriptor_count * (6 if image_output else 7),
            ),
        ]
        if image_output:
            pool_sizes.append(vk.VkDescriptorPoolSize(
                type=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                descriptorCount=descriptor_count * 9,
            ))
        self.descriptor_pool = vk.vkCreateDescriptorPool(
            self.device,
            vk.VkDescriptorPoolCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                maxSets=descriptor_count,
                poolSizeCount=len(pool_sizes),
                pPoolSizes=pool_sizes,
            ),
            None,
        )

    def _create_nv12_pipeline(self):
        """Create the final RGBA8-to-pitch-linear NV12/P010 compute pass."""
        bindings = [
            vk.VkDescriptorSetLayoutBinding(
                binding=0,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
            vk.VkDescriptorSetLayoutBinding(
                binding=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                descriptorCount=1,
                stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            ),
        ]
        self.nv12_descriptor_layout = vk.vkCreateDescriptorSetLayout(
            self.device,
            vk.VkDescriptorSetLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                bindingCount=len(bindings), pBindings=bindings,
            ), None,
        )
        push_range = vk.VkPushConstantRange(
            stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT, offset=0, size=16,
        )
        self.nv12_pipeline_layout = vk.vkCreatePipelineLayout(
            self.device,
            vk.VkPipelineLayoutCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                setLayoutCount=1,
                pSetLayouts=[self.nv12_descriptor_layout],
                pushConstantRangeCount=1,
                pPushConstantRanges=[push_range],
            ), None,
        )
        shader_bytes = files("ordinarylight").joinpath(
            "shaders/rgba_to_nv12.comp.spv"
        ).read_bytes()
        self.nv12_shader_module = vk.vkCreateShaderModule(
            self.device,
            vk.VkShaderModuleCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize=len(shader_bytes), pCode=shader_bytes,
            ), None,
        )
        stage = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=self.nv12_shader_module, pName="main",
        )
        self.nv12_pipeline = vk.vkCreateComputePipelines(
            self.device, self.pipeline_cache or vk.VK_NULL_HANDLE, 1,
            [vk.VkComputePipelineCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                stage=stage, layout=self.nv12_pipeline_layout,
            )], None,
        )[0]
        p010_shader_bytes = files("ordinarylight").joinpath(
            "shaders/hdr_to_p010.comp.spv"
        ).read_bytes()
        self.p010_shader_module = vk.vkCreateShaderModule(
            self.device,
            vk.VkShaderModuleCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize=len(p010_shader_bytes), pCode=p010_shader_bytes,
            ), None,
        )
        p010_stage = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=self.p010_shader_module, pName="main",
        )
        self.p010_pipeline = vk.vkCreateComputePipelines(
            self.device, self.pipeline_cache or vk.VK_NULL_HANDLE, 1,
            [vk.VkComputePipelineCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                stage=p010_stage, layout=self.nv12_pipeline_layout,
            )], None,
        )[0]
        self.nv12_descriptor_pool = vk.vkCreateDescriptorPool(
            self.device,
            vk.VkDescriptorPoolCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                maxSets=WINDOW_FRAMES_IN_FLIGHT * 2,
                poolSizeCount=2,
                pPoolSizes=[
                    vk.VkDescriptorPoolSize(
                        type=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                        descriptorCount=WINDOW_FRAMES_IN_FLIGHT * 2,
                    ),
                    vk.VkDescriptorPoolSize(
                        type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        descriptorCount=WINDOW_FRAMES_IN_FLIGHT * 2,
                    ),
                ],
            ), None,
        )

    def _replace_compute_pipeline(self, programs, *, attribute_layout=None):
        """Install a compute pipeline dispatching the supplied material programs."""
        programs = tuple(programs)
        if programs == self.material_programs:
            return
        from ...materials import builtin_material
        shader_source_name = (
            "ray_query_image.comp"
            if self.surface is not None or self._headless_surface
            else "ray_query.comp"
        )
        if (
            len(programs) == 1 and programs[0] is builtin_material
            and self.config.material_modifier is None
        ):
            shader_bytes = files("ordinarylight").joinpath(
                f"shaders/{shader_source_name}.spv"
            ).read_bytes()
        else:
            from ...shaders.compiler import compile_material_shader
            shader_bytes = compile_material_shader(
                shader_source_name, programs,
                attribute_layout=attribute_layout,
                material_modifier=self.config.material_modifier,
            )
        replacement_module = vk.vkCreateShaderModule(
            self.device,
            vk.VkShaderModuleCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize=len(shader_bytes),
                pCode=shader_bytes,
            ),
            None,
        )
        stage = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=replacement_module,
            pName="main",
        )
        try:
            replacement_pipeline = vk.vkCreateComputePipelines(
                self.device, self.pipeline_cache or vk.VK_NULL_HANDLE, 1,
                [vk.VkComputePipelineCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                    stage=stage, layout=self.pipeline_layout,
                )], None,
            )[0]
        except Exception:
            vk.vkDestroyShaderModule(self.device, replacement_module, None)
            raise
        previous_pipeline = self.pipeline
        previous_module = self.shader_module
        self.pipeline = replacement_pipeline
        self.shader_module = replacement_module
        self.material_programs = programs
        if previous_pipeline:
            vk.vkDestroyPipeline(self.device, previous_pipeline, None)
        if previous_module:
            vk.vkDestroyShaderModule(self.device, previous_module, None)

    def _create_fixed_pipeline(self, shader_source_name):
        shader_bytes = files("ordinarylight").joinpath(
            f"shaders/{shader_source_name}.spv"
        ).read_bytes()
        module = vk.vkCreateShaderModule(
            self.device,
            vk.VkShaderModuleCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                codeSize=len(shader_bytes), pCode=shader_bytes,
            ),
            None,
        )
        stage = vk.VkPipelineShaderStageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=module, pName="main",
        )
        pipeline = vk.vkCreateComputePipelines(
            self.device, self.pipeline_cache or vk.VK_NULL_HANDLE, 1,
            [vk.VkComputePipelineCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                stage=stage, layout=self.pipeline_layout,
            )], None,
        )[0]
        return module, pipeline

    def _ensure_scene_pipeline(self, scene):
        from ...materials import builtin_material
        default_program = self.config.material_program or builtin_material
        programs = scene.material_programs(default_program)
        if programs != self.material_programs:
            vk.vkDeviceWaitIdle(self.device)
            self._replace_compute_pipeline(
                programs,
                attribute_layout=self._material_attribute_layout(
                    scene, programs, self.config.material_modifier
                ),
            )
        return programs, default_program

    @staticmethod
    def _material_attribute_layout(scene, programs, material_modifier=None):
        """Return the opt-in custom vertex ABI required by material programs."""
        requirements = {}
        ordered_names = []
        for program in programs:
            for name, components in program.required_attributes:
                previous = requirements.get(name)
                if previous is not None and previous != components:
                    raise ValueError(
                        f"attribute {name!r} has conflicting material declarations"
                    )
                if previous is None:
                    ordered_names.append(name)
                requirements[name] = components
        from ...scene import VertexAttributeLayout
        if not ordered_names:
            from ...materials import builtin_material
            return (
                None if (
                    all(program is builtin_material for program in programs)
                    and material_modifier is None
                )
                else VertexAttributeLayout(())
            )
        layout = VertexAttributeLayout.from_scene(scene, ordered_names)
        for name, components in layout.channels:
            if requirements[name] != components:
                raise ValueError(
                    f"mesh attribute {name!r} has {components} components; "
                    f"material requires {requirements[name]}"
                )
        return layout

    def _memory_type(self, bits, flags):
        for index in range(self.memory_properties.memoryTypeCount):
            memory_type = self.memory_properties.memoryTypes[index]
            if bits & (1 << index) and memory_type.propertyFlags & flags == flags:
                return index
        raise RuntimeError(f"No Vulkan memory type satisfies flags {flags:#x}")

    def _create_buffer(self, size, usage, memory_flags, data=None, device_address=False):
        buffer = vk.vkCreateBuffer(
            self.device,
            vk.VkBufferCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                size=size,
                usage=usage,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            ),
            None,
        )
        requirements = vk.vkGetBufferMemoryRequirements(self.device, buffer)
        allocation_flags = None
        if device_address:
            allocation_flags = vk.VkMemoryAllocateFlagsInfo(
                sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
                flags=MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT,
            )
        try:
            memory = vk.vkAllocateMemory(
                self.device,
                vk.VkMemoryAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    pNext=allocation_flags,
                    allocationSize=requirements.size,
                    memoryTypeIndex=self._memory_type(
                        requirements.memoryTypeBits, memory_flags),
                ),
                None,
            )
        except vk.VkErrorOutOfDeviceMemory:
            vk.vkDestroyBuffer(self.device, buffer, None)
            raise
        vk.vkBindBufferMemory(self.device, buffer, memory, 0)
        result = Buffer(buffer, memory, size)
        self._buffers.append(result)
        if data is not None:
            payload = memoryview(data).cast("B") if not isinstance(data, bytes) else data
            mapped = vk.vkMapMemory(self.device, memory, 0, len(payload), 0)
            mapped[:] = payload
            vk.vkUnmapMemory(self.device, memory)
        return result

    def _create_exportable_buffer(self, size, usage, memory_flags):
        """Create a dedicated opaque-FD exportable buffer allocation."""
        external_info = vk.VkExternalMemoryBufferCreateInfo(
            handleTypes=vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
        )
        buffer = vk.vkCreateBuffer(
            self.device,
            vk.VkBufferCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                pNext=external_info,
                size=size, usage=usage,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            ), None,
        )
        requirements = vk.vkGetBufferMemoryRequirements(self.device, buffer)
        export_info = vk.VkExportMemoryAllocateInfo(
            handleTypes=vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
        )
        dedicated_info = vk.VkMemoryDedicatedAllocateInfo(
            pNext=export_info, image=vk.VK_NULL_HANDLE, buffer=buffer,
        )
        try:
            memory = vk.vkAllocateMemory(
                self.device,
                vk.VkMemoryAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                    pNext=dedicated_info,
                    allocationSize=requirements.size,
                    memoryTypeIndex=self._memory_type(
                        requirements.memoryTypeBits, memory_flags,
                    ),
                ), None,
            )
        except Exception:
            vk.vkDestroyBuffer(self.device, buffer, None)
            raise
        vk.vkBindBufferMemory(self.device, buffer, memory, 0)
        result = Buffer(buffer, memory, int(requirements.size))
        self._buffers.append(result)
        return result

    def _create_uploaded_device_buffer(
        self, data, usage, *, device_address=False,
    ):
        """Stage immutable data into shader-fast device-local storage."""
        payload = np.ascontiguousarray(data)
        staging = self._create_buffer(
            payload.nbytes,
            vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data=payload,
        )
        destination = self._create_buffer(
            payload.nbytes,
            usage | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            device_address=device_address,
        )
        self._single_use(lambda command: vk.vkCmdCopyBuffer(
            command, staging.buffer, destination.buffer, 1,
            [vk.VkBufferCopy(srcOffset=0, dstOffset=0, size=payload.nbytes)],
        ))
        vk.vkDestroyBuffer(self.device, staging.buffer, None)
        vk.vkFreeMemory(self.device, staging.memory, None)
        self._buffers.remove(staging)
        return destination

    def _update_device_buffers(self, updates):
        """Replace equal-sized device-local buffer contents in one submission."""
        prepared = []
        for destination, data in updates:
            payload = np.ascontiguousarray(data)
            if payload.nbytes != destination.size:
                raise ValueError("updated buffer data must retain its byte size")
            staging = self._create_buffer(
                payload.nbytes, vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                data=payload,
            )
            prepared.append((staging, destination, payload.nbytes))
        try:
            self._single_use(lambda command: [
                vk.vkCmdCopyBuffer(
                    command, staging.buffer, destination.buffer, 1,
                    [vk.VkBufferCopy(srcOffset=0, dstOffset=0, size=size)],
                )
                for staging, destination, size in prepared
            ])
        finally:
            for staging, _destination, _size in prepared:
                vk.vkDestroyBuffer(self.device, staging.buffer, None)
                vk.vkFreeMemory(self.device, staging.memory, None)
                self._buffers.remove(staging)

    def _update_device_buffer_regions(self, destination, regions):
        """Replace byte ranges of one device-local buffer in one submission."""
        prepared = []
        for offset, data in regions:
            payload = np.ascontiguousarray(data)
            offset = int(offset)
            if offset < 0 or offset + payload.nbytes > destination.size:
                raise ValueError("updated buffer region lies outside destination")
            staging = self._create_buffer(
                payload.nbytes, vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                data=payload,
            )
            prepared.append((staging, offset, payload.nbytes))
        try:
            self._single_use(lambda command: [
                vk.vkCmdCopyBuffer(
                    command, staging.buffer, destination.buffer, 1,
                    [vk.VkBufferCopy(
                        srcOffset=0, dstOffset=offset, size=size,
                    )],
                )
                for staging, offset, size in prepared
            ])
        finally:
            for staging, _offset, _size in prepared:
                vk.vkDestroyBuffer(self.device, staging.buffer, None)
                vk.vkFreeMemory(self.device, staging.memory, None)
                self._buffers.remove(staging)

    def _update_sampled_volume_regions(self, texture, volume, regions):
        """Upload z/y/x boxes into an existing shader-readable 3-D image."""
        prepared = []
        normalized = volume.normalized_data
        for offset, shape in regions:
            stop = tuple(start + size for start, size in zip(offset, shape))
            payload = np.ascontiguousarray(normalized[
                offset[0]:stop[0], offset[1]:stop[1], offset[2]:stop[2]
            ], np.float32)
            staging = self._create_buffer(
                payload.nbytes, vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                data=payload,
            )
            prepared.append((staging, offset, shape))
        subresource = vk.VkImageSubresourceRange(
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1,
        )

        def upload(command):
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, None, 0, None, 1,
                [vk.VkImageMemoryBarrier(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                    dstAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    oldLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    image=texture.image, subresourceRange=subresource,
                )],
            )
            for staging, offset, shape in prepared:
                vk.vkCmdCopyBufferToImage(
                    command, staging.buffer, texture.image,
                    vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                    [vk.VkBufferImageCopy(
                        bufferOffset=0, bufferRowLength=0, bufferImageHeight=0,
                        imageSubresource=vk.VkImageSubresourceLayers(
                            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                            mipLevel=0, baseArrayLayer=0, layerCount=1,
                        ),
                        imageOffset=vk.VkOffset3D(
                            x=offset[2], y=offset[1], z=offset[0],
                        ),
                        imageExtent=vk.VkExtent3D(
                            width=shape[2], height=shape[1], depth=shape[0],
                        ),
                    )],
                )
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                0, None, 0, None, 1,
                [vk.VkImageMemoryBarrier(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                    oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    newLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    image=texture.image, subresourceRange=subresource,
                )],
            )

        try:
            self._single_use(upload)
        finally:
            for staging, _offset, _shape in prepared:
                vk.vkDestroyBuffer(self.device, staging.buffer, None)
                vk.vkFreeMemory(self.device, staging.memory, None)
                self._buffers.remove(staging)

    def _create_sampled_texture(self, levels, texture, image_format):
        """Upload a complete RGBA8 mip pyramid to an optimal sampled image."""
        height, width, _ = levels[0].shape
        image = vk.vkCreateImage(
            self.device,
            vk.VkImageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                imageType=vk.VK_IMAGE_TYPE_2D,
                format=image_format,
                extent=vk.VkExtent3D(width=width, height=height, depth=1),
                mipLevels=len(levels), arrayLayers=1,
                samples=vk.VK_SAMPLE_COUNT_1_BIT,
                tiling=vk.VK_IMAGE_TILING_OPTIMAL,
                usage=(vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT
                       | vk.VK_IMAGE_USAGE_SAMPLED_BIT),
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
                initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            ), None,
        )
        requirements = vk.vkGetImageMemoryRequirements(self.device, image)
        memory = vk.vkAllocateMemory(
            self.device,
            vk.VkMemoryAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize=requirements.size,
                memoryTypeIndex=self._memory_type(
                    requirements.memoryTypeBits,
                    vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                ),
            ), None,
        )
        vk.vkBindImageMemory(self.device, image, memory, 0)
        offsets = []
        chunks = []
        offset = 0
        for level in levels:
            payload = np.ascontiguousarray(level, dtype=np.uint8)
            offsets.append(offset)
            chunks.append(payload.reshape(-1))
            offset += payload.nbytes
        packed = np.concatenate(chunks)
        staging = self._create_buffer(
            packed.nbytes, vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data=packed,
        )
        subresource = vk.VkImageSubresourceRange(
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0, levelCount=len(levels),
            baseArrayLayer=0, layerCount=1,
        )
        regions = [vk.VkBufferImageCopy(
            bufferOffset=offsets[index],
            bufferRowLength=0, bufferImageHeight=0,
            imageSubresource=vk.VkImageSubresourceLayers(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                mipLevel=index, baseArrayLayer=0, layerCount=1,
            ),
            imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
            imageExtent=vk.VkExtent3D(
                width=level.shape[1], height=level.shape[0], depth=1
            ),
        ) for index, level in enumerate(levels)]

        def upload(command):
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, None, 0, None, 1,
                [vk.VkImageMemoryBarrier(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=0,
                    dstAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    oldLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
                    newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    image=image, subresourceRange=subresource,
                )],
            )
            vk.vkCmdCopyBufferToImage(
                command, staging.buffer, image,
                vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                len(regions), regions,
            )
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                0, None, 0, None, 1,
                [vk.VkImageMemoryBarrier(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                    oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    newLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    image=image, subresourceRange=subresource,
                )],
            )

        self._single_use(upload)
        vk.vkDestroyBuffer(self.device, staging.buffer, None)
        vk.vkFreeMemory(self.device, staging.memory, None)
        self._buffers.remove(staging)
        view = vk.vkCreateImageView(
            self.device,
            vk.VkImageViewCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                image=image, viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                format=image_format,
                subresourceRange=subresource,
            ), None,
        )
        address_modes = {
            "repeat": vk.VK_SAMPLER_ADDRESS_MODE_REPEAT,
            "clamp": vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            "mirror": vk.VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
        }
        filtering = (
            vk.VK_FILTER_LINEAR if texture.linear_filter
            else vk.VK_FILTER_NEAREST
        )
        sampler = vk.vkCreateSampler(
            self.device,
            vk.VkSamplerCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                magFilter=filtering, minFilter=filtering,
                mipmapMode=(vk.VK_SAMPLER_MIPMAP_MODE_LINEAR
                            if texture.linear_filter
                            else vk.VK_SAMPLER_MIPMAP_MODE_NEAREST),
                addressModeU=address_modes[texture.wrap_s],
                addressModeV=address_modes[texture.wrap_t],
                addressModeW=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                minLod=0.0, maxLod=float(len(levels) - 1),
                maxAnisotropy=1.0,
            ), None,
        )
        result = SampledTexture(image, memory, view, sampler)
        self._sampled_textures.append(result)
        return result

    def _create_sampled_volume(self, data):
        """Upload one float32 scalar field to a linearly sampled 3D image."""
        payload = np.ascontiguousarray(data, dtype=np.float32)
        depth, height, width = payload.shape
        image = vk.vkCreateImage(
            self.device, vk.VkImageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                imageType=vk.VK_IMAGE_TYPE_3D,
                format=vk.VK_FORMAT_R32_SFLOAT,
                extent=vk.VkExtent3D(width=width, height=height, depth=depth),
                mipLevels=1, arrayLayers=1, samples=vk.VK_SAMPLE_COUNT_1_BIT,
                tiling=vk.VK_IMAGE_TILING_OPTIMAL,
                usage=(vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT
                       | vk.VK_IMAGE_USAGE_SAMPLED_BIT),
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
                initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            ), None,
        )
        requirements = vk.vkGetImageMemoryRequirements(self.device, image)
        memory = vk.vkAllocateMemory(
            self.device, vk.VkMemoryAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                allocationSize=requirements.size,
                memoryTypeIndex=self._memory_type(
                    requirements.memoryTypeBits,
                    vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                ),
            ), None,
        )
        vk.vkBindImageMemory(self.device, image, memory, 0)
        staging = self._create_buffer(
            payload.nbytes, vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
            | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data=payload,
        )
        subresource = vk.VkImageSubresourceRange(
            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1,
        )

        def upload(command):
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, None, 0, None, 1,
                [vk.VkImageMemoryBarrier(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=0, dstAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    oldLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
                    newLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    image=image, subresourceRange=subresource,
                )],
            )
            vk.vkCmdCopyBufferToImage(
                command, staging.buffer, image,
                vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                [vk.VkBufferImageCopy(
                    bufferOffset=0, bufferRowLength=0, bufferImageHeight=0,
                    imageSubresource=vk.VkImageSubresourceLayers(
                        aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                        mipLevel=0, baseArrayLayer=0, layerCount=1,
                    ),
                    imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
                    imageExtent=vk.VkExtent3D(
                        width=width, height=height, depth=depth,
                    ),
                )],
            )
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                0, None, 0, None, 1,
                [vk.VkImageMemoryBarrier(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
                    srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                    dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT,
                    oldLayout=vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    newLayout=vk.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                    srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                    image=image, subresourceRange=subresource,
                )],
            )

        self._single_use(upload)
        vk.vkDestroyBuffer(self.device, staging.buffer, None)
        vk.vkFreeMemory(self.device, staging.memory, None)
        self._buffers.remove(staging)
        view = vk.vkCreateImageView(
            self.device, vk.VkImageViewCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                image=image, viewType=vk.VK_IMAGE_VIEW_TYPE_3D,
                format=vk.VK_FORMAT_R32_SFLOAT,
                subresourceRange=subresource,
            ), None,
        )
        sampler = vk.vkCreateSampler(
            self.device, vk.VkSamplerCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                magFilter=vk.VK_FILTER_LINEAR, minFilter=vk.VK_FILTER_LINEAR,
                mipmapMode=vk.VK_SAMPLER_MIPMAP_MODE_NEAREST,
                addressModeU=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                addressModeV=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                addressModeW=vk.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                minLod=0.0, maxLod=0.0, maxAnisotropy=1.0,
            ), None,
        )
        result = SampledTexture(image, memory, view, sampler)
        self._sampled_textures.append(result)
        return result

    def _release_sampled_textures(self, textures):
        for texture in reversed(textures):
            vk.vkDestroySampler(self.device, texture.sampler, None)
            vk.vkDestroyImageView(self.device, texture.view, None)
            vk.vkDestroyImage(self.device, texture.image, None)
            vk.vkFreeMemory(self.device, texture.memory, None)

    def _buffer_address(self, buffer):
        info = vk.VkBufferDeviceAddressInfo(
            sType=vk.VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            buffer=buffer.buffer,
        )
        return self._raw_buffer_address(self.device, vk.ffi.addressof(info))

    def _as_address(self, structure):
        info = vk.VkAccelerationStructureDeviceAddressInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
            accelerationStructure=structure.handle,
        )
        return self._raw_as_address(self.device, vk.ffi.addressof(info))

    def _single_use(self, record):
        command = vk.vkAllocateCommandBuffers(
            self.device,
            vk.VkCommandBufferAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                commandPool=self.command_pool,
                level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1,
            ),
        )[0]
        vk.vkBeginCommandBuffer(
            command,
            vk.VkCommandBufferBeginInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            ),
        )
        record(command)
        vk.vkEndCommandBuffer(command)
        vk.vkQueueSubmit(
            self.queue,
            1,
            [
                vk.VkSubmitInfo(
                    sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    commandBufferCount=1,
                    pCommandBuffers=[command],
                )
            ],
            vk.VK_NULL_HANDLE,
        )
        vk.vkQueueWaitIdle(self.queue)
        vk.vkFreeCommandBuffers(self.device, self.command_pool, 1, [command])

    def _make_as(
        self, geometry, primitive_count, structure_type, *, allow_update=False,
    ):
        flags = vk.VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
        if allow_update:
            flags |= vk.VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR
        build = vk.VkAccelerationStructureBuildGeometryInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            type=structure_type,
            flags=flags,
            mode=vk.VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
            geometryCount=1,
            pGeometries=[geometry],
        )
        sizes = vk.VkAccelerationStructureBuildSizesInfoKHR(
            sType=vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR
        )
        self.get_as_sizes(
            self.device,
            vk.VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            build,
            [primitive_count],
            sizes,
        )
        storage = self._create_buffer(
            sizes.accelerationStructureSize,
            vk.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        )
        handle = self.create_as(
            self.device,
            vk.VkAccelerationStructureCreateInfoKHR(
                sType=vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
                buffer=storage.buffer,
                size=sizes.accelerationStructureSize,
                type=structure_type,
            ),
            None,
        )
        scratch = self._create_buffer(
            max(sizes.buildScratchSize, sizes.updateScratchSize),
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            | BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            device_address=True,
        )
        build.dstAccelerationStructure = handle
        build.scratchData.deviceAddress = self._buffer_address(scratch)
        range_info = vk.VkAccelerationStructureBuildRangeInfoKHR(
            primitiveCount=primitive_count,
            primitiveOffset=0,
            firstVertex=0,
            transformOffset=0,
        )
        range_pointer = vk.ffi.addressof(range_info)
        ranges = vk.ffi.new(
            "VkAccelerationStructureBuildRangeInfoKHR*[]", [range_pointer]
        )
        self._single_use(lambda command: self.build_as(command, 1, [build], ranges))
        result = AccelerationStructure(handle, storage, scratch)
        self._structures.append(result)
        return result

    def _scene_instance_bytes(self, instances):
        return b"".join(
            struct.pack(
                "<12fIIQ",
                *np.asarray(item.mesh.transform.matrix[:3], np.float32).reshape(-1),
                (int(item.visibility_mask) << 24) | int(item.triangle_offset),
                0,
                self._as_address(item.blas.structure),
            )
            for item in instances
        )

    @staticmethod
    def _mesh_blas_vertices(mesh):
        positions = mesh.vertices[mesh.indices].reshape((-1, 3))
        return np.ascontiguousarray(
            np.column_stack((
                positions, np.ones(len(positions), dtype=np.float32),
            )),
            dtype=np.float32,
        )

    def _refit_scene_blases(self, entries):
        """Update equal-topology BLAS bounds without reallocating them."""
        builds = []
        ranges = []
        range_storage = []
        for item in entries:
            mesh = item.mesh
            triangle_data = vk.VkAccelerationStructureGeometryTrianglesDataKHR(
                sType=(
                    vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR
                ),
                vertexFormat=vk.VK_FORMAT_R32G32B32_SFLOAT,
                vertexData=vk.VkDeviceOrHostAddressConstKHR(
                    deviceAddress=self._buffer_address(item.vertex_buffer)
                ),
                vertexStride=16,
                maxVertex=len(mesh.indices) * 3 - 1,
                indexType=vk.VK_INDEX_TYPE_NONE_KHR,
            )
            geometry = vk.VkAccelerationStructureGeometryKHR(
                sType=vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                geometryType=vk.VK_GEOMETRY_TYPE_TRIANGLES_KHR,
                geometry=vk.VkAccelerationStructureGeometryDataKHR(
                    triangles=triangle_data
                ),
                flags=vk.VK_GEOMETRY_OPAQUE_BIT_KHR,
            )
            build = vk.VkAccelerationStructureBuildGeometryInfoKHR(
                sType=(
                    vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR
                ),
                type=vk.VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                flags=(
                    vk.VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR
                    | vk.VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR
                ),
                mode=vk.VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR,
                srcAccelerationStructure=item.structure.handle,
                dstAccelerationStructure=item.structure.handle,
                geometryCount=1,
                pGeometries=[geometry],
            )
            build.scratchData.deviceAddress = self._buffer_address(
                item.structure.scratch
            )
            range_info = vk.VkAccelerationStructureBuildRangeInfoKHR(
                primitiveCount=len(mesh.indices), primitiveOffset=0,
                firstVertex=0, transformOffset=0,
            )
            range_storage.append(range_info)
            range_pointer = vk.ffi.addressof(range_info)
            ranges.append(range_pointer)
            builds.append(build)
        if builds:
            self._single_use(
                lambda command: self.build_as(
                    command, len(builds), builds, ranges
                )
            )

    def _tlas_geometry(self, instance_buffer):
        instance_data = vk.VkAccelerationStructureGeometryInstancesDataKHR(
            sType=(
                vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR
            ),
            arrayOfPointers=vk.VK_FALSE,
            data=vk.VkDeviceOrHostAddressConstKHR(
                deviceAddress=self._buffer_address(instance_buffer)
            ),
        )
        return vk.VkAccelerationStructureGeometryKHR(
            sType=vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            geometryType=vk.VK_GEOMETRY_TYPE_INSTANCES_KHR,
            geometry=vk.VkAccelerationStructureGeometryDataKHR(
                instances=instance_data
            ),
        )

    def _rebuild_scene_tlas(self, tlas, instance_buffer, instance_count):
        """Rebuild an equal-sized TLAS after instance transforms change."""
        geometry = self._tlas_geometry(instance_buffer)
        build = vk.VkAccelerationStructureBuildGeometryInfoKHR(
            sType=(
                vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR
            ),
            type=vk.VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
            flags=vk.VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
            mode=vk.VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR,
            dstAccelerationStructure=tlas.handle,
            geometryCount=1,
            pGeometries=[geometry],
        )
        build.scratchData.deviceAddress = self._buffer_address(tlas.scratch)
        range_info = vk.VkAccelerationStructureBuildRangeInfoKHR(
            primitiveCount=instance_count,
            primitiveOffset=0, firstVertex=0, transformOffset=0,
        )
        range_pointer = vk.ffi.addressof(range_info)
        ranges = vk.ffi.new(
            "VkAccelerationStructureBuildRangeInfoKHR*[]", [range_pointer]
        )
        self._single_use(
            lambda command: self.build_as(command, 1, [build], ranges)
        )

    def _build_scene(self, scene):
        stage_start = time.perf_counter()
        programs, default_program = self._ensure_scene_pipeline(scene)
        custom_attribute_layout = self._material_attribute_layout(
            scene, programs, self.config.material_modifier
        )
        triangles = scene.render_triangles()
        if not len(triangles):
            raise ValueError("Vulkan ray-query rendering requires at least one triangle")
        positions = triangles.reshape((-1, 3))
        vertices = np.ascontiguousarray(
            np.column_stack((positions, np.ones(len(positions), dtype=np.float32))),
            dtype=np.float32,
        )
        vertex_buffer = self._create_uploaded_device_buffer(
            vertices,
            vk.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
            | BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
            | vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            device_address=True,
        )
        self.scene_vertex_buffer = vertex_buffer
        self.scene_previous_vertex_buffer = self._create_uploaded_device_buffer(
            vertices, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )
        material_data = scene.triangle_material_data(programs, default_program)
        self.scene_material_buffer = self._create_uploaded_device_buffer(
            material_data,
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )
        light_data = scene.analytic_light_data()
        self.scene_light_buffer = self._create_uploaded_device_buffer(
            light_data,
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )
        area_light_data = scene.emissive_triangle_data()
        self.scene_area_light_buffer = self._create_uploaded_device_buffer(
            area_light_data,
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )
        attribute_data = scene.triangle_attribute_data()
        self.scene_attribute_buffer = self._create_uploaded_device_buffer(
            attribute_data,
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )
        self.scene_custom_attribute_layout = custom_attribute_layout
        self.scene_custom_attribute_buffer = None
        if custom_attribute_layout is not None:
            custom_attribute_data = custom_attribute_layout.pack(scene)
            if custom_attribute_data.nbytes == 0:
                custom_attribute_data = np.zeros(4, np.float32)
            self.scene_custom_attribute_buffer = (
                self._create_uploaded_device_buffer(
                    custom_attribute_data,
                    vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                )
            )
        # Native shaders only consult word zero for bounds checking. Avoid
        # building and uploading a duplicate packed mip pyramid in that mode.
        texture_data = (
            np.asarray([len(scene.textures)], dtype=np.uint32)
            if self.native_textures_enabled
            else scene.texture_data()
        )
        if self.config.wavefront_device_local_textures:
            self.scene_texture_buffer = self._create_uploaded_device_buffer(
                texture_data, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
            )
        else:
            self.scene_texture_buffer = self._create_buffer(
                texture_data.nbytes,
                vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                data=texture_data,
            )
        texture_binding_data = scene.texture_binding_data()
        self.scene_texture_binding_buffer = self._create_uploaded_device_buffer(
            texture_binding_data,
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )
        from ...volume import pack_volumes
        volume_headers, volume_scalars, volume_transfers = pack_volumes(
            scene.visible_volumes,
            empty_space_skipping=self.config.volume_empty_space_skipping,
        )
        self.scene_volume_empty_space_skipping = bool(
            np.any(volume_headers["acceleration_parameters"][:, 1:] > 0)
        )
        self.scene_volume_header_buffer = self._create_uploaded_device_buffer(
            volume_headers, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )
        self.scene_volume_scalar_buffer = self._create_uploaded_device_buffer(
            volume_scalars, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )
        self.scene_volume_transfer_buffer = self._create_uploaded_device_buffer(
            volume_transfers, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )
        self.scene_triangle_volume_buffer = self._create_uploaded_device_buffer(
            scene.triangle_volume_indices(),
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        )
        if len(scene.visible_volumes) > MAX_NATIVE_VOLUMES:
            raise ValueError(
                f"Vulkan backend supports at most {MAX_NATIVE_VOLUMES} visible volumes"
            )
        volume_payloads = [
            volume.normalized_data for volume in scene.visible_volumes
        ] or [np.zeros((2, 2, 2), np.float32)]
        self.scene_sampled_volumes = [
            self._create_sampled_volume(payload) for payload in volume_payloads
        ]
        self.scene_sampled_textures = []
        if self.native_textures_enabled:
            if len(scene.textures) > MAX_NATIVE_TEXTURES:
                raise ValueError(
                    f"Native texture backend supports at most {MAX_NATIVE_TEXTURES} "
                    f"textures; scene contains {len(scene.textures)}"
                )
            textures = scene.textures
            if not textures:
                from ...scene import Texture
                textures = (Texture(np.full((1, 1, 4), 255, np.uint8)),)
            for texture in textures:
                self.scene_sampled_textures.extend((
                    self._create_sampled_texture(
                        scene._texture_mips(texture.pixels, srgb=True), texture,
                        vk.VK_FORMAT_R8G8B8A8_SRGB,
                    ),
                    self._create_sampled_texture(
                        scene._texture_mips(texture.pixels, srgb=False), texture,
                        vk.VK_FORMAT_R8G8B8A8_UNORM,
                    ),
                ))
        self.last_timings["scene_upload_ms"] = (
            time.perf_counter() - stage_start
        ) * 1000.0
        stage_start = time.perf_counter()
        blases = []
        instances = []
        blas_by_geometry = {}
        triangle_offset = 0
        for mesh in scene.render_meshes:
            mesh_triangles = mesh.vertices[mesh.indices]
            if not len(mesh_triangles):
                continue
            if triangle_offset + len(mesh_triangles) > (1 << 24):
                raise ValueError(
                    "scene triangle offsets exceed Vulkan's 24-bit instance "
                    "custom index"
                )
            geometry_source = mesh.resource or mesh
            blas_entry = blas_by_geometry.get(id(geometry_source))
            if blas_entry is None:
                object_vertices = self._mesh_blas_vertices(geometry_source)
                blas_vertices = self._create_uploaded_device_buffer(
                    object_vertices,
                    vk.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
                    | BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    device_address=True,
                )
                triangle_data = vk.VkAccelerationStructureGeometryTrianglesDataKHR(
                    sType=(
                        vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR
                    ),
                    vertexFormat=vk.VK_FORMAT_R32G32B32_SFLOAT,
                    vertexData=vk.VkDeviceOrHostAddressConstKHR(
                        deviceAddress=self._buffer_address(blas_vertices)
                    ),
                    vertexStride=16,
                    maxVertex=len(object_vertices) - 1,
                    indexType=vk.VK_INDEX_TYPE_NONE_KHR,
                )
                geometry = vk.VkAccelerationStructureGeometryKHR(
                    sType=vk.VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                    geometryType=vk.VK_GEOMETRY_TYPE_TRIANGLES_KHR,
                    geometry=vk.VkAccelerationStructureGeometryDataKHR(
                        triangles=triangle_data
                    ),
                    flags=vk.VK_GEOMETRY_OPAQUE_BIT_KHR,
                )
                blas = self._make_as(
                    geometry, len(mesh_triangles),
                    vk.VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
                    allow_update=geometry_source.deformable,
                )
                blas_entry = SceneBlas(blas, geometry_source, blas_vertices)
                blas_by_geometry[id(geometry_source)] = blas_entry
                blases.append(blas_entry)
            instances.append(SceneTlasInstance(
                mesh, blas_entry, triangle_offset
            ))
            triangle_offset += len(mesh_triangles)
        self.scene_blases = blases
        self.scene_instances = instances
        self.last_timings["blas_count"] = len(blases)
        self.last_timings["instance_count"] = len(instances)
        self.last_timings["shared_blas_savings"] = max(
            0, len(instances) - len(blases)
        )
        self.last_timings["blas_ms"] = (time.perf_counter() - stage_start) * 1000.0

        stage_start = time.perf_counter()
        instance_bytes = self._scene_instance_bytes(instances)
        instance_buffer = self._create_buffer(
            len(instance_bytes),
            vk.VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR
            | BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            data=instance_bytes,
            device_address=True,
        )
        self.scene_instance_buffer = instance_buffer
        tlas_geometry = self._tlas_geometry(instance_buffer)
        tlas = self._make_as(
            tlas_geometry, len(instances),
            vk.VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR
        )
        self.last_timings["tlas_ms"] = (time.perf_counter() - stage_start) * 1000.0
        return tlas

    @staticmethod
    def _camera_constants(
        camera, width, height, overlay_fps=None, max_bounces=5, samples=1,
        accumulation_frame=0, previous_camera=None, temporal_history=False,
        history_valid=False, temporal_history_limit=32,
        temporal_neighborhood_clamping=True, adaptive_sampling=False,
        adaptive_variance_threshold=0.0025, adaptive_min_samples=1,
        light_count=0, area_light_count=0, area_light_weight=0.0,
        area_light_samples=1, denoiser_enabled=False,
        denoiser_variance_threshold=0.01,
    ):
        position, forward, right, up = _camera_vectors(camera)
        if not isinstance(camera, PanoramicCamera):
            right *= width / height
        include_previous = previous_camera is not None
        previous_camera = previous_camera or camera
        previous_position, previous_forward, previous_right, previous_up = (
            _camera_vectors(previous_camera)
        )
        if not isinstance(previous_camera, PanoramicCamera):
            previous_right *= width / height
        constants = [
                (*position, float(width)),
                (*forward, float(height)),
                (*right, float(denoiser_variance_threshold)),
                (*up, float(_camera_projection(camera))),
                (
                    float(overlay_fps or 0.0),
                    float(overlay_fps is not None),
                    float(max(1, min(int(max_bounces), 16))),
                    float(max(1, min(int(samples), 64))),
                ),
                (
                    float(accumulation_frame),
                    float(temporal_history),
                    float(history_valid) * (
                        1.0 if temporal_neighborhood_clamping else 2.0
                    ),
                    float(temporal_history_limit),
                ),
        ]
        if include_previous:
            constants.extend((
                (*previous_position, float(adaptive_sampling)),
                (*previous_forward, float(adaptive_variance_threshold)),
                (*previous_right, float(adaptive_min_samples)),
                (*previous_up, float(light_count)),
                (float(area_light_count), float(area_light_weight),
                 float(area_light_samples), float(denoiser_enabled)),
            ))
        return np.asarray(constants, dtype=np.float32)

    def render(self, scene, camera, width, height, samples=1, max_bounces=1):
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be positive")
        total_start = time.perf_counter()
        self.last_timings = {}
        tlas = self._build_scene(scene)
        stage_start = time.perf_counter()
        # The shader packs normalized RGBA into one uint32 per pixel.  This
        # keeps readback at 4 bytes/pixel and needs no CPU color conversion.
        output_size = width * height * 4
        output = self._create_buffer(
            output_size,
            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        descriptor_set = vk.vkAllocateDescriptorSets(
            self.device,
            vk.VkDescriptorSetAllocateInfo(
                sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                descriptorPool=self.descriptor_pool,
                descriptorSetCount=1,
                pSetLayouts=[self.descriptor_layout],
            ),
        )[0]
        as_descriptor = vk.VkWriteDescriptorSetAccelerationStructureKHR(
            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
            accelerationStructureCount=1,
            pAccelerationStructures=[tlas.handle],
        )
        buffer_info = vk.VkDescriptorBufferInfo(
            buffer=output.buffer,
            offset=0,
            range=output_size,
        )
        material_info = vk.VkDescriptorBufferInfo(
            buffer=self.scene_material_buffer.buffer,
            offset=0,
            range=self.scene_material_buffer.size,
        )
        vertex_info = vk.VkDescriptorBufferInfo(
            buffer=self.scene_vertex_buffer.buffer,
            offset=0,
            range=self.scene_vertex_buffer.size,
        )
        light_info = vk.VkDescriptorBufferInfo(
            buffer=self.scene_light_buffer.buffer,
            offset=0,
            range=self.scene_light_buffer.size,
        )
        area_light_info = vk.VkDescriptorBufferInfo(
            buffer=self.scene_area_light_buffer.buffer,
            offset=0,
            range=self.scene_area_light_buffer.size,
        )
        attribute_info = vk.VkDescriptorBufferInfo(
            buffer=self.scene_attribute_buffer.buffer,
            offset=0,
            range=self.scene_attribute_buffer.size,
        )
        writes = [
            vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                pNext=as_descriptor,
                dstSet=descriptor_set,
                dstBinding=0,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
            ),
            vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set,
                dstBinding=1,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[buffer_info],
            ),
            vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set,
                dstBinding=2,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[material_info],
            ),
            vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set,
                dstBinding=3,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[vertex_info],
            ),
            vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set,
                dstBinding=10,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[light_info],
            ),
            vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set,
                dstBinding=11,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[area_light_info],
            ),
            vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set,
                dstBinding=14,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[attribute_info],
            ),
        ]
        if self.scene_custom_attribute_buffer is not None:
            custom_attribute_info = vk.VkDescriptorBufferInfo(
                buffer=self.scene_custom_attribute_buffer.buffer,
                offset=0,
                range=self.scene_custom_attribute_buffer.size,
            )
            writes.append(vk.VkWriteDescriptorSet(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                dstSet=descriptor_set,
                dstBinding=15,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=[custom_attribute_info],
            ))
        vk.vkUpdateDescriptorSets(self.device, len(writes), writes, 0, None)
        constants = self._camera_constants(
            camera, width, height, max_bounces=max_bounces, samples=samples
        )
        self.last_timings["output_setup_ms"] = (
            time.perf_counter() - stage_start
        ) * 1000.0

        def record(command):
            vk.vkCmdBindPipeline(
                command, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline
            )
            vk.vkCmdBindDescriptorSets(
                command,
                vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                self.pipeline_layout,
                0,
                1,
                [descriptor_set],
                0,
                None,
            )
            vk.vkCmdPushConstants(
                command,
                self.pipeline_layout,
                vk.VK_SHADER_STAGE_COMPUTE_BIT,
                0,
                constants.nbytes,
                vk.ffi.from_buffer(constants),
            )
            vk.vkCmdDispatch(command, (width + 7) // 8, (height + 7) // 8, 1)

        stage_start = time.perf_counter()
        self._single_use(record)
        self.last_timings["dispatch_ms"] = (time.perf_counter() - stage_start) * 1000.0
        stage_start = time.perf_counter()
        mapped = vk.vkMapMemory(self.device, output.memory, 0, output_size, 0)
        rgba = np.frombuffer(mapped, dtype=np.uint8).copy().reshape((height, width, 4))
        vk.vkUnmapMemory(self.device, output.memory)
        self.last_timings["readback_copy_ms"] = (
            time.perf_counter() - stage_start
        ) * 1000.0
        stage_start = time.perf_counter()
        vk.vkResetDescriptorPool(self.device, self.descriptor_pool, 0)
        self._release_frame_resources()
        self.last_timings["cleanup_ms"] = (time.perf_counter() - stage_start) * 1000.0
        self.last_timings["total_ms"] = (time.perf_counter() - total_start) * 1000.0
        return rgba

    def _release_frame_resources(self):
        self._release_resources(self._structures, self._buffers)
        self._release_sampled_textures(self._sampled_textures)
        self._structures.clear()
        self._buffers.clear()
        self._sampled_textures.clear()
        self.scene_sampled_textures = []

    def _release_resources(self, structures, buffers):
        for structure in reversed(structures):
            self.destroy_as(self.device, structure.handle, None)
        for buffer in reversed(buffers):
            vk.vkDestroyBuffer(self.device, buffer.buffer, None)
            vk.vkFreeMemory(self.device, buffer.memory, None)

    def prepare_window_scene(self, scene):
        """Build persistent acceleration structures for direct presentation."""
        if self.surface is None and not self._headless_surface:
            raise RuntimeError("This Vulkan core was not created with a GLFW window")
        if (
            self.scene_resources is None
            or self.scene_resources.scene is not scene
            or self.scene_resources.scene_revision != scene.revision
        ):
            self.upload_window_scene(scene)

    def trace_wavefront_tile(
        self, camera, width, height, *, tile_origin=(0, 0), tile_extent=None,
        frame_index=0, sample_index=0, sample_count=1,
        output_image_slot=None, readback=True,
        command=None, camera_vectors=None, camera_slot=0, upload_camera=True,
        timestamp=None, restir_history_valid_override=None,
        restir_history_limit=None, primary_hit_readback=False,
    ):
        """Execute the experimental primary-generation and intersection stages."""
        if self.scene_tlas is None or self.scene_vertex_buffer is None:
            raise RuntimeError("upload a scene before tracing a wavefront tile")
        width, height = int(width), int(height)
        origin_x, origin_y = map(int, tile_origin)
        if width < 1 or height < 1:
            raise ValueError("wavefront image extent must be positive")
        if origin_x < 0 or origin_y < 0 or origin_x >= width or origin_y >= height:
            raise ValueError("wavefront tile origin must be inside the image")
        if tile_extent is None:
            remaining = self.config.wavefront_tile_capacity
            tile_width = min(width - origin_x, remaining)
            tile_height = min(height - origin_y, max(1, remaining // tile_width))
        else:
            tile_width, tile_height = map(int, tile_extent)
        if tile_width < 1 or tile_height < 1:
            raise ValueError("wavefront tile extent must be positive")
        if origin_x + tile_width > width or origin_y + tile_height > height:
            raise ValueError("wavefront tile must fit inside the image")
        if tile_width * tile_height > self.config.wavefront_tile_capacity:
            raise ValueError("wavefront tile exceeds wavefront_tile_capacity")
        self._replace_wavefront_executor_if_strategy_changed()
        if self.wavefront_executor is None:
            self.wavefront_executor = VulkanWavefrontExecutor(
                self, self.config.wavefront_tile_capacity
            )
            for slot, frame in enumerate(self.window_frames):
                if frame.get("image_view"):
                    self.wavefront_executor.bind_output_image(
                        slot, frame["wavefront_hdr_view"],
                        frame["wavefront_position_view"],
                        frame["wavefront_normal_view"],
                        frame["wavefront_material_view"], frame["image_view"],
                    )
        self.wavefront_executor.ensure_custom_material_pipelines()

        if camera_vectors is None:
            camera_vectors = _camera_vectors(camera)
        position, forward, right, up = camera_vectors
        if upload_camera:
            self.wavefront_executor.update_camera(
                camera_slot, camera_vectors, projection=_camera_projection(camera)
            )
        constants = bytearray(struct.pack(
            "8I",
            width, height, origin_x, origin_y,
            tile_width, tile_height, int(sample_count), int(sample_index),
        ))
        dispatched = self.wavefront_executor.dispatch(
            constants, tile_width, tile_height,
            output_image_slot=output_image_slot, image_width=width,
            image_height=height,
            sample_index=sample_index,
            sample_count=sample_count,
            camera_slot=camera_slot,
            readback=readback, command=command,
            timestamp=timestamp,
            primary_hit_readback=primary_hit_readback,
            restir_history_valid_override=restir_history_valid_override,
            restir_history_limit=restir_history_limit,
        )
        if not readback:
            return None
        ray, hit, continuation, resolved, packed = dispatched[:5]
        radiance = np.zeros((tile_height, tile_width, 4), dtype=np.float32)
        pixel_indices = resolved["metadata"][:, 0].astype(np.int64)
        local_x = pixel_indices % width - origin_x
        local_y = pixel_indices // width - origin_y
        valid = (
            (local_x >= 0) & (local_x < tile_width)
            & (local_y >= 0) & (local_y < tile_height)
        )
        radiance[local_y[valid], local_x[valid]] = resolved["radiance"][valid]
        rgba8 = np.zeros((tile_height, tile_width, 4), dtype=np.uint8)
        channels = np.column_stack((
            packed & 0xff,
            (packed >> 8) & 0xff,
            (packed >> 16) & 0xff,
            (packed >> 24) & 0xff,
        )).astype(np.uint8)
        rgba8[local_y[valid], local_x[valid]] = channels[valid]
        result = {
            "tile_origin": (origin_x, origin_y),
            "tile_extent": (tile_width, tile_height),
            "ray_queue": ray,
            "hit_queue": hit,
            "continuation_queue": continuation,
            "bounces_executed": self.config.max_bounces,
            "radiance": radiance,
            "rgba8": rgba8,
        }
        if primary_hit_readback:
            primary_hits = dispatched[5]
            depth = np.full((tile_height, tile_width), np.inf, np.float32)
            normal = np.zeros((tile_height, tile_width, 3), np.float32)
            primitive = np.full(
                (tile_height, tile_width), np.uint32(0xffffffff), np.uint32,
            )
            position = np.zeros((tile_height, tile_width, 3), np.float32)
            barycentric = np.zeros((tile_height, tile_width, 2), np.float32)
            path_indices = primary_hits["path_index"].astype(np.int64)
            hit_valid = (
                (primary_hits["position_t"][:, 3] >= 0.0)
                & (path_indices >= 0)
                & (path_indices < tile_width * tile_height)
            )
            local_hit_y = path_indices[hit_valid] // tile_width
            local_hit_x = path_indices[hit_valid] % tile_width
            depth[local_hit_y, local_hit_x] = primary_hits[
                "position_t"
            ][hit_valid, 3]
            normal[local_hit_y, local_hit_x] = primary_hits[
                "geometric_normal"
            ][hit_valid]
            primitive[local_hit_y, local_hit_x] = primary_hits[
                "primitive_index"
            ][hit_valid]
            position[local_hit_y, local_hit_x] = primary_hits[
                "position_t"
            ][hit_valid, :3]
            barycentric[local_hit_y, local_hit_x] = primary_hits[
                "barycentrics"
            ][hit_valid]
            result["depth"] = depth
            result["normal"] = normal
            result["primitive_id"] = primitive
            result["primary_position"] = position
            result["primary_barycentric"] = barycentric
        if self.config.denoiser_signal_capture:
            signal_index = 6 if primary_hit_readback else 5
            result["denoiser_path_signals"] = dispatched[signal_index]
        return result

    def upload_window_scene(self, scene):
        """Transactionally replace the resident scene without rebuilding output.

        The historical name is retained for compatibility, but persistent scene
        resources are also used by offscreen wavefront validation renders.  A
        replacement preserves the Vulkan device, pipelines, swapchain/headless
        output images, and exported video surfaces.  The previous scene remains
        valid until its replacement has been built successfully.
        """
        if (
            self.scene_resources is not None
            and self.scene_resources.scene is scene
            and self.scene_resources.scene_revision == scene.revision
        ):
            resources = self.scene_resources
            if resources.previous_transform_revision != scene.transform_revision:
                # The first frame after a transform consumes the preceding
                # vertex snapshot. Once that frame is complete, converge the
                # snapshot so a newly stationary object produces zero motion.
                vk.vkDeviceWaitIdle(self.device)
                self._update_device_buffers(((
                    resources.previous_vertex_buffer,
                    resources.vertex_data,
                ),))
                resources.previous_transform_revision = scene.transform_revision
            return
        if self._try_update_window_scene(scene):
            return
        vk.vkDeviceWaitIdle(self.device)
        previous = self.scene_resources
        scene_state = self._capture_scene_state()
        previous_programs = self.material_programs
        previous_attribute_layout = (
            previous.custom_attribute_layout if previous is not None else None
        )
        try:
            replacement = VulkanSceneResources(self, scene)
        except Exception:
            self._restore_scene_state(scene_state)
            if self.material_programs != previous_programs:
                self._replace_compute_pipeline(
                    previous_programs,
                    attribute_layout=previous_attribute_layout,
                )
            raise
        self._activate_scene_resources(replacement)
        self._invalidate_scene_history()
        if previous is not None:
            previous.close()

    _SCENE_STATE_NAMES = (
        "scene_tlas", "scene_vertex_buffer", "scene_previous_vertex_buffer",
        "scene_material_buffer",
        "scene_light_buffer", "scene_area_light_buffer",
        "scene_attribute_buffer", "scene_custom_attribute_buffer",
        "scene_custom_attribute_layout", "scene_texture_buffer",
        "scene_texture_binding_buffer", "scene_volume_header_buffer",
        "scene_volume_scalar_buffer", "scene_volume_transfer_buffer",
        "scene_triangle_volume_buffer", "scene_volume_empty_space_skipping",
        "scene_sampled_textures", "scene_sampled_volumes", "scene_blases",
        "scene_instances", "scene_instance_buffer",
    )

    def _capture_scene_state(self):
        return {
            name: getattr(self, name, None) for name in self._SCENE_STATE_NAMES
        }

    def _restore_scene_state(self, state):
        for name, value in state.items():
            setattr(self, name, value)

    def _activate_scene_resources(self, resources):
        self.scene_resources = resources
        self.scene_tlas = resources.tlas
        self.scene_vertex_buffer = resources.vertex_buffer
        self.scene_previous_vertex_buffer = resources.previous_vertex_buffer
        self.scene_material_buffer = resources.material_buffer
        self.scene_light_buffer = resources.light_buffer
        self.scene_area_light_buffer = resources.area_light_buffer
        self.scene_attribute_buffer = resources.attribute_buffer
        self.scene_custom_attribute_buffer = resources.custom_attribute_buffer
        self.scene_custom_attribute_layout = resources.custom_attribute_layout
        self.scene_texture_buffer = resources.texture_buffer
        self.scene_texture_binding_buffer = resources.texture_binding_buffer
        self.scene_volume_header_buffer = resources.volume_header_buffer
        self.scene_volume_scalar_buffer = resources.volume_scalar_buffer
        self.scene_volume_transfer_buffer = resources.volume_transfer_buffer
        self.scene_triangle_volume_buffer = resources.triangle_volume_buffer
        self.scene_volume_empty_space_skipping = (
            resources.volume_empty_space_skipping
        )
        self.scene_sampled_textures = list(resources.scene_sampled_textures)
        self.scene_sampled_volumes = list(resources.scene_sampled_volumes)
        self.scene_blases = list(resources.blases)
        self.scene_instances = list(resources.instances)
        self.scene_instance_buffer = resources.instance_buffer

    def _invalidate_scene_history(self):
        self.reset_accumulation()
        self.wavefront_previous_present_camera = None
        if self.wavefront_executor is not None:
            self.wavefront_executor._bound_scene_key = None
        for frame in self.window_frames:
            frame["wavefront_command_key"] = None
            frame["wavefront_history_valid"] = False
            frame["wavefront_relax_history_valid"] = False
            frame["wavefront_reservoir_valid"] = False
            frame["wavefront_indirect_reservoir_valid"] = False

    def _try_update_window_scene(self, scene):
        """Update compatible resident buffers and acceleration structures."""
        resources = self.scene_resources
        if resources is None or resources.scene is not scene:
            return False
        if scene.volumes:
            return self._try_update_volume_payloads(scene, resources)
        if resources.texture_signature != tuple(id(item) for item in scene.textures):
            return False
        geometry_changed = (
            resources.geometry_revision != scene.geometry_revision
        )
        shading_changed = resources.shading_revision != scene.shading_revision
        transform_changed = (
            resources.transform_revision != scene.transform_revision
        )
        changed_blases = []
        nonempty_meshes = [
            mesh for mesh in scene.visible_meshes if len(mesh.indices)
        ]
        if len(nonempty_meshes) != len(resources.instances):
            return False
        if any(
            mesh is not item.mesh
            for mesh, item in zip(nonempty_meshes, resources.instances)
        ):
            return False
        unique_geometry = []
        seen_geometry = set()
        for mesh in nonempty_meshes:
            source = mesh.resource or mesh
            if id(source) not in seen_geometry:
                seen_geometry.add(id(source))
                unique_geometry.append(source)
        if (
            len(unique_geometry) != len(resources.blases)
            or any(
                source is not item.mesh
                for source, item in zip(unique_geometry, resources.blases)
            )
        ):
            return False
        if geometry_changed:
            for mesh, item in zip(unique_geometry, resources.blases):
                if (
                    item.indices.shape != mesh.indices.shape
                    or not np.array_equal(item.indices, mesh.indices)
                    or item.vertices.shape != mesh.vertices.shape
                ):
                    return False
                if not np.array_equal(item.vertices, mesh.vertices):
                    if not mesh.deformable:
                        return False
                    changed_blases.append(item)
        programs, default_program = self._ensure_scene_pipeline(scene)
        custom_attribute_layout = self._material_attribute_layout(
            scene, programs, self.config.material_modifier
        )
        if custom_attribute_layout != resources.custom_attribute_layout:
            return False
        updates = [
            (resources.material_buffer,
             scene.triangle_material_data(programs, default_program)),
            (resources.light_buffer, scene.analytic_light_data()),
            (resources.area_light_buffer, scene.emissive_triangle_data()),
            (resources.attribute_buffer, scene.triangle_attribute_data()),
            (resources.texture_binding_buffer, scene.texture_binding_data()),
        ]
        if resources.custom_attribute_buffer is not None:
            custom_attribute_data = custom_attribute_layout.pack(scene)
            if custom_attribute_data.nbytes == 0:
                custom_attribute_data = np.zeros(4, np.float32)
            updates.append((
                resources.custom_attribute_buffer,
                custom_attribute_data,
            ))
        if geometry_changed or transform_changed:
            triangles, _colors, _emissions = scene.triangles()
            positions = triangles.reshape((-1, 3))
            vertices = np.ascontiguousarray(
                np.column_stack((
                    positions, np.ones(len(positions), dtype=np.float32),
                )), dtype=np.float32,
            )
            # Preserve the exact world-space geometry used by the preceding
            # frame. ReLAX reconstructs the old surface point from the stable
            # primitive index and barycentrics written by the path tracer.
            updates.append((resources.previous_vertex_buffer,
                            resources.vertex_data))
            updates.append((resources.vertex_buffer, vertices))
        for item in changed_blases:
            updates.append((
                item.vertex_buffer, self._mesh_blas_vertices(item.mesh)
            ))
        if any(np.ascontiguousarray(data).nbytes != buffer.size
               for buffer, data in updates):
            return False
        start = time.perf_counter()
        vk.vkDeviceWaitIdle(self.device)
        self._update_device_buffers(updates)
        if changed_blases:
            self._refit_scene_blases(changed_blases)
        if transform_changed:
            instance_bytes = self._scene_instance_bytes(resources.instances)
            if len(instance_bytes) != resources.instance_buffer.size:
                return False
            mapped = vk.vkMapMemory(
                self.device, resources.instance_buffer.memory,
                0, len(instance_bytes), 0,
            )
            mapped[:] = instance_bytes
            vk.vkUnmapMemory(self.device, resources.instance_buffer.memory)
            resources.transform_revision = scene.transform_revision
        if changed_blases or transform_changed:
            self._rebuild_scene_tlas(
                resources.tlas, resources.instance_buffer,
                len(resources.instances),
            )
        if geometry_changed:
            for item in resources.blases:
                item.indices = item.mesh.indices.copy()
                item.vertices = item.mesh.vertices.copy()
            resources.geometry_revision = scene.geometry_revision
        if geometry_changed or transform_changed:
            resources.vertex_data = vertices.copy()
            resources.previous_transform_revision = (
                scene.transform_revision - 1 if transform_changed
                else scene.transform_revision
            )
        resources.shading_revision = scene.shading_revision
        resources.scene_revision = scene.revision
        self.last_timings["blas_count"] = len(resources.blases)
        self.last_timings["instance_count"] = len(resources.instances)
        self.last_timings["shared_blas_savings"] = max(
            0, len(resources.instances) - len(resources.blases)
        )
        motion_only = transform_changed and not geometry_changed and not shading_changed
        self.reset_accumulation()
        for frame in self.window_frames:
            frame["wavefront_history_valid"] = False
            if not motion_only:
                frame["wavefront_relax_history_valid"] = False
            frame["wavefront_reservoir_valid"] = False
            frame["wavefront_indirect_reservoir_valid"] = False
            frame["wavefront_command_key"] = None
        self.last_timings["scene_partial_upload_ms"] = (
            time.perf_counter() - start
        ) * 1000.0
        return True

    def _try_update_volume_payloads(self, scene, resources):
        """Update same-layout volume voxels without rebuilding scene geometry."""
        signature = tuple(
            (id(volume), volume.shape, id(volume.material), volume.visible)
            for volume in scene.volumes
        )
        if signature != resources.volume_signature:
            return False
        if (
            resources.geometry_revision != scene.geometry_revision
            or resources.transform_revision != scene.transform_revision
            or resources.texture_signature != tuple(id(item) for item in scene.textures)
            or resources.volume_empty_space_skipping
        ):
            return False
        volumes = scene.visible_volumes
        revisions = tuple(volume.data_revision for volume in volumes)
        if revisions == resources.volume_data_revisions:
            return False
        if len(volumes) != len(resources.scene_sampled_volumes):
            return False
        dirty_counts = tuple(len(volume.dirty_regions) for volume in volumes)
        changed = []
        for index, volume in enumerate(volumes):
            if revisions[index] == resources.volume_data_revisions[index]:
                continue
            start = resources.volume_dirty_counts[index]
            regions = volume.dirty_regions[start:]
            if not regions:
                return False
            changed.append((index, volume, regions))

        # The packed scalar buffer is z-major. Emit one contiguous x span per
        # affected row, avoiding upload of unchanged voxels and other volumes.
        from ...volume import pack_volumes
        headers, _scalars, _transfers = pack_volumes(
            volumes, empty_space_skipping=False,
        )
        buffer_regions = []
        for index, volume, regions in changed:
            scalar_base = int(headers[index]["dimensions_offset"][3])
            _depth, height, width = volume.shape
            normalized = volume.normalized_data
            for offset, shape in regions:
                for local_z in range(shape[0]):
                    z = offset[0] + local_z
                    for local_y in range(shape[1]):
                        y = offset[1] + local_y
                        x = offset[2]
                        element_offset = scalar_base + (z * height + y) * width + x
                        row = normalized[z, y, x:x + shape[2]]
                        buffer_regions.append((element_offset * 4, row))

        vk.vkDeviceWaitIdle(self.device)
        self._update_device_buffer_regions(
            resources.volume_scalar_buffer, buffer_regions,
        )
        for index, volume, regions in changed:
            self._update_sampled_volume_regions(
                resources.scene_sampled_volumes[index], volume, regions,
            )
        resources.volume_data_revisions = revisions
        resources.volume_dirty_counts = dirty_counts
        resources.shading_revision = scene.shading_revision
        resources.scene_revision = scene.revision
        self._invalidate_scene_history()
        return True

    def _destroy_swapchain_resources(self):
        if self.device is None:
            return
        for frame in self.window_frames:
            nv12_buffer = frame.get("nv12_buffer")
            if nv12_buffer is not None:
                vk.vkDestroyBuffer(self.device, nv12_buffer.buffer, None)
                vk.vkFreeMemory(self.device, nv12_buffer.memory, None)
                if nv12_buffer in self._buffers:
                    self._buffers.remove(nv12_buffer)
                frame["nv12_buffer"] = None
            capture = frame.get("wavefront_hdr_capture_buffer")
            if capture is not None:
                vk.vkDestroyBuffer(self.device, capture.buffer, None)
                vk.vkFreeMemory(self.device, capture.memory, None)
                if capture in self._buffers:
                    self._buffers.remove(capture)
                frame["wavefront_hdr_capture_buffer"] = None
            reservoir = frame.get("wavefront_reservoir_buffer")
            if reservoir is not None:
                vk.vkDestroyBuffer(self.device, reservoir.buffer, None)
                vk.vkFreeMemory(self.device, reservoir.memory, None)
                if reservoir in self._buffers:
                    self._buffers.remove(reservoir)
                frame["wavefront_reservoir_buffer"] = None
            indirect_reservoir = frame.get("wavefront_indirect_reservoir_buffer")
            if indirect_reservoir is not None:
                vk.vkDestroyBuffer(
                    self.device, indirect_reservoir.buffer, None)
                vk.vkFreeMemory(
                    self.device, indirect_reservoir.memory, None)
                if indirect_reservoir in self._buffers:
                    self._buffers.remove(indirect_reservoir)
                frame["wavefront_indirect_reservoir_buffer"] = None
            indirect_seed = frame.get("wavefront_indirect_seed_buffer")
            if indirect_seed is not None:
                vk.vkDestroyBuffer(self.device, indirect_seed.buffer, None)
                vk.vkFreeMemory(self.device, indirect_seed.memory, None)
                if indirect_seed in self._buffers:
                    self._buffers.remove(indirect_seed)
                frame["wavefront_indirect_seed_buffer"] = None
            frame["wavefront_indirect_reservoir_extent"] = None
            if frame.get("wavefront_hdr_view"):
                vk.vkDestroyImageView(
                    self.device, frame["wavefront_hdr_view"], None
                )
                frame["wavefront_hdr_view"] = None
            if frame.get("wavefront_hdr_image"):
                vk.vkDestroyImage(
                    self.device, frame["wavefront_hdr_image"], None
                )
                frame["wavefront_hdr_image"] = None
            if frame.get("wavefront_hdr_memory"):
                vk.vkFreeMemory(
                    self.device, frame["wavefront_hdr_memory"], None
                )
                frame["wavefront_hdr_memory"] = None
            for prefix in (
                "wavefront_position", "wavefront_normal",
                "wavefront_material", "wavefront_history_color",
                "wavefront_relax_diffuse", "wavefront_relax_specular",
                "wavefront_relax_normal_roughness",
                "wavefront_relax_view_z", "wavefront_relax_motion",
                "wavefront_relax_identity",
                "wavefront_relax_temporal_diffuse",
                "wavefront_relax_temporal_specular",
                "wavefront_relax_atrous_diffuse",
                "wavefront_relax_atrous_specular",
                "wavefront_relax_diffuse_history",
                "wavefront_relax_specular_history",
            ):
                if frame.get(f"{prefix}_view"):
                    vk.vkDestroyImageView(
                        self.device, frame[f"{prefix}_view"], None
                    )
                    frame[f"{prefix}_view"] = None
                if frame.get(f"{prefix}_image"):
                    vk.vkDestroyImage(
                        self.device, frame[f"{prefix}_image"], None
                    )
                    frame[f"{prefix}_image"] = None
                if frame.get(f"{prefix}_memory"):
                    vk.vkFreeMemory(
                        self.device, frame[f"{prefix}_memory"], None
                    )
                    frame[f"{prefix}_memory"] = None
            if frame.get("image_view"):
                vk.vkDestroyImageView(self.device, frame["image_view"], None)
                frame["image_view"] = None
            if frame.get("image"):
                vk.vkDestroyImage(self.device, frame["image"], None)
                frame["image"] = None
            if frame.get("image_memory"):
                vk.vkFreeMemory(self.device, frame["image_memory"], None)
                frame["image_memory"] = None
        for view in (
            self.accumulation_views + self.gbuffer_views + self.moment_views
            + self.denoise_views
        ):
            vk.vkDestroyImageView(self.device, view, None)
        for image in (
            self.accumulation_images + self.gbuffer_images + self.moment_images
            + self.denoise_images
        ):
            vk.vkDestroyImage(self.device, image, None)
        for memory in (
            self.accumulation_memories + self.gbuffer_memories + self.moment_memories
            + self.denoise_memories
        ):
            vk.vkFreeMemory(self.device, memory, None)
        self.accumulation_views.clear()
        self.accumulation_images.clear()
        self.accumulation_memories.clear()
        self.gbuffer_views.clear()
        self.gbuffer_images.clear()
        self.gbuffer_memories.clear()
        self.moment_views.clear()
        self.moment_images.clear()
        self.moment_memories.clear()
        self.denoise_views.clear()
        self.denoise_images.clear()
        self.denoise_memories.clear()
        self.reset_accumulation()
        for view in self.swapchain_image_views:
            vk.vkDestroyImageView(self.device, view, None)
        self.swapchain_image_views = []
        if self.swapchain:
            self.destroy_swapchain(self.device, self.swapchain, None)
            self.swapchain = None
        self.swapchain_images = []
        self.swapchain_direct_storage = False
        self.swapchain_bgra_storage = False
        self.swapchain_extent = None
        self.swapchain_wavefront_only = None
        self.wavefront_previous_present_camera = None

    def _image_barrier(self, image, old_layout, new_layout, src_access, dst_access):
        return vk.VkImageMemoryBarrier(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
            srcAccessMask=src_access,
            dstAccessMask=dst_access,
            oldLayout=old_layout,
            newLayout=new_layout,
            srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
            image=image,
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                baseMipLevel=0,
                levelCount=1,
                baseArrayLayer=0,
                layerCount=1,
            ),
        )

    def _reset_wavefront_history_chain(self):
        """Return the cross-frame history semaphores to an unsignaled state."""
        if not self.window_frames:
            return
        semaphore_info = vk.VkSemaphoreCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
        )
        for frame in self.window_frames:
            semaphore = frame.get("wavefront_history_ready")
            if semaphore is not None:
                vk.vkDestroySemaphore(self.device, semaphore, None)
            frame["wavefront_history_ready"] = vk.vkCreateSemaphore(
                self.device, semaphore_info, None
            )
            frame["wavefront_history_ready_pending"] = False

    def create_window_swapchain(
        self, requested_width, requested_height, wavefront_only=False
    ):
        """Create or recreate the GLFW swapchain and GPU output image."""
        if self.surface is None and not self._headless_surface:
            raise RuntimeError("A GLFW window is required")
        self._wait_external_releases()
        vk.vkDeviceWaitIdle(self.device)
        self._reset_wavefront_history_chain()
        self._destroy_swapchain_resources()
        if self._headless_surface:
            capabilities = SimpleNamespace(
                currentExtent=vk.VkExtent2D(
                    width=int(requested_width), height=int(requested_height)
                ),
                minImageExtent=vk.VkExtent2D(width=1, height=1),
                maxImageExtent=vk.VkExtent2D(
                    width=int(requested_width), height=int(requested_height)
                ),
                minImageCount=1,
                maxImageCount=2,
                supportedUsageFlags=vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                currentTransform=0,
            )
            formats = [SimpleNamespace(
                format=vk.VK_FORMAT_R8G8B8A8_UNORM,
                colorSpace=vk.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
            )]
            present_modes = {vk.VK_PRESENT_MODE_IMMEDIATE_KHR}
        else:
            capabilities = self.get_surface_capabilities(
                self.physical_device, self.surface
            )
            formats = self.get_surface_formats(self.physical_device, self.surface)
            present_modes = set(
                self.get_surface_present_modes(self.physical_device, self.surface)
            )
        requested_mode = self.config.present_mode.lower()
        mode_options = {
            "mailbox": vk.VK_PRESENT_MODE_MAILBOX_KHR,
            "immediate": vk.VK_PRESENT_MODE_IMMEDIATE_KHR,
            "fifo": vk.VK_PRESENT_MODE_FIFO_KHR,
        }
        requested_value = mode_options.get(requested_mode, vk.VK_PRESENT_MODE_MAILBOX_KHR)
        if requested_value in present_modes:
            present_mode = requested_value
            self.present_mode_name = requested_mode.upper()
        elif vk.VK_PRESENT_MODE_MAILBOX_KHR in present_modes:
            present_mode = vk.VK_PRESENT_MODE_MAILBOX_KHR
            self.present_mode_name = "MAILBOX"
        elif vk.VK_PRESENT_MODE_IMMEDIATE_KHR in present_modes:
            present_mode = vk.VK_PRESENT_MODE_IMMEDIATE_KHR
            self.present_mode_name = "IMMEDIATE"
        else:
            present_mode = vk.VK_PRESENT_MODE_FIFO_KHR
            self.present_mode_name = "FIFO"
        rgba_surface_format = next((item for item in formats
            if item.format == vk.VK_FORMAT_R8G8B8A8_UNORM), None)
        bgra_surface_format = next((item for item in formats
            if item.format == vk.VK_FORMAT_B8G8R8A8_UNORM), None)
        surface_storage_supported = bool(
            wavefront_only
            and self.config.direct_swapchain_storage
            and capabilities.supportedUsageFlags
                & vk.VK_IMAGE_USAGE_STORAGE_BIT
        )
        rgba_storage_supported = bool(
            rgba_surface_format is not None
            and vk.vkGetPhysicalDeviceFormatProperties(
                self.physical_device, vk.VK_FORMAT_R8G8B8A8_UNORM
            ).optimalTilingFeatures & vk.VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT
        )
        bgra_storage_supported = bool(
            bgra_surface_format is not None
            and self.formatless_storage_write_supported
            and vk.vkGetPhysicalDeviceFormatProperties(
                self.physical_device, vk.VK_FORMAT_B8G8R8A8_UNORM
            ).optimalTilingFeatures & vk.VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT
        )
        direct_surface_format = (
            rgba_surface_format if rgba_storage_supported
            else bgra_surface_format if bgra_storage_supported
            else None
        )
        direct_storage = bool(
            surface_storage_supported and direct_surface_format is not None
        )
        surface_format = (direct_surface_format if direct_storage else next(
            (item for item in formats
             if item.format == vk.VK_FORMAT_B8G8R8A8_UNORM), formats[0]
        ))
        if capabilities.currentExtent.width != 0xFFFFFFFF:
            extent = capabilities.currentExtent
        else:
            extent = vk.VkExtent2D(
                width=max(capabilities.minImageExtent.width, min(requested_width, capabilities.maxImageExtent.width)),
                height=max(capabilities.minImageExtent.height, min(requested_height, capabilities.maxImageExtent.height)),
            )
        requested_image_count = self.config.swapchain_images
        image_count = requested_image_count or capabilities.minImageCount + 1
        image_count = max(image_count, capabilities.minImageCount)
        if capabilities.maxImageCount and image_count > capabilities.maxImageCount:
            image_count = capabilities.maxImageCount
        if not capabilities.supportedUsageFlags & vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT:
            raise RuntimeError("The GLFW Vulkan surface does not support transfer-destination swapchain images")
        self.swapchain = (
            None if self._headless_surface else self.create_swapchain(
                self.device,
                vk.VkSwapchainCreateInfoKHR(
                sType=vk.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                surface=self.surface,
                minImageCount=image_count,
                imageFormat=surface_format.format,
                imageColorSpace=surface_format.colorSpace,
                imageExtent=extent,
                imageArrayLayers=1,
                imageUsage=(vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT
                    | (vk.VK_IMAGE_USAGE_STORAGE_BIT if direct_storage else 0)),
                imageSharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
                preTransform=capabilities.currentTransform,
                compositeAlpha=vk.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                presentMode=present_mode,
                clipped=vk.VK_TRUE,
                oldSwapchain=vk.VK_NULL_HANDLE,
            ),
                None,
            )
        )
        self.swapchain_images = (
            [] if self._headless_surface else
            list(self.get_swapchain_images(self.device, self.swapchain))
        )
        if direct_storage and len(self.swapchain_images) > 8:
            direct_storage = False
        self.swapchain_direct_storage = direct_storage
        self.swapchain_bgra_storage = bool(
            direct_storage
            and surface_format.format == vk.VK_FORMAT_B8G8R8A8_UNORM
        )
        if direct_storage:
            self.swapchain_image_views = [vk.vkCreateImageView(
                self.device,
                vk.VkImageViewCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    image=image,
                    viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                    format=surface_format.format,
                    components=vk.VkComponentMapping(
                        r=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                        g=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                        b=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                        a=vk.VK_COMPONENT_SWIZZLE_IDENTITY,
                    ),
                    subresourceRange=vk.VkImageSubresourceRange(
                        aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                        baseMipLevel=0, levelCount=1,
                        baseArrayLayer=0, layerCount=1,
                    ),
                ), None,
            ) for image in self.swapchain_images]
        self.swapchain_image_count = len(self.swapchain_images)
        self.swapchain_generation += 1
        self.swapchain_extent = (extent.width, extent.height)
        self.swapchain_wavefront_only = bool(wavefront_only)
        self.present_id = 0
        self.last_present_id = 0

        image_info = vk.VkImageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            pNext=(
                vk.VkExternalMemoryImageCreateInfo(
                    handleTypes=vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
                ) if self._headless_surface else None
            ),
            imageType=vk.VK_IMAGE_TYPE_2D,
            format=vk.VK_FORMAT_R8G8B8A8_UNORM,
            extent=vk.VkExtent3D(
                width=1 if direct_storage and wavefront_only else extent.width,
                height=1 if direct_storage and wavefront_only else extent.height,
                depth=1,
            ),
            mipLevels=1,
            arrayLayers=1,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            tiling=vk.VK_IMAGE_TILING_OPTIMAL,
            usage=vk.VK_IMAGE_USAGE_STORAGE_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
        )
        render_width = max(1, int(round(
            extent.width * self.config.wavefront_render_scale
        )))
        render_height = max(1, int(round(
            extent.height * self.config.wavefront_render_scale
        )))
        indirect_plan = None
        if self.config.wavefront_indirect_reuse_storage:
            # Validate the complete double-buffered footprint before creating
            # any size-dependent Vulkan images or buffers.
            indirect_plan = IndirectReservoirPlan(
                render_width,
                render_height,
                scale=self.config.wavefront_indirect_reuse_scale,
                history_frames=WINDOW_FRAMES_IN_FLIGHT,
                budget_mib=self.config.wavefront_indirect_reuse_budget_mib,
            )
        wavefront_hdr_info = vk.VkImageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType=vk.VK_IMAGE_TYPE_2D,
            format=vk.VK_FORMAT_R16G16B16A16_SFLOAT,
            extent=vk.VkExtent3D(
                width=render_width, height=render_height, depth=1
            ),
            mipLevels=1, arrayLayers=1,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            tiling=vk.VK_IMAGE_TILING_OPTIMAL,
            usage=(vk.VK_IMAGE_USAGE_STORAGE_BIT
                   | (vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                      if self.config.wavefront_hdr_capture else 0)),
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
        )
        full_gbuffer = bool(
            self.config.wavefront_temporal_reconstruction
            or self.config.stationary_accumulation
            or self.config.wavefront_diffuse_filter
            or self.config.wavefront_restir_di
            or self.config.wavefront_indirect_reuse_storage
            or self.config.object_effects
            or self.config.denoiser_enabled
        )
        wavefront_position_info = vk.VkImageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType=vk.VK_IMAGE_TYPE_2D,
            format=vk.VK_FORMAT_R32_SFLOAT,
            extent=vk.VkExtent3D(
                width=render_width if full_gbuffer else 1,
                height=render_height if full_gbuffer else 1,
                depth=1,
            ),
            mipLevels=1, arrayLayers=1,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            tiling=vk.VK_IMAGE_TILING_OPTIMAL,
            usage=vk.VK_IMAGE_USAGE_STORAGE_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
        )
        wavefront_normal_info = vk.VkImageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType=vk.VK_IMAGE_TYPE_2D,
            format=vk.VK_FORMAT_R32_UINT,
            extent=vk.VkExtent3D(
                width=render_width if full_gbuffer else 1,
                height=render_height if full_gbuffer else 1,
                depth=1,
            ),
            mipLevels=1, arrayLayers=1,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            tiling=vk.VK_IMAGE_TILING_OPTIMAL,
            usage=vk.VK_IMAGE_USAGE_STORAGE_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
        )
        wavefront_material_info = vk.VkImageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType=vk.VK_IMAGE_TYPE_2D,
            format=vk.VK_FORMAT_R32_UINT,
            extent=vk.VkExtent3D(
                width=(render_width if (
                    self.config.wavefront_restir_di
                    or self.config.wavefront_indirect_reuse_storage
                    or self.config.object_effects
                    or self.config.denoiser_enabled) else 1),
                height=(render_height if (
                    self.config.wavefront_restir_di
                    or self.config.wavefront_indirect_reuse_storage
                    or self.config.object_effects
                    or self.config.denoiser_enabled) else 1),
                depth=1,
            ),
            mipLevels=1, arrayLayers=1,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            tiling=vk.VK_IMAGE_TILING_OPTIMAL,
            usage=vk.VK_IMAGE_USAGE_STORAGE_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
        )
        wavefront_history_info = vk.VkImageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType=vk.VK_IMAGE_TYPE_2D,
            format=vk.VK_FORMAT_B10G11R11_UFLOAT_PACK32,
            extent=vk.VkExtent3D(
                width=(extent.width if
                    (self.config.wavefront_temporal_reconstruction
                     or self.config.stationary_accumulation) else 1),
                height=(extent.height if
                    (self.config.wavefront_temporal_reconstruction
                     or self.config.stationary_accumulation) else 1),
                depth=1,
            ),
            mipLevels=1, arrayLayers=1,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            tiling=vk.VK_IMAGE_TILING_OPTIMAL,
            usage=vk.VK_IMAGE_USAGE_STORAGE_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
        )
        def relax_image_info(image_format):
            enabled = wavefront_only and self.config.denoiser_enabled
            return vk.VkImageCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                imageType=vk.VK_IMAGE_TYPE_2D,
                format=image_format,
                extent=vk.VkExtent3D(
                    width=render_width if enabled else 1,
                    height=render_height if enabled else 1,
                    depth=1,
                ),
                mipLevels=1, arrayLayers=1,
                samples=vk.VK_SAMPLE_COUNT_1_BIT,
                tiling=vk.VK_IMAGE_TILING_OPTIMAL,
                usage=vk.VK_IMAGE_USAGE_STORAGE_BIT,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
                initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            )
        wavefront_relax_rgba_info = relax_image_info(
            vk.VK_FORMAT_R16G16B16A16_SFLOAT
        )
        wavefront_relax_view_z_info = relax_image_info(vk.VK_FORMAT_R32_SFLOAT)
        accumulation_info = vk.VkImageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType=vk.VK_IMAGE_TYPE_2D,
            format=vk.VK_FORMAT_R32G32B32A32_SFLOAT,
            extent=vk.VkExtent3D(
                width=1 if wavefront_only else extent.width,
                height=1 if wavefront_only else extent.height,
                depth=1,
            ),
            mipLevels=1,
            arrayLayers=1,
            samples=vk.VK_SAMPLE_COUNT_1_BIT,
            tiling=vk.VK_IMAGE_TILING_OPTIMAL,
            usage=vk.VK_IMAGE_USAGE_STORAGE_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
        )
        moment_info = vk.VkImageCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            imageType=vk.VK_IMAGE_TYPE_2D,
            format=vk.VK_FORMAT_R32_SFLOAT,
            extent=vk.VkExtent3D(
                width=extent.width if (not wavefront_only and (
                    self.config.adaptive_sampling or self.config.denoiser_enabled
                )) else 1,
                height=extent.height if (not wavefront_only and (
                    self.config.adaptive_sampling or self.config.denoiser_enabled
                )) else 1,
                depth=1,
            ),
            mipLevels=1, arrayLayers=1, samples=vk.VK_SAMPLE_COUNT_1_BIT,
            tiling=vk.VK_IMAGE_TILING_OPTIMAL,
            usage=vk.VK_IMAGE_USAGE_STORAGE_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
        )
        if not self.window_frames:
            descriptor_sets = vk.vkAllocateDescriptorSets(
                self.device,
                vk.VkDescriptorSetAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                    descriptorPool=self.descriptor_pool,
                    descriptorSetCount=WINDOW_FRAMES_IN_FLIGHT,
                    pSetLayouts=[self.descriptor_layout] * WINDOW_FRAMES_IN_FLIGHT,
                ),
            )
            nv12_descriptor_sets = (
                vk.vkAllocateDescriptorSets(
                    self.device,
                    vk.VkDescriptorSetAllocateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                        descriptorPool=self.nv12_descriptor_pool,
                        descriptorSetCount=WINDOW_FRAMES_IN_FLIGHT,
                        pSetLayouts=[self.nv12_descriptor_layout]
                        * WINDOW_FRAMES_IN_FLIGHT,
                    ),
                ) if self._headless_surface else [None] * WINDOW_FRAMES_IN_FLIGHT
            )
            p010_descriptor_sets = (
                vk.vkAllocateDescriptorSets(
                    self.device,
                    vk.VkDescriptorSetAllocateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                        descriptorPool=self.nv12_descriptor_pool,
                        descriptorSetCount=WINDOW_FRAMES_IN_FLIGHT,
                        pSetLayouts=[self.nv12_descriptor_layout]
                        * WINDOW_FRAMES_IN_FLIGHT,
                    ),
                ) if self._headless_surface else [None] * WINDOW_FRAMES_IN_FLIGHT
            )
            commands = vk.vkAllocateCommandBuffers(
                self.device,
                vk.VkCommandBufferAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                    commandPool=self.command_pool,
                    level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                    commandBufferCount=WINDOW_FRAMES_IN_FLIGHT,
                ),
            )
            wavefront_commands = vk.vkAllocateCommandBuffers(
                self.device,
                vk.VkCommandBufferAllocateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                    commandPool=self.command_pool,
                    level=vk.VK_COMMAND_BUFFER_LEVEL_SECONDARY,
                    commandBufferCount=WINDOW_FRAMES_IN_FLIGHT,
                ),
            )
            semaphore_info = vk.VkSemaphoreCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
            )
            export_semaphore_info = vk.VkSemaphoreCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                pNext=vk.VkExportSemaphoreCreateInfo(
                    handleTypes=(
                        vk.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT
                    )
                ),
            )
            for index in range(WINDOW_FRAMES_IN_FLIGHT):
                self.window_frames.append({
                    "descriptor_set": descriptor_sets[index],
                    "command": commands[index],
                    "wavefront_command": wavefront_commands[index],
                    "wavefront_command_key": None,
                    "wavefront_tile_count": 0,
                    "wavefront_cached_labels": (),
                    "wavefront_history_valid": False,
                    "wavefront_reservoir_valid": False,
                    "wavefront_indirect_reservoir_valid": False,
                    "wavefront_indirect_reservoir_initialized": False,
                    "wavefront_render_extent": None,
                    "image_available": vk.vkCreateSemaphore(self.device, semaphore_info, None),
                    "render_finished": vk.vkCreateSemaphore(self.device, semaphore_info, None),
                    # Temporal images and reservoir buffers cross frame-slot
                    # submission boundaries. Swapchain acquisition orders
                    # presentation only, so use a dedicated semaphore chain
                    # to publish the preceding slot's compute writes.
                    "wavefront_history_ready": vk.vkCreateSemaphore(
                        self.device, semaphore_info, None
                    ),
                    "wavefront_history_ready_pending": False,
                    "external_ready": (
                        vk.vkCreateSemaphore(
                            self.device, export_semaphore_info, None
                        ) if self._headless_surface else None
                    ),
                    "external_release": (
                        vk.vkCreateSemaphore(
                            self.device, export_semaphore_info, None
                        ) if self._headless_surface else None
                    ),
                    "external_release_wait": False,
                    "fence": vk.vkCreateFence(
                        self.device,
                        vk.VkFenceCreateInfo(
                            sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
                            flags=vk.VK_FENCE_CREATE_SIGNALED_BIT,
                        ),
                        None,
                    ),
                    "image": None,
                    "image_memory": None,
                    "image_memory_size": 0,
                    "image_view": None,
                    "nv12_descriptor_set": nv12_descriptor_sets[index],
                    "p010_descriptor_set": p010_descriptor_sets[index],
                    "nv12_buffer": None,
                    "nv12_pitch": 0,
                    "nv12_uv_offset": 0,
                    "p010_pitch": 0,
                    "p010_uv_offset": 0,
                    "wavefront_hdr_image": None,
                    "wavefront_hdr_memory": None,
                    "wavefront_hdr_view": None,
                    "wavefront_hdr_capture_buffer": None,
                    "wavefront_position_image": None,
                    "wavefront_position_memory": None,
                    "wavefront_position_view": None,
                    "wavefront_normal_image": None,
                    "wavefront_normal_memory": None,
                    "wavefront_normal_view": None,
                    "wavefront_material_image": None,
                    "wavefront_material_memory": None,
                    "wavefront_material_view": None,
                    "wavefront_relax_diffuse_image": None,
                    "wavefront_relax_diffuse_memory": None,
                    "wavefront_relax_diffuse_view": None,
                    "wavefront_relax_specular_image": None,
                    "wavefront_relax_specular_memory": None,
                    "wavefront_relax_specular_view": None,
                    "wavefront_relax_normal_roughness_image": None,
                    "wavefront_relax_normal_roughness_memory": None,
                    "wavefront_relax_normal_roughness_view": None,
                    "wavefront_relax_view_z_image": None,
                    "wavefront_relax_view_z_memory": None,
                    "wavefront_relax_view_z_view": None,
                    "wavefront_relax_motion_image": None,
                    "wavefront_relax_motion_memory": None,
                    "wavefront_relax_motion_view": None,
                    "wavefront_relax_identity_image": None,
                    "wavefront_relax_identity_memory": None,
                    "wavefront_relax_identity_view": None,
                    "wavefront_relax_temporal_diffuse_image": None,
                    "wavefront_relax_temporal_diffuse_memory": None,
                    "wavefront_relax_temporal_diffuse_view": None,
                    "wavefront_relax_temporal_specular_image": None,
                    "wavefront_relax_temporal_specular_memory": None,
                    "wavefront_relax_temporal_specular_view": None,
                    "wavefront_relax_atrous_diffuse_image": None,
                    "wavefront_relax_atrous_diffuse_memory": None,
                    "wavefront_relax_atrous_diffuse_view": None,
                    "wavefront_relax_atrous_specular_image": None,
                    "wavefront_relax_atrous_specular_memory": None,
                    "wavefront_relax_atrous_specular_view": None,
                    "wavefront_relax_diffuse_history_image": None,
                    "wavefront_relax_diffuse_history_memory": None,
                    "wavefront_relax_diffuse_history_view": None,
                    "wavefront_relax_specular_history_image": None,
                    "wavefront_relax_specular_history_memory": None,
                    "wavefront_relax_specular_history_view": None,
                    "wavefront_relax_history_valid": False,
                    "wavefront_history_color_image": None,
                    "wavefront_history_color_memory": None,
                    "wavefront_history_color_view": None,
                    "wavefront_reservoir_buffer": None,
                    "wavefront_indirect_reservoir_buffer": None,
                    "wavefront_indirect_seed_buffer": None,
                    "wavefront_indirect_reservoir_extent": None,
                    "query_start": index * 2,
                    "has_timestamps": False,
                    "wavefront_render_scale": self.config.wavefront_render_scale,
                    "wavefront_samples_per_pixel": self.config.samples_per_pixel,
                    "wavefront_query_count": 0,
                    "wavefront_query_labels": (),
                })
            self.timestamp_query_pool = vk.vkCreateQueryPool(
                self.device,
                vk.VkQueryPoolCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                    queryType=vk.VK_QUERY_TYPE_TIMESTAMP,
                    queryCount=WINDOW_FRAMES_IN_FLIGHT * 2,
                ),
                None,
            )

        def create_history_image(create_info, image_format):
            image = vk.vkCreateImage(self.device, create_info, None)
            requirements = vk.vkGetImageMemoryRequirements(self.device, image)
            try:
                memory = vk.vkAllocateMemory(
                    self.device,
                    vk.VkMemoryAllocateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                        allocationSize=requirements.size,
                        memoryTypeIndex=self._memory_type(
                            requirements.memoryTypeBits,
                            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        ),
                    ),
                    None,
                )
            except vk.VkErrorOutOfDeviceMemory as error:
                vk.vkDestroyImage(self.device, image, None)
                extent = create_info.extent
                raise RuntimeError(
                    "Vulkan device-memory allocation failed for "
                    f"format {image_format}, {extent.width}x{extent.height}, "
                    f"{requirements.size / (1024 * 1024):.1f} MiB "
                    f"(swapchain {self.swapchain_extent}, render "
                    f"{render_width}x{render_height})"
                ) from error
            vk.vkBindImageMemory(self.device, image, memory, 0)
            view = vk.vkCreateImageView(
                self.device,
                vk.VkImageViewCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    image=image,
                    viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                    format=image_format,
                    subresourceRange=vk.VkImageSubresourceRange(
                        aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                        baseMipLevel=0, levelCount=1, baseArrayLayer=0, layerCount=1,
                    ),
                ),
                None,
            )
            return image, memory, view

        for _index in range(2):
            image, memory, view = create_history_image(
                accumulation_info, vk.VK_FORMAT_R32G32B32A32_SFLOAT
            )
            self.accumulation_images.append(image)
            self.accumulation_memories.append(memory)
            self.accumulation_views.append(view)
            if self.config.temporal_history:
                image, memory, view = create_history_image(
                    accumulation_info, vk.VK_FORMAT_R32G32B32A32_SFLOAT
                )
                self.gbuffer_images.append(image)
                self.gbuffer_memories.append(memory)
                self.gbuffer_views.append(view)
            image, memory, view = create_history_image(
                moment_info, vk.VK_FORMAT_R32_SFLOAT
            )
            self.moment_images.append(image)
            self.moment_memories.append(memory)
            self.moment_views.append(view)

        if self.config.denoiser_enabled:
            for _index in range(2):
                image, memory, view = create_history_image(
                    accumulation_info, vk.VK_FORMAT_R32G32B32A32_SFLOAT
                )
                self.denoise_images.append(image)
                self.denoise_memories.append(memory)
                self.denoise_views.append(view)

        for frame in self.window_frames:
            frame["wavefront_history_valid"] = False
            frame["wavefront_relax_history_valid"] = False
            frame["wavefront_reservoir_valid"] = False
            frame["wavefront_indirect_reservoir_valid"] = False
            frame["wavefront_indirect_reservoir_initialized"] = False
            frame["wavefront_render_extent"] = None
            (
                frame["wavefront_hdr_image"],
                frame["wavefront_hdr_memory"],
                frame["wavefront_hdr_view"],
            ) = create_history_image(
                wavefront_hdr_info, vk.VK_FORMAT_R16G16B16A16_SFLOAT
            )
            (
                frame["wavefront_position_image"],
                frame["wavefront_position_memory"],
                frame["wavefront_position_view"],
            ) = create_history_image(
                wavefront_position_info, vk.VK_FORMAT_R32_SFLOAT
            )
            (
                frame["wavefront_normal_image"],
                frame["wavefront_normal_memory"],
                frame["wavefront_normal_view"],
            ) = create_history_image(
                wavefront_normal_info, vk.VK_FORMAT_R32_UINT
            )
            (
                frame["wavefront_material_image"],
                frame["wavefront_material_memory"],
                frame["wavefront_material_view"],
            ) = create_history_image(
                wavefront_material_info, vk.VK_FORMAT_R32_UINT
            )
            for prefix in (
                "wavefront_relax_diffuse", "wavefront_relax_specular",
                "wavefront_relax_normal_roughness", "wavefront_relax_motion",
                "wavefront_relax_temporal_diffuse",
                "wavefront_relax_temporal_specular",
                "wavefront_relax_atrous_diffuse",
                "wavefront_relax_atrous_specular",
            ):
                (
                    frame[f"{prefix}_image"], frame[f"{prefix}_memory"],
                    frame[f"{prefix}_view"],
                ) = create_history_image(
                    wavefront_relax_rgba_info,
                    vk.VK_FORMAT_R16G16B16A16_SFLOAT,
                )
            (
                frame["wavefront_relax_view_z_image"],
                frame["wavefront_relax_view_z_memory"],
                frame["wavefront_relax_view_z_view"],
            ) = create_history_image(
                wavefront_relax_view_z_info, vk.VK_FORMAT_R32_SFLOAT
            )
            (
                frame["wavefront_relax_identity_image"],
                frame["wavefront_relax_identity_memory"],
                frame["wavefront_relax_identity_view"],
            ) = create_history_image(
                wavefront_material_info, vk.VK_FORMAT_R32_UINT
            )
            for prefix in (
                "wavefront_relax_diffuse_history",
                "wavefront_relax_specular_history",
            ):
                (
                    frame[f"{prefix}_image"], frame[f"{prefix}_memory"],
                    frame[f"{prefix}_view"],
                ) = create_history_image(
                    wavefront_relax_view_z_info, vk.VK_FORMAT_R32_SFLOAT
                )
            (
                frame["wavefront_history_color_image"],
                frame["wavefront_history_color_memory"],
                frame["wavefront_history_color_view"],
            ) = create_history_image(
                wavefront_history_info, vk.VK_FORMAT_B10G11R11_UFLOAT_PACK32
            )
            reservoir_bytes = (
                _restir_reservoir_storage_bytes(
                    render_width, render_height,
                    self.config.wavefront_restir_reservoirs,
                    stratified=(
                        self.config.wavefront_stratified_primary_restir
                    ),
                ) if self.config.wavefront_restir_di else 12
            )
            frame["wavefront_reservoir_buffer"] = self._create_buffer(
                reservoir_bytes,
                vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            )
            if indirect_plan is not None:
                indirect_bytes = (
                    indirect_plan.reservoir_count
                    * indirect_plan.bytes_per_reservoir
                )
                try:
                    frame["wavefront_indirect_reservoir_buffer"] = (
                        self._create_buffer(
                            indirect_bytes,
                            vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                            | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        )
                    )
                    frame["wavefront_indirect_seed_buffer"] = self._create_buffer(
                        indirect_plan.reservoir_count * 4,
                        vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                        | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                    )
                except vk.VkErrorOutOfDeviceMemory as error:
                    for allocated_frame in self.window_frames:
                        for key in (
                            "wavefront_indirect_reservoir_buffer",
                            "wavefront_indirect_seed_buffer",
                        ):
                            allocated = allocated_frame.get(key)
                            if allocated is None:
                                continue
                            vk.vkDestroyBuffer(
                                self.device, allocated.buffer, None)
                            vk.vkFreeMemory(
                                self.device, allocated.memory, None)
                            if allocated in self._buffers:
                                self._buffers.remove(allocated)
                            allocated_frame[key] = None
                        allocated_frame[
                            "wavefront_indirect_reservoir_extent"] = None
                    seed_bytes = indirect_plan.reservoir_count * 4
                    raise RuntimeError(
                        "Vulkan device-memory allocation failed for "
                        f"indirect reservoir {indirect_plan.width}x"
                        f"{indirect_plan.height}, "
                        f"{(indirect_bytes + seed_bytes) / (1024 * 1024):.1f} "
                        "MiB per "
                        f"frame ({indirect_plan.estimated_mib:.1f} MiB total)"
                    ) from error
                frame["wavefront_indirect_reservoir_extent"] = (
                    indirect_plan.width, indirect_plan.height)
            if self.config.wavefront_hdr_capture:
                frame["wavefront_hdr_capture_buffer"] = self._create_buffer(
                    render_width * render_height * 8,
                    vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                    | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                )
            frame["image"] = vk.vkCreateImage(self.device, image_info, None)
            requirements = vk.vkGetImageMemoryRequirements(self.device, frame["image"])
            frame["image_memory_size"] = int(requirements.size)
            try:
                export_allocation = (
                    vk.VkExportMemoryAllocateInfo(
                        handleTypes=vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT
                    ) if self._headless_surface else None
                )
                allocation_chain = (
                    vk.VkMemoryDedicatedAllocateInfo(
                        pNext=export_allocation,
                        image=frame["image"],
                        buffer=vk.VK_NULL_HANDLE,
                    ) if self._headless_surface else None
                )
                frame["image_memory"] = vk.vkAllocateMemory(
                    self.device,
                    vk.VkMemoryAllocateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                        pNext=allocation_chain,
                        allocationSize=requirements.size,
                        memoryTypeIndex=self._memory_type(
                            requirements.memoryTypeBits,
                            vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                        ),
                    ),
                    None,
                )
            except vk.VkErrorOutOfDeviceMemory as error:
                vk.vkDestroyImage(self.device, frame["image"], None)
                frame["image"] = None
                extent = image_info.extent
                raise RuntimeError(
                    "Vulkan device-memory allocation failed for presentation "
                    f"image {extent.width}x{extent.height}, "
                    f"{requirements.size / (1024 * 1024):.1f} MiB "
                    f"(swapchain {self.swapchain_extent}, render "
                    f"{render_width}x{render_height})"
                ) from error
            vk.vkBindImageMemory(self.device, frame["image"], frame["image_memory"], 0)
            frame["image_view"] = vk.vkCreateImageView(
                self.device,
                vk.VkImageViewCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
                    image=frame["image"],
                    viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                    format=vk.VK_FORMAT_R8G8B8A8_UNORM,
                    subresourceRange=vk.VkImageSubresourceRange(
                        aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                        baseMipLevel=0,
                        levelCount=1,
                        baseArrayLayer=0,
                        layerCount=1,
                    ),
                ),
                None,
            )
            if self._headless_surface:
                # PyNvVideoCodec currently requires tightly packed input even
                # when CUDA Array Interface strides describe a larger pitch.
                nv12_pitch = int(extent.width)
                nv12_uv_offset = nv12_pitch * int(extent.height)
                p010_pitch = int(extent.width) * 2
                p010_uv_offset = p010_pitch * int(extent.height)
                video_size = p010_uv_offset + p010_pitch * (
                    (int(extent.height) + 1) // 2
                )
                frame["nv12_buffer"] = self._create_exportable_buffer(
                    video_size,
                    vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                )
                frame["nv12_pitch"] = nv12_pitch
                frame["nv12_uv_offset"] = nv12_uv_offset
                frame["p010_pitch"] = p010_pitch
                frame["p010_uv_offset"] = p010_uv_offset
                vk.vkUpdateDescriptorSets(
                    self.device, 2,
                    [
                        vk.VkWriteDescriptorSet(
                            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                            dstSet=frame["nv12_descriptor_set"],
                            dstBinding=0, descriptorCount=1,
                            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                            pImageInfo=[vk.VkDescriptorImageInfo(
                                imageView=frame["image_view"],
                                imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
                            )],
                        ),
                        vk.VkWriteDescriptorSet(
                            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                            dstSet=frame["nv12_descriptor_set"],
                            dstBinding=1, descriptorCount=1,
                            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            pBufferInfo=[vk.VkDescriptorBufferInfo(
                                buffer=frame["nv12_buffer"].buffer,
                                offset=0, range=video_size,
                            )],
                        ),
                    ], 0, None,
                )
                vk.vkUpdateDescriptorSets(
                    self.device, 2,
                    [
                        vk.VkWriteDescriptorSet(
                            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                            dstSet=frame["p010_descriptor_set"],
                            dstBinding=0, descriptorCount=1,
                            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                            pImageInfo=[vk.VkDescriptorImageInfo(
                                imageView=frame["wavefront_hdr_view"],
                                imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
                            )],
                        ),
                        vk.VkWriteDescriptorSet(
                            sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                            dstSet=frame["p010_descriptor_set"],
                            dstBinding=1, descriptorCount=1,
                            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            pBufferInfo=[vk.VkDescriptorBufferInfo(
                                buffer=frame["nv12_buffer"].buffer,
                                offset=0, range=video_size,
                            )],
                        ),
                    ], 0, None,
                )
            as_descriptor = vk.VkWriteDescriptorSetAccelerationStructureKHR(
                sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
                accelerationStructureCount=1,
                pAccelerationStructures=[self.scene_tlas.handle],
            )
            image_descriptor = vk.VkDescriptorImageInfo(
                imageView=frame["image_view"],
                imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            )
            material_descriptor = vk.VkDescriptorBufferInfo(
                buffer=self.scene_material_buffer.buffer,
                offset=0,
                range=self.scene_material_buffer.size,
            )
            vertex_descriptor = vk.VkDescriptorBufferInfo(
                buffer=self.scene_vertex_buffer.buffer,
                offset=0,
                range=self.scene_vertex_buffer.size,
            )
            light_descriptor = vk.VkDescriptorBufferInfo(
                buffer=self.scene_light_buffer.buffer,
                offset=0,
                range=self.scene_light_buffer.size,
            )
            area_light_descriptor = vk.VkDescriptorBufferInfo(
                buffer=self.scene_area_light_buffer.buffer,
                offset=0,
                range=self.scene_area_light_buffer.size,
            )
            attribute_descriptor = vk.VkDescriptorBufferInfo(
                buffer=self.scene_attribute_buffer.buffer,
                offset=0,
                range=self.scene_attribute_buffer.size,
            )
            history_descriptors = []
            for index in range(2):
                history_descriptors.extend((
                    vk.VkDescriptorImageInfo(
                        imageView=self.accumulation_views[index],
                        imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
                    ),
                    vk.VkDescriptorImageInfo(
                        imageView=(
                            self.gbuffer_views[index]
                            if self.gbuffer_views else self.accumulation_views[index]
                        ),
                        imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
                    ),
                ))
            history_descriptors.extend(
                vk.VkDescriptorImageInfo(
                    imageView=view, imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL
                )
                for view in self.moment_views
            )
            denoise_views = self.denoise_views or self.accumulation_views
            denoise_descriptors = [
                vk.VkDescriptorImageInfo(
                    imageView=view, imageLayout=vk.VK_IMAGE_LAYOUT_GENERAL
                )
                for view in denoise_views
            ]
            descriptor_writes = [
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    pNext=as_descriptor,
                    dstSet=frame["descriptor_set"],
                    dstBinding=0,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                ),
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=frame["descriptor_set"],
                    dstBinding=1,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    pImageInfo=[image_descriptor],
                ),
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=frame["descriptor_set"],
                    dstBinding=2,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[material_descriptor],
                ),
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=frame["descriptor_set"],
                    dstBinding=3,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[vertex_descriptor],
                ),
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=frame["descriptor_set"],
                    dstBinding=10,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[light_descriptor],
                ),
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=frame["descriptor_set"],
                    dstBinding=11,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[area_light_descriptor],
                ),
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=frame["descriptor_set"],
                    dstBinding=14,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[attribute_descriptor],
                ),
            ]
            if self.scene_custom_attribute_buffer is not None:
                custom_attribute_descriptor = vk.VkDescriptorBufferInfo(
                    buffer=self.scene_custom_attribute_buffer.buffer,
                    offset=0,
                    range=self.scene_custom_attribute_buffer.size,
                )
                descriptor_writes.append(vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=frame["descriptor_set"],
                    dstBinding=15,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                    pBufferInfo=[custom_attribute_descriptor],
                ))
            descriptor_writes.extend(
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=frame["descriptor_set"],
                    dstBinding=4 + index,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    pImageInfo=[descriptor],
                )
                for index, descriptor in enumerate(history_descriptors)
            )
            descriptor_writes.extend(
                vk.VkWriteDescriptorSet(
                    sType=vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                    dstSet=frame["descriptor_set"],
                    dstBinding=12 + index,
                    descriptorCount=1,
                    descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                    pImageInfo=[descriptor],
                )
                for index, descriptor in enumerate(denoise_descriptors)
            )
            vk.vkUpdateDescriptorSets(
                self.device, len(descriptor_writes), descriptor_writes, 0, None
            )
        self._single_use(
            lambda command: vk.vkCmdPipelineBarrier(
                command,
                vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, None, 0, None,
                len(self.window_frames) * 18 + len(self.accumulation_images)
                + len(self.gbuffer_images) + len(self.moment_images)
                + len(self.denoise_images),
                [self._image_barrier(
                    frame["image"], vk.VK_IMAGE_LAYOUT_UNDEFINED,
                    vk.VK_IMAGE_LAYOUT_GENERAL, 0, vk.VK_ACCESS_SHADER_WRITE_BIT
                ) for frame in self.window_frames] + [self._image_barrier(
                    frame["wavefront_hdr_image"], vk.VK_IMAGE_LAYOUT_UNDEFINED,
                    vk.VK_IMAGE_LAYOUT_GENERAL, 0,
                    vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
                ) for frame in self.window_frames] + [self._image_barrier(
                    frame["wavefront_position_image"],
                    vk.VK_IMAGE_LAYOUT_UNDEFINED, vk.VK_IMAGE_LAYOUT_GENERAL, 0,
                    vk.VK_ACCESS_SHADER_WRITE_BIT,
                ) for frame in self.window_frames] + [self._image_barrier(
                    frame["wavefront_normal_image"],
                    vk.VK_IMAGE_LAYOUT_UNDEFINED, vk.VK_IMAGE_LAYOUT_GENERAL, 0,
                    vk.VK_ACCESS_SHADER_WRITE_BIT,
                ) for frame in self.window_frames] + [self._image_barrier(
                    frame["wavefront_material_image"],
                    vk.VK_IMAGE_LAYOUT_UNDEFINED, vk.VK_IMAGE_LAYOUT_GENERAL, 0,
                    vk.VK_ACCESS_SHADER_WRITE_BIT,
                ) for frame in self.window_frames] + [self._image_barrier(
                    frame["wavefront_history_color_image"],
                    vk.VK_IMAGE_LAYOUT_UNDEFINED, vk.VK_IMAGE_LAYOUT_GENERAL, 0,
                    vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
                ) for frame in self.window_frames
                ] + [self._image_barrier(
                    frame[f"{prefix}_image"],
                    vk.VK_IMAGE_LAYOUT_UNDEFINED,
                    vk.VK_IMAGE_LAYOUT_GENERAL, 0,
                    vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
                ) for frame in self.window_frames for prefix in (
                    "wavefront_relax_diffuse", "wavefront_relax_specular",
                    "wavefront_relax_normal_roughness",
                    "wavefront_relax_view_z", "wavefront_relax_motion",
                    "wavefront_relax_identity",
                    "wavefront_relax_temporal_diffuse",
                    "wavefront_relax_temporal_specular",
                    "wavefront_relax_atrous_diffuse",
                    "wavefront_relax_atrous_specular",
                    "wavefront_relax_diffuse_history",
                    "wavefront_relax_specular_history",
                )
                ] + [self._image_barrier(
                    image, vk.VK_IMAGE_LAYOUT_UNDEFINED,
                    vk.VK_IMAGE_LAYOUT_GENERAL, 0,
                    vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
                ) for image in (
                    self.accumulation_images + self.gbuffer_images + self.moment_images
                    + self.denoise_images
                )],
            )
        )
        if self.wavefront_executor is not None:
            for slot, frame in enumerate(self.window_frames):
                self.wavefront_executor.bind_output_image(
                    slot, frame["wavefront_hdr_view"],
                    frame["wavefront_position_view"],
                    frame["wavefront_normal_view"],
                    frame["wavefront_material_view"], frame["image_view"],
                )
                self.wavefront_executor.bind_indirect_reuse_inputs(
                    slot, frame["wavefront_hdr_view"],
                    frame["wavefront_position_view"],
                    frame["wavefront_normal_view"],
                    frame["wavefront_material_view"],
                )
                indirect_buffer = frame.get(
                    "wavefront_indirect_reservoir_buffer"
                )
                if indirect_buffer is not None:
                    self.wavefront_executor.bind_indirect_reuse_buffer(
                        slot, indirect_buffer,
                        frame["wavefront_indirect_seed_buffer"],
                    )
            if self.swapchain_direct_storage:
                self.wavefront_executor.bind_reconstruction_outputs(
                    self.swapchain_image_views
                )
        self.window_frame_index = 0

    def present_wavefront_window(
        self, scene, camera, width, height, *, pixel_format="rgba8",
        render_extent=None,
    ):
        """Render tiled wavefront paths into a Vulkan image and present it."""
        frame_start = time.perf_counter()
        if self.wavefront_last_frame_start is not None:
            interval_ms = (
                frame_start - self.wavefront_last_frame_start
            ) * 1000.0
            if self.wavefront_cadence_ms <= 0.0:
                self.wavefront_cadence_ms = interval_ms
            else:
                # Smooth compositor cadence without concealing sustained stalls.
                self.wavefront_cadence_ms += 0.12 * (
                    interval_ms - self.wavefront_cadence_ms
                )
        self.wavefront_last_frame_start = frame_start
        scene_start = time.perf_counter()
        self.prepare_window_scene(scene)
        self._replace_wavefront_executor_if_strategy_changed()
        scene_ms = (time.perf_counter() - scene_start) * 1000.0
        swapchain_start = time.perf_counter()
        if (
            self.swapchain_extent != (width, height)
            or self.swapchain_wavefront_only is not True
        ):
            self.create_window_swapchain(width, height, wavefront_only=True)
        swapchain_ms = (time.perf_counter() - swapchain_start) * 1000.0
        width, height = self.swapchain_extent
        frame_slot = self.window_frame_index
        frame = self.window_frames[frame_slot]
        profiling = self.config.wavefront_profiling
        if profiling and self.wavefront_timestamp_query_pool is None:
            self.wavefront_timestamp_query_pool = vk.vkCreateQueryPool(
                self.device,
                vk.VkQueryPoolCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                    queryType=vk.VK_QUERY_TYPE_TIMESTAMP,
                    queryCount=WINDOW_FRAMES_IN_FLIGHT * 1024,
                ), None,
            )
        wait_start = time.perf_counter()
        vk.vkWaitForFences(
            self.device, 1, [frame["fence"]], vk.VK_TRUE, (1 << 64) - 1
        )
        fence_wait_ms = (time.perf_counter() - wait_start) * 1000.0
        gpu_frame_ms = 0.0
        stage_timings = {}
        work_counters = (
            self.wavefront_executor.read_work_counters(frame_slot)
            if profiling and self.wavefront_executor is not None
            else {}
        )
        indirect_reuse_counters = (
            self.wavefront_executor.read_indirect_reuse_counters(frame_slot)
            if self.wavefront_executor is not None else {}
        )
        indirect_reuse_metrics = {}
        if indirect_reuse_counters:
            sampled = max(indirect_reuse_counters["sampled_pixels"], 1)
            temporal_attempts = max(
                indirect_reuse_counters["temporal_attempts"], 1
            )
            spatial_attempts = max(
                indirect_reuse_counters["spatial_attempts"], 1
            )
            indirect_reuse_metrics = {
                "temporal_acceptance": (
                    indirect_reuse_counters["temporal_accepted"]
                    / temporal_attempts
                ),
                "spatial_acceptance": (
                    indirect_reuse_counters["spatial_accepted"]
                    / spatial_attempts
                ),
                "average_represented_samples": (
                    indirect_reuse_counters["represented_samples"] / sampled
                ),
                "history_clamp_rate": (
                    indirect_reuse_counters["history_clamped"] / sampled
                ),
                "weight_saturation": (
                    indirect_reuse_counters["weight_saturated"] / sampled
                ),
            }
        previous_query_count = frame["wavefront_query_count"] if profiling else 0
        previous_labels = frame["wavefront_query_labels"]
        if previous_query_count > 1:
            timestamps = vk.ffi.new("uint64_t[]", previous_query_count)
            vk.vkGetQueryPoolResults(
                self.device, self.wavefront_timestamp_query_pool,
                frame_slot * 1024, previous_query_count,
                vk.ffi.sizeof(timestamps), timestamps,
                vk.ffi.sizeof("uint64_t"),
                vk.VK_QUERY_RESULT_64_BIT | vk.VK_QUERY_RESULT_WAIT_BIT,
            )
            for index, label in enumerate(previous_labels):
                duration = (
                    (timestamps[index + 1] - timestamps[index])
                    * self.timestamp_period / 1_000_000.0
                )
                stage_timings[label] = stage_timings.get(label, 0.0) + duration
        if frame["has_timestamps"]:
            timestamps = vk.ffi.new("uint64_t[]", 2)
            vk.vkGetQueryPoolResults(
                self.device, self.timestamp_query_pool, frame["query_start"], 2,
                vk.ffi.sizeof(timestamps), timestamps,
                vk.ffi.sizeof("uint64_t"),
                vk.VK_QUERY_RESULT_64_BIT | vk.VK_QUERY_RESULT_WAIT_BIT,
            )
            gpu_frame_ms = (
                (timestamps[1] - timestamps[0]) * self.timestamp_period
                / 1_000_000.0
            )
        accumulation_key = (id(scene), scene.revision, width, height)
        (
            accumulation_camera_signature,
            accumulation_active,
            interactive_active,
        ) = self._begin_accumulation_frame(accumulation_key, camera)
        render_scale = (
            self.dynamic_resolution.update(
                gpu_frame_ms, frame["wavefront_render_scale"],
                work_units=frame["wavefront_samples_per_pixel"],
                target_work_units=frame["wavefront_samples_per_pixel"],
            )
            if self.dynamic_resolution is not None
            else self.config.wavefront_render_scale
        )
        if render_extent is not None:
            requested_render_width = max(1, int(render_extent[0]))
            requested_render_height = max(1, int(render_extent[1]))
            requested_scale = min(
                requested_render_width / max(width, 1),
                requested_render_height / max(height, 1),
                1.0,
            )
            render_scale = min(render_scale, requested_scale)
        interactive_scale = self.config.wavefront_interactive_render_scale
        if interactive_active and interactive_scale is not None:
            render_scale = min(render_scale, interactive_scale)
        if interactive_active and self.interactive_dynamic_resolution is not None:
            target_work_units = (
                self.config.wavefront_interactive_min_samples
                if self.interactive_dynamic_samples is not None
                else frame["wavefront_samples_per_pixel"]
            )
            render_scale = min(
                render_scale,
                self.interactive_dynamic_resolution.update(
                    gpu_frame_ms, frame["wavefront_render_scale"],
                    work_units=frame["wavefront_samples_per_pixel"],
                    target_work_units=target_work_units,
                ),
            )
        render_width = max(1, int(round(width * render_scale)))
        render_height = max(1, int(round(height * render_scale)))
        render_extent = (render_width, render_height)
        if render_extent != self.accumulation_render_extent:
            self.accumulation_frame = 0
            self.accumulation_history_valid = False
            self.accumulation_render_extent = render_extent
        effective_samples = self.config.samples_per_pixel
        if (
            self.config.interactive_samples_per_pixel is not None
            and interactive_active
        ):
            effective_samples = self.config.interactive_samples_per_pixel
        if interactive_active and self.interactive_dynamic_samples is not None:
            interactive_max_scale = (
                self.config.wavefront_interactive_render_scale
                if self.config.wavefront_interactive_render_scale is not None
                else self.config.wavefront_render_scale
            )
            effective_samples = self.interactive_dynamic_samples.update(
                gpu_frame_ms,
                frame["wavefront_samples_per_pixel"],
                frame["wavefront_render_scale"],
                selected_scale=render_scale,
                allow_increase=(render_scale >= interactive_max_scale - 1e-6),
            )
        self.effective_samples_per_pixel = effective_samples

        acquire_start = time.perf_counter()
        if self._headless_surface:
            image_index = 0
        else:
            try:
                image_index = self.acquire_next_image(
                    self.device, self.swapchain, (1 << 64) - 1,
                    frame["image_available"], vk.VK_NULL_HANDLE,
                )
            except (vk.VkSuboptimalKhr, vk.VkErrorOutOfDateKhr):
                vk.vkDeviceWaitIdle(self.device)
                vk.vkDestroySemaphore(
                    self.device, frame["image_available"], None
                )
                frame["image_available"] = vk.vkCreateSemaphore(
                    self.device,
                    vk.VkSemaphoreCreateInfo(
                        sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
                    ), None,
                )
                self.create_window_swapchain(width, height, wavefront_only=True)
                return self.present_wavefront_window(
                    scene, camera, width, height,
                    render_extent=render_extent,
                )
        acquire_ms = (time.perf_counter() - acquire_start) * 1000.0
        vk.vkResetFences(self.device, 1, [frame["fence"]])

        record_start = time.perf_counter()
        query_base = frame_slot * 1024
        query_labels = []

        def timestamp_stage(command_buffer, label):
            if not profiling:
                return
            if len(query_labels) >= 1023:
                return
            query_labels.append(label)
            vk.vkCmdWriteTimestamp(
                command_buffer, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                self.wavefront_timestamp_query_pool,
                query_base + len(query_labels),
            )
        capacity = self.config.wavefront_tile_capacity
        camera_vectors = _camera_vectors(camera)
        if self.wavefront_executor is None:
            self.wavefront_executor = VulkanWavefrontExecutor(self, capacity)
            for slot, output_frame in enumerate(self.window_frames):
                if output_frame.get("image_view"):
                    self.wavefront_executor.bind_output_image(
                        slot, output_frame["wavefront_hdr_view"],
                        output_frame["wavefront_position_view"],
                        output_frame["wavefront_normal_view"],
                        output_frame["wavefront_material_view"],
                        output_frame["image_view"],
                    )
                    self.wavefront_executor.bind_indirect_reuse_inputs(
                        slot, output_frame["wavefront_hdr_view"],
                        output_frame["wavefront_position_view"],
                        output_frame["wavefront_normal_view"],
                        output_frame["wavefront_material_view"],
                    )
                    indirect_buffer = output_frame.get(
                        "wavefront_indirect_reservoir_buffer"
                    )
                    if indirect_buffer is not None:
                        self.wavefront_executor.bind_indirect_reuse_buffer(
                            slot, indirect_buffer,
                            output_frame["wavefront_indirect_seed_buffer"],
                        )
            if self.swapchain_direct_storage:
                self.wavefront_executor.bind_reconstruction_outputs(
                    self.swapchain_image_views
                )
        self.wavefront_executor.update_camera(
            frame_slot, camera_vectors, self.wavefront_frame_sequence,
            image_index if self.swapchain_direct_storage else 0,
            projection=_camera_projection(camera),
        )
        self.wavefront_executor.bind_scene()

        camera_motion_pixels = _camera_angular_motion_pixels(
            self.wavefront_previous_present_camera, camera, render_height
        )
        temporal_motion_valid = (
            camera_motion_pixels
            <= self.config.wavefront_temporal_motion_limit_pixels
        )
        indirect_history_limit = _motion_adaptive_history_limit(
            self.config.wavefront_indirect_reuse_history_limit,
            camera_motion_pixels,
            self.config.wavefront_indirect_reuse_history_motion_pixels,
        )
        restir_history_limit = _motion_adaptive_history_limit(
            self.config.wavefront_restir_history_limit,
            camera_motion_pixels,
            self.config.wavefront_restir_history_motion_pixels,
        )

        perspective_history_compatible = bool(
            isinstance(camera, PerspectiveCamera)
            and isinstance(
                self.wavefront_previous_present_camera, PerspectiveCamera
            )
        )
        wavefront_temporal_enabled = bool(
            self.config.wavefront_temporal_reconstruction
            or self.config.stationary_accumulation
        )
        history_valid = bool(
            wavefront_temporal_enabled
            and accumulation_active
            and self.window_frames[1 - frame_slot]["wavefront_history_valid"]
            and self.window_frames[1 - frame_slot].get(
                "wavefront_render_extent"
            ) == render_extent
            and temporal_motion_valid
            and perspective_history_compatible
        )
        relax_history_valid = bool(
            self.config.denoiser_enabled
            and self.window_frames[1 - frame_slot].get(
                "wavefront_relax_history_valid", False
            )
            and self.window_frames[1 - frame_slot].get(
                "wavefront_render_extent"
            ) == render_extent
            and temporal_motion_valid
            and perspective_history_compatible
        )
        relax_camera_motion_pixels = _camera_motion_pixels(
            self.wavefront_previous_present_camera, camera, render_height
        )
        relax_policy = _relax_temporal_policy(
            self.config, relax_camera_motion_pixels
        )
        self.wavefront_executor.update_relax_temporal_constants(
            frame_slot, render_width, render_height, relax_history_valid,
            **relax_policy,
        )
        restir_history_valid = bool(
            self.wavefront_restir_runtime_enabled
            and self.window_frames[1 - frame_slot]["wavefront_reservoir_valid"]
            and self.window_frames[1 - frame_slot].get(
                "wavefront_render_extent"
            ) == (render_width, render_height)
            and temporal_motion_valid
            and perspective_history_compatible
        )
        indirect_needs_clear = bool(
            frame.get("wavefront_indirect_reservoir_buffer") is not None
            and not frame.get("wavefront_indirect_reservoir_initialized", False)
        )
        indirect_history_valid = bool(
            self.config.wavefront_indirect_reuse_temporal
            and self.window_frames[1 - frame_slot].get(
                "wavefront_indirect_reservoir_valid", False
            )
            and self.window_frames[1 - frame_slot].get(
                "wavefront_indirect_reservoir_extent"
            ) == frame.get("wavefront_indirect_reservoir_extent")
            and temporal_motion_valid
            and perspective_history_compatible
        )

        projected_effect_camera = (
            _camera_signature(camera)
            if any(isinstance(effect, (BoundingBox, XRay))
                   for _range, effect in self.object_effect_bindings)
            else None
        )
        render_key = (
            width, height, render_width, render_height,
            capacity, self.config.max_bounces,
            effective_samples, history_valid, relax_history_valid, profiling,
            self.config.wavefront_fused_secondary,
            self.config.wavefront_subgroup_enqueue,
            self.resolved_execution_strategy,
            self.wavefront_restir_runtime_enabled,
            self.config.wavefront_restir_reservoirs,
            restir_history_valid,
            indirect_needs_clear,
            indirect_history_valid,
            indirect_history_limit,
            restir_history_limit,
            self.object_effect_bindings,
            projected_effect_camera,
        )
        secondary = frame["wavefront_command"]
        command_cache_hit = frame["wavefront_command_key"] == render_key
        if not command_cache_hit:
            vk.vkResetCommandBuffer(secondary, 0)
            inheritance = vk.VkCommandBufferInheritanceInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_INHERITANCE_INFO
            )
            vk.vkBeginCommandBuffer(
                secondary,
                vk.VkCommandBufferBeginInfo(
                    sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                    flags=0, pInheritanceInfo=inheritance,
                ),
            )
            if profiling:
                vk.vkCmdResetQueryPool(
                    secondary, self.wavefront_timestamp_query_pool,
                    query_base, 1024,
                )
                vk.vkCmdWriteTimestamp(
                    secondary, vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                    self.wavefront_timestamp_query_pool, query_base,
                )
                self.wavefront_executor.record_work_counter_reset(
                    secondary, frame_slot
                )
            if indirect_needs_clear:
                indirect_extent = frame["wavefront_indirect_reservoir_extent"]
                self.wavefront_executor.record_indirect_reuse_clear(
                    secondary, frame_slot,
                    indirect_extent[0] * indirect_extent[1],
                )
                frame["wavefront_indirect_reservoir_initialized"] = True
            tile_count = 0
            sample_count = (
                self.config.wavefront_restir_reservoirs
                if self.wavefront_restir_runtime_enabled
                else effective_samples
            )
            for sample_index in range(sample_count):
                y = 0
                while y < render_height:
                    tile_width = min(render_width, capacity)
                    tile_height = min(
                        render_height - y, max(1, capacity // tile_width)
                    )
                    x = 0
                    while x < render_width:
                        current_width = min(tile_width, render_width - x)
                        current_height = min(
                            tile_height, max(1, capacity // current_width),
                            render_height - y,
                        )
                        self.trace_wavefront_tile(
                            camera, render_width, render_height,
                            tile_origin=(x, y),
                            tile_extent=(current_width, current_height),
                            frame_index=self.wavefront_frame_sequence,
                            sample_index=sample_index,
                            sample_count=sample_count,
                            output_image_slot=frame_slot, readback=False,
                            command=secondary, camera_vectors=camera_vectors,
                            camera_slot=frame_slot, upload_camera=False,
                            timestamp=(
                                timestamp_stage if profiling else None
                            ),
                            restir_history_valid_override=restir_history_valid,
                            restir_history_limit=restir_history_limit,
                        )
                        x += current_width
                        tile_count += 1
                    y += tile_height
            if self.config.wavefront_indirect_reuse_candidates:
                indirect_extent = frame["wavefront_indirect_reservoir_extent"]
                self.wavefront_executor.record_indirect_reuse_candidates(
                    secondary, frame_slot, render_width, render_height,
                    indirect_extent[0], indirect_extent[1],
                    indirect_history_valid, self.wavefront_frame_sequence,
                    indirect_history_limit,
                )
                if profiling:
                    timestamp_stage(secondary, "indirect_candidates")
                frame["wavefront_indirect_reservoir_valid"] = True
                if (self.config.wavefront_indirect_reuse_debug_view != "off"
                        or self.config.wavefront_indirect_reuse_apply):
                    self.wavefront_executor.record_indirect_reuse_debug(
                        secondary, frame_slot, render_width, render_height,
                        indirect_extent[0], indirect_extent[1],
                    )
                    if profiling:
                        timestamp_stage(secondary, (
                            "indirect_debug"
                            if self.config.wavefront_indirect_reuse_debug_view
                            != "off" else "indirect_apply"
                        ))
            self.wavefront_executor.record_relax_temporal(
                secondary, frame_slot, render_width, render_height,
            )
            if profiling and self.config.denoiser_enabled:
                timestamp_stage(secondary, "relax_temporal")
            self.wavefront_executor.record_reconstruction(
                secondary, frame_slot, render_width, render_height,
                width, height, history_valid=history_valid,
                scene=scene, camera=camera,
            )
            if profiling:
                timestamp_stage(secondary, "reconstruct")
            vk.vkEndCommandBuffer(secondary)
            frame["wavefront_command_key"] = render_key
            frame["wavefront_tile_count"] = tile_count
            frame["wavefront_cached_labels"] = tuple(query_labels)
        else:
            query_labels = list(frame["wavefront_cached_labels"])
        tile_count = frame["wavefront_tile_count"]

        command = frame["command"]
        vk.vkResetCommandBuffer(command, 0)
        vk.vkBeginCommandBuffer(
            command,
            vk.VkCommandBufferBeginInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            ),
        )
        vk.vkCmdResetQueryPool(
            command, self.timestamp_query_pool, frame["query_start"], 2
        )
        vk.vkCmdWriteTimestamp(
            command, vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            self.timestamp_query_pool, frame["query_start"],
        )
        if self._headless_surface:
            pass
        elif self.swapchain_direct_storage:
            direct_ready = self._image_barrier(
                self.swapchain_images[image_index],
                vk.VK_IMAGE_LAYOUT_UNDEFINED, vk.VK_IMAGE_LAYOUT_GENERAL,
                0, vk.VK_ACCESS_SHADER_WRITE_BIT,
            )
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                0, None, 0, None, 1, [direct_ready],
            )
        vk.vkCmdExecuteCommands(command, 1, [secondary])
        if self._headless_surface and pixel_format in {"nv12", "p010"}:
            p010 = pixel_format == "p010"
            source_image = (
                frame["wavefront_hdr_image"] if p010 else frame["image"]
            )
            source_ready = self._image_barrier(
                source_image, vk.VK_IMAGE_LAYOUT_GENERAL,
                vk.VK_IMAGE_LAYOUT_GENERAL,
                vk.VK_ACCESS_SHADER_WRITE_BIT,
                vk.VK_ACCESS_SHADER_READ_BIT,
            )
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                0, None, 0, None, 1, [source_ready],
            )
            vk.vkCmdBindPipeline(
                command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                self.p010_pipeline if p010 else self.nv12_pipeline,
            )
            vk.vkCmdBindDescriptorSets(
                command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                self.nv12_pipeline_layout, 0, 1,
                [
                    frame["p010_descriptor_set"]
                    if p010 else frame["nv12_descriptor_set"]
                ], 0, None,
            )
            pitch = frame["p010_pitch"] if p010 else frame["nv12_pitch"]
            constants = bytearray(struct.pack(
                "3If", width, height, pitch,
                self.config.wavefront_exposure if p010 else 0.0,
            ))
            vk.vkCmdPushConstants(
                command, self.nv12_pipeline_layout,
                vk.VK_SHADER_STAGE_COMPUTE_BIT, 0, len(constants),
                vk.ffi.from_buffer(constants),
            )
            vk.vkCmdDispatch(
                command, (width + 31) // 32, (height + 15) // 16, 1,
            )
            nv12_ready = vk.VkBufferMemoryBarrier(
                sType=vk.VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
                dstAccessMask=vk.VK_ACCESS_MEMORY_READ_BIT,
                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                buffer=frame["nv12_buffer"].buffer,
                offset=0, size=frame["nv12_buffer"].size,
            )
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
                0, None, 1, [nv12_ready], 0, None,
            )
        capture_buffer = frame.get("wavefront_hdr_capture_buffer")
        if capture_buffer is not None:
            capture_ready = self._image_barrier(
                frame["wavefront_hdr_image"],
                vk.VK_IMAGE_LAYOUT_GENERAL,
                vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                vk.VK_ACCESS_SHADER_WRITE_BIT,
                vk.VK_ACCESS_TRANSFER_READ_BIT,
            )
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                0, None, 0, None, 1, [capture_ready],
            )
            vk.vkCmdCopyImageToBuffer(
                command, frame["wavefront_hdr_image"],
                vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                capture_buffer.buffer, 1,
                [vk.VkBufferImageCopy(
                    bufferOffset=0, bufferRowLength=0, bufferImageHeight=0,
                    imageSubresource=vk.VkImageSubresourceLayers(
                        aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                        mipLevel=0, baseArrayLayer=0, layerCount=1,
                    ),
                    imageOffset=vk.VkOffset3D(x=0, y=0, z=0),
                    imageExtent=vk.VkExtent3D(
                        width=render_width, height=render_height, depth=1,
                    ),
                )],
            )
            capture_complete = self._image_barrier(
                frame["wavefront_hdr_image"],
                vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                vk.VK_IMAGE_LAYOUT_GENERAL,
                vk.VK_ACCESS_TRANSFER_READ_BIT,
                vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
            )
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                0, None, 0, None, 1, [capture_complete],
            )
        if self._headless_surface:
            pass
        elif self.swapchain_direct_storage:
            present_ready = self._image_barrier(
                self.swapchain_images[image_index],
                vk.VK_IMAGE_LAYOUT_GENERAL,
                vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                vk.VK_ACCESS_SHADER_WRITE_BIT, 0,
            )
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
                0, None, 0, None, 1, [present_ready],
            )
        else:
            barriers = [
            self._image_barrier(
                frame["image"], vk.VK_IMAGE_LAYOUT_GENERAL,
                vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                vk.VK_ACCESS_SHADER_WRITE_BIT, vk.VK_ACCESS_TRANSFER_READ_BIT,
            ),
            self._image_barrier(
                self.swapchain_images[image_index], vk.VK_IMAGE_LAYOUT_UNDEFINED,
                vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                0, vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            ),
        ]
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                0, None, 0, None, len(barriers), barriers,
            )
            blit = vk.VkImageBlit(
            srcSubresource=vk.VkImageSubresourceLayers(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, mipLevel=0,
                baseArrayLayer=0, layerCount=1,
            ),
            srcOffsets=[
                vk.VkOffset3D(x=0, y=0, z=0),
                vk.VkOffset3D(x=width, y=height, z=1),
            ],
            dstSubresource=vk.VkImageSubresourceLayers(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, mipLevel=0,
                baseArrayLayer=0, layerCount=1,
            ),
            dstOffsets=[
                vk.VkOffset3D(x=0, y=0, z=0),
                vk.VkOffset3D(x=width, y=height, z=1),
            ],
        )
            vk.vkCmdBlitImage(
            command, frame["image"], vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            self.swapchain_images[image_index],
            vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, [blit], vk.VK_FILTER_NEAREST,
        )
            final_barriers = [
            self._image_barrier(
                frame["image"], vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                vk.VK_IMAGE_LAYOUT_GENERAL,
                vk.VK_ACCESS_TRANSFER_READ_BIT, vk.VK_ACCESS_SHADER_WRITE_BIT,
            ),
            self._image_barrier(
                self.swapchain_images[image_index],
                vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                vk.VK_ACCESS_TRANSFER_WRITE_BIT, 0,
            ),
        ]
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
                0, None, 0, None, len(final_barriers), final_barriers,
            )
        if profiling:
            query_labels.append("present")
            vk.vkCmdWriteTimestamp(
                command, vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                self.wavefront_timestamp_query_pool,
                query_base + len(query_labels),
            )
        vk.vkCmdWriteTimestamp(
            command, vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            self.timestamp_query_pool, frame["query_start"] + 1,
        )
        vk.vkEndCommandBuffer(command)
        record_ms = (time.perf_counter() - record_start) * 1000.0
        submit_start = time.perf_counter()
        signal_semaphores = (
            [frame["external_ready"]] if self._headless_surface
            else [frame["render_finished"]]
        )
        if self._headless_surface:
            wait_semaphores = (
                [frame["external_release"]]
                if frame.get("external_release_wait") else []
            )
            wait_stage_masks = [
                vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
            ] * len(wait_semaphores)
        else:
            wait_semaphores = [frame["image_available"]]
            wait_stage_masks = [
                # Direct presentation transitions the acquired image at the
                # start of the command buffer. Waiting at compute was too late.
                vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                if self.swapchain_direct_storage
                else vk.VK_PIPELINE_STAGE_TRANSFER_BIT
            ]
        history_source_frame = self.window_frames[1 - frame_slot]
        history_chain_enabled = bool(
            self.wavefront_restir_runtime_enabled
            or wavefront_temporal_enabled
            or self.config.wavefront_indirect_reuse_temporal
        )
        history_wait_pending, history_signal_current = (
            _wavefront_history_semaphore_plan(
                current_pending=frame.get(
                    "wavefront_history_ready_pending", False
                ),
                previous_pending=history_source_frame.get(
                    "wavefront_history_ready_pending", False
                ),
                history_enabled=history_chain_enabled,
            )
        )
        if history_wait_pending:
            wait_semaphores.append(
                history_source_frame["wavefront_history_ready"]
            )
            history_consumer_stage = vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
            if self.resolved_execution_strategy == "ser":
                history_consumer_stage |= (
                    vk.VK_PIPELINE_STAGE_RAY_TRACING_SHADER_BIT_KHR
                )
            wait_stage_masks.append(history_consumer_stage)
        if history_signal_current:
            signal_semaphores.append(frame["wavefront_history_ready"])
        vk.vkQueueSubmit(
            self.queue, 1,
            [vk.VkSubmitInfo(
                sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                waitSemaphoreCount=len(wait_semaphores),
                pWaitSemaphores=wait_semaphores,
                pWaitDstStageMask=wait_stage_masks,
                commandBufferCount=1, pCommandBuffers=[command],
                signalSemaphoreCount=len(signal_semaphores),
                pSignalSemaphores=signal_semaphores,
            )], frame["fence"],
        )
        if history_wait_pending:
            history_source_frame["wavefront_history_ready_pending"] = False
        frame["wavefront_history_ready_pending"] = history_signal_current
        if self._headless_surface and wait_semaphores:
            frame["external_release_wait"] = False
        submit_ms = (time.perf_counter() - submit_start) * 1000.0
        present_start = time.perf_counter()
        if not self._headless_surface:
            try:
                self.queue_present(
                    self.queue,
                    vk.VkPresentInfoKHR(
                        sType=vk.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                        waitSemaphoreCount=1,
                        pWaitSemaphores=[frame["render_finished"]],
                        swapchainCount=1, pSwapchains=[self.swapchain],
                        pImageIndices=[image_index],
                    ),
                )
            except (vk.VkSuboptimalKhr, vk.VkErrorOutOfDateKhr):
                self.swapchain_extent = None
        present_ms = (time.perf_counter() - present_start) * 1000.0
        frame["has_timestamps"] = True
        frame["wavefront_render_scale"] = render_scale
        frame["wavefront_samples_per_pixel"] = effective_samples
        frame["wavefront_render_extent"] = (render_width, render_height)
        frame["wavefront_query_count"] = len(query_labels) + 1 if profiling else 0
        frame["wavefront_query_labels"] = tuple(query_labels)
        frame["wavefront_history_valid"] = bool(
            wavefront_temporal_enabled and accumulation_active
        )
        frame["wavefront_relax_history_valid"] = bool(
            self.config.denoiser_enabled
        )
        frame["wavefront_reservoir_valid"] = bool(
            self.wavefront_restir_runtime_enabled
        )
        self.wavefront_frame_sequence = (
            self.wavefront_frame_sequence + 1
        ) & 0x00ffffff
        self.wavefront_previous_present_camera = camera
        self._finish_accumulation_frame(
            accumulation_camera_signature, accumulation_active,
        )
        self.window_frame_index = (frame_slot + 1) % WINDOW_FRAMES_IN_FLIGHT
        # Scene upload, pipeline creation, and swapchain recreation are setup
        # events rather than animation cadence. Do not let one such interval
        # depress the displayed FPS for dozens of otherwise steady frames.
        if scene_ms > 5.0 or swapchain_ms > 5.0:
            self.wavefront_cadence_ms = 0.0
            self.wavefront_last_frame_start = None
        self.last_timings = {
            "wavefront_frame_slot": frame_slot,
            "wavefront_history_source_slot": (
                1 - frame_slot if history_wait_pending else None
            ),
            "wavefront_history_dependency_waited": history_wait_pending,
            "wavefront_history_chain_enabled": history_chain_enabled,
            "wavefront_frame_ms": (time.perf_counter() - frame_start) * 1000.0,
            "wavefront_cadence_ms": self.wavefront_cadence_ms,
            "wavefront_fps": (
                1000.0 / self.wavefront_cadence_ms
                if self.wavefront_cadence_ms > 0.0 else 0.0
            ),
            "wavefront_scene_ms": scene_ms,
            "wavefront_swapchain_ms": swapchain_ms,
            "wavefront_acquire_ms": acquire_ms,
            "wavefront_record_ms": record_ms,
            "wavefront_submit_ms": submit_ms,
            "wavefront_present_ms": present_ms,
            "wavefront_record_submit_ms": record_ms + submit_ms,
            "gpu_frame_ms": gpu_frame_ms,
            "fence_wait_ms": fence_wait_ms,
            "wavefront_tiles": tile_count,
            "blas_count": len(self.scene_resources.blases),
            "instance_count": len(self.scene_resources.instances),
            "shared_blas_savings": max(
                0,
                len(self.scene_resources.instances)
                - len(self.scene_resources.blases),
            ),
            "wavefront_render_extent": (render_width, render_height),
            "wavefront_render_scale": render_scale,
            "wavefront_interactive_resolution": bool(
                interactive_active and (
                    interactive_scale is not None
                    or self.interactive_dynamic_resolution is not None
                )
            ),
            "wavefront_interactive_dynamic_resolution": bool(
                interactive_active
                and self.interactive_dynamic_resolution is not None
            ),
            "wavefront_interactive_dynamic_gpu_ms": (
                self.interactive_dynamic_resolution.filtered_gpu_ms
                if self.interactive_dynamic_resolution is not None else 0.0
            ),
            "wavefront_dynamic_resolution": (
                self.dynamic_resolution is not None
            ),
            "wavefront_dynamic_gpu_ms": (
                self.dynamic_resolution.filtered_gpu_ms
                if self.dynamic_resolution is not None else 0.0
            ),
            "wavefront_samples_per_pixel": effective_samples,
            "wavefront_interactive_sample_scaling": bool(
                interactive_active and self.interactive_dynamic_samples is not None
            ),
            "wavefront_interactive_sample_gpu_ms": (
                self.interactive_dynamic_samples.filtered_gpu_ms
                if self.interactive_dynamic_samples is not None else 0.0
            ),
            "accumulation_state": self.accumulation_state.value,
            "accumulated_frames": self.accumulation_frame,
            "present_mode": self.present_mode_name,
            "direct_swapchain_storage": self.swapchain_direct_storage,
            "wavefront_execution_strategy": (
                self.resolved_execution_strategy
            ),
            "wavefront_megakernel_group_swizzle": (
                self.config.wavefront_megakernel_group_swizzle
            ),
            "wavefront_persistent_coarse_tiles": (
                self.resolved_execution_strategy == "persistent"
                and self.config.wavefront_persistent_coarse_tiles
            ),
            "wavefront_persistent_continuations": (
                self.resolved_execution_strategy == "hybrid"
                and self.config.wavefront_persistent_continuations
            ),
            "wavefront_scene_specialization": (
                "opaque" if self._use_opaque_scene_specialization(scene)
                else "generic"
            ),
            "wavefront_medium_stack_bytes": (
                self.wavefront_executor.medium_buffer.size
                if self.wavefront_executor is not None else 0
            ),
            "wavefront_texture_backend": (
                "native" if self.native_textures_enabled else "packed"
            ),
            "wavefront_material_bucketing": (
                self.config.wavefront_material_bucketing
            ),
            "wavefront_material_bucketing_start_bounce": (
                self.config.wavefront_material_bucketing_start_bounce
            ),
            "wavefront_restir_di": self.wavefront_restir_runtime_enabled,
            "wavefront_restir_reservoirs": (
                self.config.wavefront_restir_reservoirs
                if self.wavefront_restir_runtime_enabled else 0
            ),
            "wavefront_restir_spatial_reuse": (
                self.wavefront_restir_runtime_enabled
                and self.config.wavefront_restir_spatial_reuse
            ),
            "wavefront_restir_pairwise_mis": (
                self.wavefront_restir_runtime_enabled
                and self.config.wavefront_restir_pairwise_mis
            ),
            "wavefront_restir_generalized_mis": (
                self.wavefront_restir_runtime_enabled
                and self.config.wavefront_restir_generalized_mis
            ),
            "wavefront_restir_generalized_balance_cap": (
                self.config.wavefront_restir_generalized_balance_cap
            ),
            "wavefront_unified_secondary_nee": (
                self.config.wavefront_unified_secondary_nee
            ),
            "wavefront_unified_primary_restir": (
                self.wavefront_restir_runtime_enabled
                and self.config.wavefront_unified_primary_restir
            ),
            "wavefront_stratified_primary_restir": (
                self.wavefront_restir_runtime_enabled
                and self.config.wavefront_stratified_primary_restir
            ),
            "wavefront_restir_reservoir_bytes": sum(
                frame["wavefront_reservoir_buffer"].size
                for frame in self.window_frames
                if frame.get("wavefront_reservoir_buffer") is not None
            ) if self.config.wavefront_restir_di else 0,
            "wavefront_indirect_reuse_storage": (
                self.config.wavefront_indirect_reuse_storage
            ),
            "wavefront_indirect_reuse_candidates": (
                self.config.wavefront_indirect_reuse_candidates
            ),
            "wavefront_indirect_reuse_temporal": (
                self.config.wavefront_indirect_reuse_temporal
            ),
            "wavefront_indirect_reuse_spatial": (
                self.config.wavefront_indirect_reuse_spatial
            ),
            "wavefront_indirect_reuse_apply": (
                self.config.wavefront_indirect_reuse_apply
            ),
            "wavefront_indirect_reuse_debug_view": (
                self.config.wavefront_indirect_reuse_debug_view
            ),
            "wavefront_indirect_reservoir_bytes": sum(
                frame["wavefront_indirect_reservoir_buffer"].size
                for frame in self.window_frames
                if frame.get("wavefront_indirect_reservoir_buffer") is not None
            ),
            "wavefront_indirect_seed_bytes": sum(
                frame["wavefront_indirect_seed_buffer"].size
                for frame in self.window_frames
                if frame.get("wavefront_indirect_seed_buffer") is not None
            ),
            "wavefront_indirect_reservoir_extent": next((
                frame.get("wavefront_indirect_reservoir_extent")
                for frame in self.window_frames
                if frame.get("wavefront_indirect_reservoir_extent") is not None
            ), None),
            "wavefront_gbuffer_bytes": (
                2 * render_width * render_height * (
                    4 + 4 + (4 if (
                        self.config.wavefront_restir_di
                        or self.config.wavefront_indirect_reuse_storage) else 0)
                )
                if (
                    wavefront_temporal_enabled
                    or self.config.wavefront_diffuse_filter
                    or self.config.wavefront_restir_di
                    or self.config.wavefront_indirect_reuse_storage
                ) else 0
            ),
            "wavefront_temporal_history_valid": history_valid,
            "wavefront_restir_history_valid": restir_history_valid,
            "wavefront_indirect_history_valid": indirect_history_valid,
            "wavefront_temporal_motion_pixels": camera_motion_pixels,
            "wavefront_temporal_motion_valid": temporal_motion_valid,
            "wavefront_indirect_reuse_effective_history_limit": (
                indirect_history_limit
            ),
            "wavefront_restir_effective_history_limit": restir_history_limit,
            "wavefront_command_cache_hit": command_cache_hit,
            "wavefront_stage_ms": stage_timings,
            "wavefront_work_counters": work_counters,
            "wavefront_indirect_reuse_counters": indirect_reuse_counters,
            "wavefront_indirect_reuse_metrics": indirect_reuse_metrics,
            "wavefront_work_counter_scope": {
                "megakernel": "full_path",
                "persistent": "full_path",
                "ser": "full_path",
                "hybrid": "inline_prefix",
                "wavefront": "full_path",
            }[self.resolved_execution_strategy],
        }
        return (width, height)

    def export_window_frame(self, slot, *, release, pixel_format="rgba8"):
        """Wrap one submitted frame as externally importable Vulkan memory."""
        from ...gpu import GpuFrame, VulkanBufferMetadata, VulkanImageMetadata

        if not self._headless_surface:
            raise RuntimeError("external image interop is not enabled")
        if not 0 <= int(slot) < len(self.window_frames):
            raise ValueError("frame slot is out of range")
        frame = self.window_frames[int(slot)]
        pixel_format = str(pixel_format).lower()
        if pixel_format not in {"rgba8", "nv12", "p010"}:
            raise ValueError("pixel_format must be 'rgba8', 'nv12', or 'p010'")
        export_buffer = (
            frame["nv12_buffer"]
            if pixel_format in {"nv12", "p010"} else None
        )

        def export_memory_fd():
            info = vk.VkMemoryGetFdInfoKHR(
                memory=(
                    export_buffer.memory if export_buffer is not None
                    else frame["image_memory"]
                ),
                handleType=vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT,
            )
            descriptor = vk.ffi.new("int *")
            result = self.get_memory_fd(
                self.device, vk.ffi.addressof(info), descriptor
            )
            if result != vk.VK_SUCCESS:
                raise RuntimeError(f"Vulkan memory FD export failed: {result}")
            return descriptor[0]

        def export_release_semaphore_fd():
            info = vk.VkSemaphoreGetFdInfoKHR(
                semaphore=frame["external_release"],
                handleType=vk.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
            )
            descriptor = vk.ffi.new("int *")
            result = self.get_semaphore_fd(
                self.device, vk.ffi.addressof(info), descriptor
            )
            if result != vk.VK_SUCCESS:
                raise RuntimeError(
                    f"Vulkan release-semaphore FD export failed: {result}"
                )
            return descriptor[0]

        def export_semaphore_fd():
            info = vk.VkSemaphoreGetFdInfoKHR(
                semaphore=frame["external_ready"],
                handleType=vk.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT,
            )
            descriptor = vk.ffi.new("int *")
            result = self.get_semaphore_fd(
                self.device, vk.ffi.addressof(info), descriptor
            )
            if result != vk.VK_SUCCESS:
                raise RuntimeError(
                    f"Vulkan semaphore FD export failed: {result}"
                )
            return descriptor[0]

        def wait_for_frame(timeout):
            timeout_ns = (
                (1 << 64) - 1 if timeout is None
                else max(0, int(float(timeout) * 1_000_000_000))
            )
            try:
                vk.vkWaitForFences(
                    self.device, 1, [frame["fence"]], vk.VK_TRUE, timeout_ns
                )
            except vk.VkTimeout:
                return False
            return True

        if pixel_format in {"nv12", "p010"}:
            p010 = pixel_format == "p010"
            metadata = VulkanBufferMetadata(
                width=int(self.swapchain_extent[0]),
                height=int(self.swapchain_extent[1]),
                format="P010" if p010 else "NV12",
                pitch=int(
                    frame["p010_pitch"] if p010 else frame["nv12_pitch"]
                ),
                y_offset=0,
                uv_offset=int(
                    frame["p010_uv_offset"]
                    if p010 else frame["nv12_uv_offset"]
                ),
                memory_size=int(frame["nv12_buffer"].size),
                memory_offset=0,
                dedicated_allocation=True,
                device_uuid=self.device_uuid,
                buffer_handle=int(vk.ffi.cast(
                    "uintptr_t", frame["nv12_buffer"].buffer
                )),
                memory_handle=int(vk.ffi.cast(
                    "uintptr_t", frame["nv12_buffer"].memory
                )),
                device_handle=int(vk.ffi.cast("uintptr_t", self.device)),
                physical_device_handle=int(vk.ffi.cast(
                    "uintptr_t", self.physical_device
                )),
                completion_fence_handle=int(vk.ffi.cast(
                    "uintptr_t", frame["fence"]
                )),
                queue_family_index=int(self.queue_family),
                bit_depth=10 if p010 else 8,
                storage_bits=16 if p010 else 8,
            )
            attributes = {
                "color_space": "bt709",
                "color_range": "limited",
                "components": pixel_format,
                "bit_depth": 10 if p010 else 8,
                "frame_slot": int(slot),
                "accumulation_state": self.accumulation_state.value,
                "accumulated_frames": int(self.accumulation_frame),
                "render_extent": tuple(frame["wavefront_render_extent"]),
                "samples_per_pixel": int(frame["wavefront_samples_per_pixel"]),
            }
        else:
            metadata = VulkanImageMetadata(
                width=int(self.swapchain_extent[0]),
                height=int(self.swapchain_extent[1]),
                format="VK_FORMAT_R8G8B8A8_UNORM",
                format_value=int(vk.VK_FORMAT_R8G8B8A8_UNORM),
                layout="VK_IMAGE_LAYOUT_GENERAL",
                memory_size=int(frame["image_memory_size"]),
                memory_offset=0,
                dedicated_allocation=True,
                device_uuid=self.device_uuid,
                image_handle=int(vk.ffi.cast("uintptr_t", frame["image"])),
                memory_handle=int(vk.ffi.cast(
                    "uintptr_t", frame["image_memory"]
                )),
                device_handle=int(vk.ffi.cast("uintptr_t", self.device)),
                physical_device_handle=int(vk.ffi.cast(
                    "uintptr_t", self.physical_device
                )),
                completion_fence_handle=int(vk.ffi.cast(
                    "uintptr_t", frame["fence"]
                )),
                queue_family_index=int(self.queue_family),
            )
            attributes = {
                "color_space": "srgb-transfer",
                "components": "rgba",
                "frame_slot": int(slot),
                "accumulation_state": self.accumulation_state.value,
                "accumulated_frames": int(self.accumulation_frame),
                "render_extent": tuple(frame["wavefront_render_extent"]),
                "samples_per_pixel": int(frame["wavefront_samples_per_pixel"]),
            }
        return GpuFrame(
            api="vulkan",
            metadata=metadata,
            export_memory_fd=export_memory_fd,
            export_ready_semaphore_fd=export_semaphore_fd,
            export_release_semaphore_fd=export_release_semaphore_fd,
            wait=wait_for_frame,
            close=release,
            attributes=attributes,
        )

    def recycle_external_frame(self, slot, release_semaphore_exported=False):
        """Reset one external binary semaphore after its consumer is done."""
        frame = self.window_frames[int(slot)]
        if release_semaphore_exported:
            frame["external_release_wait"] = True
            return
        vk.vkWaitForFences(
            self.device, 1, [frame["fence"]], vk.VK_TRUE, (1 << 64) - 1
        )
        vk.vkDestroySemaphore(self.device, frame["external_ready"], None)
        frame["external_ready"] = vk.vkCreateSemaphore(
            self.device,
            vk.VkSemaphoreCreateInfo(
                sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
                pNext=vk.VkExportSemaphoreCreateInfo(
                    handleTypes=(
                        vk.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT
                    )
                ),
            ),
            None,
        )

    def _wait_external_releases(self):
        """Drain consumer release signals before destroying shared resources."""
        if self.device is None:
            return
        frames = [
            frame for frame in self.window_frames
            if frame.get("external_release_wait")
        ]
        if not frames:
            return
        fence = vk.vkCreateFence(
            self.device,
            vk.VkFenceCreateInfo(sType=vk.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO),
            None,
        )
        try:
            semaphores = [frame["external_release"] for frame in frames]
            stages = [vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT] * len(semaphores)
            vk.vkQueueSubmit(
                self.queue, 1,
                [vk.VkSubmitInfo(
                    sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    waitSemaphoreCount=len(semaphores),
                    pWaitSemaphores=semaphores,
                    pWaitDstStageMask=stages,
                    commandBufferCount=0, pCommandBuffers=None,
                    signalSemaphoreCount=0, pSignalSemaphores=None,
                )], fence,
            )
            vk.vkWaitForFences(
                self.device, 1, [fence], vk.VK_TRUE, (1 << 64) - 1
            )
            for frame in frames:
                frame["external_release_wait"] = False
        finally:
            vk.vkDestroyFence(self.device, fence, None)

    def capture_wavefront_hdr(self):
        """Synchronously read the most recently presented internal HDR image."""
        if not self.config.wavefront_hdr_capture:
            raise RuntimeError(
                "HDR capture requires RendererConfig(wavefront_hdr_capture=True)"
            )
        if not self.window_frames:
            raise RuntimeError("no wavefront frame has been presented")
        slot = (self.window_frame_index - 1) % WINDOW_FRAMES_IN_FLIGHT
        frame = self.window_frames[slot]
        capture = frame.get("wavefront_hdr_capture_buffer")
        extent = frame.get("wavefront_render_extent")
        if capture is None or extent is None:
            raise RuntimeError("no wavefront HDR capture is available")
        width, height = extent
        size = width * height * 8
        vk.vkWaitForFences(
            self.device, 1, [frame["fence"]], vk.VK_TRUE, (1 << 64) - 1
        )
        mapped = vk.vkMapMemory(self.device, capture.memory, 0, size, 0)
        image = np.frombuffer(
            mapped, dtype=np.float16, count=width * height * 4
        ).copy().reshape((height, width, 4)).astype(np.float32)
        vk.vkUnmapMemory(self.device, capture.memory)
        return image

    def _begin_accumulation_frame(self, scene_key, camera):
        """Resolve motion policy before recording a conventional or wavefront frame."""
        now = time.perf_counter()
        camera_signature = _camera_signature(camera)
        camera_changed = (
            camera_signature != self.accumulation_camera_signature
        )
        scene_changed = scene_key != self.accumulation_key
        content_changed = camera_changed or scene_changed
        if content_changed:
            self.camera_change_time = now

        progressive = self.config.progressive_accumulation
        stationary = progressive and self.config.stationary_accumulation
        if not progressive:
            state = AccumulationState.DISABLED
        elif stationary and content_changed:
            state = AccumulationState.MOVING
        elif (
            stationary
            and now - self.camera_change_time
            < self.config.stationary_delay_seconds
        ):
            state = AccumulationState.SETTLING
        else:
            state = AccumulationState.ACCUMULATING

        reset_history = (
            not progressive
            or scene_changed
            or (camera_changed and not self.config.temporal_history)
            or (stationary and state is not AccumulationState.ACCUMULATING)
        )
        if reset_history:
            self.accumulation_frame = 0
            self.accumulation_history_valid = False
        self.accumulation_key = scene_key
        self.accumulation_state = state
        interactive_active = (
            content_changed
            or now - self.camera_change_time
            < self.config.stationary_delay_seconds
        )
        return (
            camera_signature,
            state is AccumulationState.ACCUMULATING,
            interactive_active,
        )

    def _finish_accumulation_frame(
        self, camera_signature, accumulation_active,
    ):
        self.accumulation_camera_signature = camera_signature
        if self.config.progressive_accumulation and accumulation_active:
            self.accumulation_frame += 1
            self.accumulation_history_valid = True

    def present_window(
        self, scene, camera, width, height, overlay_fps=None, max_bounces=5,
        samples=1,
    ):
        """Trace and present a frame without host readback."""
        self.prepare_window_scene(scene)
        if (
            self.swapchain_extent != (width, height)
            or self.swapchain_wavefront_only is not False
        ):
            self.create_window_swapchain(width, height)
        width, height = self.swapchain_extent
        scene_key = (id(scene), scene.revision, width, height)
        camera_signature, accumulation_active, interactive_active = (
            self._begin_accumulation_frame(scene_key, camera)
        )
        effective_samples = samples
        interactive_samples = self.config.interactive_samples_per_pixel
        if (
            interactive_samples is not None
            and interactive_active
        ):
            effective_samples = interactive_samples
        self.effective_samples_per_pixel = effective_samples
        presented_extent = self.swapchain_extent
        frame = self.window_frames[self.window_frame_index]
        present_start = time.perf_counter()
        stage_start = present_start
        timings = {}
        if self.present_pacing_enabled and self.last_present_id:
            result = self.wait_for_present(
                self.device,
                self.swapchain,
                self.last_present_id,
                (1 << 64) - 1,
            )
            if result != vk.VK_SUCCESS:
                raise RuntimeError(f"vkWaitForPresentKHR failed: {result}")
        timings["present_wait_ms"] = (time.perf_counter() - stage_start) * 1000.0
        stage_start = time.perf_counter()
        vk.vkWaitForFences(
            self.device, 1, [frame["fence"]], vk.VK_TRUE, (1 << 64) - 1
        )
        timings["fence_wait_ms"] = (time.perf_counter() - stage_start) * 1000.0
        stage_start = time.perf_counter()
        if frame["has_timestamps"]:
            timestamps = vk.ffi.new("uint64_t[]", 2)
            vk.vkGetQueryPoolResults(
                self.device,
                self.timestamp_query_pool,
                frame["query_start"],
                2,
                vk.ffi.sizeof(timestamps),
                timestamps,
                vk.ffi.sizeof("uint64_t"),
                vk.VK_QUERY_RESULT_64_BIT | vk.VK_QUERY_RESULT_WAIT_BIT,
            )
            timings["gpu_frame_ms"] = (
                (timestamps[1] - timestamps[0]) * self.timestamp_period / 1_000_000.0
            )
        timings["timestamp_read_ms"] = (time.perf_counter() - stage_start) * 1000.0
        stage_start = time.perf_counter()
        try:
            image_index = self.acquire_next_image(
                self.device, self.swapchain, (1 << 64) - 1,
                frame["image_available"], vk.VK_NULL_HANDLE,
            )
        except (vk.VkSuboptimalKhr, vk.VkErrorOutOfDateKhr):
            # The binding discards the acquired image index for SUBOPTIMAL but
            # may already have signaled this semaphore. Replace it before reuse.
            vk.vkDeviceWaitIdle(self.device)
            vk.vkDestroySemaphore(self.device, frame["image_available"], None)
            frame["image_available"] = vk.vkCreateSemaphore(
                self.device,
                vk.VkSemaphoreCreateInfo(
                    sType=vk.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO
                ),
                None,
            )
            self.create_window_swapchain(width, height)
            return self.present_window(
                scene, camera, width, height, overlay_fps=overlay_fps,
                max_bounces=max_bounces,
                samples=samples,
            )
        timings["acquire_ms"] = (time.perf_counter() - stage_start) * 1000.0
        vk.vkResetFences(self.device, 1, [frame["fence"]])

        stage_start = time.perf_counter()
        timings["descriptor_update_ms"] = (
            time.perf_counter() - stage_start
        ) * 1000.0
        stage_start = time.perf_counter()
        temporal_history = bool(
            self.config.temporal_history
            and isinstance(camera, PerspectiveCamera)
            and (
                self.previous_camera is None
                or isinstance(self.previous_camera, PerspectiveCamera)
            )
        )
        constants = self._camera_constants(
            camera, width, height, overlay_fps, max_bounces=max_bounces,
            samples=effective_samples, accumulation_frame=self.accumulation_frame,
            previous_camera=self.previous_camera or camera,
            temporal_history=temporal_history,
            history_valid=(self.accumulation_history_valid and temporal_history),
            temporal_history_limit=self.config.temporal_history_limit,
            temporal_neighborhood_clamping=(
                self.config.temporal_neighborhood_clamping
            ),
            adaptive_sampling=self.config.adaptive_sampling,
            adaptive_variance_threshold=self.config.adaptive_variance_threshold,
            adaptive_min_samples=self.config.adaptive_min_samples,
            light_count=scene.analytic_light_count,
            area_light_count=scene.emissive_triangle_count,
            area_light_weight=scene.emissive_light_weight,
            area_light_samples=self.config.area_light_samples,
            denoiser_enabled=self.config.denoiser_enabled,
            denoiser_variance_threshold=(
                self.config.denoiser_variance_threshold
            ),
        )
        timings["camera_constants_ms"] = (
            time.perf_counter() - stage_start
        ) * 1000.0
        timings["samples_per_pixel"] = effective_samples

        def record_trace(context):
            command = context["command"]
            vk.vkCmdResetQueryPool(
                command, self.timestamp_query_pool, frame["query_start"], 2
            )
            vk.vkCmdWriteTimestamp(
                command,
                vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                self.timestamp_query_pool,
                frame["query_start"],
            )
            vk.vkCmdPipelineBarrier(
                command,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, None, 0, None,
                len(self.accumulation_images) + len(self.gbuffer_images)
                + len(self.moment_images),
                [self._image_barrier(
                    image,
                    vk.VK_IMAGE_LAYOUT_GENERAL,
                    vk.VK_IMAGE_LAYOUT_GENERAL,
                    vk.VK_ACCESS_SHADER_WRITE_BIT,
                    vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
                ) for image in (
                    self.accumulation_images + self.gbuffer_images + self.moment_images
                )],
            )
            vk.vkCmdBindPipeline(command, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline)
            vk.vkCmdBindDescriptorSets(
                command, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_layout,
                0, 1, [frame["descriptor_set"]], 0, None,
            )
            vk.vkCmdPushConstants(
                command, self.pipeline_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT,
                0, constants.nbytes, vk.ffi.from_buffer(constants),
            )
            vk.vkCmdDispatch(command, (width + 7) // 8, (height + 7) // 8, 1)

        def record_denoise(context):
            command = context["command"]
            if not self.denoiser_output_enabled:
                return
            for iteration in range(self.config.denoiser_iterations):
                source_images = (
                    self.accumulation_images + self.gbuffer_images
                    + self.moment_images + self.denoise_images
                )
                barriers = [self._image_barrier(
                    image,
                    vk.VK_IMAGE_LAYOUT_GENERAL,
                    vk.VK_IMAGE_LAYOUT_GENERAL,
                    vk.VK_ACCESS_SHADER_WRITE_BIT,
                    vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
                ) for image in source_images]
                vk.vkCmdPipelineBarrier(
                    command,
                    vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0, 0, None, 0, None, len(barriers), barriers,
                )
                vk.vkCmdBindPipeline(
                    command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                    self.denoise_pipeline,
                )
                vk.vkCmdBindDescriptorSets(
                    command, vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                    self.pipeline_layout, 0, 1, [frame["descriptor_set"]],
                    0, None,
                )
                denoise_constants = constants.copy()
                denoise_constants[10, 3] = float(iteration + 1)
                vk.vkCmdPushConstants(
                    command, self.pipeline_layout,
                    vk.VK_SHADER_STAGE_COMPUTE_BIT, 0,
                    denoise_constants.nbytes,
                    vk.ffi.from_buffer(denoise_constants),
                )
                vk.vkCmdDispatch(
                    command, (width + 7) // 8, (height + 7) // 8, 1
                )

        def record_tone(context):
            command = context["command"]
            tone_source_images = (
                self.denoise_images
                if self.denoiser_output_enabled
                else self.accumulation_images
            )
            history_barriers = [
                self._image_barrier(
                    image,
                    vk.VK_IMAGE_LAYOUT_GENERAL,
                    vk.VK_IMAGE_LAYOUT_GENERAL,
                    vk.VK_ACCESS_SHADER_WRITE_BIT,
                    vk.VK_ACCESS_SHADER_READ_BIT,
                )
                for image in tone_source_images
            ]
            vk.vkCmdPipelineBarrier(
                command,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 0, None, 0, None,
                len(history_barriers), history_barriers,
            )
            vk.vkCmdBindPipeline(
                command, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.tone_pipeline
            )
            vk.vkCmdBindDescriptorSets(
                command, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self.pipeline_layout,
                0, 1, [frame["descriptor_set"]], 0, None,
            )
            tone_constants = constants.copy()
            if self.denoiser_output_enabled:
                tone_constants[10, 3] = float(self.config.denoiser_iterations)
            else:
                # Base constants use this lane to request moment output from
                # tracing. Tone mapping instead interprets it as the filtered
                # image selector, so raw presentation must clear it explicitly.
                tone_constants[10, 3] = 0.0
            vk.vkCmdPushConstants(
                command, self.pipeline_layout, vk.VK_SHADER_STAGE_COMPUTE_BIT,
                0, tone_constants.nbytes, vk.ffi.from_buffer(tone_constants),
            )
            vk.vkCmdDispatch(command, (width + 7) // 8, (height + 7) // 8, 1)

        def record_present(context):
            command = context["command"]
            barriers = [
                self._image_barrier(
                    frame["image"], vk.VK_IMAGE_LAYOUT_GENERAL,
                    vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    vk.VK_ACCESS_SHADER_WRITE_BIT, vk.VK_ACCESS_TRANSFER_READ_BIT,
                ),
                self._image_barrier(
                    self.swapchain_images[image_index], vk.VK_IMAGE_LAYOUT_UNDEFINED,
                    vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    0, vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                ),
            ]
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                0, None, 0, None, len(barriers), barriers,
            )
            blit = vk.VkImageBlit(
                srcSubresource=vk.VkImageSubresourceLayers(
                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, mipLevel=0,
                    baseArrayLayer=0, layerCount=1,
                ),
                srcOffsets=[vk.VkOffset3D(x=0, y=0, z=0), vk.VkOffset3D(x=width, y=height, z=1)],
                dstSubresource=vk.VkImageSubresourceLayers(
                    aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, mipLevel=0,
                    baseArrayLayer=0, layerCount=1,
                ),
                dstOffsets=[vk.VkOffset3D(x=0, y=0, z=0), vk.VkOffset3D(x=width, y=height, z=1)],
            )
            vk.vkCmdBlitImage(
                command, frame["image"], vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                self.swapchain_images[image_index], vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, [blit], vk.VK_FILTER_NEAREST,
            )
            final_barriers = [
                self._image_barrier(
                    frame["image"], vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    vk.VK_IMAGE_LAYOUT_GENERAL,
                    vk.VK_ACCESS_TRANSFER_READ_BIT, vk.VK_ACCESS_SHADER_WRITE_BIT,
                ),
                self._image_barrier(
                    self.swapchain_images[image_index], vk.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                    vk.VK_ACCESS_TRANSFER_WRITE_BIT, 0,
                ),
            ]
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0,
                0, None, 0, None, len(final_barriers), final_barriers,
            )
            vk.vkCmdWriteTimestamp(
                command,
                vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                self.timestamp_query_pool,
                frame["query_start"] + 1,
            )

        self.window_pipeline_stage_names = self.window_pipeline.stage_names

        stage_start = time.perf_counter()
        vk.vkResetCommandBuffer(frame["command"], 0)
        vk.vkBeginCommandBuffer(
            frame["command"],
            vk.VkCommandBufferBeginInfo(
                sType=vk.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                flags=vk.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            ),
        )
        self.window_pipeline.record({
            "command": frame["command"],
            "record_trace": record_trace,
            "record_denoise": record_denoise,
            "record_tone": record_tone,
            "record_present": record_present,
        })
        vk.vkEndCommandBuffer(frame["command"])
        timings["command_record_ms"] = (
            time.perf_counter() - stage_start
        ) * 1000.0
        stage_start = time.perf_counter()
        vk.vkQueueSubmit(
            self.queue,
            1,
            [
                vk.VkSubmitInfo(
                    sType=vk.VK_STRUCTURE_TYPE_SUBMIT_INFO,
                    waitSemaphoreCount=1,
                    pWaitSemaphores=[frame["image_available"]],
                    pWaitDstStageMask=[vk.VK_PIPELINE_STAGE_TRANSFER_BIT],
                    commandBufferCount=1,
                    pCommandBuffers=[frame["command"]],
                    signalSemaphoreCount=1,
                    pSignalSemaphores=[frame["render_finished"]],
                )
            ],
            frame["fence"],
        )
        timings["queue_submit_ms"] = (time.perf_counter() - stage_start) * 1000.0
        stage_start = time.perf_counter()
        present_id_info = None
        if self.present_pacing_enabled:
            self.present_id += 1
            present_id_info = vk.VkPresentIdKHR(
                swapchainCount=1,
                pPresentIds=[self.present_id],
            )
        try:
            self.queue_present(
                self.queue,
                vk.VkPresentInfoKHR(
                    sType=vk.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
                    pNext=present_id_info,
                    waitSemaphoreCount=1,
                    pWaitSemaphores=[frame["render_finished"]],
                    swapchainCount=1,
                    pSwapchains=[self.swapchain],
                    pImageIndices=[image_index],
                ),
            )
            if self.present_pacing_enabled:
                self.last_present_id = self.present_id
        except vk.VkSuboptimalKhr:
            # Presentation succeeded, but GLFW's framebuffer extent changed
            # during maximize/resize. Recreate safely at the next frame.
            self.swapchain_extent = None
        except vk.VkErrorOutOfDateKhr:
            self.swapchain_extent = None
        timings["queue_present_ms"] = (time.perf_counter() - stage_start) * 1000.0
        timings["direct_present_ms"] = (
            time.perf_counter() - present_start
        ) * 1000.0
        timings["frame_slot"] = self.window_frame_index
        timings["swapchain_generation"] = self.swapchain_generation
        timings["blas_count"] = len(self.scene_resources.blases)
        timings["instance_count"] = len(self.scene_resources.instances)
        timings["shared_blas_savings"] = max(
            0,
            len(self.scene_resources.instances) - len(self.scene_resources.blases),
        )
        self.last_timings = timings
        self._finish_accumulation_frame(
            camera_signature, accumulation_active,
        )
        self.previous_camera = camera
        frame["has_timestamps"] = True
        self.window_frame_index = (
            self.window_frame_index + 1
        ) % WINDOW_FRAMES_IN_FLIGHT
        return presented_extent

    def reset_accumulation(self):
        self.accumulation_frame = 0
        self.accumulation_key = None
        self.accumulation_history_valid = False
        self.accumulation_camera_signature = None
        self.accumulation_render_extent = None
        self.accumulation_state = (
            AccumulationState.SETTLING
            if self.config.progressive_accumulation
            and self.config.stationary_accumulation
            else AccumulationState.ACCUMULATING
            if self.config.progressive_accumulation
            else AccumulationState.DISABLED
        )
        self.previous_camera = None
        self.camera_change_time = time.perf_counter()

    def set_object_effect(self, triangle_range=None, effect=None):
        """Compatibility helper replacing all effects with at most one."""
        return self.set_object_effects(
            () if triangle_range is None else ((triangle_range, effect),)
        )

    def set_object_effects(self, bindings):
        """Replace the ordered set of transient packed-triangle effects."""
        bindings = tuple(bindings)
        if len(bindings) > 4:
            raise ValueError("Vulkan currently supports at most four object effects")
        normalized = []
        supported = (
            Outline, Tint, EmissiveHighlight, Isolation, BoundingBox, XRay,
        )
        for triangle_range, effect in bindings:
            if not self.config.object_effects:
                raise RuntimeError("object effects require object_effects=True")
            if not isinstance(effect, ObjectEffect):
                raise TypeError("effect must be an ordinarylight.effects object")
            if not isinstance(effect, supported):
                raise ValueError(
                    f"unsupported Vulkan object effect: {type(effect).__name__}"
                )
            if len(triangle_range) != 2:
                raise ValueError("triangle_range must contain start and end")
            start, end = (int(item) for item in triangle_range)
            if start < 0 or end <= start or end > (1 << 24):
                raise ValueError("triangle_range must be a non-empty 24-bit range")
            normalized.append(((start, end), effect))
        value = tuple(normalized)
        if value == self.object_effect_bindings:
            return value
        self.object_effect_bindings = value
        for frame in self.window_frames:
            frame["wavefront_command_key"] = None
            frame["wavefront_reservoir_valid"] = False
        return value

    @property
    def object_effect_triangle_range(self):
        return self.object_effect_bindings[0][0] if self.object_effect_bindings else None

    @property
    def object_effect(self):
        return self.object_effect_bindings[0][1] if self.object_effect_bindings else None

    def set_wavefront_restir_enabled(self, enabled):
        """Select conventional or ReSTIR direct lighting between frames."""
        enabled = bool(enabled)
        if enabled and not self.config.wavefront_restir_di:
            raise RuntimeError(
                "Runtime ReSTIR requires wavefront_restir_di=True when the "
                "renderer is constructed so reservoir resources are allocated"
            )
        if enabled == self.wavefront_restir_runtime_enabled:
            return enabled
        if self.device:
            vk.vkDeviceWaitIdle(self.device)
            self._reset_wavefront_history_chain()
        self.wavefront_restir_runtime_enabled = enabled
        for frame in self.window_frames:
            frame["wavefront_reservoir_valid"] = False
            frame["wavefront_command_key"] = None
        return enabled

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self.device:
            device_lost = False
            try:
                self._wait_external_releases()
                vk.vkDeviceWaitIdle(self.device)
            except vk.VkErrorDeviceLost:
                # Device loss invalidates pending work, but Vulkan objects and
                # the logical device still need orderly teardown.  Do not let
                # the secondary wait failure hide the operation that caused it.
                device_lost = True
            if not device_lost:
                self._save_pipeline_cache()
            self._destroy_swapchain_resources()
            commands = [
                command
                for frame in self.window_frames
                for command in (frame["command"], frame["wavefront_command"])
            ]
            if commands:
                vk.vkFreeCommandBuffers(
                    self.device, self.command_pool, len(commands), commands
                )
            for frame in self.window_frames:
                vk.vkDestroyFence(self.device, frame["fence"], None)
                vk.vkDestroySemaphore(self.device, frame["image_available"], None)
                vk.vkDestroySemaphore(self.device, frame["render_finished"], None)
                vk.vkDestroySemaphore(
                    self.device, frame["wavefront_history_ready"], None
                )
                if frame.get("external_ready") is not None:
                    vk.vkDestroySemaphore(
                        self.device, frame["external_ready"], None
                    )
                if frame.get("external_release") is not None:
                    vk.vkDestroySemaphore(
                        self.device, frame["external_release"], None
                    )
            self.window_frames.clear()
            if self.timestamp_query_pool:
                vk.vkDestroyQueryPool(self.device, self.timestamp_query_pool, None)
                self.timestamp_query_pool = None
            if self.wavefront_timestamp_query_pool:
                vk.vkDestroyQueryPool(
                    self.device, self.wavefront_timestamp_query_pool, None
                )
                self.wavefront_timestamp_query_pool = None
            if self.scene_resources is not None:
                self.scene_resources.close()
                self.scene_resources = None
            if self.wavefront_executor is not None:
                self.wavefront_executor.close()
                self.wavefront_executor = None
            self._release_frame_resources()
            if self.descriptor_pool:
                vk.vkDestroyDescriptorPool(self.device, self.descriptor_pool, None)
            if self.nv12_descriptor_pool:
                vk.vkDestroyDescriptorPool(
                    self.device, self.nv12_descriptor_pool, None
                )
            if self.pipeline:
                vk.vkDestroyPipeline(self.device, self.pipeline, None)
            if self.tone_pipeline:
                vk.vkDestroyPipeline(self.device, self.tone_pipeline, None)
            if self.denoise_pipeline:
                vk.vkDestroyPipeline(self.device, self.denoise_pipeline, None)
            if self.nv12_pipeline:
                vk.vkDestroyPipeline(self.device, self.nv12_pipeline, None)
            if self.p010_pipeline:
                vk.vkDestroyPipeline(self.device, self.p010_pipeline, None)
            if self.shader_module:
                vk.vkDestroyShaderModule(self.device, self.shader_module, None)
            if self.tone_shader_module:
                vk.vkDestroyShaderModule(self.device, self.tone_shader_module, None)
            if self.denoise_shader_module:
                vk.vkDestroyShaderModule(self.device, self.denoise_shader_module, None)
            if self.nv12_shader_module:
                vk.vkDestroyShaderModule(
                    self.device, self.nv12_shader_module, None
                )
            if self.p010_shader_module:
                vk.vkDestroyShaderModule(
                    self.device, self.p010_shader_module, None
                )
            if self.nv12_pipeline_layout:
                vk.vkDestroyPipelineLayout(
                    self.device, self.nv12_pipeline_layout, None
                )
            if self.nv12_descriptor_layout:
                vk.vkDestroyDescriptorSetLayout(
                    self.device, self.nv12_descriptor_layout, None
                )
            if self.pipeline_layout:
                vk.vkDestroyPipelineLayout(self.device, self.pipeline_layout, None)
            if self.descriptor_layout:
                vk.vkDestroyDescriptorSetLayout(self.device, self.descriptor_layout, None)
            if self.command_pool:
                vk.vkDestroyCommandPool(self.device, self.command_pool, None)
            if self.pipeline_cache:
                vk.vkDestroyPipelineCache(
                    self.device, self.pipeline_cache, None
                )
                self.pipeline_cache = None
            vk.vkDestroyDevice(self.device, None)
            self.device = None
        if self.instance:
            if self.surface and self._owns_surface:
                destroy_surface = vk.vkGetInstanceProcAddr(self.instance, "vkDestroySurfaceKHR")
                destroy_surface(self.instance, self.surface, None)
            self.surface = None
            if self._owns_instance:
                vk.vkDestroyInstance(self.instance, None)
            self.instance = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()
