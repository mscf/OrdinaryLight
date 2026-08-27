"""Vulkan hardware ray-tracing capability discovery and public API."""

from dataclasses import dataclass
import math

import numpy as np

from .materials import MaterialProgram


REQUIRED_RAY_QUERY_EXTENSIONS = frozenset(
    {
        "VK_KHR_acceleration_structure",
        "VK_KHR_deferred_host_operations",
        "VK_KHR_ray_query",
    }
)


@dataclass(frozen=True)
class RendererConfig:
    """Stable renderer settings shared by offscreen and window backends."""

    device_name: str | None = None
    max_bounces: int = 5
    present_mode: str = "mailbox"
    swapchain_images: int = 0
    present_pacing: bool = False
    material_program: MaterialProgram | None = None
    samples_per_pixel: int = 1
    progressive_accumulation: bool = False
    interactive_samples_per_pixel: int | None = None
    stationary_delay_seconds: float = 0.15
    temporal_history: bool = False
    temporal_history_limit: int = 32
    temporal_neighborhood_clamping: bool = True
    adaptive_sampling: bool = False
    adaptive_min_samples: int = 1
    adaptive_variance_threshold: float = 0.0025
    area_light_samples: int = 1
    wavefront_secondary_area_light_samples: int = 0
    wavefront_environment_samples: int = 1
    wavefront_secondary_nee_probability: float = 1.0
    wavefront_unified_secondary_nee: bool = True
    wavefront_unified_primary_restir: bool = False
    wavefront_stratified_primary_restir: bool = False
    wavefront_restir_di: bool = False
    wavefront_restir_candidates: int = 1
    wavefront_restir_history_limit: int = 20
    wavefront_restir_history_motion_pixels: float = 16.0
    wavefront_restir_spatial_reuse: bool = False
    wavefront_restir_spatial_neighbors: int = 4
    wavefront_restir_spatial_radius: int = 4
    # Pairwise balance is the production spatial-reuse estimator. Set this to
    # False explicitly to retain canonical spatial merging for comparisons.
    wavefront_restir_pairwise_mis: bool = True
    wavefront_restir_generalized_mis: bool = False
    wavefront_restir_generalized_balance_cap: float = 2.0
    wavefront_restir_specialization: bool = True
    denoiser_enabled: bool = False
    denoiser_iterations: int = 3
    denoiser_variance_threshold: float = 0.01
    wavefront_tile_capacity: int = 131072
    wavefront_exposure: float = 1.0
    wavefront_profiling: bool = False
    wavefront_pipeline_statistics: bool = False
    wavefront_hdr_capture: bool = False
    wavefront_render_scale: float = 1.0
    wavefront_dynamic_resolution: bool = False
    wavefront_dynamic_target_ms: float = 16.67
    wavefront_dynamic_min_scale: float = 0.5
    wavefront_temporal_reconstruction: bool = False
    wavefront_temporal_weight: float = 0.85
    wavefront_temporal_variance_confidence: bool = False
    wavefront_temporal_variance_strength: float = 0.5
    wavefront_temporal_material_confidence: bool = False
    wavefront_temporal_transmission_history_scale: float = 0.5
    wavefront_temporal_reprojection_search: bool = False
    wavefront_temporal_outlier_confidence: bool = False
    wavefront_temporal_outlier_strength: float = 0.75
    wavefront_temporal_motion_limit_pixels: float = 64.0
    wavefront_indirect_reuse_storage: bool = False
    wavefront_indirect_reuse_candidates: bool = False
    wavefront_indirect_reuse_temporal: bool = False
    wavefront_indirect_reuse_spatial: bool = False
    wavefront_indirect_reuse_profiling: bool = False
    wavefront_indirect_reuse_apply: bool = False
    wavefront_indirect_reuse_apply_strength: float = 0.35
    wavefront_indirect_reuse_history_limit: int = 32
    wavefront_indirect_reuse_history_motion_pixels: float = 16.0
    wavefront_indirect_reuse_debug_view: str = "off"
    wavefront_indirect_reuse_scale: float = 0.5
    wavefront_indirect_reuse_budget_mib: float = 128.0
    wavefront_diffuse_filter: bool = False
    wavefront_diffuse_filter_strength: float = 0.35
    wavefront_russian_roulette: bool = False
    wavefront_russian_roulette_start: int = 3
    wavefront_russian_roulette_min_survival: float = 0.1
    wavefront_fused_secondary: bool = True
    wavefront_subgroup_enqueue: bool = True
    direct_swapchain_storage: bool = True
    wavefront_execution_strategy: str = "wavefront"
    wavefront_auto_megakernel_transmission_fraction: float = 0.25
    wavefront_auto_megakernel_triangle_threshold: int = 16384
    wavefront_hybrid_inline_bounces: int = 3
    wavefront_device_local_textures: bool = True
    wavefront_native_textures: bool = False
    wavefront_material_bucketing: bool = False
    wavefront_material_bucketing_start_bounce: int = 2
    wavefront_persistent_coarse_tiles: bool = False
    wavefront_persistent_continuations: bool = False
    wavefront_scene_specialization: bool = True
    wavefront_untextured_specialization: bool = False
    wavefront_untextured_specialization_part: str = "full"
    wavefront_megakernel_single_warp: bool = False
    wavefront_megakernel_group_swizzle: int = 0
    wavefront_ser: bool = False
    wavefront_ser_reorder: bool = True
    # The current brick traversal is an opt-in experiment: its correctness
    # gate passes, but the sparse showcase measured a small regression.
    volume_empty_space_skipping: bool = False

    def __post_init__(self):
        if not isinstance(self.volume_empty_space_skipping, bool):
            raise TypeError("volume_empty_space_skipping must be a bool")
        if not 1 <= self.max_bounces <= 16:
            raise ValueError("max_bounces must be between 1 and 16")
        if self.present_mode.lower() not in {"mailbox", "immediate", "fifo"}:
            raise ValueError("present_mode must be 'mailbox', 'immediate', or 'fifo'")
        if self.swapchain_images < 0:
            raise ValueError("swapchain_images cannot be negative")
        if self.material_program is not None and not isinstance(
            self.material_program, MaterialProgram
        ):
            raise TypeError("material_program must be created by @material")
        if not 1 <= self.samples_per_pixel <= 64:
            raise ValueError("samples_per_pixel must be between 1 and 64")
        if (
            self.interactive_samples_per_pixel is not None
            and not 1 <= self.interactive_samples_per_pixel <= 64
        ):
            raise ValueError("interactive_samples_per_pixel must be between 1 and 64")
        if self.stationary_delay_seconds < 0.0:
            raise ValueError("stationary_delay_seconds cannot be negative")
        if self.temporal_history and not self.progressive_accumulation:
            raise ValueError(
                "temporal_history requires progressive_accumulation=True"
            )
        if not 1 <= self.temporal_history_limit <= 4096:
            raise ValueError("temporal_history_limit must be between 1 and 4096")
        if not 1 <= self.adaptive_min_samples <= 64:
            raise ValueError("adaptive_min_samples must be between 1 and 64")
        if self.adaptive_min_samples > self.samples_per_pixel:
            raise ValueError("adaptive_min_samples cannot exceed samples_per_pixel")
        if self.adaptive_variance_threshold <= 0.0:
            raise ValueError("adaptive_variance_threshold must be positive")
        if self.adaptive_sampling and not self.progressive_accumulation:
            raise ValueError("adaptive_sampling requires progressive_accumulation=True")
        if not 1 <= self.area_light_samples <= 16:
            raise ValueError("area_light_samples must be between 1 and 16")
        if not 0 <= self.wavefront_secondary_area_light_samples <= 16:
            raise ValueError(
                "wavefront_secondary_area_light_samples must be between 0 "
                "and 16"
            )
        if not 0 <= self.wavefront_environment_samples <= 4:
            raise ValueError(
                "wavefront_environment_samples must be between 0 and 4"
            )
        if not 0.0 < self.wavefront_secondary_nee_probability <= 1.0:
            raise ValueError(
                "wavefront_secondary_nee_probability must be in (0, 1]"
            )
        if not isinstance(self.wavefront_unified_secondary_nee, bool):
            raise TypeError("wavefront_unified_secondary_nee must be a bool")
        if not isinstance(self.wavefront_unified_primary_restir, bool):
            raise TypeError("wavefront_unified_primary_restir must be a bool")
        if not isinstance(self.wavefront_stratified_primary_restir, bool):
            raise TypeError("wavefront_stratified_primary_restir must be a bool")
        if (
            self.wavefront_unified_primary_restir
            and self.wavefront_stratified_primary_restir
        ):
            raise ValueError(
                "unified and stratified primary ReSTIR modes are mutually exclusive"
            )
        if not 1 <= self.wavefront_restir_history_limit <= 64:
            raise ValueError(
                "wavefront_restir_history_limit must be between 1 and 64"
            )
        if (not math.isfinite(self.wavefront_restir_history_motion_pixels)
                or self.wavefront_restir_history_motion_pixels <= 0.0):
            raise ValueError(
                "wavefront_restir_history_motion_pixels must be positive"
            )
        if not 1 <= self.wavefront_restir_candidates <= 4:
            raise ValueError(
                "wavefront_restir_candidates must be between 1 and 4"
            )
        if not isinstance(self.wavefront_restir_spatial_reuse, bool):
            raise TypeError("wavefront_restir_spatial_reuse must be a bool")
        if not 1 <= self.wavefront_restir_spatial_neighbors <= 8:
            raise ValueError(
                "wavefront_restir_spatial_neighbors must be between 1 and 8"
            )
        if not 1 <= self.wavefront_restir_spatial_radius <= 32:
            raise ValueError(
                "wavefront_restir_spatial_radius must be between 1 and 32"
            )
        if not isinstance(self.wavefront_restir_pairwise_mis, bool):
            raise TypeError("wavefront_restir_pairwise_mis must be a bool")
        if not isinstance(self.wavefront_restir_generalized_mis, bool):
            raise TypeError("wavefront_restir_generalized_mis must be a bool")
        if (self.wavefront_restir_generalized_mis
                and not self.wavefront_restir_spatial_reuse):
            raise ValueError(
                "wavefront_restir_generalized_mis requires spatial reuse"
            )
        if not 1.0 <= self.wavefront_restir_generalized_balance_cap <= 8.0:
            raise ValueError(
                "wavefront_restir_generalized_balance_cap must be between 1 and 8"
            )
        if self.wavefront_restir_di and self.samples_per_pixel != 1:
            raise ValueError(
                "wavefront_restir_di currently requires samples_per_pixel=1"
            )
        if not 1 <= self.denoiser_iterations <= 5:
            raise ValueError("denoiser_iterations must be between 1 and 5")
        if self.denoiser_enabled and not self.temporal_history:
            raise ValueError("denoiser_enabled requires temporal_history=True")
        if self.denoiser_variance_threshold <= 0.0:
            raise ValueError("denoiser_variance_threshold must be positive")
        if not 1 <= self.wavefront_tile_capacity <= 4194304:
            raise ValueError("wavefront_tile_capacity must be between 1 and 4194304")
        if self.wavefront_exposure <= 0.0:
            raise ValueError("wavefront_exposure must be positive")
        if not 0.25 <= self.wavefront_render_scale <= 1.0:
            raise ValueError("wavefront_render_scale must be between 0.25 and 1.0")
        if self.wavefront_dynamic_target_ms <= 0.0:
            raise ValueError("wavefront_dynamic_target_ms must be positive")
        if not 0.25 <= self.wavefront_dynamic_min_scale <= self.wavefront_render_scale:
            raise ValueError(
                "wavefront_dynamic_min_scale must be between 0.25 and "
                "wavefront_render_scale"
            )
        if not 0.0 <= self.wavefront_temporal_weight < 1.0:
            raise ValueError("wavefront_temporal_weight must be in [0.0, 1.0)")
        if not 0.0 <= self.wavefront_temporal_variance_strength <= 1.0:
            raise ValueError(
                "wavefront_temporal_variance_strength must be in [0.0, 1.0]"
            )
        if not isinstance(self.wavefront_temporal_variance_confidence, bool):
            raise TypeError(
                "wavefront_temporal_variance_confidence must be a bool"
            )
        if not isinstance(self.wavefront_temporal_material_confidence, bool):
            raise TypeError(
                "wavefront_temporal_material_confidence must be a bool"
            )
        if not 0.0 <= self.wavefront_temporal_transmission_history_scale <= 1.0:
            raise ValueError(
                "wavefront_temporal_transmission_history_scale must be in "
                "[0.0, 1.0]"
            )
        if not isinstance(self.wavefront_temporal_reprojection_search, bool):
            raise TypeError(
                "wavefront_temporal_reprojection_search must be a bool"
            )
        if not isinstance(self.wavefront_temporal_outlier_confidence, bool):
            raise TypeError(
                "wavefront_temporal_outlier_confidence must be a bool"
            )
        if not 0.0 <= self.wavefront_temporal_outlier_strength <= 1.0:
            raise ValueError(
                "wavefront_temporal_outlier_strength must be in [0.0, 1.0]"
            )
        if (not math.isfinite(self.wavefront_temporal_motion_limit_pixels)
                or self.wavefront_temporal_motion_limit_pixels <= 0.0):
            raise ValueError(
                "wavefront_temporal_motion_limit_pixels must be positive"
            )
        if not isinstance(self.wavefront_indirect_reuse_storage, bool):
            raise TypeError("wavefront_indirect_reuse_storage must be a bool")
        if not isinstance(self.wavefront_indirect_reuse_candidates, bool):
            raise TypeError("wavefront_indirect_reuse_candidates must be a bool")
        if (self.wavefront_indirect_reuse_candidates
                and not self.wavefront_indirect_reuse_storage):
            raise ValueError(
                "wavefront_indirect_reuse_candidates requires "
                "wavefront_indirect_reuse_storage=True"
            )
        if not isinstance(self.wavefront_indirect_reuse_temporal, bool):
            raise TypeError("wavefront_indirect_reuse_temporal must be a bool")
        if (self.wavefront_indirect_reuse_temporal
                and not self.wavefront_indirect_reuse_candidates):
            raise ValueError(
                "wavefront_indirect_reuse_temporal requires "
                "wavefront_indirect_reuse_candidates=True"
            )
        if not isinstance(self.wavefront_indirect_reuse_spatial, bool):
            raise TypeError("wavefront_indirect_reuse_spatial must be a bool")
        if (self.wavefront_indirect_reuse_spatial
                and not self.wavefront_indirect_reuse_temporal):
            raise ValueError(
                "wavefront_indirect_reuse_spatial requires "
                "wavefront_indirect_reuse_temporal=True"
            )
        if not isinstance(self.wavefront_indirect_reuse_profiling, bool):
            raise TypeError("wavefront_indirect_reuse_profiling must be a bool")
        if (self.wavefront_indirect_reuse_profiling
                and not self.wavefront_indirect_reuse_candidates):
            raise ValueError(
                "wavefront_indirect_reuse_profiling requires "
                "wavefront_indirect_reuse_candidates=True"
            )
        if not isinstance(self.wavefront_indirect_reuse_apply, bool):
            raise TypeError("wavefront_indirect_reuse_apply must be a bool")
        if (self.wavefront_indirect_reuse_apply
                and not self.wavefront_indirect_reuse_candidates):
            raise ValueError(
                "wavefront_indirect_reuse_apply requires "
                "wavefront_indirect_reuse_candidates=True"
            )
        if (not math.isfinite(self.wavefront_indirect_reuse_apply_strength)
                or not 0.0 <= self.wavefront_indirect_reuse_apply_strength <= 1.0):
            raise ValueError(
                "wavefront_indirect_reuse_apply_strength must be in [0.0, 1.0]"
            )
        if not isinstance(self.wavefront_indirect_reuse_history_limit, int):
            raise TypeError(
                "wavefront_indirect_reuse_history_limit must be an int"
            )
        if not 1 <= self.wavefront_indirect_reuse_history_limit <= 127:
            raise ValueError(
                "wavefront_indirect_reuse_history_limit must be between 1 and 127"
            )
        if (not math.isfinite(
                self.wavefront_indirect_reuse_history_motion_pixels)
                or self.wavefront_indirect_reuse_history_motion_pixels <= 0.0):
            raise ValueError(
                "wavefront_indirect_reuse_history_motion_pixels must be positive"
            )
        if self.wavefront_indirect_reuse_debug_view not in {
            "off", "radiance", "history", "validity", "acceptance"
        }:
            raise ValueError(
                "wavefront_indirect_reuse_debug_view must be 'off', "
                "'radiance', 'history', 'validity', or 'acceptance'"
            )
        if (self.wavefront_indirect_reuse_debug_view != "off"
                and not self.wavefront_indirect_reuse_candidates):
            raise ValueError(
                "wavefront_indirect_reuse_debug_view requires "
                "wavefront_indirect_reuse_candidates=True"
            )
        if (not math.isfinite(self.wavefront_indirect_reuse_scale)
                or not 0.25 <= self.wavefront_indirect_reuse_scale <= 1.0):
            raise ValueError(
                "wavefront_indirect_reuse_scale must be in [0.25, 1.0]"
            )
        if (not math.isfinite(self.wavefront_indirect_reuse_budget_mib)
                or self.wavefront_indirect_reuse_budget_mib <= 0.0):
            raise ValueError(
                "wavefront_indirect_reuse_budget_mib must be positive"
            )
        if not 0.0 <= self.wavefront_diffuse_filter_strength <= 1.0:
            raise ValueError(
                "wavefront_diffuse_filter_strength must be in [0.0, 1.0]"
            )
        if not 2 <= self.wavefront_russian_roulette_start <= 15:
            raise ValueError(
                "wavefront_russian_roulette_start must be between 2 and 15"
            )
        if not 0.01 <= self.wavefront_russian_roulette_min_survival <= 0.95:
            raise ValueError(
                "wavefront_russian_roulette_min_survival must be in [0.01, 0.95]"
            )
        if self.wavefront_execution_strategy not in {
            "wavefront", "hybrid", "megakernel", "persistent", "ser", "auto"
        }:
            raise ValueError(
                "wavefront_execution_strategy must be 'wavefront', "
                "'hybrid', 'megakernel', 'persistent', 'ser', or 'auto'"
            )
        if not 0.0 <= self.wavefront_auto_megakernel_transmission_fraction <= 1.0:
            raise ValueError(
                "wavefront_auto_megakernel_transmission_fraction must be in [0, 1]"
            )
        if self.wavefront_auto_megakernel_triangle_threshold < 1:
            raise ValueError(
                "wavefront_auto_megakernel_triangle_threshold must be positive"
            )
        if not 3 <= self.wavefront_hybrid_inline_bounces <= 15 \
                or self.wavefront_hybrid_inline_bounces % 2 == 0:
            raise ValueError(
                "wavefront_hybrid_inline_bounces must be an odd value from 3 to 15"
            )
        if not isinstance(self.wavefront_device_local_textures, bool):
            raise TypeError("wavefront_device_local_textures must be a bool")
        if not isinstance(self.wavefront_native_textures, bool):
            raise TypeError("wavefront_native_textures must be a bool")
        if not isinstance(self.wavefront_material_bucketing, bool):
            raise TypeError("wavefront_material_bucketing must be a bool")
        if not 1 <= self.wavefront_material_bucketing_start_bounce <= 15:
            raise ValueError(
                "wavefront_material_bucketing_start_bounce must be between 1 and 15"
            )
        if not isinstance(self.wavefront_persistent_coarse_tiles, bool):
            raise TypeError("wavefront_persistent_coarse_tiles must be a bool")
        if (
            self.wavefront_persistent_coarse_tiles
            and self.wavefront_execution_strategy != "persistent"
        ):
            raise ValueError(
                "wavefront_persistent_coarse_tiles requires persistent execution"
            )
        if not isinstance(self.wavefront_persistent_continuations, bool):
            raise TypeError("wavefront_persistent_continuations must be a bool")
        if (
            self.wavefront_persistent_continuations
            and self.wavefront_execution_strategy != "hybrid"
        ):
            raise ValueError(
                "wavefront_persistent_continuations requires hybrid execution"
            )
        if not isinstance(self.wavefront_scene_specialization, bool):
            raise TypeError("wavefront_scene_specialization must be a bool")
        if not isinstance(self.wavefront_untextured_specialization, bool):
            raise TypeError("wavefront_untextured_specialization must be a bool")
        if self.wavefront_untextured_specialization_part not in {
            "primary", "secondary", "full"
        }:
            raise ValueError(
                "wavefront_untextured_specialization_part must be primary, "
                "secondary, or full"
            )
        if not isinstance(self.wavefront_megakernel_single_warp, bool):
            raise TypeError("wavefront_megakernel_single_warp must be a bool")
        if self.wavefront_megakernel_group_swizzle not in (0, 8, 16, 32):
            raise ValueError(
                "wavefront_megakernel_group_swizzle must be 0, 8, 16, or 32"
            )
        if not isinstance(self.wavefront_ser, bool):
            raise TypeError("wavefront_ser must be a bool")
        if not isinstance(self.wavefront_ser_reorder, bool):
            raise TypeError("wavefront_ser_reorder must be a bool")
        if self.wavefront_execution_strategy == "ser" and not self.wavefront_ser:
            raise ValueError("ser execution requires wavefront_ser=True")
        if not isinstance(self.wavefront_hdr_capture, bool):
            raise TypeError("wavefront_hdr_capture must be a bool")
        if not isinstance(self.wavefront_pipeline_statistics, bool):
            raise TypeError("wavefront_pipeline_statistics must be a bool")
        if not isinstance(self.wavefront_restir_specialization, bool):
            raise TypeError("wavefront_restir_specialization must be a bool")
        if self.wavefront_material_bucketing and self.wavefront_execution_strategy in {
            "hybrid", "megakernel", "persistent"
        }:
            raise ValueError(
                "wavefront_material_bucketing requires wavefront or auto execution"
            )


def _resolve_execution_strategy(config, scene):
    """Choose an execution kernel without changing rendered results."""
    if config.wavefront_execution_strategy != "auto":
        return config.wavefront_execution_strategy
    if config.wavefront_material_bucketing:
        return "wavefront"
    total_triangles = sum(len(mesh.indices) for mesh in scene.visible_meshes)
    if total_triangles == 0:
        return "wavefront"
    transmission_triangles = sum(
        len(mesh.indices) for mesh in scene.visible_meshes
        if (
            mesh.material.transmission > 0.001
            or mesh.material.transmission_texture is not None
        )
    )
    fraction = transmission_triangles / total_triangles
    return (
        "megakernel"
        if (
            fraction >= config.wavefront_auto_megakernel_transmission_fraction
            or total_triangles
                >= config.wavefront_auto_megakernel_triangle_threshold
        )
        else "wavefront"
    )


def _text(value):
    if isinstance(value, bytes):
        return value.split(b"\0", 1)[0].decode("utf-8", errors="replace")
    return str(value).split("\0", 1)[0]


@dataclass(frozen=True)
class VulkanDeviceInfo:
    name: str
    device_type: int
    api_version: tuple[int, int, int]
    extensions: frozenset[str]

    @property
    def missing_ray_tracing_extensions(self):
        return REQUIRED_RAY_QUERY_EXTENSIONS - self.extensions

    @property
    def supports_ray_query(self):
        return not self.missing_ray_tracing_extensions

    @property
    def supports_ray_tracing_pipeline(self):
        """Backward-compatible name for the original capability property."""
        return self.supports_ray_query

    @property
    def is_hardware_adapter(self):
        # VkPhysicalDeviceType: integrated=1, discrete=2, virtual=3, CPU=4.
        return self.device_type in (1, 2, 3)

    @property
    def supports_hardware_ray_tracing(self):
        return self.is_hardware_adapter and self.supports_ray_query


def _version_tuple(version):
    return (version >> 22, (version >> 12) & 0x3FF, version & 0xFFF)


def _render_options(config, samples, max_bounces):
    samples = config.samples_per_pixel if samples is None else int(samples)
    if not 1 <= samples <= 64:
        raise ValueError("samples must be between 1 and 64")
    bounces = config.max_bounces if max_bounces is None else int(max_bounces)
    if not 1 <= bounces <= 16:
        raise ValueError("max_bounces must be between 1 and 16")
    return samples, bounces


def probe_vulkan_devices():
    """Enumerate Vulkan adapters without creating a logical device."""
    try:
        import vulkan as vk
    except ImportError as error:
        raise RuntimeError(
            "The Vulkan backend requires the optional Vulkan dependencies; "
            "install them with: pip install 'ordinarylight[vulkan]'"
        ) from error

    application = vk.VkApplicationInfo(
        sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName="Ordinary Light",
        applicationVersion=vk.VK_MAKE_VERSION(0, 1, 0),
        pEngineName="Ordinary Light",
        engineVersion=vk.VK_MAKE_VERSION(0, 1, 0),
        apiVersion=vk.VK_MAKE_VERSION(1, 2, 0),
    )
    create_info = vk.VkInstanceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pApplicationInfo=application,
    )
    instance = vk.vkCreateInstance(create_info, None)
    try:
        result = []
        for physical_device in vk.vkEnumeratePhysicalDevices(instance):
            properties = vk.vkGetPhysicalDeviceProperties(physical_device)
            extensions = frozenset(
                _text(item.extensionName)
                for item in vk.vkEnumerateDeviceExtensionProperties(physical_device, None)
            )
            result.append(
                VulkanDeviceInfo(
                    name=_text(properties.deviceName),
                    device_type=properties.deviceType,
                    api_version=_version_tuple(properties.apiVersion),
                    extensions=extensions,
                )
            )
        return result
    finally:
        vk.vkDestroyInstance(instance, None)


class VulkanRayTracingBackend:
    """Hardware Vulkan backend using acceleration structures and ray queries."""
    available_outputs = (
        "color", "variance", "depth", "normal", "instance_id", "object_id",
        "material_id", "motion",
    )

    def __init__(self, device_name=None, *, config=None):
        if config is not None and device_name is not None:
            raise ValueError("Pass device_name or config, not both")
        self.config = config or RendererConfig(device_name=device_name)
        device_name = self.config.device_name
        devices = probe_vulkan_devices()
        compatible = [device for device in devices if device.supports_hardware_ray_tracing]
        if device_name is not None:
            compatible = [device for device in compatible if device_name.lower() in device.name.lower()]
        if not compatible:
            details = "; ".join(
                f"{device.name}: missing {sorted(device.missing_ray_tracing_extensions)}"
                for device in devices
            ) or "no Vulkan devices found"
            raise RuntimeError(f"No compatible Vulkan ray-tracing adapter: {details}")
        self.device = compatible[0]
        from .vulkan_rt import VulkanRayQueryCore

        self._core = VulkanRayQueryCore(config=self.config)
        self._output_history = None

    def reset_output_history(self):
        """Discard prior-frame state used by opt-in motion output."""
        self._output_history = None

    @property
    def capabilities(self):
        """Backend-neutral semantic capabilities of the initialized device."""
        return {
            "backend": "vulkan-ray-query",
            "outputs": self.available_outputs,
            "features": frozenset({
                "hardware_ray_tracing", "offscreen_rendering", "instancing",
                "textures", "custom_materials", "progressive_accumulation",
                "temporal_reconstruction", "denoising", "restir_di",
                "indirect_reuse", "volumes", "volume_scattering",
                "volume_empty_space_skipping", "motion_vectors",
            }),
            "limits": {
                "max_bounces": 16,
                "max_samples_per_pixel": 64,
                "max_visible_volumes": 16,
                "max_point_lights": 64,
                "max_analytic_lights": 64,
            },
            "device": self.device,
        }

    @staticmethod
    def _project_positions(positions, camera, width, height):
        origin = np.asarray(camera.position, np.float64)
        forward = np.asarray(camera.target, np.float64) - origin
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray(camera.up, np.float64))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        relative = np.asarray(positions, np.float64) - origin
        distance = relative @ forward
        scale = np.tan(np.radians(camera.vertical_fov_degrees) * 0.5)
        safe_distance = np.maximum(distance, 1e-12)
        ndc_x = (relative @ right) / (
            safe_distance * scale * (width / height)
        )
        ndc_y = -(relative @ up) / (safe_distance * scale)
        return np.column_stack((
            (ndc_x + 1.0) * 0.5 * width,
            (ndc_y + 1.0) * 0.5 * height,
        )), distance > 1e-8

    @staticmethod
    def _capture_motion_state(scene, camera, size):
        return {
            "scene": scene, "camera": camera, "size": size,
            "meshes": {
                mesh.id: (
                    mesh.vertices.copy(), mesh.indices.copy(),
                    mesh.transform.matrix.copy(),
                )
                for mesh in scene.visible_meshes
            },
        }

    def _motion_product(
        self, scene, camera, width, height, primitive, position, barycentric,
    ):
        motion = np.zeros((height, width, 2), np.float32)
        previous = self._output_history
        if (previous is None or previous["scene"] is not scene
                or previous["size"] != (width, height)):
            return motion
        flat_primitive = primitive.reshape(-1)
        flat_position = position.reshape((-1, 3))
        flat_barycentric = barycentric.reshape((-1, 2))
        flat_motion = motion.reshape((-1, 2))
        offset = 0
        for mesh in scene.visible_meshes:
            triangle_count = len(mesh.indices)
            selected = np.flatnonzero(
                (flat_primitive >= offset)
                & (flat_primitive < offset + triangle_count)
            )
            old = previous["meshes"].get(mesh.id)
            if selected.size and old is not None:
                old_vertices, old_indices, old_transform = old
                if (old_indices.shape == mesh.indices.shape
                        and np.array_equal(old_indices, mesh.indices)
                        and old_vertices.shape == mesh.vertices.shape):
                    local = flat_primitive[selected].astype(np.int64) - offset
                    triangles = old_vertices[old_indices[local]]
                    uv = flat_barycentric[selected]
                    weights = np.column_stack((
                        1.0 - uv[:, 0] - uv[:, 1], uv[:, 0], uv[:, 1],
                    ))
                    old_object = np.sum(
                        triangles * weights[..., None], axis=1
                    )
                    old_world = (
                        old_object @ old_transform[:3, :3].T
                        + old_transform[:3, 3]
                    )
                    current_pixel, current_valid = self._project_positions(
                        flat_position[selected], camera, width, height
                    )
                    old_pixel, old_valid = self._project_positions(
                        old_world, previous["camera"], width, height
                    )
                    valid = current_valid & old_valid
                    flat_motion[selected[valid]] = (
                        current_pixel[valid] - old_pixel[valid]
                    ).astype(np.float32)
            offset += triangle_count
        return motion

    def render(self, scene, camera, width, height, samples=None, max_bounces=None):
        samples, max_bounces = _render_options(self.config, samples, max_bounces)
        return self._core.render(
            scene, camera, width, height, samples=samples, max_bounces=max_bounces
        )

    def render_to(self, scene, camera, surface, samples=None, max_bounces=None):
        rgba = self.render(
            scene,
            camera,
            surface.width,
            surface.height,
            samples=samples,
            max_bounces=max_bounces,
        )
        return surface.present(rgba)

    def render_wavefront(
        self, scene, camera, width, height, *, samples=None, frame_index=0
    ):
        """Render deterministic HDR pixels through the configured wavefront strategy.

        This readback-oriented entry point is intended for validation, tests, and
        offline tooling.  Interactive applications should use
        :class:`VulkanGlfwPresenter`, which avoids copying pixels to the host.
        """
        return self.render_wavefront_outputs(
            scene, camera, width, height, outputs=("color",),
            samples=samples, frame_index=frame_index,
        )["color"]

    def render_frame(
        self, scene, camera, width, height, *, samples=None, frame_index=0,
    ):
        """Implement the backend-neutral linear-HDR frame contract."""
        return self.render_wavefront(
            scene, camera, width, height,
            samples=samples, frame_index=frame_index,
        )

    def render_products(
        self, scene, camera, width, height, *, outputs,
        samples=None, frame_index=0,
    ):
        """Implement the optional backend-neutral named-product contract."""
        return self.render_wavefront_outputs(
            scene, camera, width, height, outputs=outputs,
            samples=samples, frame_index=frame_index,
        )

    def render_wavefront_outputs(
        self, scene, camera, width, height, *, outputs=("color",),
        samples=None, frame_index=0,
    ):
        """Render explicitly requested offscreen products.

        ``variance`` is unbiased per-pixel luminance variance across the
        independent samples used for this call. It is zero for one sample.
        """
        outputs = tuple(outputs)
        unsupported = set(outputs) - set(self.available_outputs)
        if not outputs or unsupported:
            raise ValueError(
                f"unsupported render outputs: {tuple(sorted(unsupported))}"
            )
        width, height = int(width), int(height)
        if width < 1 or height < 1:
            raise ValueError("width and height must be positive")
        samples = self.config.samples_per_pixel if samples is None else int(samples)
        if not 1 <= samples <= 64:
            raise ValueError("samples must be between 1 and 64")
        self._core.upload_window_scene(scene)
        accumulated = None
        luminance_sum = None
        luminance_square_sum = None
        variance_requested = "variance" in outputs
        primary_products_requested = bool(
            {"depth", "normal", "instance_id", "object_id", "material_id", "motion"}
            & set(outputs)
        )
        depth = (
            np.full((height, width), np.inf, np.float32)
            if "depth" in outputs else None
        )
        normal = (
            np.zeros((height, width, 3), np.float32)
            if "normal" in outputs else None
        )
        object_id = (
            np.full((height, width), np.uint32(0xffffffff), np.uint32)
            if "object_id" in outputs else None
        )
        instance_id = (
            np.full((height, width), np.uint32(0xffffffff), np.uint32)
            if "instance_id" in outputs else None
        )
        material_id = (
            np.full((height, width), np.uint32(0xffffffff), np.uint32)
            if "material_id" in outputs else None
        )
        motion_requested = "motion" in outputs
        primary_primitive = (
            np.full((height, width), np.uint32(0xffffffff), np.uint32)
            if motion_requested else None
        )
        primary_position = (
            np.zeros((height, width, 3), np.float32)
            if motion_requested else None
        )
        primary_barycentric = (
            np.zeros((height, width, 2), np.float32)
            if motion_requested else None
        )
        triangle_object_ids = (
            scene.triangle_instance_ids()
            if object_id is not None or instance_id is not None else None
        )
        triangle_material_ids = (
            scene.triangle_material_ids() if material_id is not None else None
        )
        for sample_index in range(samples):
            radiance = np.zeros((height, width, 4), dtype=np.float32)
            capacity = self.config.wavefront_tile_capacity
            y = 0
            while y < height:
                tile_width = min(width, capacity)
                tile_height = min(height - y, max(1, capacity // tile_width))
                x = 0
                while x < width:
                    current_width = min(tile_width, width - x)
                    current_height = min(
                        tile_height, max(1, capacity // current_width), height - y
                    )
                    result = self._core.trace_wavefront_tile(
                        camera,
                        width,
                        height,
                        tile_origin=(x, y),
                        tile_extent=(current_width, current_height),
                        frame_index=int(frame_index),
                        sample_index=sample_index,
                        sample_count=samples,
                        primary_hit_readback=(
                            primary_products_requested and sample_index == 0
                        ),
                    )
                    radiance[
                        y:y + current_height, x:x + current_width
                    ] = result["radiance"]
                    if sample_index == 0:
                        if depth is not None:
                            depth[
                                y:y + current_height, x:x + current_width
                            ] = result["depth"]
                        if normal is not None:
                            normal[
                                y:y + current_height, x:x + current_width
                            ] = result["normal"]
                        if (object_id is not None or instance_id is not None
                                or material_id is not None
                                or motion_requested):
                            primitive = result["primitive_id"]
                            hit = primitive != np.uint32(0xffffffff)
                            if object_id is not None:
                                tile_ids = object_id[
                                    y:y + current_height,
                                    x:x + current_width,
                                ]
                                tile_ids[hit] = triangle_object_ids[
                                    primitive[hit]
                                ]
                            if instance_id is not None:
                                tile_ids = instance_id[
                                    y:y + current_height,
                                    x:x + current_width,
                                ]
                                tile_ids[hit] = triangle_object_ids[
                                    primitive[hit]
                                ]
                            if material_id is not None:
                                tile_ids = material_id[
                                    y:y + current_height,
                                    x:x + current_width,
                                ]
                                tile_ids[hit] = triangle_material_ids[
                                    primitive[hit]
                                ]
                            if motion_requested:
                                primary_primitive[
                                    y:y + current_height,
                                    x:x + current_width,
                                ] = primitive
                                primary_position[
                                    y:y + current_height,
                                    x:x + current_width,
                                ] = result["primary_position"]
                                primary_barycentric[
                                    y:y + current_height,
                                    x:x + current_width,
                                ] = result["primary_barycentric"]
                    x += current_width
                y += tile_height
            if accumulated is None:
                accumulated = radiance.astype("float64")
            else:
                accumulated += radiance
            if variance_requested:
                luminance = (
                    radiance[..., 0] * 0.2126
                    + radiance[..., 1] * 0.7152
                    + radiance[..., 2] * 0.0722
                ).astype(np.float64)
                if luminance_sum is None:
                    luminance_sum = luminance
                    luminance_square_sum = luminance * luminance
                else:
                    luminance_sum += luminance
                    luminance_square_sum += luminance * luminance
        products = {}
        if "color" in outputs:
            products["color"] = (accumulated / samples).astype("float32")
        if variance_requested:
            if samples == 1:
                variance = np.zeros((height, width), np.float32)
            else:
                variance = (
                    luminance_square_sum
                    - luminance_sum * luminance_sum / samples
                ) / (samples - 1)
                variance = np.maximum(variance, 0.0).astype(np.float32)
            products["variance"] = variance
        if depth is not None:
            products["depth"] = depth
        if normal is not None:
            products["normal"] = normal
        if object_id is not None:
            products["object_id"] = object_id
        if instance_id is not None:
            products["instance_id"] = instance_id
        if material_id is not None:
            products["material_id"] = material_id
        if motion_requested:
            products["motion"] = self._motion_product(
                scene, camera, width, height, primary_primitive,
                primary_position, primary_barycentric,
            )
            self._output_history = self._capture_motion_state(
                scene, camera, (width, height)
            )
        return products

    @property
    def pipeline_statistics(self):
        """Driver-provided executable statistics captured in diagnostic mode."""
        executor = getattr(self._core, "wavefront_executor", None)
        return dict(getattr(executor, "pipeline_statistics", {}))

    def close(self):
        if getattr(self, "_core", None) is not None:
            self._core.close()
            self._core = None

    @property
    def last_timings(self):
        return dict(self._core.last_timings) if self._core is not None else {}

    @property
    def present_mode(self):
        return self._core.present_mode_name if self._core is not None else None

    @property
    def swapchain_image_count(self):
        return self._core.swapchain_image_count if self._core is not None else 0

    @property
    def present_pacing(self):
        if self._core is None:
            return "unavailable"
        if self._core.present_pacing_enabled:
            return "present-wait"
        return "disabled" if self._core.present_wait_supported else "unsupported"

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class VulkanGlfwPresenter:
    """Direct Vulkan swapchain presenter for a GLFW window."""

    def __init__(self, window, device_name=None, *, config=None):
        from .vulkan_rt import VulkanRayQueryCore

        if config is not None and device_name is not None:
            raise ValueError("Pass device_name or config, not both")
        self.config = config or RendererConfig(device_name=device_name)
        self._core = VulkanRayQueryCore(config=self.config, glfw_window=window)
        self.device_name = self._core.device_name

    @property
    def last_timings(self):
        return dict(self._core.last_timings) if self._core is not None else {}

    @property
    def present_mode(self):
        return self._core.present_mode_name if self._core is not None else None

    @property
    def swapchain_image_count(self):
        return self._core.swapchain_image_count if self._core is not None else 0

    @property
    def present_pacing(self):
        if self._core is None:
            return "unavailable"
        if self._core.present_pacing_enabled:
            return "present-wait"
        return "disabled" if self._core.present_wait_supported else "unsupported"

    def upload_scene(self, scene):
        """Upload or replace the scene used for direct presentation."""
        self._core.upload_window_scene(scene)

    def reset_accumulation(self):
        """Discard progressive history before the next presented frame."""
        self._core.reset_accumulation()

    @property
    def accumulated_frames(self):
        """Number of frames currently represented by progressive history."""
        return self._core.accumulation_frame if self._core is not None else 0

    @property
    def effective_samples_per_pixel(self):
        """Sample budget selected for the most recently presented frame."""
        if self._core is None:
            return 0
        return self._core.effective_samples_per_pixel

    @property
    def pipeline_stages(self):
        """Ordered names of the active executable render stages."""
        if self._core is None:
            return ()
        return self._core.window_pipeline.stage_names

    @property
    def denoiser_enabled(self):
        """Whether the denoised HDR result is currently presented."""
        return bool(
            self._core is not None and self._core.denoiser_output_enabled
        )

    def toggle_denoiser(self):
        """Toggle denoised/raw presentation and return the new state."""
        if self._core is None or not self.config.denoiser_enabled:
            return False
        self._core.denoiser_output_enabled = not self._core.denoiser_output_enabled
        return self._core.denoiser_output_enabled

    @property
    def wavefront_restir_enabled(self):
        """Whether temporal ReSTIR DI is active for the next frame."""
        return bool(
            self._core is not None
            and self._core.wavefront_restir_runtime_enabled
        )

    def set_wavefront_restir_enabled(self, enabled):
        """Select conventional or temporal ReSTIR direct lighting."""
        if self._core is None:
            return False
        return self._core.set_wavefront_restir_enabled(enabled)

    def toggle_wavefront_restir(self):
        """Toggle temporal ReSTIR DI and return the new state."""
        return self.set_wavefront_restir_enabled(
            not self.wavefront_restir_enabled
        )

    def trace_wavefront_tile(
        self, scene, camera, width, height, *, tile_origin=(0, 0), tile_extent=None,
        frame_index=0, sample_index=0,
    ):
        """Run the experimental generate/intersect wavefront GPU stages."""
        self._core.prepare_window_scene(scene)
        return self._core.trace_wavefront_tile(
            camera, width, height, tile_origin=tile_origin,
            tile_extent=tile_extent, frame_index=frame_index,
            sample_index=sample_index,
        )

    def present(
        self, scene, camera, width, height, overlay_fps=None, max_bounces=None,
        samples=None,
    ):
        samples, max_bounces = _render_options(self.config, samples, max_bounces)
        return self._core.present_window(
            scene, camera, width, height,
            overlay_fps=overlay_fps,
            max_bounces=max_bounces,
            samples=samples,
        )

    def present_wavefront(self, scene, camera, width, height):
        """Present the experimental wavefront path without pixel readback."""
        return self._core.present_wavefront_window(
            scene, camera, width, height
        )

    def capture_wavefront_hdr(self):
        """Read back the last presented internal HDR frame in capture mode."""
        return self._core.capture_wavefront_hdr()

    def close(self):
        if self._core is not None:
            self._core.close()
            self._core = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


class VulkanSurfacePresenter(VulkanGlfwPresenter):
    """Present directly to an externally owned Vulkan instance and surface.

    This is the window-toolkit integration boundary. The caller owns the
    ``VkInstance`` and ``VkSurfaceKHR`` and must keep both alive until
    :meth:`close` returns. ordinarylight owns its logical device, swapchain, and
    all rendering resources created against them.
    """

    def __init__(self, instance, surface, device_name=None, *, config=None):
        from .vulkan_rt import VulkanRayQueryCore
        from vulkan import ffi

        if config is not None and device_name is not None:
            raise ValueError("Pass device_name or config, not both")
        if instance is None or surface is None:
            raise ValueError("instance and surface are required")

        def handle(value, ctype):
            # PySide exposes Vulkan handles as Python integers, while low-level
            # callers may already hold vulkan-python CFFI objects.
            if isinstance(value, int):
                return ffi.cast(ctype, value)
            try:
                return ffi.cast(ctype, int(value))
            except (TypeError, ValueError):
                return value

        self.config = config or RendererConfig(device_name=device_name)
        self._core = VulkanRayQueryCore(
            config=self.config,
            external_instance=handle(instance, "VkInstance"),
            external_surface=handle(surface, "VkSurfaceKHR"),
        )
        self.device_name = self._core.device_name
