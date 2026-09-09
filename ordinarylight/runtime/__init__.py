"""Algorithm-independent GPU services (Vulkan loaded on demand)."""

_MODULES = {
    "primary_operation": "primary",
    "indirect_apply_operation": "indirect_apply",
    "indirect_candidates_operation": "indirect_candidates",
    "clear_indirect_reservoirs": "indirect",
    "prepare_primary_metadata_shader": "primary_metadata",
    "VulkanPrimaryMetadata": "primary_metadata",
    "VulkanRelaxPrepare": "relax_prepare",
    "VulkanPathResolve": "path_resolve",
    "reset_ray_queue": "trace",
    "split_trace_graph": "trace",
    "VulkanQueueDispatch": "dispatch",
    "shade_operation": "shading",
    "VulkanIntersection": "intersection",
    "VulkanRayGeneration": "ray_generation",
    "blit_operation": "blit",
    "VulkanFsr2": "fsr2",
    "VulkanRelaxTemporal": "relax_temporal",
    "VulkanRelaxHistory": "relax_temporal",
    "VulkanReconstruction": "reconstruction",
    "VulkanRelaxSpatial": "relax",
    "VulkanFrameRing": "frames",
    "VulkanFrameSlot": "frames",
    "VulkanToneMapTarget": "output",
    "VulkanRuntime": "vulkan",
    "VulkanCapabilities": "vulkan",
    "VulkanBuffer": "resources",
    "VulkanImage": "resources",
    "VulkanSemaphore": "resources",
    "VulkanSampler": "resources",
    "VulkanCompletion": "resources",
    "VulkanKernel": "kernel",
    "compile_compute": "kernel",
    "VulkanOutput": "output",
    "VulkanOutputFrame": "output",
}
__all__ = list(_MODULES)


def __getattr__(name):
    if name in _MODULES:
        from importlib import import_module

        return getattr(import_module(f"{__name__}.{_MODULES[name]}"), name)
    raise AttributeError(name)
