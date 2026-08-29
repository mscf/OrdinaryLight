"""Explicit backend preference and Vulkan fallback policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BackendSelection:
    """Serializable explanation of an automatic backend decision."""

    requested: str
    selected: str
    fallback: bool = False
    reason: str | None = None

    def as_dict(self):
        return {
            "requested": self.requested,
            "selected": self.selected,
            "fallback": self.fallback,
            "reason": self.reason,
        }


def _raster_backend(*, config=None, device_name=None):
    from .backends import VulkanRasterBackend
    from .raster import RasterConfig, RasterProgram

    raster_config = config if isinstance(config, RasterConfig) else None
    try:
        program = RasterProgram.scene(target="spirv")
    except RuntimeError as error:
        if "ordinaryshade" not in str(error).lower():
            raise
        raise RuntimeError(
            "Vulkan raster selection requires Ordinary Shade to compile the "
            "built-in graphics program; install ordinaryshade or pass an "
            "explicit precompiled VulkanRasterBackend"
        ) from error
    return VulkanRasterBackend(
        program, config=raster_config, device_name=device_name,
    )


def _gi_backend(*, config=None, device_name=None):
    from .backends.vulkan import RendererConfig, VulkanRayTracingBackend

    if config is not None:
        return VulkanRayTracingBackend(config=config)
    return VulkanRayTracingBackend(
        config=RendererConfig(device_name=device_name)
    )


def select_vulkan_backend(
    preference="auto", *, config=None, device_name=None,
):
    """Construct the requested Vulkan backend and record the decision.

    ``auto`` prefers the hardware GI backend and falls back to native Vulkan
    rasterization only when no matching ray-query adapter exists. Explicit
    ``gi`` and ``raster`` requests never silently change renderer class.
    """
    preference = str(preference).lower().replace("_", "-")
    aliases = {"ray-tracing": "gi", "raytracing": "gi", "rt": "gi"}
    preference = aliases.get(preference, preference)
    if preference not in {"auto", "gi", "raster"}:
        raise ValueError("backend preference must be auto, gi, or raster")
    configured_name = getattr(config, "device_name", None)
    if device_name is not None and configured_name is not None:
        raise ValueError("pass device_name directly or through config, not both")
    selected_name = device_name or configured_name

    if preference == "raster":
        backend = _raster_backend(config=config, device_name=selected_name)
        decision = BackendSelection("raster", "vulkan-raster")
    elif preference == "gi":
        backend = _gi_backend(config=config, device_name=selected_name)
        decision = BackendSelection("gi", "vulkan-gi")
    else:
        try:
            backend = _gi_backend(config=config, device_name=selected_name)
            decision = BackendSelection("auto", "vulkan-gi")
        except RuntimeError as error:
            reason = str(error)
            if not reason.startswith(
                "No compatible Vulkan ray-tracing adapter:"
            ):
                raise
            backend = _raster_backend(config=config, device_name=selected_name)
            decision = BackendSelection(
                "auto", "vulkan-raster", fallback=True, reason=reason,
            )
    backend.backend_selection = decision.as_dict()
    return backend


__all__ = ["BackendSelection", "select_vulkan_backend"]
