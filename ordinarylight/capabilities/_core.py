"""Renderer-neutral capability discovery."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class RendererCapabilities:
    """Immutable description of one initialized renderer implementation.

    Feature names describe semantic renderer behavior rather than Vulkan
    extensions. Applications can therefore select optional paths without
    importing or inspecting backend implementation classes.
    """

    renderer: str
    outputs: tuple[str, ...] = ("color",)
    features: frozenset[str] = frozenset()
    limits: Mapping[str, int | float] = field(default_factory=dict)
    device: Any | None = None
    selection: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.renderer, str) or not self.renderer:
            raise TypeError("renderer must be a non-empty string")
        outputs = tuple(self.outputs)
        if not outputs or any(not isinstance(name, str) or not name for name in outputs):
            raise TypeError("outputs must contain non-empty strings")
        features = frozenset(self.features)
        if any(not isinstance(name, str) or not name for name in features):
            raise TypeError("features must contain non-empty strings")
        limits = dict(self.limits)
        if any(not isinstance(name, str) or not name for name in limits):
            raise TypeError("limit names must be non-empty strings")
        if any(not isinstance(value, (int, float)) for value in limits.values()):
            raise TypeError("limit values must be numeric")
        selection = dict(self.selection)
        if any(not isinstance(name, str) or not name for name in selection):
            raise TypeError("selection names must be non-empty strings")
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "limits", MappingProxyType(limits))
        object.__setattr__(self, "selection", MappingProxyType(selection))

    def supports(self, feature: str) -> bool:
        """Return whether a semantic renderer feature is available."""
        return feature in self.features

    def supports_output(self, output: str) -> bool:
        """Return whether a named render product is available."""
        return output in self.outputs

    def require(self, *features: str) -> None:
        """Raise a useful error unless every requested feature is available."""
        missing = tuple(feature for feature in features if feature not in self.features)
        if missing:
            raise RuntimeError(
                f"renderer {self.renderer!r} does not support {missing}"
            )

    def as_dict(self) -> dict[str, Any]:
        """Return a serialization-friendly capability record."""
        device_name = getattr(self.device, "name", self.device)
        return {
            "renderer": self.renderer,
            "outputs": self.outputs,
            "features": tuple(sorted(self.features)),
            "limits": dict(self.limits),
            "device": device_name,
            "selection": dict(self.selection),
        }


def capabilities_from_renderer(renderer) -> RendererCapabilities:
    """Normalize an optional renderer capability declaration."""
    declared = getattr(renderer, "capabilities", None)
    if isinstance(declared, RendererCapabilities):
        selection = getattr(renderer, "renderer_selection", None)
        if selection:
            return replace(declared, selection=selection)
        return declared
    values = dict(declared or {})
    return RendererCapabilities(
        renderer=values.pop("renderer", type(renderer).__name__),
        outputs=tuple(values.pop(
            "outputs", getattr(renderer, "available_outputs", ("color",))
        )),
        features=frozenset(values.pop("features", ())),
        limits=values.pop("limits", {}),
        device=values.pop("device", getattr(renderer, "device", None)),
        selection=values.pop(
            "selection", getattr(renderer, "renderer_selection", {})
        ),
    )


__all__ = ["RendererCapabilities", "capabilities_from_renderer"]
