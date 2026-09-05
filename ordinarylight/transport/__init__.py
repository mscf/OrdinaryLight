"""Application-indexed light transport and history contracts, version 1."""

from dataclasses import dataclass
from importlib.resources import files

import numpy as np

TRANSPORT_ABI_VERSION = 1
# Each group is 16-byte aligned and maps directly to the GLSL contract.
SURFACE_SAMPLE_DTYPE = np.dtype(
    [
        ("position", "<f4", (4,)),
        ("geometric_normal", "<f4", (4,)),
        ("shading_normal", "<f4", (4,)),
        ("incoming", "<f4", (4,)),
        ("identity", "<u4", (4,)),  # owner ID, sample index, material/primitive, flags
        ("media", "<u4", (4,)),  # outside medium, inside medium, reserved, reserved
    ]
)


def shader_source(component):
    """Return a versioned, include-expanded transport GLSL component."""
    if component not in {"types", "lighting", "volumes", "contracts"}:
        raise ValueError("Unknown transport component")
    from ..shaders.compiler import _expanded_shader_source

    return _expanded_shader_source(f"transport_v1/{component}.glsl")


def shader_directory():
    """Packaged component directory; include root is ordinarylight/shaders."""
    return files("ordinarylight.shaders").joinpath("transport_v1")


@dataclass(frozen=True)
class HistoryEntry:
    value: object
    dependencies: frozenset[str]


class SampleHistory:
    """History keyed by application identity, with explicit invalidation domains.

    Owns no GPU allocations. Returned invalidated values are the application's
    responsibility to retire after their last completion. For example diffuse
    history can depend on geometry/materials/lights, and specular on camera too.
    """

    def __init__(self):
        self._entries = {}

    def set(self, identity, value, *, dependencies):
        domains = frozenset(dependencies)
        if not domains or not all(
            isinstance(domain, str) and domain for domain in domains
        ):
            raise ValueError("History requires named invalidation dependencies")
        previous = self._entries.get(identity)
        self._entries[identity] = HistoryEntry(value, domains)
        return None if previous is None else previous.value

    def get(self, identity, default=None):
        entry = self._entries.get(identity)
        return default if entry is None else entry.value

    def invalidate(self, *domains, identities=None):
        selected = set(self._entries) if identities is None else set(identities)
        affected = frozenset(domains)
        retired = {}
        for identity in selected:
            entry = self._entries.get(identity)
            if entry is not None and (not affected or entry.dependencies & affected):
                retired[identity] = self._entries.pop(identity).value
        return retired

    def __len__(self):
        return len(self._entries)
