"""Toolkit-neutral extension contract for the Ordinary Light workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib.util
import os
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Iterable, Mapping

import numpy as np

from ..cameras import PerspectiveCamera
from ..scene import Scene


@dataclass(frozen=True)
class OrbitCamera:
    """Optional authored orbit used to present a showcase scene."""

    target: tuple[float, float, float] | None = None
    radius: float | None = None
    height: float | None = None
    arc_radians: float | None = None
    vertical_fov_degrees: float = 45.0

    def __post_init__(self):
        if self.target is not None:
            target = tuple(float(value) for value in self.target)
            if len(target) != 3 or not np.all(np.isfinite(target)):
                raise ValueError("camera target must contain three finite values")
            object.__setattr__(self, "target", target)
        for name in ("radius", "height", "arc_radians"):
            value = getattr(self, name)
            if value is not None and not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 1.0 <= self.vertical_fov_degrees < 179.0:
            raise ValueError("vertical_fov_degrees must be in [1, 179)")

    def fitted(self, scene: Scene):
        """Return ``(target, radius, height)`` with missing values fitted."""
        bounds_min, bounds_max = scene.bounds()
        center = (np.asarray(bounds_min) + np.asarray(bounds_max)) * 0.5
        extent_radius = max(
            float(np.linalg.norm(np.asarray(bounds_max) - bounds_min)) * 0.5,
            0.5,
        )
        target = center if self.target is None else np.asarray(self.target)
        radius = extent_radius * 2.2 if self.radius is None else self.radius
        height = (
            float(target[1] + extent_radius * 0.25)
            if self.height is None else self.height
        )
        return target.astype(np.float64), float(radius), float(height)

    def camera(self, scene: Scene, angle=0.0):
        target, radius, height = self.fitted(scene)
        return PerspectiveCamera(
            (
                float(target[0] + radius * np.sin(angle)),
                height,
                float(target[2] + radius * np.cos(angle)),
            ),
            tuple(float(value) for value in target),
            vertical_fov_degrees=self.vertical_fov_degrees,
        )


@dataclass(frozen=True)
class Showcase:
    """One lazily constructed workbench feature demonstration."""

    id: str
    title: str
    build: Callable[[], Scene]
    description: str = ""
    camera: OrbitCamera = field(default_factory=OrbitCamera)
    renderer: Mapping[str, object] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    def __post_init__(self):
        identifier = str(self.id).strip()
        if not identifier or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-_"
            for character in identifier
        ):
            raise ValueError(
                "showcase id must use lowercase letters, digits, '-' or '_'"
            )
        if not str(self.title).strip():
            raise ValueError("showcase title cannot be empty")
        if not callable(self.build):
            raise TypeError("showcase build must be callable")
        if not isinstance(self.camera, OrbitCamera):
            raise TypeError("showcase camera must be an OrbitCamera")
        object.__setattr__(self, "id", identifier)
        object.__setattr__(self, "title", str(self.title).strip())
        object.__setattr__(self, "description", str(self.description).strip())
        object.__setattr__(self, "renderer", MappingProxyType(dict(self.renderer)))
        object.__setattr__(self, "tags", tuple(str(tag) for tag in self.tags))

    def create_scene(self):
        scene = self.build()
        if not isinstance(scene, Scene):
            raise TypeError(
                f"showcase {self.id!r} returned {type(scene).__name__}, not Scene"
            )
        return scene


class ShowcaseCatalog:
    """Ordered collection discovered from one or more simple Python scripts."""

    def __init__(self, showcases=()):
        self._items = []
        self._by_id = {}
        for showcase in showcases:
            self.add(showcase)

    def add(self, showcase):
        if not isinstance(showcase, Showcase):
            raise TypeError("catalog entries must be Showcase objects")
        if showcase.id in self._by_id:
            raise ValueError(f"duplicate showcase id {showcase.id!r}")
        self._items.append(showcase)
        self._by_id[showcase.id] = showcase
        return showcase

    def __iter__(self):
        return iter(self._items)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, key):
        return self._items[key] if isinstance(key, int) else self._by_id[key]


def _script_showcases(module, path):
    declared = getattr(module, "SHOWCASES", None)
    if declared is None:
        single = getattr(module, "SHOWCASE", None)
        if single is None:
            raise ValueError(
                f"{path} must define SHOWCASE or SHOWCASES"
            )
        declared = (single,)
    if isinstance(declared, Showcase):
        declared = (declared,)
    return tuple(declared)


def load_showcase_script(path):
    """Load showcases from one script without modifying ``sys.path``."""
    path = Path(path).expanduser().resolve()
    if path.suffix != ".py" or not path.is_file():
        raise ValueError(f"showcase script does not exist: {path}")
    digest = hashlib.sha256(str(path).encode()).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(
        f"_ordinarylight_showcase_{digest}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load showcase script {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    showcases = _script_showcases(module, path)
    if any(not isinstance(item, Showcase) for item in showcases):
        raise TypeError(f"{path} contains a non-Showcase entry")
    return showcases


def discover_showcases(paths=()):
    """Discover ``*.py`` scripts from files/directories and environment paths."""
    requested = [Path(path) for path in paths]
    environment = os.environ.get("ORDINARYLIGHT_SHOWCASE_PATH", "")
    requested.extend(Path(path) for path in environment.split(os.pathsep) if path)
    scripts = []
    for path in requested:
        path = path.expanduser()
        if path.is_dir():
            scripts.extend(sorted(
                child for child in path.glob("*.py")
                if not child.name.startswith("_")
            ))
        else:
            scripts.append(path)
    catalog = ShowcaseCatalog()
    for script in scripts:
        for showcase in load_showcase_script(script):
            catalog.add(showcase)
    return catalog


__all__ = [
    "OrbitCamera", "Showcase", "ShowcaseCatalog", "discover_showcases",
    "load_showcase_script",
]
