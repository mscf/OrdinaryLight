"""Automatic raster reflection-probe capture and refresh policies."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from ..cameras import PerspectiveCamera


_CUBE_FACES = (
    ((1, 0, 0), (0, -1, 0)), ((-1, 0, 0), (0, -1, 0)),
    ((0, 1, 0), (0, 0, 1)), ((0, -1, 0), (0, 0, -1)),
    ((0, 0, 1), (0, -1, 0)), ((0, 0, -1), (0, -1, 0)),
)


def _render_face(renderer, scene, camera, extent):
    if hasattr(renderer, "render_frame"):
        return np.asarray(renderer.render_frame(scene, camera, *extent), np.float32)
    result = renderer.render(scene, camera, extent, outputs=("color",))
    return np.asarray(result.color, np.float32)


def _cube_to_equirectangular(faces, height, width):
    yy, xx = np.mgrid[0:height, 0:width]
    longitude = (xx + 0.5) / width * (2.0 * np.pi) - np.pi
    latitude = np.pi * 0.5 - (yy + 0.5) / height * np.pi
    directions = np.stack((
        np.cos(latitude) * np.cos(longitude), np.sin(latitude),
        np.cos(latitude) * np.sin(longitude),
    ), axis=-1)
    absolute = np.abs(directions)
    axis = np.argmax(absolute, axis=-1)
    output = np.zeros((height, width, 3), np.float32)
    # Face-local coordinates match the six camera bases above.
    mappings = (
        (0, (axis == 0) & (directions[..., 0] >= 0), -2, -1),
        (1, (axis == 0) & (directions[..., 0] < 0), 2, -1),
        (2, (axis == 1) & (directions[..., 1] >= 0), 0, 2),
        (3, (axis == 1) & (directions[..., 1] < 0), 0, -2),
        (4, (axis == 2) & (directions[..., 2] >= 0), 0, -1),
        (5, (axis == 2) & (directions[..., 2] < 0), -0, -1),
    )
    # Explicit formulas avoid hidden cube-map coordinate conventions.
    components = (
        (-directions[..., 2], -directions[..., 1], absolute[..., 0]),
        (directions[..., 2], -directions[..., 1], absolute[..., 0]),
        (directions[..., 0], directions[..., 2], absolute[..., 1]),
        (directions[..., 0], -directions[..., 2], absolute[..., 1]),
        (directions[..., 0], -directions[..., 1], absolute[..., 2]),
        (-directions[..., 0], -directions[..., 1], absolute[..., 2]),
    )
    for face_index, (_unused, mask, *_rest) in enumerate(mappings):
        u, v, denominator = components[face_index]
        face = faces[face_index]
        px = np.clip(((u / denominator + 1.0) * 0.5 * face.shape[1]).astype(int), 0, face.shape[1] - 1)
        py = np.clip(((v / denominator + 1.0) * 0.5 * face.shape[0]).astype(int), 0, face.shape[0] - 1)
        output[mask] = face[py[mask], px[mask], :3]
    return output


def capture_reflection_probe(renderer, scene, probe, *, resolution=None):
    """Capture six raster views and return a probe containing HDR radiance."""
    size = int(resolution or probe.capture_resolution)
    position = np.asarray(probe.position, np.float32)
    saved = list(scene.reflection_probes)
    scene.reflection_probes.clear()
    scene._changed(shading=True)
    try:
        faces = []
        for direction, up in _CUBE_FACES:
            camera = PerspectiveCamera(
                position, position + np.asarray(direction, np.float32), up,
                vertical_fov_degrees=90.0,
            )
            faces.append(_render_face(renderer, scene, camera, (size, size)))
    finally:
        scene.reflection_probes[:] = saved
        scene._changed(shading=True)
    return probe.with_image(_cube_to_equirectangular(faces, size, size * 2))


@dataclass
class ProbeCaptureManager:
    """Track scene revisions and apply each probe's refresh policy."""

    _revisions: dict[int, int] = field(default_factory=dict, init=False)
    _requested: set[int] = field(default_factory=set, init=False)
    _aliases: dict[int, int] = field(default_factory=dict, init=False)

    def _current_id(self, probe_id):
        seen = set()
        while probe_id in self._aliases and probe_id not in seen:
            seen.add(probe_id)
            probe_id = self._aliases[probe_id]
        return probe_id

    def request(self, probe):
        self._requested.add(self._current_id(id(probe)))

    def needs_refresh(self, probe, scene):
        if not probe.captured:
            return True
        if probe.refresh_policy == "always":
            return True
        if probe.refresh_policy == "scene-change":
            return self._revisions.get(id(probe)) != scene.revision
        return probe.refresh_policy == "on-demand" and id(probe) in self._requested

    def refresh(self, renderer, scene, *, force=False):
        replacements = []
        for index, probe in enumerate(tuple(scene.reflection_probes)):
            if force or self.needs_refresh(probe, scene):
                captured = capture_reflection_probe(renderer, scene, probe)
                replacements.append((index, probe, captured))
        if replacements:
            for index, previous, captured in replacements:
                scene.reflection_probes[index] = captured
                self._requested.discard(id(previous))
                self._aliases[id(previous)] = id(captured)
            scene._changed(shading=True)
            for _index, _previous, captured in replacements:
                self._revisions[id(captured)] = scene.revision
        return tuple(captured for _index, _previous, captured in replacements)
