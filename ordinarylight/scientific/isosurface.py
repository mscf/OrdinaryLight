"""Deterministic dependency-free scalar isosurface extraction."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..materials import unlit_material
from ..scene import Material
from .clipping import ClipRegion
from .scalar_field import ProbeResult, ScalarField3D
from .transfer import TransferFunction


_CUBE_CORNERS = np.asarray((
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
), np.int64)
_TETRAHEDRA = (
    (0, 5, 1, 6), (0, 1, 2, 6), (0, 2, 3, 6),
    (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6),
)
_TETRA_EDGES = ((0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3))
_TRIANGLE_EDGES = (
    (), ((0, 2, 3),), ((0, 1, 4),), ((1, 2, 3), (1, 3, 4)),
    ((1, 2, 5),), ((0, 1, 3), (1, 5, 3)),
    ((0, 2, 4), (2, 5, 4)), ((3, 4, 5),),
    ((3, 5, 4),), ((0, 4, 2), (2, 4, 5)),
    ((0, 3, 1), (1, 3, 5)), ((1, 5, 2),),
    ((1, 4, 3), (1, 3, 2)), ((0, 4, 1),), ((0, 3, 2),), (),
)


@dataclass(frozen=True, slots=True)
class ScalarIsosurface:
    """Triangle mesh representing one physical scalar value."""

    field: ScalarField3D
    value: float
    transfer_function: TransferFunction
    clipping: ClipRegion | None
    index_vertices: np.ndarray = field(repr=False, compare=False)
    indices: np.ndarray = field(repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.field, ScalarField3D):
            raise TypeError("field must be a ScalarField3D")
        value = float(self.value)
        if not np.isfinite(value):
            raise ValueError("isosurface value must be finite")
        if not isinstance(self.transfer_function, TransferFunction):
            raise TypeError("transfer_function must be a TransferFunction")
        if self.clipping is not None and not isinstance(self.clipping, ClipRegion):
            raise TypeError("clipping must be a ClipRegion or None")
        vertices = np.ascontiguousarray(self.index_vertices, dtype=np.float32)
        indices = np.ascontiguousarray(self.indices, dtype=np.uint32)
        if vertices.ndim != 2 or vertices.shape[1:] != (3,):
            raise ValueError("index_vertices must have shape (count, 3)")
        if indices.ndim != 2 or indices.shape[1:] != (3,):
            raise ValueError("indices must have shape (count, 3)")
        vertices.flags.writeable = False
        indices.flags.writeable = False
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "index_vertices", vertices)
        object.__setattr__(self, "indices", indices)

    @classmethod
    def from_field(cls, field, value, transfer_function, *, clipping=None):
        value = float(value)
        data = np.asarray(field.data)
        vertices = []
        triangles = []
        vertex_cache = {}
        depth, height, width = data.shape

        def edge_vertex(a, b, a_value, b_value):
            a_key = tuple(map(int, a))
            b_key = tuple(map(int, b))
            key = tuple(sorted((a_key, b_key)))
            existing = vertex_cache.get(key)
            if existing is not None:
                return existing
            denominator = float(b_value - a_value)
            weight = 0.5 if denominator == 0.0 else (value - float(a_value)) / denominator
            position = np.asarray(a, np.float64) + np.clip(weight, 0.0, 1.0) * (
                np.asarray(b, np.float64) - np.asarray(a, np.float64)
            )
            index = len(vertices)
            vertices.append(position)
            vertex_cache[key] = index
            return index

        for z in range(depth - 1):
            for y in range(height - 1):
                for x in range(width - 1):
                    origin = np.asarray((x, y, z), np.int64)
                    corners = origin + _CUBE_CORNERS
                    corner_values = data[corners[:, 2], corners[:, 1], corners[:, 0]]
                    if not np.isfinite(corner_values).all():
                        continue
                    if value < float(corner_values.min()) or value > float(corner_values.max()):
                        continue
                    for tetrahedron in _TETRAHEDRA:
                        points = corners[list(tetrahedron)]
                        values = corner_values[list(tetrahedron)]
                        case = sum((float(sample) >= value) << index for index, sample in enumerate(values))
                        for triangle_edges in _TRIANGLE_EDGES[case]:
                            triangle = []
                            for edge_index in triangle_edges:
                                a_index, b_index = _TETRA_EDGES[edge_index]
                                triangle.append(edge_vertex(
                                    points[a_index], points[b_index],
                                    values[a_index], values[b_index],
                                ))
                            if len(set(triangle)) == 3:
                                triangles.append(triangle)
        index_vertices = np.asarray(vertices, np.float32).reshape((-1, 3))
        triangle_indices = np.asarray(triangles, np.uint32).reshape((-1, 3))
        if clipping is not None and len(triangle_indices):
            world_vertices = field.index_to_world(index_vertices)
            world_vertices, triangle_indices, _attributes = clipping.clip_mesh(
                field, world_vertices, triangle_indices,
            )
            index_vertices = field.world_to_index(world_vertices)
        return cls(
            field, value, transfer_function, clipping,
            index_vertices, triangle_indices,
        )

    @property
    def world_vertices(self):
        return np.asarray(self.field.index_to_world(self.index_vertices), np.float32)

    @property
    def rgba(self):
        return self.transfer_function.colors(np.asarray(self.value, np.float32))

    def probe(self, world_position):
        """Describe a picked surface point while preserving the authored iso value."""
        index_position = self.field.world_to_index(world_position)
        nearest = np.floor(index_position + 0.5).astype(np.int64)
        shape = np.asarray(self.field.data.shape[::-1])
        inside = bool(np.all(nearest >= 0) and np.all(nearest < shape))
        visible = self.clipping is None or self.clipping.contains(
            self.field, world_position, index_position,
        )
        return ProbeResult(
            tuple(map(float, world_position)), tuple(map(float, index_position)),
            tuple(map(int, nearest)), self.value, inside and visible, self.field.unit,
        )

    def add_to_scene(self, scene, *, name=None, metadata=None):
        """Add the isosurface as an unlit two-sided scene mesh."""
        if not len(self.indices):
            raise ValueError("isosurface is empty at the requested value")
        rgba = np.asarray(self.rgba, np.float64)
        surface_metadata = {} if metadata is None else dict(metadata)
        surface_metadata["scientific"] = self.snapshot()
        return scene.add_mesh(
            self.world_vertices, self.indices,
            Material(
                base_color=tuple(np.clip(rgba[:3], 0.0, 1.0)),
                opacity=float(np.clip(rgba[3], 0.0, 1.0)),
                alpha_mode="blend", emission_two_sided=True,
                program=unlit_material,
            ),
            name=name or f"{self.field.name or 'scalar'}-iso-{self.value:g}",
            metadata=surface_metadata,
        )

    def snapshot(self):
        return {
            "kind": "scalar_isosurface", "value": self.value,
            "vertex_count": len(self.index_vertices),
            "triangle_count": len(self.indices), "algorithm": "marching_tetrahedra",
            "field": self.field.snapshot(),
            "transfer_function": self.transfer_function.snapshot(self.field.data),
            "clipping": None if self.clipping is None else self.clipping.snapshot(),
        }
