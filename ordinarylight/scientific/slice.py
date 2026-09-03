"""Orthogonal scalar-field slices and scene adapters."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..materials import MaterialEvaluation, material, unlit_material, vec3
from ..scene import Material, Texture
from .clipping import ClipRegion
from .scalar_field import ProbeResult, ScalarField3D
from .transfer import TransferFunction


_AXIS_TO_DATA_DIMENSION = {"x": 2, "y": 1, "z": 0}


@material
def scientific_slice_material(ctx):
    """Emit interpolated linear float32 scientific colors without lighting."""
    color = ctx.attribute("scientific_rgba", components=4)
    return MaterialEvaluation(
        base_color=vec3(0.0), emission=color.rgb, metallic=0.0,
        roughness=1.0, transmission=0.0, ior=1.0,
        attenuation_color=vec3(1.0),
        attenuation_distance=ctx.attenuation_distance,
    )


@dataclass(frozen=True, slots=True)
class ScalarSlice:
    """One voxel-centered orthogonal plane through a :class:`ScalarField3D`."""

    field: ScalarField3D
    axis: str
    index: int
    transfer_function: TransferFunction
    clipping: ClipRegion | None = None
    values: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.field, ScalarField3D):
            raise TypeError("field must be a ScalarField3D")
        axis = str(self.axis).lower()
        if axis not in _AXIS_TO_DATA_DIMENSION:
            raise ValueError("axis must be 'x', 'y', or 'z'")
        if isinstance(self.index, bool):
            raise TypeError("index must be an integer")
        index = int(self.index)
        dimension = _AXIS_TO_DATA_DIMENSION[axis]
        if index != self.index or not 0 <= index < self.field.data.shape[dimension]:
            raise ValueError("slice index is outside the field")
        if not isinstance(self.transfer_function, TransferFunction):
            raise TypeError("transfer_function must be a TransferFunction")
        if self.clipping is not None and not isinstance(self.clipping, ClipRegion):
            raise TypeError("clipping must be a ClipRegion or None")
        selection = [slice(None)] * 3
        selection[dimension] = index
        values = self.field.data[tuple(selection)]
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "index", index)
        object.__setattr__(self, "values", values)

    @property
    def rgba(self):
        """Linear float32 colors from the shared physical transfer function."""
        return self.transfer_function.colors(self.values)

    @property
    def image_rgba8(self):
        """Renderer-compatible display texture; linear values are quantized."""
        return np.asarray(np.floor(np.clip(self.rgba, 0.0, 1.0) * 255.0 + 0.5), np.uint8)

    @property
    def index_corners(self):
        nx, ny, nz = self.field.data.shape[::-1]
        if self.axis == "x":
            return np.asarray(((self.index, 0, 0), (self.index, ny - 1, 0),
                               (self.index, ny - 1, nz - 1), (self.index, 0, nz - 1)), np.float64)
        if self.axis == "y":
            return np.asarray(((0, self.index, 0), (nx - 1, self.index, 0),
                               (nx - 1, self.index, nz - 1), (0, self.index, nz - 1)), np.float64)
        return np.asarray(((0, 0, self.index), (nx - 1, 0, self.index),
                           (nx - 1, ny - 1, self.index), (0, ny - 1, self.index)), np.float64)

    @property
    def world_corners(self):
        return np.asarray(self.field.index_to_world(self.index_corners), np.float32)

    def probe(self, world_position, *, plane_tolerance=1e-5):
        """Probe the stored value represented by a point on this slice."""
        result = self.field.probe(world_position)
        axis_index = "xyz".index(self.axis)
        on_plane = abs(result.index_position[axis_index] - self.index) <= float(plane_tolerance)
        visible = self.clipping is None or self.clipping.contains(
            self.field, world_position, result.index_position,
        )
        return ProbeResult(
            result.world_position, result.index_position, result.nearest_index,
            result.value, result.valid and on_plane and visible, result.unit,
        )

    def add_to_scene(self, scene, *, name=None, metadata=None):
        """Add an exact float32 vertex-colored slice to the scene."""
        rgba = self.rgba
        alpha = rgba[..., 3]
        if not np.allclose(alpha, alpha.flat[0], rtol=0.0, atol=1e-7):
            raise ValueError(
                "float32 slices require uniform opacity; use "
                "add_texture_to_scene() for varying opacity"
            )
        rows, columns = self.values.shape
        row_coordinates, column_coordinates = np.mgrid[:rows, :columns]
        if self.axis == "z":
            index_vertices = np.column_stack((
                column_coordinates.ravel(), row_coordinates.ravel(),
                np.full(rows * columns, self.index),
            ))
        elif self.axis == "y":
            index_vertices = np.column_stack((
                column_coordinates.ravel(), np.full(rows * columns, self.index),
                row_coordinates.ravel(),
            ))
        else:
            index_vertices = np.column_stack((
                np.full(rows * columns, self.index), column_coordinates.ravel(),
                row_coordinates.ravel(),
            ))
        vertices = np.asarray(self.field.index_to_world(index_vertices), np.float32)
        triangles = []
        for row in range(rows - 1):
            for column in range(columns - 1):
                a = row * columns + column
                triangles.extend(((a, a + 1, a + columns + 1),
                                  (a, a + columns + 1, a + columns)))
        indices = np.asarray(triangles, np.uint32)
        colors = np.ascontiguousarray(rgba.reshape((-1, 4)), np.float32)
        if self.clipping is not None:
            vertices, indices, (colors,) = self.clipping.clip_mesh(
                self.field, vertices, indices, (colors,),
            )
            if not len(indices):
                raise ValueError("slice is empty after clipping")
        slice_metadata = {} if metadata is None else dict(metadata)
        slice_metadata["scientific"] = self.snapshot()
        slice_metadata["scientific"]["representation"] = "float32_vertex_rgba"
        return scene.add_mesh(
            vertices, indices,
            Material(
                opacity=float(alpha.flat[0]), alpha_mode="blend",
                emission_two_sided=True, program=scientific_slice_material,
            ),
            attributes={"scientific_rgba": colors},
            name=name or f"{self.field.name or 'scalar'}-{self.axis}{self.index}",
            metadata=slice_metadata,
        )

    def add_texture_to_scene(self, scene, *, name=None, metadata=None):
        """Add the compatibility RGBA8 texture path supporting varying alpha."""
        texture = Texture(self.image_rgba8, wrap_s="clamp", wrap_t="clamp")
        material = Material(
            base_color=(1.0, 1.0, 1.0), base_color_texture=texture,
            alpha_mode="blend", emission_two_sided=True, program=unlit_material,
        )
        slice_metadata = {} if metadata is None else dict(metadata)
        slice_metadata["scientific"] = self.snapshot()
        slice_metadata["scientific"]["representation"] = "rgba8_texture"
        vertices = self.world_corners
        indices = np.asarray(((0, 1, 2), (0, 2, 3)), np.uint32)
        texcoords = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), np.float32)
        if self.clipping is not None:
            vertices, indices, (texcoords,) = self.clipping.clip_mesh(
                self.field, vertices, indices, (texcoords,),
            )
            if not len(indices):
                raise ValueError("slice is empty after clipping")
        return scene.add_mesh(
            vertices, indices, material, texcoords=texcoords,
            name=name or f"{self.field.name or 'scalar'}-{self.axis}{self.index}",
            metadata=slice_metadata,
        )

    def snapshot(self):
        return {
            "kind": "scalar_slice", "axis": self.axis, "index": self.index,
            "field": self.field.snapshot(),
            "transfer_function": self.transfer_function.snapshot(self.field.data),
            "clipping": None if self.clipping is None else self.clipping.snapshot(),
        }
