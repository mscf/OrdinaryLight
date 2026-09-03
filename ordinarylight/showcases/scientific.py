"""End-to-end scientific scalar-field showcase and update controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import ordinarylight as ol


@dataclass
class ScientificScalarFieldShowcase:
    """A scene whose volume, slices, and isosurface share one data model."""

    scene: ol.Scene
    field: ol.ScalarField3D
    transfer_function: ol.TransferFunction
    clipping: ol.ClipRegion
    volume: object
    slices: dict[str, ol.ScalarSlice]
    slice_meshes: dict[str, object]
    isosurface: ol.ScalarIsosurface
    isosurface_mesh: object
    synced_revision: int = 0

    def update_region(self, offset, values, *, refresh_isosurface=False):
        """Update all field-backed views while preserving scene identities."""
        self.field.update(offset, values)
        self.synced_revision = self.field.sync_volume(
            self.scene, self.volume, self.transfer_function,
            since_revision=self.synced_revision,
        )
        self.refresh_slices()
        if refresh_isosurface:
            self.refresh_isosurface()
        return self.synced_revision

    def refresh_slices(self):
        """Recompute slice colors in place from the shared transfer function."""
        for axis, previous in tuple(self.slices.items()):
            current = self.field.slice(
                axis, previous.index, self.transfer_function,
                clipping=self.clipping,
            )
            mesh = self.slice_meshes[axis]
            # Clipping may add boundary vertices, so use the same construction
            # path as initial creation and copy its validated geometry.
            scratch = ol.Scene()
            replacement = current.add_texture_to_scene(scratch)
            self.scene.update_mesh(
                mesh, vertices=replacement.vertices, indices=replacement.indices,
                material=replacement.material, texcoords=replacement.texcoords,
            )
            self.slices[axis] = current

    def refresh_isosurface(self):
        """Re-extract the authored isovalue while retaining the mesh handle."""
        current = self.field.isosurface(
            self.isosurface.value, self.transfer_function,
            clipping=self.clipping,
        )
        if not len(current.indices):
            raise ValueError("updated field has no triangles at the isovalue")
        self.scene.update_mesh(
            self.isosurface_mesh,
            vertices=current.world_vertices, indices=current.indices,
        )
        self.isosurface = current


def _example_field(resolution):
    coordinates = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    z, y, x = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    primary = np.exp(-3.8 * (x * x + 1.3 * y * y + 0.8 * z * z))
    secondary = 0.65 * np.exp(
        -14.0 * ((x + 0.42) ** 2 + (y - 0.22) ** 2 + (z + 0.1) ** 2)
    )
    ripple = 0.08 * np.sin(8.0 * x) * np.cos(6.0 * z) * primary
    return np.asarray(primary + secondary + ripple, np.float32)


def build_scientific_scalar_field_showcase(resolution=32):
    """Build the vertical-slice milestone as one inspectable state bundle."""
    resolution = int(resolution)
    if resolution < 4:
        raise ValueError("resolution must be at least 4")
    data = _example_field(resolution)
    spacing = (2.0 / (resolution - 1),) * 3
    field = ol.ScalarField3D(
        data, spacing=spacing, origin=(-1.0, -1.0, -1.0),
        unit="normalized concentration", name="scientific-scalar-field",
        metadata={"source": "analytic reproducible showcase"},
    )
    transfer = ol.TransferFunction.from_colormap(
        "viridis", samples=256, opacity=(0.0, 0.18),
        mapping=ol.ScalarMapping("linear", (0.0, 1.4)),
    )
    clipping = ol.ClipRegion(roi=ol.RegionOfInterest(
        (0, 0, 0), (resolution - 1, resolution - 1, resolution - 1),
    ))
    scene = ol.Scene(metadata={"showcase": "scientific-scalar-field"})
    volume = field.add_volume(
        scene, transfer, clipping=clipping, step_size=0.025,
        density_scale=0.7, emission_scale=0.65,
    )
    middle = resolution // 2
    slices = {
        axis: field.slice(axis, middle, transfer, clipping=clipping)
        for axis in "xyz"
    }
    slice_meshes = {
        axis: scalar_slice.add_texture_to_scene(
            scene, name=f"scientific-{axis}-slice"
        )
        for axis, scalar_slice in slices.items()
    }
    isosurface = field.isosurface(0.48, transfer, clipping=clipping)
    isosurface_mesh = isosurface.add_to_scene(scene, name="scientific-isosurface")
    return ScientificScalarFieldShowcase(
        scene, field, transfer, clipping, volume, slices, slice_meshes,
        isosurface, isosurface_mesh,
    )


def build_scientific_scalar_field_scene(resolution=32):
    """Workbench-compatible scene-only form of the scientific showcase."""
    return build_scientific_scalar_field_showcase(resolution).scene
