"""End-to-end scientific scalar-field showcase and update controller."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

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
    representation: str = "combined"
    slices_dirty: bool = False
    isosurface_dirty: bool = False

    def update_region(self, offset, values, *, refresh_isosurface=False):
        """Update all field-backed views while preserving scene identities."""
        self.field.update(offset, values)
        self.synced_revision = self.field.sync_volume(
            self.scene, self.volume, self.transfer_function,
            since_revision=self.synced_revision,
        )
        self.slices_dirty = True
        if refresh_isosurface and self.representation not in {"combined", "isosurface"}:
            self.refresh_isosurface()
        else:
            self.isosurface_dirty = True
        return self.synced_revision

    def refresh_slices(self, axes="xyz", *, geometry=True):
        """Refresh selected slices, changing topology only when required."""
        for axis in axes:
            previous = self.slices[axis]
            current = self.field.slice(
                axis, previous.index, self.transfer_function,
                clipping=self.clipping,
            )
            mesh = self.slice_meshes[axis]
            # Clipping may add boundary vertices, so use the same construction
            # path as initial creation and copy its validated geometry.
            scratch = ol.Scene()
            replacement = current.add_texture_to_scene(scratch)
            if geometry:
                self.scene.update_mesh(
                    mesh, vertices=replacement.vertices,
                    indices=replacement.indices,
                    material=replacement.material,
                    texcoords=replacement.texcoords,
                    texcoords1=None, attributes={},
                )
            else:
                self.scene.update_mesh(mesh, material=replacement.material)
            self.slices[axis] = current

    def refresh_isosurface(self, *, geometry=True):
        """Refresh isosurface material, re-extracting only when requested."""
        current = self.isosurface
        if geometry:
            current = self.field.isosurface(
                self.isosurface.value, self.transfer_function,
                clipping=self.clipping,
            )
        elif current.transfer_function is not self.transfer_function:
            current = ol.ScalarIsosurface(
                self.field, current.value, self.transfer_function,
                self.clipping, current.index_vertices, current.indices,
            )
        if not len(current.indices):
            raise ValueError("updated field has no triangles at the isovalue")
        scratch = ol.Scene()
        replacement = current.add_to_scene(scratch)
        if geometry:
            self.scene.update_mesh(
                self.isosurface_mesh, vertices=replacement.vertices,
                indices=replacement.indices, material=replacement.material,
                texcoords=None, texcoords1=None, attributes={},
            )
        else:
            self.scene.update_mesh(
                self.isosurface_mesh, material=replacement.material,
            )
        self.isosurface = current

    def replace_field(self, field):
        """Switch to a shape-compatible frame without replacing scene handles."""
        if not isinstance(field, ol.ScalarField3D):
            raise TypeError("field must be a ScalarField3D")
        if field.data.shape != self.field.data.shape:
            raise ValueError("replacement field shape must match the current field")
        self.field = field
        self.synced_revision = field.revision
        payload, value_range, _valid, reserve_missing = (
            self.transfer_function.gpu_volume_payload(field.data)
        )
        planes = tuple(
            (*plane.normal, plane.offset)
            for plane in self.clipping.world_planes(field)
        )
        self.scene.update_volume(
            self.volume, data=payload, value_range=value_range,
            value_mapping=(
                self.transfer_function.mapping.mode
                if self.transfer_function.mapping.mode in {"log", "symlog"}
                else "linear"
            ),
            linear_threshold=self.transfer_function.mapping.linear_threshold,
            missing_data=reserve_missing,
            material=self.transfer_function.volume_material(
                reserve_missing=reserve_missing,
                **{
                    name: getattr(self.volume.material, name) for name in (
                        "density_scale", "emission_scale", "step_size",
                        "scattering_scale", "scattering_color", "phase_function",
                        "anisotropy", "scattering_albedo", "scattering_orders",
                    )
                },
            ), transform=field.volume_transform(), clip_planes=planes,
        )
        self.slices_dirty = True
        self.isosurface_dirty = True

    def set_transfer_function(self, transfer_function):
        """Apply one transfer definition to volume, slices, and isosurface."""
        if not isinstance(transfer_function, ol.TransferFunction):
            raise TypeError("transfer_function must be a TransferFunction")
        previous = self.volume.material
        options = {
            name: getattr(previous, name) for name in (
                "density_scale", "emission_scale", "step_size",
                "scattering_scale", "scattering_color", "phase_function",
                "anisotropy", "scattering_albedo", "scattering_orders",
            )
        }
        payload, value_range, _valid, reserve_missing = (
            transfer_function.gpu_volume_payload(self.field.data)
        )
        material = transfer_function.volume_material(
            reserve_missing=reserve_missing, **options
        )
        current_raw = True
        next_raw = True
        if transfer_function.mapping == self.transfer_function.mapping:
            # Color/opacity edits retain the resident encoded scalar texture;
            # only the small transfer lookup and volume header change.
            self.scene.update_volume(self.volume, material=material)
        elif current_raw and next_raw:
            self.scene.update_volume(
                self.volume, value_range=value_range,
                value_mapping=(
                    transfer_function.mapping.mode
                    if transfer_function.mapping.mode in {"log", "symlog"}
                    else "linear"
                ),
                linear_threshold=transfer_function.mapping.linear_threshold,
                missing_data=reserve_missing,
                material=material,
            )
        else:
            self.scene.update_volume(
                self.volume, data=payload, value_range=value_range,
                value_mapping=(
                    transfer_function.mapping.mode
                    if transfer_function.mapping.mode in {"log", "symlog"}
                    else "linear"
                ),
                linear_threshold=transfer_function.mapping.linear_threshold,
                missing_data=reserve_missing,
                material=material,
            )
        self.transfer_function = transfer_function
        self.slices = {
            axis: self.field.slice(
                axis, scalar_slice.index, transfer_function,
                clipping=self.clipping,
            )
            for axis, scalar_slice in self.slices.items()
        }
        if self.isosurface.transfer_function is not transfer_function:
            self.isosurface = ol.ScalarIsosurface(
                self.field, self.isosurface.value, transfer_function,
                self.clipping, self.isosurface.index_vertices,
                self.isosurface.indices,
            )
        self.slices_dirty = True
        self.isosurface_dirty = True

    def set_clipping(self, clipping):
        """Apply shared clipping to every representation."""
        if not isinstance(clipping, ol.ClipRegion):
            raise TypeError("clipping must be a ClipRegion")
        planes = tuple(
            (*plane.normal, plane.offset)
            for plane in clipping.world_planes(self.field)
        )
        if len(planes) > 8:
            raise ValueError("renderer clipping supports at most 8 world planes")
        self.clipping = clipping
        self.scene.update_volume(self.volume, clip_planes=planes)
        self.slices_dirty = True
        self.isosurface_dirty = True

    def set_slice_index(self, axis, index):
        axis = str(axis).lower()
        if axis not in self.slices:
            raise ValueError("axis must be 'x', 'y', or 'z'")
        index = int(index)
        gpu_slice_unchanged = (
            self.representation != "slices"
            or (
                self.volume.slice_axis == axis
                and self.volume.slice_position
                == index / max(
                    self.field.data.shape[
                        {"x": 2, "y": 1, "z": 0}[axis]
                    ] - 1,
                    1,
                )
            )
        )
        if self.slices[axis].index == index and gpu_slice_unchanged:
            return False
        if self.slices[axis].index != index:
            self.slices[axis] = self.field.slice(
                axis, index, self.transfer_function, clipping=self.clipping,
            )
        if self.representation in {"slices", "combined"}:
            size = self.field.data.shape[{"x": 2, "y": 1, "z": 0}[axis]]
            self.scene.update_volume(
                self.volume, slice_axis=axis,
                slice_position=index / max(size - 1, 1),
                slice_positions=tuple(
                    index / max(size - 1, 1) if candidate == axis
                    else self.volume.slice_positions[position]
                    for position, candidate in enumerate("xyz")
                ),
            )
            self.slices_dirty = True
            return True
        self.refresh_slices(axis)
        return True

    def set_isovalue(self, value):
        value = float(value)
        if value == self.isosurface.value:
            return False
        if self.representation in {"combined", "isosurface"}:
            self.isosurface = ol.ScalarIsosurface(
                self.field, value, self.transfer_function, self.clipping,
                self.isosurface.index_vertices, self.isosurface.indices,
            )
            self.scene.update_volume(self.volume, isovalue=value)
            self.isosurface_dirty = True
        else:
            current = self.field.isosurface(
                value, self.transfer_function, clipping=self.clipping,
            )
            if not len(current.indices):
                raise ValueError("field has no triangles at the requested isovalue")
            self.isosurface = current
            self.refresh_isosurface()
        return True

    def set_representation(self, mode):
        """Select volume, slices, isosurface, or their combined presentation."""
        mode = str(mode).lower()
        if mode not in {"combined", "volume", "slices", "isosurface"}:
            raise ValueError("unsupported scientific representation")
        if mode == self.representation:
            return False
        self.scene.update_volume(
            self.volume, visible=True,
            render_mode=(
                "slice" if mode == "slices"
                else "isosurface" if mode == "isosurface"
                else "combined" if mode == "combined"
                else "volume"
            ),
            isovalue=self.isosurface.value,
            slice_positions=tuple(
                self.slices[axis].index / max(
                    self.field.data.shape[{"x": 2, "y": 1, "z": 0}[axis]] - 1,
                    1,
                ) for axis in "xyz"
            ),
        )
        for mesh in self.slice_meshes.values():
            self.scene.update_mesh(
                mesh, visible=False
            )
        self.scene.update_mesh(
            self.isosurface_mesh,
            visible=False,
        )
        self.representation = mode
        return True


@dataclass
class ScientificWorkbenchController:
    """Toolkit-neutral state behind the interactive scientific workbench."""

    showcase: ScientificScalarFieldShowcase
    series: ol.ScalarFieldSeries
    time_index: int = 0
    channel_index: int = 0
    playing: bool = False
    playback_fps: float = 6.0
    _last_advance: float | None = None

    def __post_init__(self):
        if not isinstance(self.showcase, ScientificScalarFieldShowcase):
            raise TypeError("showcase must be a ScientificScalarFieldShowcase")
        if not isinstance(self.series, ol.ScalarFieldSeries):
            raise TypeError("series must be a ScalarFieldSeries")
        self.playback_fps = float(self.playback_fps)
        if not np.isfinite(self.playback_fps) or self.playback_fps <= 0.0:
            raise ValueError("playback_fps must be positive and finite")

    @classmethod
    def from_array(cls, data, *, axis_order, **metadata):
        series = ol.ScalarFieldSeries.from_array(
            data, axis_order=axis_order, **metadata
        )
        showcase = _build_scientific_field_showcase(series.frame(0, 0))
        controller = cls(showcase, series)
        showcase.scene.scientific_controller = controller
        return controller

    @classmethod
    def from_numpy(cls, path, *, axis_order=None, mmap_mode="r", **metadata):
        """Memory-map a .npy dataset and infer common scientific axis orders."""
        path = Path(path)
        if path.suffix.lower() != ".npy":
            raise ValueError("scientific workbench loading currently requires .npy")
        data = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
        if axis_order is None:
            axis_order = {3: "zyx", 4: "tzyx", 5: "tczyx"}.get(data.ndim)
        if axis_order is None:
            raise ValueError("axis_order is required for arrays outside 3D–5D")
        metadata.setdefault("name", path.stem)
        metadata.setdefault("metadata", {"source": str(path.resolve())})
        return cls.from_array(data, axis_order=axis_order, **metadata)

    @property
    def scene(self):
        return self.showcase.scene

    @property
    def field(self):
        return self.showcase.field

    def select_frame(self, time_index=None, channel=None):
        previous = (self.time_index, self.channel_index)
        if time_index is not None:
            time_index = int(time_index)
            if not 0 <= time_index < len(self.series.times):
                raise IndexError("time index is outside the dataset")
            self.time_index = time_index
        if channel is not None:
            self.channel_index = self.series.channel_index(channel)
        if previous != (self.time_index, self.channel_index):
            self.showcase.replace_field(
                self.series.frame(self.time_index, self.channel_index)
            )
        return self.field

    def configure_transfer(
        self, *, colormap="viridis", mode="linear", value_range=None,
        opacity=(0.0, 0.18), reverse=False,
    ):
        transfer = ol.TransferFunction.from_colormap(
            colormap, opacity=opacity, reverse=reverse,
            mapping=ol.ScalarMapping(mode, value_range),
        )
        previous = self.showcase.transfer_function
        if (
            previous.mapping == transfer.mapping
            and previous.missing_rgba == transfer.missing_rgba
            and np.array_equal(previous.rgba, transfer.rgba)
        ):
            return previous
        self.showcase.set_transfer_function(transfer)
        return transfer

    def set_roi(self, minimum, maximum, *, space="index"):
        clipping = ol.ClipRegion(
            planes=self.showcase.clipping.planes,
            roi=ol.RegionOfInterest(minimum, maximum, space),
        )
        if clipping != self.showcase.clipping:
            self.showcase.set_clipping(clipping)
        return clipping

    def probe(self, camera, viewport_size, pixel, *, mapping=None):
        inspector = ol.ScientificInspector(
            self.field, self.showcase.transfer_function,
            clipping=self.showcase.clipping,
        )
        return inspector.probe(camera, viewport_size, pixel, mapping=mapping)

    def advance(self, now=None):
        """Advance looping playback when its stable wall-clock interval elapses."""
        if not self.playing or len(self.series.times) < 2:
            return False
        if not np.isfinite(self.playback_fps) or self.playback_fps <= 0.0:
            raise ValueError("playback_fps must be positive and finite")
        now = time.monotonic() if now is None else float(now)
        if not np.isfinite(now):
            raise ValueError("playback time must be finite")
        if self._last_advance is None:
            self._last_advance = now
            return False
        if now - self._last_advance < 1.0 / self.playback_fps:
            return False
        self._last_advance = now
        self.select_frame((self.time_index + 1) % len(self.series.times))
        return True


def _example_field(resolution):
    coordinates = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
    z, y, x = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    primary = np.exp(-3.8 * (x * x + 1.3 * y * y + 0.8 * z * z))
    secondary = 0.65 * np.exp(
        -14.0 * ((x + 0.42) ** 2 + (y - 0.22) ** 2 + (z + 0.1) ** 2)
    )
    ripple = 0.08 * np.sin(8.0 * x) * np.cos(6.0 * z) * primary
    return np.asarray(primary + secondary + ripple, np.float32)


def _build_scientific_field_showcase(field):
    resolution = min(field.data.shape)
    finite = np.asarray(field.data)[np.isfinite(field.data)]
    if not finite.size:
        raise ValueError("scientific field must contain at least one finite value")
    low, high = map(float, (np.min(finite), np.max(finite)))
    if low == high:
        high = float(np.nextafter(high, np.inf))
    transfer = ol.TransferFunction.from_colormap(
        "viridis", samples=256, opacity=(0.0, 0.18),
        mapping=ol.ScalarMapping("linear", (low, high)),
    )
    clipping = ol.ClipRegion(roi=ol.RegionOfInterest(
        (0, 0, 0), tuple(value - 1 for value in field.data.shape[::-1]),
    ))
    scene = ol.Scene(metadata={"showcase": "scientific-scalar-field"})
    volume = field.add_volume(
        scene, transfer, clipping=clipping, step_size=0.025,
        density_scale=0.7, emission_scale=0.65,
    )
    middle = {
        axis: field.data.shape[{"x": 2, "y": 1, "z": 0}[axis]] // 2
        for axis in "xyz"
    }
    slices = {
        axis: field.slice(axis, middle[axis], transfer, clipping=clipping)
        for axis in "xyz"
    }
    slice_meshes = {
        axis: scalar_slice.add_texture_to_scene(
            scene, name=f"scientific-{axis}-slice"
        )
        for axis, scalar_slice in slices.items()
    }
    isosurface = field.isosurface(low + 0.35 * (high - low), transfer, clipping=clipping)
    isosurface_mesh = isosurface.add_to_scene(scene, name="scientific-isosurface")
    scene.update_volume(
        volume, render_mode="combined",
        isovalue=isosurface.value,
        slice_positions=tuple(
            middle[axis] / max(
                field.data.shape[{"x": 2, "y": 1, "z": 0}[axis]] - 1, 1
            ) for axis in "xyz"
        ),
    )
    for mesh in slice_meshes.values():
        scene.update_mesh(mesh, visible=False)
    scene.update_mesh(isosurface_mesh, visible=False)
    return ScientificScalarFieldShowcase(
        scene, field, transfer, clipping, volume, slices, slice_meshes,
        isosurface, isosurface_mesh,
    )


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
    showcase = _build_scientific_field_showcase(field)
    series = ol.ScalarFieldSeries.from_array(
        data, axis_order="zyx", spacing=spacing,
        origin=(-1.0, -1.0, -1.0),
        channel_units=(field.unit,), name=field.name, metadata=field.metadata,
    )
    controller = ScientificWorkbenchController(showcase, series)
    showcase.scene.scientific_controller = controller
    return showcase


def build_scientific_scalar_field_scene(resolution=32):
    """Workbench-compatible scene-only form of the scientific showcase."""
    return build_scientific_scalar_field_showcase(resolution).scene
