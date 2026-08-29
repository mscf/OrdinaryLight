"""Backend-neutral world-space primitive batches lowered to mesh instances."""

from dataclasses import dataclass, field

import numpy as np

from ..scene import Material, MeshResource, Scene, Transform


def _positions(values, name):
    result = np.array(values, dtype=np.float32, copy=True, order="C")
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{name} must have shape (count, 3)")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    result.flags.writeable = False
    return result


def _positive_column(values, count, name):
    result = np.asarray(values, dtype=np.float32)
    if result.ndim == 0:
        result = np.full(count, result, np.float32)
    else:
        result = np.array(result, dtype=np.float32, copy=True).reshape(-1)
    if result.shape != (count,):
        raise ValueError(f"{name} must be scalar or have shape ({count},)")
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError(f"{name} must contain finite positive values")
    result.flags.writeable = False
    return result


def _sphere_geometry(rings, segments):
    rings, segments = int(rings), int(segments)
    if rings < 3 or segments < 3:
        raise ValueError("point sphere rings and segments must be at least 3")
    vertices = []
    normals = []
    for ring in range(rings + 1):
        theta = np.pi * ring / rings
        for segment in range(segments):
            phi = 2.0 * np.pi * segment / segments
            normal = (
                np.sin(theta) * np.cos(phi),
                np.cos(theta),
                np.sin(theta) * np.sin(phi),
            )
            vertices.append(normal)
            normals.append(normal)
    indices = []
    for ring in range(rings):
        for segment in range(segments):
            following = (segment + 1) % segments
            a = ring * segments + segment
            b = ring * segments + following
            c = (ring + 1) * segments + following
            d = (ring + 1) * segments + segment
            if ring:
                indices.append((a, b, d))
            if ring + 1 < rings:
                indices.append((b, c, d))
    return (
        np.asarray(vertices, np.float32),
        np.asarray(indices, np.uint32),
        np.asarray(normals, np.float32),
    )


def _cylinder_geometry(segments):
    segments = int(segments)
    if segments < 3:
        raise ValueError("line cylinder segments must be at least 3")
    vertices = []
    normals = []
    for y in (0.0, 1.0):
        for segment in range(segments):
            angle = 2.0 * np.pi * segment / segments
            radial = (np.cos(angle), 0.0, np.sin(angle))
            vertices.append((radial[0], y, radial[2]))
            normals.append(radial)
    side_count = len(vertices)
    for y, normal in ((0.0, (0.0, -1.0, 0.0)), (1.0, (0.0, 1.0, 0.0))):
        for segment in range(segments):
            angle = 2.0 * np.pi * segment / segments
            vertices.append((np.cos(angle), y, np.sin(angle)))
            normals.append(normal)
        vertices.append((0.0, y, 0.0))
        normals.append(normal)
    indices = []
    for segment in range(segments):
        following = (segment + 1) % segments
        indices.extend((
            (segment, following, segments + segment),
            (following, segments + following, segments + segment),
        ))
    bottom = side_count
    bottom_center = bottom + segments
    top = bottom_center + 1
    top_center = top + segments
    for segment in range(segments):
        following = (segment + 1) % segments
        indices.append((bottom_center, bottom + following, bottom + segment))
        indices.append((top_center, top + segment, top + following))
    return (
        np.asarray(vertices, np.float32),
        np.asarray(indices, np.uint32),
        np.asarray(normals, np.float32),
    )


def _point_transforms(positions, radii):
    matrices = np.repeat(np.eye(4, dtype=np.float32)[None], len(positions), axis=0)
    matrices[:, :3, :3] *= radii[:, None, None]
    matrices[:, :3, 3] = positions
    return matrices


def _line_transforms(starts, ends, radii):
    direction = ends - starts
    lengths = np.linalg.norm(direction, axis=1)
    if np.any(lengths <= 1e-8):
        raise ValueError("line segments cannot have coincident endpoints")
    y_axis = direction / lengths[:, None]
    helper = np.zeros_like(y_axis)
    helper[:, 1] = 1.0
    parallel = np.abs(y_axis[:, 1]) > 0.9
    helper[parallel] = (1.0, 0.0, 0.0)
    x_axis = np.cross(helper, y_axis)
    x_axis /= np.linalg.norm(x_axis, axis=1)[:, None]
    z_axis = np.cross(x_axis, y_axis)
    matrices = np.repeat(np.eye(4, dtype=np.float32)[None], len(starts), axis=0)
    matrices[:, :3, 0] = x_axis * radii[:, None]
    matrices[:, :3, 1] = direction
    matrices[:, :3, 2] = z_axis * radii[:, None]
    matrices[:, :3, 3] = starts
    return matrices


@dataclass
class GlyphBatch:
    """A stable batch of instances sharing arbitrary caller-owned geometry."""

    scene: Scene = field(repr=False)
    resource: MeshResource
    instances: tuple
    owns_resource: bool = False

    @property
    def ids(self):
        return tuple(instance.id for instance in self.instances)

    def update(self, *, transforms=None, materials=None, visible=None):
        self.scene.update_instance_batch(
            self.instances, transforms=transforms,
            materials=materials, visible=visible,
        )
        return self

    def remove(self):
        if self.instances:
            self.scene.remove_instances(self.instances)
            self.instances = ()
        if self.owns_resource:
            self.scene.remove_mesh_resource(self.resource)
            self.owns_resource = False


@dataclass
class PointBatch(GlyphBatch):
    """World-space spherical points with independently adjustable radii."""

    positions: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    radii: np.ndarray = field(default_factory=lambda: np.empty(0))

    def update(self, *, positions=None, radii=None, materials=None, visible=None):
        next_positions = self.positions if positions is None else _positions(
            positions, "positions"
        )
        if len(next_positions) != len(self.instances):
            raise ValueError("point count cannot change during an update")
        next_radii = self.radii if radii is None else _positive_column(
            radii, len(self.instances), "radii"
        )
        self.scene.update_instance_batch(
            self.instances,
            transforms=_point_transforms(next_positions, next_radii),
            materials=materials, visible=visible,
        )
        self.positions, self.radii = next_positions, next_radii
        return self


@dataclass
class LineBatch(GlyphBatch):
    """World-space capped line segments lowered to shared cylinder geometry."""

    starts: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    ends: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    radii: np.ndarray = field(default_factory=lambda: np.empty(0))

    def update(
        self, *, starts=None, ends=None, radii=None,
        materials=None, visible=None,
    ):
        next_starts = self.starts if starts is None else _positions(starts, "starts")
        next_ends = self.ends if ends is None else _positions(ends, "ends")
        if len(next_starts) != len(self.instances) or len(next_ends) != len(self.instances):
            raise ValueError("line count cannot change during an update")
        next_radii = self.radii if radii is None else _positive_column(
            radii, len(self.instances), "radii"
        )
        self.scene.update_instance_batch(
            self.instances,
            transforms=_line_transforms(next_starts, next_ends, next_radii),
            materials=materials, visible=visible,
        )
        self.starts, self.ends, self.radii = next_starts, next_ends, next_radii
        return self


def add_glyphs(
    scene, resource, transforms, *, materials=None, visible=True,
    names=None, metadata=None,
):
    """Place arbitrary shared mesh glyphs through the generic instance path."""
    if not isinstance(scene, Scene):
        raise TypeError("scene must be a Scene")
    instances = scene.add_instances(
        resource, transforms, materials=materials, visible=visible,
        names=names, metadata=metadata,
    )
    return GlyphBatch(scene, scene.get_mesh_resource(resource), instances)


def add_points(
    scene, positions, *, radii=0.05, material=None, materials=None,
    visible=True, names=None, metadata=None, rings=8, segments=12,
):
    """Add finite world-space points represented by one shared sphere mesh."""
    if material is not None and materials is not None:
        raise ValueError("pass material or materials, not both")
    positions = _positions(positions, "positions")
    radii = _positive_column(radii, len(positions), "radii")
    vertices, indices, normals = _sphere_geometry(rings, segments)
    resource = scene.create_mesh(
        vertices, indices, material or Material(), normals=normals,
        name="point-sphere", metadata={"primitive": "points"},
    )
    instances = scene.add_instances(
        resource, _point_transforms(positions, radii),
        materials=materials, visible=visible, names=names, metadata=metadata,
    )
    return PointBatch(
        scene, resource, instances, True, positions=positions, radii=radii
    )


def add_lines(
    scene, starts, ends=None, *, radii=0.025, material=None, materials=None,
    visible=True, names=None, metadata=None, segments=12,
):
    """Add finite world-space line segments using one shared capped cylinder."""
    if material is not None and materials is not None:
        raise ValueError("pass material or materials, not both")
    if ends is None:
        pairs = np.asarray(starts, dtype=np.float32)
        if pairs.ndim != 3 or pairs.shape[1:] != (2, 3):
            raise ValueError("segments must have shape (count, 2, 3)")
        starts, ends = pairs[:, 0], pairs[:, 1]
    starts = _positions(starts, "starts")
    ends = _positions(ends, "ends")
    if len(starts) != len(ends):
        raise ValueError("starts and ends must contain the same number of points")
    radii = _positive_column(radii, len(starts), "radii")
    vertices, indices, normals = _cylinder_geometry(segments)
    resource = scene.create_mesh(
        vertices, indices, material or Material(), normals=normals,
        name="line-cylinder", metadata={"primitive": "lines"},
    )
    instances = scene.add_instances(
        resource, _line_transforms(starts, ends, radii),
        materials=materials, visible=visible, names=names, metadata=metadata,
    )
    return LineBatch(
        scene, resource, instances, True,
        starts=starts, ends=ends, radii=radii,
    )
