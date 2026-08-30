"""Backend-neutral scene data structures."""

from dataclasses import dataclass, field
import re
from types import MappingProxyType

import numpy as np

from ..animations import (
    AnimationClip, MorphTarget, Skin, compose_matrix, decompose_matrix,
)
from ..cameras import PerspectiveCamera
from ..lights import (
    DIRECTIONAL, LIGHT_TYPES, POINT, SPOT, DirectionalLight,
    EnvironmentLight, PointLight, SpotLight,
)
from ..materials import MATERIAL_PARAMETER_LAYOUT


_UNCHANGED = object()
_ATTRIBUTE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BUILTIN_VERTEX_ATTRIBUTES = (
    "position", "normal", "texcoord0", "texcoord1", "tangent",
)


def _vec3(value, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values")
    return result


def _snapshot_value(value):
    """Convert common Python/NumPy metadata into JSON-compatible values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "items"):
        return {str(key): _snapshot_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_snapshot_value(item) for item in value]
    return repr(value)


@dataclass(frozen=True)
class Transform:
    """Immutable affine object-to-world transform."""

    matrix: np.ndarray = field(
        default_factory=lambda: np.eye(4, dtype=np.float32),
        compare=False, repr=False,
    )

    def __post_init__(self):
        matrix = np.array(self.matrix, dtype=np.float32, copy=True)
        if matrix.shape != (4, 4):
            raise ValueError("transform matrix must have shape (4, 4)")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("transform matrix must contain finite values")
        if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
            raise ValueError("transform matrix must be affine")
        if abs(float(np.linalg.det(matrix[:3, :3]))) < 1e-10:
            raise ValueError("transform linear component must be invertible")
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)

    @classmethod
    def translation(cls, offset):
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, 3] = _vec3(offset, "translation")
        return cls(matrix)

    @classmethod
    def scale(cls, factors):
        values = np.asarray(factors, dtype=np.float32)
        if values.ndim == 0:
            values = np.repeat(values, 3)
        values = _vec3(values, "scale")
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = np.diag(values)
        return cls(matrix)

    @classmethod
    def rotation(cls, axis, angle_radians):
        axis = _vec3(axis, "rotation axis")
        length = float(np.linalg.norm(axis))
        if length < 1e-8:
            raise ValueError("rotation axis cannot be zero")
        axis = axis / length
        x, y, z = axis
        cosine = float(np.cos(angle_radians))
        sine = float(np.sin(angle_radians))
        one_minus = 1.0 - cosine
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = (
            (cosine + x*x*one_minus, x*y*one_minus - z*sine,
             x*z*one_minus + y*sine),
            (y*x*one_minus + z*sine, cosine + y*y*one_minus,
             y*z*one_minus - x*sine),
            (z*x*one_minus - y*sine, z*y*one_minus + x*sine,
             cosine + z*z*one_minus),
        )
        return cls(matrix)

    def __matmul__(self, other):
        if not isinstance(other, Transform):
            return NotImplemented
        return Transform(self.matrix @ other.matrix)


@dataclass(frozen=True)
class Texture:
    """Decoded RGBA texture and its glTF-compatible sampler state."""

    pixels: np.ndarray = field(compare=False, repr=False)
    wrap_s: str = "repeat"
    wrap_t: str = "repeat"
    linear_filter: bool = True

    def __post_init__(self):
        pixels = np.ascontiguousarray(self.pixels, dtype=np.uint8)
        if pixels.ndim != 3 or pixels.shape[2] != 4:
            raise ValueError("texture pixels must have shape (height, width, 4)")
        if pixels.shape[0] < 1 or pixels.shape[1] < 1:
            raise ValueError("texture dimensions must be positive")
        if self.wrap_s not in {"repeat", "clamp", "mirror"}:
            raise ValueError("wrap_s must be repeat, clamp, or mirror")
        if self.wrap_t not in {"repeat", "clamp", "mirror"}:
            raise ValueError("wrap_t must be repeat, clamp, or mirror")
        object.__setattr__(self, "pixels", pixels)


@dataclass(frozen=True)
class Texture1D:
    """Immutable linear RGBA lookup texture with explicit sampler state."""

    values: np.ndarray = field(compare=False, repr=False)
    address_mode: str = "clamp"
    linear_filter: bool = True

    def __post_init__(self):
        values = np.array(self.values, dtype=np.float32, copy=True, order="C")
        if values.ndim != 2 or values.shape[1] not in (3, 4):
            raise ValueError("1D texture values must have shape (length, 3 or 4)")
        if len(values) < 1:
            raise ValueError("1D texture length must be positive")
        if not np.all(np.isfinite(values)):
            raise ValueError("1D texture values must be finite")
        if values.shape[1] == 3:
            values = np.column_stack((values, np.ones(len(values), np.float32)))
        if self.address_mode not in {"clamp", "repeat", "mirror"}:
            raise ValueError("address_mode must be clamp, repeat, or mirror")
        if not isinstance(self.linear_filter, bool):
            raise TypeError("linear_filter must be a bool")
        values = np.ascontiguousarray(values, dtype=np.float32)
        values.flags.writeable = False
        object.__setattr__(self, "values", values)

    def sample(self, coordinates):
        """Sample normalized coordinates and return linear float32 RGBA."""
        coordinates = np.asarray(coordinates, dtype=np.float32)
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("texture coordinates must be finite")
        if self.address_mode == "clamp":
            addressed = np.clip(coordinates, 0.0, 1.0)
        elif self.address_mode == "repeat":
            addressed = coordinates - np.floor(coordinates)
        else:
            period = np.mod(coordinates, 2.0)
            addressed = np.where(period <= 1.0, period, 2.0 - period)
        position = addressed * max(len(self.values) - 1, 0)
        if not self.linear_filter:
            index = np.floor(position + 0.5).astype(np.int64)
            return np.asarray(self.values[index], dtype=np.float32)
        lower = np.floor(position).astype(np.int64)
        upper = np.minimum(lower + 1, len(self.values) - 1)
        weight = (position - lower)[..., None]
        return np.asarray(
            self.values[lower] * (1.0 - weight) + self.values[upper] * weight,
            dtype=np.float32,
        )


@dataclass(frozen=True)
class VolumeMaterial:
    """Emission, absorption, and optional single scattering for a volume.

    Transfer-function RGB is emitted radiance and alpha is reference opacity
    per ``step_size`` world units.  ``scattering_scale`` enables point-light
    single scattering using ``scattering_color`` and the selected normalized
    phase function.  Zero scattering preserves emission--absorption behavior.
    """

    transfer_function: Texture1D = field(default_factory=lambda: Texture1D(
        np.asarray(((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0)), np.float32)
    ))
    density_scale: float = 1.0
    emission_scale: float = 1.0
    step_size: float = 0.01
    scattering_scale: float = 0.0
    scattering_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    phase_function: str = "isotropic"
    anisotropy: float = 0.0
    scattering_albedo: tuple[float, float, float] = (0.9, 0.9, 0.9)
    scattering_orders: int = 1

    def __post_init__(self):
        if not isinstance(self.transfer_function, Texture1D):
            raise TypeError("transfer_function must be a Texture1D")
        for name in (
            "density_scale", "emission_scale", "step_size",
            "scattering_scale", "anisotropy",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name == "step_size" and value <= 0.0:
                raise ValueError("step_size must be positive")
            if name in ("density_scale", "emission_scale", "scattering_scale") and value < 0.0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, value)
        color = _vec3(self.scattering_color, "scattering_color")
        if np.any(color < 0.0):
            raise ValueError("scattering_color components cannot be negative")
        object.__setattr__(self, "scattering_color", tuple(map(float, color)))
        albedo = _vec3(self.scattering_albedo, "scattering_albedo")
        if np.any(albedo < 0.0) or np.any(albedo > 1.0):
            raise ValueError("scattering_albedo components must be in [0, 1]")
        object.__setattr__(self, "scattering_albedo", tuple(map(float, albedo)))
        if self.phase_function not in {"isotropic", "henyey_greenstein"}:
            raise ValueError(
                "phase_function must be 'isotropic' or 'henyey_greenstein'"
            )
        if not -0.99 <= self.anisotropy <= 0.99:
            raise ValueError("anisotropy must be between -0.99 and 0.99")
        if isinstance(self.scattering_orders, bool):
            raise TypeError("scattering_orders must be an integer")
        orders = int(self.scattering_orders)
        if orders != self.scattering_orders or not 1 <= orders <= 8:
            raise ValueError("scattering_orders must be between 1 and 8")
        object.__setattr__(self, "scattering_orders", orders)


@dataclass
class Volume:
    """A placed dense scalar field in the local unit cube ``[0, 1]^3``."""

    data: np.ndarray = field(repr=False)
    material: VolumeMaterial = field(default_factory=VolumeMaterial)
    transform: Transform = field(default_factory=Transform)
    value_range: tuple[float, float] | None = None
    visible: bool = True
    name: str | None = None
    metadata: dict = field(default_factory=dict, repr=False)
    id: int | None = field(default=None, init=False)

    def __post_init__(self):
        data = np.array(self.data, dtype=np.float32, copy=True, order="C")
        if data.ndim != 3 or any(size < 2 for size in data.shape):
            raise ValueError("volume data must have shape (depth, height, width) with dimensions >= 2")
        if not np.all(np.isfinite(data)):
            raise ValueError("volume data must contain finite values")
        if not isinstance(self.material, VolumeMaterial):
            raise TypeError("material must be a VolumeMaterial")
        if not isinstance(self.transform, Transform):
            self.transform = Transform(self.transform)
        if self.value_range is None:
            value_range = (float(data.min()), float(data.max()))
            if value_range[0] == value_range[1]:
                value = value_range[0]
                value_range = (min(0.0, value), max(1.0, value))
        else:
            if len(self.value_range) != 2:
                raise ValueError("value_range must contain two values")
            value_range = tuple(float(value) for value in self.value_range)
        if not np.all(np.isfinite(value_range)) or value_range[1] <= value_range[0]:
            raise ValueError("value_range must be a finite increasing pair")
        if not isinstance(self.visible, bool):
            raise TypeError("visible must be a bool")
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError("name must be a string or None")
        if not hasattr(self.metadata, "items"):
            raise TypeError("metadata must be a mapping")
        data.flags.writeable = False
        self.data = data
        self.value_range = value_range
        self.metadata = dict(self.metadata)

    @property
    def shape(self):
        return self.data.shape

    @property
    def normalized_data(self):
        low, high = self.value_range
        return np.ascontiguousarray(np.clip((self.data - low) / (high - low), 0.0, 1.0))


@dataclass(frozen=True)
class TextureTransform:
    """KHR_texture_transform-compatible affine UV mapping."""

    offset: tuple[float, float] = (0.0, 0.0)
    scale: tuple[float, float] = (1.0, 1.0)
    rotation: float = 0.0
    texcoord_set: int = 0

    def __post_init__(self):
        if len(self.offset) != 2 or len(self.scale) != 2:
            raise ValueError("texture transform offset and scale must be vec2 values")
        offset = tuple(float(value) for value in self.offset)
        scale = tuple(float(value) for value in self.scale)
        rotation = float(self.rotation)
        texcoord_set = int(self.texcoord_set)
        if texcoord_set not in (0, 1):
            raise ValueError("texcoord_set must be zero or one")
        if not np.all(np.isfinite((*offset, *scale, rotation))):
            raise ValueError("texture transform values must be finite")
        if abs(scale[0]) < 1e-8 or abs(scale[1]) < 1e-8:
            raise ValueError("texture transform scale components cannot be zero")
        object.__setattr__(self, "offset", offset)
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "texcoord_set", texcoord_set)


@dataclass(frozen=True)
class Material:
    base_color: tuple[float, float, float] = (0.8, 0.8, 0.8)
    emission: tuple[float, float, float] = (0.0, 0.0, 0.0)
    metallic: float = 0.0
    roughness: float = 1.0
    transmission: float = 0.0
    clearcoat: float = 0.0
    clearcoat_roughness: float = 0.1
    sheen_color: tuple[float, float, float] = (0.0, 0.0, 0.0)
    sheen_roughness: float = 0.5
    anisotropy: float = 0.0
    thin_walled: bool = False
    subsurface: float = 0.0
    subsurface_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    subsurface_radius: float = 0.5
    ior: float = 1.5
    attenuation_color: tuple[float, float, float] = (1.0, 1.0, 1.0)
    attenuation_distance: float = float("inf")
    emission_two_sided: bool = False
    base_color_texture: Texture | None = field(default=None, compare=False)
    metallic_roughness_texture: Texture | None = field(default=None, compare=False)
    emissive_texture: Texture | None = field(default=None, compare=False)
    normal_texture: Texture | None = field(default=None, compare=False)
    normal_scale: float = 1.0
    occlusion_texture: Texture | None = field(default=None, compare=False)
    occlusion_strength: float = 1.0
    transmission_texture: Texture | None = field(default=None, compare=False)
    clearcoat_texture: Texture | None = field(default=None, compare=False)
    sheen_texture: Texture | None = field(default=None, compare=False)
    anisotropy_texture: Texture | None = field(default=None, compare=False)
    subsurface_texture: Texture | None = field(default=None, compare=False)
    base_color_transform: TextureTransform = field(default_factory=TextureTransform)
    metallic_roughness_transform: TextureTransform = field(default_factory=TextureTransform)
    emissive_transform: TextureTransform = field(default_factory=TextureTransform)
    normal_transform: TextureTransform = field(default_factory=TextureTransform)
    occlusion_transform: TextureTransform = field(default_factory=TextureTransform)
    transmission_transform: TextureTransform = field(default_factory=TextureTransform)
    program: object | None = field(default=None, compare=False)

    def __post_init__(self):
        color = _vec3(self.base_color, "base_color")
        emission = _vec3(self.emission, "emission")
        attenuation = _vec3(self.attenuation_color, "attenuation_color")
        sheen = _vec3(self.sheen_color, "sheen_color")
        subsurface_color = _vec3(self.subsurface_color, "subsurface_color")
        if np.any(color < 0.0) or np.any(color > 1.0):
            raise ValueError("base_color components must be between zero and one")
        if np.any(emission < 0.0):
            raise ValueError("emission components cannot be negative")
        if not 0.0 <= self.metallic <= 1.0:
            raise ValueError("metallic must be between zero and one")
        if not 0.0 <= self.roughness <= 1.0:
            raise ValueError("roughness must be between zero and one")
        if not 0.0 <= self.transmission <= 1.0:
            raise ValueError("transmission must be between zero and one")
        for name in (
            "clearcoat", "clearcoat_roughness", "sheen_roughness",
            "subsurface", "subsurface_radius",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if not -1.0 <= self.anisotropy <= 1.0:
            raise ValueError("anisotropy must be between minus one and one")
        if np.any(sheen < 0.0) or np.any(sheen > 1.0):
            raise ValueError("sheen_color components must be between zero and one")
        if np.any(subsurface_color < 0.0) or np.any(subsurface_color > 1.0):
            raise ValueError(
                "subsurface_color components must be between zero and one"
            )
        if not isinstance(self.thin_walled, bool):
            raise TypeError("thin_walled must be a bool")
        if self.ior <= 0.0:
            raise ValueError("ior must be positive")
        if np.any(attenuation < 0.0) or np.any(attenuation > 1.0):
            raise ValueError("attenuation_color components must be between zero and one")
        if self.attenuation_distance <= 0.0:
            raise ValueError("attenuation_distance must be positive")
        if not isinstance(self.emission_two_sided, bool):
            raise TypeError("emission_two_sided must be a bool")
        for name in (
            "base_color_texture", "metallic_roughness_texture", "emissive_texture",
            "normal_texture",
            "occlusion_texture",
            "transmission_texture",
            "clearcoat_texture", "sheen_texture", "anisotropy_texture",
            "subsurface_texture",
        ):
            texture = getattr(self, name)
            if texture is not None and not isinstance(texture, Texture):
                raise TypeError(f"{name} must be a Texture")
        if self.normal_scale < 0.0:
            raise ValueError("normal_scale cannot be negative")
        if not 0.0 <= self.occlusion_strength <= 1.0:
            raise ValueError("occlusion_strength must be between zero and one")
        for name in (
            "base_color_transform", "metallic_roughness_transform",
            "emissive_transform", "normal_transform",
            "occlusion_transform",
            "transmission_transform",
        ):
            if not isinstance(getattr(self, name), TextureTransform):
                raise TypeError(f"{name} must be a TextureTransform")
        if self.program is not None:
            from ..materials import MaterialProgram
            if not isinstance(self.program, MaterialProgram):
                raise TypeError("program must be created by @material")


@dataclass
class Mesh:
    vertices: np.ndarray
    indices: np.ndarray
    material: Material = field(default_factory=Material)
    normals: np.ndarray | None = None
    texcoords: np.ndarray | None = None
    texcoords1: np.ndarray | None = None
    tangents: np.ndarray | None = None
    transform: Transform = field(default_factory=Transform)
    deformable: bool = False
    attributes: dict = field(default_factory=dict, repr=False)
    resource: "MeshResource | None" = field(default=None, repr=False)
    visible: bool = True
    name: str | None = None
    metadata: dict = field(default_factory=dict, repr=False)
    id: int | None = field(default=None, init=False)

    def __post_init__(self):
        if self.resource is not None and not isinstance(self.resource, MeshResource):
            raise TypeError("resource must be a MeshResource")
        if not isinstance(self.visible, bool):
            raise TypeError("visible must be a bool")
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError("name must be a string or None")
        if not hasattr(self.metadata, "items"):
            raise TypeError("metadata must be a mapping")
        self.metadata = MappingProxyType(dict(self.metadata))
        if not isinstance(self.deformable, bool):
            raise TypeError("deformable must be a bool")
        if not isinstance(self.transform, Transform):
            self.transform = Transform(self.transform)
        if self.resource is not None:
            # A placed instance is intentionally lightweight: its reusable
            # object-space arrays are authoritative on MeshResource and must
            # not be normalized, validated, or copied for every placement.
            for name in (
                "vertices", "indices", "normals", "texcoords", "texcoords1",
                "tangents", "deformable", "attributes",
            ):
                setattr(self, name, getattr(self.resource, name))
            return
        self.vertices = np.ascontiguousarray(self.vertices, dtype=np.float32)
        self.indices = np.ascontiguousarray(self.indices, dtype=np.uint32)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (vertex_count, 3)")
        if self.indices.ndim != 2 or self.indices.shape[1] != 3:
            raise ValueError("indices must have shape (triangle_count, 3)")
        if self.indices.size and int(self.indices.max()) >= len(self.vertices):
            raise ValueError("an index refers to a vertex that does not exist")
        if self.normals is None:
            self.normals = self._generate_normals()
        else:
            self.normals = np.ascontiguousarray(self.normals, dtype=np.float32)
            if self.normals.shape != self.vertices.shape:
                raise ValueError("normals must have the same shape as vertices")
            lengths = np.linalg.norm(self.normals, axis=1)
            if np.any(lengths < 1e-8):
                raise ValueError("normals cannot contain zero vectors")
            self.normals /= lengths[:, None]
        if self.texcoords is None:
            self.texcoords = np.zeros((len(self.vertices), 2), dtype=np.float32)
        else:
            self.texcoords = np.ascontiguousarray(self.texcoords, dtype=np.float32)
            if self.texcoords.shape != (len(self.vertices), 2):
                raise ValueError("texcoords must have shape (vertex_count, 2)")
        if self.texcoords1 is None:
            self.texcoords1 = np.zeros((len(self.vertices), 2), dtype=np.float32)
        else:
            self.texcoords1 = np.ascontiguousarray(self.texcoords1, dtype=np.float32)
            if self.texcoords1.shape != (len(self.vertices), 2):
                raise ValueError("texcoords1 must have shape (vertex_count, 2)")
        if self.tangents is None:
            self.tangents = self._generate_tangents()
        else:
            self.tangents = np.ascontiguousarray(self.tangents, dtype=np.float32)
            if self.tangents.shape != (len(self.vertices), 4):
                raise ValueError("tangents must have shape (vertex_count, 4)")
            lengths = np.linalg.norm(self.tangents[:, :3], axis=1)
            if np.any(lengths < 1e-8):
                raise ValueError("tangent directions cannot contain zero vectors")
            self.tangents[:, :3] /= lengths[:, None]
            self.tangents[:, 3] = np.where(self.tangents[:, 3] < 0.0, -1.0, 1.0)
        if not hasattr(self.attributes, "items"):
            raise TypeError("attributes must be a mapping")
        attributes = {}
        for name, values in self.attributes.items():
            if not isinstance(name, str) or not _ATTRIBUTE_NAME.fullmatch(name):
                raise ValueError(
                    "attribute names must be valid ASCII-style identifiers"
                )
            if name in _BUILTIN_VERTEX_ATTRIBUTES:
                raise ValueError(f"{name!r} is a built-in vertex attribute")
            values = np.array(values, dtype=np.float32, copy=True, order="C")
            if values.ndim == 1:
                values = values[:, None]
            if (values.ndim != 2 or values.shape[0] != len(self.vertices)
                    or not 1 <= values.shape[1] <= 4):
                raise ValueError(
                    f"attribute {name!r} must have shape "
                    "(vertex_count,) or (vertex_count, 1..4)"
                )
            if not np.all(np.isfinite(values)):
                raise ValueError(f"attribute {name!r} must contain finite values")
            values.flags.writeable = False
            attributes[name] = values
        self.attributes = MappingProxyType(attributes)

    @property
    def vertex_attribute_names(self):
        """Built-in and user-declared interpolated vertex channel names."""
        return (*_BUILTIN_VERTEX_ATTRIBUTES, *self.attributes)

    def vertex_attribute(self, name):
        """Return one object-space vertex channel by semantic name."""
        builtin = {
            "position": self.vertices,
            "normal": self.normals,
            "texcoord0": self.texcoords,
            "texcoord1": self.texcoords1,
            "tangent": self.tangents,
        }
        if name in builtin:
            return builtin[name]
        try:
            return self.attributes[name]
        except KeyError as error:
            raise KeyError(f"mesh has no vertex attribute {name!r}") from error

    @property
    def world_vertices(self):
        """Object vertices transformed into world space."""
        linear = self.transform.matrix[:3, :3]
        translation = self.transform.matrix[:3, 3]
        return np.ascontiguousarray(
            self.vertices @ linear.T + translation, dtype=np.float32,
        )

    @property
    def world_normals(self):
        """Normals transformed by the inverse transpose."""
        normal_matrix = np.linalg.inv(self.transform.matrix[:3, :3]).T
        result = self.normals @ normal_matrix.T
        result /= np.linalg.norm(result, axis=1, keepdims=True)
        return np.ascontiguousarray(result, dtype=np.float32)

    @property
    def world_tangents(self):
        """Tangent frame transformed into world space."""
        linear = self.transform.matrix[:3, :3]
        directions = self.tangents[:, :3] @ linear.T
        normals = self.world_normals
        directions -= normals * np.sum(directions * normals, axis=1)[:, None]
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        handedness = self.tangents[:, 3].copy()
        if np.linalg.det(linear) < 0.0:
            handedness *= -1.0
        return np.ascontiguousarray(
            np.column_stack((directions, handedness)), dtype=np.float32,
        )

    def _generate_normals(self):
        normals = np.zeros_like(self.vertices)
        triangles = self.vertices[self.indices]
        face_normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        for corner in range(3):
            np.add.at(normals, self.indices[:, corner], face_normals)
        # Attribute-less procedural meshes commonly duplicate seam and pole
        # vertices. Weld identical positions for normal generation only; the
        # original topology and UV seams remain untouched.
        _positions, groups = np.unique(
            self.vertices, axis=0, return_inverse=True
        )
        group_normals = np.zeros((_positions.shape[0], 3), dtype=np.float32)
        np.add.at(group_normals, groups, normals)
        normals = group_normals[groups]
        lengths = np.linalg.norm(normals, axis=1)
        missing = lengths < 1e-8
        normals[~missing] /= lengths[~missing, None]
        normals[missing] = (0.0, 1.0, 0.0)
        return np.ascontiguousarray(normals, dtype=np.float32)

    def _generate_tangents(self):
        tangent_sum = np.zeros_like(self.vertices)
        bitangent_sum = np.zeros_like(self.vertices)
        for triangle in self.indices:
            positions = self.vertices[triangle]
            uvs = self.texcoords[triangle]
            edge_a = positions[1] - positions[0]
            edge_b = positions[2] - positions[0]
            uv_a = uvs[1] - uvs[0]
            uv_b = uvs[2] - uvs[0]
            determinant = uv_a[0] * uv_b[1] - uv_a[1] * uv_b[0]
            if abs(float(determinant)) < 1e-10:
                continue
            inverse = 1.0 / determinant
            tangent = (edge_a * uv_b[1] - edge_b * uv_a[1]) * inverse
            bitangent = (edge_b * uv_a[0] - edge_a * uv_b[0]) * inverse
            for vertex in triangle:
                tangent_sum[vertex] += tangent
                bitangent_sum[vertex] += bitangent
        result = np.empty((len(self.vertices), 4), dtype=np.float32)
        for index, normal in enumerate(self.normals):
            tangent = tangent_sum[index] - normal * np.dot(normal, tangent_sum[index])
            length = np.linalg.norm(tangent)
            if length < 1e-8:
                axis = (0.0, 0.0, 1.0) if abs(float(normal[2])) < 0.999 else (0.0, 1.0, 0.0)
                tangent = np.cross(normal, axis)
                length = np.linalg.norm(tangent)
            tangent /= max(float(length), 1e-8)
            handedness = -1.0 if np.dot(
                np.cross(normal, tangent), bitangent_sum[index]
            ) < 0.0 else 1.0
            result[index] = (*tangent, handedness)
        return np.ascontiguousarray(result)


@dataclass
class MeshResource:
    """Reusable object-space mesh data owned by one :class:`Scene`.

    Instances may independently override the default material and transform,
    while Vulkan backends build only one BLAS for this shared geometry.
    """

    vertices: np.ndarray
    indices: np.ndarray
    material: Material = field(default_factory=Material)
    normals: np.ndarray | None = None
    texcoords: np.ndarray | None = None
    texcoords1: np.ndarray | None = None
    tangents: np.ndarray | None = None
    deformable: bool = False
    attributes: dict = field(default_factory=dict, repr=False)
    name: str | None = None
    metadata: dict = field(default_factory=dict, repr=False)
    id: int | None = field(default=None, init=False)

    def __post_init__(self):
        validated = Mesh(
            self.vertices, self.indices, self.material,
            normals=self.normals, texcoords=self.texcoords,
            texcoords1=self.texcoords1, tangents=self.tangents,
            deformable=self.deformable, attributes=self.attributes,
            name=self.name, metadata=self.metadata,
        )
        for name in (
            "vertices", "indices", "material", "normals", "texcoords",
            "texcoords1", "tangents", "deformable", "attributes", "name",
            "metadata",
        ):
            setattr(self, name, getattr(validated, name))


class Instance(Mesh):
    """A placed occurrence of a reusable :class:`MeshResource`."""


@dataclass(frozen=True)
class VertexAttributeLayout:
    """Selected custom-channel ABI for an opt-in GPU attribute buffer."""

    channels: tuple[tuple[str, int], ...]

    def __post_init__(self):
        channels = tuple((name, int(components)) for name, components in self.channels)
        names = set()
        for name, components in channels:
            if not isinstance(name, str) or not _ATTRIBUTE_NAME.fullmatch(name):
                raise ValueError("attribute layout names must be valid identifiers")
            if name in _BUILTIN_VERTEX_ATTRIBUTES:
                raise ValueError("built-in attributes use the fixed vertex ABI")
            if name in names:
                raise ValueError(f"duplicate attribute layout channel {name!r}")
            if not 1 <= components <= 4:
                raise ValueError("attribute component counts must be between 1 and 4")
            names.add(name)
        object.__setattr__(self, "channels", channels)

    @classmethod
    def from_scene(cls, scene, names):
        """Derive and validate a channel layout across every scene mesh."""
        names = tuple(names)
        if len(set(names)) != len(names):
            raise ValueError("attribute names cannot contain duplicates")
        channels = []
        for name in names:
            component_count = None
            for mesh in scene.visible_meshes:
                try:
                    values = mesh.attributes[name]
                except KeyError as error:
                    raise ValueError(
                        f"mesh {mesh.id} does not provide attribute {name!r}"
                    ) from error
                if component_count is None:
                    component_count = values.shape[1]
                elif values.shape[1] != component_count:
                    raise ValueError(
                        f"attribute {name!r} has inconsistent component counts"
                    )
            if component_count is None:
                raise ValueError(f"scene has no meshes for attribute {name!r}")
            channels.append((name, component_count))
        return cls(tuple(channels))

    def pack(self, scene):
        """Pack channels as ``[triangle corner, channel, vec4]`` records."""
        corner_count = sum(mesh.indices.size for mesh in scene.visible_meshes)
        packed = np.zeros((corner_count, len(self.channels), 4), np.float32)
        offset = 0
        for mesh in scene.visible_meshes:
            count = mesh.indices.size
            for slot, (name, components) in enumerate(self.channels):
                try:
                    values = mesh.attributes[name]
                except KeyError as error:
                    raise ValueError(
                        f"mesh {mesh.id} does not provide attribute {name!r}"
                    ) from error
                if values.shape[1] != components:
                    raise ValueError(
                        f"attribute {name!r} no longer matches its layout"
                    )
                packed[offset:offset + count, slot, :components] = (
                    values[mesh.indices].reshape((-1, components))
                )
            offset += count
        return np.ascontiguousarray(packed)

    def slot(self, name):
        """Return the stable vec4 slot assigned to a channel."""
        for index, (candidate, _components) in enumerate(self.channels):
            if candidate == name:
                return index
        raise KeyError(name)


_VOLUME_BOX_VERTICES = np.asarray((
    (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
    (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
), np.float32)
_VOLUME_BOX_INDICES = np.asarray((
    (0, 2, 1), (0, 3, 2),  # z = 0
    (4, 5, 6), (4, 6, 7),  # z = 1
    (0, 1, 5), (0, 5, 4),  # y = 0
    (3, 7, 6), (3, 6, 2),  # y = 1
    (0, 4, 7), (0, 7, 3),  # x = 0
    (1, 2, 6), (1, 6, 5),  # x = 1
), np.uint32)


@dataclass
class Node:
    """Hierarchical transform grouping zero or more placed scene instances."""

    transform: Transform = field(default_factory=Transform)
    name: str | None = None
    metadata: dict = field(default_factory=dict)
    id: int | None = field(default=None, init=False)
    parent: "Node | None" = field(default=None, init=False, repr=False)
    children: list["Node"] = field(default_factory=list, init=False, repr=False)
    instances: list[Instance] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        if not isinstance(self.transform, Transform):
            self.transform = Transform(self.transform)
        if self.name is not None and not isinstance(self.name, str):
            raise TypeError("node name must be a string or None")
        if not hasattr(self.metadata, "items"):
            raise TypeError("node metadata must be a mapping")
        self.metadata = dict(self.metadata)

    @property
    def world_matrix(self):
        if self.parent is None:
            return self.transform.matrix
        return self.parent.world_matrix @ self.transform.matrix

    @property
    def world_transform(self):
        return Transform(self.world_matrix)


@dataclass
class Scene:
    meshes: list[Mesh] = field(default_factory=list)
    lights: list[PointLight | DirectionalLight | SpotLight] = field(
        default_factory=list
    )
    mesh_resources: list[MeshResource] = field(default_factory=list)
    volumes: list[Volume] = field(default_factory=list)
    environment: EnvironmentLight | None = None
    animations: list[AnimationClip] = field(default_factory=list)
    nodes: list[Node] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    _revision: int = field(default=0, init=False, repr=False)
    _geometry_revision: int = field(default=0, init=False, repr=False)
    _shading_revision: int = field(default=0, init=False, repr=False)
    _transform_revision: int = field(default=0, init=False, repr=False)
    _next_resource_id: int = field(default=1, init=False, repr=False)
    _next_mesh_resource_id: int = field(default=1, init=False, repr=False)
    _next_material_id: int = field(default=1, init=False, repr=False)
    _next_node_id: int = field(default=1, init=False, repr=False)
    _material_ids: dict = field(default_factory=dict, init=False, repr=False)
    _volume_proxy_material: Material = field(
        default_factory=lambda: Material(
            base_color=(0.0, 0.0, 0.0), transmission=1.0, ior=1.0,
            roughness=0.0,
        ), init=False, repr=False,
    )
    _environment_texture: Texture | None = field(
        default=None, init=False, repr=False,
    )
    _animation_bases: dict = field(default_factory=dict, init=False, repr=False)
    _morph_bindings: dict = field(default_factory=dict, init=False, repr=False)
    _skin_bindings: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self.meshes = list(self.meshes)
        self.lights = list(self.lights)
        self.mesh_resources = list(self.mesh_resources)
        self.volumes = list(self.volumes)
        self.animations = list(self.animations)
        self.nodes = list(self.nodes)
        if not hasattr(self.metadata, "items"):
            raise TypeError("scene metadata must be a mapping")
        self.metadata = dict(self.metadata)
        if any(not isinstance(clip, AnimationClip) for clip in self.animations):
            raise TypeError("animations must contain AnimationClip objects")
        if self.nodes:
            raise ValueError("construct scene nodes with Scene.add_node()")
        if self.environment is not None:
            if not isinstance(self.environment, EnvironmentLight):
                raise TypeError("environment must be an EnvironmentLight or None")
            self._environment_texture = self._pack_environment_texture(
                self.environment
            )[0]
        for resource in self.mesh_resources:
            if not isinstance(resource, MeshResource):
                raise TypeError("mesh_resources must contain MeshResource objects")
            self._attach_mesh_resource(resource)
        for resource in (*self.meshes, *self.lights):
            if not isinstance(resource, (Mesh, *LIGHT_TYPES)):
                raise TypeError("scene resources must be mesh or light objects")
            self._attach(resource)
            if isinstance(resource, Mesh):
                if resource.resource is not None:
                    if not any(
                        resource.resource is candidate
                        for candidate in self.mesh_resources
                    ):
                        raise ValueError("mesh instance resource is not owned by scene")
                self._register_material(resource.material)
        for volume in self.volumes:
            if not isinstance(volume, Volume):
                raise TypeError("volumes must contain Volume objects")
            self._attach(volume)
            self._register_material(self._volume_proxy_material)

    @property
    def instances(self):
        """Placed scene objects in stable insertion order."""
        return tuple(self.meshes)

    @property
    def visible_meshes(self):
        """Visible placed objects used by packing and acceleration structures."""
        return tuple(mesh for mesh in self.meshes if mesh.visible)

    @property
    def render_meshes(self):
        """Triangle objects used by hardware traversal, including volume bounds."""
        proxies = []
        for volume_index, volume in enumerate(self.visible_volumes):
            proxy = Mesh(
                _VOLUME_BOX_VERTICES, _VOLUME_BOX_INDICES,
                self._volume_proxy_material, transform=volume.transform,
                name=f"{volume.name or 'volume'}-bounds",
                metadata={"volume_index": volume_index},
            )
            proxy.id = volume.id
            proxies.append(proxy)
        return self.visible_meshes + tuple(proxies)

    @property
    def instance_count(self):
        """Number of placed objects, including currently hidden objects."""
        return len(self.meshes)

    @property
    def visible_instance_count(self):
        """Number of objects represented by the active TLAS."""
        return sum(mesh.visible and bool(len(mesh.indices)) for mesh in self.meshes)

    @property
    def visible_geometry_count(self):
        """Number of distinct object-space geometries used by visible objects."""
        return len({
            id(mesh.resource or mesh)
            for mesh in self.visible_meshes if len(mesh.indices)
        })

    @property
    def visible_volumes(self):
        """Visible dense fields in stable insertion order."""
        return tuple(volume for volume in self.volumes if volume.visible)

    def get_volume(self, reference):
        """Resolve a volume object or stable ID to its current handle."""
        return self._resolve(self.volumes, reference, "volume")

    def instancing_statistics(self):
        """Return backend-neutral expansion and geometry-sharing measurements."""
        visible = tuple(mesh for mesh in self.visible_meshes if len(mesh.indices))
        unique = []
        seen = set()
        for mesh in visible:
            geometry = mesh.resource or mesh
            if id(geometry) not in seen:
                seen.add(id(geometry))
                unique.append(geometry)

        def storage_bytes(mesh):
            arrays = (
                mesh.vertices, mesh.indices, mesh.normals, mesh.texcoords,
                mesh.texcoords1, mesh.tangents, *mesh.attributes.values(),
            )
            return sum(np.asarray(array).nbytes for array in arrays)

        return {
            "instance_count": len(visible),
            "geometry_count": len(unique),
            "shared_blas_savings": max(0, len(visible) - len(unique)),
            "expanded_triangle_count": sum(len(mesh.indices) for mesh in visible),
            "unique_triangle_count": sum(len(mesh.indices) for mesh in unique),
            "expanded_geometry_bytes": sum(storage_bytes(mesh) for mesh in visible),
            "unique_geometry_bytes": sum(storage_bytes(mesh) for mesh in unique),
        }

    def get_instance(self, reference):
        """Resolve an instance object or stable ID to its current handle."""
        return self._resolve(self.meshes, reference, "instance")

    def get_mesh_resource(self, reference):
        """Resolve reusable geometry by object or stable resource ID."""
        return self._resolve(
            self.mesh_resources, reference, "mesh resource"
        )

    def snapshot(self):
        """Return a JSON-compatible description without copying geometry arrays."""
        materials = sorted(
            self._material_ids.values(), key=lambda entry: entry[1]
        )
        return {
            "metadata": _snapshot_value(self.metadata),
            "revisions": {
                "scene": self.revision,
                "geometry": self.geometry_revision,
                "shading": self.shading_revision,
                "transform": self.transform_revision,
            },
            "instancing": self.instancing_statistics(),
            "mesh_resources": [
                {
                    "id": resource.id,
                    "name": resource.name,
                    "vertex_count": len(resource.vertices),
                    "triangle_count": len(resource.indices),
                    "deformable": resource.deformable,
                    "attribute_names": list(resource.attributes),
                    "metadata": _snapshot_value(resource.metadata),
                }
                for resource in self.mesh_resources
            ],
            "instances": [
                {
                    "id": instance.id,
                    "name": instance.name,
                    "mesh_resource_id": (
                        instance.resource.id if instance.resource is not None else None
                    ),
                    "material_id": self.material_id(instance.material),
                    "visible": instance.visible,
                    "transform": instance.transform.matrix.tolist(),
                    "metadata": _snapshot_value(instance.metadata),
                }
                for instance in self.meshes
            ],
            "materials": [
                {
                    "id": material_id,
                    "base_color": list(material.base_color),
                    "emission": list(material.emission),
                    "metallic": material.metallic,
                    "roughness": material.roughness,
                    "transmission": material.transmission,
                    "ior": material.ior,
                }
                for material, material_id in materials
            ],
            "point_lights": [
                {
                    "id": light.id,
                    "position": list(light.position),
                    "color": list(light.color),
                    "intensity": light.intensity,
                    "range": light.range,
                }
                for light in self.lights if isinstance(light, PointLight)
            ],
            "directional_lights": [
                {
                    "id": light.id,
                    "direction": list(light.direction),
                    "color": list(light.color),
                    "intensity": light.intensity,
                }
                for light in self.lights if isinstance(light, DirectionalLight)
            ],
            "spot_lights": [
                {
                    "id": light.id,
                    "position": list(light.position),
                    "direction": list(light.direction),
                    "color": list(light.color),
                    "intensity": light.intensity,
                    "inner_cone_angle": light.inner_cone_angle,
                    "outer_cone_angle": light.outer_cone_angle,
                    "range": light.range,
                }
                for light in self.lights if isinstance(light, SpotLight)
            ],
            "environment": None if self.environment is None else {
                "color": list(self.environment.color),
                "intensity": self.environment.intensity,
                "rotation": self.environment.rotation,
                "image_shape": (
                    None if self.environment.image is None
                    else list(self.environment.image.shape)
                ),
            },
            "animations": [
                {
                    "name": clip.name,
                    "start_time": clip.start_time,
                    "end_time": clip.end_time,
                    "duration": clip.duration,
                    "track_count": len(clip.tracks),
                }
                for clip in self.animations
            ],
            "nodes": [
                {
                    "id": node.id,
                    "name": node.name,
                    "parent_id": node.parent.id if node.parent is not None else None,
                    "child_ids": [child.id for child in node.children],
                    "instance_ids": [instance.id for instance in node.instances],
                    "transform": node.transform.matrix.tolist(),
                    "metadata": _snapshot_value(node.metadata),
                }
                for node in self.nodes
            ],
            "volumes": [
                {
                    "id": volume.id,
                    "name": volume.name,
                    "shape": list(volume.shape),
                    "value_range": list(volume.value_range),
                    "visible": volume.visible,
                    "transform": volume.transform.matrix.tolist(),
                    "material": {
                        "density_scale": volume.material.density_scale,
                        "emission_scale": volume.material.emission_scale,
                        "step_size": volume.material.step_size,
                        "scattering_scale": volume.material.scattering_scale,
                        "scattering_color": list(
                            volume.material.scattering_color
                        ),
                        "phase_function": volume.material.phase_function,
                        "anisotropy": volume.material.anisotropy,
                        "scattering_albedo": list(
                            volume.material.scattering_albedo
                        ),
                        "scattering_orders": volume.material.scattering_orders,
                        "transfer_function_length": len(
                            volume.material.transfer_function.values
                        ),
                    },
                    "metadata": _snapshot_value(volume.metadata),
                }
                for volume in self.volumes
            ],
        }

    @property
    def revision(self):
        """Monotonic revision of any renderer-visible scene state."""
        return self._revision

    @property
    def geometry_revision(self):
        """Revision of triangle positions, indices, or resource membership."""
        return self._geometry_revision

    @property
    def shading_revision(self):
        """Revision of materials, attributes, lights, or membership."""
        return self._shading_revision

    @property
    def transform_revision(self):
        """Revision of object-to-world transforms."""
        return self._transform_revision

    def _attach(self, resource):
        if resource.id is not None:
            raise ValueError("a scene resource cannot belong to multiple scenes")
        resource.id = self._next_resource_id
        self._next_resource_id += 1

    def _attach_mesh_resource(self, resource):
        if resource.id is not None:
            raise ValueError("a mesh resource cannot belong to multiple scenes")
        resource.id = self._next_mesh_resource_id
        self._next_mesh_resource_id += 1

    def _register_material(self, material):
        key = id(material)
        existing = self._material_ids.get(key)
        if existing is not None and existing[0] is material:
            return existing[1]
        material_id = self._next_material_id
        self._next_material_id += 1
        # Retain the object so Python cannot recycle its identity while this
        # scene promises stable material IDs.
        self._material_ids[key] = (material, material_id)
        return material_id

    def material_id(self, material):
        """Return a stable scene-local ID for an attached mesh material."""
        existing = self._material_ids.get(id(material))
        if existing is None or existing[0] is not material:
            raise KeyError("material is not attached to this scene")
        return existing[1]

    def _changed(self, *, geometry=False, shading=False, transform=False):
        self._revision += 1
        if geometry:
            self._geometry_revision += 1
        if shading:
            self._shading_revision += 1
        if transform:
            self._transform_revision += 1

    @staticmethod
    def _resolve(resources, resource, kind):
        for candidate in resources:
            if candidate is resource or candidate.id == resource:
                return candidate
        raise KeyError(f"{kind} is not part of this scene")

    def add_mesh(
        self, vertices, indices, material=None, *, normals=None, texcoords=None,
        texcoords1=None, tangents=None, attributes=None, transform=None,
        deformable=False, name=None, metadata=None,
    ):
        resource = self.create_mesh(
            vertices, indices, material,
            normals=normals, texcoords=texcoords, texcoords1=texcoords1,
            tangents=tangents, attributes={} if attributes is None else attributes,
            deformable=deformable, name=name, metadata=metadata,
        )
        return self.add_instance(
            resource, transform=transform, name=name, metadata=metadata
        )

    def add_volume(
        self, data, material=None, *, transform=None, value_range=None,
        visible=True, name=None, metadata=None,
    ):
        """Add a dense scalar field occupying its transformed local unit cube."""
        volume = Volume(
            data=data,
            material=VolumeMaterial() if material is None else material,
            transform=Transform() if transform is None else transform,
            value_range=value_range,
            visible=visible,
            name=name,
            metadata={} if metadata is None else metadata,
        )
        self._attach(volume)
        self._register_material(self._volume_proxy_material)
        self.volumes.append(volume)
        self._changed(geometry=True, shading=True, transform=True)
        return volume

    def update_volume(
        self, volume, *, data=_UNCHANGED, material=_UNCHANGED,
        transform=_UNCHANGED, value_range=_UNCHANGED, visible=_UNCHANGED,
    ):
        """Atomically replace volume data or placement while preserving its ID."""
        volume = self._resolve(self.volumes, volume, "volume")
        candidate = Volume(
            volume.data if data is _UNCHANGED else data,
            volume.material if material is _UNCHANGED else material,
            transform=(
                volume.transform if transform is _UNCHANGED else transform
            ),
            value_range=(
                (None if data is not _UNCHANGED else volume.value_range)
                if value_range is _UNCHANGED else value_range
            ),
            visible=volume.visible if visible is _UNCHANGED else visible,
            name=volume.name,
            metadata=volume.metadata,
        )
        data_changed = (
            data is not _UNCHANGED or value_range is not _UNCHANGED
        )
        material_changed = material is not _UNCHANGED
        transform_changed = transform is not _UNCHANGED and not np.array_equal(
            candidate.transform.matrix, volume.transform.matrix
        )
        visibility_changed = candidate.visible != volume.visible
        if not (data_changed or material_changed or transform_changed
                or visibility_changed):
            return volume
        for attribute in (
            "data", "material", "transform", "value_range", "visible",
        ):
            setattr(volume, attribute, getattr(candidate, attribute))
        self._changed(
            geometry=data_changed or visibility_changed,
            shading=data_changed or material_changed or visibility_changed,
            transform=transform_changed,
        )
        return volume

    def remove_volume(self, volume):
        """Remove a dense field by object or stable ID and return it."""
        volume = self._resolve(self.volumes, volume, "volume")
        self.volumes.remove(volume)
        self._changed(geometry=True, shading=True, transform=True)
        return volume

    def create_mesh(
        self, vertices, indices, material=None, *, normals=None, texcoords=None,
        texcoords1=None, tangents=None, attributes=None, deformable=False,
        name=None, metadata=None,
    ):
        """Create reusable object-space geometry without placing it."""
        resource = MeshResource(
            vertices, indices, material or Material(), normals=normals,
            texcoords=texcoords, texcoords1=texcoords1, tangents=tangents,
            attributes={} if attributes is None else attributes,
            deformable=deformable,
            name=name, metadata={} if metadata is None else metadata,
        )
        self._attach_mesh_resource(resource)
        self.mesh_resources.append(resource)
        return resource

    create_mesh_resource = create_mesh

    def update_mesh_resource(
        self, resource, *, vertices=_UNCHANGED, indices=_UNCHANGED,
        material=_UNCHANGED, normals=_UNCHANGED, texcoords=_UNCHANGED,
        texcoords1=_UNCHANGED, tangents=_UNCHANGED, attributes=_UNCHANGED,
    ):
        """Update shared geometry and synchronize every attached instance."""
        resource = self._resolve(
            self.mesh_resources, resource, "mesh resource"
        )
        requested = (
            vertices, indices, material, normals, texcoords, texcoords1,
            tangents, attributes,
        )
        if all(value is _UNCHANGED for value in requested):
            return resource
        old_default_material = resource.material
        geometry_changed = vertices is not _UNCHANGED or indices is not _UNCHANGED
        if normals is _UNCHANGED:
            normals = None if geometry_changed else resource.normals
        if tangents is _UNCHANGED:
            tangents = None if geometry_changed else resource.tangents
        candidate = MeshResource(
            resource.vertices if vertices is _UNCHANGED else vertices,
            resource.indices if indices is _UNCHANGED else indices,
            resource.material if material is _UNCHANGED else material,
            normals=normals,
            texcoords=resource.texcoords if texcoords is _UNCHANGED else texcoords,
            texcoords1=(
                resource.texcoords1 if texcoords1 is _UNCHANGED else texcoords1
            ),
            tangents=tangents,
            attributes=(
                resource.attributes if attributes is _UNCHANGED else attributes
            ),
            deformable=resource.deformable,
        )
        shading_changed = any(
            value is not _UNCHANGED
            for value in (material, normals, texcoords, texcoords1, tangents, attributes)
        ) or geometry_changed
        for name in (
            "vertices", "indices", "material", "normals", "texcoords",
            "texcoords1", "tangents", "attributes",
        ):
            setattr(resource, name, getattr(candidate, name))
        for instance in self.meshes:
            if instance.resource is not resource:
                continue
            for name in (
                "vertices", "indices", "normals", "texcoords",
                "texcoords1", "tangents", "attributes",
            ):
                setattr(instance, name, getattr(candidate, name))
            if (material is not _UNCHANGED
                    and instance.material is old_default_material):
                # Explicit per-instance overrides remain independent.
                instance.material = candidate.material
        self._register_material(candidate.material)
        self._changed(geometry=geometry_changed, shading=shading_changed)
        return resource

    def add_instance(
        self, resource, *, transform=None, material=None, visible=True,
        name=None, metadata=None, node=None,
    ):
        """Place one independently shaded occurrence of reusable geometry."""
        resource = self._resolve(self.mesh_resources, resource, "mesh resource")
        if node is not None:
            node = self.get_node(node)
            if transform is not None:
                raise TypeError(
                    "node-owned instances derive their transform from the node"
                )
            transform = node.world_transform
        instance = Instance(
            resource.vertices, resource.indices,
            resource.material if material is None else material,
            normals=resource.normals, texcoords=resource.texcoords,
            texcoords1=resource.texcoords1, tangents=resource.tangents,
            attributes=resource.attributes,
            transform=Transform() if transform is None else transform,
            deformable=resource.deformable, resource=resource,
            visible=visible, name=name,
            metadata={} if metadata is None else metadata,
        )
        self._attach(instance)
        self._register_material(instance.material)
        self.meshes.append(instance)
        if node is not None:
            node.instances.append(instance)
        self._changed(geometry=True, shading=True)
        return instance

    @staticmethod
    def _transform_batch(values, count=None):
        if isinstance(values, Transform):
            return (values,) if count is None else (values,) * count
        array = np.asarray(values) if isinstance(values, np.ndarray) else None
        if array is not None and array.shape == (4, 4):
            transform = Transform(array)
            return (transform,) if count is None else (transform,) * count
        if array is not None and array.ndim == 3 and array.shape[1:] == (4, 4):
            result = tuple(Transform(matrix) for matrix in array)
        else:
            result = tuple(
                value if isinstance(value, Transform) else Transform(value)
                for value in values
            )
        if count is not None and len(result) != count:
            raise ValueError(f"expected {count} transforms, got {len(result)}")
        return result

    @staticmethod
    def _batch_column(value, count, name, scalar_type, default):
        if value is None:
            return tuple(default(index) for index in range(count))
        if isinstance(value, scalar_type):
            return (value,) * count
        result = tuple(value)
        if len(result) != count:
            raise ValueError(f"expected {count} {name}, got {len(result)}")
        if any(not isinstance(item, scalar_type) for item in result):
            raise TypeError(f"{name} must contain {scalar_type.__name__} values")
        return result

    def add_instances(
        self, resource, transforms, *, materials=None, visible=True,
        names=None, metadata=None,
    ):
        """Create many placements atomically from column-oriented inputs."""
        resource = self._resolve(self.mesh_resources, resource, "mesh resource")
        transforms = self._transform_batch(transforms)
        count = len(transforms)
        materials = self._batch_column(
            materials, count, "materials", Material,
            lambda _index: resource.material,
        )
        visibility = self._batch_column(
            visible, count, "visibility values", bool, lambda _index: True
        )
        if names is None or isinstance(names, str):
            names = (names,) * count
        else:
            names = tuple(names)
            if len(names) != count:
                raise ValueError(f"expected {count} names, got {len(names)}")
            if any(name is not None and not isinstance(name, str) for name in names):
                raise TypeError("names must contain strings or None")
        if metadata is None or hasattr(metadata, "items"):
            metadata = ({} if metadata is None else metadata,) * count
        else:
            metadata = tuple(metadata)
            if len(metadata) != count:
                raise ValueError(
                    f"expected {count} metadata mappings, got {len(metadata)}"
                )
        candidates = tuple(
            Instance(
                resource.vertices, resource.indices, material,
                normals=resource.normals, texcoords=resource.texcoords,
                texcoords1=resource.texcoords1, tangents=resource.tangents,
                attributes=resource.attributes, transform=transform,
                deformable=resource.deformable, resource=resource,
                visible=is_visible, name=name, metadata=item_metadata,
            )
            for transform, material, is_visible, name, item_metadata
            in zip(transforms, materials, visibility, names, metadata)
        )
        if not candidates:
            return ()
        for instance in candidates:
            self._attach(instance)
            self._register_material(instance.material)
            self.meshes.append(instance)
        self._changed(geometry=True, shading=True)
        return candidates

    def update_instance_batch(
        self, instances, *, transforms=None, materials=None, visible=None,
    ):
        """Update placement columns atomically without per-instance mappings."""
        if isinstance(instances, (Mesh, int, np.integer)):
            instances = (instances,)
        resolved = tuple(
            self._resolve(self.meshes, reference, "instance")
            for reference in instances
        )
        if len({id(instance) for instance in resolved}) != len(resolved):
            raise ValueError("instances cannot contain duplicates")
        count = len(resolved)
        next_transforms = (
            tuple(instance.transform for instance in resolved)
            if transforms is None else self._transform_batch(transforms, count)
        )
        next_materials = self._batch_column(
            materials, count, "materials", Material,
            lambda index: resolved[index].material,
        )
        next_visibility = self._batch_column(
            visible, count, "visibility values", bool,
            lambda index: resolved[index].visible,
        )
        any_transform = any(
            not np.array_equal(instance.transform.matrix, transform.matrix)
            for instance, transform in zip(resolved, next_transforms)
        )
        any_material = any(
            instance.material is not material
            for instance, material in zip(resolved, next_materials)
        )
        any_visibility = any(
            instance.visible != is_visible
            for instance, is_visible in zip(resolved, next_visibility)
        )
        if not (any_transform or any_material or any_visibility):
            return resolved
        for instance, transform, material, is_visible in zip(
            resolved, next_transforms, next_materials, next_visibility
        ):
            self._register_material(material)
            instance.transform = transform
            instance.material = material
            instance.visible = is_visible
        self._changed(
            geometry=any_visibility,
            shading=any_material or any_visibility,
            transform=any_transform,
        )
        return resolved

    def update_instance_transforms(self, instances, transforms):
        """Convenience column update for animation and simulation loops."""
        return self.update_instance_batch(instances, transforms=transforms)

    def update_instance_materials(self, instances, materials):
        """Convenience column update for per-instance appearance changes."""
        return self.update_instance_batch(instances, materials=materials)

    def update_instance(
        self, instance, *, transform=_UNCHANGED, material=_UNCHANGED,
        visible=_UNCHANGED,
    ):
        """Atomically update placement state while preserving instance ID."""
        instance = self._resolve(self.meshes, instance, "instance")
        candidate_transform = (
            instance.transform if transform is _UNCHANGED
            else transform if isinstance(transform, Transform) else Transform(transform)
        )
        candidate_material = instance.material if material is _UNCHANGED else material
        if not isinstance(candidate_material, Material):
            raise TypeError("material must be a Material")
        candidate_visible = instance.visible if visible is _UNCHANGED else visible
        if not isinstance(candidate_visible, bool):
            raise TypeError("visible must be a bool")
        transform_changed = not np.array_equal(
            candidate_transform.matrix, instance.transform.matrix
        )
        material_changed = candidate_material is not instance.material
        visibility_changed = candidate_visible != instance.visible
        if not (transform_changed or material_changed or visibility_changed):
            return instance
        self._register_material(candidate_material)
        instance.transform = candidate_transform
        instance.material = candidate_material
        instance.visible = candidate_visible
        self._changed(
            geometry=visibility_changed,
            shading=material_changed or visibility_changed,
            transform=transform_changed,
        )
        return instance

    def update_instances(self, updates):
        """Validate and apply many instance updates with one revision change.

        ``updates`` is a mapping or iterable of ``(instance, changes)`` pairs;
        each changes value is a mapping containing any of ``transform``,
        ``material``, and ``visible``.
        """
        pairs = list(updates.items() if hasattr(updates, "items") else updates)
        validated = []
        for reference, changes in pairs:
            instance = self._resolve(self.meshes, reference, "instance")
            if not hasattr(changes, "items"):
                raise TypeError("instance changes must be mappings")
            unknown = set(changes) - {"transform", "material", "visible"}
            if unknown:
                raise ValueError(f"unknown instance changes: {tuple(sorted(unknown))}")
            transform = changes.get("transform", instance.transform)
            if not isinstance(transform, Transform):
                transform = Transform(transform)
            material = changes.get("material", instance.material)
            if not isinstance(material, Material):
                raise TypeError("material must be a Material")
            visible = changes.get("visible", instance.visible)
            if not isinstance(visible, bool):
                raise TypeError("visible must be a bool")
            validated.append((instance, transform, material, visible))
        any_transform = any(
            not np.array_equal(item.transform.matrix, transform.matrix)
            for item, transform, _, _ in validated
        )
        any_material = any(item.material is not material for item, _, material, _ in validated)
        any_visibility = any(item.visible != visible for item, _, _, visible in validated)
        if not (any_transform or any_material or any_visibility):
            return tuple(item for item, *_ in validated)
        for instance, transform, material, visible in validated:
            self._register_material(material)
            instance.transform = transform
            instance.material = material
            instance.visible = visible
        self._changed(
            geometry=any_visibility,
            shading=any_material or any_visibility,
            transform=any_transform,
        )
        return tuple(item for item, *_ in validated)

    def update_mesh(
        self, mesh, *, vertices=_UNCHANGED, indices=_UNCHANGED,
        material=_UNCHANGED, normals=_UNCHANGED, texcoords=_UNCHANGED,
        texcoords1=_UNCHANGED, tangents=_UNCHANGED, attributes=_UNCHANGED,
        transform=_UNCHANGED,
    ):
        """Validate and update a mesh while preserving its object and ID.

        Passing ``None`` for an optional vertex attribute regenerates its
        default. When positions or topology change, omitted normals and
        tangents are regenerated to avoid retaining stale derived data.
        """
        mesh = self._resolve(self.meshes, mesh, "mesh")
        requested = (vertices, indices, material, normals, texcoords, texcoords1,
                     tangents, attributes, transform)
        if all(value is _UNCHANGED for value in requested):
            return mesh
        geometry_changed = vertices is not _UNCHANGED or indices is not _UNCHANGED
        if normals is _UNCHANGED:
            normals = None if geometry_changed else mesh.normals
        if tangents is _UNCHANGED:
            tangents = None if geometry_changed else mesh.tangents
        candidate = Mesh(
            mesh.vertices if vertices is _UNCHANGED else vertices,
            mesh.indices if indices is _UNCHANGED else indices,
            mesh.material if material is _UNCHANGED else material,
            normals=normals,
            texcoords=mesh.texcoords if texcoords is _UNCHANGED else texcoords,
            texcoords1=(
                mesh.texcoords1 if texcoords1 is _UNCHANGED else texcoords1
            ),
            tangents=tangents,
            attributes=(
                mesh.attributes if attributes is _UNCHANGED else attributes
            ),
            transform=mesh.transform if transform is _UNCHANGED else transform,
            deformable=mesh.deformable,
            resource=None, visible=mesh.visible,
            name=mesh.name, metadata=mesh.metadata,
        )
        if geometry_changed and mesh.resource is not None:
            users = [item for item in self.meshes if item.resource is mesh.resource]
            if len(users) > 1:
                replacement = MeshResource(
                    candidate.vertices, candidate.indices, mesh.resource.material,
                    normals=candidate.normals, texcoords=candidate.texcoords,
                    texcoords1=candidate.texcoords1, tangents=candidate.tangents,
                    attributes=candidate.attributes,
                    deformable=mesh.resource.deformable,
                )
                self._attach_mesh_resource(replacement)
                self.mesh_resources.append(replacement)
                mesh.resource = replacement
            else:
                for name in (
                    "vertices", "indices", "normals", "texcoords",
                    "texcoords1", "tangents", "attributes",
                ):
                    setattr(mesh.resource, name, getattr(candidate, name))
        self._register_material(candidate.material)
        for name in (
            "vertices", "indices", "material", "normals", "texcoords",
            "texcoords1", "tangents", "attributes",
        ):
            setattr(mesh, name, getattr(candidate, name))
        mesh.transform = candidate.transform
        shading_changed = any(value is not _UNCHANGED for value in requested[2:8]) \
            or geometry_changed
        self._changed(
            geometry=geometry_changed, shading=shading_changed,
            transform=transform is not _UNCHANGED,
        )
        return mesh

    def triangle_vertex_attribute_data(self, name):
        """Pack one named channel in global triangle-corner order."""
        records = []
        component_count = None
        for mesh in self.visible_meshes:
            values = mesh.vertex_attribute(name)
            if component_count is None:
                component_count = values.shape[1]
            elif values.shape[1] != component_count:
                raise ValueError(
                    f"attribute {name!r} has inconsistent component counts"
                )
            records.append(values[mesh.indices].reshape((-1, component_count)))
        if component_count is None:
            raise KeyError(f"scene has no vertex attribute {name!r}")
        return np.ascontiguousarray(np.concatenate(records), dtype=np.float32)

    def remove_mesh(self, mesh):
        """Remove a mesh by object or stable integer ID and return it."""
        mesh = self._resolve(self.meshes, mesh, "mesh")
        self.meshes.remove(mesh)
        self._changed(geometry=True, shading=True)
        return mesh

    remove_instance = remove_mesh

    def remove_instances(self, instances):
        """Atomically remove many placements while retaining monotonic IDs."""
        resolved = tuple(
            self._resolve(self.meshes, reference, "instance")
            for reference in instances
        )
        if len({id(instance) for instance in resolved}) != len(resolved):
            raise ValueError("instances cannot contain duplicates")
        if not resolved:
            return ()
        removed = {id(instance) for instance in resolved}
        self.meshes[:] = [
            instance for instance in self.meshes if id(instance) not in removed
        ]
        self._changed(geometry=True, shading=True)
        return resolved

    def add_points(self, positions, **options):
        """Add a batch of finite world-space point primitives."""
        from ..primitives import add_points
        return add_points(self, positions, **options)

    def add_lines(self, starts, ends=None, **options):
        """Add a batch of finite world-space line-segment primitives."""
        from ..primitives import add_lines
        return add_lines(self, starts, ends, **options)

    def add_glyphs(self, resource, transforms, **options):
        """Add a batch of arbitrary shared mesh glyphs."""
        from ..primitives import add_glyphs
        return add_glyphs(self, resource, transforms, **options)

    def remove_mesh_resource(self, resource):
        """Remove unused reusable geometry without recycling its stable ID."""
        resource = self._resolve(
            self.mesh_resources, resource, "mesh resource"
        )
        if any(mesh.resource is resource for mesh in self.meshes):
            raise ValueError("remove all instances before removing a mesh resource")
        self.mesh_resources.remove(resource)
        return resource

    def triangle_attribute_data(self):
        """Pack interpolated vertex inputs as normal and UV vec4 records."""
        records = []
        for mesh in self.render_meshes:
            indices = mesh.indices
            triangles = mesh.world_vertices[indices]
            triangle_uvs = mesh.texcoords[indices]
            uv_edges = triangle_uvs[:, 1:] - triangle_uvs[:, :1]
            uv_area = np.abs(
                uv_edges[:, 0, 0] * uv_edges[:, 1, 1]
                - uv_edges[:, 0, 1] * uv_edges[:, 1, 0]
            )
            world_edges = triangles[:, 1:] - triangles[:, :1]
            world_area = np.linalg.norm(
                np.cross(world_edges[:, 0], world_edges[:, 1]), axis=1,
            )
            uv0_density = np.sqrt(uv_area / np.maximum(world_area, 1e-8))

            packed = np.empty((len(indices), 3, 3, 4), dtype=np.float32)
            packed[:, :, 0, :3] = mesh.world_normals[indices]
            packed[:, :, 0, 3] = uv0_density[:, None]
            packed[:, :, 1, :2] = triangle_uvs
            packed[:, :, 1, 2:] = mesh.texcoords1[indices]
            packed[:, :, 2] = mesh.world_tangents[indices]
            records.append(packed.reshape((-1, 3, 4)))
        if not records:
            return np.empty((0, 3, 4), dtype=np.float32)
        return np.ascontiguousarray(np.concatenate(records), dtype=np.float32)

    def add_light(self, light):
        """Attach an analytic light resource and return its stable handle."""
        if not isinstance(light, LIGHT_TYPES):
            raise TypeError("light must be a PointLight, DirectionalLight, or SpotLight")
        self._attach(light)
        self.lights.append(light)
        self._changed(shading=True)
        return light

    def add_animation(self, clip):
        """Attach a reusable animation clip to this scene."""
        if not isinstance(clip, AnimationClip):
            raise TypeError("clip must be an AnimationClip")
        self.animations.append(clip)
        return clip

    def add_node(self, *, transform=None, parent=None, name=None, metadata=None):
        """Create a hierarchical local transform node."""
        if parent is not None:
            parent = self.get_node(parent)
        node = Node(
            Transform() if transform is None else transform,
            name=name, metadata={} if metadata is None else metadata,
        )
        node.id = self._next_node_id
        self._next_node_id += 1
        node.parent = parent
        if parent is not None:
            parent.children.append(node)
        self.nodes.append(node)
        return node

    def get_node(self, reference):
        """Resolve a scene node object or stable node ID."""
        return self._resolve(self.nodes, reference, "node")

    def update_node(self, node, *, transform=_UNCHANGED):
        """Update a node's local transform and synchronize its subtree."""
        node = self.get_node(node)
        if transform is _UNCHANGED:
            return node
        transform = transform if isinstance(transform, Transform) else Transform(transform)
        if np.array_equal(transform.matrix, node.transform.matrix):
            return node
        node.transform = transform
        self._sync_node_instances(node)
        return node

    def _sync_node_instances(self, root=None):
        roots = ([node for node in self.nodes if node.parent is None]
                 if root is None else [root])
        instances = []
        transforms = []

        def visit(node):
            world = node.world_transform
            for instance in node.instances:
                instances.append(instance)
                transforms.append(world)
            for child in node.children:
                visit(child)

        for node in roots:
            visit(node)
        if instances:
            self.update_instance_transforms(instances, transforms)
        return tuple(instances)

    def apply_animation(self, clip, time, *, loop=False):
        """Sample a clip and atomically update its instance transforms."""
        if isinstance(clip, (int, np.integer)):
            clip = self.animations[int(clip)]
        if not isinstance(clip, AnimationClip):
            raise TypeError("clip must be an AnimationClip or attached clip index")
        components = {}
        morph_updates = []
        for target, property_name, value in clip.sample(time, loop=loop):
            resource = (
                self.get_node(target) if isinstance(target, Node)
                else self._resolve(self.meshes, target, "animation target")
            )
            if property_name == "weights":
                morph_updates.append((resource, value))
                continue
            key = id(resource)
            base = self._animation_bases.get(key)
            if base is None:
                base = decompose_matrix(resource.transform.matrix)
                self._animation_bases[key] = base
            pair = components.setdefault(key, (resource, {
                "translation": base[0], "rotation": base[1], "scale": base[2],
            }))
            entry = pair[1]
            if property_name not in entry:
                raise ValueError(
                    f"unsupported instance animation property {property_name!r}"
                )
            entry[property_name] = value
        resources = tuple(pair[0] for pair in components.values())
        transforms = tuple(
            Transform(compose_matrix(
                item["translation"], item["rotation"], item["scale"]
            ))
            for _instance, item in components.values()
        )
        direct_instances = []
        direct_transforms = []
        nodes_changed = False
        for resource, transform in zip(resources, transforms):
            if isinstance(resource, Node):
                resource.transform = transform
                nodes_changed = True
            else:
                direct_instances.append(resource)
                direct_transforms.append(transform)
        if direct_instances:
            self.update_instance_transforms(direct_instances, direct_transforms)
        if nodes_changed:
            self._sync_node_instances()
        for resource, weights in morph_updates:
            targets = resource.instances if isinstance(resource, Node) else (resource,)
            for instance in targets:
                self.set_morph_weights(instance, weights)
        if nodes_changed and self._skin_bindings:
            self.update_skins()
        return resources

    def bind_morph_targets(self, instance, targets, weights=None):
        """Attach deformable morph targets to one placed mesh instance."""
        instance = self._resolve(self.meshes, instance, "morph instance")
        targets = tuple(targets)
        if not targets or any(not isinstance(target, MorphTarget) for target in targets):
            raise ValueError("targets must contain at least one MorphTarget")
        if any(len(target.position_deltas) != len(instance.vertices)
               for target in targets):
            raise ValueError("morph target vertex counts must match the instance")
        if not instance.deformable:
            raise ValueError("morph target instances must be created as deformable")
        if weights is None:
            weights = np.zeros(len(targets), np.float32)
        weights = np.asarray(weights, np.float32)
        if weights.shape != (len(targets),) or not np.all(np.isfinite(weights)):
            raise ValueError("morph weights must match targets and be finite")
        self._morph_bindings[id(instance)] = {
            "instance": instance,
            "targets": targets,
            "base_vertices": instance.vertices.copy(),
            "base_normals": instance.normals.copy(),
            "weights": weights.copy(),
        }
        self.set_morph_weights(instance, weights)
        return instance

    def set_morph_weights(self, instance, weights):
        """Apply absolute morph weights and update deformable geometry."""
        instance = self._resolve(self.meshes, instance, "morph instance")
        binding = self._morph_bindings.get(id(instance))
        if binding is None:
            raise KeyError("instance has no bound morph targets")
        weights = np.asarray(weights, np.float32).reshape(-1)
        targets = binding["targets"]
        if weights.shape != (len(targets),) or not np.all(np.isfinite(weights)):
            raise ValueError("morph weights must match targets and be finite")
        vertices = binding["base_vertices"].copy()
        normals = binding["base_normals"].copy()
        has_normal_deltas = False
        for weight, target in zip(weights, targets):
            vertices += float(weight) * target.position_deltas
            if target.normal_deltas is not None:
                normals += float(weight) * target.normal_deltas
                has_normal_deltas = True
        if has_normal_deltas:
            lengths = np.linalg.norm(normals, axis=1, keepdims=True)
            normals /= np.maximum(lengths, 1e-8)
        skin_binding = self._skin_bindings.get(id(instance))
        if skin_binding is None:
            self.update_mesh(instance, vertices=vertices, normals=normals)
        else:
            skin_binding["source_vertices"] = vertices
            skin_binding["source_normals"] = normals
            self.update_skin(instance)
        binding["weights"] = weights.copy()
        return instance

    def bind_skin(
        self, instance, skin, joint_indices, joint_weights, *, mesh_node=None,
    ):
        """Bind joint influences to one deformable mesh instance."""
        instance = self._resolve(self.meshes, instance, "skin instance")
        if not isinstance(skin, Skin):
            raise TypeError("skin must be a Skin")
        if not instance.deformable:
            raise ValueError("skinned instances must be created as deformable")
        indices = np.asarray(joint_indices, np.int32)
        weights = np.asarray(joint_weights, np.float32)
        expected = (len(instance.vertices), 4)
        if indices.shape != expected or weights.shape != expected:
            raise ValueError("joint indices and weights must have shape (vertex_count, 4)")
        if np.any(indices < 0) or np.any(indices >= len(skin.joints)):
            raise ValueError("joint index is outside the skin")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("joint weights must be finite and non-negative")
        totals = weights.sum(axis=1, keepdims=True)
        weights = np.divide(
            weights, np.maximum(totals, 1e-8),
            out=np.zeros_like(weights), where=totals > 1e-8,
        )
        if mesh_node is not None:
            mesh_node = self.get_node(mesh_node)
        morph = self._morph_bindings.get(id(instance))
        # A morph binding has already applied its current absolute weights to
        # the instance.  Skin that result, rather than reverting to its bind
        # pose, so the public deformation order is morph -> skin -> node.
        source_vertices = instance.vertices.copy()
        source_normals = instance.normals.copy()
        self._skin_bindings[id(instance)] = {
            "instance": instance, "skin": skin, "indices": indices.copy(),
            "weights": weights.copy(), "mesh_node": mesh_node,
            "source_vertices": source_vertices,
            "source_normals": source_normals,
        }
        self.update_skin(instance)
        return instance

    def update_skin(self, instance):
        """Recompute one skinned mesh from current joint transforms."""
        instance = self._resolve(self.meshes, instance, "skin instance")
        binding = self._skin_bindings.get(id(instance))
        if binding is None:
            raise KeyError("instance has no bound skin")
        skin = binding["skin"]
        mesh_node = binding["mesh_node"]
        inverse_mesh = (
            np.eye(4, dtype=np.float32) if mesh_node is None
            else np.linalg.inv(mesh_node.world_matrix).astype(np.float32)
        )
        matrices = np.asarray([
            inverse_mesh @ joint.world_matrix @ inverse_bind
            for joint, inverse_bind in zip(
                skin.joints, skin.inverse_bind_matrices
            )
        ], np.float32)
        indices = binding["indices"]
        weights = binding["weights"]
        source = binding["source_vertices"]
        homogeneous = np.column_stack((source, np.ones(len(source), np.float32)))
        transformed = np.einsum(
            "nkij,nj->nki", matrices[indices], homogeneous, optimize=True
        )[..., :3]
        vertices = np.sum(transformed * weights[..., None], axis=1)
        normal_matrices = np.linalg.inv(matrices[:, :3, :3]).transpose(0, 2, 1)
        transformed_normals = np.einsum(
            "nkij,nj->nki", normal_matrices[indices],
            binding["source_normals"], optimize=True,
        )
        normals = np.sum(transformed_normals * weights[..., None], axis=1)
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-8)
        self.update_mesh(instance, vertices=vertices, normals=normals)
        return instance

    def update_skins(self):
        """Recompute all attached skins after a joint hierarchy update."""
        return tuple(
            self.update_skin(binding["instance"])
            for binding in tuple(self._skin_bindings.values())
        )

    def reset_animation(self, targets=None):
        """Restore transforms captured before their first animation sample."""
        if targets is None:
            targets = tuple(
                resource for resource in (*self.meshes, *self.nodes)
                if id(resource) in self._animation_bases
            )
        elif isinstance(targets, (Mesh, Node, int, np.integer)):
            targets = (targets,)
        resources = tuple(
            self.get_node(target) if isinstance(target, Node)
            else self._resolve(self.meshes, target, "animation target")
            for target in targets
        )
        transforms = []
        restored = []
        for resource in resources:
            base = self._animation_bases.pop(id(resource), None)
            if base is not None:
                restored.append(resource)
                transforms.append(Transform(compose_matrix(*base)))
        if restored:
            direct_instances = []
            direct_transforms = []
            nodes_changed = False
            for resource, transform in zip(restored, transforms):
                if isinstance(resource, Node):
                    resource.transform = transform
                    nodes_changed = True
                else:
                    direct_instances.append(resource)
                    direct_transforms.append(transform)
            if direct_instances:
                self.update_instance_transforms(direct_instances, direct_transforms)
            if nodes_changed:
                self._sync_node_instances()
        return tuple(restored)

    def get_light(self, reference):
        """Resolve any analytic light object or stable ID."""
        return self._resolve(self.lights, reference, "light")

    def remove_light(self, light):
        """Remove any analytic light by object or stable ID and return it."""
        light = self.get_light(light)
        self.lights.remove(light)
        self._changed(shading=True)
        return light

    def add_point_light(
        self, position, color=(1.0, 1.0, 1.0), intensity=1.0, *, range=None,
    ):
        return self.add_light(PointLight(position, color, intensity, range))

    def set_environment(self, environment=None, **parameters):
        """Set, replace, or clear the unique scene environment light.

        Pass an :class:`EnvironmentLight`, keyword constructor parameters, or
        no arguments to clear it.
        """
        if environment is not None and parameters:
            raise TypeError("pass an EnvironmentLight or keyword parameters, not both")
        if parameters:
            environment = EnvironmentLight(**parameters)
        if environment is not None and not isinstance(environment, EnvironmentLight):
            raise TypeError("environment must be an EnvironmentLight or None")
        if environment is self.environment:
            return environment
        texture, _log_range = self._pack_environment_texture(environment)
        self.environment = environment
        self._environment_texture = texture
        self._changed(shading=True)
        return environment

    @staticmethod
    def _pack_environment_texture(environment):
        if environment is None or environment.image is None:
            return None, 0.0
        image = environment.image
        log_range = max(float(np.log2(1.0 + np.max(image))), 1e-6)
        encoded = np.zeros((*image.shape[:2], 4), np.uint8)
        encoded[..., :3] = np.clip(
            np.log2(1.0 + image) / log_range * 255.0 + 0.5, 0, 255
        ).astype(np.uint8)
        encoded[..., 3] = 255
        return Texture(encoded, wrap_s="repeat", wrap_t="clamp"), log_range

    def update_point_light(
        self, light, *, position=_UNCHANGED, color=_UNCHANGED,
        intensity=_UNCHANGED, range=_UNCHANGED,
    ):
        """Validate and update a point light while preserving its object and ID."""
        light = self._resolve(self.lights, light, "point light")
        if all(value is _UNCHANGED for value in (position, color, intensity, range)):
            return light
        candidate = PointLight(
            light.position if position is _UNCHANGED else position,
            light.color if color is _UNCHANGED else color,
            light.intensity if intensity is _UNCHANGED else intensity,
            light.range if range is _UNCHANGED else range,
        )
        light.position = candidate.position
        light.color = candidate.color
        light.intensity = candidate.intensity
        light.range = candidate.range
        self._changed(shading=True)
        return light

    def remove_point_light(self, light):
        """Remove a point light by object or stable integer ID and return it."""
        light = self._resolve(
            [candidate for candidate in self.lights
             if isinstance(candidate, PointLight)], light, "point light"
        )
        return self.remove_light(light)

    def add_directional_light(
        self, direction, color=(1.0, 1.0, 1.0), intensity=1.0,
    ):
        return self.add_light(DirectionalLight(direction, color, intensity))

    def update_directional_light(
        self, light, *, direction=_UNCHANGED, color=_UNCHANGED,
        intensity=_UNCHANGED,
    ):
        candidates = [item for item in self.lights
                      if isinstance(item, DirectionalLight)]
        light = self._resolve(candidates, light, "directional light")
        if all(value is _UNCHANGED for value in (direction, color, intensity)):
            return light
        candidate = DirectionalLight(
            light.direction if direction is _UNCHANGED else direction,
            light.color if color is _UNCHANGED else color,
            light.intensity if intensity is _UNCHANGED else intensity,
        )
        light.direction = candidate.direction
        light.color = candidate.color
        light.intensity = candidate.intensity
        self._changed(shading=True)
        return light

    def remove_directional_light(self, light):
        candidates = [item for item in self.lights
                      if isinstance(item, DirectionalLight)]
        return self.remove_light(
            self._resolve(candidates, light, "directional light")
        )

    def add_spot_light(
        self, position, direction, color=(1.0, 1.0, 1.0), intensity=1.0,
        *, inner_cone_angle=0.0, outer_cone_angle=np.pi / 4.0, range=None,
    ):
        return self.add_light(SpotLight(
            position, direction, color, intensity, inner_cone_angle,
            outer_cone_angle, range,
        ))

    def update_spot_light(
        self, light, *, position=_UNCHANGED, direction=_UNCHANGED,
        color=_UNCHANGED, intensity=_UNCHANGED, inner_cone_angle=_UNCHANGED,
        outer_cone_angle=_UNCHANGED, range=_UNCHANGED,
    ):
        candidates = [item for item in self.lights if isinstance(item, SpotLight)]
        light = self._resolve(candidates, light, "spot light")
        values = (
            position, direction, color, intensity, inner_cone_angle,
            outer_cone_angle, range,
        )
        if all(value is _UNCHANGED for value in values):
            return light
        candidate = SpotLight(
            light.position if position is _UNCHANGED else position,
            light.direction if direction is _UNCHANGED else direction,
            light.color if color is _UNCHANGED else color,
            light.intensity if intensity is _UNCHANGED else intensity,
            light.inner_cone_angle if inner_cone_angle is _UNCHANGED
            else inner_cone_angle,
            light.outer_cone_angle if outer_cone_angle is _UNCHANGED
            else outer_cone_angle,
            light.range if range is _UNCHANGED else range,
        )
        for name in (
            "position", "direction", "color", "intensity",
            "inner_cone_angle", "outer_cone_angle", "range",
        ):
            setattr(light, name, getattr(candidate, name))
        self._changed(shading=True)
        return light

    def remove_spot_light(self, light):
        candidates = [item for item in self.lights if isinstance(item, SpotLight)]
        return self.remove_light(self._resolve(candidates, light, "spot light"))

    def clear(self):
        """Remove every mesh, volume, and light while retaining monotonic IDs."""
        if (not self.meshes and not self.lights and not self.mesh_resources
                and not self.volumes and self.environment is None
                and not self.nodes and not self.animations):
            return
        self.meshes.clear()
        self.lights.clear()
        self.mesh_resources.clear()
        self.volumes.clear()
        self.environment = None
        self._environment_texture = None
        self.animations.clear()
        self._animation_bases.clear()
        self.nodes.clear()
        self._morph_bindings.clear()
        self._skin_bindings.clear()
        self._changed(geometry=True, shading=True)

    def point_light_data(self):
        """Pack two vec4 values per point light for GPU direct sampling."""
        records = [
            ((*light.position, 0.0), (*light.color, light.intensity))
            for light in self.lights if isinstance(light, PointLight)
        ]
        if not records:
            records = [((0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0))]
        return np.ascontiguousarray(records, dtype=np.float32).reshape((-1, 2, 4))

    def analytic_light_data(self):
        """Pack four vec4 values per analytic light for GPU sampling.

        The records contain ``position/type``, ``direction/range``,
        ``color/intensity``, and spot cone cosines respectively.
        """
        records = []
        for light in self.lights:
            if isinstance(light, PointLight):
                records.append((
                    (*light.position, float(POINT)),
                    (0.0, 0.0, 0.0, light.range or 0.0),
                    (*light.color, light.intensity),
                    (0.0, 0.0, 0.0, 0.0),
                ))
            elif isinstance(light, DirectionalLight):
                direction = np.asarray(light.direction, dtype=np.float32)
                direction /= np.linalg.norm(direction)
                records.append((
                    (0.0, 0.0, 0.0, float(DIRECTIONAL)),
                    (*direction, 0.0),
                    (*light.color, light.intensity),
                    (0.0, 0.0, 0.0, 0.0),
                ))
            else:
                direction = np.asarray(light.direction, dtype=np.float32)
                direction /= np.linalg.norm(direction)
                records.append((
                    (*light.position, float(SPOT)),
                    (*direction, light.range or 0.0),
                    (*light.color, light.intensity),
                    (float(np.cos(light.inner_cone_angle)),
                     float(np.cos(light.outer_cone_angle)), 0.0, 0.0),
                ))
        if self.environment is not None:
            texture_index = -1.0
            log_range = 0.0
            if self._environment_texture is not None:
                texture_index = float(next(
                    index for index, texture in enumerate(self.textures)
                    if texture is self._environment_texture
                ))
                _texture, log_range = self._pack_environment_texture(
                    self.environment
                )
            records.append((
                (0.0, 0.0, 0.0, 3.0),
                (0.0, 0.0, 0.0, 0.0),
                (*self.environment.color, self.environment.intensity),
                (texture_index, self.environment.rotation, log_range, 0.0),
            ))
        if not records:
            records = [((0.0, 0.0, 0.0, 0.0),) * 4]
        return np.ascontiguousarray(records, dtype=np.float32).reshape((-1, 4, 4))

    @property
    def analytic_light_count(self):
        return len(self.lights) + int(self.environment is not None)

    def emissive_triangle_data(self):
        """Pack world-space emissive triangles for area-light sampling.

        Each record contains the three vertices followed by emission and area.
        Degenerate and black triangles are not included.
        """
        records = []
        weights = []
        for mesh in self.visible_meshes:
            emission = np.asarray(mesh.material.emission, dtype=np.float32)
            if not np.any(emission > 0.0):
                continue
            for triangle in mesh.world_vertices[mesh.indices]:
                area = 0.5 * np.linalg.norm(
                    np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
                )
                if area > 1e-12:
                    weight = area * float(np.dot(
                        emission, (0.2126, 0.7152, 0.0722)
                    ))
                    records.append((
                        (*triangle[0], 0.0),
                        (*triangle[1], 0.0),
                        (*triangle[2], 0.0),
                        (*emission, area),
                    ))
                    weights.append(weight)
        if not records:
            records = [((0.0,) * 4,) * 4]
            weights = [1.0]
        probabilities = np.asarray(weights, dtype=np.float64)
        probabilities /= probabilities.sum()
        cdf = np.cumsum(probabilities)
        cdf[-1] = 1.0
        two_sided = []
        for mesh in self.visible_meshes:
            if not any(component > 0.0 for component in mesh.material.emission):
                continue
            triangles = mesh.world_vertices[mesh.indices]
            two_sided.extend(
                float(mesh.material.emission_two_sided)
                for triangle in triangles
                if 0.5 * np.linalg.norm(
                    np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
                ) > 1e-12
            )
        if len(two_sided) != len(records):
            two_sided = [0.0] * len(records)
        packed = [
            (*record, (float(cumulative), float(probability), sided, 0.0))
            for record, cumulative, probability, sided
            in zip(records, cdf, probabilities, two_sided)
        ]
        return np.ascontiguousarray(packed, dtype=np.float32).reshape((-1, 5, 4))

    @property
    def emissive_light_weight(self):
        """Total area-luminance weight used by emissive-light sampling."""
        total = 0.0
        for mesh in self.visible_meshes:
            emission = np.asarray(mesh.material.emission, dtype=np.float64)
            luminance = float(np.dot(emission, (0.2126, 0.7152, 0.0722)))
            if luminance <= 0.0:
                continue
            triangles = mesh.world_vertices[mesh.indices]
            areas = 0.5 * np.linalg.norm(
                np.cross(triangles[:, 1] - triangles[:, 0],
                         triangles[:, 2] - triangles[:, 0]), axis=1,
            )
            total += float(areas[areas > 1e-12].sum()) * luminance
        return total

    @property
    def emissive_triangle_count(self):
        count = 0
        for mesh in self.visible_meshes:
            if not any(component > 0.0 for component in mesh.material.emission):
                continue
            triangles = mesh.world_vertices[mesh.indices]
            count += int(np.count_nonzero(
                np.linalg.norm(
                    np.cross(triangles[:, 1] - triangles[:, 0],
                             triangles[:, 2] - triangles[:, 0]), axis=1,
                ) > 2e-12
            ))
        return count

    def triangles(self):
        """Return packed triangles, material colors, and emissions."""
        triangles = []
        colors = []
        emissions = []
        for mesh in self.visible_meshes:
            mesh_triangles = mesh.world_vertices[mesh.indices]
            triangles.append(mesh_triangles)
            count = len(mesh_triangles)
            colors.extend([mesh.material.base_color] * count)
            emissions.extend([mesh.material.emission] * count)
        if not triangles:
            return (
                np.empty((0, 3, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
            )
        return (
            np.concatenate(triangles),
            np.asarray(colors, dtype=np.float32),
            np.asarray(emissions, dtype=np.float32),
        )

    def render_triangles(self):
        """Return packed traversal triangles, including volume entry proxies."""
        triangles = [
            mesh.world_vertices[mesh.indices] for mesh in self.render_meshes
        ]
        if not triangles:
            return np.empty((0, 3, 3), dtype=np.float32)
        return np.ascontiguousarray(np.concatenate(triangles), np.float32)

    def triangle_volume_indices(self):
        """Map packed traversal triangles to volumes or ``0xffffffff``."""
        records = []
        for mesh in self.render_meshes:
            volume_index = mesh.metadata.get("volume_index", None)
            value = 0xffffffff if volume_index is None else int(volume_index)
            records.append(np.full(len(mesh.indices), value, np.uint32))
        return np.concatenate(records) if records else np.asarray([0xffffffff], np.uint32)

    def triangle_instance_ids(self):
        """Stable instance IDs in globally packed triangle order."""
        values = [
            np.full(len(mesh.indices), mesh.id, dtype=np.uint32)
            for mesh in self.render_meshes if len(mesh.indices)
        ]
        return np.concatenate(values) if values else np.empty(0, np.uint32)

    def triangle_object_ids(self):
        """Compatibility alias for :meth:`triangle_instance_ids`."""
        return self.triangle_instance_ids()

    def object_triangle_range(self, reference):
        """Return the half-open packed-triangle range for an object or volume."""
        object_id = int(reference.id if hasattr(reference, "id") else reference)
        offset = 0
        for mesh in self.render_meshes:
            count = len(mesh.indices)
            if mesh.id == object_id:
                return offset, offset + count
            offset += count
        raise KeyError(f"scene has no visible object with id {object_id}")

    def triangle_material_ids(self):
        """Stable scene-local material IDs in packed triangle order."""
        values = [
            np.full(
                len(mesh.indices), self.material_id(mesh.material),
                dtype=np.uint32,
            )
            for mesh in self.render_meshes if len(mesh.indices)
        ]
        return np.concatenate(values) if values else np.empty(0, np.uint32)

    def material_programs(self, default_program):
        """Return distinct programs in stable first-use order."""
        programs = []
        for mesh in self.render_meshes:
            program = mesh.material.program or default_program
            if all(program is not existing for existing in programs):
                programs.append(program)
        return tuple(programs) or (default_program,)

    def triangle_material_data(self, programs=None, default_program=None):
        """Pack triangle materials according to ``MATERIAL_PARAMETER_LAYOUT``."""
        if programs is not None and default_program is None:
            raise ValueError("default_program is required when programs are supplied")
        records = []
        bindings = self.texture_bindings
        for mesh in self.render_meshes:
            material = mesh.material
            program_id = 0
            if programs is not None:
                selected = material.program or default_program
                program_id = next(
                    index for index, program in enumerate(programs)
                    if program is selected
                )
            uses_uv1 = any(
                texture is not None and transform.texcoord_set == 1
                for texture, transform in (
                    (material.base_color_texture, material.base_color_transform),
                    (material.metallic_roughness_texture,
                     material.metallic_roughness_transform),
                    (material.emissive_texture, material.emissive_transform),
                    (material.normal_texture, material.normal_transform),
                    (material.occlusion_texture, material.occlusion_transform),
                    (material.transmission_texture,
                     material.transmission_transform),
                    (material.clearcoat_texture, material.base_color_transform),
                    (material.sheen_texture, material.base_color_transform),
                    (material.anisotropy_texture, material.base_color_transform),
                    (material.subsurface_texture, material.base_color_transform),
                )
            )
            record = (
                (*material.base_color, material.roughness),
                (*material.emission, material.metallic),
                (*material.attenuation_color, material.transmission),
                (material.ior, material.attenuation_distance,
                 float(program_id) + (0.25 if uses_uv1 else 0.0),
                 float(material.emission_two_sided)),
                (
                    float(self._texture_binding_index(
                        bindings, material.base_color_texture,
                        material.base_color_transform,
                    )),
                    float(self._texture_binding_index(
                        bindings, material.metallic_roughness_texture,
                        material.metallic_roughness_transform,
                    )),
                    float(self._texture_binding_index(
                        bindings, material.emissive_texture,
                        material.emissive_transform,
                    )),
                    float(self._texture_binding_index(
                        bindings, material.normal_texture,
                        material.normal_transform,
                    )),
                ),
                (
                    material.normal_scale,
                    float(self._texture_binding_index(
                        bindings, material.occlusion_texture,
                        material.occlusion_transform,
                    )),
                    material.occlusion_strength,
                    float(self._texture_binding_index(
                        bindings, material.transmission_texture,
                        material.transmission_transform,
                    )),
                ),
                (
                    material.clearcoat, material.clearcoat_roughness,
                    material.sheen_roughness, material.anisotropy,
                ),
                (
                    material.subsurface, material.subsurface_radius,
                    float(material.thin_walled), 0.0,
                ),
                (*material.sheen_color, 0.0),
                (*material.subsurface_color, 0.0),
                (
                    float(self._texture_binding_index(
                        bindings, material.clearcoat_texture,
                        material.base_color_transform,
                    )),
                    float(self._texture_binding_index(
                        bindings, material.sheen_texture,
                        material.base_color_transform,
                    )),
                    float(self._texture_binding_index(
                        bindings, material.anisotropy_texture,
                        material.base_color_transform,
                    )),
                    float(self._texture_binding_index(
                        bindings, material.subsurface_texture,
                        material.base_color_transform,
                    )),
                ),
            )
            records.extend([record] * len(mesh.indices))
        return np.ascontiguousarray(records, dtype=np.float32).reshape((-1, 11, 4))

    @property
    def textures(self):
        """Return referenced textures in stable first-use order."""
        result = []
        for mesh in self.render_meshes:
            for texture in (
                mesh.material.base_color_texture,
                mesh.material.metallic_roughness_texture,
                mesh.material.emissive_texture,
                mesh.material.normal_texture,
                mesh.material.occlusion_texture,
                mesh.material.transmission_texture,
                mesh.material.clearcoat_texture,
                mesh.material.sheen_texture,
                mesh.material.anisotropy_texture,
                mesh.material.subsurface_texture,
            ):
                if texture is not None and all(texture is not item for item in result):
                    result.append(texture)
        if (self._environment_texture is not None
                and all(self._environment_texture is not item for item in result)):
            result.append(self._environment_texture)
        return tuple(result)

    @property
    def texture_bindings(self):
        result = []
        for mesh in self.render_meshes:
            material = mesh.material
            for texture, transform in (
                (material.base_color_texture, material.base_color_transform),
                (material.metallic_roughness_texture,
                 material.metallic_roughness_transform),
                (material.emissive_texture, material.emissive_transform),
                (material.normal_texture, material.normal_transform),
                (material.occlusion_texture, material.occlusion_transform),
                (material.transmission_texture, material.transmission_transform),
                (material.clearcoat_texture, material.base_color_transform),
                (material.sheen_texture, material.base_color_transform),
                (material.anisotropy_texture, material.base_color_transform),
                (material.subsurface_texture, material.base_color_transform),
            ):
                if texture is None:
                    continue
                if all(texture is not item[0] or transform != item[1]
                       for item in result):
                    result.append((texture, transform))
        return tuple(result)

    @staticmethod
    def _texture_binding_index(bindings, texture, transform):
        return next(
            (index for index, item in enumerate(bindings)
             if item[0] is texture and item[1] == transform), -1
        )

    def texture_binding_data(self):
        """Pack texture resource indices and affine UV transforms."""
        textures = self.textures
        records = []
        for texture, transform in self.texture_bindings:
            texture_index = next(
                index for index, item in enumerate(textures) if item is texture
            )
            cosine = np.cos(transform.rotation)
            sine = np.sin(transform.rotation)
            records.append((
                (
                    float(texture_index), float(cosine), float(sine),
                    float(transform.texcoord_set),
                ),
                (*transform.offset, *transform.scale),
            ))
        if not records:
            records = [((-1.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 1.0))]
        return np.ascontiguousarray(records, dtype=np.float32).reshape((-1, 2, 4))

    def texture_data(self):
        """Pack metadata plus sRGB-correct and linear RGBA8 mip pyramids."""
        textures = self.textures
        metadata_words = 1 + 8 * len(textures)
        header = np.zeros(metadata_words, dtype=np.uint32)
        header[0] = len(textures)
        chunks = [header]
        offset = metadata_words
        wrap_codes = {"repeat": 0, "clamp": 1, "mirror": 2}
        for index, texture in enumerate(textures):
            height, width, _channels = texture.pixels.shape
            flags = (
                wrap_codes[texture.wrap_s]
                | (wrap_codes[texture.wrap_t] << 2)
                | (int(texture.linear_filter) << 4)
            )
            srgb_levels = self._texture_mips(texture.pixels, srgb=True)
            linear_levels = self._texture_mips(texture.pixels, srgb=False)
            srgb_words = np.concatenate([
                level.reshape(-1, 4).copy().view(np.uint32).reshape(-1)
                for level in srgb_levels
            ])
            srgb_offset = offset
            chunks.append(srgb_words)
            offset += len(srgb_words)
            linear_words = np.concatenate([
                level.reshape(-1, 4).copy().view(np.uint32).reshape(-1)
                for level in linear_levels
            ])
            linear_offset = offset
            chunks.append(linear_words)
            offset += len(linear_words)
            start = 1 + 8 * index
            header[start:start + 8] = (
                srgb_offset, linear_offset, width, height, flags,
                len(srgb_levels), 0, 0,
            )
        return np.ascontiguousarray(np.concatenate(chunks), dtype=np.uint32)

    @staticmethod
    def _texture_mips(pixels, *, srgb):
        current = np.asarray(pixels, dtype=np.float32) / 255.0
        if srgb:
            rgb = current[..., :3]
            current[..., :3] = np.where(
                rgb <= 0.04045, rgb / 12.92,
                ((rgb + 0.055) / 1.055) ** 2.4,
            )
        levels = []
        while True:
            encoded = current.copy()
            if srgb:
                rgb = np.maximum(encoded[..., :3], 0.0)
                encoded[..., :3] = np.where(
                    rgb <= 0.0031308, rgb * 12.92,
                    1.055 * rgb ** (1.0 / 2.4) - 0.055,
                )
            levels.append(np.clip(encoded * 255.0 + 0.5, 0, 255).astype(np.uint8))
            height, width, _channels = current.shape
            if width == 1 and height == 1:
                break
            next_width = max(1, (width + 1) // 2)
            next_height = max(1, (height + 1) // 2)
            padded = np.pad(
                current,
                ((0, height & 1), (0, width & 1), (0, 0)),
                mode="edge",
            )
            current = padded.reshape(
                next_height, 2, next_width, 2, 4
            ).mean(axis=(1, 3))
        return levels

    def bounds(self):
        """Return world-space minimum and maximum corners."""
        vertices = [mesh.world_vertices for mesh in self.visible_meshes]
        for volume in self.visible_volumes:
            corners = np.column_stack((
                _VOLUME_BOX_VERTICES,
                np.ones(len(_VOLUME_BOX_VERTICES), np.float32),
            ))
            vertices.append((
                corners @ volume.transform.matrix.T
            )[:, :3])
        if not vertices:
            empty = np.zeros(3, dtype=np.float32)
            return empty.copy(), empty.copy()
        vertices = np.concatenate(vertices)
        return vertices.min(axis=0), vertices.max(axis=0)
