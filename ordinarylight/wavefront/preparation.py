"""Device-independent preparation of split wavefront shading kernels."""

from dataclasses import dataclass
from importlib.resources import files

from .shading import SHADE_BUFFER_BINDINGS


@dataclass(frozen=True)
class ShadeVariant:
    native_textures: bool = False
    profiling: bool = False
    ordinaryshade: bool = False
    overlapping_volumes: bool = False
    scattering_volumes: bool = False
    multiple_scattering_volumes: bool = False
    volume_empty_space_skipping: bool = False
    denoiser_signal_capture: bool = False
    surface_only: bool = False

    def __post_init__(self):
        if any(
            not isinstance(getattr(self, name), bool)
            for name in self.__dataclass_fields__
        ):
            raise TypeError("Shading variant flags must be bools")
        if self.multiple_scattering_volumes and not self.scattering_volumes:
            raise ValueError("Multiple scattering requires scattering")
        if self.surface_only and (not self.ordinaryshade or any((
            self.overlapping_volumes, self.scattering_volumes,
            self.multiple_scattering_volumes, self.volume_empty_space_skipping,
        ))):
            raise ValueError("Surface-only shading requires volume-free Ordinary Shade")

    @property
    def shader_name(self):
        stem = (
            "wavefront_shade_ordinaryshade" if self.ordinaryshade else "wavefront_shade"
        )
        for flag, suffix in (
            (self.native_textures, "native"),
            (self.profiling, "profile"),
            (self.overlapping_volumes, "overlap"),
            (self.scattering_volumes, "scatter"),
            (self.multiple_scattering_volumes, "multi"),
            (self.volume_empty_space_skipping, "skip"),
        ):
            if flag:
                stem += "_" + suffix
        return stem + ".comp"


@dataclass(frozen=True)
class PreparedShading:
    """Shader bytes plus binding requirements; contains no native objects."""

    spirv: bytes
    variant: ShadeVariant
    material_layout: object = None
    custom_attributes: bool = False

    @property
    def workgroup_size(self):
        return (64, 1, 1)

    @property
    def push_constant_size(self):
        return 56

    @property
    def buffer_bindings(self):
        optional = {14, 16}
        required = set(SHADE_BUFFER_BINDINGS.values()) - optional
        if self.variant.profiling:
            required.add(14)
        if self.custom_attributes:
            required.add(16)
        return tuple(sorted(required))

    @property
    def sampled_arrays(self):
        return ((13, 128), (21, 16)) if self.variant.native_textures else ((21, 16),)

    def create_kernel(
        self,
        runtime,
        bindings,
        *,
        sampled_image_arrays,
        material_resources=None,
        sampled_image_layouts=None,
    ):
        """Allocate a kernel only after explicit scene/queue resources exist."""
        from ..runtime.kernel import VulkanKernel

        if not set(self.buffer_bindings) <= bindings.keys() or 8 not in bindings:
            raise ValueError("Prepared shading bindings are incomplete")
        if bindings.keys() - (set(SHADE_BUFFER_BINDINGS.values()) | {8}):
            raise ValueError("Unknown prepared shading binding")
        for slot, resource in bindings.items():
            if resource.kind != ("acceleration_structure" if slot == 8 else "buffer"):
                raise ValueError("Prepared shading binding kind mismatch")
        arrays = {key: tuple(values) for key, values in sampled_image_arrays.items()}
        if {key: len(values) for key, values in arrays.items()} != dict(
            self.sampled_arrays
        ):
            raise ValueError("Sampled arrays do not match prepared shader variant")
        actual = (
            material_resources.resource_layout
            if material_resources is not None
            else None
        )
        if actual != self.material_layout:
            raise ValueError("Material resources do not match prepared declarations")
        return VulkanKernel(
            runtime,
            self.spirv,
            bindings,
            push_constant_size=self.push_constant_size,
            sampled_image_arrays=arrays,
            sampled_image_layouts=sampled_image_layouts,
            material_resources=material_resources,
        )


def prepare_shading(
    *,
    variant=None,
    programs=None,
    attribute_layout=None,
    material_layout=None,
    material_modifier=None,
    compiler=None,
):
    """Select packaged SPIR-V or compile a custom shading variant without a GPU.

    Custom programs require an explicit attribute layout (possibly empty).
    External declarations are supplied as a pure MaterialResourceLayout at set 1.
    Stock denoiser signals use runtime constants; their capture flag only affects
    source specialization when custom programs are compiled, matching native GI.
    """
    from ..materials.layout import MaterialResourceLayout

    variant = ShadeVariant() if variant is None else variant
    if not isinstance(variant, ShadeVariant):
        raise TypeError("Expected ShadeVariant")
    if material_layout is not None:
        if not isinstance(material_layout, MaterialResourceLayout):
            raise TypeError("Expected device-independent MaterialResourceLayout")
        if material_layout.descriptor_set != 1 or material_layout.first_binding != 0:
            raise ValueError(
                "Prepared camera shading requires material set 1 starting at binding 0"
            )
    if programs is None:
        if variant.surface_only:
            raise ValueError("Surface-only shading requires runtime shader compilation")
        if any(
            value is not None
            for value in (attribute_layout, material_layout, material_modifier)
        ):
            raise ValueError("Custom shader inputs require explicit programs")
        spirv = (
            files("ordinarylight.shaders")
            .joinpath(variant.shader_name + ".spv")
            .read_bytes()
        )
        custom = False
    else:
        from ..shaders.compiler import compile_wavefront_material_shader

        programs = tuple(programs)
        if not programs or attribute_layout is None:
            raise ValueError("Custom shading requires programs and an attribute layout")
        if material_layout is not None:
            material_layout.validate(programs)
        spirv = compile_wavefront_material_shader(
            "wavefront_shade_candidate.glsl"
            if variant.ordinaryshade
            else "wavefront_shade.comp",
            programs,
            attribute_layout=attribute_layout,
            attribute_binding=16,
            overlapping_volumes=variant.overlapping_volumes,
            scattering_volumes=variant.scattering_volumes,
            multiple_scattering_volumes=variant.multiple_scattering_volumes,
            volume_empty_space_skipping=variant.volume_empty_space_skipping,
            native_textures=variant.native_textures,
            profiling=variant.profiling,
            denoiser_signal_capture=variant.denoiser_signal_capture,
            material_modifier=material_modifier,
            material_resources=material_layout,
            compiler=compiler,
            surface_only=variant.surface_only,
        )
        custom = bool(attribute_layout.channels)
    return PreparedShading(spirv, variant, material_layout, custom)
