"""Device-independent descriptor declarations for external material resources."""

from dataclasses import dataclass
from operator import index


@dataclass(frozen=True)
class MaterialResourceLayout:
    """Immutable material resource ABI accepted by material shader preparation.

    No Vulkan runtime or allocation is needed. descriptor_set=1 is the camera GI
    contract; other executors can select their own set and first binding.
    """

    declarations: tuple
    descriptor_set: int = 1
    first_binding: int = 0

    def __post_init__(self):
        from .graph import MaterialResource

        declarations = {}
        for declaration in self.declarations:
            if not isinstance(declaration, MaterialResource):
                raise TypeError("Expected MaterialResource declarations")
            old = declarations.setdefault(declaration.name, declaration)
            if old != declaration:
                raise ValueError("Conflicting material resource declarations")
        object.__setattr__(
            self,
            "declarations",
            tuple(declarations[name] for name in sorted(declarations)),
        )
        for name in ("descriptor_set", "first_binding"):
            value = index(getattr(self, name))
            if not 0 <= value <= 0xFFFFFFFF:
                raise ValueError("Descriptor set and binding must fit uint32")
            object.__setattr__(self, name, value)
        if (
            self.first_binding
            + sum(2 if d.kind == "texture" else 1 for d in self.declarations)
            > 0x100000000
        ):
            raise ValueError("Material binding range exceeds uint32")

    @classmethod
    def from_programs(cls, programs, *, descriptor_set=1, first_binding=0):
        return cls(
            tuple(
                resource
                for program in programs
                for resource in getattr(program, "resources", ())
            ),
            descriptor_set,
            first_binding,
        )

    @property
    def entries(self):
        binding = self.first_binding
        entries = []
        for declaration in self.declarations:
            entries.append((declaration, binding))
            binding += 2 if declaration.kind == "texture" else 1
        return tuple(entries)

    def validate(self, programs):
        requested = MaterialResourceLayout.from_programs(programs)
        if not set(requested.declarations) <= set(self.declarations):
            raise ValueError(
                "Material resource layout does not cover graph declarations"
            )

    @property
    def source(self):
        sources = []
        descriptor_set = self.descriptor_set
        for declaration, binding in self.entries:
            prefix = f"ol_graph_{declaration.name}"
            if declaration.kind == "uniform":
                sources.append(
                    f"layout(set={descriptor_set},binding={binding},std140) uniform {prefix}_block {{ vec4 value; }} {prefix}_data;\n"
                )
            elif declaration.kind == "buffer":
                sources.append(
                    f"layout(set={descriptor_set},binding={binding},std430) readonly buffer {prefix}_block {{ vec4 values[]; }} {prefix}_data;\n"
                )
            else:
                sources.append(
                    f"layout(set={descriptor_set},binding={binding}) uniform texture2D {prefix}_image;\nlayout(set={descriptor_set},binding={binding + 1}) uniform sampler {prefix}_sampler;\n"
                )
            from .shade import resource_accessor
            sources.append(resource_accessor(prefix, declaration.kind))
        return "".join(sources)
