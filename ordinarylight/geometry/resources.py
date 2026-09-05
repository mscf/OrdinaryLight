"""Read-only resources declared by custom intersection programs."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class IntersectionResource:
    """A named std430 scalar/vector array or rgba32f storage image.

    Resource names are shared across programs in a scene. Bindings are assigned
    by OrdinaryLight, never by callback source. Callbacks may read but not write.
    """

    name: str
    kind: str = "buffer"
    element_type: str = "vec4"

    def __post_init__(self):
        if not re.fullmatch(
            r"[A-Za-z][A-Za-z0-9_]*", self.name
        ) or self.name.startswith(("gl_", "ordinarylight", "transport_", "OL_")):
            raise ValueError("Use an application GLSL identifier for resource names")
        if self.kind == "buffer":
            if self.element_type not in {
                "float",
                "int",
                "uint",
                "vec2",
                "vec4",
                "ivec2",
                "ivec4",
                "uvec2",
                "uvec4",
            }:
                raise ValueError(
                    "Buffer elements must be supported std430 scalars/vectors"
                )
        elif self.kind == "image":
            if self.element_type != "rgba32f":
                raise ValueError("Custom images currently require rgba32f image2D")
        else:
            raise ValueError("Custom resources must be buffers or storage images")

    @property
    def stride(self):
        return (
            4
            if self.element_type in {"float", "int", "uint"}
            else 8
            if self.element_type.endswith("2")
            else 16
        )

    def declaration(self, binding):
        if self.kind == "buffer":
            return f"layout(set=0,binding={binding},std430) readonly buffer OL_Resource_{binding} {{ {self.element_type} {self.name}[]; }};\n"
        return f"layout(set=0,binding={binding},rgba32f) readonly uniform image2D {self.name};\n"
