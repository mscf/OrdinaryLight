"""Python-authored, backend-neutral material expression programs.

Material functions are evaluated once while they are defined. Their symbolic
arguments build a small typed IR; ordinary Python is never executed on the GPU.
Backends compile the resulting :class:`MaterialProgram` to their shader language.
"""

from dataclasses import dataclass, field
import re


_TYPES = {"float", "bool", "vec2", "vec3", "vec4"}
_VECTOR_TYPES = {"vec2", "vec3", "vec4"}
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# SurfaceResponse event values are floats in the expression IR and converted
# to integers by the shader. Absorb terminates; all scattering events continue.
SCATTER_ABSORB = 0.0
SCATTER_DIFFUSE = 1.0
SCATTER_REFLECTION = 2.0
SCATTER_TRANSMISSION = 3.0

# This ordering mirrors Scene.triangle_material_data() and is the stable
# backend-neutral contract consumed by generated material programs.
MATERIAL_PARAMETER_LAYOUT = (
    ("base_color", "vec3"), ("roughness", "float"),
    ("emission", "vec3"), ("metallic", "float"),
    ("attenuation_color", "vec3"), ("transmission", "float"),
    ("ior", "float"), ("attenuation_distance", "float"),
)


def _number(value):
    if isinstance(value, bool):
        return Expression("bool", "true" if value else "false")
    if isinstance(value, (int, float)):
        text = repr(float(value))
        if "e" not in text.lower() and "." not in text:
            text += ".0"
        return Expression("float", text)
    if isinstance(value, Expression):
        return value
    raise TypeError(f"Expected a material expression or number, got {type(value).__name__}")


@dataclass(frozen=True)
class Expression:
    """A typed immutable node in a material expression graph."""

    type: str
    code: str

    def __post_init__(self):
        if self.type not in _TYPES:
            raise ValueError(f"Unsupported shader type: {self.type}")

    def _binary(self, other, operator, *, comparison=False):
        other = _number(other)
        if self.type == other.type:
            result_type = self.type
        elif self.type in _VECTOR_TYPES and other.type == "float":
            result_type = self.type
        elif self.type == "float" and other.type in _VECTOR_TYPES:
            result_type = other.type
        else:
            raise TypeError(f"Cannot apply {operator} to {self.type} and {other.type}")
        if comparison and result_type != "float":
            raise TypeError("Material comparisons currently require scalar operands")
        return Expression("bool" if comparison else result_type, f"({self.code} {operator} {other.code})")

    def __add__(self, other): return self._binary(other, "+")
    def __radd__(self, other): return _number(other)._binary(self, "+")
    def __sub__(self, other): return self._binary(other, "-")
    def __rsub__(self, other): return _number(other)._binary(self, "-")
    def __mul__(self, other): return self._binary(other, "*")
    def __rmul__(self, other): return _number(other)._binary(self, "*")
    def __truediv__(self, other): return self._binary(other, "/")
    def __rtruediv__(self, other): return _number(other)._binary(self, "/")
    def __neg__(self): return Expression(self.type, f"(-{self.code})")
    def __lt__(self, other): return self._binary(other, "<", comparison=True)
    def __le__(self, other): return self._binary(other, "<=", comparison=True)
    def __gt__(self, other): return self._binary(other, ">", comparison=True)
    def __ge__(self, other): return self._binary(other, ">=", comparison=True)
    def __and__(self, other):
        other = _number(other)
        if self.type != "bool" or other.type != "bool":
            raise TypeError("Symbolic '&' requires bool operands")
        return Expression("bool", f"({self.code} && {other.code})")
    def __or__(self, other):
        other = _number(other)
        if self.type != "bool" or other.type != "bool":
            raise TypeError("Symbolic '|' requires bool operands")
        return Expression("bool", f"({self.code} || {other.code})")

    def __bool__(self):
        raise TypeError("Symbolic material conditions must use ordinarylight.select()")

    def __getattr__(self, components):
        valid = "xyzw" if set(components) <= set("xyzw") else "rgba"
        if not components or len(components) > 4 or any(c not in valid for c in components):
            raise AttributeError(components)
        width = int(self.type[-1]) if self.type in _VECTOR_TYPES else 1
        indices = "xyzw" if valid == "xyzw" else "rgba"
        if any(indices.index(c) >= width for c in components):
            raise AttributeError(components)
        result_type = "float" if len(components) == 1 else f"vec{len(components)}"
        return Expression(result_type, f"{self.code}.{components}")


def _call(name, result_type, *arguments):
    args = [_number(argument) for argument in arguments]
    return Expression(result_type, f"{name}({', '.join(arg.code for arg in args)})")


def _coerce(value, expected_type):
    if isinstance(value, Expression):
        return value
    if expected_type == "float":
        return _number(value)
    if expected_type in _VECTOR_TYPES:
        width = int(expected_type[-1])
        if isinstance(value, (tuple, list)) and len(value) == width:
            return _call(expected_type, expected_type, *value)
        if isinstance(value, (int, float)):
            scalar = _number(value)
            return Expression(expected_type, f"{expected_type}({scalar.code})")
    raise TypeError(f"Cannot convert {value!r} to {expected_type}")


def vec2(x, y): return _call("vec2", "vec2", x, y)
def vec3(x, y=None, z=None):
    if y is None and z is None:
        value = _number(x)
        return Expression("vec3", f"vec3({value.code})")
    return _call("vec3", "vec3", x, y, z)
def vec4(x, y, z, w): return _call("vec4", "vec4", x, y, z, w)
def dot(a, b): return _call("dot", "float", a, b)
def normalize(value): return _call("normalize", _number(value).type, value)
def reflect(direction, normal): return _call("reflect", "vec3", direction, normal)
def refract(direction, normal, eta): return _call("refract", "vec3", direction, normal, eta)
def maximum(a, b):
    a, b = _number(a), _number(b)
    if a.type != b.type:
        raise TypeError("maximum operands must have matching types")
    return _call("max", a.type, a, b)
def fresnel_schlick(cosine, ior_from, ior_to):
    return _call("waveFresnelSchlick", "float", cosine, ior_from, ior_to)
def cosine_sample_hemisphere(normal, random_u, random_v):
    return _call("waveCosineHemisphere", "vec3", normal, random_u, random_v)


def mix(a, b, amount):
    a, b = _number(a), _number(b)
    if a.type != b.type:
        raise TypeError("mix endpoints must have matching types")
    return _call("mix", a.type, a, b, amount)


def select(condition, when_true, when_false):
    condition, when_true, when_false = map(_number, (condition, when_true, when_false))
    if condition.type != "bool":
        raise TypeError("select condition must be bool")
    if when_true.type != when_false.type:
        raise TypeError("select branches must have matching types")
    return Expression(
        when_true.type,
        f"({condition.code} ? {when_true.code} : {when_false.code})",
    )


@dataclass(frozen=True)
class MaterialContext:
    normal: Expression
    uv: Expression
    direction: Expression
    entering: Expression
    base_color: Expression
    emission: Expression
    metallic: Expression
    roughness: Expression
    transmission: Expression
    ior: Expression
    attenuation_color: Expression
    attenuation_distance: Expression
    random_u: Expression
    random_v: Expression
    bounce: Expression
    current_ior: Expression
    exterior_ior: Expression
    _attribute_requests: dict = field(
        default_factory=dict, compare=False, repr=False
    )

    def attribute(self, name, *, components=1):
        """Declare and access an interpolated custom vertex attribute."""
        if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name):
            raise ValueError("attribute name must be a valid identifier")
        components = int(components)
        if not 1 <= components <= 4:
            raise ValueError("attribute components must be between 1 and 4")
        previous = self._attribute_requests.get(name)
        if previous is not None and previous != components:
            raise ValueError(
                f"attribute {name!r} was already declared with {previous} components"
            )
        self._attribute_requests[name] = components
        expression_type = "float" if components == 1 else f"vec{components}"
        return Expression(
            expression_type,
            f"waveVertexAttribute{components}(WAVE_ATTRIBUTE_{name})",
        )

    @classmethod
    def shader_inputs(cls):
        return cls(
            Expression("vec3", "normal"), Expression("vec2", "uv"),
            Expression("vec3", "direction"),
            Expression("bool", "entering"), Expression("vec3", "material.base_roughness.rgb"),
            Expression("vec3", "material.emission_metallic.rgb"),
            Expression("float", "material.emission_metallic.a"),
            Expression("float", "material.base_roughness.a"),
            Expression("float", "material.attenuation_transmission.a"),
            Expression("float", "material.ior_distance.x"),
            Expression("vec3", "material.attenuation_transmission.rgb"),
            Expression("float", "material.ior_distance.y"),
            Expression("float", "random_u"), Expression("float", "random_v"),
            Expression("float", "bounce_index"),
            Expression("float", "current_ior"), Expression("float", "exterior_ior"),
        )


@dataclass(frozen=True)
class MaterialEvaluation:
    base_color: Expression
    emission: Expression
    metallic: Expression
    roughness: Expression
    transmission: Expression
    ior: Expression
    attenuation_color: Expression
    attenuation_distance: Expression

    def __post_init__(self):
        types = (
            ("base_color", "vec3"), ("emission", "vec3"),
            ("metallic", "float"), ("roughness", "float"),
            ("transmission", "float"), ("ior", "float"),
            ("attenuation_color", "vec3"), ("attenuation_distance", "float"),
        )
        for name, expected in types:
            object.__setattr__(self, name, _coerce(getattr(self, name), expected))


@dataclass(frozen=True)
class SurfaceResponse:
    """A material-controlled path-scattering result."""

    emission: Expression
    weight: Expression
    next_direction: Expression
    event: Expression
    pdf: Expression

    def __post_init__(self):
        for name, expected in (
            ("emission", "vec3"), ("weight", "vec3"),
            ("next_direction", "vec3"), ("event", "float"), ("pdf", "float"),
        ):
            object.__setattr__(self, name, _coerce(getattr(self, name), expected))


@dataclass(frozen=True)
class MaterialProgram:
    """Compiled material IR plus GLSL generation and ABI metadata."""

    name: str
    evaluation: MaterialEvaluation | SurfaceResponse
    required_attributes: tuple[tuple[str, int], ...] = ()

    @property
    def parameter_layout(self):
        return MATERIAL_PARAMETER_LAYOUT

    @property
    def raster_kind(self):
        """Portable real-time approximation selected by raster renderers.

        Path-controlled programs cannot execute stochastic continuation in a
        raster pipeline.  Their declared event is therefore mapped to the
        closest deterministic surface model while parameter-evaluation
        programs retain the standard PBR model.
        """
        if isinstance(self.evaluation, MaterialEvaluation):
            return "unlit" if self.name == "unlit_material" else "pbr"
        event = self.evaluation.event.code
        if event == repr(float(SCATTER_DIFFUSE)):
            return "diffuse"
        if event == repr(float(SCATTER_REFLECTION)):
            return "mirror"
        if event == repr(float(SCATTER_TRANSMISSION)) or "3.0" in event:
            return "glass"
        return "pbr"

    def glsl(self, function_name="evaluateMaterial", *, attribute_slots=None):
        if not _IDENTIFIER.match(function_name):
            raise ValueError("function_name must be a valid shader identifier")
        material_expected = {
            "base_color": "vec3", "emission": "vec3", "metallic": "float",
            "roughness": "float", "transmission": "float", "ior": "float",
            "attenuation_color": "vec3", "attenuation_distance": "float",
        }
        lines = [
            f"MaterialEvaluation {function_name}(MaterialData material, vec3 normal, vec2 uv, vec3 direction, bool entering, float random_u, float random_v, float bounce_index, float current_ior, float exterior_ior)",
            "{", "    MaterialEvaluation result;",
        ]
        if isinstance(self.evaluation, MaterialEvaluation):
            for field_name, field_type in material_expected.items():
                expression = getattr(self.evaluation, field_name)
                self._assign(lines, field_name, field_type, expression)
            lines.extend((
                "    result.custom_scattering = 0.0;",
                "    result.weight = vec3(0.0);",
                "    result.next_direction = direction;",
                "    result.event = 0.0;",
                "    result.pdf = 1.0;",
            ))
        else:
            response_expected = {
                "emission": "vec3", "weight": "vec3",
                "next_direction": "vec3", "event": "float", "pdf": "float",
            }
            # Preserve the parameter fields so the ABI remains common across
            # old and new programs, even though the custom path does not use them.
            lines.extend((
                "    result.base_color = material.base_roughness.rgb;",
                "    result.metallic = material.emission_metallic.a;",
                "    result.roughness = material.base_roughness.a;",
                "    result.transmission = material.attenuation_transmission.a;",
                "    result.ior = material.ior_distance.x;",
                "    result.attenuation_color = material.attenuation_transmission.rgb;",
                "    result.attenuation_distance = material.ior_distance.y;",
                "    result.custom_scattering = 1.0;",
            ))
            for field_name, field_type in response_expected.items():
                expression = getattr(self.evaluation, field_name)
                self._assign(lines, field_name, field_type, expression)
        lines.extend(("    return result;", "}"))
        source = "\n".join(lines)
        if attribute_slots is not None:
            for name, components in self.required_attributes:
                try:
                    slot = int(attribute_slots[name])
                except KeyError as error:
                    raise ValueError(
                        f"no shader slot was supplied for attribute {name!r}"
                    ) from error
                if slot < 0:
                    raise ValueError("attribute slots cannot be negative")
                source = source.replace(f"WAVE_ATTRIBUTE_{name}", f"{slot}u")
        return source

    @staticmethod
    def _assign(lines, field_name, field_type, expression):
        if not isinstance(expression, Expression) or expression.type != field_type:
            raise TypeError(f"{field_name} must be a {field_type} expression")
        lines.append(f"    result.{field_name} = {expression.code};")


def material_dispatch_glsl(programs, *, attribute_slots=None):
    """Generate evaluator functions and a material-ID dispatcher."""
    programs = tuple(programs)
    if not programs:
        raise ValueError("At least one material program is required")
    functions = [
        program.glsl(
            f"evaluateMaterial_{index}", attribute_slots=attribute_slots
        )
        for index, program in enumerate(programs)
    ]
    lines = [
        *functions,
        "MaterialEvaluation evaluateMaterial(MaterialData material, vec3 normal, vec2 uv, vec3 direction, bool entering, float random_u, float random_v, float bounce_index, float current_ior, float exterior_ior)",
        "{",
        "    int program_id = int(floor(material.ior_distance.z));",
    ]
    for index in range(1, len(programs)):
        lines.append(
            f"    if (program_id == {index}) return evaluateMaterial_{index}(material, normal, uv, direction, entering, random_u, random_v, bounce_index, current_ior, exterior_ior);"
        )
    lines.extend((
        "    return evaluateMaterial_0(material, normal, uv, direction, entering, random_u, random_v, bounce_index, current_ior, exterior_ior);",
        "}",
    ))
    return "\n".join(lines)


def material(function):
    """Trace a restricted Python material function into a MaterialProgram."""
    context = MaterialContext.shader_inputs()
    evaluation = function(context)
    if not isinstance(evaluation, (MaterialEvaluation, SurfaceResponse)):
        raise TypeError("A material function must return MaterialEvaluation or SurfaceResponse")
    program = MaterialProgram(
        function.__name__, evaluation,
        tuple(context._attribute_requests.items()),
    )
    program.glsl()  # Eagerly validate all output types.
    return program


@material
def builtin_material(ctx):
    return MaterialEvaluation(
        base_color=ctx.base_color,
        emission=ctx.emission,
        metallic=ctx.metallic,
        roughness=ctx.roughness,
        transmission=ctx.transmission,
        ior=ctx.ior,
        attenuation_color=ctx.attenuation_color,
        attenuation_distance=ctx.attenuation_distance,
    )


@material
def unlit_material(ctx):
    """Emit the textured base color without receiving scene lighting."""
    return MaterialEvaluation(
        base_color=vec3(0.0),
        emission=ctx.base_color,
        metallic=0.0,
        roughness=1.0,
        transmission=0.0,
        ior=1.0,
        attenuation_color=vec3(1.0),
        attenuation_distance=ctx.attenuation_distance,
    )
