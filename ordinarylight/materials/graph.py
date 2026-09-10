"""Declarative material DAGs shared by all OrdinaryLight consumers.

This module does not depend on OrdinaryLattice or a rendering target. Its output
is the existing MaterialProgram contract, also accepted by camera materials.
"""

from dataclasses import dataclass, field
from types import MappingProxyType
import math

from ._core import (
    Expression,
    MaterialContext,
    MaterialEvaluation,
    MaterialProgram,
    MATERIAL_PARAMETER_LAYOUT,
    _coerce,
    _number,
    mix,
    select,
    dot,
    normalize,
)


@dataclass(frozen=True)
class MaterialResource:
    """A named vec4 uniform/buffer or separately sampled 2D color texture."""

    name: str
    kind: str

    def __post_init__(self):
        if not self.name.isidentifier() or not self.name.isascii():
            raise ValueError("Material resource names must be ASCII identifiers")
        if self.kind not in {"uniform", "buffer", "texture"}:
            raise ValueError("Unknown material resource kind")


@dataclass(frozen=True)
class MaterialNode:
    """A typed DAG node; arguments name other nodes, never shader source."""

    operation: str
    arguments: tuple[str, ...] = ()
    value: object = None
    type: str = "float"

    def __post_init__(self):
        object.__setattr__(self, "arguments", tuple(self.arguments))
        if not all(isinstance(name, str) and name for name in self.arguments):
            raise ValueError("Node arguments must be nonempty node names")
        if isinstance(self.value, list):
            object.__setattr__(self, "value", tuple(self.value))


@dataclass(frozen=True)
class MaterialGraph:
    """Named material nodes and output-to-node connections.

    Unconnected outputs inherit the fixed material's values. All nodes, including
    disconnected nodes, are validated. Compilation preserves the camera material
    API; a graph can be reused by any application without a lattice dependency.
    """

    nodes: object
    outputs: object
    name: str = "material_graph"
    resources: tuple[MaterialResource, ...] = field(default_factory=tuple)

    def __post_init__(self):
        resources = tuple(self.resources)
        if not all(isinstance(item, MaterialResource) for item in resources) or len(
            {item.name for item in resources}
        ) != len(resources):
            raise ValueError(
                "Graph resources require unique MaterialResource declarations"
            )
        object.__setattr__(self, "resources", resources)
        nodes, outputs = dict(self.nodes), dict(self.outputs)
        if not all(
            isinstance(name, str) and name and isinstance(node, MaterialNode)
            for name, node in nodes.items()
        ):
            raise TypeError("Graph nodes must map names to MaterialNode values")
        if not self.name.isidentifier() or not self.name.isascii():
            raise ValueError("Graph name must be an ASCII identifier")
        object.__setattr__(self, "nodes", MappingProxyType(nodes))
        object.__setattr__(self, "outputs", MappingProxyType(outputs))
        self.compile()

    def compile(self):
        context = MaterialContext.shader_inputs()
        values, visiting = {}, set()
        operations = {
            "add": (2, lambda a, b: a + b),
            "subtract": (2, lambda a, b: a - b),
            "multiply": (2, lambda a, b: a * b),
            "divide": (2, lambda a, b: a / b),
            "mix": (3, mix),
            "select": (3, select),
            "dot": (2, dot),
            "normalize": (1, normalize),
        }

        def evaluate(name):
            if name in visiting:
                raise ValueError(f"Material graph cycle at {name!r}")
            if name in values:
                return values[name]
            if name not in self.nodes:
                raise ValueError(f"Missing material node {name!r}")
            visiting.add(name)
            node = self.nodes[name]
            arguments = [evaluate(argument) for argument in node.arguments]
            if node.operation in {"constant", "input"}:
                if arguments:
                    raise ValueError("Constant/input nodes cannot have arguments")
                if node.operation == "constant":
                    numbers = (
                        node.value
                        if isinstance(node.value, (list, tuple))
                        else (node.value,)
                    )
                    if not all(
                        isinstance(value, (int, float)) and math.isfinite(value)
                        for value in numbers
                    ):
                        raise ValueError("Graph constants must contain finite numbers")
                    result = (
                        _number(node.value)
                        if node.type == "bool"
                        else _coerce(node.value, node.type)
                    )
                else:
                    if not isinstance(node.value, str) or node.value.startswith("_"):
                        raise ValueError("Invalid material context input")
                    result = getattr(context, node.value, None)
                    if not isinstance(result, Expression):
                        raise ValueError(f"Unknown material input {node.value!r}")
            elif node.operation in {"uniform", "buffer", "texture"}:
                declarations = {item.name: item.kind for item in self.resources}
                if declarations.get(node.value) != node.operation:
                    raise ValueError("Resource node must match a declared resource")
                expected = {"uniform": (), "buffer": ("float",), "texture": ("vec2",)}[
                    node.operation
                ]
                if tuple(argument.type for argument in arguments) != expected:
                    raise TypeError(
                        "Material resource coordinates have incorrect types"
                    )
                code = (
                    f"ol_graph_{node.value}("
                    + ", ".join(argument.code for argument in arguments)
                    + ")"
                )
                result = Expression("vec4", code, f"ol_graph_{node.value}(" + ", ".join(argument.python for argument in arguments) + ")")
            elif node.operation == "components":
                if len(arguments) != 1 or not isinstance(node.value, str):
                    raise ValueError("Component nodes require one vector and a swizzle")
                if (
                    not node.value
                    or len(node.value) > 4
                    or not (
                        set(node.value) <= set("xyzw") or set(node.value) <= set("rgba")
                    )
                ):
                    raise ValueError("Invalid component swizzle")
                result = getattr(arguments[0], node.value)
            elif node.operation in operations:
                arity, function = operations[node.operation]
                if len(arguments) != arity:
                    raise ValueError(f"{node.operation} requires {arity} arguments")
                if node.operation == "dot" and (
                    arguments[0].type != arguments[1].type
                    or arguments[0].type not in {"vec2", "vec3", "vec4"}
                ):
                    raise TypeError("dot requires matching vectors")
                if node.operation == "normalize" and arguments[0].type not in {
                    "vec2",
                    "vec3",
                    "vec4",
                }:
                    raise TypeError("normalize requires a vector")
                if node.operation in {"add", "subtract", "multiply", "divide"} and any(
                    argument.type == "bool" for argument in arguments
                ):
                    raise TypeError("Arithmetic nodes require numeric inputs")
                if node.operation == "mix" and arguments[2].type not in {
                    "float",
                    arguments[0].type,
                }:
                    raise TypeError("Mix weight must be scalar or match the endpoints")
                result = function(*arguments)
            else:
                raise ValueError(f"Unknown material operation {node.operation!r}")
            if result.type != node.type:
                raise TypeError(
                    f"Node {name!r} declares {node.type}, produces {result.type}"
                )
            visiting.remove(name)
            values[name] = result
            return result

        for name in self.nodes:
            evaluate(name)
        output_types = dict(MATERIAL_PARAMETER_LAYOUT)
        if set(self.outputs) - output_types.keys():
            raise ValueError("Unknown material output")
        outputs = {}
        for name, expected in output_types.items():
            result = (
                evaluate(self.outputs[name])
                if name in self.outputs
                else getattr(context, name)
            )
            if result.type != expected:
                raise TypeError(f"Material output {name} must be {expected}")
            outputs[name] = result
        return MaterialProgram(
            self.name, MaterialEvaluation(**outputs), resources=self.resources
        )
