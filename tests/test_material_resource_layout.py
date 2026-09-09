"""Material shader preparation must not require a native runtime."""

from dataclasses import FrozenInstanceError
import pytest
import ordinarylight as ol
from ordinarylight.materials import (
    MaterialGraph,
    MaterialNode,
    MaterialResource,
    MaterialResourceLayout,
)
from ordinarylight.shaders.compiler import (
    compile_wavefront_material_shader,
    wavefront_material_shader_source,
    find_glsl_compiler,
)


def program():
    return MaterialGraph(
        {
            "value": MaterialNode("uniform", value="emission", type="vec4"),
            "rgb": MaterialNode("components", ("value",), value="rgb", type="vec3"),
        },
        {"emission": "rgb"},
        resources=(MaterialResource("emission", "uniform"),),
    ).compile()


def test_layout_orders_bindings_and_validates_declarations():
    declarations = (
        MaterialResource("z", "buffer"),
        MaterialResource("a", "texture"),
        MaterialResource("b", "uniform"),
    )
    layout = MaterialResourceLayout(declarations, descriptor_set=2, first_binding=4)
    assert [(d.name, b) for d, b in layout.entries] == [("a", 4), ("b", 6), ("z", 7)]
    assert "set=2,binding=5" in layout.source
    assert "isnan(index)||isinf(index)" in layout.source
    with pytest.raises(FrozenInstanceError):
        layout.first_binding = 9
    with pytest.raises(ValueError, match="Conflicting"):
        MaterialResourceLayout(
            (MaterialResource("x", "uniform"), MaterialResource("x", "buffer"))
        )
    with pytest.raises(ValueError, match="cover"):
        layout.validate([program()])


def test_shader_prepares_without_device(monkeypatch):
    from ordinarylight.runtime import VulkanRuntime

    def forbidden(*args, **kwargs):
        raise AssertionError("Shader preparation must not create a runtime")

    monkeypatch.setattr(VulkanRuntime, "__init__", forbidden)
    material = program()
    layout = MaterialResourceLayout.from_programs([material])
    options = dict(
        attribute_layout=ol.VertexAttributeLayout(()),
        attribute_binding=16,
        material_resources=layout,
    )
    source = wavefront_material_shader_source(
        "wavefront_shade.comp", [material], **options
    )
    assert layout.source in source
    assert "set=1,binding=0,std140" in source
    compiler = find_glsl_compiler()
    if compiler is not None:
        binary = compile_wavefront_material_shader(
            "wavefront_shade.comp", [material], compiler=compiler, **options
        )
        assert binary[:4] == b"\x03\x02#\x07"
