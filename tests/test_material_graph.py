"""Shared graph compilation and validation independent of a GPU backend."""

import pytest
from ordinarylight.materials import MaterialGraph, MaterialNode


def test_graph_compiles_to_camera_contract():
    graph = MaterialGraph(
        {
            "color": MaterialNode("constant", value=(0.2, 0.4, 0.6), type="vec3"),
            "gain": MaterialNode("constant", value=2),
            "emission": MaterialNode("multiply", ("color", "gain"), type="vec3"),
        },
        {"emission": "emission"},
    )
    import ordinarylight as ol

    material = ol.Material(program=graph.compile())
    assert "result.emission = (vec3(0.2, 0.4, 0.6) * 2.0)" in material.program.glsl()


def test_graph_rejects_invalid_connections():
    with pytest.raises(ValueError, match="cycle"):
        MaterialGraph({"a": MaterialNode("add", ("a", "a"))}, {})
    with pytest.raises(ValueError, match="Missing"):
        MaterialGraph({}, {"emission": "absent"})
    with pytest.raises(TypeError, match="must be vec3"):
        MaterialGraph({"a": MaterialNode("constant", value=1)}, {"emission": "a"})
    with pytest.raises(ValueError, match="Unknown material input"):
        MaterialGraph({"a": MaterialNode("input", value="bad")}, {})
