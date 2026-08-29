"""Built-in raster shaders authored with Ordinary Shade."""

import ordinaryshade as osh


@osh.structure
class SceneVertexOutput:
    position: osh.builtin(osh.vec4, "position")
    color: osh.location(osh.vec4, 0)


@osh.vertex
def scene_vertex(
    position: osh.location(osh.vec4, 0),
    color: osh.location(osh.vec4, 1),
) -> SceneVertexOutput:
    return SceneVertexOutput(position, color)


@osh.fragment
def scene_fragment(color: osh.location(osh.vec4, 0)) -> osh.location(osh.vec4, 0):
    return color


__all__ = ["SceneVertexOutput", "scene_fragment", "scene_vertex"]
