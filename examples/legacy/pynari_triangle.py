"""Legacy minimal PyNARI triangle render."""

import numpy as np
import pynari as anari

width, height = 1024, 768
device = anari.newDevice("default")

# Camera
camera = device.newCamera("perspective")
camera.setParameter("aspect", anari.FLOAT32, width / height)
camera.setParameter("position", anari.FLOAT32_VEC3, [0.0, 0.0, 0.0])
camera.setParameter("direction", anari.FLOAT32_VEC3, [0.0, 0.0, 1.0])
camera.setParameter("up", anari.FLOAT32_VEC3, [0.0, 1.0, 0.0])
camera.commitParameters()

# Triangle geometry
vertices = np.array([
    [-1.0, -1.0, 3.0],
    [-1.0,  1.0, 3.0],
    [ 1.0, -1.0, 3.0],
], dtype=np.float32)

mesh = device.newGeometry("triangle")
vertex_array = device.newArray1D(anari.FLOAT32_VEC3, vertices)
mesh.setParameter("vertex.position", anari.ARRAY1D, vertex_array)
mesh.commitParameters()

# Geometry needs a material and surface
material = device.newMaterial("matte")
material.setParameter("color", anari.FLOAT32_VEC3, [0.8, 0.3, 0.2])
material.commitParameters()

surface = device.newSurface()
surface.setParameter("geometry", anari.GEOMETRY, mesh)
surface.setParameter("material", anari.MATERIAL, material)
surface.commitParameters()

# World: pass a real list, not ...
world = device.newWorld()
world.setParameterArray1D("surface", anari.SURFACE, [surface])
world.commitParameters()

# Renderer
renderer = device.newRenderer("default")
renderer.commitParameters()

# Frame
frame = device.newFrame()
frame.setParameter("size", anari.UINT32_VEC2, [width, height])
frame.setParameter(
    "channel.color",
    anari.DATA_TYPE,
    anari.UFIXED8_RGBA_SRGB,
)
frame.setParameter("renderer", anari.RENDERER, renderer)
frame.setParameter("camera", anari.CAMERA, camera)
frame.setParameter("world", anari.WORLD, world)
frame.commitParameters()

# Render
frame.render()
fb_color = np.array(frame.get("channel.color"))
print(fb_color.shape)
