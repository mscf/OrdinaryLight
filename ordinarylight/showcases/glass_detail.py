"""High-contrast glass fixtures for inspecting denoising during motion."""

import ordinarylight as ol

from .materials import diffuse, fresnel_glass, quad, sphere


def build_glass_detail():
    scene = ol.Scene()
    vertices, indices = sphere((0, 1, 0), 1.0)
    scene.add_mesh(vertices, indices, ol.Material(
        base_color=(0.97, 0.99, 1), transmission=1, roughness=0,
        ior=1.52, program=fresnel_glass,
    ), name="glass-sphere")
    for i in range(32):
        x = -4 + i * 0.25
        vertices, indices = quad((x, -2, 2), (x + .25, -2, 2),
                                 (x + .25, 4, 2), (x, 4, 2))
        color = ((2.0, .3, .05) if i % 4 == 0 else (1.5, 1.5, 1.5)
                 if i % 2 == 0 else (.015, .015, .015))
        scene.add_mesh(vertices, indices, ol.Material(
            emission=color, emission_two_sided=True, program=diffuse,
        ), name=f"target-bar-{i}")
    return scene


def _animated_detail(target):
    scene = build_glass_detail()
    meshes = [mesh for mesh in scene.meshes
              if (mesh.name == "glass-sphere") == (target == "glass")]
    # Move for one second, hold for two, then reverse and hold again.
    scene.add_animation(ol.AnimationClip(tuple(
        ol.AnimationTrack(mesh, "translation", (0, 1, 3, 4, 6),
                          ((-.3, 0, 0), (.3, 0, 0), (.3, 0, 0),
                           (-.3, 0, 0), (-.3, 0, 0)))
        for mesh in meshes
    ), name=f"glass-detail-{target}"))
    scene.apply_animation(scene.animations[0], 0)
    return scene


def build_moving_glass_detail():
    return _animated_detail("glass")


def build_moving_glass_target():
    return _animated_detail("target")


def animate_glass_detail(scene, time):
    scene.apply_animation(scene.animations[0], time, loop=True)
