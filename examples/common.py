"""Small scene shared by supported Ordinary Light examples."""

import ordinarylight as ol


def scene_and_camera():
    scene = ol.Scene()
    scene.add_mesh(
        ((-2, -1, 0), (2, -1, 0), (0, 2, 0)),
        ((0, 1, 2),),
        ol.Material(base_color=(0.8, 0.2, 0.08)),
    )
    scene.add_point_light((2, 3, -2), intensity=25.0)
    camera = ol.PerspectiveCamera((0, 0, -4), (0, 0, 0))
    return scene, camera
