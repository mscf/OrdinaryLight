"""Update stable scene resources without rebuilding the scene object."""

import ordinarylight as ol
from examples.common import scene_and_camera


def main():
    scene, camera = scene_and_camera()
    mesh = scene.meshes[0]
    scene.update_mesh(mesh, transform=ol.Transform.translation((0.5, 0, 0)))
    with ol.Renderer(
        backend=ol.backends.ReferenceBackend(samples_per_pixel=1)
    ) as renderer:
        image = renderer.render(scene, camera, (64, 36))
    print(image.shape, mesh.id, scene.revision)


if __name__ == "__main__":
    main()
