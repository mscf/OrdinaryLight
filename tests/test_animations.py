import unittest

import numpy as np

import ordinarylight as ol


def triangle():
    return ((0, 0, 0), (1, 0, 0), (0, 1, 0)), ((0, 1, 2),)


class AnimationTests(unittest.TestCase):
    def test_linear_step_and_quaternion_sampling(self):
        target = object()
        linear = ol.AnimationTrack(
            target, "translation", (0, 2), ((0, 0, 0), (4, 2, 0))
        )
        step = ol.AnimationTrack(
            target, "scale", (0, 2), ((1, 1, 1), (2, 2, 2)), "step"
        )
        rotation = ol.AnimationTrack(
            target, "rotation", (0, 1),
            ((0, 0, 0, 1), (0, 1, 0, 0)),
        )
        np.testing.assert_allclose(linear.sample(1), (2, 1, 0))
        np.testing.assert_allclose(step.sample(1), (1, 1, 1))
        sampled = rotation.sample(0.5)
        self.assertAlmostEqual(float(np.linalg.norm(sampled)), 1.0, places=6)
        np.testing.assert_allclose(abs(sampled[[1, 3]]), np.sqrt(0.5), atol=1e-6)

    def test_cubic_spline_uses_gltf_tangent_layout(self):
        track = ol.AnimationTrack(
            object(), "translation", (0, 1),
            (
                (0, 0, 0), (0, 0, 0), (2, 0, 0),
                (2, 0, 0), (1, 0, 0), (0, 0, 0),
            ),
            "cubic",
        )
        np.testing.assert_allclose(track.sample(0.5), (0.5, 0, 0))

    def test_scene_applies_and_resets_transform_clip(self):
        scene = ol.Scene()
        vertices, indices = triangle()
        mesh = scene.add_mesh(
            vertices, indices, transform=ol.Transform.translation((1, 0, 0))
        )
        clip = scene.add_animation(ol.AnimationClip((
            ol.AnimationTrack(
                mesh, "translation", (0, 2), ((1, 0, 0), (5, 0, 0))
            ),
            ol.AnimationTrack(
                mesh, "scale", (0, 2), ((1, 1, 1), (2, 3, 4))
            ),
        ), name="move"))
        scene.apply_animation(clip, 1)
        np.testing.assert_allclose(mesh.transform.matrix[:3, 3], (3, 0, 0))
        np.testing.assert_allclose(
            np.linalg.norm(mesh.transform.matrix[:3, :3], axis=0),
            (1.5, 2, 2.5),
        )
        self.assertEqual(scene.snapshot()["animations"][0]["name"], "move")
        scene.reset_animation()
        np.testing.assert_allclose(mesh.transform.matrix, ol.Transform.translation(
            (1, 0, 0)
        ).matrix)

    def test_player_loops_and_updates_scene(self):
        scene = ol.Scene()
        vertices, indices = triangle()
        mesh = scene.add_mesh(vertices, indices)
        clip = ol.AnimationClip((ol.AnimationTrack(
            mesh, "translation", (0, 1), ((0, 0, 0), (2, 0, 0))
        ),))
        player = ol.AnimationPlayer(scene, clip, time=0.75)
        player.update(0.5)
        self.assertAlmostEqual(player.time, 1.25)
        np.testing.assert_allclose(mesh.transform.matrix[:3, 3], (0.5, 0, 0))

    def test_parent_node_animation_updates_descendant_instances(self):
        scene = ol.Scene()
        vertices, indices = triangle()
        resource = scene.create_mesh(vertices, indices)
        parent = scene.add_node(transform=ol.Transform.translation((1, 0, 0)))
        child = scene.add_node(
            parent=parent, transform=ol.Transform.translation((0, 2, 0))
        )
        instance = scene.add_instance(resource, node=child)
        clip = ol.AnimationClip((ol.AnimationTrack(
            parent, "translation", (0, 1), ((1, 0, 0), (3, 0, 0))
        ),), name="parent-motion")
        scene.add_animation(clip)
        scene.apply_animation(clip, 0.5)
        np.testing.assert_allclose(instance.transform.matrix[:3, 3], (2, 2, 0))
        self.assertEqual(scene.snapshot()["nodes"][1]["parent_id"], parent.id)
        scene.reset_animation()
        np.testing.assert_allclose(instance.transform.matrix[:3, 3], (1, 2, 0))

    def test_morph_weights_deform_absolute_bind_pose(self):
        scene = ol.Scene()
        vertices, indices = triangle()
        mesh = scene.add_mesh(vertices, indices, deformable=True)
        first = ol.MorphTarget(((0, 0, 0), (1, 0, 0), (0, 0, 0)))
        second = ol.MorphTarget(((0, 0, 0), (0, 0, 0), (0, 2, 0)))
        scene.bind_morph_targets(mesh, (first, second))
        scene.set_morph_weights(mesh, (0.5, 0.25))
        np.testing.assert_allclose(
            mesh.vertices, ((0, 0, 0), (1.5, 0, 0), (0, 1.5, 0))
        )
        scene.set_morph_weights(mesh, (0, 0))
        np.testing.assert_allclose(mesh.vertices, vertices)

    def test_node_weight_track_updates_bound_morph_instances(self):
        scene = ol.Scene()
        vertices, indices = triangle()
        resource = scene.create_mesh(vertices, indices, deformable=True)
        node = scene.add_node()
        mesh = scene.add_instance(resource, node=node)
        scene.bind_morph_targets(mesh, (ol.MorphTarget(
            ((0, 0, 0), (2, 0, 0), (0, 0, 0))
        ),))
        clip = ol.AnimationClip((ol.AnimationTrack(
            node, "weights", (0, 1), ((0,), (1,))
        ),))
        scene.apply_animation(clip, 0.5)
        np.testing.assert_allclose(mesh.vertices[1], (2, 0, 0))

    def test_skinning_composes_morph_joint_and_mesh_node_transforms(self):
        scene = ol.Scene()
        vertices, indices = triangle()
        resource = scene.create_mesh(vertices, indices, deformable=True)
        mesh_node = scene.add_node(
            transform=ol.Transform.translation((10, 0, 0)), name="mesh"
        )
        joint = scene.add_node(name="joint")
        mesh = scene.add_instance(resource, node=mesh_node)
        scene.bind_morph_targets(mesh, (ol.MorphTarget(
            ((0, 0, 0), (1, 0, 0), (0, 0, 0))
        ),), (0.5,))
        skin = ol.Skin((joint,), np.eye(4, dtype=np.float32)[None, ...])
        scene.bind_skin(
            mesh, skin,
            np.zeros((3, 4), np.uint16),
            np.column_stack((np.ones(3), np.zeros((3, 3)))),
            mesh_node=mesh_node,
        )
        clip = ol.AnimationClip((ol.AnimationTrack(
            joint, "translation", (0, 1), ((10, 0, 0), (10, 2, 0))
        ),))
        scene.apply_animation(clip, 1.0)
        # The mesh-local result includes the morph and joint delta, while its
        # node owns the final world placement.
        np.testing.assert_allclose(
            mesh.vertices, ((0, 2, 0), (1.5, 2, 0), (0, 3, 0)), atol=1e-6
        )
        np.testing.assert_allclose(
            mesh.world_vertices,
            ((10, 2, 0), (11.5, 2, 0), (10, 3, 0)), atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
