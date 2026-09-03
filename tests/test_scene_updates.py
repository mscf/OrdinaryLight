import unittest

import numpy as np

import ordinarylight as ol


def triangle(offset=0.0):
    return np.asarray(
        ((-1 + offset, 0, 0), (1 + offset, 0, 0), (offset, 1, 0)),
        np.float32,
    )


class SceneUpdateTests(unittest.TestCase):
    def test_volume_range_update_preserves_resident_scalar_revision(self):
        scene = ol.Scene()
        volume = scene.add_volume(
            np.arange(8, dtype=np.float32).reshape(2, 2, 2),
            value_range=(0.0, 7.0),
        )
        data_revision = volume.data_revision
        shading_revision = scene.shading_revision

        scene.update_volume(volume, value_range=(1.0, 6.0))

        self.assertEqual(volume.data_revision, data_revision)
        self.assertEqual(volume.value_range, (1.0, 6.0))
        self.assertGreater(scene.shading_revision, shading_revision)

        scene.update_volume(
            volume, value_range=(1.0, 6.0), value_mapping="symlog",
            linear_threshold=0.25,
        )
        self.assertEqual(volume.data_revision, data_revision)
        self.assertEqual(volume.value_mapping, "symlog")
        self.assertEqual(volume.linear_threshold, 0.25)

    def test_volume_slice_controls_are_shading_only(self):
        scene = ol.Scene()
        volume = scene.add_volume(np.ones((2, 3, 4), np.float32))
        geometry_revision = scene.geometry_revision
        data_revision = volume.data_revision

        scene.update_volume(
            volume, render_mode="slice", slice_axis="x", slice_position=0.25,
        )

        self.assertEqual(scene.geometry_revision, geometry_revision)
        self.assertEqual(volume.data_revision, data_revision)
        self.assertEqual(volume.render_mode, "slice")
        self.assertEqual(volume.slice_axis, "x")
        self.assertEqual(volume.slice_position, 0.25)

    def test_volume_isosurface_controls_are_shading_only(self):
        scene = ol.Scene()
        volume = scene.add_volume(np.ones((2, 3, 4), np.float32))
        geometry_revision = scene.geometry_revision
        data_revision = volume.data_revision

        scene.update_volume(
            volume, render_mode="isosurface", isovalue=0.75,
        )

        self.assertEqual(scene.geometry_revision, geometry_revision)
        self.assertEqual(volume.data_revision, data_revision)
        self.assertEqual(volume.render_mode, "isosurface")
        self.assertEqual(volume.isovalue, 0.75)

    def test_material_ids_are_stable_and_triangle_aligned(self):
        scene = ol.Scene()
        shared = ol.Material(base_color=(0.2, 0.3, 0.4))
        first = scene.add_mesh(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0)), ((0, 1, 2),), shared,
        )
        second = scene.add_mesh(
            ((0, 0, 1), (1, 0, 1), (0, 1, 1)), ((0, 1, 2),), shared,
        )
        shared_id = scene.material_id(shared)
        np.testing.assert_array_equal(
            scene.triangle_object_ids(), (first.id, second.id),
        )
        np.testing.assert_array_equal(
            scene.triangle_material_ids(), (shared_id, shared_id),
        )
        replacement = ol.Material(base_color=(0.8, 0.1, 0.1))
        scene.update_mesh(first, material=replacement)
        replacement_id = scene.material_id(replacement)
        self.assertNotEqual(shared_id, replacement_id)
        self.assertEqual(scene.material_id(shared), shared_id)
        np.testing.assert_array_equal(
            scene.triangle_material_ids(), (replacement_id, shared_id),
        )

    def test_deformable_is_an_explicit_stable_mesh_capability(self):
        scene = ol.Scene()
        mesh = scene.add_mesh(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0)), ((0, 1, 2),),
            deformable=True,
        )
        self.assertTrue(mesh.deformable)
        scene.update_mesh(
            mesh, vertices=((0, 0, 0), (1.1, 0, 0), (0, 1, 0)),
        )
        self.assertTrue(mesh.deformable)
        with self.assertRaises(TypeError):
            ol.Mesh(
                ((0, 0, 0), (1, 0, 0), (0, 1, 0)), ((0, 1, 2),),
                deformable=1,
            )

    def test_affine_transform_preserves_object_data_and_updates_world_data(self):
        scene = ol.Scene()
        transform = ol.Transform.translation((2, 3, 4)) @ ol.Transform.scale(2)
        mesh = scene.add_mesh(
            triangle(), ((0, 1, 2),), transform=transform,
        )
        np.testing.assert_array_equal(mesh.vertices, triangle())
        expected = triangle() * 2 + np.asarray((2, 3, 4), np.float32)
        np.testing.assert_allclose(mesh.world_vertices, expected)
        np.testing.assert_allclose(scene.triangles()[0][0], expected)
        bounds = scene.bounds()
        np.testing.assert_allclose(bounds[0], expected.min(axis=0))
        np.testing.assert_allclose(bounds[1], expected.max(axis=0))

    def test_transform_update_has_independent_revision(self):
        scene = ol.Scene()
        mesh = scene.add_mesh(triangle(), ((0, 1, 2),))
        geometry_revision = scene.geometry_revision
        shading_revision = scene.shading_revision
        transform_revision = scene.transform_revision
        scene.update_mesh(mesh, transform=ol.Transform.translation((1, 0, 0)))
        self.assertEqual(scene.geometry_revision, geometry_revision)
        self.assertEqual(scene.shading_revision, shading_revision)
        self.assertGreater(scene.transform_revision, transform_revision)
        np.testing.assert_allclose(mesh.world_vertices, triangle(1.0))

    def test_visibility_update_preserves_identity_and_invalidates_geometry(self):
        scene = ol.Scene()
        mesh = scene.add_mesh(triangle(), ((0, 1, 2),))
        geometry_revision = scene.geometry_revision
        self.assertIs(scene.update_mesh(mesh, visible=False), mesh)
        self.assertFalse(mesh.visible)
        self.assertNotIn(mesh, scene.visible_meshes)
        self.assertGreater(scene.geometry_revision, geometry_revision)

    def test_transform_validation_and_composition(self):
        with self.assertRaises(ValueError):
            ol.Transform(np.eye(3))
        with self.assertRaises(ValueError):
            ol.Transform.scale((1, 0, 1))
        rotation = ol.Transform.rotation((0, 0, 1), np.pi / 2)
        point = np.append((1, 0, 0), 1).astype(np.float32)
        np.testing.assert_allclose(
            rotation.matrix @ point, (0, 1, 0, 1), atol=1e-6,
        )

    def test_mesh_identity_and_revisions_survive_validated_updates(self):
        scene = ol.Scene()
        mesh = scene.add_mesh(triangle(), ((0, 1, 2),))
        resource_id = mesh.id
        initial = (scene.revision, scene.geometry_revision, scene.shading_revision)

        result = scene.update_mesh(mesh, vertices=triangle(2.0))
        self.assertIs(result, mesh)
        self.assertEqual(mesh.id, resource_id)
        self.assertGreater(scene.revision, initial[0])
        self.assertGreater(scene.geometry_revision, initial[1])
        self.assertGreater(scene.shading_revision, initial[2])
        np.testing.assert_array_equal(mesh.vertices, triangle(2.0))

        geometry_revision = scene.geometry_revision
        material = ol.Material(base_color=(0.2, 0.3, 0.4))
        scene.update_mesh(resource_id, material=material)
        self.assertIs(mesh.material, material)
        self.assertEqual(scene.geometry_revision, geometry_revision)

    def test_invalid_update_is_atomic(self):
        scene = ol.Scene()
        mesh = scene.add_mesh(triangle(), ((0, 1, 2),))
        revision = scene.revision
        original = mesh.vertices.copy()
        with self.assertRaises(ValueError):
            scene.update_mesh(mesh, vertices=np.zeros((2, 2), np.float32))
        self.assertEqual(scene.revision, revision)
        np.testing.assert_array_equal(mesh.vertices, original)

    def test_named_attribute_update_is_atomic_and_shading_only(self):
        scene = ol.Scene()
        mesh = scene.add_mesh(
            triangle(), ((0, 1, 2),), attributes={"value": (0, 1, 2)},
        )
        geometry_revision = scene.geometry_revision
        shading_revision = scene.shading_revision
        scene.update_mesh(mesh, attributes={"value": (3, 4, 5)})
        self.assertEqual(scene.geometry_revision, geometry_revision)
        self.assertGreater(scene.shading_revision, shading_revision)
        np.testing.assert_array_equal(
            mesh.vertex_attribute("value")[:, 0], (3, 4, 5)
        )
        revision = scene.revision
        with self.assertRaises(ValueError):
            scene.update_mesh(mesh, attributes={"value": (1, 2)})
        self.assertEqual(scene.revision, revision)
        np.testing.assert_array_equal(
            mesh.vertex_attribute("value")[:, 0], (3, 4, 5)
        )

    def test_remove_clear_and_light_updates_use_stable_ids(self):
        scene = ol.Scene()
        first = scene.add_mesh(triangle(), ((0, 1, 2),))
        second = scene.add_mesh(triangle(3.0), ((0, 1, 2),))
        light = scene.add_point_light((0, 2, 0))
        light_id = light.id
        self.assertIs(
            scene.update_point_light(light_id, intensity=5.0), light,
        )
        self.assertEqual(light.intensity, 5.0)
        self.assertIs(scene.remove_mesh(first.id), first)
        self.assertEqual(scene.meshes, [second])
        self.assertIs(scene.remove_point_light(light_id), light)
        revision = scene.revision
        scene.clear()
        self.assertFalse(scene.meshes)
        self.assertGreater(scene.revision, revision)
        new_mesh = scene.add_mesh(triangle(), ((0, 1, 2),))
        self.assertGreater(new_mesh.id, second.id)

    def test_resource_cannot_be_attached_to_multiple_scenes(self):
        mesh = ol.Mesh(triangle(), ((0, 1, 2),))
        ol.Scene(meshes=[mesh])
        with self.assertRaises(ValueError):
            ol.Scene(meshes=[mesh])


if __name__ == "__main__":
    unittest.main()
