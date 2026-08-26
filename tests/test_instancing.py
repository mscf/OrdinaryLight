import unittest
import json

import numpy as np

import ordinarylight as ol


def triangle():
    return np.asarray(((-0.5, 0, 0), (0.5, 0, 0), (0, 1, 0)), np.float32)


INDICES = np.asarray(((0, 1, 2),), np.uint32)


class InstancingTests(unittest.TestCase):
    def test_shared_resource_has_independent_instances_and_stable_ids(self):
        scene = ol.Scene()
        default = ol.Material(base_color=(1, 0, 0))
        override = ol.Material(base_color=(0, 1, 0))
        resource = scene.create_mesh(triangle(), INDICES, default)
        first = scene.add_instance(resource)
        second = scene.add_instance(
            resource, transform=ol.Transform.translation((2, 0, 0)),
            material=override,
        )

        self.assertIsInstance(resource, ol.MeshResource)
        self.assertIsInstance(first, ol.Instance)
        self.assertIs(first.resource, second.resource)
        self.assertIs(first.vertices, resource.vertices)
        self.assertIs(second.normals, resource.normals)
        self.assertIs(first.attributes, resource.attributes)
        self.assertNotEqual(first.id, second.id)
        self.assertEqual(tuple(scene.triangle_instance_ids()), (first.id, second.id))
        self.assertEqual(tuple(scene.triangle_object_ids()), (first.id, second.id))
        self.assertEqual(
            tuple(scene.triangle_material_ids()),
            (scene.material_id(default), scene.material_id(override)),
        )

    def test_visibility_excludes_instance_without_recycling_id(self):
        scene = ol.Scene()
        resource = scene.create_mesh(triangle(), INDICES)
        first = scene.add_instance(resource)
        second = scene.add_instance(resource)
        scene.update_instance(first, visible=False)

        self.assertEqual(len(scene.triangles()[0]), 1)
        self.assertEqual(tuple(scene.triangle_object_ids()), (second.id,))
        scene.update_instance(first, visible=True)
        self.assertEqual(tuple(scene.triangle_object_ids()), (first.id, second.id))

    def test_batch_update_is_atomic_and_changes_revision_once(self):
        scene = ol.Scene()
        resource = scene.create_mesh(triangle(), INDICES)
        first = scene.add_instance(resource)
        second = scene.add_instance(resource)
        revision = scene.revision
        transform_revision = scene.transform_revision
        shading_revision = scene.shading_revision
        blue = ol.Material(base_color=(0, 0, 1))

        updated = scene.update_instances((
            (first, {"transform": ol.Transform.translation((1, 0, 0))}),
            (second, {"material": blue}),
        ))
        self.assertEqual(updated, (first, second))
        self.assertEqual(scene.revision, revision + 1)
        self.assertEqual(scene.transform_revision, transform_revision + 1)
        self.assertEqual(scene.shading_revision, shading_revision + 1)

        snapshot = first.transform
        with self.assertRaises(TypeError):
            scene.update_instances((
                (first, {"transform": ol.Transform.translation((4, 0, 0))}),
                (second, {"visible": "yes"}),
            ))
        np.testing.assert_array_equal(first.transform.matrix, snapshot.matrix)

    def test_legacy_geometry_update_detaches_one_shared_instance(self):
        scene = ol.Scene()
        resource = scene.create_mesh(triangle(), INDICES)
        first = scene.add_instance(resource)
        second = scene.add_instance(resource)
        changed = triangle().copy()
        changed[2, 1] = 2.0

        scene.update_mesh(first, vertices=changed)
        self.assertIsNot(first.resource, resource)
        self.assertIs(second.resource, resource)
        self.assertEqual(first.vertices[2, 1], 2.0)
        self.assertEqual(second.vertices[2, 1], 1.0)

    def test_shared_resource_update_reaches_all_instances(self):
        scene = ol.Scene()
        resource = scene.create_mesh(triangle(), INDICES, deformable=True)
        first = scene.add_instance(resource)
        second = scene.add_instance(resource)
        changed = triangle().copy()
        changed[:, 0] *= 2.0
        scene.update_mesh_resource(resource, vertices=changed)
        np.testing.assert_array_equal(first.vertices, changed)
        np.testing.assert_array_equal(second.vertices, changed)

    def test_statistics_measure_shared_geometry_without_expansion(self):
        scene = ol.Scene()
        resource = scene.create_mesh(triangle(), INDICES)
        for index in range(8):
            scene.add_instance(
                resource,
                transform=ol.Transform.translation((index, 0, 0)),
            )
        statistics = scene.instancing_statistics()
        self.assertEqual(statistics["instance_count"], 8)
        self.assertEqual(statistics["geometry_count"], 1)
        self.assertEqual(statistics["shared_blas_savings"], 7)
        self.assertEqual(statistics["expanded_triangle_count"], 8)
        self.assertEqual(statistics["unique_triangle_count"], 1)
        self.assertGreater(
            statistics["expanded_geometry_bytes"],
            statistics["unique_geometry_bytes"],
        )

    def test_resource_cannot_be_removed_while_instances_reference_it(self):
        scene = ol.Scene()
        resource = scene.create_mesh(triangle(), INDICES)
        instance = scene.add_instance(resource)
        with self.assertRaises(ValueError):
            scene.remove_mesh_resource(resource)
        scene.remove_instance(instance)
        self.assertIs(scene.remove_mesh_resource(resource.id), resource)

    def test_column_oriented_creation_and_updates_are_single_revisions(self):
        scene = ol.Scene()
        resource = scene.create_mesh(
            triangle(), INDICES, name="triangle",
            metadata={"source": "test"},
        )
        transforms = np.stack([
            ol.Transform.translation((index, 0, 0)).matrix
            for index in range(4)
        ])
        revision = scene.revision
        instances = scene.add_instances(
            resource, transforms,
            names=[f"item-{index}" for index in range(4)],
            metadata=[{"index": np.int64(index)} for index in range(4)],
        )
        self.assertEqual(scene.revision, revision + 1)
        self.assertEqual(tuple(item.name for item in instances), (
            "item-0", "item-1", "item-2", "item-3",
        ))

        revision = scene.revision
        transform_revision = scene.transform_revision
        shifted = transforms.copy()
        shifted[:, 2, 3] = 2.0
        blue = ol.Material(base_color=(0, 0, 1))
        updated = scene.update_instance_batch(
            [item.id for item in instances],
            transforms=shifted, materials=blue,
        )
        self.assertEqual(updated, instances)
        self.assertEqual(scene.revision, revision + 1)
        self.assertEqual(scene.transform_revision, transform_revision + 1)
        self.assertTrue(all(item.material is blue for item in instances))

        snapshot = scene.snapshot()
        self.assertEqual(snapshot["mesh_resources"][0]["name"], "triangle")
        self.assertEqual(snapshot["instances"][2]["metadata"]["index"], 2)
        json.dumps(snapshot)

    def test_bulk_creation_validates_before_attaching_anything(self):
        scene = ol.Scene()
        resource = scene.create_mesh(triangle(), INDICES)
        with self.assertRaises(TypeError):
            scene.add_instances(
                resource,
                [ol.Transform(), ol.Transform.translation((1, 0, 0))],
                metadata=[{}, "not metadata"],
            )
        self.assertEqual(scene.instance_count, 0)


if __name__ == "__main__":
    unittest.main()
