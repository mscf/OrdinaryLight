import unittest

import numpy as np

import ordinarylight as ol
from ordinarylight.showcases.primitives import flat_shaded_pyramid


TRIANGLE = np.asarray(((-0.5, 0, 0), (0.5, 0, 0), (0, 1, 0)), np.float32)
INDICES = np.asarray(((0, 1, 2),), np.uint32)


class PrimitiveBatchTests(unittest.TestCase):
    def test_showcase_pyramid_is_flat_shaded_and_outward_wound(self):
        vertices, indices, normals = flat_shaded_pyramid()
        interior = np.asarray((0.0, 0.2, 0.0), np.float32)
        for triangle_index, triangle in enumerate(vertices[indices]):
            face_normal = normals[indices[triangle_index, 0]]
            np.testing.assert_allclose(
                normals[indices[triangle_index]],
                np.repeat(face_normal[None], 3, axis=0),
            )
            self.assertGreater(
                float(np.dot(face_normal, triangle.mean(axis=0) - interior)),
                0.0,
            )

    def test_points_share_geometry_and_preserve_world_centers(self):
        scene = ol.Scene()
        positions = np.asarray(((0, 1, 2), (3, 4, 5), (-2, 0, 1)), np.float32)
        batch = scene.add_points(positions, radii=(0.1, 0.2, 0.3))

        self.assertIsInstance(batch, ol.PointBatch)
        self.assertEqual(len(batch.instances), 3)
        self.assertEqual(scene.visible_geometry_count, 1)
        self.assertEqual(scene.instancing_statistics()["shared_blas_savings"], 2)
        for instance, center in zip(batch.instances, positions):
            np.testing.assert_allclose(
                (instance.world_vertices.min(axis=0)
                 + instance.world_vertices.max(axis=0)) * 0.5,
                center, atol=1e-6,
            )

    def test_lines_reach_both_endpoints_and_update_atomically(self):
        scene = ol.Scene()
        starts = np.asarray(((0, 0, 0), (1, 1, 1)), np.float32)
        ends = np.asarray(((0, 2, 0), (4, 1, 1)), np.float32)
        batch = scene.add_lines(starts, ends, radii=0.1)
        ids = batch.ids
        for instance, start, end in zip(batch.instances, starts, ends):
            vertices = instance.world_vertices
            self.assertLess(np.linalg.norm(vertices - start, axis=1).min(), 1e-6)
            self.assertLess(np.linalg.norm(vertices - end, axis=1).min(), 1e-6)

        revision = scene.revision
        transform_revision = scene.transform_revision
        moved = ends + (0, 0, 2)
        batch.update(ends=moved, radii=(0.15, 0.2))
        self.assertEqual(batch.ids, ids)
        self.assertEqual(scene.revision, revision + 1)
        self.assertEqual(scene.transform_revision, transform_revision + 1)
        np.testing.assert_array_equal(batch.ends, moved)

    def test_arbitrary_glyphs_use_existing_resource(self):
        scene = ol.Scene()
        resource = scene.create_mesh(TRIANGLE, INDICES, name="arrow")
        transforms = np.stack((
            ol.Transform.translation((0, 0, 0)).matrix,
            ol.Transform.translation((2, 0, 0)).matrix,
        ))
        batch = scene.add_glyphs(resource, transforms)
        self.assertIsInstance(batch, ol.GlyphBatch)
        self.assertIs(batch.resource, resource)
        self.assertFalse(batch.owns_resource)
        self.assertEqual(scene.visible_geometry_count, 1)

        batch.remove()
        self.assertEqual(scene.instance_count, 0)
        self.assertIn(resource, scene.mesh_resources)

    def test_generated_resource_is_removed_with_batch(self):
        scene = ol.Scene()
        batch = ol.add_points(scene, ((0, 0, 0),))
        resource = batch.resource
        batch.remove()
        self.assertEqual(scene.instance_count, 0)
        self.assertNotIn(resource, scene.mesh_resources)

    def test_lowered_primitives_retain_named_output_identity(self):
        scene = ol.Scene()
        points = scene.add_points(((0, 0, 0), (1, 0, 0)), radii=0.1)
        lines = scene.add_lines(((0, 0, 0),), ((0, 1, 0),), radii=0.05)
        packed = scene.triangle_instance_ids()
        point_triangles = len(points.resource.indices)
        line_triangles = len(lines.resource.indices)
        self.assertTrue(np.all(packed[:point_triangles] == points.ids[0]))
        self.assertTrue(np.all(
            packed[point_triangles:2 * point_triangles] == points.ids[1]
        ))
        self.assertTrue(np.all(packed[-line_triangles:] == lines.ids[0]))

    def test_invalid_line_update_does_not_mutate_batch_or_scene(self):
        scene = ol.Scene()
        batch = scene.add_lines(((0, 0, 0),), ((0, 1, 0),))
        revision = scene.revision
        before = batch.ends.copy()
        with self.assertRaises(ValueError):
            batch.update(ends=((0, 0, 0),))
        self.assertEqual(scene.revision, revision)
        np.testing.assert_array_equal(batch.ends, before)


if __name__ == "__main__":
    unittest.main()
