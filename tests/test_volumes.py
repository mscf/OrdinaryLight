import json
import unittest

import numpy as np

import ordinarylight as ol
from ordinarylight.showcases.volumes import build_volume_showcase
from ordinarylight.showcases.multivolume import build_multivolume_showcase
from ordinarylight.showcases.volume_scattering import build_volume_scattering_showcase
from ordinarylight.vulkan_rt import SceneTlasInstance
from ordinarylight.volume import (
    VOLUME_HEADER_DTYPE, integrate_volume, integrate_volumes,
    intersect_unit_boxes, pack_volumes, phase_function, sample_trilinear,
    volume_brick_occupancy, volume_empty_space_statistics,
)


def ramp_volume():
    z, y, x = np.mgrid[0:4, 0:4, 0:4]
    return (x + y + z).astype(np.float32)


class VolumeResourceTests(unittest.TestCase):
    def test_brick_occupancy_is_sparse_conservative_and_packable(self):
        data = np.zeros((17, 17, 17), np.float32)
        data[2:5, 2:5, 2:5] = 1.0
        transfer = ol.Texture1D(((0, 0, 0, 0), (1, 1, 1, 0.5)))
        scene = ol.Scene()
        volume = scene.add_volume(
            data, ol.VolumeMaterial(transfer), name="sparse",
        )
        bricks = volume_brick_occupancy(volume)
        self.assertEqual(bricks.shape, (2, 2, 2))
        self.assertGreater(np.count_nonzero(bricks), 0)
        self.assertLess(np.count_nonzero(bricks), bricks.size)
        statistics = volume_empty_space_statistics((volume,))
        self.assertGreater(statistics["empty_fraction"], 0.0)
        headers, scalars, _transfers = pack_volumes((volume,))
        np.testing.assert_array_equal(
            headers[0]["acceleration_parameters"][1:], (2, 2, 2),
        )
        offset = int(headers[0]["acceleration_parameters"][0])
        np.testing.assert_array_equal(scalars[offset:offset + 8], bricks.ravel())
        disabled_headers, disabled_scalars, _ = pack_volumes(
            (volume,), empty_space_skipping=False,
        )
        np.testing.assert_array_equal(
            disabled_headers[0]["acceleration_parameters"], 0,
        )
        self.assertEqual(disabled_scalars.size, data.size)

    def test_brick_occupancy_checks_transfer_knots_inside_scalar_range(self):
        data = np.linspace(0.2, 0.8, 9 ** 3, dtype=np.float32).reshape(9, 9, 9)
        transfer = ol.Texture1D((
            (0, 0, 0, 0), (1, 1, 1, 0.5), (0, 0, 0, 0),
        ))
        volume = ol.Volume(data, ol.VolumeMaterial(transfer))
        self.assertEqual(float(volume_brick_occupancy(volume)[0, 0, 0]), 1.0)

    def test_scattering_showcase_is_nonemissive_and_light_driven(self):
        scene = build_volume_scattering_showcase(20)
        self.assertEqual(len(scene.visible_volumes), 1)
        material = scene.visible_volumes[0].material
        self.assertEqual(material.emission_scale, 0.0)
        self.assertGreater(material.scattering_scale, 0.0)
        self.assertEqual(material.phase_function, "henyey_greenstein")
        self.assertGreater(material.anisotropy, 0.0)
        self.assertGreaterEqual(len(scene.lights), 2)

    def test_volume_scattering_api_is_validated_and_snapshotted(self):
        transfer = ol.Texture1D(((0, 0, 0, 0.2), (0, 0, 0, 0.2)))
        material = ol.VolumeMaterial(
            transfer, scattering_scale=0.7,
            scattering_color=(0.2, 0.5, 1.0),
            phase_function="henyey_greenstein", anisotropy=0.65,
            scattering_albedo=(0.3, 0.6, 0.9), scattering_orders=4,
        )
        scene = ol.Scene()
        scene.add_volume(np.ones((2, 2, 2), np.float32), material)
        snapshot = scene.snapshot()["volumes"][0]["material"]
        self.assertEqual(snapshot["phase_function"], "henyey_greenstein")
        np.testing.assert_allclose(
            snapshot["scattering_color"], [0.2, 0.5, 1.0], atol=1e-7,
        )
        self.assertAlmostEqual(snapshot["anisotropy"], 0.65)
        np.testing.assert_allclose(
            snapshot["scattering_albedo"], [0.3, 0.6, 0.9], atol=1e-7,
        )
        self.assertEqual(snapshot["scattering_orders"], 4)
        with self.assertRaises(ValueError):
            ol.VolumeMaterial(phase_function="unsupported")
        with self.assertRaises(ValueError):
            ol.VolumeMaterial(anisotropy=1.0)
        with self.assertRaises(ValueError):
            ol.VolumeMaterial(scattering_albedo=(1.1, 0.5, 0.5))
        with self.assertRaises(ValueError):
            ol.VolumeMaterial(scattering_orders=0)
        with self.assertRaises(ValueError):
            ol.VolumeMaterial(scattering_orders=9)

    def test_multiple_scattering_is_bounded_and_optical_depth_dependent(self):
        light = ol.PointLight((0.5, 0.5, 1.8), intensity=20.0)
        origins = np.asarray(((0.5, 0.5, -1.0),), np.float32)
        directions = np.asarray(((0.0, 0.0, 1.0),), np.float32)

        def render(opacity, orders):
            transfer = ol.Texture1D(((0, 0, 0, opacity),) * 2)
            volume = ol.Volume(
                np.ones((2, 2, 2), np.float32),
                ol.VolumeMaterial(
                    transfer, emission_scale=0.0, step_size=0.05,
                    scattering_scale=1.0, scattering_orders=orders,
                    scattering_albedo=(0.9, 0.9, 0.9),
                ),
            )
            entries, exits = intersect_unit_boxes(origins, directions, (volume,))
            return integrate_volume(
                volume, origins, directions, entries[0], exits[0],
                lights=(light,),
            )[0]

        thin_single = render(0.005, 1)
        thin_multiple = render(0.005, 4)
        thick_single = render(0.08, 1)
        thick_multiple = render(0.08, 4)
        thin_ratio = float(np.mean(thin_multiple) / np.mean(thin_single))
        thick_ratio = float(np.mean(thick_multiple) / np.mean(thick_single))
        self.assertGreater(thin_ratio, 1.0)
        self.assertGreater(thick_ratio, thin_ratio + 0.5)
        self.assertLess(thick_ratio, 4.0)

    def test_scattering_order_one_is_the_existing_direct_estimator(self):
        transfer = ol.Texture1D(((0, 0, 0, 0.04),) * 2)
        common = dict(
            emission_scale=0.0, step_size=0.05, scattering_scale=1.0,
            scattering_color=(0.4, 0.7, 1.0),
            phase_function="henyey_greenstein", anisotropy=0.4,
        )
        default_volume = ol.Volume(
            np.ones((2, 2, 2), np.float32),
            ol.VolumeMaterial(transfer, **common),
        )
        explicit_volume = ol.Volume(
            np.ones((2, 2, 2), np.float32),
            ol.VolumeMaterial(
                transfer, scattering_orders=1,
                scattering_albedo=(0.0, 1.0, 0.3), **common,
            ),
        )
        origins = np.asarray(((0.5, 0.5, -1.0),), np.float32)
        directions = np.asarray(((0.0, 0.0, 1.0),), np.float32)
        light = ol.PointLight((0.5, 0.5, 1.8), intensity=20.0)

        def render(volume):
            entries, exits = intersect_unit_boxes(origins, directions, (volume,))
            return integrate_volume(
                volume, origins, directions, entries[0], exits[0],
                lights=(light,),
            )

        default_radiance, default_transmittance = render(default_volume)
        explicit_radiance, explicit_transmittance = render(explicit_volume)
        np.testing.assert_array_equal(default_radiance, explicit_radiance)
        np.testing.assert_array_equal(
            default_transmittance, explicit_transmittance,
        )

    def test_phase_functions_are_normalized_and_directional(self):
        cosine = np.linspace(-1.0, 1.0, 20001, dtype=np.float32)
        isotropic = phase_function(cosine)
        forward = phase_function(cosine, 0.7, "henyey_greenstein")
        # Integrating over solid angle reduces to 2 pi times the cosine integral.
        self.assertAlmostEqual(
            float(2.0 * np.pi * np.trapezoid(isotropic, cosine)), 1.0,
            places=4,
        )
        self.assertAlmostEqual(
            float(2.0 * np.pi * np.trapezoid(forward, cosine)), 1.0,
            places=3,
        )
        self.assertGreater(float(forward[-1]), float(forward[0]) * 100.0)

    def test_point_light_single_scattering_adds_radiance(self):
        transfer = ol.Texture1D(((0, 0, 0, 0.08), (0, 0, 0, 0.08)))
        volume = ol.Volume(
            np.ones((2, 2, 2), np.float32),
            ol.VolumeMaterial(
                transfer, emission_scale=0.0, step_size=0.05,
                scattering_scale=1.0, scattering_color=(0.2, 0.6, 1.0),
            ),
        )
        light = ol.PointLight((0.5, 0.5, -0.5), intensity=20.0)
        origins = np.asarray(((0.5, 0.5, -1.0),), np.float32)
        directions = np.asarray(((0.0, 0.0, 1.0),), np.float32)
        entries, exits = intersect_unit_boxes(origins, directions, (volume,))
        dark, _ = integrate_volume(
            volume, origins, directions, entries[0], exits[0],
        )
        lit, _ = integrate_volume(
            volume, origins, directions, entries[0], exits[0], lights=(light,),
        )
        np.testing.assert_array_equal(dark, 0.0)
        self.assertGreater(float(lit[0, 2]), float(lit[0, 1]))
        self.assertGreater(float(lit[0, 1]), float(lit[0, 0]))

    def test_multivolume_showcase_has_transparent_overlapping_media(self):
        scene = build_multivolume_showcase(20)
        self.assertEqual(len(scene.visible_volumes), 3)
        bounds = []
        for volume in scene.visible_volumes:
            density = volume.data
            boundary = np.concatenate((
                density[0].ravel(), density[-1].ravel(),
                density[:, 0].ravel(), density[:, -1].ravel(),
                density[:, :, 0].ravel(), density[:, :, -1].ravel(),
            ))
            np.testing.assert_array_equal(boundary, 0.0)
            matrix = volume.transform.matrix
            lower = (matrix @ np.asarray((0, 0, 0, 1), np.float32))[:3]
            upper = (matrix @ np.asarray((1, 1, 1, 1), np.float32))[:3]
            bounds.append((np.minimum(lower, upper), np.maximum(lower, upper)))
        for first, second in zip(bounds, bounds[1:]):
            overlap = np.minimum(first[1], second[1]) - np.maximum(
                first[0], second[0]
            )
            self.assertTrue(np.all(overlap > 0.0))

    def test_showcase_density_is_transparent_at_its_grid_boundary(self):
        density = build_volume_showcase(24).volumes[0].data
        boundary = np.concatenate((
            density[0].ravel(), density[-1].ravel(),
            density[:, 0].ravel(), density[:, -1].ravel(),
            density[:, :, 0].ravel(), density[:, :, -1].ravel(),
        ))
        np.testing.assert_array_equal(boundary, 0.0)

    def test_volume_proxy_uses_a_distinct_tlas_visibility_mask(self):
        scene = ol.Scene()
        surface = scene.add_mesh(
            ((0, 0, 0), (1, 0, 0), (0, 1, 0)), ((0, 1, 2),)
        )
        scene.add_volume(np.ones((2, 2, 2), np.float32))
        proxy = scene.render_meshes[-1]
        surface_instance = SceneTlasInstance(surface, None, 0)
        volume_instance = SceneTlasInstance(proxy, None, 1)
        self.assertEqual(surface_instance.visibility_mask, 0x01)
        self.assertEqual(volume_instance.visibility_mask, 0x02)

    def test_volume_is_immutable_normalized_and_snapshot_safe(self):
        source = ramp_volume()
        scene = ol.Scene()
        volume = scene.add_volume(source, name="density", metadata={"unit": "a.u."})
        source[:] = -1
        self.assertFalse(volume.data.flags.writeable)
        self.assertEqual(volume.value_range, (0.0, 9.0))
        self.assertAlmostEqual(float(volume.normalized_data[-1, -1, -1]), 1.0)
        snapshot = scene.snapshot()
        self.assertEqual(snapshot["volumes"][0]["shape"], [4, 4, 4])
        self.assertNotIn("data", snapshot["volumes"][0])
        json.dumps(snapshot)

    def test_update_is_atomic_and_preserves_identity(self):
        scene = ol.Scene()
        volume = scene.add_volume(ramp_volume())
        revision = scene.revision
        volume_id = volume.id
        scene.update_volume(
            volume, transform=ol.Transform.translation((1, 2, 3)),
            material=ol.VolumeMaterial(density_scale=2.0),
        )
        self.assertEqual(volume.id, volume_id)
        self.assertGreater(scene.revision, revision)
        np.testing.assert_allclose(volume.transform.matrix[:3, 3], (1, 2, 3))
        revision = scene.revision
        before = volume.data.copy()
        with self.assertRaises(ValueError):
            scene.update_volume(volume, data=np.zeros((2, 2), np.float32))
        self.assertEqual(scene.revision, revision)
        np.testing.assert_array_equal(volume.data, before)

    def test_sampling_intersection_and_gpu_pack(self):
        scene = ol.Scene()
        volume = scene.add_volume(
            ramp_volume(),
            transform=ol.Transform.translation((-0.5, -0.5, 0.0)),
        )
        sampled = sample_trilinear(volume, ((0.0, 0.0, 0.5),))
        np.testing.assert_allclose(sampled, (0.5,), atol=1e-6)
        origins = np.asarray(((0, 0, -2), (3, 0, -2)), np.float32)
        directions = np.asarray(((0, 0, 1), (0, 0, 1)), np.float32)
        entry, exit = intersect_unit_boxes(origins, directions, (volume,))
        np.testing.assert_allclose(entry[0, 0], 2.0)
        np.testing.assert_allclose(exit[0, 0], 3.0)
        self.assertTrue(np.isinf(entry[0, 1]))
        headers, scalars, transfers = pack_volumes((volume,))
        self.assertEqual(headers.dtype, VOLUME_HEADER_DTYPE)
        np.testing.assert_array_equal(headers[0]["dimensions_offset"][:3], (4, 4, 4))
        self.assertEqual(headers[0]["render_parameters"][2], 1.0)
        np.testing.assert_array_equal(
            headers[0]["acceleration_parameters"], 0,
        )
        np.testing.assert_array_equal(
            headers[0]["scattering_parameters"], (1.0, 1.0, 1.0, 0.0),
        )
        np.testing.assert_allclose(
            headers[0]["multiple_scattering_parameters"],
            (0.9, 0.9, 0.9, 1.0), atol=1e-7,
        )
        self.assertEqual(scalars.size, 64)
        self.assertEqual(transfers.shape[1], 4)
        np.testing.assert_array_equal(
            scene.triangle_volume_indices(),
            np.zeros(12, np.uint32),
        )
        np.testing.assert_array_equal(
            scene.triangle_instance_ids(),
            np.full(12, volume.id, np.uint32),
        )
        bounds_min, bounds_max = scene.bounds()
        np.testing.assert_allclose(bounds_min, (-0.5, -0.5, 0.0))
        np.testing.assert_allclose(bounds_max, (0.5, 0.5, 1.0))

    def test_emission_absorption_integrator_is_bounded(self):
        transfer = ol.Texture1D(((1, 0, 0, 0.5), (1, 0, 0, 0.5)))
        scene = ol.Scene()
        volume = scene.add_volume(
            np.ones((2, 2, 2), np.float32),
            ol.VolumeMaterial(transfer, step_size=0.1),
        )
        origins = np.asarray(((0.5, 0.5, -1),), np.float32)
        directions = np.asarray(((0, 0, 1),), np.float32)
        entry, exit = intersect_unit_boxes(origins, directions, (volume,))
        radiance, transmittance = integrate_volume(
            volume, origins, directions, entry[0], exit[0]
        )
        self.assertGreater(radiance[0, 0], 0.99)
        self.assertLess(transmittance[0], 0.001)
        np.testing.assert_allclose(radiance[0, 1:], 0.0)

    def test_overlapping_volume_integration_is_order_independent(self):
        red = ol.Volume(
            np.ones((2, 2, 2), np.float32),
            ol.VolumeMaterial(
                ol.Texture1D(((1, 0, 0, 0.5), (1, 0, 0, 0.5))),
                step_size=0.1,
            ),
        )
        blue = ol.Volume(
            np.ones((2, 2, 2), np.float32),
            ol.VolumeMaterial(
                ol.Texture1D(((0, 0, 1, 0.5), (0, 0, 1, 0.5))),
                step_size=0.1,
            ),
        )
        origins = np.asarray(((0.5, 0.5, -1.0),), np.float32)
        directions = np.asarray(((0.0, 0.0, 1.0),), np.float32)

        first_order = (red, blue)
        entries, exits = intersect_unit_boxes(origins, directions, first_order)
        first_radiance, first_transmittance = integrate_volumes(
            first_order, origins, directions, entries, exits,
        )
        second_order = tuple(reversed(first_order))
        entries, exits = intersect_unit_boxes(origins, directions, second_order)
        second_radiance, second_transmittance = integrate_volumes(
            second_order, origins, directions, entries, exits,
        )

        np.testing.assert_allclose(first_radiance, second_radiance, atol=1e-6)
        np.testing.assert_allclose(
            first_transmittance, second_transmittance, atol=1e-7,
        )
        np.testing.assert_allclose(first_radiance[0], (0.5, 0.0, 0.5), atol=0.01)
        self.assertLess(first_transmittance[0], 1e-4)

    def test_reference_renderer_renders_volume_without_meshes(self):
        transfer = ol.Texture1D(((0, 0, 0, 0), (2, 0.2, 0.05, 0.8)))
        scene = ol.Scene()
        scene.add_volume(
            np.ones((4, 4, 4), np.float32),
            ol.VolumeMaterial(transfer, step_size=0.05),
            transform=ol.Transform.translation((-0.5, -0.5, 0)),
        )
        camera = ol.PerspectiveCamera((0, 0, -2), (0, 0, 0.5))
        image = ol.ReferencePathTracer(seed=3).render(
            scene, camera, 32, 24, samples=1, max_bounces=1
        )
        center = image[12, 16, :3]
        corner = image[0, 0, :3]
        self.assertGreater(int(center[0]), int(corner[0]))
        self.assertGreater(int(center[0]), int(center[1]))


if __name__ == "__main__":
    unittest.main()
