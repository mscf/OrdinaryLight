"""Opt-in GPU tests against closed-form transport and intersection results."""

import os

import numpy as np
import pytest

import ordinarylight as ol
from ordinarylight.geometry import SdfSphere
from ordinarylight.transport import (
    VulkanTransportScene,
    TransportMaterial,
    VulkanTransportIntegrator,
    GpuSampleAccumulator,
    OpticalMedium,
    MediumBoundary,
    ray_samples,
    surface_samples,
    intersect_rays,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_TRANSPORT") != "1",
    reason="opt-in transport GPU validation",
)


@pytest.fixture(scope="module")
def runtime():
    with ol.VulkanRuntime() as runtime:
        yield runtime


def box(scene, *, material=None):
    vertices = [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ]
    indices = [
        [4, 5, 6],
        [4, 6, 7],
        [0, 2, 1],
        [0, 3, 2],
        [1, 2, 6],
        [1, 6, 5],
        [0, 4, 7],
        [0, 7, 3],
        [3, 7, 6],
        [3, 6, 2],
        [0, 1, 5],
        [0, 5, 4],
    ]
    return scene.add_mesh(vertices, indices, material or ol.Material())


def test_common_hit_contract_and_closest_triangle_custom_geometry(runtime):
    scene = ol.Scene()
    mesh = scene.add_mesh([[-4, -4, -2], [4, -4, -2], [0, 4, -2]], [[0, 1, 2]])
    with VulkanTransportScene(
        runtime,
        scene,
        custom_geometry=[SdfSphere().geometry(identity=73)],
        custom_materials=[TransportMaterial()],
    ) as resident:
        hits = intersect_rays(
            resident,
            [[0, 0, 3], [1.5, 0, 3], [0, 0, 0]],
            [[0, 0, -1], [0, 0, -1], [0, 0, 1]],
        )
        np.testing.assert_array_equal(hits["identity"][:, 0], [2, 1, 2])
        np.testing.assert_array_equal(hits["identity"][:, 2], [73, mesh.id, 73])
        np.testing.assert_allclose(
            hits["position_distance"][:, 3], [2, 5, 1], atol=1e-5
        )
        np.testing.assert_allclose(
            hits["geometric_normal"][:, :3], [[0, 0, 1]] * 3, atol=1e-5
        )
        assert not np.any(hits["boundary"][:, 3])


def test_non_camera_two_diffuse_bounces_match_exact_cavity_series(runtime):
    emitter = TransportMaterial("diffuse", (0.25, 0.25, 0.25), (2, 1, 0.5), True)
    source = TransportMaterial("diffuse", (0.6, 0.6, 0.6))
    with VulkanTransportScene(
        runtime,
        custom_geometry=[SdfSphere().geometry()],
        custom_materials=[emitter, source],
    ) as scene:
        samples = surface_samples(
            np.zeros((64, 3)),
            np.tile([0, 0, 1], (64, 1)),
            materials=1,
            identities=np.arange(63, -1, -1),
        )
        with GpuSampleAccumulator(runtime, 64, extent=(8, 8)) as accumulation:
            with VulkanTransportIntegrator(scene, samples, accumulation) as integrator:
                integrator.accumulate(
                    samples_per_element=16, max_bounces=2, max_steps=1024
                ).wait()
                np.testing.assert_allclose(
                    accumulation.means(),
                    np.tile(np.array([2, 1, 0.5]) * 0.6 * 1.25, (64, 1)),
                    rtol=2e-5,
                )
                records = accumulation.read()
                assert np.all(records["events"][:, 0] == 32)
                assert np.all(records["counts"][:, 3] == 16)
                hdr = accumulation.hdr
                accumulation.resolve().wait()
                accumulation.resolve().wait()
                assert accumulation.hdr is hdr


@pytest.mark.parametrize("representation", ["sdf", "triangles"])
def test_dielectric_absorption_and_fresnel_energy(runtime, representation):
    sigma = np.array([0.2, 0.4, 0.6])
    media = [OpticalMedium(), OpticalMedium(1.5, tuple(sigma))]
    boundary = MediumBoundary(19, 0, 1)
    if representation == "sdf":
        resident = VulkanTransportScene(
            runtime,
            custom_geometry=[SdfSphere().geometry(boundary=19)],
            custom_materials=[TransportMaterial("dielectric")],
            media=media,
            boundaries=[boundary],
        )
    else:
        source = ol.Scene()
        mesh = box(source, material=ol.Material(transmission=1, roughness=0))
        resident = VulkanTransportScene(
            runtime,
            source,
            media=media,
            boundaries=[boundary],
            triangle_boundaries={mesh.id: 19},
        )
    with resident:
        samples = ray_samples(
            np.tile([0, 0, 3], (128, 1)), np.tile([0, 0, -1], (128, 1))
        )
        with GpuSampleAccumulator(runtime, 128) as accumulation:
            with VulkanTransportIntegrator(
                resident, samples, accumulation
            ) as integrator:
                integrator.accumulate(
                    samples_per_element=256,
                    max_bounces=32,
                    environment=(1, 1, 1),
                    seed=29,
                ).wait()
                attenuation = np.exp(-2 * sigma)
                fresnel = 0.04
                expected = fresnel + (1 - fresnel) ** 2 * attenuation / (
                    1 - fresnel * attenuation
                )
                np.testing.assert_allclose(
                    accumulation.means().mean(axis=0), expected, atol=0.004, rtol=0
                )
                records = accumulation.read()
                assert not records["counts"][:, 3].any()
                reflection_rate = (
                    records["events"][:, 1].sum() / records["counts"][:, 0].sum()
                )
                assert reflection_rate == pytest.approx(0.08, abs=0.007)


def test_nested_absorption_and_initial_inside_medium(runtime):
    a = np.array([0.1, 0.2, 0.3])
    b = np.array([0.4, 0.3, 0.2])
    media = [OpticalMedium(), OpticalMedium(1, tuple(a)), OpticalMedium(1, tuple(b))]
    boundaries = [MediumBoundary(10, 0, 1), MediumBoundary(20, 1, 2)]
    geometry = [
        SdfSphere(radius=2).geometry(boundary=10),
        SdfSphere(radius=1).geometry(boundary=20),
    ]
    with VulkanTransportScene(
        runtime,
        custom_geometry=geometry,
        custom_materials=[TransportMaterial("dielectric")],
        media=media,
        boundaries=boundaries,
    ) as scene:
        for origin, initial, expected, events in [
            ([0, 0, 3], (), np.exp(-2 * a - 2 * b), 4),
            ([0, 0, 0], (10, 20), np.exp(-a - b), 2),
        ]:
            with GpuSampleAccumulator(runtime, 1) as accumulation:
                with VulkanTransportIntegrator(
                    scene,
                    ray_samples([origin], [[0, 0, -1]]),
                    accumulation,
                    initial_boundaries=initial,
                ) as integrator:
                    integrator.accumulate(max_bounces=8, environment=(1, 1, 1)).wait()
                    np.testing.assert_allclose(
                        accumulation.means()[0], expected, rtol=2e-4
                    )
                    assert accumulation.read()["events"][0, 2] == events


def test_total_internal_reflection_preserves_medium(runtime):
    with VulkanTransportScene(
        runtime,
        custom_geometry=[SdfSphere().geometry(boundary=7)],
        custom_materials=[TransportMaterial("dielectric")],
        media=[OpticalMedium(), OpticalMedium(1.5)],
        boundaries=[MediumBoundary(7, 0, 1)],
    ) as scene:
        with GpuSampleAccumulator(runtime, 1) as accumulation:
            with VulkanTransportIntegrator(
                scene,
                ray_samples([[0, 0, 0.9]], [[1, 0, 0]]),
                accumulation,
                initial_boundaries=[7],
            ) as integrator:
                integrator.accumulate(
                    max_bounces=6, environment=(1, 1, 1), max_steps=1024
                ).wait()
                records = accumulation.read()
                assert records["events"][0, 3] == 6
                assert records["events"][0, 2] == 0
                assert records["counts"][0, 3] == 1
                np.testing.assert_array_equal(accumulation.means(), [[0, 0, 0]])


def test_non_nested_overlap_is_reported_not_silently_averaged(runtime):
    geometry = [
        SdfSphere((-0.5, 0, 0)).geometry(boundary=1),
        SdfSphere((0.5, 0, 0)).geometry(boundary=2),
    ]
    with VulkanTransportScene(
        runtime,
        custom_geometry=geometry,
        custom_materials=[TransportMaterial("dielectric")],
        media=[OpticalMedium(), OpticalMedium(1)],
        boundaries=[MediumBoundary(1, 0, 1), MediumBoundary(2, 0, 1)],
    ) as scene:
        with GpuSampleAccumulator(runtime, 1) as accumulation:
            with VulkanTransportIntegrator(
                scene, ray_samples([[-3, 0, 0]], [[1, 0, 0]]), accumulation
            ) as integrator:
                integrator.accumulate(max_bounces=8, environment=(1, 1, 1)).wait()
                with pytest.raises(RuntimeError, match="invalid paths"):
                    accumulation.read()
                assert accumulation.read(strict=False)["counts"][0, 2] & 2


def test_persistent_identity_accumulation_and_selective_reset(runtime):
    with VulkanTransportScene(
        runtime,
        custom_geometry=[SdfSphere().geometry()],
        custom_materials=[TransportMaterial()],
    ) as scene:
        samples = ray_samples(
            [[3, 0, 0], [4, 0, 0]], [[1, 0, 0], [1, 0, 0]], identities=[2, 0]
        )
        with GpuSampleAccumulator(runtime, 3) as accumulation:
            with VulkanTransportIntegrator(scene, samples, accumulation) as integrator:
                integrator.accumulate(
                    samples_per_element=3, environment=(0.2, 0.4, 0.6)
                )
                integrator.accumulate(
                    samples_per_element=2, environment=(0.2, 0.4, 0.6)
                ).wait()
                np.testing.assert_array_equal(
                    accumulation.read()["counts"][:, 1], [5, 0, 5]
                )
                accumulation.reset([2]).wait()
                np.testing.assert_array_equal(
                    accumulation.read()["counts"][:, 1], [5, 0, 0]
                )
                integrator.accumulate(environment=(0.2, 0.4, 0.6)).wait()
                np.testing.assert_array_equal(
                    accumulation.read()["counts"][:, 1], [6, 0, 1]
                )
                np.testing.assert_allclose(
                    accumulation.means()[[0, 2]], [[0.2, 0.4, 0.6]] * 2, rtol=1e-6
                )
                with pytest.raises(RuntimeError, match="integrators"):
                    scene.close()
