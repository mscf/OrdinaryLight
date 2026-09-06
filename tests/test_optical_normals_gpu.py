"""Explicit optical normals preserve topology and retain null-sample weights."""

import os
import numpy as np
import pytest
import ordinarylight as ol
from ordinarylight.geometry import SdfSphere, CustomGeometry, IntersectionProgram
from ordinarylight.transport import (
    VulkanTransportScene,
    TransportMaterial,
    OpticalMedium,
    MediumBoundary,
    GpuSampleAccumulator,
    VulkanTransportIntegrator,
    surface_samples,
    ray_samples,
    dielectric_event,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_TRANSPORT") != "1",
    reason="opt-in Vulkan optical-normal validation",
)


@pytest.fixture(scope="module")
def runtime():
    with ol.VulkanRuntime() as runtime:
        yield runtime


def run_surface(
    runtime, scene, normal, policy, *, incoming=None, bounces=1, initial_boundaries=()
):
    inputs = surface_samples(
        [[0, 0, 1]],
        [[0, 0, 1]],
        materials=0,
        boundaries=7,
        shading_normals=[normal],
        incoming=incoming,
    )
    with GpuSampleAccumulator(runtime, 1) as output:
        with VulkanTransportIntegrator(
            scene, inputs, output, initial_boundaries=initial_boundaries
        ) as integrator:
            integrator.accumulate(
                samples_per_element=16384,
                max_bounces=bounces,
                dielectric_normal_policy=policy,
                environment=(1, 1, 1),
            ).wait()
            return output.read(), output.means()


def test_equal_optical_normal_matches_geometric_and_invalid_policy(runtime):
    with VulkanTransportScene(
        runtime,
        custom_geometry=[SdfSphere().geometry(boundary=7)],
        custom_materials=[TransportMaterial("dielectric")],
        media=[OpticalMedium(), OpticalMedium(1.5, (0.2, 0.4, 0.6))],
        boundaries=[MediumBoundary(7, 0, 1)],
    ) as scene:
        geometric, _ = run_surface(runtime, scene, [0, 0, 1], "geometric", bounces=16)
        shading, _ = run_surface(
            runtime, scene, [0, 0, 1], "shading_clipped", bounces=16
        )
        np.testing.assert_array_equal(geometric, shading)
        with pytest.raises(ValueError, match="dielectric_normal_policy"):
            run_surface(runtime, scene, [0, 0, 1], "unknown")


def test_wrong_hemisphere_is_null_not_renormalized(runtime):
    with VulkanTransportScene(
        runtime,
        custom_geometry=[SdfSphere().geometry(boundary=7)],
        custom_materials=[TransportMaterial("dielectric")],
        media=[OpticalMedium(), OpticalMedium(1.5)],
        boundaries=[MediumBoundary(7, 0, 1)],
    ) as scene:
        records, _ = run_surface(runtime, scene, [0.8, 0, 0.6], "shading_clipped")
        count = records["counts"][0]
        assert tuple(count[:3]) == (16384, 16384, 0)
        # Optical reflection points into the geometric solid and is discarded;
        # surviving transmissions keep their original Fresnel probability.
        assert records["events"][0, 1] == 0
        fresnel = dielectric_event([0, 0, -1], [0.8, 0, 0.6], 1, 1.5, 0).fresnel
        np.testing.assert_allclose(
            records["events"][0, 2] / 16384, 1 - fresnel, atol=0.008
        )
        hidden, means = run_surface(
            runtime,
            scene,
            [0.8, 0, 0.6],
            "shading_clipped",
            incoming=[[0.99, 0, -np.sqrt(1 - 0.99**2)]],
        )
        assert hidden["counts"][0, 1] == 16384
        assert not hidden["events"].any()
        np.testing.assert_array_equal(means, [[0, 0, 0]])


@pytest.mark.parametrize("roughness", [0.0, 0.25])
@pytest.mark.parametrize("nested", [False, True])
def test_custom_optical_normals_keep_nested_medium_transitions(
    runtime, roughness, nested
):
    legacy = IntersectionProgram.sdf_sphere()
    program = IntersectionProgram(
        "opticalSphere",
        legacy.source
        + """
uint opticalSphere(vec3 origin,vec3 direction,float t_min,float t_max,vec4 parameters,
    float tolerance,uint max_steps,inout OrdinaryLightCustomHit hit) {
    uint status=ordinarylightSdfSphere(origin,direction,t_min,t_max,parameters,
        tolerance,max_steps,hit.distance,hit.geometric_normal);
    vec3 n=hit.geometric_normal;
    hit.shading_normal=vec3(0.6*n.x+0.8*n.z,n.y,-0.8*n.x+0.6*n.z);
    hit.flags=OL_HIT_SHADING_NORMAL;
    return status;
}
""",
        hit_version=2,
        sampling=legacy.sampling,
    )
    geometry = CustomGeometry(
        ((-1, -1, -1), (1, 1, 1)), program, (0, 0, 0, 1), boundary=7
    )
    with VulkanTransportScene(
        runtime,
        custom_geometry=[geometry]
        + (
            [
                CustomGeometry(
                    ((-0.4, -0.4, -0.4), (0.4, 0.4, 0.4)),
                    program,
                    (0, 0, 0, 0.4),
                    material=1,
                    boundary=8,
                )
            ]
            if nested
            else []
        ),
        custom_materials=[TransportMaterial("dielectric", roughness=roughness)] * 2,
        media=[OpticalMedium(), OpticalMedium(1.5), OpticalMedium(1.3)],
        boundaries=[MediumBoundary(7, 0, 1), MediumBoundary(8, 1, 2)],
    ) as scene:
        results = []
        for policy in ["geometric", "shading_clipped"]:
            with GpuSampleAccumulator(runtime, 64) as output:
                inputs = ray_samples(
                    np.tile([0, 0, 3], (64, 1)), np.tile([0, 0, -1], (64, 1))
                )
                with VulkanTransportIntegrator(scene, inputs, output) as integrator:
                    integrator.accumulate(
                        samples_per_element=256,
                        max_bounces=32,
                        dielectric_normal_policy=policy,
                        environment=(1, 1, 1),
                    ).wait()
                    records = output.read(strict=False)
                    assert not records["counts"][:, 2].any(), (
                        policy,
                        records["counts"][:, 2],
                    )
                    results.append(output.means().mean(axis=0))
        assert np.isfinite(results).all()
        assert np.all(results[1] >= 0)
        assert np.all(results[1] <= 1.04)
        assert np.max(np.abs(results[0] - results[1])) > 0.01


def test_optical_tir_and_eta_weight_use_classified_media(runtime):
    with VulkanTransportScene(
        runtime,
        custom_geometry=[SdfSphere().geometry(boundary=7)],
        custom_materials=[TransportMaterial("dielectric")],
        media=[OpticalMedium(), OpticalMedium(1.5)],
        boundaries=[MediumBoundary(7, 0, 1)],
    ) as scene:
        options = dict(incoming=[[0.8, 0, 0.6]], initial_boundaries=[7])
        geometric, geometric_mean = run_surface(
            runtime, scene, [0.8, 0, 0.6], "geometric", **options
        )
        optical, optical_mean = run_surface(
            runtime, scene, [0.8, 0, 0.6], "shading_clipped", **options
        )
        assert geometric["events"][0, 3] == 16384
        assert optical["events"][0, 3] == 0
        np.testing.assert_array_equal(geometric_mean, [[0, 0, 0]])
        fresnel = dielectric_event([0.8, 0, 0.6], [-0.8, 0, -0.6], 1.5, 1, 0).fresnel
        np.testing.assert_allclose(optical_mean, (1 - fresnel) * 1.5**2, atol=0.02)
