"""Area-light MIS against independent integrals and BSDF-only estimates."""

import os

import numpy as np
import pytest
import ordinarylight as ol
import ordinaryshade as osh
from ordinarylight.shaders.transport_programs import (
    ordinarylightSampleSphere, ordinarylightSampleSphere_pdf,
)
from ordinarylight.geometry import SdfSphere
from ordinarylight.transport import (
    VulkanTransportScene,
    TransportMaterial,
    GpuSampleAccumulator,
    VulkanTransportIntegrator,
    surface_samples,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_TRANSPORT") != "1",
    reason="opt-in transport GPU validation",
)


@pytest.fixture(scope="module")
def runtime():
    with ol.VulkanRuntime() as runtime:
        yield runtime


def receiver_samples(material, count=64):
    return surface_samples(
        np.zeros((count, 3)),
        np.tile([0, 0, 1], (count, 1)),
        materials=material,
        identities=np.arange(count),
    )


def estimate(runtime, scene, material, *, nee, environment=(0, 0, 0), samples=256):
    with GpuSampleAccumulator(runtime, 64) as accumulator:
        with VulkanTransportIntegrator(
            scene, receiver_samples(material), accumulator
        ) as integrator:
            integrator.accumulate(
                samples_per_element=samples,
                max_bounces=1,
                emissive_nee=nee,
                environment=environment,
                environment_nee=any(environment),
                seed=19,
            ).wait()
            return accumulator.means()


def test_small_triangle_light_integral_and_variance(runtime):
    source = ol.Scene()
    h = 0.2
    source.add_mesh(
        [[-h, -h, 2], [h, -h, 2], [h, h, 2], [-h, h, 2]],
        [[0, 2, 1], [0, 3, 2]],
        ol.Material(emission=(1, 2, 3)),
    )
    with VulkanTransportScene(
        runtime,
        source,
        custom_materials=[TransportMaterial(albedo=(0.5, 0.5, 0.5))],
    ) as scene:
        sampled = estimate(runtime, scene, 2, nee=True)
        plain = estimate(runtime, scene, 2, nee=False)
    # Independent Gauss-Legendre integral of cos(theta_r)*cos(theta_l)/(pi*r^2).
    nodes, weights = np.polynomial.legendre.leggauss(64)
    x, y = np.meshgrid(h * nodes, h * nodes)
    integral = (
        np.sum(np.outer(weights, weights) * 4 / (np.pi * (x * x + y * y + 4) ** 2))
        * h
        * h
    )
    expected = 0.5 * integral * np.array([1, 2, 3])
    np.testing.assert_allclose(sampled.mean(axis=0), expected, rtol=0.02)
    assert sampled[:, 0].var() < plain[:, 0].var() / 4


@pytest.mark.parametrize("environment", [(0, 0, 0), (0.2, 0.2, 0.2)])
def test_sampled_sdf_sphere_matches_projected_solid_angle(runtime, environment):
    with VulkanTransportScene(
        runtime,
        custom_geometry=[SdfSphere(center=(0, 0, 2), radius=0.2).geometry()],
        custom_materials=[
            TransportMaterial("emission", emission=(1, 1, 1)),
            TransportMaterial(albedo=(1, 1, 1)),
        ],
    ) as scene:
        actual = estimate(
            runtime, scene, 1, nee=True, environment=environment, samples=512
        )
    fraction = 0.2**2 / 2**2
    expected = fraction + (1 - fraction) * np.array(environment)
    np.testing.assert_allclose(actual.mean(axis=0), expected, atol=0.0015, rtol=0.025)


def test_heterogeneous_chunk_emission_and_resource_update(runtime):
    from ordinarylight.geometry import (
        CustomGeometry,
        IntersectionProgram,
        SurfaceSamplingProgram,
    )
    from ordinarylight.materials import MaterialGraph, MaterialNode, MaterialResource

    sampler = SurfaceSamplingProgram(
        "samplePlane",
        """
uint samplePlane(vec4 parameters,vec3 randoms,out vec3 position,out vec3 normal,out float area_pdf) {
    position=vec3((randoms.xy*2.0-1.0)*0.2,2.0);
    normal=vec3(0,0,-1); area_pdf=6.25; return 1u;
}
float samplePlane_pdf(vec4 parameters,vec3 position,vec3 normal) { return 6.25; }
""",
    )
    program = IntersectionProgram(
        "plane",
        """
uint plane(vec3 origin,vec3 direction,float t_min,float t_max,vec4 parameters,
    float tolerance,uint max_steps,inout OrdinaryLightCustomHit hit) {
    if(abs(direction.z)<1e-20) return 0u;
    hit.distance=(2.0-origin.z)/direction.z;
    if(hit.distance<t_min || hit.distance>t_max) return 0u;
    hit.geometric_normal=vec3(0,0,-1);
    hit.flags=OL_HIT_MATERIAL;
    hit.material=(origin+hit.distance*direction).x<0?0u:1u;
    return 1u;
}
""",
        hit_version=2,
        sampling=sampler,
    )
    graph = MaterialGraph(
        {
            "value": MaterialNode("uniform", value="color", type="vec4"),
            "rgb": MaterialNode("components", ("value",), value="rgb", type="vec3"),
        },
        {"emission": "rgb"},
        resources=(MaterialResource("color", "uniform"),),
    )
    with runtime.buffer(16, data=np.array([1, 0, 0, 1], np.float32)) as color:
        with VulkanTransportScene(
            runtime,
            custom_geometry=[
                CustomGeometry(
                    ((-0.2, -0.2, 1.9), (0.2, 0.2, 2.1)), program, (0, 0, 0, 0)
                )
            ],
            custom_capacity=4,
            custom_materials=[
                TransportMaterial("emission", program=graph),
                TransportMaterial("emission", emission=(0, 0, 1)),
                TransportMaterial(albedo=(1, 1, 1)),
            ],
            material_resources={"color": color},
        ) as scene:
            first = estimate(runtime, scene, 2, nee=True, samples=1024).mean(axis=0)
            color.upload(np.array([2, 0, 0, 1], np.float32))
            second = estimate(runtime, scene, 2, nee=True, samples=1024).mean(axis=0)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    x, y = np.meshgrid(0.2 * nodes, 0.2 * nodes)
    half = (
        np.sum(np.outer(weights, weights) * 4 / (np.pi * (x * x + y * y + 4) ** 2))
        * 0.2**2
        / 2
    )
    np.testing.assert_allclose(first, [half, 0, half], atol=0.0003)
    np.testing.assert_allclose(second, [2 * half, 0, half], atol=0.0005)
    np.testing.assert_allclose(second, first * [2, 1, 1], rtol=1e-5, atol=1e-6)


@osh.function(name="invalidSample")
def sample_negative_pdf(parameters: osh.vec4, randoms: osh.vec3,
        position: osh.out(osh.vec3), normal: osh.out(osh.vec3),
        area_pdf: osh.out(osh.f32)) -> osh.u32:
    result = ordinarylightSampleSphere(parameters, randoms, position, normal, area_pdf)
    area_pdf = -1.0
    return result


@osh.function(name="invalidSample")
def sample_nan_pdf(parameters: osh.vec4, randoms: osh.vec3,
        position: osh.out(osh.vec3), normal: osh.out(osh.vec3),
        area_pdf: osh.out(osh.f32)) -> osh.u32:
    result = ordinarylightSampleSphere(parameters, randoms, position, normal, area_pdf)
    area_pdf = osh.uint_bits_to_float(osh.u32(0x7fc00000))
    return result


@osh.function(name="invalidSample")
def sample_inconsistent_pdf(parameters: osh.vec4, randoms: osh.vec3,
        position: osh.out(osh.vec3), normal: osh.out(osh.vec3),
        area_pdf: osh.out(osh.f32)) -> osh.u32:
    result = ordinarylightSampleSphere(parameters, randoms, position, normal, area_pdf)
    area_pdf = 2.0 * area_pdf
    return result


@osh.function(name="invalidSample")
def sample_zero_normal(parameters: osh.vec4, randoms: osh.vec3,
        position: osh.out(osh.vec3), normal: osh.out(osh.vec3),
        area_pdf: osh.out(osh.f32)) -> osh.u32:
    result = ordinarylightSampleSphere(parameters, randoms, position, normal, area_pdf)
    normal = osh.vec3(0.0)
    return result


@osh.function(name="invalidSample")
def sample_reversed_normal(parameters: osh.vec4, randoms: osh.vec3,
        position: osh.out(osh.vec3), normal: osh.out(osh.vec3),
        area_pdf: osh.out(osh.f32)) -> osh.u32:
    result = ordinarylightSampleSphere(parameters, randoms, position, normal, area_pdf)
    normal = -normal
    return result


@osh.function(name="invalidSample")
def sample_off_surface(parameters: osh.vec4, randoms: osh.vec3,
        position: osh.out(osh.vec3), normal: osh.out(osh.vec3),
        area_pdf: osh.out(osh.f32)) -> osh.u32:
    result = ordinarylightSampleSphere(parameters, randoms, position, normal, area_pdf)
    position = osh.vec3(1000.0)
    return result


@osh.function(name="invalidSample_pdf")
def invalid_sample_pdf(parameters: osh.vec4, position: osh.vec3, normal: osh.vec3) -> osh.f32:
    return ordinarylightSampleSphere_pdf(parameters, position, normal)


@pytest.mark.parametrize("fault", [
    sample_negative_pdf,
    sample_nan_pdf,
    sample_inconsistent_pdf,
    sample_zero_normal,
    sample_reversed_normal,
    sample_off_surface,
])
def test_invalid_custom_sampler_fails_paths(runtime, fault):
    from dataclasses import replace
    from ordinarylight.geometry import SurfaceSamplingProgram

    geometry = SdfSphere(center=(0, 0, 2), radius=0.2).geometry()
    sampler = geometry.program.sampling
    # Author the faulty callbacks in OrdinaryShade: emitted GLSL spelling must
    # never decide whether the regression actually injects its intended fault.
    source = sampler.source + osh.compile_function(
        fault, externals=(osh.external(ordinarylightSampleSphere.function),)
    ).source + osh.compile_function(
        invalid_sample_pdf, externals=(osh.external(ordinarylightSampleSphere_pdf.function),)
    ).source
    invalid = replace(
        geometry,
        program=replace(
            geometry.program,
            sampling=SurfaceSamplingProgram("invalidSample", source),
        ),
    )
    with VulkanTransportScene(
        runtime,
        custom_geometry=[invalid],
        custom_materials=[
            TransportMaterial("emission", emission=(1, 1, 1)),
            TransportMaterial(),
        ],
    ) as scene:
        with pytest.raises(RuntimeError, match="invalid"):
            estimate(runtime, scene, 1, nee=True, samples=1)


def test_unregistered_custom_sampler_preserves_bsdf_emission(runtime):
    from dataclasses import replace

    geometry = SdfSphere(center=(0, 0, 2), radius=0.8).geometry()
    geometry = replace(geometry, program=replace(geometry.program, sampling=None))
    with VulkanTransportScene(
        runtime,
        custom_geometry=[geometry],
        custom_materials=[
            TransportMaterial("emission", emission=(1, 1, 1)),
            TransportMaterial(albedo=(1, 1, 1)),
        ],
    ) as scene:
        actual = estimate(runtime, scene, 1, nee=True, samples=1024)
    np.testing.assert_allclose(actual.mean(axis=0), 0.8**2 / 2**2, atol=0.008)
