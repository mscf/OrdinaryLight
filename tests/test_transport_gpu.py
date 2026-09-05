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


def test_mixed_initial_media_in_one_batch(runtime):
    absorption = np.array([0.2, 0.4, 0.6])
    with (
        VulkanTransportScene(
            runtime,
            custom_geometry=[SdfSphere(radius=1).geometry(boundary=7)],
            custom_materials=[TransportMaterial("dielectric")],
            media=[OpticalMedium(), OpticalMedium(1, tuple(absorption))],
            boundaries=[MediumBoundary(7, 0, 1)],
        ) as scene,
        GpuSampleAccumulator(runtime, 2) as accumulation,
    ):
        samples = ray_samples(
            [[0, 0, 2], [0, 0, 0]],
            [[0, 0, -1]] * 2,
            initial_stack=[0, 1],
        )
        with VulkanTransportIntegrator(
            scene, samples, accumulation, initial_stacks=[(7,)]
        ) as integrator:
            integrator.accumulate(max_bounces=4, environment=(1, 1, 1)).wait()
            np.testing.assert_allclose(
                accumulation.means(),
                np.exp(-np.array([2, 1])[:, None] * absorption),
                rtol=2e-4,
            )
            assert not accumulation.read()["counts"][:, 2].any()
        with pytest.raises(ValueError, match="stack index"):
            VulkanTransportIntegrator(scene, samples, accumulation)


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


def test_resource_backed_field_and_image_updates(runtime):
    from ordinarylight.geometry import (
        CustomGeometry,
        IntersectionProgram,
        IntersectionResource,
    )
    from ordinarylight.runtime import VulkanKernel, compile_compute
    from ordinarylight.pipeline.vulkan import (
        VulkanResource,
        VulkanResourceUse,
        VulkanPass,
        VulkanPassPipeline,
    )
    import vulkan as vk

    # A sampled planar SDF is exact under linear interpolation. Two z layers
    # make the stepping guarantee independently checkable.
    source = """
uint gridHit(vec3 o,vec3 d,float lo,float hi,vec4 p,float eps,uint steps,out float t,out vec3 n) {
    t=lo; n=vec3(0,0,1);
    for(uint j=0u;j<steps;++j) {
        float z=(o.z+t*d.z+1.0)*0.5;
        float value=mix(distanceGrid[0],distanceGrid[1],z);
        if(abs(value)<=eps) return 1u;
        t+=abs(value); if(t>hi) return 0u;
    }
    return 2u;
}
"""
    program = IntersectionProgram(
        "gridHit",
        source,
        resources=[IntersectionResource("distanceGrid", element_type="float")],
    )
    with runtime.buffer(8, data=np.array([-1, 1], np.float32)) as data:
        geometry = CustomGeometry(((-1, -1, -1), (1, 1, 1)), program, (0, 0, 0, 0))
        with VulkanTransportScene(
            runtime,
            custom_geometry=[geometry],
            custom_materials=[TransportMaterial()],
            custom_resources={"distanceGrid": data},
        ) as scene:
            hit = intersect_rays(scene, [[0, 0, 3]], [[0, 0, -1]])
            assert hit["position_distance"][0, 3] == pytest.approx(3)
            with pytest.raises(RuntimeError, match="borrowers"):
                data.close()
            data.upload(np.array([-1.25, 0.75], np.float32))
            hit = intersect_rays(scene, [[0, 0, 3]], [[0, 0, -1]])
            assert hit["position_distance"][0, 3] == pytest.approx(2.75)
            unresolved = intersect_rays(scene, [[0, 0, 3]], [[0, 0, -1]], max_steps=1)
            assert unresolved["boundary"][0, 3] == 1

    image_source = (
        SdfSphere().geometry().program.source
        + """
uint imageHit(vec3 o,vec3 d,float lo,float hi,vec4 p,float eps,uint steps,out float t,out vec3 n) {
    return ordinarylightSdfSphere(o,d,lo,hi,imageLoad(sphereImage,ivec2(0)),eps,steps,t,n);
}
"""
    )
    image_program = IntersectionProgram(
        "imageHit",
        image_source,
        resources=[IntersectionResource("sphereImage", "image", "rgba32f")],
    )
    with runtime.image(1, 1) as field:
        resource = VulkanResource.image(field)
        writer_source = """#version 460
layout(local_size_x=1) in;
layout(set=0,binding=0,rgba32f) writeonly uniform image2D field;
void main() { imageStore(field,ivec2(0),vec4(0,0,0,0.5)); }
"""
        with VulkanKernel(
            runtime, compile_compute(writer_source), {0: resource}
        ) as writer:
            ready = VulkanPassPipeline(
                [
                    VulkanPass(
                        "field_producer",
                        (
                            VulkanResourceUse(
                                resource,
                                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                vk.VK_ACCESS_SHADER_WRITE_BIT,
                                vk.VK_IMAGE_LAYOUT_GENERAL,
                            ),
                        ),
                        writer.bind,
                        (1, 1, 1),
                    )
                ]
            ).execute(runtime)
            geometry = CustomGeometry(
                ((-1, -1, -1), (1, 1, 1)), image_program, (0, 0, 0, 0)
            )
            with VulkanTransportScene(
                runtime,
                custom_geometry=[geometry],
                custom_materials=[TransportMaterial("diffuse", (0, 0, 0), (2, 1, 0.5))],
                custom_resources={"sphereImage": field},
            ) as scene:
                hit = intersect_rays(scene, [[0, 0, 3]], [[0, 0, -1]], after=[ready])
                assert hit["position_distance"][0, 3] == pytest.approx(2.5, abs=1e-5)
                with GpuSampleAccumulator(runtime, 1) as output:
                    with VulkanTransportIntegrator(
                        scene, ray_samples([[0, 0, 3]], [[0, 0, -1]]), output
                    ) as transport:
                        transport.accumulate()
                        np.testing.assert_allclose(output.means(), [[2, 1, 0.5]])
                with pytest.raises(RuntimeError, match="borrowers"):
                    field.close()


def test_reusable_inputs_and_many_to_one_reduction(runtime):
    from ordinarylight.transport import GpuTransportSamples, SampleReduction

    with VulkanTransportScene(
        runtime,
        custom_geometry=[SdfSphere().geometry()],
        custom_materials=[
            TransportMaterial("diffuse", (0, 0, 0)),
            TransportMaterial("diffuse", (0, 0, 0), (1, 2, 3)),
            TransportMaterial("diffuse", (0, 0, 0), (3, 4, 5)),
        ],
    ) as scene:
        inputs = surface_samples([[0, 0, 2]] * 3, [[0, 0, 1]] * 3, materials=[1, 2, 1])
        with (
            GpuTransportSamples(runtime, 4, samples=inputs) as samples,
            GpuSampleAccumulator(runtime, 2) as output,
        ):
            with VulkanTransportIntegrator(
                scene, samples, output, reduction=SampleReduction([0, 0, 1])
            ) as transport:
                kernel, buffer = transport._kernel, samples.buffer
                transport.accumulate(samples_per_element=3)
                np.testing.assert_allclose(output.means(), [[2, 3, 4], [1, 2, 3]])
                np.testing.assert_array_equal(output.read()["counts"][:, 1], [6, 3])
                transport.update_samples(inputs[:2], reduction=SampleReduction([1, 1]))
                output.reset()
                transport.accumulate(samples_per_element=2)
                np.testing.assert_allclose(output.means(), [[0, 0, 0], [2, 3, 4]])
                assert transport._kernel is kernel and samples.buffer is buffer
                with pytest.raises(RuntimeError, match="integrators"):
                    samples.close()
                samples.set_count(1)
                with pytest.raises(ValueError, match="count changed"):
                    transport.accumulate()
                transport.set_reduction(SampleReduction([0]))
                transport.accumulate()
                np.testing.assert_allclose(output.means()[0], [1, 2, 3])


def test_gpu_generated_samples_without_input_readback_and_invalid_guard(
    runtime, monkeypatch
):
    import vulkan as vk
    from ordinarylight.runtime import VulkanKernel, compile_compute
    from ordinarylight.pipeline.vulkan import (
        VulkanResource,
        VulkanResourceUse,
        VulkanPass,
        VulkanPassPipeline,
    )
    from ordinarylight.transport import (
        GpuTransportSamples,
        SampleReduction,
        shader_source,
    )

    with VulkanTransportScene(
        runtime,
        custom_geometry=[SdfSphere().geometry()],
        custom_materials=[TransportMaterial()],
    ) as scene:
        with (
            GpuTransportSamples(runtime, 2) as inputs,
            GpuSampleAccumulator(runtime, 1) as output,
        ):
            resource = VulkanResource.buffer(inputs.buffer)
            source = (
                "#version 460\nlayout(local_size_x=2) in;\n"
                + shader_source("contracts")
                + """
layout(set=0,binding=0,std430) buffer Inputs { OrdinaryLightSurfaceSample samples[]; };
layout(push_constant) uniform Constants { uint invalid; } pc;
void main() {
    uint i=gl_GlobalInvocationID.x;
    OrdinaryLightSurfaceSample s;
    s.position=vec4(3,0,0,0); s.incoming=vec4(1,0,0,0);
    s.geometric_normal=vec4(0,0,1,0); s.shading_normal=s.geometric_normal;
    s.identity=uvec4(i,0,0,0); s.media=uvec4(0,0,0xffffffffu,0);
    if(pc.invalid==1u && i==0u) { s.identity.w=1u; s.identity.z=0xffffffffu; }
    samples[i]=s;
}
"""
            )
            with VulkanKernel(
                runtime, compile_compute(source), {0: resource}, push_constant_size=4
            ) as generator:

                def produce(invalid=0):
                    return VulkanPassPipeline(
                        [
                            VulkanPass(
                                "generate",
                                (
                                    VulkanResourceUse(
                                        resource,
                                        vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                        vk.VK_ACCESS_SHADER_WRITE_BIT,
                                    ),
                                ),
                                lambda command: generator.bind(
                                    command, int(invalid).to_bytes(4, "little")
                                ),
                                (1, 1, 1),
                            )
                        ]
                    ).execute(runtime)

                monkeypatch.setattr(
                    inputs.buffer, "read", lambda: pytest.fail("GPU sample readback")
                )
                with VulkanTransportIntegrator(
                    scene, inputs, output, reduction=SampleReduction([0, 0])
                ) as transport:
                    ready = produce()
                    transport.accumulate(
                        samples_per_element=4,
                        environment=(0.2, 0.4, 0.6),
                        after=[ready],
                    )
                    np.testing.assert_allclose(
                        output.means(), [[0.2, 0.4, 0.6]], rtol=1e-6
                    )
                    assert output.read()["counts"][0, 1] == 8
                    output.reset()
                    transport.accumulate(after=[produce(1)])
                    records = output.read(strict=False)
                    np.testing.assert_array_equal(records["counts"][0, :3], [2, 1, 32])
                    with pytest.raises(RuntimeError, match="invalid paths"):
                        output.read()
                with pytest.raises(RuntimeError, match="borrowers"):
                    inputs.buffer.close()


def test_dielectric_shading_normal_does_not_change_boundary_scattering(runtime):
    with VulkanTransportScene(
        runtime,
        custom_geometry=[SdfSphere().geometry(boundary=7)],
        custom_materials=[TransportMaterial("dielectric")],
        media=[OpticalMedium(), OpticalMedium(1.5, (0.2, 0.4, 0.6))],
        boundaries=[MediumBoundary(7, 0, 1)],
    ) as scene:
        results = []
        for shading in ([0, 0, 1], [0.8, 0, 0.6]):
            inputs = surface_samples(
                [[0, 0, 1]],
                [[0, 0, 1]],
                shading_normals=[shading],
                materials=0,
                boundaries=7,
            )
            with GpuSampleAccumulator(runtime, 1) as output:
                with VulkanTransportIntegrator(scene, inputs, output) as transport:
                    transport.accumulate(
                        samples_per_element=256, max_bounces=24, environment=(1, 1, 1)
                    )
                    results.append(output.read())
        np.testing.assert_array_equal(results[0], results[1])


def test_weighted_reduction_and_explicit_normalization(runtime):
    from ordinarylight.transport import SampleReduction

    with (
        VulkanTransportScene(
            runtime,
            custom_geometry=[SdfSphere(center=(100, 0, 0)).geometry()],
            custom_materials=[
                TransportMaterial(albedo=(0, 0, 0), emission=(2, 0, 0)),
                TransportMaterial(albedo=(0, 0, 0), emission=(0, 4, 0)),
            ],
        ) as scene,
        GpuSampleAccumulator(runtime, 1) as accumulation,
    ):
        samples = surface_samples(
            [[0, 0, 0]] * 2,
            [[0, 0, 1]] * 2,
            materials=[0, 1],
        )
        for normalization, expected in [(None, [0.5, 3, 0]), (1, [1, 6, 0])]:
            accumulation.reset().wait()
            with VulkanTransportIntegrator(
                scene,
                samples,
                accumulation,
                reduction=SampleReduction(
                    [0, 0], weights=[1, 3], normalization_weights=normalization
                ),
            ) as integrator:
                integrator.accumulate(samples_per_element=3).wait()
                np.testing.assert_allclose(accumulation.means()[0], expected)
                assert accumulation.read()["counts"][0, 1] == 6
                accumulation.resolve().wait()
                assert accumulation.hdr.layout != 0


def test_graph_emission_matches_fixed_material(runtime):
    from ordinarylight.materials import MaterialGraph, MaterialNode

    graph = MaterialGraph(
        {
            "color": MaterialNode("constant", value=(0.2, 0.4, 0.6), type="vec3"),
            "gain": MaterialNode("constant", value=2),
            "emission": MaterialNode("multiply", ("color", "gain"), type="vec3"),
        },
        {"emission": "emission"},
    )
    with (
        VulkanTransportScene(
            runtime,
            custom_geometry=[SdfSphere(center=(100, 0, 0)).geometry()],
            custom_materials=[
                TransportMaterial(albedo=(0, 0, 0), program=graph),
                TransportMaterial(albedo=(0, 0, 0), emission=(0.4, 0.8, 1.2)),
            ],
        ) as scene,
        GpuSampleAccumulator(runtime, 2) as accumulation,
    ):
        with VulkanTransportIntegrator(
            scene,
            surface_samples([[0, 0, 0]] * 2, [[0, 0, 1]] * 2, materials=[0, 1]),
            accumulation,
        ) as integrator:
            integrator.accumulate().wait()
            np.testing.assert_allclose(accumulation.means(), [[0.4, 0.8, 1.2]] * 2)


def test_graph_metal_white_furnace_and_rough_glass(runtime):
    from ordinarylight.materials import MaterialGraph, MaterialNode

    graph = MaterialGraph(
        {
            "rough": MaterialNode("constant", value=0.4),
            "metal": MaterialNode("constant", value=1),
        },
        {"roughness": "rough", "metallic": "metal"},
    )
    with (
        VulkanTransportScene(
            runtime,
            custom_geometry=[SdfSphere(center=(100, 0, 0)).geometry()],
            custom_materials=[
                TransportMaterial("pbr", albedo=(1, 1, 1), program=graph),
                TransportMaterial("pbr", albedo=(1, 1, 1), metallic=1, roughness=0.4),
                TransportMaterial("pbr", albedo=(1, 1, 1), metallic=1),
            ],
        ) as scene,
        GpuSampleAccumulator(runtime, 3) as accumulation,
    ):
        with VulkanTransportIntegrator(
            scene,
            surface_samples([[0, 0, 0]] * 3, [[0, 0, 1]] * 3, materials=[0, 1, 2]),
            accumulation,
        ) as integrator:
            integrator.accumulate(
                samples_per_element=32768, max_bounces=2, environment=(1, 1, 1)
            ).wait()
            means = accumulation.means()
            assert np.all((means >= 0.85) & (means <= 1.01))
            np.testing.assert_allclose(means[0], means[1], atol=0.01)
            np.testing.assert_allclose(means[2], [1, 1, 1], atol=1e-5)
    with (
        VulkanTransportScene(
            runtime,
            custom_geometry=[SdfSphere().geometry(boundary=7)],
            custom_materials=[TransportMaterial("dielectric", roughness=0.25)],
            media=[OpticalMedium(), OpticalMedium(1.5)],
            boundaries=[MediumBoundary(7, 0, 1)],
        ) as scene,
        GpuSampleAccumulator(runtime, 1) as accumulation,
    ):
        with VulkanTransportIntegrator(
            scene, ray_samples([[0, 0, 2]], [[0, 0, -1]]), accumulation
        ) as integrator:
            integrator.accumulate(
                samples_per_element=32768,
                max_bounces=32,
                max_steps=8192,
                environment=(1, 1, 1),
            ).wait()
            assert np.all((accumulation.means() > 0.9) & (accumulation.means() < 1.02))
            events = accumulation.read()["events"][0]
            assert events[1] > 0 and events[2] > 0


def test_analytic_light_nee_and_environment_mis(runtime):
    with (
        VulkanTransportScene(
            runtime,
            custom_geometry=[SdfSphere(center=(100, 0, 0)).geometry()],
            custom_materials=[TransportMaterial(albedo=(0.5, 0.25, 0.75))],
            lights=[ol.PointLight((0, 0, 2), intensity=4)],
        ) as scene,
        GpuSampleAccumulator(runtime, 1) as accumulation,
    ):
        with VulkanTransportIntegrator(
            scene, surface_samples([[0, 0, 0]], [[0, 0, 1]], materials=0), accumulation
        ) as integrator:
            integrator.accumulate(max_bounces=1).wait()
            expected = np.array([0.5, 0.25, 0.75]) / np.pi
            np.testing.assert_allclose(accumulation.means()[0], expected, rtol=1e-5)
            accumulation.reset().wait()
            integrator.accumulate(
                samples_per_element=32768,
                max_bounces=1,
                environment=(1, 1, 1),
                environment_nee=True,
            ).wait()
            np.testing.assert_allclose(
                accumulation.means()[0], expected + [0.5, 0.25, 0.75], atol=0.01
            )


def test_material_graph_uniform_buffer_and_texture_resources(runtime):
    from ordinarylight.materials import MaterialGraph, MaterialNode, MaterialResource
    from ordinarylight.runtime import VulkanKernel, VulkanSampler, compile_compute
    from ordinarylight.pipeline.vulkan import (
        VulkanResource,
        VulkanResourceUse,
        VulkanPass,
    )
    from ordinarylight.pipeline.graph import VulkanGraph
    import vulkan as vk

    graph = MaterialGraph(
        {
            "uv": MaterialNode("constant", value=(0.5, 0.5), type="vec2"),
            "pixel": MaterialNode("texture", ("uv",), value="texture", type="vec4"),
            "gain": MaterialNode("uniform", value="gain", type="vec4"),
            "index": MaterialNode("constant", value=0),
            "tint": MaterialNode("buffer", ("index",), value="tint", type="vec4"),
            "scaled": MaterialNode("multiply", ("pixel", "gain"), type="vec4"),
            "colored": MaterialNode("multiply", ("scaled", "tint"), type="vec4"),
            "rgb": MaterialNode("components", ("colored",), value="rgb", type="vec3"),
        },
        {"emission": "rgb"},
        resources=(
            MaterialResource("texture", "texture"),
            MaterialResource("gain", "uniform"),
            MaterialResource("tint", "buffer"),
        ),
    )
    fill = """#version 460
layout(local_size_x=1) in;
layout(set=0,binding=0,rgba32f) writeonly uniform image2D target;
void main() { imageStore(target,ivec2(0),vec4(1,2,3,1)); }
"""
    with (
        runtime.image(1, 1) as image,
        VulkanSampler(runtime) as sampler,
        runtime.buffer(16, data=np.ones(4, np.float32)) as gain,
        runtime.buffer(16, data=np.array([1, 0.5, 0.25, 1], np.float32)) as tint,
    ):
        with (
            VulkanTransportScene(
                runtime,
                custom_geometry=[SdfSphere(center=(100, 0, 0)).geometry()],
                custom_materials=[TransportMaterial("emission", program=graph)],
                material_resources={
                    "texture": (image, sampler),
                    "gain": gain,
                    "tint": tint,
                },
            ) as scene,
            GpuSampleAccumulator(runtime, 1) as accumulation,
            VulkanKernel(
                runtime, compile_compute(fill), {0: VulkanResource.image(image)}
            ) as producer,
        ):
            with VulkanTransportIntegrator(
                scene,
                surface_samples([[0, 0, 0]], [[0, 0, 1]], materials=0),
                accumulation,
            ) as integrator:
                operation = integrator.accumulate_operation()
                schedule = (
                    VulkanGraph()
                    .add("shade", operation)
                    .add(
                        "texture",
                        VulkanPass(
                            "texture",
                            (
                                VulkanResourceUse(
                                    VulkanResource.image(image),
                                    vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    vk.VK_ACCESS_SHADER_WRITE_BIT,
                                    vk.VK_IMAGE_LAYOUT_GENERAL,
                                ),
                            ),
                            lambda command: producer.bind(command),
                            (1, 1, 1),
                        ),
                    )
                    .compile()
                )
                schedule.execute(runtime).wait()
                np.testing.assert_allclose(accumulation.means(), [[1, 1, 0.75]])
                kernel = integrator._kernel
                gain.upload(np.full(4, 2, np.float32))
                accumulation.reset().wait()
                schedule.execute(runtime).wait()
                np.testing.assert_allclose(accumulation.means(), [[2, 2, 1.5]])
                assert kernel is integrator._kernel


def test_equal_ior_rough_boundary_is_transparent(runtime):
    with (
        VulkanTransportScene(
            runtime,
            custom_geometry=[SdfSphere().geometry(boundary=7)],
            custom_materials=[TransportMaterial("dielectric", roughness=0.9)],
            media=[OpticalMedium(), OpticalMedium(1)],
            boundaries=[MediumBoundary(7, 0, 1)],
        ) as scene,
        GpuSampleAccumulator(runtime, 2) as accumulation,
    ):
        with VulkanTransportIntegrator(
            scene,
            ray_samples(
                [[0.99, 0, 2], [0.99, 0, 0]], [[0, 0, -1]] * 2, initial_stack=[0, 1]
            ),
            accumulation,
            initial_stacks=[(7,)],
        ) as integrator:
            integrator.accumulate(
                samples_per_element=512, max_bounces=4, environment=(1, 1, 1)
            ).wait()
            np.testing.assert_allclose(accumulation.means(), np.ones((2, 3)), atol=1e-5)


def test_rough_bsdf_sampling_matches_solid_angle_quadrature(runtime):
    from ordinarylight.transport import shader_source
    from ordinarylight.runtime import VulkanKernel, compile_compute
    from ordinarylight.pipeline.vulkan import (
        VulkanResource,
        VulkanResourceUse,
        VulkanPass,
        VulkanPassPipeline,
    )
    import vulkan as vk
    import struct

    count = 256 * 256
    source = (
        "#version 460\n"
        + "\n".join(
            shader_source(component)
            for component in ("material_contracts", "sampling", "dielectric", "bsdf")
        )
        + """
layout(local_size_x=64) in;
layout(set=0,binding=0,std430) writeonly buffer Results { vec4 results[]; };
layout(push_constant) uniform Constants { uint kind; } pc;
void main() {
    uint i=gl_GlobalInvocationID.x;
    float z=1.0-2.0*(float(i/256u)+0.5)/256.0;
    float phi=2.0*OL_PI*(float(i%256u)+0.5)/256.0;
    vec3 outgoing=vec3(sqrt(1.0-z*z)*cos(phi),sqrt(1.0-z*z)*sin(phi),z);
    MaterialEvaluation m; m.base_color=vec3(0.7,0.4,0.2); m.metallic=0.7; m.roughness=0.6; m.ior=1.5;
    vec4 value=olBsdf(m,pc.kind,vec3(0,0,1),vec3(0.6,0,0.8),outgoing,1.0,1.5);
    results[i*2u]=vec4(value.rgb*abs(z)*4.0*OL_PI,0);
    vec3 randoms=vec3(secondaryNeeHash(i*3u+1u),secondaryNeeHash(i*3u+2u),secondaryNeeHash(i*3u+3u))*(1.0/4294967296.0);
    vec3 weight; float pdf; bool reflected,tir;
    olSampleBsdf(m,pc.kind,vec3(0,0,1),vec3(0.6,0,0.8),1.0,1.5,randoms,weight,pdf,reflected,tir);
    results[i*2u+1u]=vec4(weight,pdf);
}
"""
    )
    with (
        runtime.buffer(count * 32) as output,
        VulkanKernel(
            runtime,
            compile_compute(source),
            {0: VulkanResource.buffer(output)},
            push_constant_size=4,
        ) as kernel,
    ):
        for kind in (1, 2):
            VulkanPassPipeline(
                [
                    VulkanPass(
                        "quadrature",
                        (
                            VulkanResourceUse(
                                VulkanResource.buffer(output),
                                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                vk.VK_ACCESS_SHADER_WRITE_BIT,
                            ),
                        ),
                        lambda command: kernel.bind(command, struct.pack("I", kind)),
                        (count // 64, 1, 1),
                    )
                ]
            ).execute(runtime).wait()
            records = np.frombuffer(output.read(), np.float32).reshape(count, 2, 4)
            assert np.isfinite(records).all()
            integral = records[:, 0, :3].mean(axis=0, dtype=np.float64)
            estimate = records[:, 1, :3].mean(axis=0, dtype=np.float64)
            np.testing.assert_allclose(estimate, integral, rtol=0.02, atol=0.005)
