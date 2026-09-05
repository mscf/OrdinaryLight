"""Resource-backed camera graphs must agree with their constant equivalents."""

import os
import numpy as np
import pytest
import ordinarylight as ol
from ordinarylight.materials import MaterialGraph, MaterialNode, MaterialResource

GPU = pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_TRANSPORT") != "1",
    reason="opt-in Vulkan material validation",
)


def emission_graph():
    return MaterialGraph(
        {
            "value": MaterialNode("uniform", value="emission", type="vec4"),
            "rgb": MaterialNode("components", ("value",), value="rgb", type="vec3"),
        },
        {"emission": "rgb"},
        resources=(MaterialResource("emission", "uniform"),),
    ).compile()


@GPU
@pytest.mark.parametrize("staged", [False, True])
def test_camera_uniform_matches_fixed_emission_and_updates(staged):
    from ordinarylight.renderers.gi import VulkanGlobalIlluminationRenderer

    graph = emission_graph()
    with ol.VulkanRuntime() as runtime:
        with runtime.buffer(
            16, data=np.array([0.2, 0.4, 0.6, 1], np.float32)
        ) as uniform:
            with ol.VulkanMaterialResources(
                runtime, [graph], {"emission": uniform}
            ) as resources:
                config = ol.RendererConfig(material_resources=resources, max_bounces=1)
                with VulkanGlobalIlluminationRenderer(
                    runtime=runtime, config=config
                ) as renderer:
                    scene = ol.Scene()
                    scene.add_mesh(
                        [[-100, -100, 0], [100, -100, 0], [0, 100, 0]],
                        [[0, 1, 2]],
                        ol.Material(base_color=(0, 0, 0), program=graph),
                    )
                    camera = ol.PerspectiveCamera((0, 0, 2), (0, 0, 0))
                    render = renderer.render_wavefront if staged else renderer.render
                    actual = render(scene, camera, 4, 4, samples=1)
                    with pytest.raises(RuntimeError, match="attached renderers"):
                        resources.close()
                    with pytest.raises(RuntimeError):
                        uniform.close()
                    uniform.upload(np.array([0.4, 0.8, 1.2, 1], np.float32))
                    resources.synchronize()
                    changed = render(scene, camera, 4, 4, samples=1)
                    assert np.isfinite(actual).all()
                    assert np.any(changed != actual)
                # Compare identical evaluation through a resource-free graph.
                constant = MaterialGraph(
                    {
                        "rgb": MaterialNode(
                            "constant", value=(0.4, 0.8, 1.2), type="vec3"
                        )
                    },
                    {"emission": "rgb"},
                ).compile()
                fixed = ol.Scene()
                fixed.add_mesh(
                    [[-100, -100, 0], [100, -100, 0], [0, 100, 0]],
                    [[0, 1, 2]],
                    ol.Material(base_color=(0, 0, 0), program=constant),
                )
                with VulkanGlobalIlluminationRenderer(
                    runtime=runtime,
                    config=ol.RendererConfig(max_bounces=1),
                ) as renderer:
                    render = renderer.render_wavefront if staged else renderer.render
                    expected = render(fixed, camera, 4, 4, samples=1)
                np.testing.assert_allclose(changed, expected, rtol=1e-5, atol=1e-5)


def test_unbound_camera_graph_is_rejected():
    from ordinarylight.shaders.compiler import material_shader_source

    with pytest.raises(ValueError, match="material_resources"):
        material_shader_source("ray_query.comp", emission_graph())


@GPU
def test_camera_texture_buffer_uniform_match_transport():
    import vulkan as vk
    from ordinarylight.runtime import VulkanKernel, VulkanSampler, compile_compute
    from ordinarylight.pipeline.vulkan import (
        VulkanResource,
        VulkanResourceUse,
        VulkanPass,
        VulkanPassPipeline,
    )
    from ordinarylight.renderers.gi import VulkanGlobalIlluminationRenderer
    from ordinarylight.geometry import SdfSphere
    from ordinarylight.transport import (
        VulkanTransportScene,
        TransportMaterial,
        GpuSampleAccumulator,
        VulkanTransportIntegrator,
        surface_samples,
    )

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
    program = graph.compile()
    with ol.VulkanRuntime() as runtime:
        with (
            runtime.image(1, 1) as image,
            VulkanSampler(runtime) as sampler,
            runtime.buffer(16, data=np.ones(4, np.float32)) as gain,
            runtime.buffer(16, data=np.array([1, 0.5, 0.25, 1], np.float32)) as tint,
        ):
            supplied = {"texture": (image, sampler), "gain": gain, "tint": tint}
            with ol.VulkanMaterialResources(runtime, [program], supplied) as bindings:
                resource = VulkanResource.image(image)
                with VulkanKernel(
                    runtime,
                    compile_compute("""
#version 460
layout(local_size_x=1) in;
layout(set=0,binding=0,rgba32f) writeonly uniform image2D target;
void main() { imageStore(target,ivec2(0),vec4(1,2,3,1)); }
"""),
                    {0: resource},
                ) as producer:
                    with VulkanPassPipeline(
                        [
                            VulkanPass(
                                "fill",
                                (
                                    VulkanResourceUse(
                                        resource,
                                        vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                        vk.VK_ACCESS_SHADER_WRITE_BIT,
                                        vk.VK_IMAGE_LAYOUT_GENERAL,
                                    ),
                                ),
                                lambda command: producer.bind(command),
                                (1, 1, 1),
                            )
                        ]
                    ).execute(runtime) as ready:
                        bindings.synchronize(after=[ready])
                with VulkanGlobalIlluminationRenderer(
                    runtime=runtime,
                    config=ol.RendererConfig(
                        material_resources=bindings, max_bounces=1
                    ),
                ) as renderer:
                    scene = ol.Scene()
                    scene.add_mesh(
                        [[-100, -100, 0], [100, -100, 0], [0, 100, 0]],
                        [[0, 1, 2]],
                        ol.Material(base_color=(0, 0, 0), program=program),
                    )
                    camera = ol.PerspectiveCamera((0, 0, 2), (0, 0, 0))
                    actual = renderer.render_wavefront(scene, camera, 4, 4, samples=1)
                with (
                    VulkanTransportScene(
                        runtime,
                        custom_geometry=[SdfSphere(center=(100, 0, 0)).geometry()],
                        custom_materials=[TransportMaterial("emission", program=graph)],
                        material_resources=supplied,
                    ) as transport,
                    GpuSampleAccumulator(runtime, 1) as accumulator,
                ):
                    with VulkanTransportIntegrator(
                        transport,
                        surface_samples([[0, 0, 0]], [[0, 0, 1]], materials=0),
                        accumulator,
                    ) as integrator:
                        integrator.accumulate().wait()
                        expected = accumulator.means()[0]
                np.testing.assert_allclose(expected, [1, 1, 0.75])
                np.testing.assert_allclose(
                    actual[..., :3],
                    np.broadcast_to(expected, actual[..., :3].shape),
                    atol=1e-5,
                )


@GPU
def test_camera_resource_validation():
    from ordinarylight.materials import builtin_material

    graph = emission_graph()
    with ol.VulkanRuntime() as runtime:
        with runtime.buffer(16, data=np.ones(4, np.float32)) as uniform:
            with pytest.raises(ValueError, match="exactly match"):
                ol.VulkanMaterialResources(runtime, [graph], {})
            with pytest.raises(ValueError, match="exactly match"):
                ol.VulkanMaterialResources(
                    runtime, [graph], {"emission": uniform, "extra": uniform}
                )
            with pytest.raises(TypeError, match="compiled"):
                ol.VulkanMaterialResources(runtime, [object()], {})
            with ol.VulkanMaterialResources(
                runtime, [graph], {"emission": uniform}
            ) as resources:
                resources.validate([builtin_material])
                with pytest.raises(ValueError, match="require wavefront"):
                    ol.RendererConfig(
                        material_resources=resources,
                        wavefront_execution_strategy="megakernel",
                    )
            with pytest.raises(RuntimeError, match="closed"):
                resources.validate([graph])


@GPU
def test_camera_rough_metal_resource_matches_constant_scattering():
    from ordinarylight.renderers.gi import VulkanGlobalIlluminationRenderer

    graph = MaterialGraph(
        {
            "optics": MaterialNode("uniform", value="optics", type="vec4"),
            "rough": MaterialNode("components", ("optics",), value="x"),
            "metal": MaterialNode("components", ("optics",), value="y"),
        },
        {"roughness": "rough", "metallic": "metal"},
        resources=(MaterialResource("optics", "uniform"),),
    ).compile()
    constant = MaterialGraph(
        {
            "rough": MaterialNode("constant", value=0.4),
            "metal": MaterialNode("constant", value=1.0),
        },
        {"roughness": "rough", "metallic": "metal"},
    ).compile()
    with ol.VulkanRuntime() as runtime:
        with runtime.buffer(16, data=np.array([0.4, 1, 0, 0], np.float32)) as uniform:
            with ol.VulkanMaterialResources(
                runtime, [graph], {"optics": uniform}
            ) as resources:
                results = []
                for program, bindings in [(graph, resources), (constant, None)]:
                    scene = ol.Scene()
                    scene.add_mesh(
                        [[-100, -100, 0], [100, -100, 0], [0, 100, 0]],
                        [[0, 1, 2]],
                        ol.Material(base_color=(0.7, 0.4, 0.2), program=program),
                    )
                    scene.add_light(ol.PointLight((0, 0, 2), intensity=4))
                    with VulkanGlobalIlluminationRenderer(
                        runtime=runtime,
                        config=ol.RendererConfig(
                            material_resources=bindings, max_bounces=2
                        ),
                    ) as renderer:
                        results.append(
                            renderer.render_wavefront(
                                scene,
                                ol.PerspectiveCamera((0, 0, 2), (0, 0, 0)),
                                4,
                                4,
                                samples=8,
                            )
                        )
                assert np.isfinite(results[0]).all()
                assert np.any(results[1][..., :3] > 0)
                np.testing.assert_allclose(results[0], results[1], rtol=1e-4, atol=1e-5)
