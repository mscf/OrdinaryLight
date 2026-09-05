"""Graph dependency proofs and dynamic single-queue GPU integration."""

import os

import numpy as np
import pytest
import vulkan as vk

from ordinarylight.pipeline.vulkan import VulkanResource, VulkanResourceUse, VulkanPass
from ordinarylight.pipeline.graph import VulkanGraph


def stage(resource, *, read=False, write=False):
    return VulkanPass(
        "test",
        (
            VulkanResourceUse(
                resource,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                (vk.VK_ACCESS_SHADER_READ_BIT if read else 0)
                | (vk.VK_ACCESS_SHADER_WRITE_BIT if write else 0),
            ),
        ),
        lambda command: None,
    )


def test_versions_aliases_cycles_and_ambiguous_writers():
    resource = VulkanResource(object(), "buffer", 7, 16)
    alias = VulkanResource(object(), "buffer", 7, 16)
    graph = (
        VulkanGraph()
        .add("read", stage(alias, read=True))
        .add("write", stage(resource, write=True))
    )
    assert graph.compile().order == ("write", "read")
    graph = (
        VulkanGraph()
        .add("old_read", stage(alias, read=True), reads=[alias.version(0)])
        .add("write", stage(resource, write=True))
    )
    assert graph.compile().order == ("old_read", "write")
    with pytest.raises(ValueError, match="Ambiguous"):
        VulkanGraph().add("a", stage(resource, write=True)).add(
            "b", stage(alias, write=True)
        ).compile()
    graph = VulkanGraph().add(
        "later", stage(alias, read=True, write=True), writes=[alias.version(2)]
    )
    graph.add("first", stage(resource, write=True), writes=[resource.version(1)])
    assert graph.compile().order == ("first", "later")
    graph = (
        VulkanGraph()
        .add("a", stage(resource, write=True), after=["b"])
        .add("b", stage(alias, read=True), after=["a"])
    )
    with pytest.raises(ValueError, match="cycle"):
        graph.compile()
    with pytest.raises(ValueError, match="producer"):
        VulkanGraph().add(
            "missing", stage(resource, read=True), reads=[resource.version(1)]
        ).compile()

    image = VulkanResource(object(), "image", 9)
    draw = VulkanPass(
        "draw",
        (
            VulkanResourceUse(
                image,
                vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            ),
        ),
        lambda command: None,
    )
    sample = VulkanPass(
        "sample",
        (
            VulkanResourceUse(
                image,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_ACCESS_SHADER_READ_BIT,
                vk.VK_IMAGE_LAYOUT_GENERAL,
            ),
        ),
        lambda command: None,
    )
    assert VulkanGraph().add("sample", sample).add("draw", draw).compile().order == (
        "draw",
        "sample",
    )


GPU = pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_GRAPH") != "1",
    reason="opt-in graph GPU tests",
)


@GPU
def test_dynamic_geometry_graph_refits_activation_growth_and_persistent_output(
    monkeypatch,
):
    from ordinarylight.runtime import VulkanRuntime, VulkanOutput, VulkanFrameRing
    from ordinarylight.geometry import SdfSphere
    from ordinarylight.transport import (
        VulkanTransportScene,
        TransportMaterial,
        GpuSampleAccumulator,
        VulkanTransportIntegrator,
        ray_samples,
        intersect_rays,
    )

    with VulkanRuntime() as runtime:
        with VulkanTransportScene(
            runtime,
            custom_geometry=[SdfSphere().geometry()],
            custom_capacity=2,
            custom_materials=[TransportMaterial("diffuse", (0, 0, 0), (2, 1, 0.5))],
        ) as scene:
            with GpuSampleAccumulator(runtime, 1) as accumulator:
                with VulkanTransportIntegrator(
                    scene, ray_samples([[0, 0, 3]], [[0, 0, -1]]), accumulator
                ) as transport:
                    with (
                        VulkanOutput(runtime) as output,
                        output.prepare(accumulator.hdr) as target,
                        VulkanFrameRing(runtime, 2) as ring,
                    ):
                        hdr, image, kernel, tlas = (
                            accumulator.hdr,
                            target.image,
                            target.kernel,
                            scene.tlas.handle,
                        )
                        first = None
                        for step in range(4):
                            patch = (
                                {0: None, 1: SdfSphere(radius=0.5).geometry()}
                                if step % 2
                                else {0: SdfSphere().geometry(), 1: None}
                            )
                            graph = VulkanGraph()
                            graph.add("tone", target.operation())
                            graph.add("resolve", accumulator.resolve_operation())
                            graph.add(
                                "trace",
                                transport.accumulate_operation(),
                                after=["reset"],
                            )
                            graph.add("reset", accumulator.reset_operation())
                            graph.add(
                                "geometry",
                                scene.update_custom_geometry_operation(
                                    patch, mode="rebuild" if step == 3 else "refit"
                                ),
                            )
                            compiled = graph.compile()
                            ring.acquire()
                            if step == 1:
                                # Queue dependencies must not call wait on the prior token.
                                monkeypatch.setattr(
                                    first,
                                    "wait",
                                    lambda: pytest.fail(
                                        "per-stage host dependency wait"
                                    ),
                                )
                            ready = ring.submit(compiled)
                            if step == 0:
                                first = ready
                            if step == 1:
                                monkeypatch.undo()
                        ready.wait()
                        np.testing.assert_allclose(accumulator.means(), [[2, 1, 0.5]])
                        assert (
                            accumulator.hdr is hdr
                            and target.image is image
                            and target.kernel is kernel
                        )
                        assert scene.tlas.handle == tlas
                        hit = intersect_rays(scene, [[0, 0, 3]], [[0, 0, -1]])
                        assert hit["position_distance"][0, 3] == pytest.approx(
                            2.5, abs=1e-5
                        )
                        old_material = scene.resource("materials").handle
                        old_samples = transport.samples.buffer
                        stale = (
                            VulkanGraph()
                            .add("trace", transport.accumulate_operation())
                            .compile()
                        )
                        assert scene.reserve_custom_geometry(4)
                        assert scene.resource("materials").handle == old_material
                        assert transport.samples.buffer is old_samples
                        with pytest.raises(ValueError, match="recompile"):
                            stale.execute(runtime)
                        scene.update_custom_geometry(
                            {3: SdfSphere((0, 0, 2), 0.25).geometry()}
                        )
                        accumulator.reset()
                        transport.accumulate()
                        hit = intersect_rays(scene, [[0, 0, 3]], [[0, 0, -1]])
                        assert hit["identity"][0, 1] == 3
                        assert hit["position_distance"][0, 3] == pytest.approx(
                            0.75, abs=1e-5
                        )
                        scene.update_custom_geometry({1: None, 3: None})
                        accumulator.reset()
                        transport.accumulate(environment=(0.1, 0.2, 0.3))
                        np.testing.assert_allclose(
                            accumulator.means(), [[0.1, 0.2, 0.3]], rtol=1e-5
                        )


@GPU
def test_reflection_generated_access_and_gpu_producer_consumer(monkeypatch):
    from ordinarylight.runtime import VulkanRuntime, VulkanKernel, compile_compute
    from ordinarylight.pipeline.graph import reflected_operation

    with VulkanRuntime() as runtime:
        with runtime.buffer(16) as buffer:
            resource = VulkanResource.buffer(buffer)
            import ordinaryshade as osh

            @osh.compute(workgroup_size=(1, 1, 1))
            def write_value(
                values: osh.storage_buffer(osh.u32, access="write", binding=0),
            ):
                values[osh.global_invocation_id.x] = osh.u32(42)

            program = osh.compile(write_value)
            with VulkanKernel(
                runtime, compile_compute(program.source), {0: resource}
            ) as kernel:
                reflection = program.reflection
                op = reflected_operation(kernel, reflection, workgroups=(1, 1, 1))
                completion = (
                    VulkanGraph().add("producer", op).compile().execute(runtime)
                )
                with monkeypatch.context() as patch:
                    patch.setattr(
                        completion,
                        "wait",
                        lambda: pytest.fail("dependency waited on host"),
                    )
                    second = (
                        VulkanGraph()
                        .add("producer", op)
                        .compile()
                        .execute(runtime, after=[completion])
                    )
                second.wait()
                assert np.frombuffer(buffer.read(), np.uint32)[0] == 42


@GPU
@pytest.mark.skipif(
    not os.environ.get("DISPLAY"), reason="native presentation needs a display"
)
def test_presentation_reuse_resize_and_cancel():
    import glfw
    from ordinarylight.runtime import (
        VulkanRuntime,
        VulkanOutput,
        VulkanKernel,
        compile_compute,
    )
    from ordinarylight.pipeline.graph import VulkanOperation

    assert glfw.init()
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    window = glfw.create_window(
        96, 64, "OrdinaryLight graph lifecycle test", None, None
    )
    assert window is not None
    try:
        with VulkanRuntime(glfw_window=window) as runtime:
            with runtime.image(8, 8) as hdr:
                source = """#version 460
layout(local_size_x=8,local_size_y=8) in;
layout(set=0,binding=0,rgba32f) writeonly uniform image2D hdr;
void main(){ imageStore(hdr,ivec2(gl_GlobalInvocationID.xy),vec4(.2,.3,.4,1)); }
"""
                resource = VulkanResource.image(hdr)
                with VulkanKernel(
                    runtime, compile_compute(source), {0: resource}
                ) as writer:
                    from ordinarylight.pipeline.vulkan import VulkanPassPipeline

                    class ImageAlias:
                        def __init__(self):
                            self.runtime = runtime
                            self.layout = hdr.layout

                        def require_open(self):
                            hdr.require_open()

                    alias = ImageAlias()
                    VulkanPassPipeline(
                        [
                            VulkanPass(
                                "initialize",
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
                            ),
                            VulkanPass(
                                "alias_transition",
                                (
                                    VulkanResourceUse(
                                        VulkanResource(alias, "image", hdr.image),
                                        vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                                        vk.VK_ACCESS_TRANSFER_READ_BIT,
                                        vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    ),
                                ),
                                lambda command: None,
                            ),
                        ]
                    ).execute(runtime).wait()
                    assert (
                        hdr.layout
                        == alias.layout
                        == vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
                    )
                    with VulkanOutput(runtime) as output, output.prepare(hdr) as target:
                        persistent_image, persistent_kernel = (
                            target.image,
                            target.kernel,
                        )
                        abandoned = output.present_operation(target)
                        assert abandoned is not None
                        output.cancel_presentation()
                        with pytest.raises(RuntimeError, match="single-use"):
                            abandoned.execute(runtime)
                        rendered = 0
                        for frame in range(6):
                            if frame == 2:
                                glfw.set_window_size(window, 128, 80)
                            glfw.poll_events()
                            presentation = output.present_operation(
                                target, surface_size=glfw.get_framebuffer_size(window)
                            )
                            if presentation is None:
                                continue
                            write = VulkanOperation(
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
                                        writer.bind,
                                        (1, 1, 1),
                                    )
                                ],
                                validate=writer.require_open,
                            )
                            graph = (
                                VulkanGraph()
                                .add("present", presentation)
                                .add("tone", target.operation())
                                .add("fill", write)
                            )
                            graph.compile().execute(runtime)
                            rendered += 1
                        assert rendered >= 2
                        target.wait()
                        assert (
                            target.image is persistent_image
                            and target.kernel is persistent_kernel
                        )
                        pixels = np.frombuffer(output.read(target), np.uint8).reshape(
                            (-1, 4)
                        )
                        assert np.all(pixels[:, 3] == 255) and pixels[:, :3].min() > 0
    finally:
        glfw.destroy_window(window)
        glfw.terminate()


@GPU
def test_uniform_and_sampled_texture_graph():
    from ordinarylight.runtime import (
        VulkanRuntime,
        VulkanKernel,
        VulkanSampler,
        compile_compute,
    )
    from ordinarylight.pipeline.graph import reflected_operation
    import ordinaryshade as osh

    @osh.structure
    class Parameters:
        gain: osh.vec4

    @osh.compute(workgroup_size=(1, 1, 1))
    def consume(
        pixels: osh.storage_buffer(osh.vec4, access="write", binding=0),
        params: osh.uniform_buffer(Parameters, binding=1),
        texture: osh.sampled_texture_2d(binding=2),
        filtering: osh.sampler(binding=3),
    ):
        pixels[0] = (
            texture.sample_level_with(filtering, osh.vec2(0.5, 0.5), 0.0) * params.gain
        )

    program = osh.compile(consume)
    producer_source = """#version 460
layout(local_size_x=1) in;
layout(set=0,binding=0,rgba32f) writeonly uniform image2D output_image;
void main() { imageStore(output_image,ivec2(0),vec4(1,2,3,4)); }
"""
    with (
        VulkanRuntime() as runtime,
        runtime.buffer(16) as output,
        runtime.buffer(16, data=np.array([2, 3, 4, 5], np.float32)) as uniform,
        runtime.image(1, 1) as texture,
        VulkanSampler(runtime) as sampler,
    ):
        image_resource = VulkanResource.image(texture)
        with (
            VulkanKernel(
                runtime, compile_compute(producer_source), {0: image_resource}
            ) as producer,
            VulkanKernel(
                runtime,
                compile_compute(program.source),
                {
                    0: VulkanResource.buffer(output),
                    1: VulkanResource.uniform_buffer(uniform),
                    2: VulkanResource.sampled_image(texture),
                    3: VulkanResource.sampler(sampler),
                },
            ) as consumer,
        ):
            with pytest.raises(RuntimeError, match="borrowers"):
                sampler.close()
            graph = VulkanGraph().add(
                "consumer",
                reflected_operation(consumer, program.reflection, workgroups=(1, 1, 1)),
            )
            graph.add(
                "producer",
                VulkanPass(
                    "fill",
                    (
                        VulkanResourceUse(
                            image_resource,
                            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            vk.VK_ACCESS_SHADER_WRITE_BIT,
                            vk.VK_IMAGE_LAYOUT_GENERAL,
                        ),
                    ),
                    lambda command: producer.bind(command),
                    (1, 1, 1),
                ),
            )
            compiled = graph.compile()
            assert compiled.order == ("producer", "consumer")
            compiled.execute(runtime).wait()
            np.testing.assert_allclose(
                np.frombuffer(output.read(), np.float32), [2, 6, 12, 20]
            )


def test_buffer_range_hazards_and_overlapping_versions():
    buffer = VulkanResource(object(), "buffer", 71, 64)
    left, right = buffer.byte_range(0, 32), buffer.byte_range(32, 32)
    graph = (
        VulkanGraph()
        .add("left", stage(left, write=True))
        .add("right", stage(right, write=True))
    )
    graph.add("read_all", stage(buffer, read=True))
    assert graph.compile().order == ("left", "right", "read_all")
    # An explicit reverse edge between disjoint writers is legal.
    graph = VulkanGraph().add("left", stage(left, write=True), after=["right"])
    graph.add("right", stage(right, write=True))
    assert graph.compile().order == ("right", "left")
    overlap = buffer.byte_range(16, 32)
    with pytest.raises(ValueError, match="Ambiguous"):
        VulkanGraph().add("left", stage(left, write=True)).add(
            "overlap", stage(overlap, write=True)
        ).compile()
    graph = VulkanGraph().add("old", stage(left, read=True), reads=[left.version(0)])
    graph.add("write_all", stage(buffer, write=True))
    assert graph.compile().order == ("old", "write_all")
    with pytest.raises(ValueError, match="outside"):
        left.byte_range(16, 32)


@GPU
def test_gpu_discovered_geometry_bounds_validation_and_growth():
    from ordinarylight.runtime import VulkanRuntime, VulkanKernel, compile_compute
    from ordinarylight.geometry import SdfSphere
    from ordinarylight.transport import (
        VulkanTransportScene,
        TransportMaterial,
        VulkanTransportIntegrator,
        GpuSampleAccumulator,
        GpuCustomGeometry,
        ray_samples,
    )

    source = """#version 460
layout(local_size_x=1) in;
struct Record { vec4 lower; vec4 upper; vec4 parameters; uvec4 metadata; };
layout(set=0,binding=0,std430) writeonly buffer Slots { Record slots[]; };
void main() {
    slots[0]=Record(vec4(-1,-1,-1,0),vec4(1,1,1,0),vec4(0,0,0,1),uvec4(0,0,0xffffffffu,42));
    slots[1]=Record(vec4(1),vec4(-1),vec4(0),uvec4(0,0,0xffffffffu,43));
}
"""
    with (
        VulkanRuntime() as runtime,
        VulkanTransportScene(
            runtime,
            custom_geometry=[SdfSphere(center=(10, 0, 0)).geometry()],
            custom_capacity=2,
            custom_materials=[TransportMaterial("emission", emission=(2, 3, 4))],
        ) as scene,
        runtime.buffer(128) as input_buffer,
        GpuSampleAccumulator(runtime, 1) as accumulation,
    ):
        with (
            VulkanKernel(
                runtime,
                compile_compute(source),
                {0: VulkanResource.buffer(input_buffer)},
            ) as producer,
            GpuCustomGeometry(scene, input_buffer) as updater,
            VulkanTransportIntegrator(
                scene,
                ray_samples([[0, 0, 3]], [[0, 0, -1]]),
                accumulation,
            ) as transport,
        ):
            graph = VulkanGraph().add("trace", transport.accumulate_operation())
            graph.add("refit", updater.operation())
            graph.add(
                "discover",
                VulkanPass(
                    "discover",
                    (
                        VulkanResourceUse(
                            VulkanResource.buffer(input_buffer),
                            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                            vk.VK_ACCESS_SHADER_WRITE_BIT,
                        ),
                    ),
                    lambda command: producer.bind(command),
                    (1, 1, 1),
                ),
            )
            compiled = graph.compile()
            assert compiled.order == ("discover", "refit", "trace")
            compiled.execute(runtime).wait()
            np.testing.assert_allclose(accumulation.means(), [[2, 3, 4]])
            np.testing.assert_array_equal(updater.read_diagnostics(), [0, 2])
        # Explicit host allocation boundary preserves the last GPU-authored state.
        scene.reserve_custom_geometry(4)
        assert scene.custom_geometry[0].identity == 42
        assert scene.custom_geometry[1] is None
