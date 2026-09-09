"""Desktop GPU regressions for composed native GI; opt in with VULKAN_GRAPH.

The direct recorder is a deliberately independent oracle for dispatch parity.
These tests require a desktop display and the built FSR2 bridge.
"""

import os
from types import SimpleNamespace
import numpy as np
import pytest
import vulkan as vk
import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.integrations.raster_workbench import _gi_config
from ordinarylight.pipeline import RenderStage
from tools.denoiser_motion.glass_detail import fixture, pose
import ordinarylight.targets.vulkan.primary_graph as primary_graph

pytestmark = pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_GRAPH") != "1",
    reason="opt-in desktop Vulkan integration",
)


@pytest.fixture
def stress_multiplier(request):
    value = int(os.environ.get("ORDINARYLIGHT_STRESS_MULTIPLIER", "1"))
    if not 1 <= value <= 100:
        pytest.fail("ORDINARYLIGHT_STRESS_MULTIPLIER must be between 1 and 100")
    request.node.user_properties.append(("stress_multiplier", value))
    return value


@pytest.fixture
def desktop():
    glfw = load_glfw()
    assert glfw.init(), "Desktop GPU tests require a working display"
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = None
    try:
        window = glfw.create_window(320, 240, "GI regression", None, None)
        assert window, "Could not create Vulkan test window"
        yield glfw, window
    finally:
        if window:
            glfw.destroy_window(window)
        glfw.terminate()


def direct_primary(executor, command, pipeline, slot, constants, groups):
    vk.vkCmdBindPipeline(command, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
    vk.vkCmdBindDescriptorSets(
        command,
        vk.VK_PIPELINE_BIND_POINT_COMPUTE,
        executor.primary_pipeline_layout,
        0,
        1,
        [executor.primary_sets[slot]],
        0,
        None,
    )
    executor.core._bind_material_resources(command, executor.primary_pipeline_layout)
    vk.vkCmdPushConstants(
        command,
        executor.primary_pipeline_layout,
        vk.VK_SHADER_STAGE_COMPUTE_BIT,
        0,
        len(constants),
        vk.ffi.from_buffer(constants),
    )
    vk.vkCmdDispatch(command, *groups)


@pytest.mark.parametrize("mode", ["bilinear", "fsr1-shade", "fsr2"])
def test_direct_composed_motion_parity(desktop, monkeypatch, tmp_path, mode):
    _glfw, window = desktop
    composed_primary = primary_graph.record_primary
    results = []
    for custom in (False, True):
        monkeypatch.setattr(
            primary_graph,
            "record_primary",
            composed_primary if custom else direct_primary,
        )
        scene, glass, bars = fixture()
        config = _gi_config(
            SimpleNamespace(id="glass-detail", renderer={}),
            render_scale=0.25,
            upscale_filter=mode,
            capture=True,
        )
        calls = []
        with ol.VulkanGlfwPresenter(window, config=config) as presenter:

            def builder(default, frame):
                assert frame.render_extent == (80, 60)
                assert frame.output_extent == (320, 240)

                def checkpoint(context):
                    assert context["frame"] is frame
                    assert frame.resources["wavefront_hdr_image"] is not None
                    calls.append(frame.slot)
                    # Exercise recording an application command in the
                    # renderer's secondary buffer without changing pixels.
                    vk.vkCmdPipelineBarrier(
                        frame.command,
                        vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0,
                        0,
                        None,
                        0,
                        None,
                        0,
                        None,
                    )

                return default.insert_before(
                    "gi.reconstruct",
                    RenderStage(
                        "app.checkpoint",
                        reads={"gi.hdr"},
                        recorder=checkpoint,
                    ),
                )

            if custom:
                presenter.set_gi_pipeline_builder(builder)
            frames = []
            for index in range(8):
                presenter.present_wavefront(
                    scene,
                    pose(scene, glass, bars, "camera", index * 0.001),
                    320,
                    240,
                )
                image = presenter.capture_wavefront_hdr()
                assert np.isfinite(image).all()
                assert np.any(image[..., :3] > 0), "Vacuous black-frame parity"
                frames.append(image)
            if custom:
                assert len(calls) == 8
                presenter.set_gi_pipeline_builder(None)
                presenter.present_wavefront(
                    scene, pose(scene, glass, bars, "camera", 0.008), 320, 240
                )
            results.append(np.array(frames))
    if not np.array_equal(*results):
        np.savez_compressed(
            tmp_path / "primary-parity.npz", direct=results[0], composed=results[1]
        )
    np.testing.assert_array_equal(results[0], results[1])


@pytest.mark.parametrize("mode", ["fsr1-shade", "fsr2"])
def test_native_resize_scale_and_retirement(desktop, mode):
    glfw, window = desktop
    scene, glass, bars = fixture()
    cfg = _gi_config(
        SimpleNamespace(id="glass-detail", renderer={}),
        present=True,
        upscale_filter=mode,
    )
    with ol.VulkanGlfwPresenter(window, config=cfg) as p:
        camera = pose(scene, glass, bars, "camera", 0)
        hits = 0
        for i in range(8):
            p.present_wavefront(scene, camera, 320, 240)
            hits += int(p.last_timings["wavefront_command_cache_hit"])
        preparation = {
            slot: stage
            for slot, (
                _key,
                stage,
            ) in p._core.wavefront_executor.relax_preparation_stages.items()
        }
        assert len(preparation) == 2
        assert p._core.wavefront_executor.relax_prepare_pipeline is None
        assert p._core.wavefront_executor.relax_prepare_sets == []
        resolve = dict(p._core.wavefront_executor.path_resolve_stages)
        assert len(resolve) == 2
        assert p._core.wavefront_executor.wavefront_image_pipeline is None
        assert p._core.wavefront_executor.wavefront_image_sets == []
        graph_cache = p._core.wavefront_executor.relax_preparation_graphs
        original_graphs = {slot: dict(graphs) for slot, graphs in graph_cache.items()}
        assert original_graphs and all(original_graphs.values())
        # Force rebuilt native commands to exercise graph-cache reuse,
        # rather than merely replaying cached Vulkan commands.
        for _ in range(2):
            for frame in p._core.window_frames:
                frame["wavefront_command_key"] = None
            p.present_wavefront(scene, camera, 320, 240)
        assert all(
            graph_cache[slot][key] is graph
            for slot, graphs in original_graphs.items()
            for key, graph in graphs.items()
        )
        assert all(len(graphs) <= 32 for graphs in graph_cache.values())
        adapter = p._core.denoiser_graph
        reconstruction = p._core.reconstruction_graph
        assert p._core.wavefront_executor.reconstruct_pipeline is None
        assert p._core.wavefront_executor.reconstruct_sets == []
        assert all(not stage._owns_scratch for stage in adapter.spatial)
        assert all(not history._owns_images for history in adapter.histories)
        assert p._core.wavefront_executor.relax_temporal_pipeline is None
        assert p._core.wavefront_executor.relax_atrous_pipeline is None
        if mode != "fsr2":
            assert hits > 0
        for extent in ((160, 120), (80, 60), (320, 240)):
            for i in range(3):
                p.present_wavefront(scene, camera, 320, 240, render_extent=extent)
                assert p._core.denoiser_graph is adapter
                assert all(
                    p._core.wavefront_executor.relax_preparation_stages[slot][1]
                    is stage
                    for slot, stage in preparation.items()
                )
                if mode != "fsr2":
                    assert p._core.reconstruction_graph is reconstruction
                assert tuple(p.last_timings["wavefront_render_extent"]) == extent
        glfw.set_window_size(window, 400, 300)
        glfw.poll_events()
        for i in range(3):
            p.present_wavefront(scene, camera, 400, 300)
        assert all(kernel.closed for _key, kernel, _graphs in resolve.values())
        assert all(stage.closed for stage in preparation.values())
        assert adapter.closed
        assert reconstruction.closed
        assert p._core.denoiser_graph is not adapter
        assert p._core.swapchain_extent == (400, 300)
        adapter = p._core.denoiser_graph
        # Restore before the next mode and exercise resize in both directions.
        glfw.set_window_size(window, 320, 240)
        glfw.poll_events()
        p.present_wavefront(scene, camera, 320, 240)
        assert adapter.closed
        active = p._core.denoiser_graph
        active_reconstruction = p._core.reconstruction_graph
    assert active.closed
    assert active_reconstruction.closed


@pytest.mark.parametrize("mode", ["fsr1-shade", "fsr2"])
def test_repeated_scene_replacement(desktop, mode, stress_multiplier):
    _glfw, window = desktop
    scenes = [fixture(), fixture()]
    cfg = _gi_config(
        SimpleNamespace(id="glass-detail", renderer={}),
        render_scale=0.25,
        upscale_filter=mode,
        capture=True,
    )
    retired = []
    with ol.VulkanGlfwPresenter(window, config=cfg) as presenter:
        previous = None
        for switch in range(6 * stress_multiplier):
            scene, glass, bars = scenes[switch % 2]
            camera = pose(scene, glass, bars, "glass", 0.15 * (switch % 2))
            for frame in range(3):
                presenter.present_wavefront(scene, camera, 320, 240)
                image = presenter.capture_wavefront_hdr()
                assert np.isfinite(image).all()
                assert np.any(image[..., :3] > 0)
                active = presenter._core.scene_resources
                assert active.scene is scene
                if frame == 0 and previous is not None:
                    assert active is not previous
                    with pytest.raises(RuntimeError, match="closed"):
                        previous.require_open()
                    retired.append(previous)
            previous = active
        retired.append(previous)
    for resource in retired:
        with pytest.raises(RuntimeError, match="closed"):
            resource.require_open()


def test_repeated_gi_raster_surface_handoff(desktop, stress_multiplier):
    """Retire each device/swapchain before reusing the viewer's external surface."""
    from ordinarylight.runtime import VulkanRuntime
    from ordinarylight.integrations.raster_workbench import _raster_config

    _glfw, window = desktop
    scene, glass, bars = fixture()
    camera = pose(scene, glass, bars, "camera", 0)
    settings = {"denoiser_enabled": True}
    gi_config = _gi_config(
        SimpleNamespace(id="glass-detail", renderer=settings),
        render_scale=0.25,
        upscale_filter="fsr1-shade",
        capture=True,
    )
    raster_config = _raster_config(settings, shadows=False, shadow_map_size=256)
    program = ol.RasterProgram.scene(
        target="spirv",
        validate=False,
        material_programs=scene.material_programs(ol.builtin_material),
    )
    with VulkanRuntime(glfw_window=window) as surface_owner:
        for _ in range(3 * stress_multiplier):
            with ol.VulkanSurfacePresenter(
                surface_owner.instance, surface_owner.surface, config=gi_config
            ) as presenter:
                for _frame in range(3):
                    presenter.present_wavefront(scene, camera, 320, 240)
                    image = presenter.capture_wavefront_hdr()
                    assert np.isfinite(image).all()
                    assert np.any(image[..., :3] > 0)
                stage = presenter._core.denoiser_graph
            assert stage.closed
            renderer = ol.renderers.raster.VulkanRasterRenderer(
                program,
                config=raster_config,
                instance=surface_owner.instance,
                surface=surface_owner.surface,
            )
            try:
                for frame in range(3):
                    renderer.present_frame(scene, camera, 320, 240, frame_index=frame)
            finally:
                renderer.close()
            surface_owner.require_open()
