"""CPU contract checks for the native frame composition boundary."""

from types import SimpleNamespace

import pytest

from ordinarylight.pipeline import RenderPipeline, RenderStage
from ordinarylight.pipeline.gi import GiFrame, create_gi_pipeline, record_gi_pipeline
from ordinarylight.targets.vulkan.api import VulkanGlfwPresenter


def frame():
    return GiFrame(None, None, 0, (160, 120), (640, 480), {}, {})


def test_primary_hit_export_requires_a_supported_execution_kernel():
    from ordinarylight.targets.vulkan.api import RendererConfig, _resolve_execution_strategy
    from ordinarylight.wavefront import PRIMARY_HIT_DTYPE, primary_bindings

    config = RendererConfig(wavefront_primary_hits=True, wavefront_execution_strategy="auto")
    # Auto must not select a megakernel that lacks primary-hit export, even for
    # a scene which normally triggers its triangle/transmission thresholds.
    assert _resolve_execution_strategy(config, object()) == "wavefront"
    for strategy in ("megakernel", "hybrid", "persistent"):
        with pytest.raises(ValueError, match="Primary hit outputs"):
            RendererConfig(wavefront_primary_hits=True, wavefront_execution_strategy=strategy)
    with pytest.raises(TypeError, match="bools"):
        primary_bindings(primary_hits=1)
    assert PRIMARY_HIT_DTYPE.itemsize == 96


def test_default_and_temporal_upscaling_dependency_order():
    recorded = []

    def recorder(name):
        return lambda context: recorded.append((name, context["frame"].render_extent))

    pipeline = create_gi_pipeline(
        trace=recorder("trace"),
        indirect=recorder("indirect"),
        denoise=recorder("denoise"),
        upscale=recorder("upscale"),
        reconstruct=recorder("display"),
    )
    record_gi_pipeline(pipeline, frame())
    assert [name for name, extent in recorded] == [
        "trace",
        "indirect",
        "denoise",
        "upscale",
        "display",
    ]
    assert all(extent == (160, 120) for name, extent in recorded)
    # An application cannot accidentally feed display conversion before FSR.
    with pytest.raises(ValueError, match="gi.upscaled_hdr"):
        RenderPipeline(
            (pipeline.stages[-1],), pipeline.initial_resources | {"gi.guides"}
        )


def test_application_can_insert_a_stage_using_produced_hdr():
    calls = []
    pipeline = create_gi_pipeline(
        trace=lambda context: calls.append("trace"),
        reconstruct=lambda context: calls.append("display"),
    )

    def builder(default, frame):
        return default.insert_before(
            "gi.reconstruct",
            RenderStage(
                "app.exposure",
                reads={"gi.hdr"},
                writes={"gi.hdr"},
                recorder=lambda context: calls.append("app"),
            ),
        )

    record_gi_pipeline(pipeline, frame(), builder=builder)
    assert calls == ["trace", "app", "display"]
    with pytest.raises(ValueError, match="gi.display"):
        record_gi_pipeline(pipeline, frame(), builder=lambda p, f: RenderPipeline([]))
    with pytest.raises(TypeError, match="RenderPipeline"):
        record_gi_pipeline(pipeline, frame(), builder=lambda p, f: None)


def test_frame_resources_are_borrowed_read_only_snapshots():
    resources = {"hdr": object()}
    f = GiFrame(None, None, 0, (160, 120), (640, 480), resources, {})
    resources.clear()
    assert "hdr" in f.resources
    with pytest.raises(TypeError):
        f.resources["hdr"] = None
    with pytest.raises(ValueError):
        GiFrame(None, None, 0, (0, 120), (640, 480), {}, {})


def test_changing_composition_invalidates_commands_and_history():
    state = {
        "wavefront_command_key": ("cached",),
        "wavefront_relax_history_valid": True,
    }
    resets = []
    core = SimpleNamespace(
        window_frames=[state], reset_accumulation=lambda: resets.append(True)
    )
    presenter = VulkanGlfwPresenter.__new__(VulkanGlfwPresenter)
    presenter._core = core

    def builder(p, f):
        return p

    presenter.set_gi_pipeline_builder(builder)
    assert core.gi_pipeline_builder is builder
    assert state["wavefront_command_key"] is None
    assert not state["wavefront_relax_history_valid"]
    assert resets == [True]
    presenter.set_gi_pipeline_builder()
    assert core.gi_pipeline_builder is None


def test_lighting_only_composition_and_hdr_processing_order():
    calls = []
    record = lambda name: lambda context: calls.append(name)
    for upscale in (None, record("upscale")):
        calls.clear()
        pipeline = create_gi_pipeline(
            trace=record("trace"), denoise=record("denoise"),
            upscale=upscale, reconstruct=record("display"),
        )
        pipeline = frame().process_hdr(pipeline, RenderStage(
            "app.average", reads={"gi.hdr"}, writes={"gi.hdr"},
            recorder=record("average"),
        ))
        record_gi_pipeline(pipeline, frame())
        assert calls == ["trace", "denoise", "average"] + (
            ["upscale"] if upscale else []
        ) + ["display"]
    calls.clear()
    lighting = create_gi_pipeline(trace=record("trace"), denoise=record("denoise"))
    record_gi_pipeline(lighting, frame(), required_outputs={"gi.hdr"})
    assert calls == ["trace", "denoise"]
    with pytest.raises(ValueError, match="gi.display"):
        record_gi_pipeline(lighting, frame())


def test_reusable_composition_and_separate_history_invalidation():
    state = {
        "wavefront_command_key": ("cached",),
        "wavefront_relax_history_valid": True,
        "wavefront_history_ready_pending": True,
    }
    core = SimpleNamespace(window_frames=[state], reset_accumulation=lambda: None)
    presenter = VulkanGlfwPresenter.__new__(VulkanGlfwPresenter)
    presenter._core = core
    presenter.set_gi_pipeline_builder(lambda p, f: p, reuse_commands=True)
    assert core.gi_reuse_commands
    state["wavefront_command_key"] = ("cached",)
    state["wavefront_relax_history_valid"] = True
    presenter.invalidate_gi_commands()
    assert state["wavefront_command_key"] is None
    assert state["wavefront_relax_history_valid"]
    presenter.invalidate_gi_history()
    assert not state["wavefront_relax_history_valid"]
    # Never discard a pending semaphore signal when rejecting history contents.
    assert state["wavefront_history_ready_pending"]
    with pytest.raises(TypeError):
        presenter.set_gi_pipeline_builder(reuse_commands="yes")
    presenter._core = None
    with pytest.raises(RuntimeError, match="closed"):
        presenter.invalidate_gi_history()


def test_named_image_views_reject_retirement_and_are_cached():
    from ordinarylight.targets.vulkan.gi_images import native_gi_images

    class Core:
        pass

    core = Core()
    core.device = object()
    core.runtime = SimpleNamespace(require_open=lambda: None)
    core.swapchain_generation = 1
    core.window_frames = [{
        "wavefront_allocation_extent": (160, 120),
        "wavefront_hdr_image": object(), "wavefront_hdr_view": object(),
    }]
    images = native_gi_images(core, 0)
    assert native_gi_images(core, 0) is images
    image = images["hdr"]
    assert (image.width, image.height) == (160, 120)
    image.require_open()
    with pytest.raises(TypeError):
        images["hdr"] = None
    core.window_frames = [dict(core.window_frames[0])]
    with pytest.raises(RuntimeError, match="retired"):
        image.require_open()


def test_presenter_reconfiguration_validates_before_mutation():
    from ordinarylight.targets.vulkan.api import RendererConfig
    presenter = VulkanGlfwPresenter.__new__(VulkanGlfwPresenter)
    presenter.config = RendererConfig()
    original = presenter.config
    presenter._core = SimpleNamespace(
        config=original, window_frames=[], reset_accumulation=lambda: None,
        swapchain_extent=(160, 120),
    )
    assert presenter.reconfigure() is original
    with pytest.raises(ValueError):
        presenter.reconfigure(max_bounces=-1)
    with pytest.raises(ValueError, match="recreation required"):
        presenter.reconfigure(device_name="another device")
    assert presenter._core.config is original
    presenter.reconfigure(wavefront_exposure=2.0)
    assert presenter.config.wavefront_exposure == 2.0
    assert presenter._core.swapchain_extent == (160, 120)
    presenter.reconfigure(samples_per_pixel=2)
    assert presenter._core.swapchain_extent is None
    assert presenter._core.config is presenter.config
