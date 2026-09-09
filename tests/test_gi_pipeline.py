"""CPU contract checks for the native frame composition boundary."""

from types import SimpleNamespace

import pytest

from ordinarylight.pipeline import RenderPipeline, RenderStage
from ordinarylight.pipeline.gi import GiFrame, create_gi_pipeline, record_gi_pipeline
from ordinarylight.targets.vulkan.api import VulkanGlfwPresenter


def frame():
    return GiFrame(None, None, 0, (160, 120), (640, 480), {}, {})


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
