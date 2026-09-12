"""External recording publishes resource/history state only after submission."""

from types import SimpleNamespace

import pytest
import vulkan as vk

from ordinarylight.pipeline.graph import VulkanGraph, VulkanOperation
from ordinarylight.pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse


def test_external_recording_defers_publication_and_supports_replay(monkeypatch):
    monkeypatch.setattr(vk, "vkCmdPipelineBarrier", lambda *a: None)
    monkeypatch.setattr(vk, "vkCmdDispatch", lambda *a: None)
    runtime = SimpleNamespace(require_open=lambda: None)

    class Owner:
        layout = vk.VK_IMAGE_LAYOUT_UNDEFINED

        def require_open(self):
            pass

    owner = Owner()
    owner.runtime = runtime
    resource = VulkanResource(owner, "image", vk.ffi.cast("VkImage", 1))
    calls = []
    operation = VulkanOperation(
        [
            VulkanPass(
                "write",
                (
                    VulkanResourceUse(
                        resource,
                        vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        vk.VK_ACCESS_SHADER_WRITE_BIT,
                        vk.VK_IMAGE_LAYOUT_GENERAL,
                    ),
                ),
                lambda command: calls.append("record"),
                (1, 1, 1),
            )
        ],
        submitted=lambda completion: calls.append(completion),
    )
    graph = VulkanGraph().add("work", operation).compile()
    recording = graph.prepare_recording(runtime)
    with pytest.raises(RuntimeError, match="Record commands"):
        recording.submitted("early")
    assert owner.layout == vk.VK_IMAGE_LAYOUT_UNDEFINED
    recording.record(None)
    assert calls == ["record"]
    assert owner.layout == vk.VK_IMAGE_LAYOUT_UNDEFINED
    recording.submitted("first")
    assert owner.layout == vk.VK_IMAGE_LAYOUT_GENERAL
    # Replayed commands publish the latest completion without re-recording.
    recording.submitted("replay")
    assert calls == ["record", "first", "replay"]
    with pytest.raises(RuntimeError, match="new recording"):
        recording.record(None)


def test_recording_failure_cannot_publish_or_retry(monkeypatch):
    runtime = SimpleNamespace(require_open=lambda: None)

    def fail(command):
        raise RuntimeError("record failed")

    operation = VulkanOperation([VulkanPass("fail", (), fail)])
    recording = (
        VulkanGraph().add("fail", operation).compile().prepare_recording(runtime)
    )
    with pytest.raises(RuntimeError, match="record failed"):
        recording.record(None)
    with pytest.raises(RuntimeError, match="Record commands"):
        recording.submitted(None)


def test_callback_failure_still_publishes_other_consumers(monkeypatch):
    monkeypatch.setattr(vk, "vkCmdPipelineBarrier", lambda *a: None)
    runtime = SimpleNamespace(require_open=lambda: None)
    published = []

    def fail(completion):
        raise RuntimeError("application failed")

    graph = VulkanGraph()
    for name, callback in (("app", fail), ("kernel", published.append)):
        graph.add(name, VulkanOperation(
            [VulkanPass(name, (), lambda command: None)], submitted=callback,
        ))
    recording = graph.compile().prepare_recording(runtime)
    recording.record(None)
    completion = object()
    with pytest.raises(RuntimeError, match="application failed"):
        recording.submitted(completion)
    assert published == [completion]
    with pytest.raises(RuntimeError, match="new recording"):
        recording.record(None)
