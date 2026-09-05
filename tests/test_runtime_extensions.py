"""Compatibility and executable extension contracts; GPU checks are opt-in."""

import os
from threading import RLock
from unittest.mock import patch

import numpy as np
import pytest

from ordinarylight.transport import SampleHistory, SURFACE_SAMPLE_DTYPE, shader_source


def test_history_camera_and_local_geometry_invalidation():
    history = SampleHistory()
    history.set(("diffuse", 7), "diffuse7", dependencies={"geometry", "lights"})
    history.set(
        ("specular", 7), "specular7", dependencies={"geometry", "lights", "camera"}
    )
    history.set(("diffuse", 8), "diffuse8", dependencies={"geometry", "lights"})
    assert history.invalidate("camera") == {("specular", 7): "specular7"}
    assert history.get(("diffuse", 7)) == "diffuse7"
    assert history.invalidate("geometry", identities=[("diffuse", 7)]) == {
        ("diffuse", 7): "diffuse7"
    }
    assert history.get(("diffuse", 8)) == "diffuse8"


def test_surface_contract_has_independent_normals_media_and_identity():
    assert SURFACE_SAMPLE_DTYPE.itemsize == 96
    assert [
        SURFACE_SAMPLE_DTYPE.fields[name][1] for name in SURFACE_SAMPLE_DTYPE.names
    ] == list(range(0, 96, 16))
    sample = np.zeros(1, SURFACE_SAMPLE_DTYPE)
    sample["identity"][0] = [92, 3, 2, 0]
    sample["media"][0, :2] = [5, 9]
    sample["geometric_normal"][0, 2] = 1
    sample["shading_normal"][0, 1] = 1
    assert not np.array_equal(sample["geometric_normal"], sample["shading_normal"])
    source = shader_source("contracts")
    assert "surface.media.y : surface.media.x" in source
    assert "geometric_normal" in source


def test_runtime_rejects_early_close_without_destroying_device():
    from ordinarylight.runtime import VulkanRuntime

    runtime = object.__new__(VulkanRuntime)
    runtime.lock = RLock()
    runtime._closed = False
    runtime._consumers = set()
    owner = object()
    runtime.retain(owner)
    with pytest.raises(RuntimeError, match="consumers"):
        runtime.close()
    assert not runtime._closed
    runtime.release(owner)
    assert not runtime._consumers


def test_typed_pass_validation():
    from ordinarylight.pipeline.vulkan import (
        VulkanPass,
        VulkanResource,
        VulkanResourceUse,
        VulkanPassPipeline,
    )
    import vulkan as vk

    image = VulkanResource(object(), "image", 17)
    with pytest.raises(ValueError, match="layout"):
        VulkanResourceUse(
            image,
            vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.VK_ACCESS_SHADER_WRITE_BIT,
        )
    with pytest.raises(ValueError, match="three positive"):
        VulkanPass("sample", (), lambda command: None, (0, 1, 1))
    stage = VulkanPass("sample", (), lambda command: None, (17, 3, 1))
    with pytest.raises(ValueError, match="unique"):
        VulkanPassPipeline([stage, stage])


GPU = pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_RUNTIME") != "1",
    reason="opt-in Vulkan GPU test",
)


@GPU
def test_standalone_surface_transport():
    from examples.runtime_surface_samples import run
    from ordinarylight.targets.vulkan.core import VulkanRayQueryCore

    with patch.object(
        VulkanRayQueryCore,
        "__init__",
        side_effect=AssertionError("GI must not be constructed"),
    ):
        result = run()
    np.testing.assert_allclose(
        result[:, :3],
        [[0.5654066, 0.49565288, 0.42589906], [0.61911273, 0.5427183, 0.46632397]],
        rtol=2e-4,
    )


@GPU
def test_ordered_image_tone_mapping_and_export():
    import vulkan as vk
    from ordinarylight.runtime import VulkanRuntime, VulkanOutput
    from ordinarylight.pipeline.vulkan import (
        VulkanResource,
        VulkanResourceUse,
        VulkanPass,
        VulkanPassPipeline,
    )

    with VulkanRuntime(headless_surface=True) as runtime:
        with runtime.image(17, 9) as hdr, VulkanOutput(runtime) as output:
            use = VulkanResourceUse(
                VulkanResource.image(hdr),
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                vk.VK_IMAGE_LAYOUT_GENERAL,
            )

            def clear(command):
                vk.vkCmdClearColorImage(
                    command,
                    hdr.image,
                    vk.VK_IMAGE_LAYOUT_GENERAL,
                    vk.VkClearColorValue(float32=[1, 0.5, 0, 1]),
                    1,
                    [
                        vk.VkImageSubresourceRange(
                            aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT,
                            levelCount=1,
                            layerCount=1,
                        )
                    ],
                )

            done = VulkanPassPipeline([VulkanPass("producer", (use,), clear)]).execute(
                runtime
            )
            with output.tone_map(hdr, after=done) as frame:
                pixels = np.frombuffer(output.read(frame), np.uint8).reshape(9, 17, 4)
                assert np.all(pixels == [232, 206, 0, 255])
            # Exercise layout tracking again after the first readback transition.
            with output.tone_map(hdr, after=done, exposure=0) as frame:
                pixels = np.frombuffer(output.read(frame), np.uint8).reshape(9, 17, 4)
                assert np.all(pixels == [0, 0, 0, 255])
            with output.export(hdr, after=done) as frame:
                assert frame.wait()
                assert frame.metadata.dedicated_allocation
                os.close(frame.export_memory_fd())
                os.close(frame.export_ready_semaphore_fd())


@GPU
def test_shared_scene_survives_renderer_close():
    import ordinarylight as ol
    from ordinarylight.renderers.gi import VulkanGlobalIlluminationRenderer

    with ol.VulkanRuntime() as runtime:
        scene = ol.Scene()
        scene.add_mesh([[-1, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
        with runtime.upload_scene(scene) as resident:
            renderer = VulkanGlobalIlluminationRenderer(runtime=runtime)
            try:
                renderer.use_scene_resources(resident)
                assert renderer.compute_context is runtime
                assert renderer._core.scene_resources is resident
                camera = ol.PerspectiveCamera((0, 0.5, 3), (0, 0.5, 0))
                image = renderer.render_frame(scene, camera, 16, 16, samples=1)
                assert image.shape == (16, 16, 4) and np.isfinite(image).all()
                assert renderer._core.scene_resources is resident
                with pytest.raises(RuntimeError, match="Unbind"):
                    resident.close()
                with pytest.raises(RuntimeError, match="consumers"):
                    runtime.close()
                scene.add_light(ol.PointLight(position=(0, 0, 2), intensity=2.0))
                with pytest.raises(ValueError, match="Borrowed scene changed"):
                    renderer.render_frame(scene, camera, 16, 16, samples=1)
            finally:
                renderer.close()
            resident.require_open()
            assert resident.resource("tlas").owner is resident
            with runtime.upload_scene(scene) as replacement:
                assert replacement.geometry_revision == resident.geometry_revision
                assert (
                    replacement.content_signatures["materials"]
                    == resident.content_signatures["materials"]
                )
                assert (
                    replacement.content_signatures["lighting"]
                    != resident.content_signatures["lighting"]
                )
                np.testing.assert_array_equal(
                    replacement.primitive_ids, resident.primitive_ids
                )
        runtime.require_open()
