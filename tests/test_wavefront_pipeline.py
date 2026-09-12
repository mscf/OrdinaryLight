"""Native lighting in an application graph, with no window or video interop."""

import os
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import vulkan as vk

from ordinarylight.integrations.raster_workbench import _gi_config
from ordinarylight.pipeline.graph import VulkanGraph
from ordinarylight.runtime import VulkanRuntime, VulkanWavefrontPipeline
from tools.denoiser_motion.glass_detail import fixture, pose

pytestmark = pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_GRAPH") != "1", reason="opt-in Vulkan GPU test",
)


def test_standalone_native_frames_cancel_resize_and_history(monkeypatch):
    scene, glass, bars = fixture()
    config = replace(_gi_config(
        SimpleNamespace(id="glass-detail", renderer={}),
        render_scale=0.5, upscale_filter="bilinear", capture=True,
    ), max_bounces=4, samples_per_pixel=1, wavefront_raw_hdr_output=True)
    with VulkanRuntime(config=config) as runtime, runtime.upload_scene(scene) as resident:
        with VulkanWavefrontPipeline(runtime, resident, config=config) as pipeline:
            assert pipeline._core.surface is None
            reset_pools = []
            reset_query_pool = vk.vkCmdResetQueryPool

            def track_reset(command, pool, first, count):
                reset_pools.append(pool)
                return reset_query_pool(command, pool, first, count)

            monkeypatch.setattr(vk, "vkCmdResetQueryPool", track_reset)
            results, hits = [], 0
            for i in range(8):
                reset_pools.clear()
                frame = pipeline.prepare(pose(scene, glass, bars, "camera", i * 0.001), (160, 120))
                assert pipeline._core.timestamp_query_pool not in reset_pools, (
                    "Preparation must not record an unused primary command buffer"
                )
                assert frame.render_extent == (80, 60)
                assert frame.images["hdr"].image != frame.images["raw_hdr"].image
                with pytest.raises(RuntimeError, match="Submit or cancel"):
                    pipeline.prepare(pose(scene, glass, bars, "camera", 0), (160, 120))
                graph = VulkanGraph().add("lighting", frame.operation).compile()
                completion = graph.execute(runtime)
                assert frame.completion is completion
                completion.wait()
                results.append(pipeline.capture_wavefront_hdr())
                hits += int(pipeline.last_timings["wavefront_command_cache_hit"])
                with pytest.raises(RuntimeError, match="submitted or cancelled"):
                    graph.execute(runtime)
            assert hits > 0
            assert np.isfinite(results).all()
            assert np.any(np.array(results)[..., :3] > 0)
            assert pipeline._core.reconstruction_graph is None
            assert pipeline._core.swapchain is None
            old_image = frame.images["hdr"]
            cancelled = pipeline.prepare(pose(scene, glass, bars, "camera", 0), (160, 120))
            cancelled.cancel()
            pipeline.invalidate_gi_history()
            frame = pipeline.render(pose(scene, glass, bars, "camera", 0), (200, 140))
            frame.completion.wait()
            assert frame.render_extent == (100, 70)
            with pytest.raises(RuntimeError, match="retired"):
                old_image.require_open()


def test_primary_outputs_match_actual_sampled_rays():
    from ordinarylight.pipeline.vulkan import VulkanPass, VulkanResource, VulkanResourceUse
    from ordinarylight.wavefront.primary_outputs import PRIMARY_HIT_DTYPE
    scene, glass, bars = fixture()
    config = replace(_gi_config(
        SimpleNamespace(id="glass-detail", renderer={}),
        render_scale=0.5, upscale_filter="bilinear",
    ), max_bounces=4, samples_per_pixel=2, wavefront_primary_hits=True)
    with VulkanRuntime(config=config) as runtime, runtime.upload_scene(scene) as resident:
        with VulkanWavefrontPipeline(runtime, resident, config=config) as pipeline:
            camera = replace(pose(scene, glass, bars, "camera", 0), vertical_fov_degrees=70)
            frame = pipeline.prepare(camera, (160, 120))
            hits = frame.buffers["primary_hits"]
            assert frame.sample_count == 2
            assert hits.byte_size == 2 * 80 * 60 * PRIMARY_HIT_DTYPE.itemsize
            with runtime.buffer(hits.byte_size) as staging:
                graph = VulkanGraph().add("lighting", frame.operation)
                graph.add("diagnostics", VulkanPass("copy", (
                    VulkanResourceUse(VulkanResource.buffer(hits), vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      vk.VK_ACCESS_TRANSFER_READ_BIT),
                    VulkanResourceUse(VulkanResource.buffer(staging), vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                                      vk.VK_ACCESS_TRANSFER_WRITE_BIT),
                ), lambda command: vk.vkCmdCopyBuffer(command, hits.buffer, staging.buffer, 1,
                                                     [vk.VkBufferCopy(size=hits.byte_size)])))
                graph.compile().execute(runtime).wait()
                data = np.frombuffer(staging.read(), PRIMARY_HIT_DTYPE).reshape(2, 60, 80)
            valid = data["position_distance"][..., 3] >= 0
            assert np.any(valid) and np.any(~valid)
            np.testing.assert_array_equal(data["identity"][~valid], np.uint32(0xffffffff))
            origin, direction = data["ray_origin"][..., :3], data["ray_direction"][..., :3]
            np.testing.assert_allclose(np.linalg.norm(direction, axis=-1), 1, atol=1e-5)
            distance = data["position_distance"][..., 3]
            np.testing.assert_allclose(data["position_distance"][..., :3][valid],
                                       (origin + direction * distance[..., None])[valid], atol=2e-5)
            assert np.any(direction[0] != direction[1]), "Each path sample must retain its own jitter"
            triangles = scene.render_triangles().astype(np.float64)
            a, edge1, edge2 = triangles[:, 0], triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            for flat in np.linspace(0, valid.size - 1, 64, dtype=int):
                ray = data.reshape(-1)[flat]
                o, d = ray["ray_origin"][:3], ray["ray_direction"][:3]
                h = np.cross(d, edge2)
                det = np.einsum("ij,ij->i", edge1, h)
                inv = np.divide(1.0, det, out=np.zeros_like(det), where=np.abs(det) > 1e-9)
                relative = o - a
                u = np.einsum("ij,ij->i", relative, h) * inv
                q = np.cross(relative, edge1)
                v = q @ d * inv
                t = np.einsum("ij,ij->i", edge2, q) * inv
                candidates = (np.abs(det) > 1e-9) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= 0.001)
                if np.any(candidates):
                    primitive = np.argmin(np.where(candidates, t, np.inf))
                    np.testing.assert_allclose(ray["position_distance"][3], t[primitive], rtol=1e-5, atol=1e-5)
                    assert ray["identity"][2] == primitive
                else:
                    assert ray["position_distance"][3] < 0


def test_resident_replacement_leases_and_content_updates():
    from ordinarylight.pipeline.vulkan import VulkanPass, VulkanResourceUse
    scene, glass, bars = fixture()
    config = replace(_gi_config(
        SimpleNamespace(id="glass-detail", renderer={}),
        render_scale=0.5, upscale_filter="bilinear", capture=True,
    ), max_bounces=4, samples_per_pixel=1)
    camera = pose(scene, glass, bars, "camera", 0)
    with VulkanRuntime(config=config) as runtime:
        # Close the scene before its borrowed replacement allocation.
        with runtime.upload_scene(scene) as source:
            size = source.bindings["material"].size
        with runtime.buffer(size) as replacement, runtime.upload_scene(scene) as resident:
            with VulkanWavefrontPipeline(runtime, resident, config=config) as pipeline:
                original = resident.resource("material")
                stale = VulkanGraph().add("read", VulkanPass("read", (
                    VulkanResourceUse(original, vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                      vk.VK_ACCESS_SHADER_READ_BIT),
                ), lambda command: None)).compile()
                frame = pipeline.prepare(camera, (160, 120))
                for change in (lambda: resident.notify_content_changed(),
                               lambda: resident.replace_resources(buffers={"material": replacement}),
                               lambda: pipeline.reconfigure(max_bounces=3),
                               pipeline.invalidate_gi_history):
                    with pytest.raises(RuntimeError, match="Submit or cancel"):
                        change()
                frame.cancel()
                with pytest.raises(ValueError, match="leased same-runtime"):
                    resident.replace_resources(acceleration=resident.resource("tlas"))
                assert resident.binding_revision == 0
                # Upload the fixture's native packing at this explicit replacement boundary.
                replacement.upload(scene.triangle_material_data())
                assert resident.replace_resources(buffers={"material": replacement}) == 1
                with pytest.raises(RuntimeError, match="borrow"):
                    replacement.close()
                with pytest.raises(ValueError, match="recompile"):
                    stale.execute(runtime)
                frame = pipeline.render(camera, (160, 120))
                frame.completion.wait()
                assert np.any(pipeline.capture_wavefront_hdr()[..., :3] > 0)
                commands = [f.get("wavefront_command_key") for f in pipeline._core.window_frames]
                assert resident.notify_content_changed(after=(frame.completion,),
                                                        invalidate_history=False) == 2
                assert commands == [f.get("wavefront_command_key") for f in pipeline._core.window_frames]
                pipeline.render(camera, (160, 120)).completion.wait()
            assert resident.resource("material").handle == replacement.buffer
        assert replacement.closed


def test_native_linear_hdr_can_be_tone_mapped_in_the_same_graph():
    from ordinarylight.runtime import VulkanOutput
    scene, glass, bars = fixture()
    config = replace(_gi_config(
        SimpleNamespace(id="glass-detail", renderer={}),
        render_scale=0.5, upscale_filter="bilinear", capture=True,
    ), max_bounces=4, samples_per_pixel=1)
    with VulkanRuntime(config=config) as runtime, runtime.upload_scene(scene) as resident:
        with VulkanWavefrontPipeline(runtime, resident, config=config) as pipeline, VulkanOutput(runtime) as output:
            frame = pipeline.prepare(pose(scene, glass, bars, "camera", 0), (160, 120))
            with output.prepare(frame.images["hdr"], extent=frame.render_extent) as tone:
                graph = VulkanGraph().add("lighting", frame.operation).add("tone", tone.operation())
                compiled = graph.compile()
                assert compiled.order == ("lighting", "tone")
                completion = compiled.execute(runtime)
                assert frame.completion is tone.completion is completion
                completion.wait()
                hdr = np.maximum(pipeline.capture_wavefront_hdr()[..., :3], 0)
                expected = np.clip((hdr * (2.51 * hdr + 0.03)) /
                                   (hdr * (2.43 * hdr + 0.59) + 0.14), 0, 1)
                expected = np.where(expected <= 0.0031308, 12.92 * expected,
                                    1.055 * expected ** (1 / 2.4) - 0.055)
                actual = np.frombuffer(output.read(tone), np.uint8).reshape(60, 80, 4)
                np.testing.assert_allclose(actual[..., :3], expected * 255, atol=1)
                assert np.all(actual[..., 3] == 255)


def test_imported_native_scene_matches_uploaded_lighting_and_retains_owners():
    from ordinarylight.materials import builtin_material
    scene, glass, bars = fixture()
    config = replace(_gi_config(
        SimpleNamespace(id="glass-detail", renderer={}),
        render_scale=0.5, upscale_filter="bilinear", capture=True,
    ), max_bounces=4, samples_per_pixel=1)
    camera = pose(scene, glass, bars, "camera", 0)
    default = config.material_program or builtin_material
    material_data = scene.triangle_material_data(scene.material_programs(default), default)
    with VulkanRuntime(config=config) as runtime:
        with runtime.upload_scene(scene) as source, runtime.buffer(material_data.nbytes, data=material_data) as materials:
            borrowers = set(source._borrowers)
            with runtime.buffer(16) as too_small:
                with pytest.raises(ValueError, match="smaller than"):
                    runtime.import_scene(scene, acceleration=source.resource("tlas"),
                                         buffers={"material": too_small})
                assert source._borrowers == borrowers
                assert not too_small._borrowers
            with pytest.raises(ValueError, match="Unknown native scene buffer"):
                runtime.import_scene(scene, acceleration=source.resource("tlas"),
                                     buffers={"typo": materials})
            with runtime.import_scene(scene, acceleration=source.resource("tlas"),
                                      buffers={"material": materials}) as imported:
                assert not imported._structures
                assert imported.instance_buffer is None
                assert imported.bindings["material"] is materials
                assert all(item.buffer != materials.buffer for item in imported._buffers)
                with pytest.raises(RuntimeError, match="borrowers"):
                    materials.close()
                with pytest.raises(RuntimeError, match="consumers"):
                    source.close()
                images = []
                for resident in (source, imported):
                    with VulkanWavefrontPipeline(runtime, resident, config=config) as pipeline:
                        frame = pipeline.render(camera, (128, 96))
                        frame.completion.wait()
                        images.append(pipeline.capture_wavefront_hdr())
                np.testing.assert_allclose(images[0], images[1], atol=2e-3, rtol=2e-3)
                assert np.any(images[1][..., :3] > 0)
            assert source._borrowers == borrowers
            assert not materials._borrowers
