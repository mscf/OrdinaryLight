"""Native denoiser cache, active extent, resize and disposal checks."""

from types import SimpleNamespace
from dataclasses import replace
import json
from pathlib import Path

import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.integrations.raster_workbench import _gi_config
from tools.denoiser_motion.glass_detail import fixture, pose


def main():
    glfw = load_glfw()
    assert glfw.init()
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(320, 240, "Denoiser lifecycle", None, None)
    report = []
    try:
        for mode in ("fsr1-shade", "fsr2"):
            scene, glass, bars = fixture()
            cfg = _gi_config(
                SimpleNamespace(id="glass-detail", renderer={}),
                present=True,
                upscale_filter=mode,
            )
            cfg=replace(cfg,wavefront_indirect_reuse_storage=True,wavefront_indirect_reuse_candidates=True)
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
                original_graphs = {
                    slot: dict(graphs) for slot, graphs in graph_cache.items()
                }
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
                        p.present_wavefront(
                            scene, camera, 320, 240, render_extent=extent
                        )
                        assert p._core.denoiser_graph is adapter
                        assert all(
                            p._core.wavefront_executor.relax_preparation_stages[slot][1]
                            is stage
                            for slot, stage in preparation.items()
                        )
                        if mode != "fsr2":
                            assert p._core.reconstruction_graph is reconstruction
                        assert (
                            tuple(p.last_timings["wavefront_render_extent"]) == extent
                        )
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
            report.append(
                dict(
                    filter=mode,
                    cache_hits=hits,
                    extent_changes_reused_bindings=True,
                    resize_retired_bindings=True,
                    close_retired_bindings=True,
                    reconstruction_uses_component=True,
                    reconstruction_retired_bindings=True,
                )
            )
            print(report[-1], flush=True)
    finally:
        glfw.destroy_window(window)
        glfw.terminate()
    Path("/tmp/gi-denoiser-lifecycle.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
