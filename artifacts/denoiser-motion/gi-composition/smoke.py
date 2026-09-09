"""Run with PYTHONPATH=. .venv/bin/python this_file.py on a Vulkan desktop."""

from types import SimpleNamespace

import numpy as np
import ordinarylight as ol
import vulkan as vk
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.integrations.raster_workbench import _gi_config
from ordinarylight.pipeline import RenderStage
from tools.denoiser_motion.glass_detail import fixture, pose


def main():
    glfw = load_glfw()
    assert glfw.init()
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(320, 240, "GI composition", None, None)
    try:
        for mode in ("bilinear", "fsr1-shade", "fsr2"):
            results = []
            for custom in (False, True):
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
                        frames.append(presenter.capture_wavefront_hdr())
                    if custom:
                        assert len(calls) == 8
                        presenter.set_gi_pipeline_builder(None)
                        presenter.present_wavefront(
                            scene, pose(scene, glass, bars, "camera", 0.008), 320, 240
                        )
                    results.append(np.array(frames))
            assert np.array_equal(*results), mode
            print(
                mode, "default and custom graph HDR identical; reset passed", flush=True
            )
    finally:
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    main()
