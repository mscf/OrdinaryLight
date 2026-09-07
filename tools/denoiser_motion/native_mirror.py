"""Smoke-test native static planar-mirror guides with camera motion."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.showcases.rooms import build_planar_mirror_guides
from tests.gates.relax_motion_quality import _config
from tools.denoiser_motion.live_edges import read_guides


def run():
    output = Path("/tmp/native-mirror-guides")
    output.mkdir(exist_ok=True)
    glfw = load_glfw()
    if not glfw.init():
        raise RuntimeError("GLFW unavailable")
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(160, 120, "Native mirror validation", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Window creation failed")
    guides = {}
    args = SimpleNamespace(width=160, height=120, bounces=3, reference_samples=1,
                           atrous_iterations=3, history_floor=3)
    try:
        for enabled in (False, True):
            config = replace(_config(args, reference=False), denoiser_signal_capture=True,
                             denoiser_planar_mirror_guides=enabled)
            scene = build_planar_mirror_guides()
            with ol.VulkanGlfwPresenter(window, config=config) as presenter:
                for x in (0, 0.1, 0.2, 0.2):
                    camera = ol.PerspectiveCamera(position=(x, 2, -7), target=(x, 2, 0))
                    presenter.present_wavefront(scene, camera, 160, 120)
                guides[enabled] = read_guides(presenter._core)
                image = presenter.capture_wavefront_hdr()
                assert np.isfinite(image).all()
                np.save(output / f"image-{enabled}.npy", image)
                np.savez_compressed(output / f"guides-{enabled}.npz", **guides[enabled])
        base, optical = guides[False], guides[True]
        mirror = np.isclose(base["view_z"][..., 0], 7, atol=1e-4)
        changed = abs(base["view_z"][..., 0]-optical["view_z"][..., 0]) > 1e-4
        assert (changed & mirror).sum() > 20
        assert ((optical["view_z"][..., 0] > 0) & mirror).sum() > 20
        np.testing.assert_array_equal(base["view_z"][~mirror], optical["view_z"][~mirror])
        print(f"mirror pixels={mirror.sum()}, changed={int((changed & mirror).sum())}")
    finally:
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    run()
