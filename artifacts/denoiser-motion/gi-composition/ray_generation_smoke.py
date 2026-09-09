"""Split tracing and medium-buffer rebinding with the extracted generator."""

from dataclasses import replace
from types import SimpleNamespace
import numpy as np
import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.integrations.raster_workbench import _gi_config
from tools.denoiser_motion.glass_detail import fixture, pose


def main():
    glfw = load_glfw()
    assert glfw.init()
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(64, 48, "Ray generation graph", None, None)
    try:
        for opaque, native, profiling in (
            (False, False, False),
            (True, False, False),
            (False, True, False),
            (True, False, True),
        ):
            scene, glass, bars = fixture()
            camera = pose(scene, glass, bars, "camera", 0)
            if opaque:
                scene.remove_mesh(glass)
            config = replace(
                _gi_config(
                    SimpleNamespace(id="glass-detail", renderer={}), capture=True
                ),
                wavefront_tile_capacity=64,
                wavefront_fused_secondary=False,
                wavefront_native_textures=native,
                wavefront_profiling=profiling,
            )
            with ol.VulkanGlfwPresenter(window, config=config) as presenter:
                results = []
                for _ in range(2):
                    results.append(
                        presenter.trace_wavefront_tile(
                            scene,
                            camera,
                            17,
                            13,
                            tile_origin=(2, 1),
                            tile_extent=(7, 5),
                        )["radiance"]
                    )
                    ex = presenter._core.wavefront_executor
                    assert ex.ray_generation_graph is not None
                    assert ex.intersection_graph is not None
                    assert ex.intersect_pipeline is None
                    assert len(ex.shade_descriptors) == (4 if profiling else 2)
                    assert all(
                        d.descriptor == handle
                        for d, handle in zip(ex.shade_descriptors, ex.shade_sets)
                    )
                    assert ex.generate_pipeline is None
                    assert ex.medium_capacity == ex.capacity
                    presenter.present_wavefront(scene, camera, 64, 48)
                assert np.isfinite(results).all()
                np.testing.assert_array_equal(*results)
            print(
                "opaque" if opaque else "glass",
                f"native={native} profiling={profiling} split tracing/rebind/close passed",
                flush=True,
            )
    finally:
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    main()
