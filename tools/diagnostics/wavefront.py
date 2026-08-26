"""Exercise the low-level Vulkan wavefront generate/intersect stages."""

import ordinarylight as ol
from ordinarylight.integrations.glfw_platform import load_glfw


glfw = load_glfw()

from ordinarylight.showcases.vertex_attributes import build_vertex_attribute_showcase


def main():
    if not glfw.init():
        raise RuntimeError("GLFW initialization failed")
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(64, 64, "wavefront probe", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("GLFW Vulkan window creation failed")
    try:
        scene = build_vertex_attribute_showcase()
        camera = ol.PerspectiveCamera(
            position=(0.0, 3.0, 8.5), target=(0.0, 1.25, 0.0)
        )
        config = ol.RendererConfig(wavefront_tile_capacity=4096)
        with ol.VulkanGlfwPresenter(window, config=config) as presenter:
            result = presenter.trace_wavefront_tile(
                scene, camera, 64, 64, tile_extent=(64, 64)
            )
        summary = {
            key: value for key, value in result.items()
            if key not in {"radiance", "rgba8"}
        }
        radiance = result["radiance"]
        rgba8 = result["rgba8"]
        summary["radiance_shape"] = radiance.shape
        summary["mean_radiance"] = tuple(radiance[..., :3].mean(axis=(0, 1)))
        summary["mean_rgba8"] = tuple(rgba8.mean(axis=(0, 1)))
        print(summary)
        assert result["ray_queue"]["count"] == 4096
        assert result["ray_queue"]["overflow"] == 0
        assert result["hit_queue"]["count"] <= 4096
        assert result["hit_queue"]["overflow"] == 0
        assert result["continuation_queue"]["count"] == 0
        assert result["continuation_queue"]["overflow"] == 0
        assert result["bounces_executed"] == config.max_bounces
        assert radiance.shape == (64, 64, 4)
        assert radiance.dtype.name == "float32"
        assert radiance[..., 3].min() == 1.0
        assert radiance[..., :3].max() > 0.0
        assert rgba8.shape == (64, 64, 4)
        assert rgba8.dtype.name == "uint8"
        assert rgba8[..., 3].min() == 255
    finally:
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    main()
