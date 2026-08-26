"""Print driver-provided Vulkan compute pipeline statistics."""

import ordinarylight as ol


def main():
    config = ol.RendererConfig(
        max_bounces=5,
        samples_per_pixel=1,
        wavefront_execution_strategy="hybrid",
        wavefront_persistent_continuations=True,
        wavefront_pipeline_statistics=True,
    )
    scene = ol.build_feature_parity_scene()
    camera = ol.feature_parity_camera()
    with ol.VulkanRayTracingBackend(config=config) as renderer:
        renderer.render_wavefront(scene, camera, 64, 64, samples=1)
        for shader, executables in sorted(renderer.pipeline_statistics.items()):
            print(shader)
            for executable in executables:
                print(f"  {executable['name']}")
                for name, value in sorted(executable["statistics"].items()):
                    print(f"    {name}: {value}")


if __name__ == "__main__":
    main()
