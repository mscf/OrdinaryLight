"""Check native optics capture lobe attribution and raw-output preservation."""
import numpy as np

import ordinarylight as ol
from ordinarylight.renderers.gi.vulkan import VulkanGlobalIlluminationRenderer
from tools.denoiser_motion.run import camera, fixture


def main():
    scene, regions = fixture()
    raw_images = []
    for enabled in (False, True):
        config = ol.RendererConfig(
            # One bounce terminates before primary direct lighting.
            max_bounces=2, denoiser_signal_capture=True,
            denoiser_sampled_indirect=enabled, wavefront_tile_capacity=4096,
        )
        with VulkanGlobalIlluminationRenderer(config=config) as renderer:
            raw = renderer.capture_denoiser_raw(
                scene, camera(0), 96, 48, frame_index=9,
            )
            diffuse = raw["path_state"]["diffuse_radiance_hit_distance"][..., :3]
            specular = raw["path_state"]["specular_radiance_hit_distance"][..., :3]
            np.testing.assert_allclose(
                diffuse + specular, np.maximum(raw["radiance"][..., :3], 0),
                rtol=2e-6, atol=1e-6,
            )
            ids = raw["primitive_id"]
            valid = ids != np.uint32(0xffffffff)
            metal = np.zeros(ids.shape, bool)
            metal[valid] = (
                scene.triangle_material_ids()[ids[valid]] == regions["glossy"]
            )
            print(f"enabled={enabled}: metal diffuse={diffuse[metal].sum():.8g}, "
                  f"specular={specular[metal].sum():.8g}", flush=True)
            if enabled:
                assert specular[metal].sum() > 0
                # Independent accumulation orders may leave float32-scale
                # residuals; measure them relative to the conserved energy.
                np.testing.assert_allclose(
                    diffuse[metal] + specular[metal], specular[metal],
                    rtol=2e-6, atol=1e-7,
                )
            raw_images.append(raw["radiance"])
    np.testing.assert_array_equal(*raw_images)
    print("Passed: metal attribution, recomposition, identical raw output")


if __name__ == "__main__":
    main()
