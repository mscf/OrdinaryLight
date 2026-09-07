"""GPU contract checks for per-sample indirect classification and accumulation.

Run with python -m tools.denoiser_motion.check_sampled_indirect.
"""
from pathlib import Path

import numpy as np
import wgpu

from ordinarylight.wavefront import PATH_STATE_DTYPE, SECONDARY_PATH_STATE_DTYPE
from tools.denoiser_motion.replay import ShaderReplay


def main():
    replay = ShaderReplay(1, 1)
    device = replay.device
    source = (Path(__file__).resolve().parents[2] /
              "ordinarylight/shaders/denoiser_relax_prepare.comp.wgsl").read_text()
    pipeline = device.create_compute_pipeline(
        layout="auto", compute={
            "module": device.create_shader_module(code=source), "entry_point": "main",
        },
    )
    textures = {
        key: replay.texture(np.zeros((1, 1, 4 if fmt == "rgba16float" else 1)), fmt)
        for key, fmt in {
            2: "r32uint", 3: "r32uint", 4: "rgba16float", 5: "rgba16float",
            6: "rgba16float", 7: "r32float", 8: "rgba16float", 12: "r32uint",
        }.items()
    }

    def dispatch(index, count, specular, total, primary, probability, *, enabled=True, primary_specular=None, indirect_fraction=None):
        paths = np.zeros(1, PATH_STATE_DTYPE)
        paths["radiance"][0, :3] = total
        secondary = np.zeros(1, SECONDARY_PATH_STATE_DTYPE)
        secondary["primary_radiance"][0] = (*([primary] * 3), probability)
        secondary["primary_throughput"][0, 3] = 2 if specular else 1
        secondary["primary_position"][0] = (0, 0, 1, 1)
        secondary["position_valid"][0] = (3, 4, 1, 1)
        secondary["diffuse_radiance_hit_distance"][0] = (2, 2, 2, 5)
        secondary["specular_radiance_hit_distance"][0] = (6, 6, 6, 5)
        if primary_specular is not None:
            secondary["specular_radiance_hit_distance"][0] = (
                primary_specular, primary_specular, primary_specular, -1,
            )
        if indirect_fraction is not None:
            secondary["diffuse_radiance_hit_distance"][0] = (
                *indirect_fraction, -1,
            )
        camera = np.array(((0, 0, 0, 0), (0, 0, 1, 0),
                           (1, 0, 0, 0), (0, 1, 0, 0)), np.float32)
        values = {
            0: paths, 1: secondary, 9: camera, 10: camera,
            11: np.array(((0, 0, 1, 0),) * 3, np.float32),
            13: np.array((1, 1, 1, 0, index, count, int(enabled), 0), np.uint32),
        }
        buffers = {key: device.create_buffer_with_data(
            data=value, usage=wgpu.BufferUsage.UNIFORM if key == 13
            else wgpu.BufferUsage.STORAGE,
        ) for key, value in values.items()}
        entries = [{"binding": key, "resource": tex.create_view()}
                   for key, tex in textures.items() if key != 3]
        entries += [{"binding": key, "resource": {"buffer": buf}}
                    for key, buf in buffers.items()]
        group = device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0), entries=entries,
        )
        encoder = device.create_command_encoder()
        compute = encoder.begin_compute_pass()
        compute.set_pipeline(pipeline)
        compute.set_bind_group(0, group)
        compute.dispatch_workgroups(1)
        compute.end()
        device.queue.submit([encoder.finish()])
        result = [replay.read(textures[key])[0, 0] for key in (4, 5)]
        for buffer in buffers.values():
            buffer.destroy()
        return result

    try:
        # Two samples deliberately have opposite branches and probabilities.
        # Mean raw = 12; expected diffuse = (7.75 + 0.25)/2, specular = 8.
        dispatch(0, 2, False, 8, 1, 0.25)
        diffuse, specular = dispatch(1, 2, True, 16, 1, 0.75)
        np.testing.assert_array_equal(diffuse[:3], [4, 4, 4])
        np.testing.assert_array_equal(specular[:3], [8, 8, 8])
        np.testing.assert_array_equal(diffuse[:3] + specular[:3], [12, 12, 12])
        assert diffuse[3] == 0 and specular[3] == 5
        # A new frame must overwrite the previous sample sum.
        diffuse, specular = dispatch(0, 1, False, 8, 0, 0.9)
        np.testing.assert_array_equal(diffuse[:3], [8, 8, 8])
        np.testing.assert_array_equal(specular[:3], [0, 0, 0])
        assert diffuse[3] == 5 and specular[3] == 0
        # Evaluated direct specular overrides the material probability.
        diffuse, specular = dispatch(
            0, 1, False, 8, 8, 0.1, primary_specular=8,
        )
        np.testing.assert_array_equal(diffuse[:3], [0, 0, 0])
        np.testing.assert_array_equal(specular[:3], [8, 8, 8])
        # Primary emission (no recorded specular) stays outside specular.
        diffuse, specular = dispatch(
            0, 1, True, 8, 8, 0.9, primary_specular=0,
        )
        np.testing.assert_array_equal(diffuse[:3], [8, 8, 8])
        np.testing.assert_array_equal(specular[:3], [0, 0, 0])
        # Mixed-BSDF weights are split componentwise, regardless of branch.
        diffuse, specular = dispatch(
            0, 1, True, 8, 0, 0.9, primary_specular=0,
            indirect_fraction=(0.25, 0.5, 0.75),
        )
        np.testing.assert_array_equal(diffuse[:3], [6, 4, 2])
        np.testing.assert_array_equal(specular[:3], [2, 4, 6])
        assert diffuse[3] == 5 and specular[3] == 5
        # Disabled mode must retain the pre-existing resolved-channel behavior.
        diffuse, specular = dispatch(0, 1, False, 100, 0, 0, enabled=False)
        np.testing.assert_array_equal(diffuse, [2, 2, 2, 5])
        np.testing.assert_array_equal(specular, [6, 6, 6, 5])
        print("GPU checks passed: mixed-sample accumulation, energy conservation, "
              "frame reset, per-branch distance, disabled-mode compatibility")
    finally:
        for texture in textures.values():
            texture.destroy()
        replay.close()


if __name__ == "__main__":
    main()
