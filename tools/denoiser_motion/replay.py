"""Offline replay of production OrdinaryShade denoiser shaders via WebGPU.

Inputs are uploaded and outputs read back: this is a quality diagnostic, not
an interactive runtime or a timing benchmark. Camera-only motion is supported.
"""

from pathlib import Path

import numpy as np
import wgpu


def instrument_acceptance(source):
    marker = (
        "    textureStore(output_history_length, pixel, vec4<f32>(history_length));"
    )
    if source.count(marker) != 1:
        raise ValueError("temporal shader diagnostic hook changed")
    return (
        "@group(0) @binding(15) var debug_acceptance: texture_storage_2d<r32float, write>;\n"
        + source.replace(
            marker,
            marker
            + "\n    textureStore(debug_acceptance, pixel, vec4<f32>(select(0.0, 1.0, accepted)));",
        )
    )


class ShaderReplay:
    def __init__(self, width, height, *, depth_footprint=False, plane_gate=False):
        self.width, self.height = width, height
        self.plane_gate = plane_gate
        adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
        self.adapter_info = dict(adapter.info)
        self.device = adapter.request_device_sync(
            required_features=["texture-adapter-specific-format-features"],
            required_limits={"max-storage-textures-per-shader-stage": 17 if plane_gate else 15},
        )
        root = Path(__file__).resolve().parents[2] / "ordinarylight/shaders"
        self.pipelines = {}
        for name in ("temporal", "atrous"):
            source = (root / f"denoiser_relax_{name}.comp.wgsl").read_text()
            if name == "temporal":
                source = instrument_acceptance(source)
                if depth_footprint:
                    from tools.denoiser_motion.footprint import instrument_depth_footprint
                    source = instrument_depth_footprint(source)
                if plane_gate:
                    from tools.denoiser_motion.footprint import instrument_plane_gate
                    source = instrument_plane_gate(source)
            module = self.device.create_shader_module(code=source)
            self.pipelines[name] = self.device.create_compute_pipeline(
                layout="auto",
                compute={"module": module, "entry_point": "main"},
            )
        self.previous = None

    def texture(self, value, fmt):
        dtype = (
            np.float16
            if fmt == "rgba16float"
            else (np.uint32 if fmt == "r32uint" else np.float32)
        )
        data = np.ascontiguousarray(value, dtype=dtype)
        texture = self.device.create_texture(
            size=(self.width, self.height, 1),
            format=fmt,
            usage=wgpu.TextureUsage.STORAGE_BINDING
            | wgpu.TextureUsage.COPY_SRC
            | wgpu.TextureUsage.COPY_DST,
        )
        self.device.queue.write_texture(
            {"texture": texture},
            data,
            {"bytes_per_row": data.nbytes // self.height},
            (self.width, self.height, 1),
        )
        return texture

    def read(self, texture, channels=4):
        dtype = np.float16 if channels == 4 else np.float32
        row = self.width * channels * np.dtype(dtype).itemsize
        padded = (row + 255) // 256 * 256
        data = self.device.queue.read_texture(
            {"texture": texture},
            {"bytes_per_row": padded},
            (self.width, self.height, 1),
        )
        packed = (
            np.frombuffer(data, np.uint8).reshape(self.height, padded)[:, :row].copy()
        )
        return (
            packed.view(dtype)
            .reshape(self.height, self.width, channels)
            .astype(np.float32)
        )

    def dispatch(self, name, textures, constants, binding):
        buffer = self.device.create_buffer_with_data(
            data=np.asarray(constants, np.float32),
            usage=wgpu.BufferUsage.UNIFORM,
        )
        pipeline = self.pipelines[name]
        entries = [
            {"binding": key, "resource": tex.create_view()}
            for key, tex in textures.items()
        ]
        entries.append({"binding": binding, "resource": {"buffer": buffer}})
        group = self.device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=entries,
        )
        encoder = self.device.create_command_encoder()
        compute = encoder.begin_compute_pass()
        compute.set_pipeline(pipeline)
        compute.set_bind_group(0, group)
        compute.dispatch_workgroups((self.width + 7) // 8, (self.height + 7) // 8)
        compute.end()
        self.device.queue.submit([encoder.finish()])
        buffer.destroy()

    def denoise(
        self, signal, previous_depth, policy, iterations=3, *,
        spatial_normal_power=32, color_weight=4, positions=None,
    ):
        zeros = np.zeros((self.height, self.width), np.float32)
        rgba = np.zeros((*zeros.shape, 4), np.float32)
        motion = rgba.copy()
        motion[..., :2] = signal.motion
        motion[..., 2] = previous_depth
        guides = [
            self.texture(signal.normal_roughness, "rgba16float"),
            self.texture(signal.view_z, "r32float"),
            self.texture(signal.material_id, "r32uint"),
        ]
        position_textures = []
        if self.plane_gate:
            if positions is None:
                raise ValueError("plane gate requires world positions")
            packed = rgba.copy()
            packed[..., :3] = positions
            position_textures = [self.texture(packed, "rgba32float")]
        motion_tex = self.texture(motion, "rgba16float")
        valid = self.previous is not None and not signal.frame.camera_cut
        old = self.previous if valid else (guides, [None, None], [None, None])
        temporals, lengths, outputs, acceptance = [], [], [], []
        for lobe, value in enumerate(
            (
                signal.diffuse_radiance_hit_distance,
                signal.specular_radiance_hit_distance,
            )
        ):
            current = self.texture(value, "rgba16float")
            temporal = self.texture(rgba, "rgba16float")
            length = self.texture(zeros, "r32float")
            previous = old[1][lobe] if valid else current
            old_length = old[2][lobe] if valid else length
            # Each diagnostic object has a distinct material, so material ID
            # also supplies stable instance identity for this fixture only.
            bindings = {
                0: current,
                1: guides[0],
                2: guides[1],
                3: motion_tex,
                4: guides[2],
                5: previous,
                6: old[0][0],
                7: old[0][1],
                8: old[0][2],
                9: old_length,
                10: temporal,
                11: length,
                13: guides[2],
                14: old[0][2],
            }
            debug = self.texture(zeros, "r32float")
            bindings[15] = debug
            if self.plane_gate:
                bindings[16] = position_textures[0]
                bindings[17] = old[3][0] if valid else position_textures[0]
            # An output texture cannot alias a read binding, even on reset.
            empty_length = self.texture(zeros, "r32float")
            if not valid:
                bindings[9] = empty_length
            self.dispatch(
                "temporal",
                bindings,
                [
                    self.width,
                    self.height,
                    policy["history_limit"],
                    float(valid),
                    policy["normal_threshold"],
                    policy["depth_threshold"],
                    policy["clamp_sigma"],
                    policy["reactive_sigma"],
                ],
                12,
            )
            acceptance.append(self.read(debug, 1)[..., 0])
            debug.destroy()
            source = temporal
            for iteration in range(iterations):
                target = self.texture(rgba, "rgba16float")
                self.dispatch(
                    "atrous",
                    {0: source, 1: guides[0], 2: guides[1], 3: guides[2], 4: target},
                    [
                        self.width,
                        self.height,
                        1 << iteration,
                        0,
                        spatial_normal_power,
                        0.02,
                        color_weight,
                        float(lobe == 1 and iteration == 0),
                    ],
                    5,
                )
                if source is not temporal:
                    source.destroy()
                source = target
            outputs.append(self.read(source)[..., :3])
            if source is not temporal:
                source.destroy()
            current.destroy()
            empty_length.destroy()
            temporals.append(temporal)
            lengths.append(length)
        history = [self.read(value, 1)[..., 0] for value in lengths]
        if self.previous is not None:
            for group in self.previous:
                for tex in group:
                    tex.destroy()
        self.previous = (guides, temporals, lengths, position_textures)
        motion_tex.destroy()
        self.last_acceptance = acceptance
        return outputs, history

    def close(self):
        self.device.destroy()
