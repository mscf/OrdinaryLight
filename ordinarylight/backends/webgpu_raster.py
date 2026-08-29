"""Minimal real WebGPU offscreen raster backend."""

from __future__ import annotations

import numpy as np

from ..capabilities import RendererCapabilities
from ..raster import RasterMesh


class WebGpuRasterBackend:
    """Draw Ordinary Shade WGSL programs into an offscreen RGBA target."""

    def __init__(self, program, *, power_preference="high-performance"):
        try:
            import wgpu
        except ImportError as error:
            raise RuntimeError(
                "WebGPU rasterization requires: pip install 'ordinarylight[webgpu]'"
            ) from error
        if program.vertex.target != "wgsl" or program.fragment.target != "wgsl":
            raise ValueError("WebGpuRasterBackend requires a WGSL RasterProgram")
        self._wgpu = wgpu
        self.program = program
        self.adapter = wgpu.gpu.request_adapter_sync(
            power_preference=power_preference,
        )
        self.device = self.adapter.request_device_sync()
        vertex_module = self.device.create_shader_module(code=program.vertex.source)
        fragment_module = self.device.create_shader_module(code=program.fragment.source)
        self.pipeline = self.device.create_render_pipeline(
            layout="auto",
            vertex={
                "module": vertex_module,
                "entry_point": "main",
                "buffers": ({
                    "array_stride": 8,
                    "step_mode": "vertex",
                    "attributes": ({
                        "format": "float32x2", "offset": 0,
                        "shader_location": 0,
                    },),
                },),
            },
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            fragment={
                "module": fragment_module,
                "entry_point": "main",
                "targets": ({"format": "rgba8unorm"},),
            },
        )
        info = self.adapter.info
        self.capabilities = RendererCapabilities(
            backend="webgpu-raster", features=frozenset({"raster", "offscreen"}),
            device=info.get("device", info.get("description", "WebGPU adapter")),
        )

    def render(self, mesh: RasterMesh, width: int, height: int) -> np.ndarray:
        wgpu = self._wgpu
        width, height = int(width), int(height)
        if width < 1 or height < 1:
            raise ValueError("raster target dimensions must be positive")
        texture = self.device.create_texture(
            size=(width, height, 1), format="rgba8unorm",
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        vertex_buffer = self.device.create_buffer_with_data(
            data=mesh.vertices, usage=wgpu.BufferUsage.VERTEX,
        )
        encoder = self.device.create_command_encoder()
        render_pass = encoder.begin_render_pass(color_attachments=({
            "view": texture.create_view(), "resolve_target": None,
            "load_op": "clear", "store_op": "store",
            "clear_value": (0.04, 0.06, 0.1, 1.0),
        },))
        render_pass.set_pipeline(self.pipeline)
        render_pass.set_vertex_buffer(0, vertex_buffer)
        if mesh.indices is None:
            render_pass.draw(mesh.vertices.shape[0])
        else:
            index_buffer = self.device.create_buffer_with_data(
                data=mesh.indices, usage=wgpu.BufferUsage.INDEX,
            )
            render_pass.set_index_buffer(index_buffer, "uint32")
            render_pass.draw_indexed(mesh.indices.size)
        render_pass.end()
        row_bytes = width * 4
        padded_row_bytes = (row_bytes + 255) & ~255
        readback = self.device.create_buffer(
            size=padded_row_bytes * height,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
        )
        encoder.copy_texture_to_buffer(
            {"texture": texture},
            {"buffer": readback, "bytes_per_row": padded_row_bytes,
             "rows_per_image": height},
            (width, height, 1),
        )
        self.device.queue.submit((encoder.finish(),))
        raw = np.frombuffer(self.device.queue.read_buffer(readback), np.uint8)
        return raw.reshape(height, padded_row_bytes)[:, :row_bytes].reshape(height, width, 4).copy()

    def close(self):
        self.pipeline = None
        self.device = None
        self.adapter = None


__all__ = ["WebGpuRasterBackend"]
