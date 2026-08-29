"""Minimal real WebGPU offscreen raster backend."""

from __future__ import annotations

import numpy as np

from ..capabilities import RendererCapabilities
from ..raster import RasterMesh, RasterState, scene_mesh


class WebGpuRasterBackend:
    """Draw Ordinary Shade WGSL programs into an offscreen RGBA target."""

    def __init__(self, program, *, state=None, power_preference="high-performance"):
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
        self.config = self.state = state or RasterState()
        self.available_outputs = ("color",)
        self.last_timings = {}
        self.adapter = wgpu.gpu.request_adapter_sync(
            power_preference=power_preference,
        )
        self.device = self.adapter.request_device_sync()
        vertex_module = self.device.create_shader_module(code=program.vertex.source)
        fragment_module = self.device.create_shader_module(code=program.fragment.source)
        self._vertex_module = vertex_module
        self._fragment_module = fragment_module
        self._pipelines = {}
        info = self.adapter.info
        self.capabilities = RendererCapabilities(
            backend="webgpu-raster", features=frozenset({"raster", "offscreen", "depth"}),
            outputs=("color",),
            device=info.get("device", info.get("description", "WebGPU adapter")),
        )

    def _pipeline(self, layout):
        key = (layout, self.state)
        if key in self._pipelines:
            return self._pipelines[key]
        blend = None
        if self.state.blend_mode == "alpha":
            blend = {"color": {"src_factor": "src-alpha", "dst_factor": "one-minus-src-alpha", "operation": "add"}, "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha", "operation": "add"}}
        elif self.state.blend_mode == "additive":
            blend = {"color": {"src_factor": "one", "dst_factor": "one", "operation": "add"}, "alpha": {"src_factor": "one", "dst_factor": "one", "operation": "add"}}
        pipeline = self.device.create_render_pipeline(
            layout="auto",
            vertex={
                "module": self._vertex_module,
                "entry_point": "main",
                "buffers": ({
                    "array_stride": layout.stride,
                    "step_mode": "vertex",
                    "attributes": tuple({"format": item.format, "offset": item.offset, "shader_location": item.location} for item in layout.attributes),
                },),
            },
            primitive={"topology": self.state.topology, "cull_mode": self.state.cull_mode, "front_face": self.state.front_face},
            depth_stencil={"format": "depth32float", "depth_write_enabled": self.state.depth_write, "depth_compare": self.state.depth_compare} if self.state.depth_test else None,
            fragment={
                "module": self._fragment_module,
                "entry_point": "main",
                "targets": ({"format": "rgba8unorm", "blend": blend},),
            },
        )
        self._pipelines[key] = pipeline
        return pipeline

    def render(self, mesh: RasterMesh, width: int, height: int) -> np.ndarray:
        wgpu = self._wgpu
        width, height = int(width), int(height)
        if width < 1 or height < 1:
            raise ValueError("raster target dimensions must be positive")
        texture = self.device.create_texture(
            size=(width, height, 1), format="rgba8unorm",
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        depth = None
        if self.state.depth_test:
            depth = self.device.create_texture(size=(width, height, 1), format="depth32float", usage=wgpu.TextureUsage.RENDER_ATTACHMENT)
        vertex_buffer = self.device.create_buffer_with_data(
            data=mesh.vertices, usage=wgpu.BufferUsage.VERTEX,
        )
        encoder = self.device.create_command_encoder()
        render_pass = encoder.begin_render_pass(color_attachments=({
            "view": texture.create_view(), "resolve_target": None,
            "load_op": "clear", "store_op": "store",
            "clear_value": (0.04, 0.06, 0.1, 1.0),
        },), depth_stencil_attachment={"view": depth.create_view(), "depth_clear_value": 1.0, "depth_load_op": "clear", "depth_store_op": "store"} if depth is not None else None)
        render_pass.set_pipeline(self._pipeline(mesh.layout))
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

    def render_frame(self, scene, camera, width, height, *, samples=None, frame_index=0):
        import time
        started = time.perf_counter()
        image = self.render(scene_mesh(scene, camera, width, height), width, height)
        self.last_timings = {"total_ms": (time.perf_counter() - started) * 1000.0}
        return image.astype(np.float32) / 255.0

    def close(self):
        self._pipelines.clear()
        self.device = None
        self.adapter = None


__all__ = ["WebGpuRasterBackend"]
