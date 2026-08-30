"""WebGPU offscreen raster renderer."""

from __future__ import annotations

import numpy as np

from ...capabilities import RendererCapabilities
from ...raster import (
    RasterConfig, RasterMesh, RasterPostProcessor, RasterState,
    CAMERA_DTYPE, MATERIAL_DTYPE, camera_matrix, create_raster_pipeline,
    rasterize_geometry_products, scene_mesh,
)
from ..base import RendererImplementation, RendererImplementationInfo


class WebGpuRasterRenderer(RendererImplementation):
    """Draw Ordinary Shade WGSL programs into an offscreen RGBA target."""

    implementation = RendererImplementationInfo(
        name="webgpu-raster", family="raster", graphics_api="webgpu",
    )

    def __init__(self, program, *, config=None, state=None, power_preference="high-performance"):
        try:
            import wgpu
        except ImportError as error:
            raise RuntimeError(
                "WebGPU rasterization requires: pip install 'ordinarylight[webgpu]'"
            ) from error
        if program.vertex.target != "wgsl" or program.fragment.target != "wgsl":
            raise ValueError("WebGpuRasterRenderer requires a WGSL RasterProgram")
        self._wgpu = wgpu
        self.program = program
        if config is not None and state is not None:
            raise TypeError("pass config or state, not both")
        self.config = config or RasterConfig(state=state or RasterState())
        self.state = self.config.state
        self.pipeline_graph = create_raster_pipeline(self.config)
        self._post = RasterPostProcessor(self.config)
        self.available_outputs = ("color", "depth", "normal", "object_id")
        self.last_timings = {}
        self.adapter = wgpu.gpu.request_adapter_sync(
            power_preference=power_preference,
        )
        self.device = self.adapter.request_device_sync()
        vertex_module = self.device.create_shader_module(code=program.vertex.source)
        fragment_module = self.device.create_shader_module(code=program.fragment.source)
        self._vertex_module = vertex_module
        self._fragment_module = fragment_module
        shadow_program = type(program).shadow(target="wgsl", validate=False)
        self._shadow_vertex_module = self.device.create_shader_module(
            code=shadow_program.vertex.source,
        )
        self._shadow_fragment_module = self.device.create_shader_module(
            code=shadow_program.fragment.source,
        )
        self._pipelines = {}
        self._shadow_pipeline = None
        info = self.adapter.info
        self.capabilities = RendererCapabilities(
            renderer="webgpu-raster", features=frozenset({"raster", "offscreen", "depth"}),
            outputs=self.available_outputs,
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
                "targets": ({"format": "rgba16float", "blend": blend},),
            },
        )
        self._pipelines[key] = pipeline
        return pipeline

    def render(self, mesh: RasterMesh, width: int, height: int) -> np.ndarray:
        wgpu = self._wgpu
        width, height = int(width), int(height)
        if width < 1 or height < 1:
            raise ValueError("raster target dimensions must be positive")
        if not len(mesh.vertices):
            clear = np.array((0.04, 0.06, 0.1, 1.0), np.float32)
            return np.broadcast_to(clear, (height, width, 4)).copy()
        texture = self.device.create_texture(
            size=(width, height, 1), format="rgba16float",
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        depth = None
        if self.state.depth_test:
            depth = self.device.create_texture(size=(width, height, 1), format="depth32float", usage=wgpu.TextureUsage.RENDER_ATTACHMENT)
        vertex_buffer = self.device.create_buffer_with_data(
            data=mesh.vertices, usage=wgpu.BufferUsage.VERTEX,
        )
        bind_group = None
        atlas_texture = atlas_sampler = shadow_depth = None
        resources = getattr(self.program.fragment.reflection, "resources", ())
        if resources:
            atlas = np.ascontiguousarray(
                mesh.resources.get("base_color_atlas", np.full((1, 1, 4), 255, np.uint8)),
                dtype=np.uint8,
            )
            atlas_height, atlas_width = atlas.shape[:2]
            atlas_texture = self.device.create_texture(
                size=(atlas_width, atlas_height, 1), format="rgba8unorm-srgb",
                usage=(wgpu.TextureUsage.TEXTURE_BINDING
                       | wgpu.TextureUsage.COPY_DST
                       | wgpu.TextureUsage.RENDER_ATTACHMENT),
            )
            self.device.queue.write_texture(
                {"texture": atlas_texture}, atlas,
                {"bytes_per_row": atlas_width * 4, "rows_per_image": atlas_height},
                (atlas_width, atlas_height, 1),
            )
            atlas_sampler = self.device.create_sampler(
                mag_filter="linear", min_filter="linear",
                address_mode_u="clamp-to-edge", address_mode_v="clamp-to-edge",
            )
            shadow_sampler = self.device.create_sampler(
                mag_filter="linear", min_filter="linear",
                address_mode_u="clamp-to-edge", address_mode_v="clamp-to-edge",
                compare="less-equal",
            )
        encoder = self.device.create_command_encoder()
        shadow_vertices = mesh.resources.get("shadow_vertices")
        shadow_indices = mesh.resources.get("shadow_indices")
        shadow_rectangle = mesh.resources.get("shadow_rectangle")
        if (
            atlas_texture is not None and shadow_rectangle is not None
            and shadow_vertices is not None and len(shadow_vertices)
            and shadow_indices is not None and len(shadow_indices)
        ):
            if self._shadow_pipeline is None:
                self._shadow_pipeline = self.device.create_render_pipeline(
                    layout="auto",
                    vertex={
                        "module": self._shadow_vertex_module,
                        "entry_point": "main",
                        "buffers": ({
                            "array_stride": 16,
                            "step_mode": "vertex",
                            "attributes": ({
                                "format": "float32x4", "offset": 0,
                                "shader_location": 0,
                            },),
                        },),
                    },
                    primitive={
                        "topology": "triangle-list",
                        "cull_mode": self.config.shadow_cull_mode,
                        "front_face": self.state.front_face,
                    },
                    depth_stencil={
                        "format": "depth32float", "depth_write_enabled": True,
                        "depth_compare": "less",
                        "depth_bias": 2,
                        "depth_bias_slope_scale": 1.75,
                        "depth_bias_clamp": 0.0,
                    },
                    fragment={
                        "module": self._shadow_fragment_module,
                        "entry_point": "main",
                        "targets": (),
                    },
                )
            shadow_width, shadow_height = shadow_rectangle[2:4]
            shadow_depth = self.device.create_texture(
                size=(shadow_width, shadow_height, 1), format="depth32float",
                usage=(wgpu.TextureUsage.RENDER_ATTACHMENT
                       | wgpu.TextureUsage.TEXTURE_BINDING),
            )
            shadow_vertex_buffer = self.device.create_buffer_with_data(
                data=shadow_vertices, usage=wgpu.BufferUsage.VERTEX,
            )
            shadow_index_buffer = self.device.create_buffer_with_data(
                data=shadow_indices, usage=wgpu.BufferUsage.INDEX,
            )
            shadow_pass = encoder.begin_render_pass(
                color_attachments=(),
                depth_stencil_attachment={
                    "view": shadow_depth.create_view(),
                    "depth_clear_value": 1.0, "depth_load_op": "clear",
                    "depth_store_op": "store",
                },
            )
            sx, sy, sw, sh, _aw, _ah = shadow_rectangle
            shadow_pass.set_viewport(float(sx), float(sy), float(sw), float(sh), 0.0, 1.0)
            shadow_pass.set_scissor_rect(int(sx), int(sy), int(sw), int(sh))
            shadow_pass.set_pipeline(self._shadow_pipeline)
            shadow_pass.set_vertex_buffer(0, shadow_vertex_buffer)
            shadow_pass.set_index_buffer(shadow_index_buffer, "uint32")
            shadow_pass.draw_indexed(shadow_indices.size)
            shadow_pass.end()
        elif atlas_texture is not None:
            # The scene shader always declares a depth texture.  Keep a valid,
            # fully visible fallback bound when the scene has no shadow caster.
            shadow_depth = self.device.create_texture(
                size=(1, 1, 1), format="depth32float",
                usage=(wgpu.TextureUsage.RENDER_ATTACHMENT
                       | wgpu.TextureUsage.TEXTURE_BINDING),
            )
            shadow_clear_pass = encoder.begin_render_pass(
                color_attachments=(),
                depth_stencil_attachment={
                    "view": shadow_depth.create_view(),
                    "depth_clear_value": 1.0, "depth_load_op": "clear",
                    "depth_store_op": "store",
                },
            )
            shadow_clear_pass.end()
        if atlas_texture is not None:
            entries = [
                {"binding": 0, "resource": atlas_texture.create_view()},
                {"binding": 1, "resource": atlas_sampler},
                {"binding": 2, "resource": shadow_depth.create_view()},
                {"binding": 4, "resource": shadow_sampler},
            ]
            camera_payload = mesh.resources.get("camera_uniform")
            if camera_payload is not None:
                camera_buffer = self.device.create_buffer_with_data(
                    data=camera_payload,
                    usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
                )
                entries.append({"binding": 3, "resource": camera_buffer})
            material_payload = mesh.resources.get("material_buffer")
            if material_payload is not None:
                if not material_payload:
                    material_payload = bytes(MATERIAL_DTYPE.itemsize)
                material_buffer = self.device.create_buffer_with_data(
                    data=material_payload,
                    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
                )
                entries.append({
                    "binding": 5,
                    "resource": {
                        "buffer": material_buffer,
                        "offset": 0,
                        "size": len(material_payload),
                    },
                })
            bind_group = self.device.create_bind_group(
                layout=self._pipeline(mesh.layout).get_bind_group_layout(0),
                entries=tuple(entries),
            )
        render_pass = encoder.begin_render_pass(color_attachments=({
            "view": texture.create_view(), "resolve_target": None,
            "load_op": "clear", "store_op": "store",
            "clear_value": (0.04, 0.06, 0.1, 1.0),
        },), depth_stencil_attachment={"view": depth.create_view(), "depth_clear_value": 1.0, "depth_load_op": "clear", "depth_store_op": "store"} if depth is not None else None)
        render_pass.set_pipeline(self._pipeline(mesh.layout))
        if bind_group is not None:
            render_pass.set_bind_group(0, bind_group)
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
        row_bytes = width * 4 * np.dtype(np.float16).itemsize
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
        packed = raw.reshape(height, padded_row_bytes)[:, :row_bytes].copy()
        return packed.view(np.float16).reshape(height, width, 4).astype(np.float32)

    def render_frame(self, scene, camera, width, height, *, samples=None, frame_index=0):
        import time
        started = time.perf_counter()
        mesh = scene_mesh(
            scene, camera, width, height, self.config,
            native_shadow_maps=True, gpu_camera=True,
        )
        camera_data = np.zeros(1, CAMERA_DTYPE)
        camera_data["view_projection"][0] = camera_matrix(camera, width, height).T
        camera_data["position_exposure"][0] = (*camera.position, 1.0)
        mesh.resources["camera_uniform"] = camera_data.tobytes()
        image = self.render(mesh, width, height)
        self.last_timings = {"total_ms": (time.perf_counter() - started) * 1000.0}
        return self._post.process(image, scene, camera)

    def render_products(self, scene, camera, width, height, *, outputs, samples=None, frame_index=0):
        mesh = scene_mesh(
            scene, camera, width, height, self.config,
            native_shadow_maps=True,
        )
        products = rasterize_geometry_products(mesh, width, height)
        if "color" in outputs:
            color_mesh = scene_mesh(
                scene, camera, width, height, self.config,
                native_shadow_maps=True, gpu_camera=True,
            )
            camera_data = np.zeros(1, CAMERA_DTYPE)
            camera_data["view_projection"][0] = camera_matrix(
                camera, width, height,
            ).T
            camera_data["position_exposure"][0] = (*camera.position, 1.0)
            color_mesh.resources["camera_uniform"] = camera_data.tobytes()
            image = self.render(color_mesh, width, height)
            products["color"] = self._post.process(image, scene, camera)
        return {name: products[name] for name in outputs}

    @property
    def accumulated_frames(self):
        return self._post.accumulated_frames

    def reset_output_history(self):
        self._post.reset()

    def close(self):
        self._pipelines.clear()
        self._shadow_pipeline = None
        self.device = None
        self.adapter = None


__all__ = ["WebGpuRasterRenderer"]
