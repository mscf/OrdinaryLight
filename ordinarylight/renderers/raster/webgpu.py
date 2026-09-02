"""WebGPU offscreen raster renderer."""

from __future__ import annotations

import numpy as np

_OPTICAL_DEBUG_MODES = {
    "off": 0.0, "hit": 1.0, "uv": 2.0, "depth-delta": 3.0,
    "confidence": 4.0, "object-id": 5.0, "depth-trace": 6.0,
    "refraction-hit": 7.0, "refraction-uv": 8.0,
    "refraction-source": 9.0,
}

from ...capabilities import RendererCapabilities
from ...raster import (
    RasterConfig, RasterMesh, RasterPostProcessor, RasterState,
    CAMERA_DTYPE, LIGHT_DTYPE, MATERIAL_DTYPE, SHADOW_DTYPE, camera_matrix, create_raster_pipeline,
    geometry_product_mesh, scene_mesh,
)
from ..base import RendererImplementation, RendererImplementationInfo


class WebGpuRasterRenderer(RendererImplementation):
    """Draw Ordinary Shade WGSL programs into an offscreen RGBA target."""

    implementation = RendererImplementationInfo(
        name="webgpu-raster", family="raster", graphics_api="webgpu",
    )

    def request_probe_refresh(self, probe):
        """Request recapture of an ``on-demand`` reflection probe."""
        self.probe_capture.request(probe)

    def refresh_reflection_probes(self, scene, *, force=False):
        """Capture due probes immediately and return their replacements."""
        return self.probe_capture.refresh(self, scene, force=force)

    @staticmethod
    def _opaque_camera_payload(payload):
        camera = np.frombuffer(payload, dtype=CAMERA_DTYPE).copy()
        mode = camera["viewport_optics"][0, 2]
        camera["viewport_optics"][0, 2] = 2.0 if mode > 0.0 else 0.0
        camera["optical_diagnostic"][0] = 0.0
        return camera.tobytes()

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
        self.available_outputs = ("color", "depth", "normal", "object_id", "motion")
        self._output_history = None
        self.last_timings = {}
        from ...probes import ProbeCaptureManager
        self.probe_capture = ProbeCaptureManager()
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
        product_program = type(program).geometry_products(
            target="wgsl", validate=False,
        )
        self._product_vertex_module = self.device.create_shader_module(
            code=product_program.vertex.source,
        )
        self._product_fragment_module = self.device.create_shader_module(
            code=product_program.fragment.source,
        )
        volume_program = type(program).volume(target="wgsl", validate=False)
        self._volume_vertex_module = self.device.create_shader_module(
            code=volume_program.vertex.source,
        )
        self._volume_fragment_module = self.device.create_shader_module(
            code=volume_program.fragment.source,
        )
        self._product_pipelines = {}
        self._pipelines = {}
        self._shadow_pipeline = None
        self._volume_pipeline = None
        info = self.adapter.info
        self.capabilities = RendererCapabilities(
            renderer="webgpu-raster", features=frozenset({
                "raster", "offscreen", "depth", "volumes",
                "volume_scattering", "volume-shadowing",
                "overlapping-volume-extinction",
                "volume-empty-space-skipping",
                "native-volume-ray-march",
            }),
            outputs=self.available_outputs,
            limits={"max_volume_slices": 1024},
            device=info.get("device", info.get("description", "WebGPU adapter")),
        )

    def _native_volume_pipeline(self):
        if self._volume_pipeline is not None:
            return self._volume_pipeline
        self._volume_pipeline = self.device.create_render_pipeline(
            layout="auto",
            vertex={
                "module": self._volume_vertex_module,
                "entry_point": "main",
                "buffers": ({
                    "array_stride": 16,
                    "step_mode": "vertex",
                    "attributes": ({
                        "format": "float32x2", "offset": 0,
                        "shader_location": 0,
                    }, {
                        "format": "float32x2", "offset": 8,
                        "shader_location": 1,
                    }),
                },),
            },
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            fragment={
                "module": self._volume_fragment_module,
                "entry_point": "main",
                "targets": ({"format": "rgba16float"},),
            },
        )
        return self._volume_pipeline

    def _composite_native_volumes(
        self, encoder, mesh, source, depth, shadow_depth, shadow_sampler,
        width, height,
    ):
        resources = mesh.resources.get("volume_resources")
        if resources is None or not resources.scalar_fields or depth is None:
            return source
        wgpu = self._wgpu
        destination = self.device.create_texture(
            size=(width, height, 1), format="rgba16float",
            usage=(wgpu.TextureUsage.RENDER_ATTACHMENT
                   | wgpu.TextureUsage.COPY_SRC
                   | wgpu.TextureUsage.TEXTURE_BINDING),
        )
        matrix = np.asarray(mesh.resources["volume_inverse_view_projection"], np.float32)
        camera = np.zeros(1, dtype=np.dtype([
            ("inverse_view_projection", np.float32, (4, 4)),
            ("camera_position", np.float32, (4,)),
            ("viewport_steps", np.float32, (4,)),
            ("volume_count", np.uint32, (4,)),
        ], align=True))
        camera["inverse_view_projection"][0] = matrix.T
        camera["camera_position"][0] = mesh.resources["volume_camera_position"]
        camera["viewport_steps"][0] = (
            width, height, mesh.resources["volume_step_scale"],
            mesh.resources["volume_max_steps"],
        )
        camera["volume_count"][0, 0] = min(len(resources.scalar_fields), 4)
        camera["volume_count"][0, 1] = min(mesh.resources.get("light_count", 0), 8)
        camera["volume_count"][0, 2] = int(mesh.resources.get(
            "volume_empty_space_skipping", False,
        ))
        camera["volume_count"][0, 3] = min(mesh.resources.get("shadow_count", 0), 24)
        uniform = self.device.create_buffer_with_data(
            data=camera, usage=wgpu.BufferUsage.UNIFORM,
        )
        headers = self.device.create_buffer_with_data(
            data=resources.headers, usage=wgpu.BufferUsage.STORAGE,
        )
        transfers = self.device.create_buffer_with_data(
            data=resources.transfers, usage=wgpu.BufferUsage.STORAGE,
        )
        light_payload = mesh.resources.get("light_buffer", b"") or bytes(64)
        light_buffer = self.device.create_buffer_with_data(
            data=light_payload, usage=wgpu.BufferUsage.STORAGE,
        )
        shadow_payload = mesh.resources.get("shadow_buffer", b"") or bytes(96)
        shadow_buffer = self.device.create_buffer_with_data(
            data=shadow_payload, usage=wgpu.BufferUsage.STORAGE,
        )
        textures = []
        fields = list(resources.scalar_fields[:4])
        while len(fields) < 4:
            fields.append(np.zeros((1, 1, 1), np.float32))
        for field in fields:
            field = np.ascontiguousarray(field, dtype=np.float16)
            depth_size, height_size, width_size = field.shape
            texture = self.device.create_texture(
                size=(width_size, height_size, depth_size), dimension="3d",
                format="r16float",
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            )
            self.device.queue.write_texture(
                {"texture": texture}, field,
                {"bytes_per_row": width_size * 2, "rows_per_image": height_size},
                (width_size, height_size, depth_size),
            )
            textures.append(texture)
        occupancy_textures = []
        occupancy_fields = list(resources.occupancy_fields[:4])
        while len(occupancy_fields) < 4:
            occupancy_fields.append(np.ones((1, 1, 1), np.float32))
        for field in occupancy_fields:
            field = np.ascontiguousarray(field, dtype=np.float16)
            depth_size, height_size, width_size = field.shape
            texture = self.device.create_texture(
                size=(width_size, height_size, depth_size), dimension="3d",
                format="r16float",
                usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST,
            )
            self.device.queue.write_texture(
                {"texture": texture}, field,
                {"bytes_per_row": width_size * 2, "rows_per_image": height_size},
                (width_size, height_size, depth_size),
            )
            occupancy_textures.append(texture)
        linear = self.device.create_sampler(
            mag_filter="linear", min_filter="linear",
            address_mode_u="clamp-to-edge", address_mode_v="clamp-to-edge",
            address_mode_w="clamp-to-edge",
        )
        nearest = self.device.create_sampler(
            mag_filter="nearest", min_filter="nearest",
            address_mode_u="clamp-to-edge", address_mode_v="clamp-to-edge",
        )
        pipeline = self._native_volume_pipeline()
        bind_group = self.device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=(
                {"binding": 0, "resource": {"buffer": uniform}},
                {"binding": 1, "resource": {"buffer": headers}},
                {"binding": 2, "resource": {"buffer": transfers}},
                {"binding": 3, "resource": source.create_view()},
                {"binding": 4, "resource": depth.create_view()},
                *({"binding": 5 + index, "resource": texture.create_view()}
                  for index, texture in enumerate(textures)),
                {"binding": 9, "resource": linear},
                {"binding": 10, "resource": nearest},
                {"binding": 11, "resource": {"buffer": light_buffer}},
                *({"binding": 12 + index, "resource": texture.create_view()}
                  for index, texture in enumerate(occupancy_textures)),
                {"binding": 16, "resource": shadow_depth.create_view()},
                {"binding": 17, "resource": shadow_sampler},
                {"binding": 18, "resource": {"buffer": shadow_buffer}},
            ),
        )
        fullscreen = self.device.create_buffer_with_data(
            # position.xy, top-left-oriented texture coordinate.xy. WebGPU's
            # framebuffer Y is opposite its clip-space Y, so UV is explicit.
            data=np.asarray((
                (-1, -1, 0, 1), (3, -1, 2, 1), (-1, 3, 0, -1),
            ), np.float32),
            usage=wgpu.BufferUsage.VERTEX,
        )
        render_pass = encoder.begin_render_pass(color_attachments=({
            "view": destination.create_view(), "resolve_target": None,
            "load_op": "clear", "store_op": "store",
            "clear_value": (0.0, 0.0, 0.0, 1.0),
        },))
        render_pass.set_pipeline(pipeline)
        render_pass.set_bind_group(0, bind_group)
        render_pass.set_vertex_buffer(0, fullscreen)
        render_pass.draw(3)
        render_pass.end()
        return destination

    def _product_pipeline(self, layout):
        pipeline = self._product_pipelines.get(layout)
        if pipeline is not None:
            return pipeline
        pipeline = self.device.create_render_pipeline(
            layout="auto",
            vertex={
                "module": self._product_vertex_module,
                "entry_point": "main",
                "buffers": ({
                    "array_stride": layout.stride,
                    "step_mode": "vertex",
                    "attributes": tuple({
                        "format": item.format, "offset": item.offset,
                        "shader_location": item.location,
                    } for item in layout.attributes),
                },),
            },
            primitive={
                "topology": "triangle-list", "cull_mode": self.state.cull_mode,
                "front_face": self.state.front_face,
            },
            depth_stencil={
                "format": "depth32float", "depth_write_enabled": True,
                "depth_compare": "less",
            },
            fragment={
                "module": self._product_fragment_module,
                "entry_point": "main",
                "targets": (
                    {"format": "rgba32float"},
                    {"format": "rgba32float"},
                ),
            },
        )
        self._product_pipelines[layout] = pipeline
        return pipeline

    def _render_native_products(self, mesh, width, height):
        """Rasterize depth/normal/object-ID/motion into native MRTs."""
        if not len(mesh.vertices) and mesh.resources.get("volume_resources") is None:
            return {
                "depth": np.ones((height, width), np.float32),
                "normal": np.zeros((height, width, 3), np.float32),
                "object_id": np.zeros((height, width), np.uint32),
                "motion": np.zeros((height, width, 2), np.float32),
            }
        wgpu = self._wgpu
        formats = ("rgba32float", "rgba32float")
        textures = tuple(self.device.create_texture(
            size=(width, height, 1), format=format_name,
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        ) for format_name in formats)
        depth = self.device.create_texture(
            size=(width, height, 1), format="depth32float",
            usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
        )
        vertex = self.device.create_buffer_with_data(
            data=mesh.vertices, usage=wgpu.BufferUsage.VERTEX,
        )
        index = self.device.create_buffer_with_data(
            data=mesh.indices, usage=wgpu.BufferUsage.INDEX,
        ) if mesh.indices is not None and mesh.indices.size else None
        uniform = self.device.create_buffer_with_data(
            data=mesh.resources["geometry_product_camera"],
            usage=wgpu.BufferUsage.UNIFORM,
        )
        pipeline = self._product_pipeline(mesh.layout)
        bind_group = self.device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=({
                "binding": 0,
                "resource": {"buffer": uniform, "offset": 0,
                             "size": len(mesh.resources["geometry_product_camera"])},
            },),
        )
        encoder = self.device.create_command_encoder()
        render_pass = encoder.begin_render_pass(
            color_attachments=tuple({
                "view": texture.create_view(), "resolve_target": None,
                "load_op": "clear", "store_op": "store",
                "clear_value": (0, 0, 0, 0),
            } for index, texture in enumerate(textures)),
            depth_stencil_attachment={
                "view": depth.create_view(), "depth_load_op": "clear",
                "depth_store_op": "store", "depth_clear_value": 1.0,
            },
        )
        render_pass.set_pipeline(pipeline)
        render_pass.set_bind_group(0, bind_group, (), 0, 0)
        render_pass.set_vertex_buffer(0, vertex, 0, mesh.vertices.nbytes)
        if index is not None:
            render_pass.set_index_buffer(index, "uint32", 0, mesh.indices.nbytes)
            render_pass.draw_indexed(mesh.indices.size, 1, 0, 0, 0)
        else:
            render_pass.draw(len(mesh.vertices), 1, 0, 0)
        render_pass.end()
        specs = (
            (textures[0], 16, np.float32, 4),
            (textures[1], 16, np.float32, 4),
        )
        readbacks = []
        for texture, bytes_per_pixel, _dtype, _components in specs:
            row_bytes = width * bytes_per_pixel
            padded = (row_bytes + 255) & ~255
            buffer = self.device.create_buffer(
                size=padded * height,
                usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
            )
            encoder.copy_texture_to_buffer(
                {"texture": texture},
                {"buffer": buffer, "bytes_per_row": padded,
                 "rows_per_image": height},
                (width, height, 1),
            )
            readbacks.append((buffer, padded, row_bytes, _dtype, _components))
        self.device.queue.submit((encoder.finish(),))
        arrays = []
        for buffer, padded, row_bytes, dtype, components in readbacks:
            raw = np.frombuffer(self.device.queue.read_buffer(buffer), np.uint8)
            packed = raw.reshape(height, padded)[:, :row_bytes].copy()
            values = packed.view(dtype)
            if components > 1:
                values = values.reshape(height, width, components)
            else:
                values = values.reshape(height, width)
            arrays.append(values)
        normal_depth, motion_object = arrays
        object_ids = np.rint(motion_object[..., 2]).astype(np.uint32)
        depth_output = normal_depth[..., 3].astype(np.float32)
        depth_output[object_ids == 0] = np.inf
        motion = motion_object[..., :2].astype(np.float32)
        motion[np.abs(motion) < 1e-6] = 0.0
        return {
            "depth": depth_output,
            "normal": normal_depth[..., :3].astype(np.float32),
            "object_id": object_ids,
            "motion": motion,
        }

    def _pipeline(self, layout, *, pass_kind="opaque"):
        optical = pass_kind in {
            "optical-opaque", "transmissive", "transparent",
        }
        transparent = pass_kind in {"transmissive", "transparent"}
        key = (layout, self.state, pass_kind)
        if key in self._pipelines:
            return self._pipelines[key]
        # Source-alpha blending is harmless for opaque fragments (alpha=1)
        # and lets material-level ``alpha_mode='blend'`` work without forcing
        # applications to replace the renderer-wide pipeline state.
        blend = None
        if transparent or self.state.blend_mode in {"opaque", "alpha"}:
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
            primitive={
                "topology": self.state.topology,
                "cull_mode": (
                    "back" if pass_kind == "transmissive"
                    else self.state.cull_mode
                ),
                "front_face": self.state.front_face,
            },
            depth_stencil={
                "format": "depth32float",
                "depth_write_enabled": self.state.depth_write and not optical,
                "depth_compare": (
                    "less-equal" if optical and
                    self.config.optical_quality == "screen-space" else
                    self.state.depth_compare
                ),
            } if self.state.depth_test else None,
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
            usage=(wgpu.TextureUsage.RENDER_ATTACHMENT
                   | wgpu.TextureUsage.COPY_SRC
                   | wgpu.TextureUsage.COPY_DST
                   | wgpu.TextureUsage.TEXTURE_BINDING),
        )
        depth = None
        if self.state.depth_test:
            depth = self.device.create_texture(
                size=(width, height, 1), format="depth32float",
                usage=(wgpu.TextureUsage.RENDER_ATTACHMENT
                       | wgpu.TextureUsage.TEXTURE_BINDING),
            )
        vertex_buffer = self.device.create_buffer_with_data(
            data=(mesh.vertices if len(mesh.vertices) else
                  np.zeros((1, mesh.layout.stride // 4), np.float32)),
            usage=wgpu.BufferUsage.VERTEX,
        )
        bind_group = None
        transparent_bind_group = None
        optical_entries = None
        atlas_texture = atlas_sampler = shadow_depth = shadow_sampler = None
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
            scene_depth_sampler = self.device.create_sampler(
                mag_filter="nearest", min_filter="nearest",
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
        screen_space_optics = bool(
            self.config.optical_quality == "screen-space"
            and mesh.resources.get("optical_index_count", 0) and depth is not None
        )
        if atlas_texture is not None:
            entries = [
                {"binding": 0, "resource": atlas_texture.create_view()},
                {"binding": 1, "resource": atlas_sampler},
                {"binding": 2, "resource": shadow_depth.create_view()},
                {"binding": 4, "resource": shadow_sampler},
                {"binding": 6, "resource": atlas_texture.create_view()},
                {"binding": 7, "resource": shadow_depth.create_view()},
                {"binding": 8, "resource": atlas_sampler},
                {"binding": 9, "resource": scene_depth_sampler},
            ]
            camera_payload = mesh.resources.get("camera_uniform")
            optical_camera_buffer = None
            if camera_payload is not None:
                camera_buffer = self.device.create_buffer_with_data(
                    data=(
                        self._opaque_camera_payload(camera_payload)
                        if screen_space_optics else camera_payload
                    ),
                    usage=wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST,
                )
                entries.append({"binding": 3, "resource": camera_buffer})
                if screen_space_optics:
                    optical_camera_buffer = self.device.create_buffer_with_data(
                        data=camera_payload,
                        usage=(wgpu.BufferUsage.UNIFORM
                               | wgpu.BufferUsage.COPY_DST),
                    )
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
            light_payload = mesh.resources.get("light_buffer", b"")
            if not light_payload:
                light_payload = bytes(LIGHT_DTYPE.itemsize)
            light_buffer = self.device.create_buffer_with_data(
                data=light_payload,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            )
            entries.append({
                "binding": 10,
                "resource": {
                    "buffer": light_buffer,
                    "offset": 0,
                    "size": len(light_payload),
                },
            })
            shadow_payload = mesh.resources.get("shadow_buffer", b"")
            if not shadow_payload:
                shadow_payload = bytes(SHADOW_DTYPE.itemsize)
            shadow_buffer = self.device.create_buffer_with_data(
                data=shadow_payload,
                usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST,
            )
            entries.append({
                "binding": 11,
                "resource": {
                    "buffer": shadow_buffer,
                    "offset": 0,
                    "size": len(shadow_payload),
                },
            })
            bind_group = self.device.create_bind_group(
                layout=self._pipeline(mesh.layout).get_bind_group_layout(0),
                entries=tuple(entries),
            )
            transparent_bind_group = self.device.create_bind_group(
                layout=self._pipeline(
                    mesh.layout, pass_kind="transparent",
                ).get_bind_group_layout(0),
                entries=tuple(entries),
            )
            optical_bind_group = None
            if screen_space_optics:
                optical_entries = [dict(entry) for entry in entries]
                for entry in optical_entries:
                    if entry["binding"] == 6:
                        entry["resource"] = texture.create_view()
                    elif entry["binding"] == 7:
                        entry["resource"] = depth.create_view()
                    elif entry["binding"] == 3:
                        entry["resource"] = optical_camera_buffer
                optical_bind_group = self.device.create_bind_group(
                    layout=self._pipeline(
                        mesh.layout, pass_kind="optical-opaque",
                    ).get_bind_group_layout(0),
                    entries=tuple(optical_entries),
                )
        output_texture = texture
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
            authored_opaque_count = int(mesh.resources.get(
                "opaque_index_count", mesh.indices.size,
            ))
            opaque_count = (
                int(mesh.resources.get("opaque_prepass_index_count", 0))
                if screen_space_optics else authored_opaque_count
            )
            if opaque_count:
                render_pass.draw_indexed(opaque_count)
            transparent_count = mesh.indices.size - opaque_count
            if transparent_count and not screen_space_optics:
                render_pass.set_pipeline(
                    self._pipeline(mesh.layout, pass_kind="transparent"),
                )
                if transparent_bind_group is not None:
                    render_pass.set_bind_group(0, transparent_bind_group)
                render_pass.draw_indexed(
                    transparent_count, 1, opaque_count, 0, 0,
                )
        render_pass.end()
        if screen_space_optics:
            output_texture = self.device.create_texture(
                size=(width, height, 1), format="rgba16float",
                usage=(wgpu.TextureUsage.RENDER_ATTACHMENT
                       | wgpu.TextureUsage.COPY_SRC
                       | wgpu.TextureUsage.COPY_DST
                       | wgpu.TextureUsage.TEXTURE_BINDING),
            )
            optical_ping_entries = [dict(entry) for entry in entries]
            for entry in optical_ping_entries:
                if entry["binding"] == 6:
                    entry["resource"] = output_texture.create_view()
                elif entry["binding"] == 7:
                    entry["resource"] = depth.create_view()
                elif entry["binding"] == 3:
                    entry["resource"] = optical_camera_buffer
            encoder.copy_texture_to_texture(
                {"texture": texture}, {"texture": output_texture},
                (width, height, 1),
            )
            optical_pass = encoder.begin_render_pass(
                color_attachments=({
                    "view": output_texture.create_view(),
                    "resolve_target": None,
                    "load_op": "load", "store_op": "store",
                },),
                depth_stencil_attachment={
                    "view": depth.create_view(),
                    "depth_read_only": True,
                },
            )
            optical_opaque_pipeline = self._pipeline(
                mesh.layout, pass_kind="optical-opaque",
            )
            optical_pass.set_pipeline(optical_opaque_pipeline)
            optical_pass.set_bind_group(0, optical_bind_group)
            optical_pass.set_vertex_buffer(0, vertex_buffer)
            optical_pass.set_index_buffer(index_buffer, "uint32")
            optical_opaque_count = int(mesh.resources.get(
                "optical_opaque_index_count", 0,
            ))
            alpha_transparent_count = int(mesh.resources.get(
                "transparent_index_count", 0,
            ))
            transmissive_count = int(mesh.resources.get(
                "optical_transmissive_index_count", 0,
            ))
            authored_transparent_count = (
                alpha_transparent_count - transmissive_count
            )
            if optical_opaque_count:
                optical_pass.draw_indexed(
                    optical_opaque_count, 1, opaque_count, 0, 0,
                )
            optical_pass.end()
            final_texture = output_texture

            def composite_optical_layer(
                source_texture, destination_texture, source_entries,
                pass_kind, draw_count, first_index,
            ):
                encoder.copy_texture_to_texture(
                    {"texture": source_texture},
                    {"texture": destination_texture},
                    (width, height, 1),
                )
                layer_pass = encoder.begin_render_pass(
                    color_attachments=({
                        "view": destination_texture.create_view(),
                        "resolve_target": None,
                        "load_op": "load", "store_op": "store",
                    },),
                    depth_stencil_attachment={
                        "view": depth.create_view(),
                        "depth_read_only": True,
                    },
                )
                layer_pipeline = self._pipeline(
                    mesh.layout, pass_kind=pass_kind,
                )
                source_bind_group = self.device.create_bind_group(
                    layout=layer_pipeline.get_bind_group_layout(0),
                    entries=tuple(source_entries),
                )
                layer_pass.set_pipeline(layer_pipeline)
                layer_pass.set_bind_group(0, source_bind_group)
                layer_pass.set_vertex_buffer(0, vertex_buffer)
                layer_pass.set_index_buffer(index_buffer, "uint32")
                layer_pass.draw_indexed(
                    draw_count, 1, first_index, 0, 0,
                )
                layer_pass.end()

            transmissive_counts = tuple(mesh.resources.get(
                "optical_transmissive_index_counts", (),
            ))
            retained_counts = transmissive_counts[
                -int(self.config.screen_space_optical_layers):
            ]
            next_index = (
                opaque_count + optical_opaque_count
                + sum(transmissive_counts[:-len(retained_counts)])
                if retained_counts else
                opaque_count + optical_opaque_count
            )
            for draw_count in retained_counts:
                if final_texture is output_texture:
                    destination_texture = texture
                    source_entries = optical_ping_entries
                else:
                    destination_texture = output_texture
                    source_entries = optical_entries
                composite_optical_layer(
                    final_texture, destination_texture, source_entries,
                    "transmissive", int(draw_count), int(next_index),
                )
                final_texture = destination_texture
                next_index += int(draw_count)
            if authored_transparent_count:
                if final_texture is output_texture:
                    destination_texture = texture
                    source_entries = optical_ping_entries
                else:
                    destination_texture = output_texture
                    source_entries = optical_entries
                composite_optical_layer(
                    final_texture, destination_texture, source_entries,
                    "transparent", authored_transparent_count,
                    opaque_count + optical_opaque_count + transmissive_count,
                )
                final_texture = destination_texture
            output_texture = final_texture
        output_texture = self._composite_native_volumes(
            encoder, mesh, output_texture, depth, shadow_depth, shadow_sampler,
            width, height,
        )
        row_bytes = width * 4 * np.dtype(np.float16).itemsize
        padded_row_bytes = (row_bytes + 255) & ~255
        readback = self.device.create_buffer(
            size=padded_row_bytes * height,
            usage=wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC,
        )
        encoder.copy_texture_to_buffer(
            {"texture": output_texture},
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
        self.probe_capture.refresh(self, scene)
        mesh = scene_mesh(
            scene, camera, width, height, self.config,
            native_shadow_maps=True, gpu_camera=True,
        )
        camera_data = np.zeros(1, CAMERA_DTYPE)
        camera_data["view_projection"][0] = camera_matrix(camera, width, height).T
        camera_data["position_exposure"][0] = (*camera.position, 1.0)
        camera_data["viewport_optics"][0] = (width, height, 1.0 if self.config.optical_quality == "screen-space" else 0.0, self.config.screen_space_ray_steps)
        camera_data["optical_diagnostic"][0, 0] = _OPTICAL_DEBUG_MODES[
            self.config.optical_debug_view
        ]
        camera_data["optical_diagnostic"][0, 1] = mesh.resources.get(
            "light_count", 0,
        )
        camera_data["optical_diagnostic"][0, 2] = mesh.resources.get(
            "shadow_count", 0,
        )
        camera_data["optical_diagnostic"][0, 3] = float(mesh.resources.get(
            "optical_transmissive_layers_nested", False,
        ))
        mesh.resources["camera_uniform"] = camera_data.tobytes()
        mesh.resources["volume_inverse_view_projection"] = np.linalg.inv(
            camera_matrix(camera, width, height),
        )
        mesh.resources["volume_camera_position"] = (*camera.position, 1.0)
        image = self.render(mesh, width, height)
        self.last_timings = {"total_ms": (time.perf_counter() - started) * 1000.0}
        return self._post.process(image, scene, camera)

    def render_products(self, scene, camera, width, height, *, outputs, samples=None, frame_index=0):
        mesh, next_history = geometry_product_mesh(
            scene, camera, width, height, self._output_history,
        )
        products = self._render_native_products(mesh, width, height)
        self._output_history = next_history
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
            camera_data["viewport_optics"][0] = (width, height, 1.0 if self.config.optical_quality == "screen-space" else 0.0, self.config.screen_space_ray_steps)
            camera_data["optical_diagnostic"][0, 0] = _OPTICAL_DEBUG_MODES[
                self.config.optical_debug_view
            ]
            camera_data["optical_diagnostic"][0, 1] = color_mesh.resources.get(
                "light_count", 0,
            )
            camera_data["optical_diagnostic"][0, 2] = color_mesh.resources.get(
                "shadow_count", 0,
            )
            camera_data["optical_diagnostic"][0, 3] = float(
                color_mesh.resources.get(
                    "optical_transmissive_layers_nested", False,
                )
            )
            color_mesh.resources["camera_uniform"] = camera_data.tobytes()
            color_mesh.resources["volume_inverse_view_projection"] = np.linalg.inv(
                camera_matrix(camera, width, height),
            )
            color_mesh.resources["volume_camera_position"] = (
                *camera.position, 1.0,
            )
            image = self.render(color_mesh, width, height)
            products["color"] = self._post.process(image, scene, camera)
        return {name: products[name] for name in outputs}

    @property
    def accumulated_frames(self):
        return self._post.accumulated_frames

    def reset_output_history(self):
        self._post.reset()
        self._output_history = None

    def close(self):
        self._pipelines.clear()
        self._product_pipelines.clear()
        self._shadow_pipeline = None
        self.device = None
        self.adapter = None


__all__ = ["WebGpuRasterRenderer"]
