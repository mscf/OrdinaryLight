import sys, json, os
from dataclasses import replace
from PySide6 import QtCore, QtWidgets
from ordinarylight.integrations import raster_workbench
original_config = raster_workbench._gi_config
def config(*args, **kwargs):
    result = replace(original_config(*args, **kwargs), wavefront_restir_di=True)
    if os.environ.get("MODE") == "no-restir":
        result = replace(result, wavefront_restir_di=False)
    if os.environ.get("MODE") == "no-lobes":
        result = replace(result, denoiser_sampled_indirect=False)
    if os.environ.get("MODE") == "mirror-on":
        result = replace(result, denoiser_planar_mirror_guides=True)
    if os.environ.get("MODE") == "generic":
        result = replace(result, wavefront_restir_specialization=False)
    if os.environ.get("MODE") == "textured-path":
        result = replace(result, wavefront_untextured_specialization=False)
    if os.environ.get("MODE") == "one-stream":
        result = replace(result, wavefront_restir_reservoirs=1)
    if os.environ.get("MODE") == "two-stream":
        result = replace(result, wavefront_restir_reservoirs=2)
    return result
raster_workbench._gi_config = config
if os.environ.get("MODE") == "fresh":
    from ordinarylight.targets.vulkan.core import VulkanWavefrontExecutor
    original_dispatch = VulkanWavefrontExecutor.dispatch
    def dispatch(self, *args, **kwargs):
        kwargs["restir_history_valid_override"] = False
        return original_dispatch(self, *args, **kwargs)
    VulkanWavefrontExecutor.dispatch = dispatch
if os.environ.get("MODE") == "bounds":
    from pathlib import Path
    from ordinarylight.targets.vulkan import core
    original_files = core.files
    class ShaderResources:
        def __init__(self, package):
            self.base = original_files(package)
        def joinpath(self, name):
            candidate = Path("/tmp/restir-bounds-spv") / Path(name).name
            return candidate if candidate.is_file() else self.base.joinpath(name)
    core.files = ShaderResources

from ordinarylight.showcases import rooms
if os.environ.get("MATERIALS") == "builtin-diffuse":
    rooms.diffuse = None
# Select custom diffuse surfaces by their order in this dedicated fixture.
selected_materials = os.environ.get("MATERIALS", "custom-diffuse")
material_groups = {"floor": {0}, "emitter": {1}, "spheres": {3, 4},
                   "floor-spheres": {0, 3, 4}, "floor-emitter": {0, 1},
                   "emitter-spheres": {1, 3, 4}}
if selected_materials in material_groups:
    original_add_mesh = rooms.ol.Scene.add_mesh
    mesh_index = 0
    def selected_add_mesh(self, vertices, indices, material=None, **kwargs):
        global mesh_index
        index = mesh_index
        mesh_index += 1
        keep = material_groups[selected_materials]
        if material is not None and material.program is rooms.diffuse and index not in keep:
            material = replace(material, program=None)
        print(json.dumps({"mesh": index, "custom_diffuse": material.program is rooms.diffuse}), flush=True)
        return original_add_mesh(self, vertices, indices, material, **kwargs)
    rooms.ol.Scene.add_mesh = selected_add_mesh
from ordinarylight.targets.vulkan import core
original_dispatch_trace = core.VulkanWavefrontExecutor.dispatch
original_bind_trace = core.vk.vkCmdBindPipeline
pipeline_names = {}
bound_names = set()
def trace_dispatch(self, *args, **kwargs):
    for name, value in vars(self).items():
        if name.endswith("_pipeline") and value is not None:
            pipeline_names[str(value)] = name
    return original_dispatch_trace(self, *args, **kwargs)
def trace_bind(command, point, pipeline):
    name = pipeline_names.get(str(pipeline), "other")
    if name not in bound_names:
        bound_names.add(name)
        print("BOUND " + name, flush=True)
    return original_bind_trace(command, point, pipeline)
core.VulkanWavefrontExecutor.dispatch = trace_dispatch
core.vk.vkCmdBindPipeline = trace_bind

# Append an unreachable program only during shader generation. Scene packing
# retains the original program tuple, so existing material IDs are unchanged.
from pathlib import Path
from ordinarylight.shaders import compiler as material_compiler
from ordinarylight.materials import builtin_material
original_shader_source = material_compiler.wavefront_material_shader_source
def diagnostic_shader_source(shader_name, programs, **kwargs):
    programs = tuple(programs)
    original_count = len(programs)
    if (os.environ.get("EXTRA_PROGRAM") == "builtin"
            and os.environ.get("EXTRA_SHADER", "all") in {"all", shader_name}):
        assert all(program is not builtin_material for program in programs)
        programs = programs + (builtin_material,)
    source = original_shader_source(shader_name, programs, **kwargs)
    reduction = os.environ.get("PRIMARY_REDUCTION")
    if reduction and shader_name == "wavefront_primary.comp":
        anchor = "    SurfaceParameters surface = SurfaceParameters(evaluated.base_color"
        assert source.count(anchor) == 1
        if reduction == "runtime-flag":
            addition = "    if (program_id == 2) evaluated.custom_scattering = 0.0;\n"
        elif reduction == "duplicate-custom":
            addition = "    if (program_id == 2) evaluated = evaluateMaterial_0(material, normal, uv, direction, entering, random_u, random_v, bounce_index, current_ior, exterior_ior);\n"
        elif reduction in {"loop-resample", "loop-scatter", "primary-resample", "primary-scatter"}:
            addition = ""
            condition = "if (wave_surface_response.custom_scattering > 0.5)"
            parts = source.split(condition)
            assert len(parts) == 5
            selected = ["loop-resample", "loop-scatter", "primary-resample", "primary-scatter"].index(reduction)
            source = parts[0]
            for index, part in enumerate(parts[1:]):
                changed = "if (wave_surface_response.custom_scattering > 0.5 && int(floor(material.ior_distance.z)) != 2)"
                source += (changed if index == selected else condition) + part
        else:
            raise ValueError("Unknown PRIMARY_REDUCTION: " + reduction)
        source = source.replace(anchor, addition + anchor)
    output_dir = os.environ.get("SHADER_DUMP")
    if output_dir:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / (shader_name + ".glsl")).write_text(source)
        binary = material_compiler._compile_source(source, material_compiler.find_glsl_compiler())
        (directory / (shader_name + ".spv")).write_bytes(binary)
    print(json.dumps({"shader": shader_name, "scene_program_count": original_count,
                      "compiled_program_count": len(programs)}), flush=True)
    return source
material_compiler.wavefront_material_shader_source = diagnostic_shader_source

if os.environ.get("DETERMINISTIC_CAMERA") == "1" or os.environ.get("TRACE_CAMERA") == "1":
    import math
    original_present = core.VulkanRayQueryCore.present_wavefront_window
    def deterministic_present(self, scene, camera, width, height, **kwargs):
        index = getattr(self, "diagnostic_present_index", 0)
        angle = index * (0.35 / 60.0)
        position = (-7.0 * math.sin(angle), 2.0, -7.0 * math.cos(angle))
        if os.environ.get("DETERMINISTIC_CAMERA") == "1":
            camera = rooms.ol.PerspectiveCamera(position=position, target=(0.0, 1.8, 0.0))
        position = list(camera.position)
        if index < 12:
            print(json.dumps({"submit_index": index, "position": position,
                              "target": list(camera.target)}), flush=True)
        if os.environ.get("FORCE_RECREATE") == "1" and index in {2, 20, 40}:
            self.swapchain_extent = None
        result = original_present(self, scene, camera, width, height, **kwargs)
        if result is None:
            print(json.dumps({"skipped_acquire": index}), flush=True)
            return result
        if index < 12:
            keys = ["wavefront_frame_slot", "wavefront_history_source_slot",
                    "wavefront_history_dependency_waited", "wavefront_history_chain_enabled",
                    "wavefront_restir_history_valid", "wavefront_command_cache_hit",
                    "wavefront_temporal_motion_pixels", "wavefront_render_extent",
                    "wavefront_render_scale"]
            print(json.dumps({"completed_submit": index, "state": {
                key: self.last_timings.get(key) for key in keys}}), flush=True)
        self.diagnostic_present_index = index + 1
        return result
    core.VulkanRayQueryCore.present_wavefront_window = deterministic_present

if os.environ.get("TRACE_RESOURCES") == "1":
    def resource_state(self):
        return {"extent": self.swapchain_extent,
                "previous_camera": self.wavefront_previous_present_camera is not None,
                "frames": [{key: frame.get(key) for key in (
                    "wavefront_render_extent", "wavefront_reservoir_valid",
                    "wavefront_history_ready_pending")} | {
                    "cached": frame.get("wavefront_command_key") is not None}
                    for frame in self.window_frames]}
    def trace_method(name):
        original = getattr(core.VulkanRayQueryCore, name)
        def traced(self, *args, **kwargs):
            print(json.dumps({"resource_event": name, "phase": "before", "state": resource_state(self)}), flush=True)
            result = original(self, *args, **kwargs)
            print(json.dumps({"resource_event": name, "phase": "after", "state": resource_state(self)}), flush=True)
            return result
        setattr(core.VulkanRayQueryCore, name, traced)
    for name in ["create_window_swapchain", "_destroy_swapchain_resources", "_invalidate_scene_history", "_reset_wavefront_history_chain"]:
        trace_method(name)

app = QtWidgets.QApplication([])
timer = QtCore.QTimer()
ticks = 0
def poll():
    global ticks
    ticks += 1
    for window in app.topLevelWidgets():
        if hasattr(window, "completed_frame_count"):
            if os.environ.get("MODE") == "still":
                window.animate.setChecked(False)
            count = window.completed_frame_count
            failed = window.presentation_failed
            if count >= int(os.environ.get("FRAME_LIMIT", "120")) or failed or ticks >= 600:
                print(json.dumps({"frames":count,"failed":failed,
                    "status":window.status.text(),"extent":window.extent,
                    "animated":window.animate.isChecked(),
                    "slow":window.slow_diagnostic.isChecked()}),flush=True)
                timer.stop()
                if failed:
                    os._exit(2)
                window.close()
timer.timeout.connect(poll)
timer.start(100)
sys.argv = ["raster_feature_viewer.py", "--target", "wavefront-gi",
            "--showcase", "planar-mirror-guides"]
raise SystemExit(raster_workbench.main())
