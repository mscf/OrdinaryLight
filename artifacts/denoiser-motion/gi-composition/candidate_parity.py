"""Compare legacy and shared candidate recording, including GPU reservoir words."""

import ast
from dataclasses import replace
from importlib.resources import files
import struct
import subprocess
import textwrap
from types import SimpleNamespace
import numpy as np
import vulkan as vk
import ordinarylight as ol
from ordinarylight.targets.vulkan.core import VulkanWavefrontExecutor
from ordinarylight.targets.vulkan.indirect_candidates_graph import _Kernel
from ordinarylight.runtime import VulkanKernel, compile_compute
from ordinarylight.pipeline.graph import VulkanGraph
from ordinarylight.pipeline.vulkan import VulkanResource, VulkanResourceUse, VulkanPass
from ordinarylight.integrations.glfw_platform import load_glfw
from ordinarylight.integrations.raster_workbench import _gi_config
from tools.denoiser_motion.glass_detail import fixture, pose


def main():
    source = subprocess.check_output(
        ["git", "show", "966c745fc5ea6dbc879c285d174998ccb520619a:ordinarylight/targets/vulkan/core.py"], text=True
    )
    cls = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == "VulkanWavefrontExecutor"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef)
        and n.name == "record_indirect_reuse_candidates"
    )
    namespace = {"vk": vk, "struct": struct}
    exec(textwrap.dedent(ast.get_source_segment(source, method)), namespace)
    legacy = namespace[method.name]
    shared = VulkanWavefrontExecutor.record_indirect_reuse_candidates

    def old(self, command, slot, *args):
        if slot not in self.candidate_stages:
            kernel = VulkanKernel(
                self.core.runtime,
                files("ordinarylight.shaders")
                .joinpath("wavefront_indirect_candidates.comp.spv")
                .read_bytes(),
                _Kernel(self, slot).bindings,
                push_constant_size=36,
            )
            self.candidate_stages[slot] = (None, kernel)
        kernel = self.candidate_stages[slot][1]
        self.indirect_reuse_candidate_pipeline = kernel.pipeline
        self.indirect_reuse_candidate_pipeline_layout = kernel.pipeline_layout
        self.indirect_reuse_candidate_sets = [kernel.descriptor] * 2
        try:
            legacy(self, command, slot, *args)
        finally:
            self.indirect_reuse_candidate_pipeline = None
            self.indirect_reuse_candidate_pipeline_layout = None
            self.indirect_reuse_candidate_sets = []

    code = compile_compute("""#version 460
layout(local_size_x=64) in;
layout(binding=0,std430) readonly buffer Source {uint source_words[];};
layout(binding=1,std430) writeonly buffer Target {uint target_words[];};
void main(){uint i=gl_GlobalInvocationID.x;if(i<source_words.length())target_words[i]=source_words[i];}
""")
    glfw = load_glfw()
    assert glfw.init()
    glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(160, 120, "candidate parity", None, None)
    captures = []
    try:
        for recorder in (old, shared):
            VulkanWavefrontExecutor.record_indirect_reuse_candidates = recorder
            scene, glass, bars = fixture()
            cfg = replace(
                _gi_config(
                    SimpleNamespace(id="glass-detail", renderer={}),
                    capture=True,
                    present=True,
                    render_scale=0.5,
                ),
                wavefront_indirect_reuse_storage=True,
                wavefront_indirect_reuse_candidates=True,
                wavefront_indirect_reuse_temporal=True,
                wavefront_indirect_reuse_spatial=True,
                wavefront_indirect_reuse_profiling=True,
            )
            frames = []
            with ol.VulkanGlfwPresenter(window, config=cfg) as p:
                for frame in range(4):
                    p.present_wavefront(
                        scene,
                        pose(scene, glass, bars, "camera", frame * 0.001),
                        160,
                        120,
                    )
                    hdr = p.capture_wavefront_hdr()
                    slot = (p._core.window_frame_index - 1) % 2
                    source_resource = _Kernel(
                        p._core.wavefront_executor, slot
                    ).bindings[0]
                    with p._core.runtime.buffer(source_resource.size) as target:
                        with VulkanKernel(
                            p._core.runtime,
                            code,
                            {0: source_resource, 1: VulkanResource.buffer(target)},
                        ) as reader:
                            uses = tuple(
                                VulkanResourceUse(
                                    r,
                                    vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    vk.VK_ACCESS_SHADER_READ_BIT
                                    if b == 0
                                    else vk.VK_ACCESS_SHADER_WRITE_BIT,
                                )
                                for b, r in reader.bindings.items()
                            )
                            VulkanGraph().add(
                                "read",
                                VulkanPass(
                                    "read",
                                    uses,
                                    reader.bind,
                                    ((source_resource.size // 4 + 63) // 64, 1, 1),
                                ),
                            ).compile().execute(p._core.runtime).wait()
                            frames.append((hdr, target.read()))
            captures.append(frames)
        for old_frame, new_frame in zip(*captures):
            assert np.array_equal(old_frame[0], new_frame[0])
            assert old_frame[1] == new_frame[1]
        assert any(any(data) for _hdr, data in captures[0]), (
            "Reservoir comparison must include nonzero data"
        )
        print("4 moving frames: HDR and compact reservoir bytes match legacy exactly")
    finally:
        VulkanWavefrontExecutor.record_indirect_reuse_candidates = shared
        glfw.destroy_window(window)
        glfw.terminate()


if __name__ == "__main__":
    main()
