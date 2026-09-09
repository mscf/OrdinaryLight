"""Compare typed GLSL/WGSL EASU against the upstream FP32 algorithm on GPU."""

from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import ordinaryshade as osh
import wgpu
from ordinarylight.shaders.easu import EASU_HELPERS, easu_resolve

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from compile_shaders import find_compiler


@osh.compute(workgroup_size=(64, 1, 1))
def evaluate(
    inputs: osh.storage_buffer(osh.vec4, binding=0, access="read"),
    outputs: osh.storage_buffer(osh.vec4, binding=1),
):
    index = osh.global_invocation_id.x
    base = index * osh.u32(17)
    outputs[index] = osh.vec4(
        easu_resolve(
            inputs[base].xy,
            inputs[base + osh.u32(2)].rgb,
            inputs[base + osh.u32(3)].rgb,
            inputs[base + osh.u32(5)].rgb,
            inputs[base + osh.u32(6)].rgb,
            inputs[base + osh.u32(7)].rgb,
            inputs[base + osh.u32(8)].rgb,
            inputs[base + osh.u32(9)].rgb,
            inputs[base + osh.u32(10)].rgb,
            inputs[base + osh.u32(11)].rgb,
            inputs[base + osh.u32(12)].rgb,
            inputs[base + osh.u32(14)].rgb,
            inputs[base + osh.u32(15)].rgb,
        ),
        1.0,
    )


def main():
    out = Path("/tmp/easu-numerical")
    out.mkdir(exist_ok=True)
    reference = """#version 450
#extension GL_GOOGLE_include_directive : require
#define A_GPU 1
#define A_GLSL 1
#include "ffx_a.h"
#define FSR_EASU_F 1
#include "ffx_fsr1.h"
layout(local_size_x=64) in;
layout(set=0,binding=0,std430) readonly buffer Input { vec4 inputs[]; };
layout(set=0,binding=1,std430) buffer Output { vec4 outputs[]; };
vec4 gatherValue(vec2 p,int channel) {
 ivec2 q=ivec2(floor(p*4.0-0.5))+ivec2(1);
 ivec2 a=clamp(q+ivec2(0,1),ivec2(0),ivec2(3));
 ivec2 b=clamp(q+ivec2(1,1),ivec2(0),ivec2(3));
 ivec2 c=clamp(q+ivec2(1,0),ivec2(0),ivec2(3));
 ivec2 d=clamp(q,ivec2(0),ivec2(3));
 uint base=gl_GlobalInvocationID.x*17u+1u;
 return vec4(inputs[base+uint(a.y*4+a.x)][channel],inputs[base+uint(b.y*4+b.x)][channel],inputs[base+uint(c.y*4+c.x)][channel],inputs[base+uint(d.y*4+d.x)][channel]);
}
vec4 FsrEasuRF(vec2 p){return gatherValue(p,0);}
vec4 FsrEasuGF(vec2 p){return gatherValue(p,1);}
vec4 FsrEasuBF(vec2 p){return gatherValue(p,2);}
void main(){
 vec3 result;vec2 pp=inputs[gl_GlobalInvocationID.x*17u].xy;
 FsrEasuF(result,uvec2(0),floatBitsToUint(vec4(0,0,pp)),
 floatBitsToUint(vec4(.25,.25,.25,-.25)),floatBitsToUint(vec4(-.25,.5,.25,.5)),floatBitsToUint(vec4(0,1,0,0)));
 outputs[gl_GlobalInvocationID.x]=vec4(result,1);
}
"""
    (out / "reference.comp").write_text(reference)
    subprocess.run(
        [
            find_compiler(),
            "-V",
            "-S",
            "comp",
            "-I" + str(ROOT / "ordinarylight/shaders/third_party/fsr1"),
            str(out / "reference.comp"),
            "-o",
            str(out / "reference.spv"),
        ],
        check=True,
    )
    wgsl = osh.compile(
        evaluate, helpers=EASU_HELPERS, target="wgsl", validate=True
    ).source
    spirv = osh.compile(
        evaluate,
        helpers=EASU_HELPERS,
        target="spirv",
        spirv_compiler=find_compiler(),
        validate=True,
    ).binary
    (out / "port.wgsl").write_text(wgsl)
    rng = np.random.default_rng(419)
    values = rng.random((4096, 17, 4), dtype=np.float32)
    values[0, 1:, :3] = 0
    values[1, 1:, :3] = 1
    for index in range(2, 258):
        y, x = np.mgrid[:4, :4]
        angle = index * 0.11
        patch = (x * np.cos(angle) + y * np.sin(angle) > np.sin(index) * 2).astype(
            np.float32
        )
        values[index, 1:, :3] = patch.reshape(16, 1)
    for index in range(258, 514):
        patch = np.linspace(0, 1, 16, dtype=np.float32)
        values[index, 1:, :3] = patch[:, None]
    values[514:770, 1:, :3] = (np.indices((4, 4)).sum(0) % 2).reshape(1, 16, 1)
    hdr = np.exp(rng.uniform(-12, 12, (1024, 16, 3)))
    values[770:1794, 1:, :3] = (hdr / (1 + hdr)) ** (1 / 2.2)
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = adapter.request_device_sync()
    results = {}
    for name, code in [
        ("amd", (out / "reference.spv").read_bytes()),
        ("shade-glsl", bytes(spirv)),
        ("shade-wgsl", wgsl),
    ]:
        pipeline = device.create_compute_pipeline(
            layout="auto",
            compute={
                "module": device.create_shader_module(code=code),
                "entry_point": "main",
            },
        )
        inp = device.create_buffer_with_data(
            data=values, usage=wgpu.BufferUsage.STORAGE
        )
        output = device.create_buffer(
            size=4096 * 16, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )
        group = device.create_bind_group(
            layout=pipeline.get_bind_group_layout(0),
            entries=[
                {"binding": 0, "resource": {"buffer": inp}},
                {"binding": 1, "resource": {"buffer": output}},
            ],
        )
        encoder = device.create_command_encoder()
        compute = encoder.begin_compute_pass()
        compute.set_pipeline(pipeline)
        compute.set_bind_group(0, group)
        compute.dispatch_workgroups(64)
        compute.end()
        device.queue.submit([encoder.finish()])
        results[name] = (
            np.frombuffer(device.queue.read_buffer(output), np.float32)
            .copy()
            .reshape(-1, 4)
        )
        assert np.isfinite(results[name]).all()
        centers = values[:, [6, 7, 10, 11], :3]
        assert np.all(results[name][:, :3] >= centers.min(1) - 1e-6)
        assert np.all(results[name][:, :3] <= centers.max(1) + 1e-6)
        inp.destroy()
        output.destroy()
    report = {}
    for name in ("shade-glsl", "shade-wgsl"):
        delta = results[name] - results["amd"]
        report[name] = {
            "max_abs": float(np.abs(delta).max()),
            "rmse": float(np.sqrt(np.mean(delta**2))),
            "cases": 4096,
        }
        assert np.abs(delta).max() < 2e-5, report[name]
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    print(report)


if __name__ == "__main__":
    main()
