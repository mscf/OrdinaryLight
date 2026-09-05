"""Ray diagnostics through the same traversal used by multi-bounce transport."""

from contextlib import ExitStack
import struct

import numpy as np

HIT_DTYPE = np.dtype(
    [
        ("position_distance", "<f4", (4,)),
        ("geometric_normal", "<f4", (4,)),
        ("shading_normal", "<f4", (4,)),
        ("identity", "<u4", (4,)),
        ("boundary", "<u4", (4,)),
    ]
)


def intersect_rays(
    scene, origins, directions, *, t_min=1e-4, t_max=1e6, tolerance=1e-5, max_steps=256
):
    import vulkan as vk
    from ..runtime import VulkanKernel, compile_compute
    from ..pipeline.vulkan import (
        VulkanResource,
        VulkanResourceUse,
        VulkanPass,
        VulkanPassPipeline,
    )
    from ._shaders import scene_source, SCENE_BINDINGS

    scene.require_open()
    origins = np.asarray(origins, np.float32)
    directions = np.asarray(directions, np.float32)
    if (
        origins.ndim != 2
        or origins.shape[1] != 3
        or origins.shape != directions.shape
        or not len(origins)
        or not np.isfinite(origins).all()
        or not np.isfinite(directions).all()
        or not np.allclose(np.linalg.norm(directions, axis=1), 1, rtol=1e-5)
    ):
        raise ValueError("Supply finite (N,3) origins and normalized directions")
    if (
        not np.isfinite([t_min, t_max, tolerance]).all()
        or not 0 <= t_min < t_max
        or tolerance <= 0
        or not 1 <= max_steps <= 65536
    ):
        raise ValueError("Invalid intersection limits")
    packed = np.zeros((len(origins), 2, 4), np.float32)
    packed[:, 0, :3] = origins
    packed[:, 1, :3] = directions
    source = (
        scene_source(scene)
        + """
struct RayInput { vec4 origin; vec4 direction; };
layout(set=0,binding=8,std430) readonly buffer Rays { RayInput rays[]; };
layout(set=0,binding=9,std430) writeonly buffer Hits { OrdinaryLightHit hits[]; };
layout(push_constant) uniform Constants { uint count; float t_min; float t_max; float tolerance; uint max_steps; } pc;
void main() {
    uint i=gl_GlobalInvocationID.x; if(i>=pc.count) return;
    hits[i]=ordinarylightIntersect(rays[i].origin.xyz,rays[i].direction.xyz,pc.t_min,pc.t_max,pc.tolerance,pc.max_steps);
}
"""
    )
    with scene.runtime.lock, ExitStack() as stack:
        inputs = stack.enter_context(scene.runtime.buffer(packed.nbytes, data=packed))
        outputs = stack.enter_context(
            scene.runtime.buffer(len(origins) * HIT_DTYPE.itemsize)
        )
        bindings = {i: scene.resource(name) for i, name in enumerate(SCENE_BINDINGS)}
        bindings.update(
            {8: VulkanResource.buffer(inputs), 9: VulkanResource.buffer(outputs)}
        )
        kernel = stack.enter_context(
            VulkanKernel(
                scene.runtime, compile_compute(source), bindings, push_constant_size=20
            )
        )
        uses = tuple(
            VulkanResourceUse(
                resource,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_ACCESS_SHADER_WRITE_BIT
                if binding == 9
                else vk.VK_ACCESS_SHADER_READ_BIT,
            )
            for binding, resource in bindings.items()
        )
        VulkanPassPipeline(
            [
                VulkanPass(
                    "intersections",
                    uses,
                    lambda command: kernel.bind(
                        command,
                        struct.pack(
                            "IfffI", len(origins), t_min, t_max, tolerance, max_steps
                        ),
                    ),
                    ((len(origins) + 63) // 64, 1, 1),
                )
            ]
        ).execute(scene.runtime).wait()
        return np.frombuffer(outputs.read(), HIT_DTYPE).copy()
