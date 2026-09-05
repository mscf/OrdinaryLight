"""Sample analytic lighting by application ID on a runtime with no GI renderer.

Run with the Vulkan extra and glslangValidator installed. This deliberately
small fixture uses untextured surfaces and vacuum; arbitrary material programs,
texture sampling and participating media require their documented adapters.
"""

from contextlib import ExitStack
import struct

import numpy as np
import vulkan as vk

import ordinarylight as ol
from ordinarylight.pipeline.vulkan import (
    VulkanPass,
    VulkanPassPipeline,
    VulkanResource,
    VulkanResourceUse,
)
from ordinarylight.runtime import VulkanKernel, compile_compute
from ordinarylight.transport import SURFACE_SAMPLE_DTYPE, shader_source


def sample_shader():
    return (
        """#version 460
#extension GL_EXT_ray_query : require
layout(local_size_x=64) in;
"""
        + shader_source("types")
        + shader_source("contracts")
        + """
layout(set=0,binding=0,std430) readonly buffer Samples { OrdinaryLightSurfaceSample samples[]; };
layout(set=0,binding=1,std430) writeonly buffer Results { vec4 results[]; };
layout(set=0,binding=2,std430) readonly buffer Materials { MaterialData materials[]; };
layout(set=0,binding=3,std430) readonly buffer Lights { PointLightData point_lights[]; };
layout(set=0,binding=4,std430) readonly buffer AreaLights { AreaLightData area_lights[]; };
layout(set=0,binding=5) uniform accelerationStructureEXT scene_tlas;
layout(push_constant) uniform Constants { uint count; uint lights; } pc;
#define OL_TRANSPORT_POINT_LIGHT_COUNT pc.lights
#define OL_TRANSPORT_AREA_LIGHT_COUNT 0u
#define OL_TRANSPORT_AREA_LIGHT_WEIGHT 0.0
#define OL_TRANSPORT_ENVIRONMENT_SAMPLES 0u
#define OL_TRANSPORT_SECONDARY_AREA_LIGHT_SAMPLES 0u
vec3 geometric_normal;
#define OL_TRANSPORT_RAY_ORIGIN(hit, normal) ((hit) + geometric_normal * 0.002)
// This fixture explicitly selects vacuum and no environment textures.
float volumeShadowTransmittance(vec3 origin,vec3 direction,float distance) { return 1.0; }
vec4 sampleSceneTexture(int texture_index,vec2 uv) { return vec4(0.0); }
"""
        + shader_source("lighting")
        + """
void main() {
    uint i=gl_GlobalInvocationID.x;
    if (i>=pc.count) return;
    OrdinaryLightSurfaceSample surface=samples[i];
    geometric_normal=surface.geometric_normal.xyz;
    vec3 value=samplePointLights(surface.position.xyz,surface.shading_normal.xyz,
        surface.incoming.xyz,materials[surface.identity.z]);
    results[surface.identity.x]=vec4(value,1.0);
}
"""
    )


def run():
    scene = ol.Scene()
    scene.add_mesh(
        [[-2, -2, 0], [2, -2, 0], [0, 2, 0]],
        [[0, 1, 2]],
        ol.Material(base_color=(0.8, 0.7, 0.6)),
    )
    scene.add_light(ol.PointLight(position=(0, 0, 2), intensity=10.0))
    samples = np.zeros(2, SURFACE_SAMPLE_DTYPE)
    samples["position"][:, :3] = [[0, 0, 0], [0.5, 0, 0]]
    samples["geometric_normal"][:, 2] = 1
    samples["shading_normal"][:, 2] = 1
    samples["incoming"][:, 2] = -1
    samples["identity"][:, 0] = [1, 0]  # intentionally not dispatch or pixel order
    with ExitStack() as stack:
        runtime = stack.enter_context(ol.VulkanRuntime())
        resident = stack.enter_context(runtime.upload_scene(scene))
        inputs = stack.enter_context(runtime.buffer(samples.nbytes, data=samples))
        output = stack.enter_context(runtime.buffer(2 * 16))
        bindings = {
            0: VulkanResource.buffer(inputs),
            1: VulkanResource.buffer(output),
            2: resident.resource("material"),
            3: resident.resource("light"),
            4: resident.resource("area_light"),
            5: resident.resource("tlas"),
        }
        kernel = stack.enter_context(
            VulkanKernel(
                runtime,
                compile_compute(sample_shader()),
                bindings,
                push_constant_size=8,
            )
        )
        uses = tuple(
            VulkanResourceUse(
                resource,
                vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.VK_ACCESS_SHADER_WRITE_BIT
                if binding == 1
                else vk.VK_ACCESS_SHADER_READ_BIT,
            )
            for binding, resource in bindings.items()
        )
        VulkanPassPipeline(
            [
                VulkanPass(
                    "sample_surfaces",
                    uses,
                    lambda command: kernel.bind(command, struct.pack("II", 2, 1)),
                    (1, 1, 1),
                )
            ]
        ).execute(runtime).wait()
        result = np.frombuffer(output.read(), np.float32).reshape(2, 4).copy()
        assert np.isfinite(result).all() and np.all(result[:, :3] > 0)
        # The centered, closer sample writes slot 1 and should receive more light.
        assert np.all(result[1, :3] > result[0, :3])
        print("Radiance by application ID:", result)
        return result


if __name__ == "__main__":
    run()
