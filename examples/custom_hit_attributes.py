"""One chunk AABB with two material/identity regions, using public APIs.

Run with a Vulkan ray-query GPU: python examples/custom_hit_attributes.py
The split plane is a diagnostic surface, not a closed optical region.
"""

import ordinarylight as ol
from ordinarylight.geometry import CustomGeometry, IntersectionProgram
from ordinarylight.transport import (
    VulkanTransportScene,
    TransportMaterial,
    intersect_rays,
)

PROGRAM = IntersectionProgram(
    "splitPlane",
    """
uint splitPlane(vec3 origin,vec3 direction,float t_min,float t_max,
    vec4 parameters,float tolerance,uint max_steps,inout OrdinaryLightCustomHit hit) {
    if(abs(direction.z)<1e-20) return 0u;
    float distance=-origin.z/direction.z;
    if(distance<t_min || distance>t_max) return 0u;
    vec3 position=origin+distance*direction;
    hit.distance=distance;
    hit.geometric_normal=vec3(0,0,1);
    hit.flags=OL_HIT_MATERIAL|OL_HIT_IDENTITY|OL_HIT_UV;
    hit.material=position.x<0?0u:1u;
    hit.identity=position.x<0?101u:102u;
    hit.uv=position.xy*0.5+0.5;
    return 1u;
}
""",
    hit_version=2,
)


def main():
    chunk = CustomGeometry(((-1, -1, -0.1), (1, 1, 0.1)), PROGRAM, (0, 0, 0, 0))
    with ol.VulkanRuntime() as runtime:
        with VulkanTransportScene(
            runtime,
            custom_geometry=[chunk],
            custom_materials=[
                TransportMaterial(albedo=(0.8, 0.1, 0.1)),
                TransportMaterial(albedo=(0.1, 0.1, 0.8)),
            ],
        ) as scene:
            hits = intersect_rays(
                scene,
                [[-0.5, 0, 2], [0.5, 0, 2]],
                [[0, 0, -1], [0, 0, -1]],
            )
            for hit in hits:
                print(
                    "kind/primitive/application/material:",
                    hit["identity"],
                    "status:",
                    hit["boundary"][3],
                )


if __name__ == "__main__":
    main()
