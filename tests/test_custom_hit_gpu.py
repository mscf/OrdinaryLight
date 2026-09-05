"""Heterogeneous custom primitives exercise the public v2 hit ABI."""

import os
import numpy as np
import pytest
import ordinarylight as ol
from ordinarylight.geometry import CustomGeometry, IntersectionProgram
from ordinarylight.transport import (
    VulkanTransportScene,
    TransportMaterial,
    OpticalMedium,
    MediumBoundary,
    intersect_rays,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("ORDINARYLIGHT_TEST_VULKAN_TRANSPORT") != "1",
    reason="opt-in transport GPU validation",
)

SOURCE = """
uint chunkHit(vec3 origin,vec3 direction,float t_min,float t_max,
    vec4 parameters,float tolerance,uint max_steps,inout OrdinaryLightCustomHit hit) {
    float t=(parameters.x-origin.z)/direction.z;
    if(t<t_min || t>t_max) return 0u;
    hit.distance=t; hit.geometric_normal=vec3(0,0,1);
    hit.flags=OL_HIT_MATERIAL|OL_HIT_BOUNDARY|OL_HIT_IDENTITY|OL_HIT_UV|OL_HIT_SHADING_NORMAL;
    hit.material=origin.x<0?0u:1u;
    hit.boundary=origin.x<0?OL_NO_BOUNDARY:73u;
    hit.identity=origin.x<0?101u:102u;
    hit.uv=vec2(0.25,0.75);
    hit.shading_normal=normalize(vec3(0.2,0,1));
    REPLACE
    return 1u;
}
"""


def geometry(z=0, replacement="", **kwargs):
    return CustomGeometry(
        ((-2, -2, z - 0.1), (2, 2, z + 0.1)),
        IntersectionProgram(
            "chunkHit", SOURCE.replace("REPLACE", replacement), hit_version=2
        ),
        (z, 0, 0, 0),
        **kwargs,
    )


def test_heterogeneous_chunks_and_closest_hits():
    with ol.VulkanRuntime() as runtime:
        triangles = ol.Scene()
        triangles.add_mesh([[-2, -2, -2], [2, -2, -2], [0, 2, -2]], [[0, 1, 2]])
        with VulkanTransportScene(
            runtime,
            triangles,
            custom_geometry=[geometry(-1), geometry(0)],
            custom_materials=[TransportMaterial(), TransportMaterial("dielectric")],
            media=[OpticalMedium(), OpticalMedium(ior=1.5)],
            boundaries=[MediumBoundary(73, 0, 1)],
        ) as scene:
            hits = intersect_rays(scene, [[-0.5, 0, 3], [0.5, 0, 3]], [[0, 0, -1]] * 2)
            assert not hits["boundary"][:, 3].any()
            np.testing.assert_array_equal(hits["identity"][:, 1], [1, 1])
            np.testing.assert_array_equal(hits["identity"][:, 2], [101, 102])
            np.testing.assert_array_equal(hits["identity"][:, 3], [1, 2])
            np.testing.assert_array_equal(hits["boundary"][:, 0], [0xFFFFFFFF, 0])
            np.testing.assert_allclose(hits["position_distance"][:, 3], 3)
            np.testing.assert_allclose(hits["geometric_normal"][:, 3], 0.25)
            np.testing.assert_allclose(hits["shading_normal"][:, 3], 0.75)
            # A ray beginning between chunks hits the other chunk with the same region.
            other = intersect_rays(scene, [[0.5, 0, -0.5]], [[0, 0, -1]])
            assert other["identity"][0, 1] == 0
            assert other["boundary"][0, 0] == 0


@pytest.mark.parametrize(
    "replacement,status",
    [
        ("hit.material=999u;", 32),
        ("hit.boundary=999u;", 32),
        ("hit.boundary=OL_NO_BOUNDARY;", 32),
        ("hit.flags=128u;", 32),
        ("hit.shading_normal=vec3(0,0,-1);", 1),
        ("hit.uv=vec2(uintBitsToFloat(0x7fc00000u));", 1),
    ],
)
def test_invalid_custom_attributes(replacement, status):
    with ol.VulkanRuntime() as runtime:
        with VulkanTransportScene(
            runtime,
            custom_geometry=[geometry(replacement=replacement)],
            custom_materials=[TransportMaterial(), TransportMaterial("dielectric")],
            media=[OpticalMedium(), OpticalMedium(ior=1.5)],
            boundaries=[MediumBoundary(73, 0, 1)],
        ) as scene:
            hit = intersect_rays(scene, [[0.5, 0, 3]], [[0, 0, -1]])
            assert hit["boundary"][0, 3] == status


@pytest.mark.parametrize("clear", [False, True])
def test_inherited_attributes_and_explicit_boundary_clear(clear):
    replacement = (
        "hit.flags=OL_HIT_MATERIAL|OL_HIT_BOUNDARY; hit.material=0u; hit.boundary=OL_NO_BOUNDARY;"
        if clear
        else "hit.flags=0u;"
    )
    with ol.VulkanRuntime() as runtime:
        with VulkanTransportScene(
            runtime,
            custom_geometry=[
                geometry(replacement=replacement, material=1, boundary=73, identity=99)
            ],
            custom_materials=[TransportMaterial(), TransportMaterial("dielectric")],
            media=[OpticalMedium(), OpticalMedium(ior=1.5)],
            boundaries=[MediumBoundary(73, 0, 1)],
        ) as scene:
            hit = intersect_rays(scene, [[0.5, 0, 3]], [[0, 0, -1]])[0]
            assert hit["boundary"][3] == 0
            assert hit["identity"][2] == 99
            assert hit["identity"][3] == (0 if clear else 1)
            assert hit["boundary"][0] == (0xFFFFFFFF if clear else 0)
            np.testing.assert_array_equal(hit["geometric_normal"], [0, 0, 1, 0])
            np.testing.assert_array_equal(hit["shading_normal"], [0, 0, 1, 0])


def test_per_hit_dielectric_transport_matches_legacy_sphere():
    from ordinarylight.geometry import SdfSphere
    from ordinarylight.transport import (
        VulkanTransportIntegrator,
        GpuSampleAccumulator,
        ray_samples,
    )

    legacy = IntersectionProgram.sdf_sphere()
    program = IntersectionProgram(
        "attributedSphere",
        legacy.source
        + """
uint attributedSphere(vec3 origin,vec3 direction,float t_min,float t_max,
    vec4 parameters,float tolerance,uint max_steps,inout OrdinaryLightCustomHit hit) {
    uint status=ordinarylightSdfSphere(origin,direction,t_min,t_max,parameters,
        tolerance,max_steps,hit.distance,hit.geometric_normal);
    hit.flags=OL_HIT_MATERIAL|OL_HIT_BOUNDARY|OL_HIT_IDENTITY;
    hit.material=1u; hit.boundary=73u; hit.identity=123u;
    return status;
}
""",
        hit_version=2,
    )
    sphere = SdfSphere().geometry(material=1, boundary=73)
    attributed = CustomGeometry(sphere.bounds, program, sphere.parameters)
    results = []
    with ol.VulkanRuntime() as runtime:
        for shape in [sphere, attributed]:
            with VulkanTransportScene(
                runtime,
                custom_geometry=[shape],
                custom_materials=[TransportMaterial(), TransportMaterial("dielectric")],
                media=[
                    OpticalMedium(),
                    OpticalMedium(ior=1.5, absorption=(0.1, 0.2, 0.3)),
                ],
                boundaries=[MediumBoundary(73, 0, 1)],
            ) as scene:
                with GpuSampleAccumulator(runtime, 1) as accumulator:
                    with VulkanTransportIntegrator(
                        scene,
                        ray_samples([[0, 0, 3]], [[0, 0, -1]]),
                        accumulator,
                    ) as integrator:
                        integrator.accumulate(
                            samples_per_element=64, max_bounces=8, environment=(1, 1, 1)
                        ).wait()
                        results.append(accumulator.read())
    np.testing.assert_array_equal(results[0], results[1])
