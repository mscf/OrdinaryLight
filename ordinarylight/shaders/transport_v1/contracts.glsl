#ifndef ORDINARYLIGHT_TRANSPORT_CONTRACTS_V1
#define ORDINARYLIGHT_TRANSPORT_CONTRACTS_V1 1
// All vectors occupy 16 bytes. See SURFACE_SAMPLE_DTYPE; no camera/pixel ABI.
struct OrdinaryLightSurfaceSample {
    vec4 position;
    vec4 geometric_normal; // oriented intersection normal for ray offsets
    vec4 shading_normal;   // scattering frame; need not match geometry
    vec4 incoming;         // direction travelling toward the surface
    uvec4 identity;        // application owner, sample index, material, flags
    uvec4 media;           // outside, inside medium IDs; application boundary ID (or 0xffffffff), reserved
};
// Medium membership is not inferred from a shading normal. The application
// supplies IDs and chooses its boundary representation/traversal algorithm.
bool ordinarylightEnteringMedium(vec3 direction, vec3 geometric_normal) {
    return dot(direction, geometric_normal) < 0.0;
}
uint ordinarylightDestinationMedium(OrdinaryLightSurfaceSample surface, vec3 direction) {
    return ordinarylightEnteringMedium(direction, surface.geometric_normal.xyz)
        ? surface.media.y : surface.media.x;
}
#endif
