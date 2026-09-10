"""Mechanical resource ABI and variant guards. Algorithms are typed helpers."""
LAYOUT = r'''// Bound resources are declared by the integrator/diagnostic client.
struct OrdinaryLightHit {
    vec4 position_distance;
    vec4 geometric_normal;
    vec4 shading_normal;
    uvec4 identity; // kind (0 miss,1 triangle,2 custom), primitive, app ID, material
    uvec4 boundary; // boundary index, outside medium, inside medium, status
};
@ordinarylightBounds@
@ordinarylightIntersect@
'''
