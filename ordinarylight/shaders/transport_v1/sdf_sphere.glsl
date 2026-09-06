#ifndef ORDINARYLIGHT_SDF_SPHERE_V1
#define ORDINARYLIGHT_SDF_SPHERE_V1 1
// Exact world-space sphere SDF. Called only inside its declared AABB interval.
uint ordinarylightSdfSphere(vec3 origin, vec3 direction, float t_min, float t_max,
    vec4 parameters, float tolerance, uint max_steps, out float distance,
    out vec3 geometric_normal)
{
    distance = t_min;
    vec3 start_delta=origin+distance*direction-parameters.xyz;
    float start_value=length(start_delta)-parameters.w;
    // AABB clipping can place t_min just inside an entry surface through
    // rounding. Only suppress a departing root when the ray itself starts
    // near that surface, not when the clipped interval starts there.
    float origin_value=length(origin-parameters.xyz)-parameters.w;
    bool leaving_origin_root=abs(origin_value)<=tolerance &&
        abs(start_value)<=tolerance && start_value*dot(start_delta,direction)>0.0;
    for (uint step=0u; step<max_steps; ++step) {
        vec3 delta=origin+distance*direction-parameters.xyz;
        float value=length(delta)-parameters.w;
        if (leaving_origin_root && abs(value)>tolerance) leaving_origin_root=false;
        if (abs(value)<=tolerance && !leaving_origin_root) {
            geometric_normal=normalize(delta);
            return 1u;
        }
        distance+=leaving_origin_root?max(abs(value),tolerance):abs(value);
        if (distance>t_max+tolerance) return 0u;
    }
    return 2u;
}

#endif
