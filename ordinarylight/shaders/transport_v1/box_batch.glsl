
uint OL_BOX_ENTRY(vec3 origin,vec3 direction,float t_min,float t_max,
    vec4 parameters,float tolerance,uint max_steps,inout OrdinaryLightCustomHit hit) {
    bool found=false;
    float closest=t_max;
    if(any(isnan(parameters.xy)) || any(isinf(parameters.xy)) ||
       parameters.x<0.0 || parameters.y<1.0 ||
       any(notEqual(floor(parameters.xy),parameters.xy)) ||
       any(greaterThan(parameters.xy,vec2(16777216.0)))) return 2u;
    uint first=uint(parameters.x),count=uint(parameters.y);
    if(count>max_steps || first+count>OL_BOX_INDEX_COUNT) return 2u;
    for(uint slot=first;slot<first+count;++slot) {
        uint i=OL_BOX_INDEX;
        if(i>=uint(OL_BOX_RESOURCE.length())/3u) return 2u;
        vec4 lower=OL_BOX_RESOURCE[i*3u],upper=OL_BOX_RESOURCE[i*3u+1u];
        if(lower.w==0.0) continue;
        if(lower.w!=1.0 || any(isnan(lower.xyz)) || any(isinf(lower.xyz)) ||
           any(isnan(upper.xyz)) || any(isinf(upper.xyz)) ||
           any(greaterThanEqual(lower.xyz,upper.xyz))) return 2u;
        float near_t=-1e30,far_t=1e30;
        vec3 near_n=vec3(0),far_n=vec3(0);
        bool missed=false;
        for(int axis=0;axis<3;++axis) {
            if(abs(direction[axis])<1e-20) {
                if(origin[axis]<lower[axis] || origin[axis]>upper[axis]) missed=true;
            } else {
                float a=(lower[axis]-origin[axis])/direction[axis];
                float b=(upper[axis]-origin[axis])/direction[axis];
                vec3 n=vec3(0); n[axis]=direction[axis]>0.0?-1.0:1.0;
                if(min(a,b)>near_t) { near_t=min(a,b); near_n=n; }
                if(max(a,b)<far_t) { far_t=max(a,b); far_n=-n; }
            }
        }
        if(missed || near_t>far_t) continue;
        // Bounds clipping may round the entry just below t_min.
        bool entry=near_t>=t_min-tolerance;
        float distance=entry?max(near_t,t_min):far_t;
        if(distance<t_min || distance>closest) continue;
        uvec4 metadata=floatBitsToUint(OL_BOX_RESOURCE[i*3u+2u]);
        hit.distance=distance;
        hit.geometric_normal=entry?near_n:far_n;
        hit.material=metadata.x;
        hit.boundary=metadata.y;
        hit.identity=metadata.z;
        hit.flags=OL_HIT_MATERIAL|OL_HIT_BOUNDARY|OL_HIT_IDENTITY;
        closest=distance; found=true;
    }
    return found?1u:0u;
}
