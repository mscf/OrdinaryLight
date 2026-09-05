// Bound resources are declared by the integrator/diagnostic client.
struct OrdinaryLightHit {
    vec4 position_distance;
    vec4 geometric_normal;
    vec4 shading_normal;
    uvec4 identity; // kind (0 miss,1 triangle,2 custom), primitive, app ID, material
    uvec4 boundary; // boundary index, outside medium, inside medium, status
};
bool ordinarylightBounds(vec3 origin,vec3 direction,vec3 lower,vec3 upper,
                        inout float near_t,inout float far_t) {
    for (int axis=0;axis<3;++axis) {
        if (abs(direction[axis])<1e-20) {
            if (origin[axis]<lower[axis] || origin[axis]>upper[axis]) return false;
        } else {
            float a=(lower[axis]-origin[axis])/direction[axis];
            float b=(upper[axis]-origin[axis])/direction[axis];
            near_t=max(near_t,min(a,b)); far_t=min(far_t,max(a,b));
        }
    }
    return near_t<=far_t;
}
OrdinaryLightHit ordinarylightIntersect(vec3 origin,vec3 direction,float t_min,float t_max,
                                     float tolerance,uint max_steps) {
    OrdinaryLightHit hit;
    hit.position_distance=vec4(0,0,0,t_max);
    hit.geometric_normal=vec4(0); hit.shading_normal=vec4(0);
    hit.identity=uvec4(0); hit.boundary=uvec4(0xffffffffu,0,0,0);
    rayQueryEXT query;
    rayQueryInitializeEXT(query,transport_tlas,gl_RayFlagsOpaqueEXT,0xff,
        origin,t_min,direction,t_max);
    vec3 generated_normal=vec3(0);
    while (rayQueryProceedEXT(query)) {
        if (rayQueryGetIntersectionTypeEXT(query,false)!=gl_RayQueryCandidateIntersectionAABBEXT) continue;
        uint index=rayQueryGetIntersectionPrimitiveIndexEXT(query,false);
        CustomRecord geometry=custom_geometry[index];
        float near_t=t_min,far_t=t_max;
        if (rayQueryGetIntersectionTypeEXT(query,true)!=gl_RayQueryCommittedIntersectionNoneEXT)
            far_t=min(far_t,rayQueryGetIntersectionTEXT(query,true));
        if (!ordinarylightBounds(origin,direction,geometry.lower.xyz,geometry.upper.xyz,near_t,far_t)) continue;
        float distance=0; vec3 normal=vec3(0);
        uint status=ordinarylightCustomIntersect(geometry.metadata.x,origin,direction,near_t,far_t,
            geometry.parameters,tolerance,max_steps,distance,normal);
        if (status>1u || (status==1u && (isnan(distance)||isinf(distance)||
            any(isnan(normal))||any(isinf(normal))||abs(dot(normal,normal)-1.0)>0.001||distance<near_t-tolerance||distance>far_t+tolerance))) {
            hit.boundary.w=1u;
            rayQueryTerminateEXT(query);
            return hit;
        }
        if (status==1u && distance<=far_t && distance>=t_min) {
            rayQueryGenerateIntersectionEXT(query,distance);
            generated_normal=normalize(normal);
        }
    }
    uint kind=rayQueryGetIntersectionTypeEXT(query,true);
    if (kind==gl_RayQueryCommittedIntersectionNoneEXT) return hit;
    float distance=rayQueryGetIntersectionTEXT(query,true);
    uint index=rayQueryGetIntersectionPrimitiveIndexEXT(query,true);
    hit.position_distance=vec4(origin+distance*direction,distance);
    if (kind==gl_RayQueryCommittedIntersectionTriangleEXT) {
        index+=rayQueryGetIntersectionInstanceCustomIndexEXT(query,true);
        vec3 a=transport_vertices[index*3u].xyz;
        vec3 b=transport_vertices[index*3u+1u].xyz;
        vec3 c=transport_vertices[index*3u+2u].xyz;
        vec3 geometric=normalize(cross(b-a,c-a));
        vec2 uv=rayQueryGetIntersectionBarycentricsEXT(query,true);
        vec3 shading=transport_attributes[index*9u].xyz*(1.0-uv.x-uv.y)
                    +transport_attributes[index*9u+3u].xyz*uv.x
                    +transport_attributes[index*9u+6u].xyz*uv.y;
        if (dot(shading,shading)<1e-12) shading=geometric;
        shading=normalize(shading);
        if (dot(shading,geometric)<0) shading=-shading;
        uvec4 metadata=triangle_records[index];
        hit.identity=uvec4(1,index,metadata.z,metadata.x);
        hit.boundary.x=metadata.y;
        hit.geometric_normal=vec4(geometric,0);
        hit.shading_normal=vec4(shading,0);
    } else {
        CustomRecord geometry=custom_geometry[index];
        hit.identity=uvec4(2,index,geometry.metadata.w,geometry.metadata.y);
        hit.boundary.x=geometry.metadata.z;
        hit.geometric_normal=vec4(generated_normal,0);
        hit.shading_normal=vec4(generated_normal,0);
    }
    if (hit.boundary.x!=0xffffffffu)
        hit.boundary.yz=medium_boundaries[hit.boundary.x].xy;
    return hit;
}
