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
    OrdinaryLightHit generated_hit=hit;
    while (rayQueryProceedEXT(query)) {
        if (rayQueryGetIntersectionTypeEXT(query,false)!=gl_RayQueryCandidateIntersectionAABBEXT) continue;
        uint index=rayQueryGetIntersectionPrimitiveIndexEXT(query,false);
        CustomRecord geometry=custom_geometry[index];
        if(geometry.metadata.x==0xffffffffu) continue;
        float near_t=t_min,far_t=t_max;
        if (rayQueryGetIntersectionTypeEXT(query,true)!=gl_RayQueryCommittedIntersectionNoneEXT)
            far_t=min(far_t,rayQueryGetIntersectionTEXT(query,true));
        if (!ordinarylightBounds(origin,direction,geometry.lower.xyz,geometry.upper.xyz,near_t,far_t)) continue;
        OrdinaryLightCustomHit result;
        result.distance=0; result.geometric_normal=vec3(0);
        result.flags=0u; result.material=0u; result.boundary=OL_NO_BOUNDARY;
        result.identity=0u; result.uv=vec2(0); result.shading_normal=vec3(0);
        uint status=ordinarylightCustomIntersect(geometry.metadata.x,origin,direction,near_t,far_t,
            geometry.parameters,tolerance,max_steps,result);
        float distance=result.distance; vec3 normal=result.geometric_normal;
        if (status>1u || (status==1u && (isnan(distance)||isinf(distance)||
            any(isnan(normal))||any(isinf(normal))||abs(dot(normal,normal)-1.0)>0.001||distance<near_t-tolerance||distance>far_t+tolerance))) {
            hit.boundary.w=1u;
            rayQueryTerminateEXT(query);
            return hit;
        }
        if (status==1u && distance<=far_t && distance>=t_min) {
            uint material=geometry.metadata.y, boundary=geometry.metadata.z;
            uint identity=geometry.metadata.w;
            if ((result.flags & ~31u)!=0u) {
                hit.boundary.w=32u; rayQueryTerminateEXT(query); return hit;
            }
            if ((result.flags & OL_HIT_MATERIAL)!=0u) {
                if (result.material>=OL_MATERIAL_COUNT-OL_CUSTOM_MATERIAL_OFFSET) {
                    hit.boundary.w=32u; rayQueryTerminateEXT(query); return hit;
                }
                material=OL_CUSTOM_MATERIAL_OFFSET+result.material;
            }
            if ((result.flags & OL_HIT_BOUNDARY)!=0u) {
                boundary=OL_NO_BOUNDARY;
                if (result.boundary!=OL_NO_BOUNDARY) {
                    for(uint i=0u;i<OL_BOUNDARY_COUNT;++i)
                        if(medium_boundaries[i].z==result.boundary) { boundary=i; break; }
                    if(boundary==OL_NO_BOUNDARY) {
                        hit.boundary.w=32u; rayQueryTerminateEXT(query); return hit;
                    }
                }
            }
            if(material>=OL_MATERIAL_COUNT ||
               (boundary!=OL_NO_BOUNDARY && boundary>=OL_BOUNDARY_COUNT)) {
                hit.boundary.w=32u; rayQueryTerminateEXT(query); return hit;
            }
            if((transport_materials[material].albedo_kind.w==1.0)!=(boundary!=OL_NO_BOUNDARY)) {
                hit.boundary.w=32u; rayQueryTerminateEXT(query); return hit;
            }
            if ((result.flags & OL_HIT_IDENTITY)!=0u) identity=result.identity;
            vec2 uv=(result.flags & OL_HIT_UV)!=0u?result.uv:vec2(0);
            vec3 shading=(result.flags & OL_HIT_SHADING_NORMAL)!=0u?result.shading_normal:normal;
            if(any(isnan(uv))||any(isinf(uv))||any(isnan(shading))||any(isinf(shading))||
               abs(dot(shading,shading)-1.0)>0.001||dot(shading,normal)<=0.0) {
                hit.boundary.w=1u; rayQueryTerminateEXT(query); return hit;
            }
            rayQueryGenerateIntersectionEXT(query,distance);
            // Ray queries carry no user attributes: retain only the committed candidate.
            if(rayQueryGetIntersectionTypeEXT(query,true)==gl_RayQueryCommittedIntersectionGeneratedEXT &&
               rayQueryGetIntersectionPrimitiveIndexEXT(query,true)==index &&
               rayQueryGetIntersectionTEXT(query,true)==distance) {
                generated_hit.identity=uvec4(2,index,identity,material);
                generated_hit.boundary.x=boundary;
                generated_hit.geometric_normal=vec4(normalize(normal),uv.x);
                generated_hit.shading_normal=vec4(normalize(shading),uv.y);
            }
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
        vec2 texcoord=transport_attributes[index*9u+1u].xy*(1.0-uv.x-uv.y)
                    +transport_attributes[index*9u+4u].xy*uv.x
                    +transport_attributes[index*9u+7u].xy*uv.y;
        hit.geometric_normal=vec4(geometric,texcoord.x);
        hit.shading_normal=vec4(shading,texcoord.y);
    } else {
        hit.identity=generated_hit.identity;
        hit.boundary.x=generated_hit.boundary.x;
        hit.geometric_normal=generated_hit.geometric_normal;
        hit.shading_normal=generated_hit.shading_normal;
    }
    if (hit.boundary.x!=0xffffffffu)
        hit.boundary.yz=medium_boundaries[hit.boundary.x].xy;
    return hit;
}
