// Uniform primitive-slot selection deliberately includes null/inactive slots.
// This distribution remains stable under GPU custom-geometry edits.
struct OrdinaryLightEmitterSample {
    vec3 position;
    vec3 normal;
    float area_pdf;
    uint kind;
    uint primitive;
};
bool ordinarylightValidAreaPdf(float pdf) {
    return !isnan(pdf) && !isinf(pdf) && pdf>=0.0;
}
float ordinarylightTriangleArea(uint index) {
    vec3 a=transport_vertices[index*3u].xyz;
    vec3 b=transport_vertices[index*3u+1u].xyz;
    vec3 c=transport_vertices[index*3u+2u].xyz;
    return 0.5*length(cross(b-a,c-a));
}
uint ordinarylightSampleSurface(vec4 randoms,out OrdinaryLightEmitterSample selected) {
    randoms=min(randoms,vec4(0.99999994));
    uint slot=min(uint(randoms.x*float(OL_SURFACE_SLOT_COUNT)),OL_SURFACE_SLOT_COUNT-1u);
    uint status;
    if(slot<OL_TRIANGLE_COUNT) {
        vec3 a=transport_vertices[slot*3u].xyz;
        vec3 b=transport_vertices[slot*3u+1u].xyz;
        vec3 c=transport_vertices[slot*3u+2u].xyz;
        float area=ordinarylightTriangleArea(slot);
        if(area<=0.0) return 0u;
        float root=sqrt(randoms.y);
        selected.position=a*(1.0-root)+b*(root*(1.0-randoms.z))+c*(root*randoms.z);
        selected.normal=normalize(cross(b-a,c-a));
        selected.area_pdf=1.0/area;
        selected.kind=1u; selected.primitive=slot;
        status=1u;
    } else {
        uint index=slot-OL_TRIANGLE_COUNT;
        CustomRecord geometry=custom_geometry[index];
        if(geometry.metadata.x==0xffffffffu) return 0u;
        selected.kind=2u; selected.primitive=index;
        status=ordinarylightCustomSurfaceSample(geometry.metadata.x,geometry.parameters,
            randoms.yzw,selected.position,selected.normal,selected.area_pdf);
        if(status!=1u) return status>1u?2u:0u;
        if(any(lessThan(selected.position,geometry.lower.xyz)) ||
           any(greaterThan(selected.position,geometry.upper.xyz))) return 2u;
        float reverse_pdf=ordinarylightCustomSurfacePdf(geometry.metadata.x,
            geometry.parameters,selected.position,selected.normal);
        if(!ordinarylightValidAreaPdf(reverse_pdf) ||
           abs(reverse_pdf-selected.area_pdf)>1e-4*max(reverse_pdf,selected.area_pdf)) return 2u;
    }
    if(any(isnan(selected.position))||any(isinf(selected.position))||
       any(isnan(selected.normal))||any(isinf(selected.normal))||
       abs(dot(selected.normal,selected.normal)-1.0)>0.001||
       !ordinarylightValidAreaPdf(selected.area_pdf)||selected.area_pdf==0.0) return 2u;
    return status;
}
float ordinarylightSurfaceLightPdf(vec3 origin,OrdinaryLightHit hit,inout uint status) {
    float area_pdf=0.0;
    if(hit.identity.x==1u) {
        float area=ordinarylightTriangleArea(hit.identity.y);
        if(area>0.0) area_pdf=1.0/area;
    } else if(hit.identity.x==2u) {
        CustomRecord geometry=custom_geometry[hit.identity.y];
        area_pdf=ordinarylightCustomSurfacePdf(geometry.metadata.x,geometry.parameters,
            hit.position_distance.xyz,hit.geometric_normal.xyz);
    }
    if(!ordinarylightValidAreaPdf(area_pdf)) { status|=1u; return 0.0; }
    vec3 offset=hit.position_distance.xyz-origin;
    float distance_squared=dot(offset,offset);
    if(area_pdf==0.0 || distance_squared==0.0) return 0.0;
    float cosine=abs(dot(hit.geometric_normal.xyz,-normalize(offset)));
    if(cosine<=0.0) return 0.0;
    float pdf=area_pdf*distance_squared/(float(OL_SURFACE_SLOT_COUNT)*cosine);
    if(!ordinarylightValidAreaPdf(pdf)) { status|=1u; return 0.0; }
    return pdf;
}

bool ordinarylightValidTransportMaterial(MaterialEvaluation evaluated,TransportMaterialRecord material,OrdinaryLightHit hit) {
    return !(isnan(evaluated.metallic) || isinf(evaluated.metallic) || isnan(evaluated.roughness) || isinf(evaluated.roughness) ||
                isnan(evaluated.ior) || isinf(evaluated.ior) || evaluated.ior<=0.0 ||
                (material.albedo_kind.w==0.0 && (evaluated.metallic!=0.0 || evaluated.roughness!=0.0)) ||
                (material.albedo_kind.w==1.0 && (evaluated.metallic!=0.0 || evaluated.ior!=optical_media[hit.boundary.z].a)) ||
                any(notEqual(evaluated.attenuation_color,vec3(1))) || evaluated.attenuation_distance!=1e30 ||
                evaluated.transmission!=float(material.albedo_kind.w==1.0) ||
                any(isnan(evaluated.base_color)) || any(isinf(evaluated.base_color)) ||
                any(isnan(evaluated.emission)) || any(isinf(evaluated.emission)));
}


// Scaling avoids overflow and preserves normalization when both PDFs are small.
float ordinarylightEmissiveMisWeight(float first_pdf,float second_pdf) {
    float scale=max(first_pdf,second_pdf);
    if(scale<=0.0) return 0.0;
    float a=first_pdf/scale, b=second_pdf/scale;
    return a*a/(a*a+b*b);
}
