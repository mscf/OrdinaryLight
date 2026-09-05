// One invocation per application identity. Samples accumulate without float
// atomics because identities are unique within a dispatch; batches are ordered.
struct SampleAccumulation { vec4 radiance; uvec4 counts; uvec4 events; };
layout(set=0,binding=8,std430) readonly buffer Samples { OrdinaryLightSurfaceSample transport_samples[]; };
layout(set=0,binding=9,std430) buffer Accumulation { SampleAccumulation accumulated[]; };
layout(set=0,binding=10,std430) readonly buffer InitialStack { uvec4 initial_stack[]; };
layout(push_constant) uniform Constants {
    uint count; uint samples_per_element; uint max_bounces; uint sample_offset;
    uint seed; uint initial_depth; float tolerance; float ray_epsilon;
    uint max_steps; float max_distance; uint padding0; uint padding1;
    vec4 environment;
} pc;

void main() {
    uint i=gl_GlobalInvocationID.x; if(i>=pc.count) return;
    OrdinaryLightSurfaceSample input_sample=transport_samples[i];
    uint owner=input_sample.identity.x;
    SampleAccumulation result=accumulated[owner];
    for(uint sample_index=0u;sample_index<pc.samples_per_element;++sample_index) {
        uint rng=secondaryNeeHash(owner^secondaryNeeHash(pc.seed)^secondaryNeeHash(
            pc.sample_offset+sample_index)^secondaryNeeHash(input_sample.identity.y));
        uint medium_stack[8]; uint boundary_stack[8];
        uint depth=pc.initial_depth;
        for(uint j=0u;j<depth;++j) {
            medium_stack[j]=initial_stack[j].x;
            boundary_stack[j]=initial_stack[j].y;
        }
        vec3 origin=input_sample.position.xyz;
        vec3 direction=input_sample.incoming.xyz;
        vec3 throughput=vec3(1); vec3 radiance=vec3(0);
        uint status=0u; bool truncated=false;
        uvec4 events=uvec4(0);
        for(uint bounce=0u;bounce<=pc.max_bounces;++bounce) {
            OrdinaryLightHit hit;
            if(bounce==0u && (input_sample.identity.w&1u)!=0u) {
                hit.position_distance=vec4(origin,0);
                hit.geometric_normal=input_sample.geometric_normal;
                hit.shading_normal=input_sample.shading_normal;
                hit.identity=uvec4(3,0,owner,input_sample.identity.z);
                hit.boundary=uvec4(input_sample.media.z,0,0,0);
                if(hit.boundary.x!=0xffffffffu) hit.boundary.yz=medium_boundaries[hit.boundary.x].xy;
            } else {
                hit=ordinarylightIntersect(origin,direction,0.0,pc.max_distance,pc.tolerance,pc.max_steps);
            }
            if(hit.boundary.w!=0u) { status|=hit.boundary.w; break; }
            if(hit.identity.x==0u) {
                if(depth!=1u) status|=8u; // An unmatched exit is not an environment sample.
                else radiance+=throughput*pc.environment.rgb;
                break;
            }
            throughput*=ordinarylightBeer(optical_media[medium_stack[depth-1u]].rgb,hit.position_distance.w);
            TransportMaterialRecord material=transport_materials[hit.identity.w];
            if(dot(direction,hit.geometric_normal.xyz)<0.0 || material.emission.a>0.5)
                radiance+=throughput*material.emission.rgb;
            bool dielectric=material.albedo_kind.w>0.5;
            if(!dielectric && max(material.albedo_kind.r,max(material.albedo_kind.g,material.albedo_kind.b))==0.0) break;
            if(bounce==pc.max_bounces) { truncated=true; break; }
            vec3 geometric=hit.geometric_normal.xyz;
            bool entering=dot(direction,geometric)<0.0;
            if(dielectric) {
                if(hit.boundary.x==0xffffffffu) { status|=2u; break; }
                uvec4 boundary=medium_boundaries[hit.boundary.x];
                uint target=entering?boundary.y:boundary.x;
                if(entering) {
                    if(medium_stack[depth-1u]!=boundary.x) { status|=2u; break; }
                    for(uint j=1u;j<depth;++j) if(boundary_stack[j]==hit.boundary.x) status|=2u;
                    if(status!=0u) break;
                } else {
                    if(depth<2u) { status|=2u; break; }
                    if(boundary_stack[depth-1u]!=hit.boundary.x || medium_stack[depth-1u]!=boundary.y ||
                        medium_stack[depth-2u]!=boundary.x) { status|=2u; break; }
                }
                OrdinaryLightDielectricEvent event=ordinarylightDielectric(direction,entering?geometric:-geometric,
                    optical_media[medium_stack[depth-1u]].a,optical_media[target].a,min(randomFloat(rng),0.99999994));
                direction=event.direction;
                throughput*=event.throughput;
                if(event.reflected) { events.y++; if(event.tir) events.w++; }
                else {
                    events.z++;
                    if(entering) {
                        if(depth==8u) { status|=4u; break; }
                        medium_stack[depth]=target; boundary_stack[depth]=hit.boundary.x; depth++;
                    } else depth--;
                }
            } else {
                vec3 normal=entering?hit.shading_normal.xyz:-hit.shading_normal.xyz;
                direction=cosineHemisphere(normal,randomFloat(rng),randomFloat(rng));
                if(dot(direction,entering?geometric:-geometric)<=0.0) break;
                throughput*=material.albedo_kind.rgb;
                events.x++;
            }
            // Account for the deliberate ray-start displacement as optical
            // distance in the new medium, avoiding thickness-dependent bias.
            throughput*=ordinarylightBeer(optical_media[medium_stack[depth-1u]].rgb,pc.ray_epsilon);
            origin=hit.position_distance.xyz+direction*pc.ray_epsilon;
            if(any(isnan(throughput))||any(isinf(throughput))||any(isnan(radiance))||any(isinf(radiance))) { status|=16u; break; }
            if(max(throughput.r,max(throughput.g,throughput.b))==0.0) break;
        }
        if(any(isnan(throughput))||any(isinf(throughput))||any(isnan(radiance))||any(isinf(radiance))) status|=16u;
        result.counts.x++;
        result.counts.z|=status;
        if(status==0u) { result.radiance.rgb+=radiance; result.counts.y++; }
        if(truncated) result.counts.w++;
        result.events+=events;
    }
    accumulated[owner]=result;
}
