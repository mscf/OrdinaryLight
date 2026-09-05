// One invocation per input slot. A separate deterministic pass reduces slots
// into output identities, so duplicate destinations never race.
struct SampleAccumulation { vec4 radiance; uvec4 counts; uvec4 events; };
layout(set=0,binding=8,std430) readonly buffer Samples { OrdinaryLightSurfaceSample transport_samples[]; };
layout(set=0,binding=9,std430) buffer Accumulation { SampleAccumulation accumulated[]; };
layout(set=0,binding=10,std430) readonly buffer InitialStack { uvec4 initial_stack[]; };
layout(push_constant) uniform Constants {
    uint count; uint samples_per_element; uint max_bounces; uint sample_offset;
    uint seed; uint initial_depth; float tolerance; float ray_epsilon;
    uint max_steps; float max_distance; uint initial_stack_count; uint environment_nee;
    vec4 environment;
} pc;

void main() {
    uint i=gl_GlobalInvocationID.x; if(i>=pc.count) return;
    OrdinaryLightSurfaceSample input_sample=transport_samples[i];
    uint owner=input_sample.identity.x;
    SampleAccumulation result=SampleAccumulation(vec4(0),uvec4(0),uvec4(0));
    bool invalid=any(isnan(input_sample.position))||any(isinf(input_sample.position))||
        any(isnan(input_sample.incoming))||any(isinf(input_sample.incoming))||
        abs(dot(input_sample.incoming.xyz,input_sample.incoming.xyz)-1.0)>0.001||
        input_sample.identity.w>1u || input_sample.media.w>=pc.initial_stack_count;
    if(input_sample.identity.w==1u) {
        vec3 gn=input_sample.geometric_normal.xyz, sn=input_sample.shading_normal.xyz;
        invalid=invalid||isnan(input_sample.geometric_normal.w)||isinf(input_sample.geometric_normal.w)||
            isnan(input_sample.shading_normal.w)||isinf(input_sample.shading_normal.w)||any(isnan(gn))||any(isinf(gn))||any(isnan(sn))||any(isinf(sn))||
            abs(dot(gn,gn)-1.0)>0.001||abs(dot(sn,sn)-1.0)>0.001||dot(gn,sn)<=0.0||
            input_sample.identity.z>=OL_MATERIAL_COUNT;
        uint boundary_id=input_sample.media.z, boundary_index=0xffffffffu;
        if(boundary_id!=0xffffffffu) {
            for(uint j=0u;j<OL_BOUNDARY_COUNT;++j)
                if(medium_boundaries[j].z==boundary_id) { boundary_index=j; break; }
            if(boundary_index==0xffffffffu) invalid=true;
        }
        input_sample.media.z=boundary_index;
        if(!invalid) {
            bool dielectric=transport_materials[input_sample.identity.z].albedo_kind.w==1.0;
            if(dielectric!=(boundary_index!=0xffffffffu)) invalid=true;
        }
    }
    if(invalid) {
        result.counts=uvec4(pc.samples_per_element,0,32,0);
        accumulated[i]=result; return;
    }
    for(uint sample_index=0u;sample_index<pc.samples_per_element;++sample_index) {
        uint rng=secondaryNeeHash(owner^secondaryNeeHash(pc.seed)^secondaryNeeHash(
            pc.sample_offset+sample_index)^secondaryNeeHash(input_sample.identity.y)^secondaryNeeHash(i));
        uint medium_stack[8]; uint boundary_stack[8];
        uint stack_offset=input_sample.media.w*8u;
        uint depth=initial_stack[stack_offset].z;
        for(uint j=0u;j<depth;++j) {
            medium_stack[j]=initial_stack[stack_offset+j].x;
            boundary_stack[j]=initial_stack[stack_offset+j].y;
        }
        vec3 origin=input_sample.position.xyz;
        vec3 direction=input_sample.incoming.xyz;
        vec3 throughput=vec3(1); vec3 radiance=vec3(0);
        uint status=0u; bool truncated=false;
        float previous_pdf=0.0; bool previous_delta=true;
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
                else radiance+=throughput*pc.environment.rgb*(pc.environment_nee!=0u && !previous_delta?powerHeuristic(previous_pdf,1.0/(4.0*OL_PI)):1.0);
                break;
            }
            throughput*=ordinarylightBeer(optical_media[medium_stack[depth-1u]].rgb,hit.position_distance.w);
            TransportMaterialRecord material=transport_materials[hit.identity.w];
            MaterialEvaluation evaluated=ordinarylightEvaluateMaterial(hit.identity.w,hit,direction,float(bounce),
                optical_media[medium_stack[depth-1u]].a,depth>1u?optical_media[medium_stack[depth-2u]].a:1.0,vec2(randomFloat(rng),randomFloat(rng)));
            if(isnan(evaluated.metallic) || isinf(evaluated.metallic) || isnan(evaluated.roughness) || isinf(evaluated.roughness) ||
                isnan(evaluated.ior) || isinf(evaluated.ior) || evaluated.ior<=0.0 ||
                (material.albedo_kind.w==0.0 && (evaluated.metallic!=0.0 || evaluated.roughness!=0.0)) ||
                (material.albedo_kind.w==1.0 && (evaluated.metallic!=0.0 || evaluated.ior!=optical_media[hit.boundary.z].a)) ||
                any(notEqual(evaluated.attenuation_color,vec3(1))) || evaluated.attenuation_distance!=1e30 ||
                evaluated.transmission!=float(material.albedo_kind.w==1.0) ||
                any(isnan(evaluated.base_color)) || any(isinf(evaluated.base_color)) ||
                any(isnan(evaluated.emission)) || any(isinf(evaluated.emission))) { status|=16u; break; }
            evaluated.metallic=clamp(evaluated.metallic,0.0,1.0);
            evaluated.roughness=clamp(evaluated.roughness,0.0,1.0);
            evaluated.base_color=clamp(evaluated.base_color,vec3(0),vec3(1));
            material.albedo_kind.rgb=clamp(evaluated.base_color,vec3(0),vec3(1));
            material.emission.rgb=max(evaluated.emission,vec3(0));
            if(dot(direction,hit.geometric_normal.xyz)<0.0 || material.emission.a>0.5)
                radiance+=throughput*material.emission.rgb;
            bool dielectric=material.albedo_kind.w==1.0;
            if(material.albedo_kind.w==3.0) break;
            if(material.albedo_kind.w==0.0 && max(material.albedo_kind.r,max(material.albedo_kind.g,material.albedo_kind.b))==0.0) break;
            if(bounce==pc.max_bounces) { truncated=true; break; }
            vec3 geometric=hit.geometric_normal.xyz;
            bool entering=dot(direction,geometric)<0.0;
            vec3 scattering_normal=entering?hit.shading_normal.xyz:-hit.shading_normal.xyz;
            if(dielectric) scattering_normal=entering?geometric:-geometric;
            float eta_i=optical_media[medium_stack[depth-1u]].a, eta_t=eta_i;
            uint target_medium=medium_stack[depth-1u];
            if(dielectric && hit.boundary.x!=0xffffffffu) {
                target_medium=entering?medium_boundaries[hit.boundary.x].y:medium_boundaries[hit.boundary.x].x;
                eta_t=optical_media[target_medium].a;
            }
            for(uint light_index=0u;light_index<OL_ANALYTIC_LIGHT_COUNT;++light_index) {
                vec4 p=analytic_lights[light_index*4u], d=analytic_lights[light_index*4u+1u];
                vec4 color=analytic_lights[light_index*4u+2u], cone=analytic_lights[light_index*4u+3u];
                bool directional=p.w==1.0;
                vec3 offset=p.xyz-hit.position_distance.xyz;
                float distance=directional?pc.max_distance:length(offset);
                if(distance<=2.0*pc.ray_epsilon || (!directional && d.w>0.0 && distance>d.w)) continue;
                vec3 outgoing=directional?-d.xyz:offset/distance;
                float attenuation=directional?1.0:1.0/(distance*distance);
                if(p.w==2.0) attenuation*=cone.x==cone.y?float(dot(d.xyz,-outgoing)>=cone.x):smoothstep(cone.y,cone.x,dot(d.xyz,-outgoing));
                if(dot(outgoing,scattering_normal)*dot(outgoing,entering?geometric:-geometric)<=0.0) continue;
                vec4 bsdf=olBsdf(evaluated,uint(material.albedo_kind.w),scattering_normal,-direction,outgoing,eta_i,eta_t);
                if(max(bsdf.r,max(bsdf.g,bsdf.b))<=0.0) continue;
                uint shadow_medium=dot(outgoing,scattering_normal)>0.0?medium_stack[depth-1u]:target_medium;
                if(directional && shadow_medium!=0u) continue;
                OrdinaryLightHit blocker=ordinarylightIntersect(hit.position_distance.xyz+outgoing*pc.ray_epsilon,
                    outgoing,0.0,distance-2.0*pc.ray_epsilon,pc.tolerance,pc.max_steps);
                status|=blocker.boundary.w;
                if(blocker.identity.x==0u && blocker.boundary.w==0u)
                    radiance+=throughput*bsdf.rgb*abs(dot(scattering_normal,outgoing))*color.rgb*color.a*attenuation*
                        ordinarylightBeer(optical_media[shadow_medium].rgb,directional?0.0:distance);
            }
            if(pc.environment_nee!=0u && max(pc.environment.r,max(pc.environment.g,pc.environment.b))>0.0) {
                float z=1.0-2.0*randomFloat(rng), phi=2.0*OL_PI*randomFloat(rng);
                float radius=sqrt(max(0.0,1.0-z*z));
                vec3 outgoing=vec3(radius*cos(phi),radius*sin(phi),z);
                vec4 bsdf=olBsdf(evaluated,uint(material.albedo_kind.w),scattering_normal,-direction,outgoing,eta_i,eta_t);
                uint shadow_medium=dot(outgoing,scattering_normal)>0.0?medium_stack[depth-1u]:target_medium;
                if(bsdf.a>0.0 && shadow_medium==0u && dot(outgoing,scattering_normal)*dot(outgoing,entering?geometric:-geometric)>0.0) {
                    OrdinaryLightHit blocker=ordinarylightIntersect(hit.position_distance.xyz+outgoing*pc.ray_epsilon,
                        outgoing,0.0,pc.max_distance,pc.tolerance,pc.max_steps);
                    status|=blocker.boundary.w;
                    if(blocker.identity.x==0u && blocker.boundary.w==0u)
                        radiance+=throughput*bsdf.rgb*abs(dot(scattering_normal,outgoing))*pc.environment.rgb*
                            (4.0*OL_PI)*powerHeuristic(1.0/(4.0*OL_PI),bsdf.a);
                }
            }
            previous_delta=dielectric?(evaluated.roughness==0.0 || eta_i==eta_t):false;
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
                vec3 weight; float pdf; bool reflected, tir;
                direction=olSampleBsdf(evaluated,1u,entering?geometric:-geometric,-direction,
                    optical_media[medium_stack[depth-1u]].a,optical_media[target].a,
                    vec3(randomFloat(rng),randomFloat(rng),min(randomFloat(rng),0.99999994)),weight,pdf,reflected,tir);
                if(pdf==0.0) break;
                throughput*=weight;
                previous_pdf=pdf;
                if(reflected) { events.y++; if(tir) events.w++; }
                else {
                    events.z++;
                    if(entering) {
                        if(depth==8u) { status|=4u; break; }
                        medium_stack[depth]=target; boundary_stack[depth]=hit.boundary.x; depth++;
                    } else depth--;
                }
            } else {
                vec3 normal=entering?hit.shading_normal.xyz:-hit.shading_normal.xyz;
                vec3 weight; float pdf; bool reflected, tir;
                vec3 mirror=reflect(direction,normal);
                direction=olSampleBsdf(evaluated,uint(material.albedo_kind.w),normal,-direction,1.0,1.0,
                    vec3(randomFloat(rng),randomFloat(rng),randomFloat(rng)),weight,pdf,reflected,tir);
                if(pdf==0.0 || dot(direction,entering?geometric:-geometric)<=0.0) break;
                throughput*=weight;
                previous_pdf=pdf;
                previous_delta=material.albedo_kind.w==2.0 && evaluated.roughness==0.0 && dot(direction,mirror)>0.999999;
                events.x++;
            }
            // Account for the deliberate ray-start displacement as optical
            // distance in the new medium, avoiding thickness-dependent bias.
            throughput*=ordinarylightBeer(optical_media[medium_stack[depth-1u]].rgb,pc.ray_epsilon);
            origin=hit.position_distance.xyz+direction*pc.ray_epsilon;
            if(dielectric && evaluated.roughness>0.0) {
                // A direction-only offset can jump across a short grazing chord
                // and lose the matching exit. Preserve the classified side.
                origin=hit.position_distance.xyz+geometric*(dot(direction,geometric)>0.0?pc.ray_epsilon:-pc.ray_epsilon);
            }
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
    accumulated[i]=result;
}
