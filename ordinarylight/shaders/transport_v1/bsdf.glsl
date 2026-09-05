// Isotropic GGX evaluation and matched NDF sampling. Directions point away
// from the surface. Rejected microfacets are null samples, never resampled.
#ifndef ORDINARYLIGHT_BSDF_V1
#define ORDINARYLIGHT_BSDF_V1
const float OL_PI=3.141592653589793;
float olDistribution(float cosine,float roughness) {
    if(cosine<=0.0) return 0.0;
    float a=max(roughness*roughness,1e-4), a2=a*a;
    float q=(1.0-cosine*cosine)+a2*cosine*cosine;
    return a2/(OL_PI*q*q);
}
float olMask(float cosine,float roughness) {
    cosine=abs(cosine); if(cosine==0.0) return 0.0;
    float a=max(roughness*roughness,1e-4);
    return 2.0*cosine/(cosine+sqrt(a*a+(1.0-a*a)*cosine*cosine));
}
vec3 olMicroNormal(vec3 normal,float roughness,vec2 randoms) {
    float a=max(roughness*roughness,1e-4);
    float u=min(randoms.x,0.99999994);
    float cosine=sqrt((1.0-u)/(1.0-u+a*a*u));
    float sine=sqrt(max(0.0,1.0-cosine*cosine));
    vec3 t=normalize(cross(normal,abs(normal.z)<0.99?vec3(0,0,1):vec3(0,1,0)));
    return normalize(normal*cosine+sine*(t*cos(2.0*OL_PI*randoms.y)+cross(normal,t)*sin(2.0*OL_PI*randoms.y)));
}
// Returns BSDF RGB and PDF per unit solid angle. kind: 0 diffuse,1 glass,2 PBR,3 emission.
vec4 olBsdf(MaterialEvaluation material,uint kind,vec3 normal,vec3 view,vec3 outgoing,float eta_i,float eta_t) {
    float cv=dot(normal,view), co=dot(normal,outgoing);
    if(cv<=0.0 || co==0.0 || kind==3u) return vec4(0);
    if(kind==0u) return co>0.0?vec4(material.base_color/OL_PI,co/OL_PI):vec4(0);
    if(material.roughness==0.0) {
        if(kind!=2u || co<=0.0 || material.metallic==1.0) return vec4(0);
        vec3 f0=mix(vec3(pow((material.ior-1.0)/(material.ior+1.0),2.0)),material.base_color,material.metallic);
        vec3 fresnel=f0+(vec3(1)-f0)*pow(1.0-cv,5.0);
        vec3 exit_fresnel=f0+(vec3(1)-f0)*pow(1.0-co,5.0);
        return vec4((vec3(1)-fresnel)*(vec3(1)-exit_fresnel)*(1.0-material.metallic)*material.base_color/OL_PI,0.5*co/OL_PI);
    }
    bool reflected=co>0.0;
    if(kind!=1u && !reflected) return vec4(0);
    float eta=eta_t/eta_i;
    vec3 sum=view+outgoing*(reflected?1.0:eta);
    if(dot(sum,sum)<1e-20) return vec4(0);
    vec3 half_vector=normalize(sum);
    if(dot(normal,half_vector)<0.0) half_vector=-half_vector;
    float vh=dot(view,half_vector), oh=dot(outgoing,half_vector), nh=dot(normal,half_vector);
    if(vh<=0.0 || oh*co<=0.0) return vec4(0);
    float d=olDistribution(nh,material.roughness);
    float g=olMask(cv,material.roughness)*olMask(co,material.roughness);
    float normal_pdf=d*nh;
    if(kind==1u) {
        float fresnel=ordinarylightDielectric(-view,half_vector,eta_i,eta_t,0.0).fresnel;
        if(reflected) return vec4(vec3(fresnel*d*g/(4.0*cv*co)),normal_pdf*fresnel/(4.0*vh));
        float denominator=oh+vh/eta;
        denominator*=denominator;
        if(denominator<1e-30) return vec4(0);
        float pdf=normal_pdf*(1.0-fresnel)*abs(oh)/denominator;
        float value=(1.0-fresnel)*d*g*abs(vh*oh/(cv*co*denominator))/(eta*eta);
        return vec4(vec3(value),pdf);
    }
    vec3 f0=mix(vec3(pow((material.ior-1.0)/(material.ior+1.0),2.0)),material.base_color,material.metallic);
    vec3 fresnel=f0+(vec3(1)-f0)*pow(1.0-vh,5.0);
    vec3 value=fresnel*d*g/(4.0*cv*co)+(vec3(1)-fresnel)*(1.0-material.metallic)*material.base_color/OL_PI;
    float probability=material.metallic==1.0?1.0:0.5;
    return vec4(value,mix(co/OL_PI,normal_pdf/(4.0*vh),probability));
}
vec3 olSampleBsdf(MaterialEvaluation material,uint kind,vec3 normal,vec3 view,
    float eta_i,float eta_t,vec3 randoms,out vec3 weight,out float pdf,out bool reflected,out bool tir) {
    weight=vec3(0); pdf=0.0; reflected=true; tir=false;
    vec3 outgoing;
    if(kind==0u) {
        outgoing=cosineHemisphere(normal,randoms.x,randoms.y);
    } else if(kind==1u) {
        if(eta_i==eta_t) { weight=vec3(1); pdf=1.0; reflected=false; return -view; }
        vec3 micro=material.roughness==0.0?normal:olMicroNormal(normal,material.roughness,randoms.xy);
        if(dot(view,micro)<=0.0) return normal;
        OrdinaryLightDielectricEvent event=ordinarylightDielectric(-view,micro,eta_i,eta_t,randoms.z);
        outgoing=event.direction; reflected=event.reflected; tir=event.tir;
        if((dot(outgoing,normal)>0.0)!=reflected) return outgoing;
        if(material.roughness==0.0 || eta_i==eta_t) {
            weight=vec3(event.throughput); pdf=1.0; return outgoing;
        }
    } else if(kind==2u) {
        float probability=material.metallic==1.0?1.0:0.5;
        if(material.roughness==0.0) {
            vec3 f0=mix(vec3(pow((material.ior-1.0)/(material.ior+1.0),2.0)),material.base_color,material.metallic);
            vec3 fresnel=f0+(vec3(1)-f0)*pow(1.0-dot(normal,view),5.0);
            if(randoms.z<probability) {
                weight=fresnel/probability; pdf=1.0;
                return reflect(-view,normal);
            }
            outgoing=cosineHemisphere(normal,randoms.x,randoms.y);
            vec3 exit_fresnel=f0+(vec3(1)-f0)*pow(1.0-dot(normal,outgoing),5.0);
            weight=(vec3(1)-fresnel)*(vec3(1)-exit_fresnel)*(1.0-material.metallic)*material.base_color/(1.0-probability);
            pdf=(1.0-probability)*max(dot(normal,outgoing),0.0)/OL_PI;
            return outgoing;
        }
        outgoing=randoms.z<probability?reflect(-view,olMicroNormal(normal,material.roughness,randoms.xy)):
            cosineHemisphere(normal,randoms.x,randoms.y);
    } else return normal;
    vec4 value=olBsdf(material,kind,normal,view,outgoing,eta_i,eta_t);
    pdf=value.a;
    if(pdf>0.0) weight=value.rgb*abs(dot(normal,outgoing))/pdf;
    return outgoing;
}
#endif
