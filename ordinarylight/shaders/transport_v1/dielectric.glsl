#ifndef ORDINARYLIGHT_DIELECTRIC_V1
#define ORDINARYLIGHT_DIELECTRIC_V1 1
struct OrdinaryLightDielectricEvent {
    vec3 direction;
    float throughput;
    bool reflected;
    bool tir;
    float fresnel;
};
vec3 ordinarylightBeer(vec3 absorption, float distance) {
    return exp(-absorption*distance);
}
OrdinaryLightDielectricEvent ordinarylightDielectric(
    vec3 direction, vec3 incident_normal, float eta_i, float eta_t, float selector)
{
    OrdinaryLightDielectricEvent event;
    float ci=clamp(-dot(direction,incident_normal),0.0,1.0);
    float eta=eta_i/eta_t;
    float st2=eta*eta*max(0.0,1.0-ci*ci);
    event.tir=st2>=1.0;
    float ct=sqrt(max(0.0,1.0-st2));
    float rs=(eta_i*ci-eta_t*ct)/max(eta_i*ci+eta_t*ct,1e-20);
    float rp=(eta_t*ci-eta_i*ct)/max(eta_t*ci+eta_i*ct,1e-20);
    event.fresnel=event.tir?1.0:(eta_i==eta_t?0.0:0.5*(rs*rs+rp*rp));
    event.reflected=event.tir || selector<event.fresnel;
    event.direction=normalize(event.reflected?reflect(direction,incident_normal)
        :eta*direction+(eta*ci-ct)*incident_normal);
    event.throughput=event.reflected?1.0:eta*eta;
    return event;
}
#endif
