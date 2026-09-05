#ifndef ORDINARYLIGHT_MATERIAL_CONTRACTS_V1
#define ORDINARYLIGHT_MATERIAL_CONTRACTS_V1
struct MaterialEvaluation
{
    vec3 base_color;
    vec3 emission;
    float metallic;
    float roughness;
    float transmission;
    float ior;
    vec3 attenuation_color;
    float attenuation_distance;
    float custom_scattering;
    vec3 weight;
    vec3 next_direction;
    float event;
    float pdf;
};
#endif
