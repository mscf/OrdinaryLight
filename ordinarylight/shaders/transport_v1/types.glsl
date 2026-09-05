#ifndef ORDINARYLIGHT_TRANSPORT_TYPES_V1
#define ORDINARYLIGHT_TRANSPORT_TYPES_V1 1
struct MaterialData {
    vec4 base_roughness;
    vec4 emission_metallic;
    vec4 attenuation_transmission;
    vec4 ior_distance;
    vec4 texture_indices;
    vec4 texture_parameters;
    vec4 advanced0;
    vec4 advanced1;
    vec4 sheen_color;
    vec4 subsurface_color;
    vec4 advanced_texture_indices;
    vec4 optical;
};
struct PointLightData {
    vec4 position_type;
    vec4 direction_range;
    vec4 color_intensity;
    vec4 spot_parameters;
};
struct AreaLightData {
    vec4 a; vec4 b; vec4 c; vec4 emission_area; vec4 distribution;
};
#endif
