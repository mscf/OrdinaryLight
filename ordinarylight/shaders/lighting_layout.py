"""Mechanical resource ABI and variant guards. Algorithms are typed helpers."""
LAYOUT = r'''// OrdinaryLight transport ABI v1. See docs/renderer_extensions.md.
#if !defined(ORDINARYLIGHT_TRANSPORT_LIGHTING_V1)
#define ORDINARYLIGHT_TRANSPORT_LIGHTING_V1 1
#if !defined(OL_TRANSPORT_RAY_ORIGIN)
#define OL_TRANSPORT_RAY_ORIGIN ordinarylightDefaultRayOrigin
@ordinarylightDefaultRayOrigin@
#endif
#if !defined(OL_TRANSPORT_AREA_LIGHT_COUNT)
#error Define OL_TRANSPORT_AREA_LIGHT_COUNT before including transport lighting v1
#endif
#if !defined(OL_TRANSPORT_AREA_LIGHT_WEIGHT)
#error Define OL_TRANSPORT_AREA_LIGHT_WEIGHT before including transport lighting v1
#endif
#if !defined(OL_TRANSPORT_ENVIRONMENT_SAMPLES)
#error Define OL_TRANSPORT_ENVIRONMENT_SAMPLES before including transport lighting v1
#endif
#if !defined(OL_TRANSPORT_POINT_LIGHT_COUNT)
#error Define OL_TRANSPORT_POINT_LIGHT_COUNT before including transport lighting v1
#endif
#if !defined(OL_TRANSPORT_SECONDARY_AREA_LIGHT_SAMPLES)
#error Define OL_TRANSPORT_SECONDARY_AREA_LIGHT_SAMPLES before including transport lighting v1
#endif
#include "wavefront_restir.glsl"

#include "transport_v1/sampling.glsl"

@pbrSpecularProbability@

// Per-invocation scratch for the most recent BSDF evaluation. Callers
// consume it only for accepted contributions, never candidate target sums.
vec3 transportLastSpecularFraction = vec3(0.0);
vec3 transportPointSpecular = vec3(0.0);

@evaluatePbr@

@pbrPdf@

@sampleGgxHalfVector@

@samplePbr@

@samplePointLights@

@sampleAreaLightTechnique@

@sampleAreaLight@

struct AreaLightCandidate {
    uint light_index;
    vec2 barycentrics;
    float target;
};

const uint ENVIRONMENT_LIGHT_CANDIDATE_INDEX = 0x01fffffeu;

@unifiedAreaDomainProbability@

@evaluateAreaLightCandidateTechnique@

@evaluateAreaLightCandidate@

@generateAreaLightCandidate@

@areaLightCandidateVisibility@

@environmentColor@

@sampleEnvironmentTechnique@

@sampleEnvironment@

@sampleUnifiedSecondaryLight@

@encodeEnvironmentCandidateDirection@

@decodeEnvironmentCandidateDirection@

@evaluateEnvironmentCandidate@

@generateUnifiedPrimaryCandidate@

@evaluateUnifiedPrimaryCandidate@

#endif
'''
