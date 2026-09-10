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

// Per-invocation primary BRDF scratch. Only production ReSTIR enables this:
// generalized proposal evaluations can use a different normal/view, so their
// path must remain uncached. Primary disables the cache before continuation.
#if !defined(WAVE_PREPARED_PRIMARY_PBR)
#define WAVE_PREPARED_PRIMARY_PBR 1
#endif
bool transportPbrPrepared = false;
vec3 transportPbrF0;
vec4 transportPbrLobes;
float transportPbrProbability;
float transportPbrViewCosine;
vec3 transportPbrTangent;
vec3 transportPbrBitangent;
float transportPbrGeometry;
float transportPbrCoatGeometry;
@preparePrimaryPbr@

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
