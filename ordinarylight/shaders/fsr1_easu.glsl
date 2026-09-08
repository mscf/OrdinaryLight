// FSR 1 EASU adapter. Upstream algorithm is unmodified; RCAS is disabled.
#define A_GPU 1
#define A_GLSL 1
#include "third_party/fsr1/ffx_a.h"
#define FSR_EASU_F 1
#include "third_party/fsr1/ffx_fsr1.h"

vec3 fsr1Source(ivec2 pixel) {
    return clamp(linearToSrgb(acesApproximation(
        max(reconstructLoadHdr(pixel), vec3(0.0)) * push.exposure)), 0.0, 1.0);
}
vec4 fsr1Gather(vec2 p, int channel) {
    // GLSL gather order: bottom-left, bottom-right, top-right, top-left.
    ivec2 base = ivec2(floor(p * vec2(push.source_width, push.source_height) - 0.5));
    return vec4(fsr1Source(base + ivec2(0, 1))[channel],
                fsr1Source(base + ivec2(1, 1))[channel],
                fsr1Source(base + ivec2(1, 0))[channel],
                fsr1Source(base)[channel]);
}
vec4 FsrEasuRF(vec2 p) { return fsr1Gather(p, 0); }
vec4 FsrEasuGF(vec2 p) { return fsr1Gather(p, 1); }
vec4 FsrEasuBF(vec2 p) { return fsr1Gather(p, 2); }
vec3 reconstructFsr1(ivec2 pixel, ivec2 size) {
    uvec4 con0, con1, con2, con3;
    FsrEasuCon(con0, con1, con2, con3,
        float(push.source_width), float(push.source_height),
        float(push.source_width), float(push.source_height), float(size.x), float(size.y));
    vec3 result;
    FsrEasuF(result, uvec2(pixel), con0, con1, con2, con3);
    return clamp(result, 0.0, 1.0);
}
