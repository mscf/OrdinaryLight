# Deterministic camera and early presentation state

The diagnostic supplies a frame-indexed orbit directly to native presentation:
angle = index * 0.35 / 60, position = (-7 sin(angle), 2, -7 cos(angle)),
target = (0, 1.8, 0). No shader changes are applied. First twelve camera poses,
history dependency state, ReSTIR history validity and command-cache reuse are
logged. This controls camera inputs, not GPU scheduling or startup timing.

| Run | Frames | Result |
| --- | --- | --- |
| Deterministic A | 124 | Clean exit |
| Deterministic B, 600-frame target | 4 | Device lost |
| Deterministic C, skip-acquisition-aware counter | 4 | Device lost |
| Ordinary camera, trace enabled | 122 | Clean exit |

Captured primary and secondary binaries are identical across these runs.
The deterministic A/B submitted camera poses match through the failing prefix,
but the state first differs at index 2: A reports valid ReSTIR history, a
history dependency wait and a command-cache miss; B reports invalid history,
no dependency wait and a cache hit. This is an observation, not proof that
command caching caused the fault. The state logs do not yet include every
resource reset, render-extent transition, or adaptive-resolution decision.

The harness now explicitly logs acquisition skips and advances its diagnostic
index only when presentation returns an extent. This correction was applied
before run C. No skipped-acquire message was observed in C.

Reproduce with DETERMINISTIC_CAMERA=1 and SHADER_DUMP set to a temporary directory
using the parent harness. TRACE_CAMERA=1 instead records the ordinary camera.
Leave shader mutation and material-selection variables unset. FRAME_LIMIT=600
extends the requested observation window but does not override device loss.

Next trace render extents, adaptive resolution, resource resets and command-key
transitions around the first reuse of frame slot 0. Shader optimization alone
and wall-clock camera differences are not sufficient explanations for the
observed outcomes. The existing scene-specific ReSTIR-off workaround remains.
