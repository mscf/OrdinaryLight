# ReSTIR control-flow trace corrects the working hypothesis

An unsampled atomic bitmask traced the following events:
1: primary invocation; 2: direct-light/ReSTIR branch;
4: temporal branch; 8: successful reprojection; 16: history load;
32: rejected history load; 64: rejected current store;
128: history-valid push flag seen at primary entry.

The normal-paced animated run completed 124 frames. OR-reduced flags were
129: primary invocation and history-valid flag only. No direct-light branch,
temporal branch, reprojection or history load was observed. Host metadata:
production selection true, resolved strategy wavefront, custom primary
pipeline present, ReSTIR runtime enabled.

Source inspection explains this: compiler.py inserts custom scattering before
the built-in transmission/diffuse path. The fixture's materials use custom
diffuse/mirror responses, so built-in direct-light/ReSTIR work is bypassed.
Reservoir initialization stores still execute at primary entry, explaining
the prior store counters.

## Implication

The earlier configuration matrix remains valid, but interpreting it as proof
of a failing executed temporal reservoir-read loop was too strong. Those
configuration changes also alter dispatch counts, resident command keys,
pipeline creation/selection and compiled code. The instrumented history-load
guard was not exercised and its clean run does not establish a bounds fix.

Next isolate custom-material dispatch and resource/pipeline differences rather
than instrumenting a branch this fixture does not enter. Compare built-in
diffuse materials against custom diffuse responses while retaining the mirror,
and record the actually bound pipeline and command-cache history settings.

Temporary instrumentation and core descriptor allocation changes were restored.
Packaged shaders and the working scene-specific workaround remain unchanged.
The retained patch requires diagnostic binding support and is not a production
patch. No root-cause fix or driver/compiler diagnosis is claimed.
