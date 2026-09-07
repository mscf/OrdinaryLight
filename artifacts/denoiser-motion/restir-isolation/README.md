# ReSTIR startup isolation

Normal animated Qt viewer, planar-mirror-guides scene, 1280x720, direct
presentation, no diagnostic pacing. The mirror-guide option remains off.
The diagnostic deliberately overrides the scene's ReSTIR-off workaround.

| Configuration | Observed result |
| --- | --- |
| Production specialization, temporal history, four streams | Previously reproduced device loss after 3–4 completed frames |
| Same, force restir_history_valid_override=False in executor dispatch | 123 frames, clean exit |
| Same, wavefront_restir_specialization=False | 120 frames, clean exit |
| Same, wavefront_untextured_specialization=False only | Device loss after 4 frames |
| Same, wavefront_restir_reservoirs=1 | 131 frames, clean exit |

These isolate a failing combination: specialized temporal ReSTIR with multiple
streams. They do not prove all stream counts greater than one fail, or identify
an offending shader instruction. A clean bounded run is not a long-duration
stability guarantee.

The generic-path toggle also affects pipeline selection. The unsuccessful
untextured-specialization ablation narrows that confound but does not establish
that a specific preprocessor constant alone causes the fault.

Inspection targets:
- wavefront_primary_impl.glsl: restir_reservoir_count/restir_stream indexing
  and previous reservoir loads in the temporal loop.
- VulkanWavefrontExecutor: production shader selection and per-stream dispatch.
- Production macros constant-fold optional estimator branches while preserving
  the push-constant ABI. A driver/compiler issue is possible but not established.

No Khronos validation layer is installed in this environment (only Intel null
hardware and Mesa overlay layers were present), so no validation-layer-clean
claim is made. GPU faults are still reported as Xid 109 context-switch timeouts.

The showcase workaround remains unchanged. Next compare specialized and generic
temporal loop code with explicit bounds instrumentation, and test two streams
before narrowing further. Do not globally disable ReSTIR on this evidence.

Follow-up correction: [control-flow tracing](../restir-flow/README.md) found
the custom-material path bypasses the built-in ReSTIR temporal loads in the
instrumented scene. The matrix identifies configuration correlations, not proof
that the executed temporal load loop causes the fault.
