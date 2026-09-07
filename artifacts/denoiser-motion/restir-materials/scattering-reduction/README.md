# Reduction to the custom-scattering flag

The diagnostic modifies only generated primary GLSL. Scene geometry, material
IDs 0/1, visible material behavior, and the secondary shader remain unchanged.

| Primary change | Frames | Result |
| --- | --- | --- |
| `if (program_id == 2) evaluated.custom_scattering = 0.0;` | 121 | Clean exit |
| Extra ID 2 dispatch to the existing custom diffuse evaluator | 4 | Device lost |

No visible material selects ID 2. The first variant removes the compile-time
certainty that every material evaluation has custom scattering. The second adds
a dispatch branch while preserving that certainty. This is stronger evidence
of sensitivity to optimization around the custom-scattering branch than the
previous full built-in-program experiment. It is not proof of driver miscompilation:
undefined behavior or timing-sensitive resource access remains possible.
Both modified primary binaries pass SPIR-V validation for Vulkan 1.2.

Reproduce with the existing harness using `PRIMARY_REDUCTION=runtime-flag` or
`PRIMARY_REDUCTION=duplicate-custom`, with `EXTRA_PROGRAM`, `MODE`, and `MATERIALS`
unset. `SHADER_DUMP` saves exact GLSL and SPIR-V. `FRAME_LIMIT` defaults to 120;
the harness also stops after 600 timer ticks. Use the X11 environment and
repository-root command documented in the parent investigation.

These are diagnostic mutations only. The renderer retains its scene-specific
ReSTIR-off workaround. A production change needs a causal explanation and
numerical validation, not merely a branch that prevents this reproduction.
Next separate the primary material resampling branch from its continuation
scattering branch, then inspect the resulting resource accesses and optimized
control flow for the failing path.

An independent runtime-flag repeat completed 602 animated frames with
a clean exit. Its primary SPIR-V was byte-identical to the first runtime-flag
run. This extends the successful observation but does not establish long-term
stability or image equivalence.
