# Branch isolation exposes a confounding factor

Each diagnostic adds `&& int(floor(material.ior_distance.z)) != 2` to exactly
one of the four custom-scattering conditions in generated primary source.
Scene material IDs remain 0/1, so intended visible behavior is unchanged.

| Condition changed | Completed frames | Result |
| --- | --- | --- |
| Primary scattering | 121 | Clean exit |
| Primary resampling | 4 | Device lost |
| Continuation scattering | 124 | Clean exit |
| Continuation resampling | 125 | Clean exit |
| Unchanged control repeat (600-frame target) | 4 | Device lost |

Crucially, both continuation variants compile to **exactly the same primary
SPIR-V as the original failing control**. Their secondary SPIR-V is identical
too. Thus a clean run does not, by itself, identify a causal shader change.
All captured binaries pass SPIR-V validation; hashes and outcomes are retained
in results.json. The continuation conditions are eliminated in this compiled
variant, so these runs are effectively controls at the captured-binary level.

This weakens the earlier inference that custom-scattering optimization alone
explains the failure. The primary scattering variant remains interesting, but
startup timing, camera trajectory, pipeline cache state, resource handling, or
other nondeterminism must be controlled before attributing the failure to it.
No specific alternative cause has been established.

Reproduce with PRIMARY_REDUCTION set to primary-scatter, primary-resample,
loop-scatter, or loop-resample using the existing harness. Omit it for the
control. Use SHADER_DUMP to verify binary identities. Other diagnostic mode
variables should be unset. The unchanged repeat used FRAME_LIMIT=600 but failed
at four frames. Logs and exact source diffs are retained here.

Next use a deterministic frame-indexed camera trajectory and repeated baseline
runs, recording initial poses and command-cache state. Check whether shader
compilation/startup timing shifts the trajectory or resource initialization.
The scene-specific ReSTIR-off workaround remains unchanged. No production
shader modification was made in this investigation.
