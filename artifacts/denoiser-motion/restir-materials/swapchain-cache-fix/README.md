# Swapchain recreation retained stale recorded commands

The failing resource trace shows swapchain recreation after the first two
frames. Images and reservoir buffers are replaced, and history validity resets,
but both `wavefront_command_key` values remain populated. The replacement has
the same dimensions. Frame slot 0 then matches its original no-history render
key and reuses commands referencing destroyed resources. Device loss follows
at the fence wait after four completed frames.

The successful pre-fix trace did not recreate the swapchain during the observed
startup. This explains why identical shaders and camera poses could produce
different outcomes; earlier shader changes were not evidence of a compiler bug.

The fix clears both slots' command keys before swapchain resource destruction.
The next submission must re-record commands against the replacement resources,
even when dimensions and other render-key fields match. No shader changes or
material behavior changes are needed for this fix.

Validation with the original custom materials and ReSTIR enabled:

| Guides | Forced recreation indices | Completed frames | Result |
| --- | --- | --- | --- |
| Off | 2, 20, 40 | 607 | Clean exit |
| On | 2, 20, 40 | 604 | Clean exit |

The regression test checks that both slots are invalidated before the first
resource destruction. Focused config/viewer tests: 56 passed, eight subtests.
`git diff --check` passed.

The actual swapchain in these desktop runs was 1026x882 and the internal render
extent was 838x720. Earlier 1280x720 descriptions quoted the viewer's requested
extent/status, not the actual render extent. No claim of full 1280x720 rendering
is made for these runs.

Reproduce using the parent harness with DETERMINISTIC_CAMERA=1,
TRACE_RESOURCES=1, FORCE_RECREATE=1 and FRAME_LIMIT=600. Set MODE=mirror-on
for the guide-enabled run; otherwise leave MODE unset. ReSTIR is forced on by
the harness. This remains a finite reproduction check, not exhaustive Vulkan
validation. The scene default has subsequently been restored to ReSTIR enabled.

After restoring the showcase default, normal animated startup completed 602
frames and exited cleanly. This run used the actual default configuration,
without forcing ReSTIR on, a deterministic camera, or swapchain recreation.
The 56 focused tests and eight subtests passed again.
