# Reservoir guard counter experiment

Dedicated counters were attached at primary binding 15 without enabling
wavefront_profiling (which changes production shader selection). Two host-visible
buffers were zeroed before dispatch. Production variants were compiled into
/tmp/restir-counter-spv and loaded through the temporary viewer harness.

Counting every access caused device loss after four completed frames. That
instrumentation adds high atomic contention and cannot establish the original
failure's cause. Reducing attempt sampling to reservoir_index % 4093 == 0
completed 124 frames; rejected-access counters remained unsampled/exact.

Readback after waiting for queued work:
- sampled load attempts: 0
- rejected loads: 0
- sampled store attempts: 73,750
- rejected stores: 0

This is inconclusive for the temporal load guard. Nonzero stores establish
that instrumentation executed, but zero sampled loads do not demonstrate that
the relevant history branch ran. Sparse sampling can also miss executed indices.
Do not conclude that no invalid temporal access exists or that a compiler bug
has been established.

Next instrument temporal-branch entry and successful reprojection, plus an
unsampled low-contention flag proving at least one history load executes.
Then inspect actual history eligibility and selected pipeline if necessary.

Temporary core allocation/binding changes and shared GLSL edits were restored.
Packaged shaders and the showcase workaround are unchanged. The shader patch
is retained solely as diagnostic source; it requires the dedicated counter
descriptor and must not be applied as a standalone production fix.
