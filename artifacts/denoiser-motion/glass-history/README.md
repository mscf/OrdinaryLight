# Stale previous geometry after glass motion

After 32 stopped frames the moved-glass patch still reported mean absolute
motion of 1.505 pixels/component, versus 0.251 in a fresh stationary render.
The stationary value includes ray-sample jitter. Resetting ReLAX history once
at the stop did not help: the same incorrect guides kept arriving.

prepare_window_scene bypassed upload_window_scene when the scene revision was
unchanged, refreshing only GPU volume sources. That skipped the resident fast
path which converges the previous vertex snapshot after a transform. The
preceding moving pose stayed in the previous-geometry buffer indefinitely.

The fix routes preparation through the existing resident-scene logic. It waits
for outstanding work and updates the previous snapshot once after motion stops;
subsequent stationary calls need no extra upload or wait. Borrowed-scene handling
remains in the existing upload method.

| Glass after 32 stopped frames | Before | After |
| --- | --- | --- |
| No diagnostic history reset: reference RMSE | 0.1305 | 0.0376 |
| Reset once at stop: reference RMSE | 0.1305 | 0.0175 |
| Fresh stationary reference RMSE | 0.0170 | 0.0170 |
| Mean absolute guide motion, no reset | 1.505 | 0.251 |
| Mean specular history length, no reset | 13.0 | 31.8 |

The motion mismatch is resolved. Some retained-history error remains without
the diagnostic reset. No automatic reset or rejection rule is promoted here.
The diagnostic reset clears history validity and command keys once at the first
stopped frame. Camera/target motion matrix metrics are unchanged after this fix;
glass moving-end metrics are unchanged, and the settled result improves.

The regression test exercises the presentation entry point and checks that
previous geometry converges exactly once on repeated stationary calls.
71 focused tests and ten subtests passed. Source whitespace checks passed.

Run capture.py from the repository root with PYTHONPATH=. after glass_detail
has generated its reference under /tmp/glass-detail. Captures go to
/tmp/glass-history. The before records were captured before this fix;
rerunning now produces fixed behavior. Raw guides remain in temporary output.
This is a correction to stationary geometry history, not full refraction-aware
reprojection or a validated solution for every moving refractive scene.
