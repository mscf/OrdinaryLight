# Incoming-path emission MIS correction

The typed staged shade shader passed `push.unified_secondary_nee` to emissive
hit weighting. That describes the sampling mode of the next continuation,
not the mode that generated the incoming ray. The primary bounce can use a
different mode, notably the viewer's separate primary / unified secondary
configuration. The inline shader and handwritten staged fallback already
consulted PATH_PREVIOUS_UNIFIED_NEE_BIT on the incoming path.

The typed source now passes that incoming-path bit too. Generated GLSL and
all ordinaryshade staged SPIR-V variants were rebuilt. This is a transport
correctness fix, not a sampling reduction or a performance optimization.
It intentionally corrects affected existing staged results.

A two-bounce raw-HDR comparison (denoising off) is now byte-identical between
standard and the diagnostic custom-inline path. Before the fix its maximum
absolute difference was .12598 and RMSE .00213. This isolates the first
continuation independently of temporal filtering.

48-frame denoised sequences after correction:

| Motion | HDR RMSE | Max absolute | Pixel-frames with RGB difference >.01 |
| --- | ---: | ---: | ---: |
| Glass | .00001791 | .01367 | 11 |
| Background | .00020472 | .51270 | 9 |
| Camera | .000001062 | .0009766 | 0 |

Each sequence contains 3,686,400 pixel-frames. Most of the earlier discrepancy
is explained, but the isolated multi-bounce outliers remain unresolved.
Do not call the prototype fully equivalent or enable it by default on this
evidence. In particular the .51 outlier is not dismissed as harmless merely
because average error is low. The next check is to isolate later-bounce
hit/transport divergence at those pixels. Concurrent external GPU work makes
new performance claims inappropriate in this session.

72 focused shader/compiler/denoising tests passed. The diagnostic used the
existing custom-inline shim; it remains outside production dispatch policy.

Reproduce the raw two-bounce check from repository root with native Vulkan:

```sh
BOUNCES=2 OUT=/tmp/mis-standard PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/incoming-emission-mis/check_depth.py
INLINE=1 BOUNCES=2 OUT=/tmp/mis-inline PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/incoming-emission-mis/check_depth.py
```

Compare the resulting glass-result.npy arrays. Full corrected motion arrays
from this run remain in /tmp/inline-standard-corrected and
/tmp/inline-prototype-corrected. The custom-inline directory's check.py can
regenerate those full sequences with current source.

Follow-up: [handoff replay](../inline-handoff/README.md) isolated the large
remaining outlier to sub-microunit ray geometry differences and reproduced
the standard result exactly by substituting its handoff ray. No hard-coded
replay or precision workaround was added to production.
