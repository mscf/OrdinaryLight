# Canonical motion audit and corrected mirror replay

## Root cause and fix

General output motion is current projected sample minus previous projected
sample. Canonical DenoiserSignals specifies the opposite direction, measured
from the integer current pixel center. Capture previously reused general
motion unchanged. Replay was correctly adding motion to the current pixel.

Canonical capture now subtracts the general displacement from the current
projected sample offset relative to the pixel center. This handles both sign
and stochastic sample offset. General output motion and native GPU denoiser
preparation are unchanged. Reflection still flips image X and negates the X
motion component; this coordinate transformation remains valid.

Regression tests cover stationary subpixel samples, camera translation, and
object translation. GPU captured motion agrees with independent previous-camera
projection in the rendered sloped fixture to within 0.0002 pixels at all three
resolutions. That diagnostic now uses captured motion and asserts agreement,
rather than replacing it with reconstructed motion.

## Corrected mirror results

Both ten-frame sequences were recaptured with 128-sample references and all
nine ablation variants rerun. Reference/radiance setup and policies are unchanged.

| Motion | Policy | Finite-hit history acceptance | Finite-hit log-RGB RMSE |
| --- | --- | ---: | ---: |
| Camera | Primary guides | 93.6% | 0.03527 |
| Camera | Reflected, original depth | 47.9% | 0.06176 |
| Camera | Reflected, blanket 2% | 81.6% | 0.04671 |
| Camera | Reflected, adaptive | 80.5% | 0.04722 |
| Object | Primary guides | 84.9% | 0.03795 |
| Object | Reflected, original depth | 33.0% | 0.06169 |
| Object | Reflected, blanket 2% | 71.5% | 0.04908 |
| Object | Reflected, adaptive | 70.5% | 0.04948 |

Acceptance excludes the initial frame; error includes all frames. Adaptive
error is 23.5% lower for camera motion and 19.8% lower for object motion than
original reflected guides. Primary guides still have lower error with stronger
smoothing. The occlusion false-acceptance findings remain a reason not to
promote adaptive depth.

[Camera animation](camera-motion.apng) and
[object animation](object-motion.apng) replace the earlier motion comparisons.
Playback is deliberately slowed, with a final hold before the loop resets.

All 18 mirror replays completed; unmodified primary/reflected replay outputs
exactly matched the new capture baselines. Nine rendered sloped replays passed
the independent motion assertion. CPU validation: 42 tests passed, including
general-output tests; lint and diff checks passed.

## Historical results

Previous offline captures used incorrect canonical motion; their quality
numbers are historical, not current evidence. Mirror results above supersede
the prior mirror studies. The broader-filter study uses live native preparation and is unaffected.
Spatial ablations replay only the spatial stage from live temporal outputs,
so they are also unaffected. The early `run.py` offline motion studies require
recapture before further quality/default decisions.

No native shader or viewer integration changed. Next: use these corrected inputs
to evaluate geometric history validation beyond flat parallel patches.


## Study execution-path audit

| Study | Input / execution path | Capture-motion fix impact |
| --- | --- | --- |
| Early moderate/fast offline comparisons | `run.py`: canonical capture, then temporal replay | Affected; historical only pending rerun |
| Planar mirror and depth ablations | `planar_mirror.py`: canonical capture, then temporal replay | Affected; recaptured results above supersede prior runs |
| Live edge and lobe-quality studies | `live_edges.py`: presenter, native temporal preparation | Unaffected |
| Spatial ablation | `ablate_spatial.py`: preserved live temporal outputs, only a-trous replay | Unaffected; merely constructing ShaderReplay does not use canonical motion |
| Broader-filter validation | `validate_broader.py`: presenter/native denoiser | Unaffected; no recapture required for this fix |
| Synthetic footprint stress | Constructed signals, no canonical capture | Unaffected |
| Rendered occlusion | Originally reconstructed correspondence; now corrected capture with projection assertion | Validated against independent projection |

The prior statement that broader-filter results needed recapture was incorrect.
Their recommendation remains: retain weight 4 by default and expose weight 2
as a noise/detail preference. This audit establishes which motion producer each
study uses; it is not a new quality measurement.
