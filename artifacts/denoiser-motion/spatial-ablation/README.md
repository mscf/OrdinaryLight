# Spatial-filter ablation, 2026-09-07

The room's evaluated-lobe regression enters primarily in the spatial stage.
The recombined temporal outputs have nearly equal error between modes:
0.377681 baseline versus 0.377846 evaluated in the room, and 0.544760
versus 0.544806 in optics. Independent nonlinear filtering of differently
partitioned signals changes the result after this stage.

**A broader luminance tolerance helps both signal modes.** It is available as
`RendererConfig(denoiser_color_weight=2.0)`, with the existing default of 4.0
unchanged. It does not eliminate the room's evaluated-lobe disadvantage when
comparing both modes at the same setting.

All values below use the existing three-frame history floor.

| Scene / mode | Color weight | Overall normalized error | Motion-region error | Temporal residual | Geometry-band error |
| --- | ---: | ---: | ---: | ---: | ---: |
| Room / baseline | 4 | 0.183300 | 0.180588 | 0.335512 | 0.014014 |
| Room / baseline | 2 | 0.150663 | 0.182366 | 0.257343 | 0.011612 |
| Room / evaluated | 4 | 0.208773 | 0.197093 | 0.386287 | 0.015634 |
| Room / evaluated | 2 | 0.171022 | 0.198773 | 0.295162 | 0.012925 |
| Optics / baseline | 4 | 0.477343 | 1.987430 | 0.484591 | 0.092231 |
| Optics / baseline | 2 | 0.417447 | 1.859469 | 0.344909 | 0.085885 |
| Optics / evaluated | 4 | 0.412189 | 1.947242 | 0.293003 | 0.087179 |
| Optics / evaluated | 2 | 0.387905 | 1.843894 | 0.247019 | 0.083093 |

Lower error is better. The broader filter reduces baseline overall error
about 18% in the room and 13% in optics. Room motion-region error increases
about 1%; this remains a noise/detail tradeoff, not a universal improvement.
No default promotion or timing benefit is claimed.

Disabling the first-pass specular firefly clamp does not repair the room
regression. Combining temporal channels before spatial filtering also does
not repair it in that fixture. Full metrics for these ablations are retained
in the four JSON reports beside this file.

[Comparison image](comparison.png): room above, optics below; baseline weight
4, baseline weight 2, evaluated weight 2. All use the same display transform.

## Method and interpretation

Replay uses the packaged GPU A-trous shader and preserved live temporal
outputs from the [paired live captures](../lobe-quality/README.md). Normal,
depth and material guides are taken directly from those captures. Sky pixels
retain their live output. The three spatial passes retain their normal/depth
weights, step sizes and specular first-pass clamp, except where a variant
explicitly changes them.

The replay is image-exact for the evaluated room. Other variants have small
Vulkan/WebGPU arithmetic/half-float differences: maximum absolute RGB
difference is 0.000244 for baseline room, 0.015625 for baseline optics, and
0.003906 for evaluated optics. Full replay RMSE is recorded in each report;
replayed aggregate metrics closely match live capture. Ablation values are
compared within the replay, rather than assuming all outputs are image-exact.

Important capture detail: `diffuse` and `specular` images are reused as
A-trous ping-pong targets. After three iterations they contain the second
spatial pass, not the original prepared signals. `temporal_diffuse` and
`temporal_specular` preserve the actual temporal outputs and are the replay
inputs. Replaying spatial filters does not feed changes back into temporal
history: native history already stores the pre-spatial result.

Run:

```bash
python -m tools.denoiser_motion.ablate_spatial /tmp/lobe-room-evaluated --output /tmp/new-ablation
```

The captured sequences cover camera motion only. Longer sequences, moving
objects, fine material detail and optical-region validation are needed before
changing defaults. These runs isolate spatial behavior but do not prove the
temporal stage is optimal.
