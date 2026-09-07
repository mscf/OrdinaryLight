# Evaluated-lobe motion quality, 2026-09-07

**Keep evaluated lobes experimental and disabled by default.** They improve
aggregate error in the optics fixture but regress the original motion-room
fixture. Correct attribution alone does not establish better denoising.

## Measured results

Both comparisons below use a three-frame history floor. Lower error is better.

| Metric | Optics baseline | Optics evaluated | Change | Room baseline | Room evaluated | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Normalized log-luminance RMSE | 0.477310 | 0.412188 | -13.6% | 0.183244 | 0.208773 | +13.9% |
| Motion-region RMSE | 1.987351 | 1.947244 | -2.0% | 0.180716 | 0.197093 | +9.1% |
| Stationary-region temporal residual | 0.484480 | 0.293002 | -39.5% | 0.335395 | 0.386287 | +15.2% |
| Geometry-band log-luminance RMSE | 0.092226 | 0.087179 | -5.5% | 0.014009 | 0.015634 | +11.6% |

The one-frame floor shows the same aggregate direction: optics overall error
improves 15.3%, while room error worsens 14.7%.

Optics primary-surface regions, at floor three:

| Region | Baseline log-luminance RMSE | Evaluated | Change |
| --- | ---: | ---: | ---: |
| Glossy metal | 0.090631 | 0.090019 | -0.7% |
| Glass | 0.114285 | 0.109617 | -4.1% |
| Mirror | 0.021734 | 0.022622 | +4.1% |

The aggregate optics gain is not a large improvement on the optical objects
themselves. Small regional differences should not be treated as robust wins or
losses without more independent sequences. Glass remains visibly noisy.

- [Optics comparison image](optics/comparison.png), [complete metrics](optics/report.json).
- [Room comparison image](room/comparison.png), [complete metrics](room/report.json).

Images show the last moving frame, with reference, baseline, and evaluated
columns, and floor-one/floor-three rows. All panels use the same
`(rgb / (1 + rgb)) ** (1 / 2.2)` display transform.

## Protocol

Seven live Vulkan frames at 320×180, eight bounces, three A-trous iterations.
Candidates use two ReSTIR reservoirs, four candidates, and the existing
temporal policy. Both one- and three-frame history floors are captured.
Optics camera X moves from -0.4 to +0.4; the room uses the original camera
arc of 0.2 radians. Objects are stationary in both sequences.

References use the same live presenter with denoising and ReSTIR disabled,
eight independent 64-sample batches per camera pose (512 samples total),
and frame seeds starting at 1000. References are image-exact between the
enabled and disabled runs. The split-reference log-luminance RMSE is
0.014689 for optics and 0.003377 for the room, so the references are not
noise-free.

These measure the entire experiment: evaluated lobe attribution together with
per-sample channel accumulation. They do not isolate each change's benefit.
No timing claim, object-motion validation, or camera-cut/recovery claim is
made by these short camera-motion sequences.

For the room after the first frame, the fraction of foreground pixels with
history above one remains similar: diffuse 83.5% → 84.1%, specular
84.1% → 83.5%. The regression is not explained by wholesale history rejection.
A plausible next investigation is how the existing nonlinear spatial and
temporal filters respond to independently varying lobes; this is a hypothesis,
not an established cause.

## Reproduction

For each scene (`optics` or `object-motion`), use new output directories:

```bash
python -m tools.denoiser_motion.live_edges --scene optics --output /tmp/lobes-baseline
python -m tools.denoiser_motion.live_edges --scene optics --evaluated-lobes --output /tmp/lobes-evaluated
python -m tools.denoiser_motion.compare_lobes /tmp/lobes-baseline /tmp/lobes-evaluated --output /tmp/lobes-report
```

The comparison tool rejects mismatched configurations and nonidentical
references. Native arrays and guide captures remain in the temporary capture
directories; the compact reports and comparison images are retained here.

Next: isolate spatial filtering from temporal accumulation on these captures,
then test lobe-specific filtering settings against both fixtures. Do not
promote the experiment based only on the optics aggregate.
