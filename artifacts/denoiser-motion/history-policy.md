# Motion history policy experiment — 2026-09-06

**Follow-up:** the [matched-reference edge audit](edge-audit/README.md) found
that the original edge-correlation failure was confounded by the noisy
reference. Read that audit before interpreting the historical results below.

The three-frame floor is available as
`RendererConfig(denoiser_motion_history_floor=3)`. It remains experimental;
the default is 1. We did not replace or relax the accepted quality baseline.

## Identical captured optics inputs

The saved fast-camera sequence was replayed with floors of one, two and three,
and with no motion cap. All other filter constants and inputs were identical.

| Policy | Log-luminance RMSE ↓ | Motion-region RMSE ↓ | Stationary temporal residual ↓ | Edge correlation ↑ |
|---|---:|---:|---:|---:|
| Previous/default floor 1 | 0.6994 | 1.4135 | 0.7525 | 0.4217 |
| Floor 2 | 0.6227 | 1.2767 | 0.6306 | 0.4175 |
| Floor 3 | 0.6099 | 1.2700 | 0.6131 | 0.4197 |
| No motion cap | 0.6037 | 1.2705 | 0.6017 | 0.4072 |

Floor three reduces overall error by 12.8% and stationary temporal residual
by 18.5%, while slightly reducing the edge metric. At moderate camera speed,
the existing cap is six; floors two and three produce image-identical outputs
to the previous policy.

## Live Vulkan gate

The existing gate uses a separate room fixture with camera and object motion.
The three-frame policy **fails** its accepted camera edge threshold. A control
run with the previous policy passes and reproduces the accepted camera image
metrics. This isolates the regression to retaining additional history, rather
than the offscreen random-sequence fix or the NRD diagnostic bridge.

| Camera result | Previous policy | Floor 3 |
|---|---:|---:|
| Log-luminance RMSE ↓ | 0.2969 | 0.2263 |
| Motion-region RMSE ↓ | 0.4422 | 0.3688 |
| Stationary temporal residual ↓ | 0.3726 | 0.2908 |
| Edge correlation ↑ | 0.1122 | -0.0133 |

The accepted minimum edge correlation is 0.0922. This is not a borderline
failure. Both object-motion runs have identical image metrics, as expected
when the camera-motion policy is unchanged.

Reports: [previous live policy](live-previous.json),
[three-frame live policy](live-floor3.json), and
[three-frame optics replay](fast-floor3.json). The live captures predate the
CLI flag and selected the alternative policy in-process; both used the same
reported scene/render configuration. The runner now exposes `--history-floor`
and records non-default choices explicitly.

## Targeted rejection checks

`python -m tools.denoiser_motion.check_rejection` executes the production WGSL
shaders with diagnostic acceptance output. It verifies:

- A moving foreground retains correctly reprojected history.
- Newly revealed background rejects old foreground, resets history length,
  and does not retain its radiance.
- Abrupt lighting changes reject stale radiance.
- Camera cuts reject all history.
- Retained fast-motion history is bounded by three frames.

These pass, but do not supersede the live edge failure. They isolate basic
contracts; the room gate exercises more complex sampling and reconstruction.

## Next boundary

A global history floor provides a useful A/B option, but is not ready as the
default. Further work should diagnose the live edge failure and introduce
local confidence or rejection that preserves detail while retaining history
in stable interiors. That work needs live signal/guide captures around failing
edges, rather than simply increasing the global accumulation limit.
