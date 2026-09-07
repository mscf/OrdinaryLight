# Glass-detail viewer scenes

Start with the moving sphere:

```bash
.venv/bin/python tools/raster_feature_viewer.py --target wavefront-gi --showcase glass-detail-motion
```

The scene selector offers three related fixtures:

- **Glass detail: camera motion** (`glass-detail-camera`): stationary glass and
  target, using the viewer's normal camera orbit. Manual orbit and zoom work too.
- **Glass detail: moving sphere** (`glass-detail-motion`): only the glass moves.
- **Glass detail: moving target** (`glass-detail-target`): only the striped target moves.

Keep **Animate scene / camera** enabled to run the selected motion. Sphere and
target motion repeat a six-second cycle: move for one second, hold for two,
reverse for one, then hold for two. Uncheck Animate to freeze the current pose
and inspect denoiser settling. The glass and target modes keep the camera still
unless you manipulate it manually. Switching scenes creates a fresh history;
it is not a continuation of the preceding scene's history.

ReLAX and ReSTIR use the normal GI viewer defaults. The planar-mirror guide
checkbox does not apply to these scenes. Bright/dark/orange emissive bars make
blurred or displaced refracted detail easier to identify. The glass uses IOR
1.52 and radius 1; its denoising guides still describe the front surface.

The offline comparison shares this fixture builder:

```bash
PYTHONPATH=. .venv/bin/python -m tools.denoiser_motion.glass_detail
```

That diagnostic uses frame-indexed translation, independent reference renders,
and fixed comparison patches. The interactive camera orbit is intentionally not
an exact replay of its horizontal camera translation. See the
[history investigation](../artifacts/denoiser-motion/glass-history/README.md) for
known residual error. The optional cap below is disabled by default.

Validation: 24 fixture/viewer tests passed. Native viewer checks completed
603 frames (motion), 600 frames (target), 121 frames (camera).
All exited cleanly. Status dimensions describe the requested viewer extent,
not a verified internal rendering resolution.

## Experimental glass motion history cap

Enable **Glass motion history cap (experimental)**, then click **Apply and
restart renderer**. The option is off by default. It limits history to four
frames on marked transmissive primary surfaces whose screen-space guide motion
exceeds one pixel. History grows normally after motion stops. Both denoiser
lobes use the same cap. Compare identical movement/hold phases with it off/on.

The low-level option is `RendererConfig.denoiser_transmission_motion_cap`.
The marker currently comes from dynamically compiled custom-material primary
pipelines; stock primary variants do not emit it. The glass showcase uses the
supported custom Fresnel material. Opaque surfaces and unmarked samples retain
the existing temporal policy. This is not a refracted-background motion model:
a stationary glass surface cannot detect a moving target through this marker.

See the [integration validation](../artifacts/denoiser-motion/glass-cap-native/README.md)
and [eight-case diagnostic sweep](../artifacts/denoiser-motion/glass-cap-sweep/README.md).
