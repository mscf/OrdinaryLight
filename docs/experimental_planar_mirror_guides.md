# Experimental native planar-mirror guides

Start the dedicated static fixture:

```bash
.venv/bin/python tools/raster_feature_viewer.py --target wavefront-gi --showcase planar-mirror-guides
```

Enable **Planar mirror guides (experimental)**, then click **Apply and restart
renderer**. Orbit and zoom to compare with the checkbox off. Keep the built-in
denoiser enabled. The checkbox defaults off and only enables the feature for
this dedicated showcase; it does not affect other catalog scenes.

Native preparation reads the first secondary hit already captured by the
wavefront renderer. It reflects that position and normal across z=0 to construct
virtual-world depth, normal and camera-motion guides. Misses use zero guide
depth to avoid retaining a prior finite reflected surface. The geometric
point-to-plane gate from the offline experiments is not enabled.

## Scope

This first prototype is restricted to a static z=0 plane with near-zero packed
roughness and static reflected geometry. Eligibility is an explicit designation
by configuration, not automatic detection of every mirror BSDF. The renderer
clamps nominal zero roughness to approximately 0.001. The low-level
RendererConfig.denoiser_planar_mirror_guides option is false by default.

Primary mirror material/instance identity and roughness are retained. The
prototype does not export secondary material/instance identity, track moving
reflected objects, handle recursive mirror chains, refraction or curved
mirrors, or implement a general PSR path. This limits temporal rejection between
different reflected objects. Canonical CPU signal capture still describes
primary guides; this feature changes native preparation only.

No quality improvement or performance benefit is claimed. Use the dedicated
scene for interactive assessment before extending the resource contract.

## Validation

The live Vulkan smoke test runs camera motion and a stop with the toggle off
and on. At 160x120, all 6,634 mirror pixels change guide depth, including 1,163
finite reflected hits. Non-mirror guide depths are identical and both rendered
outputs are finite. Run:

```bash
.venv/bin/python -m tools.denoiser_motion.native_mirror
```

The diagnostic saves images and guides under /tmp/native-mirror-guides.
83 focused tests plus eight subtests passed; generated SPIR-V/WGSL artifacts
were verified. Existing broad-file lint findings include shader DSL globals
and pre-existing style issues; no blanket lint-clean claim is made.

## ReSTIR and swapchain recreation

ReSTIR direct-light reuse is enabled by default for this showcase again.
The startup device-loss investigation found cached commands referencing destroyed
resources after swapchain recreation. Teardown now invalidates both command-cache
slots before destroying those resources.

With ReSTIR enabled, three forced recreations completed 607 frames with guides
off and 604 with guides on. The actual internal extent in those runs was
838x720; the viewer status showed the requested 1280x720 extent. See
[the fix and evidence](../artifacts/denoiser-motion/restir-materials/swapchain-cache-fix/README.md).

## Glass sphere

The foreground sphere uses Fresnel glass. Custom-material primary transmission
now supplies front-surface denoising guides when signal capture is active,
avoiding the previous raw-output bypass for most transmitted samples. This is
not refraction-aware background reprojection; fine moving detail through glass
still needs separate validation. See the [glass comparison](../artifacts/denoiser-motion/glass-transmission/README.md).
