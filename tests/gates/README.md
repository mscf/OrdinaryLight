# Renderer gates

This package owns executable correctness, image-quality, hardware-feature, and
performance gates. Ordinary unit tests remain directly under `tests/`; probes
which only print device information remain development diagnostics at the
repository root.

Run the normal, hardware-independent suite with:

```bash
python -m pytest -q tests
```

The manually dispatched `GPU gates` workflow first builds the FSR2 bridge and
runs the GPU-enabled pytest suite, then runs the formal gate wrapper below.
It installs pytest and enables both graph and transport tests. The pytest step
disables the gate-wrapper flags so the expensive gates are not run twice.
Its JUnit report and failure diagnostics are uploaded as `gpu-pytest-results`,
including on test failure.

The self-hosted `ordinarylight-gpu` runner needs a suitable Vulkan ray-tracing
GPU/driver, a desktop display, g++, Vulkan development headers/libraries, and
`glslangValidator`. FSR2 is built from this checkout rather than relying on an
old bridge left on the runner. To reproduce the new step locally:

```bash
python scripts/build_fsr2.py
ORDINARYLIGHT_TEST_VULKAN_GRAPH=1 \
ORDINARYLIGHT_TEST_VULKAN_TRANSPORT=1 \
ORDINARYLIGHT_RUN_GPU_GATES=0 \
ORDINARYLIGHT_RUN_PERFORMANCE_GATES=0 \
ORDINARYLIGHT_FSR2_LIBRARY="$PWD/.tools/fsr2/libordinarylight_fsr2.so" \
python -m pytest -q tests --junitxml=test-results/gpu-pytest.xml
```

This does not enable every optional backend or hardware gate, and the workflow
still requires manual dispatch; ordinary pull-request CI remains GPU-independent.

The formal GPU gate wrapper is intentionally opt-in because it opens Vulkan
windows, writes captures, and may take several minutes:

```bash
ORDINARYLIGHT_RUN_GPU_GATES=1 \
python -m unittest tests.gates.test_gpu_gates -v
```

Add the 4K performance stage explicitly:

```bash
ORDINARYLIGHT_RUN_GPU_GATES=1 \
ORDINARYLIGHT_RUN_PERFORMANCE_GATES=1 \
python -m unittest tests.gates.test_gpu_gates -v
```

Individual gates remain directly runnable as modules, for example:

```bash
python -m tests.gates.execution_parity --help
python -m tests.gates.restir_matrix --help
tests/gates/run_4k_performance.sh
```

Vulkan and WebGPU raster backends have a shared visual-parity gate covering
the feature scene, direct triangle path, and volume slicing. It writes paired
captures, difference images, and a JSON report outside the source tree:

```bash
python -m tests.gates.raster_parity
```

The ray-marched volume parity matrix separately exercises every authored
volume showcase: emissive transfer functions, overlapping media, single
scattering with an embedded opaque shadow caster, and bounded multiple
scattering. It compares native Vulkan and WebGPU HDR output and writes both
captures, amplified difference images, and metrics:

```bash
PYTHONPATH=../ordinaryshade python -m tests.gates.volume_raster_parity
```

Native analytic-light coverage separately verifies directional and spot
shadows, point-light cube shadows at an oblique long-shadow regression pose,
multiple simultaneous shadow-casting lights, and GPU light-array accumulation.
It writes the shadowed/unshadowed evidence for visual inspection:

```bash
PYTHONPATH=../ordinaryshade python -m tests.gates.raster_lighting --target vulkan
PYTHONPATH=../ordinaryshade python -m tests.gates.raster_lighting --target webgpu
```

Sparse-volume empty-space skipping can be checked independently for the GI
renderer or the native Vulkan raster ray marcher. The gate requires identical
HDR output and a measurable end-to-end speedup:

```bash
PYTHONPATH=../ordinaryshade python -m tests.gates.volume_empty_space --target gi
PYTHONPATH=../ordinaryshade python -m tests.gates.volume_empty_space --target vulkan-raster
```

Raster/GI approximate visual parity uses a path-traced reference while
separately measuring exposure-normalized color, edge structure, and foreground
coverage. This intentionally does not demand numerical identity for indirect
illumination or traced transmission:

```bash
python -m tests.gates.renderer_visual_parity
python -m tests.gates.renderer_visual_parity --scene modifier
```

The opt-in GPU suite additionally evaluates the refraction scene from every
camera in `tests/gates/poses/refraction_parity.json`. These fixed oblique,
front, and reverse views prevent a probe/refraction fix from overfitting a
single camera pose. Nested dielectric composition is likewise checked from
the fixed front and oblique views in
`tests/gates/poses/nested_dielectric_parity.json`. Those checks include an
object-local edge correlation inside the outer shell, so losing the inner
boundary cannot be hidden by otherwise similar room pixels.

For interactive inspection, render the same evidence into a side-by-side Qt
viewer. The raster half is exposure-matched and the metric summary remains
visible above both images:

```bash
python tools/renderer_parity_viewer.py
```

Resident scene/settings transitions have a dedicated gate. It verifies that
the Vulkan device and two-frame external P010 pool survive scene replacement,
and separately budgets startup and transition latency:

```bash
python -m tests.gates.transition_latency
```

The accepted multi-scene noise baseline exercises the presented ReLAX output
at one sample per pixel across diffuse, area-light, glossy/glass, fast-motion
dense geometry, and both static and camera-moving volume rendering. Independent
high-sample frames remain the reference:

```bash
python -m tests.gates.noise_quality
```

Replacing that baseline is intentionally reviewable and requires an explicit
reason. This is appropriate when an understood quality tradeoff is accepted,
or when a change materially improves the baseline:

```bash
ORDINARYLIGHT_NOISE_GATE_OVERRIDE_REASON="explain the accepted change" \
python -m tests.gates.noise_quality --accept-baseline
```

The tracked baseline contains aggregate metrics and configuration, not large
HDR captures. Its edge policy measures gradients after a small triangular
prefilter, preventing sample-scale noise in the reference from masquerading as
detail that the denoiser should preserve. A separate bright-boundary temporal
metric catches unstable subpixel coverage around emitters and reflected
highlights, including cases where a jittered primary ray alternates between
geometry and the environment. Per-frame reference and ReLAX sequences, the
complete CSV, and the run report remain under
`/tmp/ordinarylight_noise_quality` for inspection.
The same baseline also guards median candidate GPU time per scene with a 20%
relative margin and a 0.35 ms absolute floor for sub-millisecond captures.

Raw estimator correctness is intentionally kept separate from presented
denoised quality. The indirect/reuse stages in `validation_matrix` compare raw
HDR estimators, while `path_termination_quality` checks Russian roulette
against full paths without allowing denoising to conceal bias. Because
roulette is inherently a variance/performance tradeoff, that gate enforces a
strict absolute-bias limit and bounded raw variance amplification; it does not
require roulette and full-path one-sample captures to have identical noise.

Rigid-object motion has a separate focused ReLAX gate. It compares actual HDR pixels
against independent high-sample frames for both object and camera motion,
including stationary-background stability, moving/disoccluded regions, edge
preservation, and median GPU time:

```bash
python -m tests.gates.relax_motion_quality
```

Smooth matte tessellation has a focused ReLAX gate using the material-program
room and a close view of its cyan sphere. It projects the sphere's actual mesh
edges and rejects sparse dark residual concentrated along those edges:

```bash
python -m tests.gates.relax_tessellation_quality
```

Replacing its tracked baseline also requires an explicit review reason:

```bash
ORDINARYLIGHT_RELAX_MOTION_GATE_OVERRIDE_REASON="explain the accepted change" \
python -m tests.gates.relax_motion_quality --accept-baseline
```

Gate reports default to `/tmp` or to an explicitly supplied output directory.
Tests must not write generated captures into the source tree.

GitHub-hosted runners execute the normal suite only. `.github/workflows/gpu-gates.yml`
provides a manual hardware workflow for a self-hosted Linux runner carrying the
`ordinarylight-gpu` label. Its `performance` input controls the renderer's 4K
stage and its `nvenc` input installs the optional video dependencies and runs
the end-to-end 4K encoding gate.
# GPU video output

The optional NVENC gate validates the entire 4K path, including Vulkan tone
mapping and YUV conversion, CUDA external-memory/semaphore interop, and NVENC
encoding. It covers 8-bit NV12/H.264 and 10-bit P010/HEVC:

```bash
python -m tests.gates.nvenc_zero_copy
python -m tests.gates.nvenc_zero_copy --pixel-format p010
```

It requires `ordinarylight[video-gpu]`, an NVIDIA GPU, and a driver exposing
Vulkan ray tracing, CUDA external interop, and NVENC. The default median budget
is 16.67 ms after four warm-up frames; pass `--maximum-median-ms` when recording
an explicit hardware-specific exception.

## Native composition regressions

`tests/test_native_gi_composition.py` is discovered by the GPU pytest step. It
checks exact direct/composed HDR parity across eight moving-camera frames with
bilinear, FSR1 OrdinaryShade and FSR2, including an application-inserted stage and
reset to the default graph. Black or nonfinite frames fail independently of parity.
On mismatch, the test saves both image sequences in pytest's temporary directory;
the GPU workflow retains these alongside the JUnit report.

FSR1 and FSR2 lifecycle cases exercise command rebuilds, cached graph reuse,
active render-scale changes, resizing in both directions and resource retirement
on resize and close. Each case owns a fresh hidden window; opting into these
integration tests requires a working desktop display rather than silently skipping
them when window creation fails. The direct primary recorder is an independent
regression oracle; do not replace it with a call to the composed operation.

These maintained tests supersede the corresponding one-off `primary_parity.py`
and `viewer_lifecycle.py` checks under `artifacts/denoiser-motion/gi-composition`.
Additional bounded regressions alternate two live scenes six times under both
FSR paths, requiring finite/nonblack output and closure of retired scene resources.
A shared-surface test performs three GI/raster cycles with three frames per backend,
using GI settings projected through the viewer's raster configuration helper.
It checks native surface reuse and teardown, not Qt event delivery or raster image
quality. CPU lifecycle tests separately verify that bursts of restart requests
cannot retire resources while render/start/update futures are pending.
Long-duration stress and real Qt interaction automation remain separate work.

## Extended switching stress

Enable the `stress` checkbox when dispatching `GPU gates` to run the three
switching cases with `ORDINARYLIGHT_STRESS_MULTIPLIER=10`: 60 scene selections
under each of FSR1 and FSR2, and 30 complete GI/raster cycles on one surface.
Each selection/backend renders three frames, for 540 rendered frames in total.
The normal suite remains at multiplier 1, regardless of the runner environment.
The stress step has a 20-minute timeout; its JUnit report is included in the
`gpu-pytest-results` artifact. The multiplier is recorded as a test property.

```bash
ORDINARYLIGHT_TEST_VULKAN_GRAPH=1 ORDINARYLIGHT_STRESS_MULTIPLIER=10 \
python -m pytest -q tests/test_native_gi_composition.py \
  -k 'repeated_scene_replacement or repeated_gi_raster_surface_handoff' \
  --junitxml=/tmp/gpu-stress.xml
```

Multipliers from 1 to 100 are accepted for local runs. This is an extended,
finite lifecycle test, not an hours-long soak, a VRAM leak measurement, or a Qt
input/event-loop test. It checks live output, retired-resource closure and
successful repeated presentation/device teardown.
