# Renderer gates

This package owns executable correctness, image-quality, hardware-feature, and
performance gates. Ordinary unit tests remain directly under `tests/`; probes
which only print device information remain development diagnostics at the
repository root.

Run the normal, hardware-independent suite with:

```bash
python -m unittest discover -s tests
```

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

The accepted multi-scene noise baseline covers diffuse, area-light,
glossy/glass, fast-motion dense geometry, and volume rendering:

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
HDR captures. Per-frame HDR sequences, the complete CSV, and the run report
remain under `/tmp/ordinarylight_noise_quality` for inspection.

Rigid-object motion has a separate ReLAX gate. It compares actual HDR pixels
against independent high-sample frames for both object and camera motion,
including stationary-background stability, moving/disoccluded regions, edge
preservation, and median GPU time:

```bash
python -m tests.gates.relax_motion_quality
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
