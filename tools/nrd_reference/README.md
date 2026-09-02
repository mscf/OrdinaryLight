# Optional NRD reference bridge

This directory is reserved for Ordinary Light's optional native NVIDIA NRD
bridge. It is a validation dependency, not part of the `ordinarylight` wheel or
runtime renderer.

The bridge consumes `ordinarylight.DenoiserSignals`, runs
`RELAX_DIFFUSE_SPECULAR`, and returns separate linear diffuse and specular
images. Ordinary Light discovers it as the Python module `ordinarylight_nrd`.

The integration is intentionally one-way:

1. Ordinary Light captures canonical, renderer-neutral denoiser signals.
2. NRD/ReLAX produces the reference output.
3. The Ordinary Shade denoiser consumes the same signals.
4. Validation compares both denoisers to each other and to a high-sample GI
   reference. NRD never becomes an application dependency.

The native implementation is built separately because NRD and NRI have their
own SDK build, platform, and license requirements. The pinned SDK revision and
build helper live in this directory rather than in package metadata.

## Preparing the reference SDK

```bash
python tools/nrd_reference/bootstrap.py --jobs 8
```

This fetches the revision recorded in `pins.json`, initializes the NRI
submodules selected by that NRD revision, and builds NRD with its NRI
integration and embedded SPIR-V shaders. Nothing is installed into the active
Python environment and the generated `_build` directory is ignored by Git.

NRD is a GPU dispatch library, not a NumPy callable. The native reference
bridge owns a minimal Vulkan 1.3 device, wraps it with NRI, retains NRD history
textures, uploads canonical signals, reads back the final reference result,
and measures the dispatch list with Vulkan timestamp queries. The Python
bridge exports:

```python
version() -> str
denoise_relax(signals: ordinarylight.DenoiserSignals,
              settings: dict) -> tuple[numpy.ndarray, numpy.ndarray]
denoise_relax_sequence(signals: tuple[ordinarylight.DenoiserSignals, ...],
                       settings: dict) -> list[tuple[numpy.ndarray,
                                                     numpy.ndarray]]

benchmark_relax(signals: tuple[ordinarylight.DenoiserSignals, ...],
                settings: dict,
                *, warmup: int,
                iterations: int) -> dict
```

``benchmark_relax`` sizes resident synthetic signal surfaces from the supplied
sequence, retains NRD history resources, and returns ``median_gpu_ms``,
``p95_gpu_ms``, ``wall_ms``,
``persistent_mib``, ``transient_mib``, and ``measured_frames``. GPU time must
come from Vulkan timestamp queries enclosing only NRD's dispatch list. It must
not be inferred from the blocking Python call, which includes upload, readback,
and synchronization.

The adapter intentionally refuses to substitute the portable denoiser when the
module is absent: doing so would invalidate the comparison.

## Visual A/B in the Qt showcase

After building the bridge, run:

```bash
python tools/raster_feature_viewer.py --target wavefront-gi
```

The **GI denoiser implementation** selector offers the production
**Ordinary Shade ReLAX (live)** path and **NVIDIA NRD ReLAX (reference
capture)**. The NRD choice captures up to four canonical GI signal frames,
runs them as one temporal NRD sequence, and displays the final linear HDR
result through the Qt readback viewport. The sequence is reduced at large
resolutions to keep host signal storage below 384 MiB. Stable
scene/camera/extent results are cached.

This mode is deliberately labelled a reference capture: the bridge currently
owns a separate Vulkan device and includes readback, upload, and subprocess
overhead. It is suitable for image-quality A/B comparisons, not real-time
performance comparisons. A future shared-device NRD runtime can implement the
same selector without changing its user-facing semantics.

## Comparing a captured sequence

A capture directory contains ordered `frame-*.npz` files written by
`DenoiserSignals.save()` and one linear RGB `ground_truth.npy` image:

```bash
python -m tests.gates.denoiser_reference_quality captures/glossy-motion \
  --output /tmp/glossy-motion.json

# Add the independently built NRD bridge to PYTHONPATH first:
python -m tests.gates.denoiser_reference_quality captures/glossy-motion \
  --with-nrd --output /tmp/glossy-motion-with-nrd.json

# Include repeatable GPU-only timing and allocation telemetry:
python -m tests.gates.denoiser_reference_quality captures/glossy-motion \
  --benchmark-nrd --nrd-warmup 16 --nrd-iterations 64 \
  --output /tmp/glossy-motion-nrd-benchmark.json
```

Baseline replacement is explicit and auditable:

```bash
ORDINARYLIGHT_QUALITY_OVERRIDE_REASON='accepted after visual review' \
python -m tests.gates.denoiser_reference_quality captures/glossy-motion \
  --baseline tests/gates/baselines/glossy-motion-denoiser.json \
  --accept-baseline
```
