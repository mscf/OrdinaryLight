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

NRD is a GPU dispatch library, not a NumPy callable. The native bridge must own
or wrap a Vulkan device, upload the canonical signal images, retain NRD history
textures across calls, execute the dispatch list, and read back only the final
reference result. It must export exactly:

```python
version() -> str
denoise_relax(signals: ordinarylight.DenoiserSignals,
              settings: dict) -> tuple[numpy.ndarray, numpy.ndarray]
```

The adapter intentionally refuses to substitute the portable denoiser when the
module is absent: doing so would invalidate the comparison.

## Comparing a captured sequence

A capture directory contains ordered `frame-*.npz` files written by
`DenoiserSignals.save()` and one linear RGB `ground_truth.npy` image:

```bash
python -m tests.gates.denoiser_reference_quality captures/glossy-motion \
  --output /tmp/glossy-motion.json

# Add the independently built NRD bridge to PYTHONPATH first:
python -m tests.gates.denoiser_reference_quality captures/glossy-motion \
  --with-nrd --output /tmp/glossy-motion-with-nrd.json
```

Baseline replacement is explicit and auditable:

```bash
ORDINARYLIGHT_QUALITY_OVERRIDE_REASON='accepted after visual review' \
python -m tests.gates.denoiser_reference_quality captures/glossy-motion \
  --baseline tests/gates/baselines/glossy-motion-denoiser.json \
  --accept-baseline
```
