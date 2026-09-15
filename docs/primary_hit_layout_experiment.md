# Lossless primary-hit field planes

The diagnostic `scripts/primary_hit_layout_experiment.py` generates an
OrdinaryShade producer and matching vxl8r face-averaging consumer in temporary
modules. No production shader, buffer ABI, allocation or renderer changes are
enabled. `profile_primary_prefixes.py --stage full-hit-planes` runs the complete
lighting, denoising and face-averaging pipeline against the current layout.

## Representation

The current buffer stores six 16-byte fields per pixel in a 96-byte record.
The prototype instead stores six contiguous arrays per sample, in this order:
position/distance, geometric normal, shading normal, identity, ray origin,
ray direction. Addressing in 16-byte units is:

```
sample * pixel_count * 6 + field * pixel_count + pixel
```

Identity uses a bit reinterpretation, not integer-to-float numeric conversion.
All field precision, padding, jitter components and miss sentinels remain
unchanged. Memory allocation and total output byte count remain unchanged too:
759.375 MiB per sample at 3840×2160. This tests access organization, not
compression. The producer assembles its initial record locally before storing
each field; the later shading-normal update remains. Thus it includes the
single-record-write idea tested previously, not solely a layout substitution.

The face-averaging consumer reads position/distance and identity directly from
their arrays. It does not require a conversion pass. Explicit CPU diagnostic
readback reorders the fields into the existing structured dtype; normal frame
execution has no readback, new allocations, CPU conversion or inter-stage wait.

## Animated native 4K measurements

RTX 5090 Laptop GPU, perspective camera, 512×512 voxel scene, four bounces,
one sample per pixel, 524,288 paths, GPU animation/layout and tight bounds.
Each run alternates baseline/candidate order across 40 samples after 12 warmup
frames. Setup has its own timestamp; primary below excludes that interval.
Headless serialized GPU timing excludes presentation and is not viewer FPS.

| Capture | Original primary | Field-plane primary | Original total | Field-plane total |
| --- | ---: | ---: | ---: | ---: |
| First | 5.905 ms | 5.848 ms | 22.896 ms | 22.667 ms |
| Repeat | 5.894 ms | 5.846 ms | 22.953 ms | 22.534 ms |
| Wide overview | 5.405 ms | 5.324 ms | 16.453 ms | 16.057 ms |

Total savings were 0.23–0.42 ms (about 1–2%). Primary saved only 0.05–0.06 ms.
In the repeat, face output improved from 3.322 to 3.077 ms, accounting for much
of the benefit. The earlier experiment disabling exports cannot be interpreted
as a recoverable 1.35 ms gain from changing layout alone.

The overview saved 0.396 ms total (2.4%); face output improved from 3.591 to
3.177 ms. Raw, denoised and face-averaged HDR were identical there, and all six
sampled hit fields matched exactly. The benefit is consistent but modest and
primarily consumer-side; this experiment does not justify silently changing
the public buffer ABI. The prototype remains diagnostic-only.

Both close-up captures preserved raw and denoised HDR exactly. Face-averaged
maximum error was 8.94e-8 / 1.19e-7. All six exported fields compared exactly
at the final sampled frame. The 128×96 GPU smoke test also passed output and
all-field parity. CPU readback reordering was checked with two samples, odd
extent, arbitrary bit patterns and all-ones miss sentinels; that is not a
multi-sample GPU rendering test.

## Replay and scope

From the Ordinary repository root:

```sh
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-hit-planes
PYTHONPATH=../vxl8r/src .venv/bin/python components/OrdinaryLight/scripts/profile_primary_prefixes.py --stage full-hit-planes --overview
```

The harness temporarily selects producer compilation and the matching consumer
in its own single-threaded process. This is not an application integration API
and must not be copied into vxl8r production code. Existing public primary-hit
buffers remain arrays of 96-byte records. Any adoption needs an explicit public
layout contract, corresponding readback/shader accessors, and validation across
resize, multiple samples and other consumers. Raw logs are archived under
`artifacts/primary-hit-layout/`.
