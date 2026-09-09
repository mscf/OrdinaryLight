# Experimental FSR 2 Vulkan integration

The optional native bridge runs AMD FSR 2.2.1's host code, Vulkan backend and
FP32 shaders on the presenter's existing device. Pinned upstream sources,
license notices and build instructions are in `native/fsr2/README.md`.
Sharpening is disabled. This is a temporal integration, not an FSR 1 filter
renamed to FSR 2.

## Inputs and lifecycle

Primary rays use the SDK's coherent Halton jitter instead of independent
per-pixel jitter. The exact half-quantized offset travels through the camera
buffer and is also supplied to FSR. ReLAX supplies linear view depth and
previous-minus-current pixel displacement; the preparation pass converts
depth to reversed finite [0,1] depth (.1 to 10000) and removes current jitter
from displacement. Glossy or invalid surfaces receive conservative reactivity.
This does not reconstruct reflected or refracted-object motion.

The SDK owns temporal resources; frame-local preparation/output images are
owned by the bridge wrapper. GPU dispatch remains on the existing queue.
External input layouts are restored after SDK dispatch. Contexts are recreated
on extent changes, history resets with invalid ReLAX history, and frame time
is measured at submission. Dispatch is recorded every frame rather than
replayed from cached commands. At native scale FSR still provides temporal AA.
The feature currently requires perspective cameras, ReLAX, primary-surface
guides, and no separate reconstruction filters or object effects.

## Validation

Actual RGBA output in `comparison.png` compares bilinear / FSR 1 / FSR 2 on
three 24-frame sequences (12 moving, 12 stationary) at 640×480 output and
half internal resolution. FSR 2 improves the visible background stripe edges
and reduces settled pixel variation. Glass rims remain imperfect. Normalized
RGB temporal standard deviation over the final eight frames:

| Motion | Bilinear | FSR 1 | FSR 2 |
| --- | ---: | ---: | ---: |
| Glass | .00595 | .00693 | .00366 |
| Target | .00592 | .00699 | .00421 |
| Camera | .00560 | .00665 | .00383 |

This is a preliminary fixture check, not a high-sample reference quality gate.
Lower variation does not establish absence of ghosting. Arbitrary glass,
reflection and disocclusion behavior needs broader validation. FSR 2 changes
sampling, so internal-HDR parity is not expected. Bilinear and FSR 1 output
sequences are still byte-identical to their pre-integration captures.

The native viewer activated FSR 2, rendered at 50%, entered 4K fullscreen,
returned to windowed presentation, and switched to 75% without presentation
failure (`viewer.log`). Native smoke tests completed. 104 focused tests and
16 subtests passed across the two test batches; generated source checks and
all affected primary-ray shader variants were rebuilt. Sdist/wheel builds
passed: the sdist carries optional bridge sources, while the wheel carries
Python/shader integration and loads a separately built library.

## Provisional cost

4K output, half internal resolution, one reservoir, moving camera; eight
warmup + sixteen measured frames per run. Reconstruction includes the FSR 2
stage plus final tone mapping/output:

| Order | Mode | Median reconstruction ms | Median GPU frame ms |
| --- | --- | ---: | ---: |
| 1 | FSR 1 | 1.69 | 25.28 |
| 2 | FSR 2 | 3.42 | 26.41 |
| 3 | FSR 2 | 3.54 | 26.58 |
| 4 | FSR 1 | 1.77 | 25.09 |

GPU contention is uncontrolled. These are provisional costs, not a 4K FPS
guarantee or a clean isolation of algorithm overhead. Bilinear stays default.

## Reproduce

```sh
.venv/bin/python scripts/build_fsr2.py
PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/fsr2-upscale/capture.py
PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/fsr2-upscale/profile.py
```

Sequences are saved under `/tmp/fsr2-upscale`; raw timing data is in
`/tmp/fsr2-profile.json`. The locally built `.so` is intentionally not committed.
