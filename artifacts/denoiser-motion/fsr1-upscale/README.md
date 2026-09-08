# Experimental FSR 1 EASU

Uses unmodified AMD FP32 EASU from FidelityFX-FSR revision
`a21ffb8f6c13233ba336352bdff293894c706575`. Headers and MIT notices are
vendored in `ordinarylight/shaders/third_party/fsr1`. RCAS is disabled.

The adapter emulates the gather callbacks using clamped HDR image loads,
ACES tone mapping and sRGB encoding before EASU. Results feed the existing
output/effects path without another tone-map operation. This avoids a new
image/pass but repeats per-sample tone mapping; it is a correctness/quality
prototype, not an optimized texture-gather implementation. No extra AA
stage is added; denoised input can still contain aliasing and noise.

FSR 1 runs only below native scale. The separate temporal reconstruction,
stationary accumulation and diffuse reconstruction filters are rejected for
this mode because EASU bypasses them. ReLAX and its histories still operate.

## Captures

`comparison.png` shows actual GPU RGBA output: bilinear, clamped cubic, and
FSR 1, with rows for glass/target/camera motion. Each 640×480 sequence has
12 moving + 12 stationary frames at half internal resolution. Every internal
HDR frame is byte-identical across modes. A repeat at native resolution
produces byte-identical final output between bilinear and FSR 1.

Inspection suggests cleaner curved edges with FSR 1 than cubic. It does not
recover missing detail or eliminate jaggies. Mean settled pixel standard
deviation over eight frames in normalized display RGB is:

| Motion | Bilinear | Cubic | FSR 1 |
| --- | ---: | ---: | ---: |
| Glass | .00595 | .00739 | .00693 |
| Target | .00592 | .00722 | .00699 |
| Camera | .00560 | .00707 | .00665 |

These short sequences and final screenshots are not a high-sample reference
quality gate or evidence of general motion/planar-mirror stability. FSR uses
tone mapping before filtering; bilinear/cubic filter HDR first, so differences
include that ordering as well as the filter kernel.

## Provisional cost

Direct swapchain, 4K output, 50% internal resolution, one ReSTIR reservoir,
moving camera. Eight warmup + sixteen measured frames per run:

| Order | Filter | Median reconstruction ms | Median GPU frame ms |
| --- | --- | ---: | ---: |
| 1 | Bilinear | .315 | 19.36 |
| 2 | Cubic | .512 | 19.52 |
| 3 | FSR 1 | 1.600 | 22.19 |
| 4 | FSR 1 | 1.590 | 24.21 |
| 5 | Cubic | .852 | 19.66 |
| 6 | Bilinear | .314 | 19.25 |

GPU contention is uncontrolled; the roughly 1.3 ms reconstruction increment
is provisional and whole-frame differences should not be attributed solely
to the filter. Bilinear remains default.

## Validation and reproduction

66 focused tests + 8 subtests and 3 shader-manifest tests passed. Both RGBA
and BGRA reconstruction binaries compile; generated sources are current.
The wheel includes the exact adapter, headers and provenance notices.

```sh
PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/fsr1-upscale/capture.py
SCALE=1 OUT=/tmp/fsr1-native PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/fsr1-upscale/capture.py
PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/fsr1-upscale/profile.py
```

Arrays remain in `/tmp/fsr1-upscale` and `/tmp/fsr1-native`.

The actual Qt viewer selected FSR 1 through its control and restart, verified
active config/timings, and completed native → 50% → F11 fullscreen 4K →
Escape/windowed 75% transitions without presentation failure.
