# Experimental clamped cubic upscaling

The viewer exposes bilinear (default) and clamped Catmull–Rom cubic filtering.
Cubic is evaluated only when the internal extent is smaller than output.
The 4×4 HDR neighborhood is weighted separably, then clamped per color channel
to the central 2×2 range. It uses the existing reconstruction pass and replaces
a padding word in its 256-byte push constants; no buffers or history are added.
Both RGBA and BGRA SPIR-V variants were rebuilt.

## Quality checks

`comparison.png` contains actual GPU reconstruction output from 24-frame
sequences at 640×480 output / 320×240 internal. Rows isolate glass, target and
camera motion, each followed by 12 stationary frames. The non-direct RGBA
output is read back before presentation; no host filter creates these images.
Internal HDR is byte-identical between modes for all frames. Native-resolution
runs also produce byte-identical final output with cubic selected.

Cubic makes sphere rims and stripes crisper but also emphasizes jagged edges.
Mean pixel temporal standard deviation over the last eight stationary frames
increases from ~0.0056–0.0060 to ~0.0071–0.0074 (normalized display RGB), about
22–26%. These short captures are not a high-sample reference quality gate.
This tradeoff is why bilinear remains default. No planar-mirror acceptance
or general temporal stability claim is made.

## Provisional cost

Four sequential runs in bilinear/cubic/cubic/bilinear order used 4K output,
half internal resolution, one ReSTIR reservoir, moving camera, and direct
swapchain storage. Each run discarded eight frames and recorded sixteen.
Median reconstruction times: 0.329 / 0.658 / 0.492 / 0.644 ms.
Median whole GPU frame times: 31.37 / 31.36 / 31.38 / 29.30 ms.
Competing GPU load was uncontrolled and produces substantial timing outliers;
these measurements do not establish a precise overhead or speedup.

The actual Qt viewer activated cubic through the control and restart, then
completed F11/escape transitions at 50% scale, including 1920×1080 → 4K,
and returned to 75% windowed rendering without presentation failure.
65 focused tests and 8 subtests passed; generated-source checks passed.

## Reproduce

```sh
PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/cubic-upscale/capture.py
SCALE=1 OUT=/tmp/cubic-native PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/cubic-upscale/capture.py
PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/cubic-upscale/profile.py
```

HDR/output checks and image arrays remain under `/tmp/cubic-upscale` and
`/tmp/cubic-native`; timing results are in `/tmp/cubic-profile.json`.
