# 4K GI cost investigation

Glass-detail scene, NVIDIA laptop GPU, 3840×2160 swapchain, eight warm-up
frames then 12 moving and 12 stationary frames. Medians exclude the first two
frames per phase. Ordinary Shade denoising and glass history cap enabled.
Default: four ReSTIR streams, eight bounces, 131072-path tiles. Short runs on
one desktop session; numbers are indicative, not universal FPS guarantees.
Captures happen after each phase and are outside the timed loop.

The new `wavefront_timestamps` option records GPU stage timestamps without
selecting work-counter profiling shaders or disabling production shader
specializations. It is enabled in the direct viewer. CPU timings and delayed
GPU samples are separate; they must not be added as if they were disjoint.
The GPU samples belong to the completed frame slot (typically two submissions
behind). Query capacity now accommodates the thousands of stages at 4K;
previous 1024-query storage was inadequate. HDR resolve and denoiser guide
preparation now have separate labels.

## Measurements

| Experiment | Moving wall ms | Stationary wall ms | Moving GPU ms |
| --- | ---: | ---: | ---: |
| 4K baseline, intermediate presentation | 312.0 | 304.3 | 304.0 |
| Same, timestamps disabled | 308.1 | 300.6 | 301.5 |
| 1048576-path tiles | 352.4 | 347.6 | 343.5 |
| 32768-path tiles | 392.8 | 380.9 | 385.0 |
| One stream, native 4K | 101.6 | 95.6 | 92.3 |
| Four streams, direct presentation | 311.3 | 301.4 | 302.9 |
| One stream, 1080p internal, direct 4K presentation | 28.8 | 22.3 | 21.9 |

Both tile changes were rejected. Timestamp overhead was roughly 1–2% in this
sample. No tile size, sample count or resolution default was changed.
The lower-resolution configuration meets a 33.3 ms moving frame budget in
this test, but is explicitly upscaled output, not native 4K. No visual-quality
acceptance or native-4K 30/60 FPS claim is made.

Native direct moving GPU breakdown (mean stages, not median total): primary
125.8 ms; secondary intersect/shade 91.5 ms; guide preparation 37.6 ms;
HDR resolve 21.9 ms; temporal/spatial denoising 15.1 ms; presentation 8.8 ms;
reconstruction 1.1 ms. Stationary guide preparation averaged 43.7 ms.
Guide preparation dispatches only each tile's paths, not the full image per
tile. Repeated full-image computation was investigated and not found there.

Next work: profile primary and secondary shader costs more closely, reduce
per-sample guide/resolve bandwidth, and evaluate fewer streams with objective
noise/motion comparisons. 60 FPS requires 16.7 ms total, 30 FPS 33.3 ms.
These native measurements require architectural/shader work, not merely CPU
upload tuning. Captured HDR can vary with queue ordering; instrumentation and
tile experiments are not claimed to be bit-identical here.

## Reproduction

From repository root with native Vulkan/display access:

```sh
DIRECT=1 PROFILE=0 OUT=/tmp/4k-native PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/4k-performance/capture.py
DIRECT=1 STREAMS=1 SCALE=0.5 PROFILE=0 OUT=/tmp/4k-upscaled PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/4k-performance/capture.py
```

`CAPACITY`, `STREAMS`, `SCALE`, `TIMESTAMPS`, `FRAMES` select experiments.
`PROFILE=0` disables Python cProfile, not GPU timestamps. GPU timestamping is
controlled separately by `TIMESTAMPS`. Raw HDR remains in /tmp/profile-4k-*.
