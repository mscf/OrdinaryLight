# Viewer render-scale integration

The native Qt/Vulkan viewer completed native → half → fullscreen half →
windowed 75% transitions without presentation failure. `viewer-extents.json`
records actual internal/output dimensions, including 1920×1080 → 3840×2160.
Each phase completed at least 12 frames. This is functional validation,
not an isolated GPU benchmark.

`capture.py` captures 24-frame sequences (12 moving, 12 settled) for glass,
target, and camera motion at 640×480 output and 100%/50% internal scale.
All six sequences produced finite HDR with expected dimensions. Run with:

```sh
PYTHONPATH=. .venv/bin/python artifacts/denoiser-motion/render-scale/capture.py
```

Arrays are written to `/tmp/render-scale-quality`. `comparison.png` shows
final internal HDR frames with display-only bilinear enlargement, not actual
swapchain screenshots or high-sample references. Qt screenshot capture of the
native Vulkan surface returned black, so it was not used as visual evidence.

Inspection shows preserved broad refraction patterns but softer sphere rims
and stripe edges at half resolution. This supports keeping native as default;
it does not establish temporal stability or planar-mirror quality. A full
moving-image/reference gate and controlled 4K timing remain follow-up work.
The production reconstruction shader uses bilinear HDR sampling here; temporal
reconstruction remains disabled. No reconstruction shader was changed.
