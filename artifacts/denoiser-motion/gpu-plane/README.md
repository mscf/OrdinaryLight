# Experimental WGSL geometric gate

ShaderReplay now has an optional plane_gate path. Two rgba32float world-position
textures supply current/previous static hit positions. The gate uses the same
rgba16float normal guides as temporal validation and rejects history before
history sampling/length accumulation. Unlike the previous CPU-only experiment,
its decision feeds back into later temporal history.

The diagnostic shader assumes static geometry and a 45-degree vertical FOV.
It requests 17 storage textures per shader stage, up from the replay's 15;
this is an offline diagnostic requirement, not a portable/runtime contract.
Native renderer shaders and viewer behavior are unchanged.

## Four twelve-frame folded sequences

| Width | Grid | Accepted valid, baseline | Accepted valid, GPU gate | Wrong accepts, baseline | Wrong accepts, GPU gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 160 | 16 | 93,182 | 93,181 | 99 | 0 |
| 160 | 64 | 95,719 | 95,718 | 103 | 0 |
| 320 | 16 | 379,930 | 379,930 | 129 | 1 |
| 320 | 64 | 385,180 | 385,180 | 137 | 0 |

The GPU gate reduces observed wrong accepts from 468 to one, with two fewer
valid accepts. This is close to, but not identical to, the earlier full-precision
CPU result. Precision and pixel-selection differences need isolation before
claiming parity: CPU masks use float32 motion while replay uploads rgba16float
motion, and the shader uses half-precision normals.

Radiance is constant, so this tests acceptance/history updates, not visual
quality under changing illumination. Patch-label masks only distinguish the
two known patches. Counts are correlated across frames and resolutions.

Next retain per-pixel GPU rejection details for the three discrepant cases,
compare CPU predictions using the exact uploaded guide precision and selected
pixels, then test varying radiance and optical/moving geometry. Do not tune a
threshold solely to remove these discrepancies.

All four sequences (48 GPU frames) completed. Existing 38 regression tests
and lint/diff checks passed.

Reproduce:
```bash
.venv/bin/python -m tools.denoiser_motion.folded_sequence --plane-gate --output /tmp/folded-gpu-plane
```

Follow-up: [matching guide precision resolves the discrepancy](../visual-plane/README.md).
The same report includes the first rendered-radiance animation; it does not
demonstrate a visual-quality improvement.
