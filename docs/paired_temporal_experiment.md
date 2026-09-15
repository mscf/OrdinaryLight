# Paired temporal filtering experiment — not enabled

Implemented a combined OrdinaryShade diffuse/specular temporal shader. It shared
reprojection, geometry validation and neighborhood eligibility, while retaining
separate radiance statistics, reactive rejection and history lengths. Runtime
recording used one dispatch and borrowed the same input/output images. Six
existing temporal/preparation GPU tests passed; generated SPIR-V and WGSL were
validated. The implementation is archived, not used by the production runtime.

## Measurements

RTX 5090 Laptop GPU, true 3840×2160 GI/output, vxl8r 512×512 scene, four bounces,
1 spp, 524,288 paths, tight bounds and GPU animation/layout, prior retained
averaging/resolve/reset optimizations. Each pair alternated renderer order across
48 warmed timestamp samples. GPU timestamps are serialized diagnostics and
exclude actual presentation.

| Workload | Separate total | Paired total | Separate denoiser marker | Paired marker |
| --- | ---: | ---: | ---: | ---: |
| Animated orthographic close-up | 23.004 ms | 22.938 ms | 2.820 ms | 2.814 ms |
| Stationary orthographic close-up | 22.494 ms | 22.540 ms | 2.848 ms | 2.841 ms |
| Stationary perspective close-up | 22.809 ms | 22.619 ms | 3.549 ms | 3.324 ms |

No useful gain for the current viewer. Valid perspective history saves approximately
0.22 ms at the denoiser marker, about 0.19 ms / 0.8% of total time in this capture.
Camera projection changes the workload; compare variants within each row only.

Raw and denoised HDR were identical in the captured outputs. Averaged-output
maximum differences were at most 1.5e-7. After eight further stationary frames,
raw and denoised outputs also remained identical. The perspective diagnostic
confirmed 142 history-valid updates out of 146 calls, and verified that baseline
stages had two binding sets while candidates had one.

## Orthographic history finding

`VulkanRayQueryCore.prepare_wavefront_window` gates `relax_history_valid` on
`perspective_history_compatible`: both current and previous cameras must be
PerspectiveCamera instances. vxl8r's viewer uses OrthographicCamera. Thus even a
stationary viewer scene does not exercise denoiser temporal-history reuse.
This statement concerns the temporal denoiser specifically, not all accumulation
or averaging mechanisms in the renderer.

The earlier stationary parity claim in temporal_reset_optimization.md did not
prove valid-history behavior and has been corrected. No camera/history eligibility
changes were made in this experiment. Orthographic history support should be
evaluated separately for reprojection, rejection, visual stability and GPU cost;
enabling it is not automatically a performance optimization.

## Disposition

Restored the existing two-pass runtime and shader source. The larger shader also
requires 19 storage-image bindings versus 14 per original pass, so promoting it
would require device-limit handling and broader independent-lobe rejection tests.
The measured benefit does not justify that work for the current viewer workload.

Prototype source, generated shader, restoration patch, comparison scripts and
raw logs are archived under `artifacts/paired-temporal/`. Scripts are original
session diagnostics referencing `/tmp/paired-temporal-baseline`; reconstruct
that baseline from the pre-experiment runtime before replaying them. The
prototype.patch applies only to the corresponding pre-experiment source state.
