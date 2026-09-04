# Renderer roadmap status

This is the current implementation summary. Historical benchmark captures and
gate evidence are retained under [`../artifacts/`](../artifacts/README.md);
the [public API](public_api.md) describes the supported application contract.

## Renderer and target contract

- Algorithms live in `ordinarylight.renderers`: Vulkan global illumination,
  Vulkan/WebGPU raster, CPU reference, and hybrid composition.
- Execution APIs live in `ordinarylight.targets`: Vulkan, WebGPU, and CPU.
  There is no `ordinarylight.backends` namespace.
- `Renderer(renderer_preference="auto")` prefers hardware ray-query GI and
  selects Vulkan raster when GI is unavailable. Explicit `"gi"` and `"raster"`
  requests fail rather than silently changing renderer family. Selection and
  fallback reasons are inspectable through `renderer.renderer_selection`.
- Vulkan paths use a Vulkan 1.2 baseline. Generated SPIR-V targets that baseline.
- Raster and GI consume the same scene, camera, material, lighting, animation,
  interaction, and output abstractions where their techniques overlap. Indirect
  illumination, traced reflection/refraction, and multiple scattering remain
  GI-specific rather than raster parity requirements.

## Implemented foundation

- Renderer-neutral HDR arrays, named products, structured statistics,
  caller-owned outputs, capability discovery, and ordered asynchronous jobs.
- Stable mesh resources and instances, batch mutation, textures, lights,
  hierarchical animation, morph targets, skinning, and glTF/GLB ingestion.
- Perspective, orthographic, and panoramic cameras; points, lines, glyphs,
  materials, optical surfaces, volumes, clipping, and object effects.
- Picking with stable object identities, asynchronous GPU picking, CPU fallback,
  and viewport/DPI/dynamic-resolution coordinate mapping.
- Native raster shadow maps, GPU MRT depth/normal/object-ID/motion products,
  GPU volume ray marching, and screen-space optical compositing on Vulkan and
  WebGPU. CPU reference helpers remain available for validation.
- Direct Vulkan presentation with resident resources, GPU-resident video frames,
  FFmpeg sinks, and optional Vulkan/CUDA/NVENC H.264 and 10-bit encoding.
- Scene replacement and supported runtime reconfiguration retain the renderer
  device and reusable resources. Structural changes require recreation.
- Qt workbench scene and renderer startup runs off the event thread, with
  progress reporting, cancellable extension initialization, deferred close,
  serialized live resource updates, and compute/presentation scheduling.
- Generic showcase and workbench-extension hooks allow downstream applications
  to supply domain catalogs, panels, and controllers.

## Shader authoring and compute

- Ordinary Shade Python definitions are authoritative for built-in raster
  shaders. CI compiles and packages SPIR-V/WGSL; stale source-checkout artifacts
  trigger recompilation. Ordinary Shade is a declared core dependency.
- Generated wavefront modules cover transport policy, materials, sampling,
  volumes, reconstruction, and Vulkan orchestration, including the fused
  secondary loop and persistent scheduler. Handwritten branches remain as
  diagnostic fallbacks and ABI adapters. Shader migration and execution-parity
  gates guard the generated boundary.
- Device-keyed persistent Vulkan pipeline caching and lazy execution-strategy
  pipeline creation reduce repeated startup work. First-use compilation after
  a cache miss remains a performance concern.
- `ordinarylight.compute` provides persistent WebGPU sessions and WebGPU/Vulkan
  multi-pass sequences with reflected bindings and shared buffers. Domain
  compilers supply programs and resource plans without entering renderer code.
- Same-device resident compute buffers can feed WebGPU raster, Vulkan raster,
  and Wavefront GI volumes without host scalar-field readback. Buffer views are
  non-owning and their producer must outlive the renderer's use of the volume.
- OrdinaryScience owns scientific data semantics and controls. OrdinaryLattice
  owns model lowering; neither is a dependency of OrdinaryLight.

## Deferred, measured work

- Overlap headless readback through a Vulkan staging ring. Public async jobs
  are ordered, and execution through one high-level renderer remains serialized.
- Move animation deformation and frequent scene mutation toward GPU updates
  where measurements justify it; improve fine-grained resource invalidation.
- Continue measuring native raster allocation/upload costs and pass timings;
  see the [CPU/GPU audit](raster_cpu_gpu_audit.md).
- Improve interactive path-tracing noise and reconstruction while preserving
  the multi-scene HDR quality baselines for temporal stability, fireflies,
  structured noise, and edge detail.
- Redesign GI volume empty-space skipping before enabling it by default. The
  retained sparse-volume measurement was image-exact but slower (0.93x).
- Continue the 4K performance program toward at least 50 FPS without lowering
  image quality; record environmental exceptions with benchmark evidence.

## Extension points

- Formats belong in `ordinarylight.loaders.<format>` and register with
  `ordinarylight.loaders.load()`.
- Output transports belong in `ordinarylight.outputs`; GUI adapters belong in
  `ordinarylight.integrations`.
- Renderer implementations belong in `ordinarylight.renderers` and implement
  the same HDR frame and capability contracts. Execution targets remain separate.
