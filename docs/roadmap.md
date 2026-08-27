# Renderer roadmap status

## Implemented foundation

- Semantic public namespaces for loaders, cameras, lights, animations,
  materials, outputs, integrations, capabilities, and backends.
- A backend-neutral `Renderer`, named render products, structured statistics,
  caller-owned NumPy outputs, and explicit capability discovery.
- Perspective, orthographic, and panoramic cameras plus point, directional,
  spot, environment, emissive-triangle, and area-light workflows.
- Stable scene resources, instancing, mutation, morph targets, skinning, node
  animation, and deterministic animation playback.
- glTF/GLB ingestion with sparse and normalized accessors, scene transforms,
  instancing, morph targets, skins, animation, texture transforms,
  transmission, emissive strength, and unlit materials.
- Python-authored material programs and custom vertex attributes, including a
  deterministic staged wavefront path.
- General-purpose points, lines, glyphs, textures, volumes, overlapping media,
  transfer functions, and multi-volume composition.
- Headless linear-HDR NumPy rendering, SDR conversion, and optional streamed
  FFmpeg video encoding.
- Vulkan ray-query acceleration and a deterministic CPU reference backend.
- A script-extensible Qt workbench with worker-owned Vulkan presentation,
  background scene loading, and one-frame backpressure. Individual interactive
  showcase programs have been replaced by reusable scene builders and
  declarative workbench catalog scripts; automation gates remain standalone.
- Ordered non-blocking `RenderJob` submissions for headless/notebook/GUI use.
- Runtime-checkable backend protocols with `render_frame()` as the canonical
  entry point. Vulkan is isolated behind the optional `vulkan` extra, while
  the core package and CPU reference backend remain independently usable.
- Consumer-level wheel verification covering HDR arrays, named products,
  asynchronous jobs, and glTF loading from a clean downstream environment.
- A manifest-driven, deterministic shader inventory with checks for missing
  sources, missing binaries, and unmanaged compiled outputs.
- Focused examples for headless, asynchronous, video, custom-material, scene-
  update, and workbench workflows, backed by a concise getting-started guide.

## Deferred, measured work

- Add a Vulkan headless staging ring so multiple readback jobs can overlap GPU
  execution, synchronization, and CPU consumption. The current public async
  contract is stable, while backend execution is deliberately serialized.
- Move animation deformation and high-frequency scene updates from CPU array
  rebuilding toward GPU skinning/morphing and asynchronous acceleration-
  structure updates where measurements justify the complexity.

- Revisit interactive path-tracing noise and reconstruction quality. Current
  behavior is accepted for development, but quality remains equal in priority
  to performance. A tracked multi-scene HDR baseline now prevents silent
  regressions in accuracy, temporal stability, structured noise, fireflies,
  banding, and edge-detail preservation while this work proceeds.
- Redesign volume empty-space skipping before enabling it by default. The
  current sparse-volume gate is image-exact but measured 0.93x rather than a
  speedup, so the feature remains opt-in and experimental.
- Continue the 4K performance program without lowering image quality. The
  persistent target remains at least 50 FPS; environmental exceptions require
  an explicit recorded reason.

## Natural extensions

- Additional file formats belong in `ordinarylight.loaders.<format>` and register
  with `ordinarylight.loaders.load()`.
- Additional output transports belong in `ordinarylight.outputs`; GUI-specific
  adapters remain in `ordinarylight.integrations`.
- Additional renderer implementations belong in `ordinarylight.backends` and
  implement the same HDR frame and capability contracts.
