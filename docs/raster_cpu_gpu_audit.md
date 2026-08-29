# Raster CPU/GPU work audit

This audit tracks work currently performed on the CPU in the raster renderer
and identifies the intended ownership. It distinguishes correctness scaffolding
from the steady-state rendering architecture.

## Addressed

- **Scene-static shadow preparation:** light-space planning, the shadow-sized
  texture packing and shadow geometry transformation are
  cached by `(scene identity, scene revision, RasterConfig)`. Camera motion no
  longer repeats this work. `scene_prepare_ms` reports cache misses separately
  from per-frame `scene_pack_ms`.
- **Native shadow generation:** Vulkan and WebGPU use a dedicated sampled D32
  depth texture produced by a GPU depth-only pass. Native paths neither
  allocate CPU shadow pixels nor embed encoded depth in the color atlas.
- **Camera-only frames:** native scene shaders transform resident world-space
  vertices with an 80-byte camera uniform. After swapchain warm-up, camera
  motion no longer rebuilds or uploads the 156-byte interleaved vertex stream.
- **Static initialization commands:** direct Vulkan presentation records atlas
  upload and shadow generation in a one-shot setup command for each resident
  swapchain resource set. Steady submissions replay only drawing/presentation.
- **Resident Vulkan objects:** swapchain-image command buffers, render targets,
  pipelines, descriptors, and scene-sized buffers are retained after warm-up.
- **CPU readback in the viewer:** direct presentation uses a Qt-owned Vulkan
  surface and never creates a NumPy/QImage frame.

## Remaining high-priority transfers

1. **Object/light/material mutation.** Camera state is resident, but object,
   light, and material mutations still rebuild the flattened stream. Complete
   the existing storage-buffer ABI so these update compact records instead.
2. **Fine-grained scene mutation.** Scene revision invalidation currently
   rebuilds all resident sets. Track dirty textures, shadow casters, transforms,
   lights, and materials independently.
3. **Volume slice construction.** Slice topology, transfer-function sampling,
   sorting, and vertex generation currently run in NumPy. Store volume data and
   transfer functions as GPU textures and generate/sort sampling in shaders.
4. **Named geometry products.** The offscreen depth/normal/object-ID products
   use a CPU rasterizer. Replace this with MRT attachments and asynchronous GPU
   readback only when a CPU result is explicitly requested.
5. **CPU fallback lighting/shadows.** Diffuse compatibility paths evaluate
   lighting and ray/triangle visibility on the CPU. Keep them for reference
   tests, but native backends should never select them during interactive use.

## Measurement gaps

- Add Vulkan timestamp queries for shadow, main raster, and final blit passes.
- Record fence wait and swapchain-image acquisition independently.
- Report the selected swapchain present mode. The direct viewer now reports
  `MAILBOX` or the required `FIFO` fallback; FPS alone is not renderer GPU time.

The target steady frame is: update compact camera/object/light buffers, submit
resident GPU work, and present. Scene flattening, atlas construction, and full
geometry uploads belong to scene mutation rather than camera animation.
