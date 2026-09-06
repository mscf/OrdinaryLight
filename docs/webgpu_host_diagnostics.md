# Local WebGPU host diagnosis — September 6, 2026

The example's null adapter was reproduced in a fresh Chrome profile, separately
from the user's existing browser profile. No user settings were changed.

## Environment

- Chrome: 152.0.7977.82.
- NVIDIA RTX 4070 Laptop GPU, driver 580.95.05; Intel integrated graphics also present.
- Desktop session: Wayland. Chrome's GPU report confirms Ozone `wayland`.
- VS Code: 1.136.1 (Snap), Electron 42.10.0, Chromium 148.0.7778.280.
- Example origin: `http://127.0.0.1:8765/`, verified as a secure context.

## Controlled Chrome tests

Every configuration used a separate temporary profile and the same example.

| GPU flags | Actual result |
| --- | --- |
| None | `navigator.gpu` exists, but `requestAdapter()` returns null. |
| `--use-vulkan` alone | Adapter still null; GPU report still shows ANGLE/OpenGL and Vulkan disabled. |
| `--enable-unsafe-webgpu` alone | Adapter/device creation and example startup succeed, using **SwiftShader (CPU)**. |
| `--use-angle=vulkan --enable-features=Vulkan,VulkanFromANGLE` | Adapter/device creation and example startup succeed, using **NVIDIA/Lovelace**. No unsafe-WebGPU override used. |

Default, high-performance, low-power and compatibility adapter requests all
returned null in the default profile. With the unsafe override alone, all four
selected SwiftShader.

The working hardware configuration also passed an execution/readback check:
224 histogram samples, 1,048,576 RGBA pixel bytes (512 × 512), 229 distinct byte
values, and no reported device loss after submitted work completed.

## What this establishes

The default renderer is Intel/Mesa via ANGLE/OpenGL (`GaneshGL`). The working
renderer is NVIDIA via ANGLE/Vulkan (`GaneshVulkan`). The default `chrome://gpu`
report nevertheless labels WebGPU hardware accelerated and lists both Vulkan
hardware adapters as available. That summary alone does not prove that a page
can acquire a usable adapter.

The problem is narrowed to Chromium's graphics-backend/adapter eligibility path
on this hybrid-GPU configuration. It is not simply an old Chrome version,
non-secure origin, missing `navigator.gpu`, or a shader error. These experiments
do not identify the exact internal check that rejects the hardware adapters in
the default configuration. Adapter enumeration and backend/external-image
requirements are separate checks in the
[installed Chromium revision's adapter-selection code](https://github.com/chromium/chromium/blob/d04cdb24d67b081f6cf80200ffc5233f44b61109/gpu/command_buffer/service/webgpu_decoder_impl.cc).

VS Code's bundled runtime version was inspected, but its embedded viewer was
not retested with these flags. Chrome's result must not be taken as proof that
the same configuration fixes VS Code.

## Reproduce the working hardware configuration

With the demo server running:

```sh
google-chrome \
  --user-data-dir=/tmp/ordinarylight-chrome-vulkan \
  --use-angle=vulkan \
  --enable-features=Vulkan,VulkanFromANGLE \
  http://127.0.0.1:8765/
```

The separate profile prevents an already-running Chrome process from absorbing
the request without adopting the new startup flags. This is a tested local
configuration, not a universal browser requirement.

For a manual adapter check in the page's developer console:

```js
const adapter = await navigator.gpu.requestAdapter();
console.log(adapter && {
  vendor: adapter.info.vendor,
  architecture: adapter.info.architecture,
  description: adapter.info.description,
});
```

In the working test this returned vendor `nvidia`, architecture `lovelace`.
Vendor `google`, architecture `swiftshader` identifies the software path observed
in the unsafe-flag-only test.
