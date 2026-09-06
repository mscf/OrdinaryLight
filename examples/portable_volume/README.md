# Portable scientific volume

Prepare a DMC vector histogram without allocating a GPU device, then execute
and render the exported package with browser WebGPU:

```sh
bash examples/portable_volume/run-local.sh --output /tmp/portable-volume
```

The local launcher uses this checkout's `.venv` and the sibling OrdinaryScience,
OrdinaryLattice and ordinaryshade source checkouts. It uses a sibling LatticeModel
checkout when available, otherwise `.staging/LatticeModel` in this repository.

Open `http://127.0.0.1:8765/`. This example needs the current upstream preparation
modules from OrdinaryScience, OrdinaryLattice, OrdinaryShade and LatticeModel
in the Python environment. If those are already installed in your active
environment, you can instead run `python examples/portable_volume/serve.py`.

See [the portable contract, update semantics and verification instructions](../../docs/portable_volume.md).

## Automated conformance

Using the wheel environment from `scripts/check_portable_install.py`, install
the native executor and run the gate:

```sh
/tmp/ordinary-portable-install/venv/bin/python -m pip install 'wgpu>=0.32'
/tmp/ordinary-portable-install/venv/bin/python examples/portable_volume/run_conformance.py \
  --output /tmp/portable-conformance
```

The output directory must be new. Chrome and Node 20+ must be on PATH (override
with `--chrome` / `--node`). The default Linux hardware configuration uses
X11/ANGLE/Vulkan and opens a dedicated window, requiring a desktop display.
`--headless` is available for hosts supporting hardware WebGPU in headless Chrome.
Native execution uses the adapter selected by wgpu. There is no automatic
software fallback; inspect the reported adapter identities for each run.
This is a conformance test, not a performance benchmark.

The runner owns its server, debugging port and temporary Chrome profile and
stops them on success or failure. It compares initial, live-update and
structurally replaced outputs against native execution and a NumPy histogram
reference. It checks same-runtime and fresh-runtime snapshot restoration,
pixel restoration, shared-device replacement, failed-update recovery, and
diagnostics for incompatible/corrupt packages. Floating outputs use `1e-5`
absolute/relative tolerance; histogram counts must match exactly. Pixel checks
compare browser snapshots on the same device, not browser/native raster images.

Results include `verification.json` (actual adapter identities and numerical
errors), `browser.json`, `browser-volume.png`, exported packages, and server,
Chrome, probe and native logs. Any failed check exits nonzero. The manually
dispatched **Portable browser/native conformance** workflow runs this same gate
on an `ordinarylight-gpu` self-hosted runner with upstream Git read access.
The runner needs an accessible desktop display for the default Chrome mode.
