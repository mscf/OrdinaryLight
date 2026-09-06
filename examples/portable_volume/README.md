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

## Hodgkin–Huxley voltage example

With the current OrdinaryScience and OrdinaryLattice development sources:

```sh
bash examples/portable_volume/run-local.sh --example hh --output /tmp/hh-volume
```

Open `http://127.0.0.1:8765/`. This prepares a single-compartment squid-axon HH
model at 6.3 °C with constant injected current. The grid sweeps `current`
(0–15 µA/cm²) and maximal sodium conductance `g_na` (80–140 mS/cm²). The vector
output is `voltage` in mV. The volume shows voltage-bin counts over the complete
simulation; counts multiplied by the timestep give time spent in each bin.
The central grid point also has a voltage-versus-time preview, making spike
timing visible. That diagnostic view reads back the voltage vector;
compute and volume rendering continue to share GPU-resident buffers.

Defaults are `g_k=36 mS/cm²`, `g_l=0.3 mS/cm²`, `C_m=1 µF/cm²`, reversal
potentials `E_Na=50`, `E_K=-77`, `E_l=-54.3 mV`, and initial voltage `-65 mV`.
Gates begin at their steady-state values at the initial voltage. Rate equations
use the absolute-voltage convention of [NEURON's HH mechanism](https://github.com/neuronsimulator/nrn/blob/master/src/nrnoc/hh.mod),
with a polynomial limit near the removable rate singularities at -40/-55 mV.

Integration is explicit Euler, initially `dt_ms=0.01` over `duration_ms=20`.
Samples are post-step: 0.01 through 20 ms. The step count is rounded to the
nearest integer, so a duration not divisible by dt ends at `round(duration/dt)*dt`.
Timestep and duration are structural controls; potassium conductance and initial
voltage are live controls. The current and sodium-conductance axes supply their
own per-cell values. Time units, spacing and sample count survive package
replacement and snapshot restoration. Pulse stimulation is not included yet.

To run the complete comparison in an environment with the modified projects
installed and native WebGPU available:

```sh
python examples/portable_volume/run_conformance.py --example hh \
  --output /tmp/hh-conformance
```

This tests a live `g_k` edit, replacement at `dt_ms=0.005` with 44 histogram bins,
restoration, rendering and package rejection. Browser/native histogram counts
must agree exactly; voltage tolerance is 0.02 mV with zero relative tolerance.
Native traces are additionally compared against a float64 CPU implementation
of the same Euler equations. Scientific tests check resting behavior, sodium
block, spiking, finite singular rates, and timestep refinement against RK4.
Agreement between executors is distinct from timestep accuracy: refine dt when
spike timing matters. Reports include CPU voltage errors, `*-voltage.npz` trace
data, the volume image, and a PNG of the browser's trace preview.

These changes require development versions until the next coordinated release.

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
