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
