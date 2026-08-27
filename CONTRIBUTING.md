# Contributing to Ordinary Light

Ordinary Light is a general-purpose renderer with a backend-neutral public API.
Changes should preserve that separation: application-specific behavior belongs
outside the core package, while reusable renderer concepts belong in semantic
modules such as `loaders`, `outputs`, `backends`, and `integrations`.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
```

GPU gates are opt-in because they require suitable Vulkan ray-tracing hardware:

```bash
ORDINARYLIGHT_RUN_GPU_GATES=1 \
python -m unittest tests.gates.test_gpu_gates -v
```

See `tests/gates/README.md` for the 4K performance gate and direct gate usage.
GitHub-hosted CI runs only the hardware-independent suite. The manually
dispatched `GPU gates` workflow targets a self-hosted Linux runner labeled
`ordinarylight-gpu`; it is not a required check unless the project has such a
runner registered.

## Pull requests

- Add focused tests for behavior and API changes.
- Keep GPU implementation details behind the public renderer abstractions.
- Preserve image quality unless a documented tradeoff has been explicitly
  accepted.
- Run `python -m build` and `python scripts/verify_wheel.py dist/*.whl` after
  changing package data or build configuration.
- Do not commit local captures, caches, or generated benchmark output.

## Shader inventory

`ordinarylight/shaders/manifest.json` owns the compiled shader families and
target environment. Validate the checked-in SPIR-V inventory without requiring
a compiler:

```bash
python scripts/compile_shaders.py --check
```

Use `--list` to inspect the expanded plan, `--only GLOB` for a focused rebuild,
or run without either flag to rebuild the complete inventory with
`glslangValidator`. New specializations must be added to the manifest and must
not leave unmanaged `.spv` files behind.

## Releases

1. Update `version` in `pyproject.toml` and `CHANGELOG.md`.
2. Merge the release changes and create a GitHub Release tagged `vVERSION`.
3. Publishing the release runs the wheel workflow, validates the tag/version,
   and attaches the wheel to the release.
