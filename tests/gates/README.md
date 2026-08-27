# Renderer gates

This package owns executable correctness, image-quality, hardware-feature, and
performance gates. Ordinary unit tests remain directly under `tests/`; probes
which only print device information remain development diagnostics at the
repository root.

Run the normal, hardware-independent suite with:

```bash
python -m unittest discover -s tests
```

The formal GPU gate wrapper is intentionally opt-in because it opens Vulkan
windows, writes captures, and may take several minutes:

```bash
ORDINARYLIGHT_RUN_GPU_GATES=1 \
python -m unittest tests.gates.test_gpu_gates -v
```

Add the 4K performance stage explicitly:

```bash
ORDINARYLIGHT_RUN_GPU_GATES=1 \
ORDINARYLIGHT_RUN_PERFORMANCE_GATES=1 \
python -m unittest tests.gates.test_gpu_gates -v
```

Individual gates remain directly runnable as modules, for example:

```bash
python -m tests.gates.execution_parity --help
python -m tests.gates.restir_matrix --help
tests/gates/run_4k_performance.sh
```

Gate reports default to `/tmp` or to an explicitly supplied output directory.
Tests must not write generated captures into the source tree.

GitHub-hosted runners execute the normal suite only. `.github/workflows/gpu-gates.yml`
provides a manual hardware workflow for a self-hosted Linux runner carrying the
`ordinarylight-gpu` label. Its `performance` input controls the 4K stage.
