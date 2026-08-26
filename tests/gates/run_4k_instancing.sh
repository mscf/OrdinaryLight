#!/usr/bin/env bash
# Formal Ordinary Light GPU gate launcher.
set -euo pipefail

export WAVE_RENDER_SCENE=dense
export WAVE_RENDER_INSTANCING_GATE=1
export WAVE_RENDER_INSTANCING_GATE_MIN_SHARED_BLAS_SAVINGS=39
export WAVE_RENDER_BENCHMARK_CSV="${WAVE_RENDER_BENCHMARK_CSV:-/tmp/ordinarylight_4k_instancing_gate.csv}"
export WAVE_RENDER_BENCHMARK_SUMMARY="${WAVE_RENDER_BENCHMARK_SUMMARY:-/tmp/ordinarylight_4k_instancing_gate.json}"

exec "$(dirname "$0")/run_4k_performance.sh"
