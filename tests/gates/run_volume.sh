#!/usr/bin/env bash
# Formal Ordinary Light GPU gate launcher.
set -euo pipefail

gate_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$gate_dir/../.." && pwd)"
task_python="${WAVE_RENDER_PYTHON:-$repo_root/.venv/bin/python}"

export WAVE_RENDER_SCENE=volumes
export WAVE_RENDER_WIDTH=${WAVE_RENDER_WIDTH:-1920}
export WAVE_RENDER_HEIGHT=${WAVE_RENDER_HEIGHT:-1080}
export WAVE_RENDER_BENCHMARK_WARMUP_FRAMES=${WAVE_RENDER_BENCHMARK_WARMUP_FRAMES:-20}
export WAVE_RENDER_BENCHMARK_FRAMES=${WAVE_RENDER_BENCHMARK_FRAMES:-100}
export WAVE_RENDER_PERFORMANCE_GATE=1
export WAVE_RENDER_PERFORMANCE_GATE_MIN_FPS=${WAVE_RENDER_PERFORMANCE_GATE_MIN_FPS:-30}
export WAVE_RENDER_PERFORMANCE_GATE_TARGET_WIDTH=$WAVE_RENDER_WIDTH
export WAVE_RENDER_PERFORMANCE_GATE_TARGET_HEIGHT=$WAVE_RENDER_HEIGHT
export WAVE_RENDER_BENCHMARK_SUMMARY=${WAVE_RENDER_BENCHMARK_SUMMARY:-/tmp/wave_volume_gate.json}

cd "$repo_root"
exec "$task_python" -m tools.wavefront_present
