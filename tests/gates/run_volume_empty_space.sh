#!/usr/bin/env bash
# Formal Ordinary Light GPU gate launcher.
set -euo pipefail

gate_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$gate_dir/../.." && pwd)"
task_python="${WAVE_RENDER_PYTHON:-$repo_root/.venv/bin/python}"

exec "$task_python" "$gate_dir/volume_empty_space.py" "$@"
