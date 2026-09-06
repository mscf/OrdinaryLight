#!/usr/bin/env bash
# Launch with the upstream source checkouts used by this development workspace.
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${repo_root}/.venv/bin/python"
model_root="${repo_root}/../LatticeModel/src"
if [[ ! -d "${model_root}/latticemodel" ]]; then
    model_root="${repo_root}/.staging/LatticeModel/src"
fi
source_roots=(
    "${repo_root}"
    "${repo_root}/../OrdinaryScience/src"
    "${repo_root}/../OrdinaryLattice/src"
    "${repo_root}/../ordinaryshade"
    "${model_root}"
)
if [[ ! -x "${python_bin}" ]]; then
    echo "Missing ${python_bin}; create the project's Python environment first." >&2
    exit 1
fi
for source_root in "${source_roots[@]}"; do
    if [[ ! -d "${source_root}" ]]; then
        echo "Missing local source checkout: ${source_root}" >&2
        exit 1
    fi
done
source_path="$(IFS=:; echo "${source_roots[*]}")"
export PYTHONPATH="${source_path}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${python_bin}" "${repo_root}/examples/portable_volume/serve.py" "$@"
