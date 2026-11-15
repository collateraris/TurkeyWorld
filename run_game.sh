#!/usr/bin/env bash
set -euo pipefail
task_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$task_root"
task_python="${PYTHON:-$task_root/.venv/bin/python}"
if [[ ! -x "$task_python" ]]; then task_python=python3; fi
exec "$task_python" -m turkeyworld.play --size 256 --display-size 1024 --fps 60 "$@"
