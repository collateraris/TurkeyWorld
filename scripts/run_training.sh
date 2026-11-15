#!/usr/bin/env bash
# Explicit foreground launcher. Nothing starts during setup/install.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--help" || $# -eq 0 ]]; then
    echo 'Usage: bash scripts/run_training.sh [--allow-sleep] --module package.module [target arguments]'
    echo 'Default: systemd-inhibit --what=sleep --mode=block; display/lock policies unchanged.'
    echo '--allow-sleep skips this wrapper only; workers may still request sleep protection.'
    echo 'For terminal disconnects, run this command inside tmux or an explicit systemd service.'
    exit 0
fi
ALLOW_SLEEP=0
if [[ "${1:-}" == "--allow-sleep" ]]; then ALLOW_SLEEP=1; shift; fi
if [[ "${1:-}" != "--module" && "${1:-}" != "--script" ]]; then
    echo 'Expected --module or --script; use --help.' >&2; exit 2
fi
if [[ -n "${TURKEYWORLD_PYTHON:-}" ]]; then
    PYTHON="$TURKEYWORLD_PYTHON"
    if [[ ! -x "$PYTHON" ]]; then echo 'TURKEYWORLD_PYTHON is not executable.' >&2; exit 2; fi
elif [[ -n "${VIRTUAL_ENV:-}" && -x "$VIRTUAL_ENV/bin/python" ]]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif [[ -x "$ROOT/.venv-ubuntu/bin/python" ]]; then
    PYTHON="$ROOT/.venv-ubuntu/bin/python"
elif [[ -x "$ROOT/.venv/bin/python" ]]; then
    PYTHON="$ROOT/.venv/bin/python"
else
    PYTHON="$(command -v python3)"
fi
cd -- "$ROOT"
if [[ "$ALLOW_SLEEP" == 1 ]]; then
    echo 'Outer sleep inhibition explicitly disabled by --allow-sleep; worker KeepAwake requests remain available.' >&2
    exec "$PYTHON" -m turkeyworld.bootstrap "$@"
fi
if ! command -v systemd-inhibit >/dev/null 2>&1; then
    echo 'systemd-inhibit is unavailable. Install systemd, or explicitly pass --allow-sleep.' >&2
    exit 2
fi
exec systemd-inhibit --what=sleep --mode=block --who=TurkeyWorld \
    --why='Neural game training' -- "$PYTHON" -m turkeyworld.bootstrap "$@"
