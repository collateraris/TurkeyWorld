#!/usr/bin/env bash
# Create a Linux environment only. Never starts a worker or removes stop flags.
# Official matching binaries: https://pytorch.org/get-started/previous-versions/#v2110
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
python_bin="${TW_PYTHON:-python3.12}"
venv_path="${project_root}/.venv-ubuntu"
venv_explicit=0
dry_run=0
cpu_only=0

usage() {
  cat <<'HELP'
Usage: bash scripts/setup_ubuntu.sh [--dry-run] [--cpu] [--python PATH] [--venv PATH]
  --dry-run     Print the installation commands without running them.
  --cpu         Install official CPU PyTorch into .venv-ubuntu-cpu by default.
  --python PATH Use an existing Python 3.12 interpreter (default python3.12).
  --venv PATH   Use/create this dedicated environment.

Ubuntu 24.04 system prerequisites (run separately when needed):
  sudo apt-get update
  sudo apt-get install python3.12-venv python3-pip rsync tmux libgl1 libsdl2-2.0-0
No NVIDIA driver, checkpoint, immutable plan, data file or stop marker is modified.
HELP
}

while (($#)); do
  case "$1" in
    --dry-run) dry_run=1; shift ;;
    --cpu) cpu_only=1; shift ;;
    --python|--venv)
      if (($# < 2)); then printf 'Missing value for %s\n' "$1" >&2; exit 2; fi
      if [[ "$1" == --python ]]; then python_bin="$2"; else venv_path="$2"; venv_explicit=1; fi
      shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

torch_index='https://download.pytorch.org/whl/cu128'
torch_package='torch==2.11.0+cu128'
vision_package='torchvision==0.26.0+cu128'
if ((cpu_only)); then
  torch_index='https://download.pytorch.org/whl/cpu'
  torch_package='torch==2.11.0+cpu'
  vision_package='torchvision==0.26.0+cpu'
  if ((!venv_explicit)); then venv_path="${project_root}/.venv-ubuntu-cpu"; fi
fi

run() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if ((!dry_run)); then "$@"; fi
}

if ((!dry_run)); then
  if [[ "$(uname -s)" != Linux ]]; then
    printf 'This installer requires Linux. Use --dry-run to inspect it elsewhere.\n' >&2
    exit 2
  fi
  "$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12 to match the saved reference"'
fi

if [[ -e "$venv_path" && ! -f "$venv_path/pyvenv.cfg" ]]; then
  printf 'Destination exists but is not a Python virtual environment: %s\n' "$venv_path" >&2
  exit 2
fi
if [[ -f "$venv_path/pyvenv.cfg" && ! -x "$venv_path/bin/python" ]]; then
  printf 'Not a usable Linux venv (do not reuse a copied Windows environment): %s\n' "$venv_path" >&2
  exit 2
fi
if ((!dry_run)) && [[ -f "$venv_path/pyvenv.cfg" ]]; then
  "$venv_path/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Existing environment must also use Python 3.12"'
fi

run "$python_bin" -m venv "$venv_path"
venv_python="$venv_path/bin/python"
run "$venv_python" -m pip install --upgrade pip
run "$venv_python" -m pip install "$torch_package" "$vision_package" --index-url "$torch_index"
run "$venv_python" -m pip install -r "$project_root/requirements-ubuntu.txt"
run "$venv_python" -m pip check
run "$venv_python" -c 'import sys, torch, torchvision, numpy, PIL, pygame; print("Python:", sys.version); print("Torch:", torch.__version__, "torchvision:", torchvision.__version__); print("CUDA runtime:", torch.version.cuda, "available:", torch.cuda.is_available()); print("NumPy:", numpy.__version__, "Pillow:", PIL.__version__, "pygame:", pygame.version.ver); print("Environment only; training remains paused.")'
suffix=''
if ((dry_run)); then suffix=' (dry run)'; fi
printf 'Setup complete%s. No training was started; stop markers were preserved.\n' "$suffix"
