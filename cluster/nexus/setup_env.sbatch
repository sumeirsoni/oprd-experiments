#!/bin/bash
#SBATCH --job-name=oprd-setup
#SBATCH --partition=cml-furongh
#SBATCH --account=cml-furongh
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:h200-sxm:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/cmlscratch/%u/oprd-experiments/cluster/logs/setup_%j.out

# One-shot: create venv + install vLLM/deps on an H200 node (not on the login node).
# Nexus module initialization reads optional unset variables.
set -eo pipefail

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

source /etc/profile.d/modules.sh 2>/dev/null || fail "cannot initialize environment modules"
module use /opt/common/.modulefiles
module load cuda/12.8.1 || fail "cannot load cuda/12.8.1"

SCRATCH_ROOT=${SCRATCH_ROOT:-/cmlscratch/$USER}
ROOT=${ROOT:-$SCRATCH_ROOT/oprd-experiments}
CLUSTER_DIR=$ROOT/cluster
BRIDGE_HF_HOME=${BRIDGE_HF_HOME:-$SCRATCH_ROOT/hf-cache}
UV_BOOTSTRAP=${UV_BOOTSTRAP:-$ROOT/.oprd-uv-bootstrap}
UV=${UV:-$UV_BOOTSTRAP/bin/uv}
UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$ROOT/.oprd-python}
UV_CACHE_DIR=${UV_CACHE_DIR:-$SCRATCH_ROOT/uv-cache}
MANAGED_PYTHON_VERSION=3.12.7
ATTEMPT_TAG=${SLURM_JOB_ID:-$$}.${SLURM_RESTART_COUNT:-0}
VENV=${VENV:-$ROOT/venv}
ENV_READY=$VENV/.oprd-env-ready.json

cd "$CLUSTER_DIR"
mkdir -p logs runs "$BRIDGE_HF_HOME" "$UV_PYTHON_INSTALL_DIR" "$UV_CACHE_DIR"
export HF_HOME="$BRIDGE_HF_HOME"
export UV_CACHE_DIR UV_PYTHON_INSTALL_DIR

# Furong login and compute nodes expose different node-local Python modules.
# Bootstrap uv once on a compute node, then use its relocatable managed CPython
# from shared scratch for setup, preflight, status, resume, and every DAG phase.
if [[ ! -x $UV ]]; then
  module use /opt/local/stow/.modulefiles
  module load Python3/3.12.3 || fail "cannot load bootstrap Python3/3.12.3"
  python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' ||
    fail "bootstrap module Python is not Python 3.12"
  if [[ -e $UV_BOOTSTRAP ]]; then
    UV_BOOTSTRAP_BACKUP=$ROOT/.oprd-uv-bootstrap.pre-managed.$ATTEMPT_TAG
    [[ ! -e $UV_BOOTSTRAP_BACKUP ]] || fail "uv bootstrap backup already exists: $UV_BOOTSTRAP_BACKUP"
    mv "$UV_BOOTSTRAP" "$UV_BOOTSTRAP_BACKUP"
  fi
  python3 -m venv "$UV_BOOTSTRAP"
  "$UV_BOOTSTRAP/bin/python" -m pip install \
    --index-url https://pypi.org/simple --upgrade pip
  "$UV_BOOTSTRAP/bin/python" -m pip install \
    --index-url https://pypi.org/simple "uv==0.9.8"
fi
UV_VERSION=$("$UV" --version) || fail "cannot execute uv bootstrap: $UV"
[[ $UV_VERSION == "uv 0.9.8" ]] || fail "unexpected uv version: $UV_VERSION"
"$UV" python install --install-dir "$UV_PYTHON_INSTALL_DIR" --no-bin \
  "$MANAGED_PYTHON_VERSION"
MANAGED_PYTHON=$(
  "$UV" --managed-python --no-python-downloads python find \
    "$MANAGED_PYTHON_VERSION"
) || fail "cannot find managed Python $MANAGED_PYTHON_VERSION"
case $MANAGED_PYTHON in
  "$UV_PYTHON_INSTALL_DIR"/*) ;;
  *) fail "managed Python escaped install directory: $MANAGED_PYTHON" ;;
esac
"$MANAGED_PYTHON" -c \
  'import sys; assert sys.version_info[:3] == (3, 12, 7), sys.version' ||
  fail "managed Python is not 3.12.7: $MANAGED_PYTHON"
set -u

VENV_REUSABLE=0
if [[ -x $VENV/bin/python ]] &&
  "$VENV/bin/python" - "$MANAGED_PYTHON" <<'PY'
import pathlib
import sys
expected = pathlib.Path(sys.argv[1]).resolve(strict=True)
actual = pathlib.Path(sys._base_executable).resolve(strict=True)
if sys.version_info[:3] != (3, 12, 7) or actual != expected:
    raise SystemExit(1)
PY
then
  VENV_REUSABLE=1
fi
if [[ $VENV_REUSABLE == 0 ]]; then
  if [[ -e $VENV ]]; then
    VENV_BACKUP=$ROOT/venv.pre-managed.$ATTEMPT_TAG
    [[ ! -e $VENV_BACKUP ]] || fail "venv backup already exists: $VENV_BACKUP"
    mv "$VENV" "$VENV_BACKUP"
    printf 'Preserved previous venv at %s\n' "$VENV_BACKUP"
  fi
  "$MANAGED_PYTHON" -m venv "$VENV"
fi
"$VENV/bin/python" - "$MANAGED_PYTHON" <<'PY'
import pathlib
import sys
expected = pathlib.Path(sys.argv[1]).resolve(strict=True)
actual = pathlib.Path(sys._base_executable).resolve(strict=True)
assert sys.version_info[:3] == (3, 12, 7), sys.version
assert actual == expected, (actual, expected)
PY

# A ready marker authenticates the exact installed package set. Invalidate it
# before any in-place mutation so preflight cannot accept a partial or failed rerun.
if [[ -e $ENV_READY ]]; then
  ENV_READY_BACKUP=$VENV/.oprd-env-ready.pre-setup.$ATTEMPT_TAG.json
  [[ ! -e $ENV_READY_BACKUP ]] || fail "ready-marker backup already exists: $ENV_READY_BACKUP"
  mv "$ENV_READY" "$ENV_READY_BACKUP"
  printf 'Invalidated previous environment marker at %s\n' "$ENV_READY_BACKUP"
fi

# Torch 2.11 requires setuptools<82. Keep the packaging toolchain within that
# runtime constraint instead of leaving a dependency-inconsistent environment.
"$VENV/bin/python" -m pip install --index-url https://pypi.org/simple \
  --upgrade pip wheel "setuptools<82"
# vLLM first selects its compatible Torch build; then install the remaining floors.
"$VENV/bin/python" -m pip install --index-url https://pypi.org/simple "vllm>=0.8"
"$VENV/bin/python" -m pip install --index-url https://pypi.org/simple \
  -r requirements.txt
# Furong compute nodes do not expose the login node's FFmpeg modules. Use the
# platform wheel's bundled executable from the shared venv on every node.
"$VENV/bin/python" -m pip install --index-url https://pypi.org/simple \
  "imageio-ffmpeg==0.6.0"
"$VENV/bin/python" -m pip check
BUNDLED_FFMPEG=$(
  "$VENV/bin/python" -c \
    'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())'
) || fail "cannot locate bundled FFmpeg"
[[ -x $BUNDLED_FFMPEG ]] || fail "bundled FFmpeg is not executable: $BUNDLED_FFMPEG"
ln -sfn "$BUNDLED_FFMPEG" "$VENV/bin/ffmpeg"
export PATH="$VENV/bin:$PATH"

FFMPEG_PATH=$(command -v ffmpeg)
FFMPEG_OUTPUT=$(ffmpeg -version) || fail "ffmpeg -version failed"
FFMPEG_VERSION=${FFMPEG_OUTPUT%%$'\n'*}
[[ -n "$FFMPEG_VERSION" ]] || fail "ffmpeg returned no version string"

"$VENV/bin/python" - \
  "$CLUSTER_DIR/requirements.txt" "$ENV_READY" "$FFMPEG_PATH" \
  "$FFMPEG_VERSION" "$MANAGED_PYTHON" "$UV_VERSION" <<'PY'
import hashlib
import importlib.metadata
import json
import os
import pathlib
import sys
import tempfile

import datasets
import numpy
import safetensors
import torch
import transformers
import vllm

requirements = pathlib.Path(sys.argv[1]).resolve()
destination = pathlib.Path(sys.argv[2]).resolve()
ffmpeg_path = pathlib.Path(sys.argv[3]).resolve(strict=True)
ffmpeg_version = sys.argv[4]
managed_python = pathlib.Path(sys.argv[5]).resolve(strict=True)
uv_version = sys.argv[6]
if sys.version_info[:3] != (3, 12, 7):
    raise SystemExit(f"expected Python 3.12.7, got {sys.version}")
if pathlib.Path(sys._base_executable).resolve(strict=True) != managed_python:
    raise SystemExit("venv is not based on the configured managed Python")
if not torch.cuda.is_available():
    raise SystemExit("Torch cannot see CUDA in the setup allocation")
payload = {
    "schema_version": 1,
    "python": sys.version,
    "python_executable": sys.executable,
    "managed_python_executable": os.fspath(managed_python),
    "uv_version": uv_version,
    "cuda_available": True,
    "cuda_device": torch.cuda.get_device_name(0),
    "torch_cuda": torch.version.cuda,
    "ffmpeg_path": os.fspath(ffmpeg_path),
    "ffmpeg_version": ffmpeg_version,
    "requirements_sha256": hashlib.sha256(requirements.read_bytes()).hexdigest(),
    "packages": {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "numpy",
            "safetensors",
            "transformers",
            "datasets",
            "vllm",
            "setuptools",
            "imageio-ffmpeg",
        )
    },
}
destination.parent.mkdir(parents=True, exist_ok=True)
fd, temporary_name = tempfile.mkstemp(
    prefix=f".{destination.name}.", dir=destination.parent
)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_name, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
finally:
    pathlib.Path(temporary_name).unlink(missing_ok=True)
print(json.dumps(payload, sort_keys=True))
PY

printf '%s\n' "$FFMPEG_VERSION"
printf 'SETUP_OK marker=%s hf_home=%s\n' "$ENV_READY" "$HF_HOME"
