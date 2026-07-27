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
set -euo pipefail

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

source /etc/profile.d/modules.sh 2>/dev/null || fail "cannot initialize environment modules"
module use /opt/local/stow/.modulefiles
module load Python3/3.12.3 || fail "cannot load Python3/3.12.3"
module use /opt/common/.modulefiles
module load cuda/12.8.1 || fail "cannot load cuda/12.8.1"
module load ffmpeg/7.1 || module load ffmpeg/6.1 || fail "cannot load a supported FFmpeg module"
command -v ffmpeg >/dev/null 2>&1 || fail "ffmpeg is unavailable after module load"
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' ||
  fail "module Python is not Python 3.12"

SCRATCH_ROOT=${SCRATCH_ROOT:-/cmlscratch/$USER}
ROOT=${ROOT:-$SCRATCH_ROOT/oprd-experiments}
CLUSTER_DIR=$ROOT/cluster
BRIDGE_HF_HOME=${BRIDGE_HF_HOME:-$SCRATCH_ROOT/hf-cache}
VENV=${VENV:-$ROOT/venv}
ENV_READY=$VENV/.oprd-env-ready.json

cd "$CLUSTER_DIR"
mkdir -p logs runs "$BRIDGE_HF_HOME"
export HF_HOME="$BRIDGE_HF_HOME"

if [[ ! -x $VENV/bin/python ]]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' ||
  fail "existing venv is not based on Python 3.12: $VENV"

"$VENV/bin/python" -m pip install -U pip setuptools wheel
# vLLM first selects its compatible Torch build; then install the remaining floors.
"$VENV/bin/python" -m pip install "vllm>=0.8"
"$VENV/bin/python" -m pip install -r requirements.txt

FFMPEG_PATH=$(command -v ffmpeg)
FFMPEG_OUTPUT=$(ffmpeg -version) || fail "ffmpeg -version failed"
FFMPEG_VERSION=${FFMPEG_OUTPUT%%$'\n'*}
[[ -n "$FFMPEG_VERSION" ]] || fail "ffmpeg returned no version string"

"$VENV/bin/python" - "$CLUSTER_DIR/requirements.txt" "$ENV_READY" "$FFMPEG_PATH" "$FFMPEG_VERSION" <<'PY'
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
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"expected Python 3.12, got {sys.version}")
if not torch.cuda.is_available():
    raise SystemExit("Torch cannot see CUDA in the setup allocation")
payload = {
    "schema_version": 1,
    "python": sys.version,
    "python_executable": sys.executable,
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
finally:
    pathlib.Path(temporary_name).unlink(missing_ok=True)
print(json.dumps(payload, sort_keys=True))
PY

printf '%s\n' "$FFMPEG_VERSION"
printf 'SETUP_OK marker=%s hf_home=%s\n' "$ENV_READY" "$HF_HOME"
