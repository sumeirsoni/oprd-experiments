#!/bin/bash
#SBATCH --job-name=oprd-official-setup
#SBATCH --partition=cml-furongh
#SBATCH --account=cml-furongh
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:h200-sxm:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/cmlscratch/%u/oprd-experiments-main/cluster/logs/oprd_official_setup_%j.out
#SBATCH --export=NONE

set -euo pipefail

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

readonly RUN_USER=${SLURM_JOB_USER:?SLURM_JOB_USER is required}
readonly SCRATCH_ROOT=/cmlscratch/$RUN_USER
readonly PROJECT_ROOT=$SCRATCH_ROOT/oprd-experiments-main
readonly OFFICIAL_ROOT=$SCRATCH_ROOT/oprd-official-stage2
readonly SOURCE=$OFFICIAL_ROOT/source/OPRD
readonly SOURCE_MANIFEST=$OFFICIAL_ROOT/source/manifest.json
readonly ENVIRONMENT=$OFFICIAL_ROOT/env-a63236c
readonly READY_MARKER=$ENVIRONMENT/.oprd-official-ready.json
readonly LOCK_DIR=$OFFICIAL_ROOT/locks
readonly INPUT_REQUIREMENTS=$PROJECT_ROOT/cluster/official_stage2/official_requirements.in
readonly CONTRACT_TOOL=$PROJECT_ROOT/cluster/official_stage2/contract.py
readonly UV=$PROJECT_ROOT/.oprd-uv-bootstrap/bin/uv
readonly UV_PYTHON_INSTALL_DIR=$OFFICIAL_ROOT/python
readonly UV_CACHE_DIR=$OFFICIAL_ROOT/cache/uv
readonly HF_HOME=$OFFICIAL_ROOT/cache/huggingface
readonly VLLM_CACHE_ROOT=$OFFICIAL_ROOT/cache/vllm
readonly TRITON_CACHE_DIR=$OFFICIAL_ROOT/cache/triton
readonly FLASHINFER_WORKSPACE_BASE=$OFFICIAL_ROOT/cache/flashinfer
readonly OFFICIAL_CUDA_HOME=/opt/common/cuda/cuda-12.8.1
readonly PYTHON_VERSION=3.11.13
readonly BUILD_ROOT=$OFFICIAL_ROOT/build/verl-${SLURM_JOB_ID}
readonly FAILED_ROOT=$OFFICIAL_ROOT/failed/setup-${SLURM_JOB_ID}

[[ $# -eq 0 ]] || fail "setup_official_stage2.sbatch accepts no arguments"
[[ -x $UV ]] || fail "authenticated uv bootstrap is missing: $UV"
[[ -r $INPUT_REQUIREMENTS ]] || fail "official requirements input is missing"
[[ -d $SOURCE && ! -L $SOURCE ]] || fail "pinned official source is missing: $SOURCE"
[[ -r $SOURCE_MANIFEST ]] || fail "official source manifest is missing: $SOURCE_MANIFEST"
INPUT_SHA256=$(sha256sum "$INPUT_REQUIREMENTS" | cut -d' ' -f1)
readonly INPUT_SHA256
[[ ! -e $FAILED_ROOT ]] || fail "failed-attempt preservation root already exists: $FAILED_ROOT"
if [[ -e $ENVIRONMENT ]]; then
  PRESERVED_ENVIRONMENT=env-a63236c.partial
  if [[ -e $READY_MARKER ]]; then
    OLD_INPUT_SHA256=$(
      "$PROJECT_ROOT/venv/bin/python" - "$READY_MARKER" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["lock"]["input_sha256"])
PY
    )
    [[ $OLD_INPUT_SHA256 != "$INPUT_SHA256" ]] ||
      fail "ready official environment already matches current requirements"
    PRESERVED_ENVIRONMENT=env-a63236c.incompatible-$OLD_INPUT_SHA256
  fi
  mkdir -p "$FAILED_ROOT"
  mv "$ENVIRONMENT" "$FAILED_ROOT/$PRESERVED_ENVIRONMENT"
fi
if [[ -e $BUILD_ROOT ]]; then
  mkdir -p "$FAILED_ROOT"
  mv "$BUILD_ROOT" "$FAILED_ROOT/verl-build.partial"
fi

source /etc/profile.d/modules.sh 2>/dev/null || fail "cannot initialize environment modules"
module use /opt/common/.modulefiles
module load cuda/12.8.1 || fail "cannot load cuda/12.8.1"
[[ ${CUDA_HOME:-} == "$OFFICIAL_CUDA_HOME" ]] ||
  fail "cuda/12.8.1 resolved an unexpected CUDA_HOME"
readonly CUDA_HOME=$OFFICIAL_CUDA_HOME
readonly FLASHINFER_NVCC=$CUDA_HOME/bin/nvcc
readonly LD_LIBRARY_PATH=$CUDA_HOME/lib64
readonly PATH=$ENVIRONMENT/bin:$CUDA_HOME/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/bin:/sbin

export PATH CUDA_HOME FLASHINFER_NVCC LD_LIBRARY_PATH
export UV_CACHE_DIR UV_PYTHON_INSTALL_DIR HF_HOME VLLM_CACHE_ROOT TRITON_CACHE_DIR
export FLASHINFER_WORKSPACE_BASE
mkdir -p "$LOCK_DIR" "$UV_CACHE_DIR" "$HF_HOME" "$VLLM_CACHE_ROOT" \
  "$TRITON_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" "$UV_PYTHON_INSTALL_DIR"

"$PROJECT_ROOT/venv/bin/python" "$CONTRACT_TOOL" validate >/dev/null
"$UV" python install --install-dir "$UV_PYTHON_INSTALL_DIR" --no-bin "$PYTHON_VERSION"
MANAGED_PYTHON=$(
  "$UV" --managed-python --no-python-downloads python find "$PYTHON_VERSION"
) || fail "cannot find managed Python $PYTHON_VERSION"
readonly MANAGED_PYTHON

LOCK=$LOCK_DIR/requirements-$INPUT_SHA256.txt
readonly LOCK
if [[ -e $LOCK ]]; then
  mkdir -p "$FAILED_ROOT"
  mv "$LOCK" "$FAILED_ROOT/requirements.partial.txt"
fi
"$UV" pip compile \
  --python "$MANAGED_PYTHON" \
  --python-platform x86_64-unknown-linux-gnu \
  --generate-hashes \
  --output-file "$LOCK" \
  "$INPUT_REQUIREMENTS"

"$UV" venv --seed --python "$MANAGED_PYTHON" "$ENVIRONMENT"
"$UV" pip install \
  --python "$ENVIRONMENT/bin/python" \
  "torch==2.8.0" packaging ninja
UV_CONCURRENT_BUILDS=1 MAX_JOBS=4 "$UV" pip sync \
  --python "$ENVIRONMENT/bin/python" \
  --no-build-isolation \
  "$LOCK"
mkdir -p "$BUILD_ROOT"
cp -a "$SOURCE/verl/." "$BUILD_ROOT/"
"$ENVIRONMENT/bin/python" -m pip install --no-deps "$BUILD_ROOT"
"$ENVIRONMENT/bin/python" -m pip check

"$ENVIRONMENT/bin/python" - <<'PY'
import importlib
import torch
for name in (
    "ray", "vllm", "verl", "verl.trainer.main_ppo", "flash_attn",
    "flashinfer", "datasets", "pyarrow"
):
    importlib.import_module(name)
from transformers import AutoModelForVision2Seq
assert AutoModelForVision2Seq is not None
assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
print(torch.cuda.get_device_name(0))
PY

LOCK_SHA256=$(sha256sum "$LOCK" | cut -d' ' -f1)
SOURCE_MANIFEST_SHA256=$(sha256sum "$SOURCE_MANIFEST" | cut -d' ' -f1)
FREEZE=$ENVIRONMENT/pip-freeze.txt
"$ENVIRONMENT/bin/python" -m pip freeze --all | LC_ALL=C sort > "$FREEZE"
FREEZE_SHA256=$(sha256sum "$FREEZE" | cut -d' ' -f1)
readonly LOCK_SHA256 SOURCE_MANIFEST_SHA256 FREEZE FREEZE_SHA256

"$ENVIRONMENT/bin/python" - \
  "$READY_MARKER" "$LOCK" "$LOCK_SHA256" "$INPUT_SHA256" \
  "$SOURCE_MANIFEST" "$SOURCE_MANIFEST_SHA256" "$FREEZE" "$FREEZE_SHA256" <<'PY'
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import subprocess
import sys
import tempfile

marker = pathlib.Path(sys.argv[1])
lock = pathlib.Path(sys.argv[2]).resolve(strict=True)
source_manifest = pathlib.Path(sys.argv[5]).resolve(strict=True)
freeze = pathlib.Path(sys.argv[7]).resolve(strict=True)
packages = {
    name: importlib.metadata.version(name)
    for name in (
        "torch", "vllm", "ray", "flash-attn", "flashinfer-python",
        "transformers", "datasets", "pyarrow", "numpy", "tensordict"
    )
}
payload = {
    "schema_version": 1,
    "status": "ready",
    "python": {
        "executable": str(pathlib.Path(sys.executable).resolve(strict=True)),
        "version": platform.python_version(),
    },
    "packages": packages,
    "lock": {"path": str(lock), "sha256": sys.argv[3], "input_sha256": sys.argv[4]},
    "official_source_manifest": {
        "path": str(source_manifest), "sha256": sys.argv[6]
    },
    "pip_freeze": {"path": str(freeze), "sha256": sys.argv[8]},
    "cuda": {
        "torch_version": __import__("torch").version.cuda,
        "device": __import__("torch").cuda.get_device_name(0),
        "driver": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            text=True,
        ).strip(),
    },
}
marker.parent.mkdir(parents=True, exist_ok=True)
fd, temporary_name = tempfile.mkstemp(prefix=f".{marker.name}.", dir=marker.parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_name, marker)
finally:
    pathlib.Path(temporary_name).unlink(missing_ok=True)
PY

printf 'official Stage-2 environment ready: %s\n' "$READY_MARKER"
