#!/bin/bash
#SBATCH --job-name=oprd-bridge-teacher
#SBATCH --partition=cml-furongh
#SBATCH --account=cml-furongh
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:h200-sxm:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/cmlscratch/%u/oprd-experiments/cluster/logs/oprd_bridge_teacher_%j.out

# Fit the frozen teacher PCA basis, then materialize immutable teacher targets.
set -euo pipefail

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
check_space() {
  local available_kb
  available_kb=$(df -Pk "$SCRATCH_ROOT" | awk 'NR == 2 {print $4}')
  [[ "$available_kb" =~ ^[0-9]+$ ]] || fail "could not determine free space on $SCRATCH_ROOT"
  (( available_kb >= MIN_FREE_GIB * 1024 * 1024 )) ||
    fail "$SCRATCH_ROOT has less than ${MIN_FREE_GIB} GiB free"
}

set +u
source /etc/profile.d/modules.sh 2>/dev/null || fail "cannot initialize environment modules"
module use /opt/common/.modulefiles
module load cuda/12.8.1 || fail "cannot load cuda/12.8.1"
MODULE_CUDA_HOME=${CUDA_HOME:-}
set -u

SCRATCH_ROOT=${SCRATCH_ROOT:-/cmlscratch/$USER}
ROOT=${ROOT:-$SCRATCH_ROOT/oprd-experiments}
CLUSTER_DIR=$ROOT/cluster
NEXUS_DIR=$CLUSTER_DIR/nexus
PYTHON=${PYTHON:-$ROOT/venv/bin/python}
BRIDGE_RUNTIME_PATH=${BRIDGE_RUNTIME_PATH:?submit_oprd_bridge.sh must export BRIDGE_RUNTIME_PATH}
BRIDGE_CUDA_HOME=${BRIDGE_CUDA_HOME:?submit_oprd_bridge.sh must export BRIDGE_CUDA_HOME}
BRIDGE_CUDA_NVCC=${BRIDGE_CUDA_NVCC:?submit_oprd_bridge.sh must export BRIDGE_CUDA_NVCC}
BRIDGE_RUNTIME_LD_LIBRARY_PATH=${BRIDGE_RUNTIME_LD_LIBRARY_PATH:?submit_oprd_bridge.sh must export BRIDGE_RUNTIME_LD_LIBRARY_PATH}
VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:?submit_oprd_bridge.sh must export VLLM_CACHE_ROOT}
TRITON_CACHE_DIR=${TRITON_CACHE_DIR:?submit_oprd_bridge.sh must export TRITON_CACHE_DIR}
FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WORKSPACE_BASE:?submit_oprd_bridge.sh must export FLASHINFER_WORKSPACE_BASE}
[[ "$MODULE_CUDA_HOME" == "$BRIDGE_CUDA_HOME" ]] || fail "cuda/12.8.1 resolved unexpected CUDA_HOME: $MODULE_CUDA_HOME"
export PATH="$BRIDGE_RUNTIME_PATH"
export CUDA_HOME="$BRIDGE_CUDA_HOME"
export FLASHINFER_NVCC="$BRIDGE_CUDA_NVCC"
export LD_LIBRARY_PATH="$BRIDGE_RUNTIME_LD_LIBRARY_PATH"
export VLLM_CACHE_ROOT TRITON_CACHE_DIR FLASHINFER_WORKSPACE_BASE
command -v ffmpeg >/dev/null 2>&1 || fail "bundled FFmpeg is unavailable from the venv"
OPS=$NEXUS_DIR/oprd_bridge_ops.py
BRIDGE_BUILDER=${BRIDGE_BUILDER:-$CLUSTER_DIR/build_oprd_bridge.py}
BRIDGE_BUILD_DIR=${BRIDGE_BUILD_DIR:?submit_oprd_bridge.sh must export BRIDGE_BUILD_DIR}
BRIDGE_ARTIFACT=${BRIDGE_ARTIFACT:?submit_oprd_bridge.sh must export BRIDGE_ARTIFACT}
BRIDGE_HF_HOME=${BRIDGE_HF_HOME:?submit_oprd_bridge.sh must export BRIDGE_HF_HOME}
BRIDGE_RUN_ID=${BRIDGE_RUN_ID:?submit_oprd_bridge.sh must export BRIDGE_RUN_ID}
JOB_RECORD=${JOB_RECORD:?submit_oprd_bridge.sh must export JOB_RECORD}
MIN_FREE_GIB=${TEACHER_MIN_FREE_GIB:-50}
ENV_READY=$ROOT/venv/.oprd-env-ready.json

[[ -d "$SCRATCH_ROOT" && -w "$SCRATCH_ROOT" ]] || fail "scratch is unavailable or not writable: $SCRATCH_ROOT"
[[ -d "$BRIDGE_BUILD_DIR" ]] || fail "bridge workspace is missing: $BRIDGE_BUILD_DIR"
[[ -x "$PYTHON" ]] || fail "venv Python is missing or not executable: $PYTHON"
[[ -f "$BRIDGE_BUILDER" ]] || fail "bridge builder is missing: $BRIDGE_BUILDER"
[[ -r "$ENV_READY" ]] || fail "environment-ready marker is missing: $ENV_READY"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' || fail "venv Python is not 3.12"
case "$BRIDGE_BUILD_DIR/" in "$SCRATCH_ROOT"/*) ;; *) fail "build directory must be on scratch: $BRIDGE_BUILD_DIR" ;; esac
case "$BRIDGE_ARTIFACT/" in "$SCRATCH_ROOT"/*) ;; *) fail "artifact must be on scratch: $BRIDGE_ARTIFACT" ;; esac
case "$BRIDGE_HF_HOME/" in "$SCRATCH_ROOT"/*) ;; *) fail "HF cache must be on scratch: $BRIDGE_HF_HOME" ;; esac
case "$VLLM_CACHE_ROOT/" in "$SCRATCH_ROOT"/*) ;; *) fail "vLLM cache must be on scratch: $VLLM_CACHE_ROOT" ;; esac
case "$TRITON_CACHE_DIR/" in "$SCRATCH_ROOT"/*) ;; *) fail "Triton cache must be on scratch: $TRITON_CACHE_DIR" ;; esac
case "$FLASHINFER_WORKSPACE_BASE/" in "$SCRATCH_ROOT"/*) ;; *) fail "FlashInfer workspace must be on scratch: $FLASHINFER_WORKSPACE_BASE" ;; esac
[[ ! -e "$BRIDGE_ARTIFACT" ]] || fail "refusing to overwrite canonical artifact: $BRIDGE_ARTIFACT"
"$PYTHON" "$OPS" check-runtime \
  --scratch "$SCRATCH_ROOT" \
  --path "$BRIDGE_BUILD_DIR" \
  --path "$BRIDGE_HF_HOME" \
  --path "$VLLM_CACHE_ROOT" \
  --path "$TRITON_CACHE_DIR" \
  --path "$FLASHINFER_WORKSPACE_BASE" \
  --artifact "$BRIDGE_ARTIFACT" \
  --run-id "$BRIDGE_RUN_ID" \
  --record "$JOB_RECORD" \
  --cuda-home "$BRIDGE_CUDA_HOME" \
  --cuda-nvcc "$BRIDGE_CUDA_NVCC" \
  --runtime-ld-library-path "$BRIDGE_RUNTIME_LD_LIBRARY_PATH" \
  --flashinfer-workspace-base "$FLASHINFER_WORKSPACE_BASE"
"$PYTHON" "$OPS" check-environment \
  --marker "$ENV_READY" \
  --requirements-file "$CLUSTER_DIR/requirements.txt" \
  --python "$PYTHON" \
  --scratch "$SCRATCH_ROOT" \
  --runtime-path "$BRIDGE_RUNTIME_PATH" \
  --cuda-home "$BRIDGE_CUDA_HOME" \
  --cuda-nvcc "$BRIDGE_CUDA_NVCC" \
  --runtime-ld-library-path "$BRIDGE_RUNTIME_LD_LIBRARY_PATH" \
  --flashinfer-workspace-base "$FLASHINFER_WORKSPACE_BASE"
"$PYTHON" "$OPS" check-toolchain --scratch "$SCRATCH_ROOT"
check_space

cd "$CLUSTER_DIR"
export HF_HOME="$BRIDGE_HF_HOME"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" \
  "$FLASHINFER_WORKSPACE_BASE"

srun "$PYTHON" "$BRIDGE_BUILDER" fit-teacher-pca \
  --workspace "$BRIDGE_BUILD_DIR"

srun "$PYTHON" "$BRIDGE_BUILDER" materialize-teacher-targets \
  --workspace "$BRIDGE_BUILD_DIR"

printf 'TEACHER_TARGETS_OK workspace=%s\n' "$BRIDGE_BUILD_DIR"
