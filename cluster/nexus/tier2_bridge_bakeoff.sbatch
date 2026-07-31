#!/bin/bash
#SBATCH --job-name=oprd-t2-paper
#SBATCH --partition=cml-furongh
#SBATCH --account=cml-furongh
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:h200-sxm:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=72:00:00
#SBATCH --array=0-26%4
#SBATCH --output=/cmlscratch/%u/oprd-experiments-main/cluster/logs/t2_paper_%A_%a.out
#SBATCH --export=NONE

# Fixed Qwen3-4B -> Qwen3-1.7B paper matrix. Every task consumes the
# identity contract from one independently verified production bridge build.
set -euo pipefail

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ $# -eq 3 ]] ||
  fail "usage: tier2_bridge_bakeoff.sbatch BUILD_DIR ARTIFACT_DIR RESULT_NAMESPACE"

readonly RUN_USER=${SLURM_JOB_USER:?SLURM_JOB_USER is required}
readonly SCRATCH_ROOT="/cmlscratch/$RUN_USER"
readonly ROOT="$SCRATCH_ROOT/oprd-experiments-main"
readonly CLUSTER_DIR="$ROOT/cluster"
readonly NEXUS_DIR="$CLUSTER_DIR/nexus"
readonly PYTHON="$ROOT/venv/bin/python"
readonly ENV_READY="$ROOT/venv/.oprd-env-ready.json"
readonly BRIDGE_BUILD_INPUT=$1
readonly BRIDGE_ARTIFACT_INPUT=$2
readonly RESULT_NS=$3

set +u
source /etc/profile.d/modules.sh 2>/dev/null || fail "cannot initialize environment modules"
module use /opt/common/.modulefiles
module load cuda/12.8.1 || fail "cannot load cuda/12.8.1"
MODULE_CUDA_HOME=${CUDA_HOME:-}
set -u

readonly BRIDGE_RUNTIME_PATH="$ROOT/venv/bin:/opt/common/cuda/cuda-12.8.1/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/bin:/sbin"
readonly BRIDGE_CUDA_HOME=/opt/common/cuda/cuda-12.8.1
readonly BRIDGE_CUDA_NVCC="$BRIDGE_CUDA_HOME/bin/nvcc"
readonly BRIDGE_RUNTIME_LD_LIBRARY_PATH="$BRIDGE_CUDA_HOME/lib64"
export PATH="$BRIDGE_RUNTIME_PATH"
export COMPILER_PATH=/usr/bin
export CUDA_HOME="$BRIDGE_CUDA_HOME"
export FLASHINFER_NVCC="$BRIDGE_CUDA_NVCC"
export LD_LIBRARY_PATH="$BRIDGE_RUNTIME_LD_LIBRARY_PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="$SCRATCH_ROOT/hf-cache"
export XDG_CACHE_HOME="$SCRATCH_ROOT/vllm-cache"
export VLLM_CACHE_ROOT="$SCRATCH_ROOT/vllm-cache"
export TRITON_CACHE_DIR="$SCRATCH_ROOT/triton-cache"
export FLASHINFER_WORKSPACE_BASE="$SCRATCH_ROOT/flashinfer-workspace"
readonly SRUN_EXPORTS=PATH,COMPILER_PATH,CUDA_HOME,FLASHINFER_NVCC,LD_LIBRARY_PATH,PYTORCH_CUDA_ALLOC_CONF,HF_HOME,XDG_CACHE_HOME,VLLM_CACHE_ROOT,TRITON_CACHE_DIR,FLASHINFER_WORKSPACE_BASE

[[ "$MODULE_CUDA_HOME" == "$BRIDGE_CUDA_HOME" ]] ||
  fail "cuda/12.8.1 resolved unexpected CUDA_HOME: $MODULE_CUDA_HOME"
[[ -x "$PYTHON" ]] || fail "venv Python is missing or not executable: $PYTHON"
[[ -r "$ENV_READY" ]] || fail "environment-ready marker is missing: $ENV_READY"
command -v ffmpeg >/dev/null 2>&1 || fail "bundled FFmpeg is unavailable from the venv"
TRITON_LINKER=$(/usr/bin/gcc -print-prog-name=ld) || fail "cannot resolve GCC linker"
[[ "$TRITON_LINKER" == /usr/bin/ld* && -x "$TRITON_LINKER" ]] ||
  fail "GCC linker is not an executable under /usr/bin: $TRITON_LINKER"
readonly TRITON_LINKER
[[ -d "$BRIDGE_BUILD_INPUT" && ! -L "$BRIDGE_BUILD_INPUT" ]] ||
  fail "BUILD_DIR must be a real directory: $BRIDGE_BUILD_INPUT"
[[ -d "$BRIDGE_ARTIFACT_INPUT" && ! -L "$BRIDGE_ARTIFACT_INPUT" ]] ||
  fail "ARTIFACT_DIR must be a real directory: $BRIDGE_ARTIFACT_INPUT"
SCRATCH_CANONICAL=$(realpath -e -- "$SCRATCH_ROOT") || fail "cannot resolve scratch root"
BRIDGE_BUILD_DIR=$(realpath -e -- "$BRIDGE_BUILD_INPUT") || fail "cannot resolve BUILD_DIR"
BRIDGE_ARTIFACT=$(realpath -e -- "$BRIDGE_ARTIFACT_INPUT") || fail "cannot resolve ARTIFACT_DIR"
readonly SCRATCH_CANONICAL BRIDGE_BUILD_DIR BRIDGE_ARTIFACT
case "$BRIDGE_BUILD_DIR/" in
  "$SCRATCH_CANONICAL/"*) ;;
  *) fail "resolved BUILD_DIR is outside scratch: $BRIDGE_BUILD_DIR" ;;
esac
case "$BRIDGE_ARTIFACT/" in
  "$SCRATCH_CANONICAL/"*) ;;
  *) fail "resolved ARTIFACT_DIR is outside scratch: $BRIDGE_ARTIFACT" ;;
esac
[[ "$RESULT_NS" =~ ^[a-z0-9][a-z0-9._-]{5,79}$ ]] ||
  fail "RESULT_NAMESPACE must be 6-80 lowercase letters, digits, dots, underscores, or hyphens"

cd "$CLUSTER_DIR"
readonly OPS="$NEXUS_DIR/oprd_bridge_ops.py"
"$PYTHON" "$OPS" check-environment \
  --marker "$ENV_READY" \
  --requirements-file "$CLUSTER_DIR/requirements.txt" \
  --python "$PYTHON" \
  --scratch "$SCRATCH_ROOT" \
  --runtime-path "$BRIDGE_RUNTIME_PATH" \
  --cuda-home "$BRIDGE_CUDA_HOME" \
  --cuda-nvcc "$BRIDGE_CUDA_NVCC" \
  --runtime-ld-library-path "$BRIDGE_RUNTIME_LD_LIBRARY_PATH" \
  --flashinfer-workspace-base "$FLASHINFER_WORKSPACE_BASE" ||
  fail "authenticated environment validation failed"
"$PYTHON" "$OPS" validate-published \
  --build-dir "$BRIDGE_BUILD_DIR" \
  --artifact "$BRIDGE_ARTIFACT" ||
  fail "published bridge artifact validation failed"

readonly CONTRACT="$BRIDGE_BUILD_DIR/contract.json"
readonly VERIFY="$BRIDGE_BUILD_DIR/verify.json"
[[ -r "$CONTRACT" ]] || fail "bridge contract is missing: $CONTRACT"
[[ -r "$VERIFY" ]] || fail "bridge verification report is missing: $VERIFY"

CONTRACT_METADATA=$("$PYTHON" - "$CONTRACT" <<'PY'
import json
import pathlib
import sys

contract = json.loads(pathlib.Path(sys.argv[1]).read_text())
identity = contract["identity"]
values = [
    identity["student_model_id"],
    identity["student_model_revision"],
    identity["teacher_model_id"],
    identity["teacher_model_revision"],
    identity["tokenizer_id"],
    identity["tokenizer_revision"],
    identity["dataset_id"],
    identity["dataset_revision"],
    identity["dataset_split"],
    identity["prompt_key"],
    contract["stage1_contract_sha256"],
]
if any(not isinstance(value, str) or not value for value in values):
    raise SystemExit("bridge contract has an empty identity field")
print("\n".join(values))
PY
) || fail "bridge contract validation failed"
mapfile -t CONTRACT_VALUES <<<"$CONTRACT_METADATA"
[[ ${#CONTRACT_VALUES[@]} -eq 11 ]] || fail "unexpected bridge contract field count"

readonly TEACHER=Qwen/Qwen3-4B
readonly STUDENT=Qwen/Qwen3-1.7B-Base
readonly PAIR=qwen3_4b_to_1p7b
[[ "${CONTRACT_VALUES[0]}" == "$STUDENT" ]] || fail "contract student does not match Tier-2 pair"
[[ "${CONTRACT_VALUES[2]}" == "$TEACHER" ]] || fail "contract teacher does not match Tier-2 pair"
readonly PROMPTS=${CONTRACT_VALUES[6]}
EXPECTED_PRODUCTION_CONTRACT_SHA256=$("$PYTHON" -c \
  'from oprd_bridge import stage1_contract_sha256; print(stage1_contract_sha256())') ||
  fail "could not compute the production Stage-1 contract digest"
readonly EXPECTED_PRODUCTION_CONTRACT_SHA256
[[ "${CONTRACT_VALUES[10]}" == "$EXPECTED_PRODUCTION_CONTRACT_SHA256" ]] ||
  fail "production Tier-2 refuses a non-production Stage-1 contract"

BRIDGE_MANIFEST_SHA256=$("$PYTHON" - "$VERIFY" <<'PY'
import json
import pathlib
import sys

verify = json.loads(pathlib.Path(sys.argv[1]).read_text())
if verify.get("status") != "ok":
    raise SystemExit("verify.json does not report status=ok")
digest = verify.get("artifact_manifest_sha256")
if not isinstance(digest, str) or len(digest) != 64:
    raise SystemExit("verify.json has no valid artifact manifest SHA-256")
print(digest)
PY
) || fail "bridge verification report validation failed"
readonly BRIDGE_MANIFEST_SHA256

readonly ARMS=(
  oprd_bridge
  linear_map_mse
  minilmv2_selfrel
  ln_dist
  soft_knn_kl
  cka_gram
  depth_gram
  opd_topk
  untrained_baseline
)
readonly SEEDS=(0 1 2)
readonly N_ARMS=${#ARMS[@]}
readonly TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
(( TASK_ID >= 0 && TASK_ID < N_ARMS * ${#SEEDS[@]} )) || fail "array task is out of range: $TASK_ID"
readonly ARM=${ARMS[$((TASK_ID % N_ARMS))]}
readonly SEED=${SEEDS[$((TASK_ID / N_ARMS))]}
git -C "$ROOT" diff --quiet --exit-code || fail "tracked source tree has unstaged changes"
git -C "$ROOT" diff --cached --quiet --exit-code || fail "tracked source tree has staged changes"
SOURCE_REVISION=$(git -C "$ROOT" rev-parse HEAD) || fail "could not resolve deployed source revision"
readonly SOURCE_REVISION
readonly OUT="$CLUSTER_DIR/runs/${RESULT_NS}_${PAIR}_${ARM}_s${SEED}"

case "$OUT/" in
  "$CLUSTER_DIR/runs/"*) ;;
  *) fail "Tier-2 output escaped the runs directory: $OUT" ;;
esac
mkdir "$OUT" || fail "Tier-2 output already exists; choose a fresh namespace: $OUT"

readonly COMMON_ARGS=(
  --teacher "$TEACHER"
  --student "$STUDENT"
  --prompts "$PROMPTS"
  --student-revision "${CONTRACT_VALUES[1]}"
  --teacher-revision "${CONTRACT_VALUES[3]}"
  --tokenizer "${CONTRACT_VALUES[4]}"
  --tokenizer-revision "${CONTRACT_VALUES[5]}"
  --dataset-revision "${CONTRACT_VALUES[7]}"
  --dataset-split "${CONTRACT_VALUES[8]}"
  --prompt-key "${CONTRACT_VALUES[9]}"
  --seed "$SEED"
  --max-tokens 24000
  --eval aime24,aime25
  --eval-avg-at 16
  --run-label "$ARM"
  --result-namespace "$RESULT_NS"
  --source-revision "$SOURCE_REVISION"
  --out "$OUT"
)

if [[ "$ARM" == untrained_baseline ]]; then
  srun --export="$SRUN_EXPORTS" "$PYTHON" train_harness.py \
    --loss opd_topk \
    --steps 0 \
    "${COMMON_ARGS[@]}"
  exit 0
fi

BRIDGE_ARGS=()
if [[ "$ARM" == oprd_bridge ]]; then
  BRIDGE_ARGS=(
    --bridge-artifact "$BRIDGE_ARTIFACT"
    --bridge-manifest-sha256 "$BRIDGE_MANIFEST_SHA256"
    --bridge-stage1-contract-sha256 "${CONTRACT_VALUES[10]}"
  )
fi

srun --export="$SRUN_EXPORTS" "$PYTHON" train_harness.py \
  --loss "$ARM" \
  --n-prompts 4000 \
  --supervised-layers all \
  --steps 2000 \
  --batch-tokens 16384 \
  --lr 1e-5 \
  --rollout-tokens 16000 \
  --supervise-last-k 2000 \
  --require-finished \
  --diagnostics calibrated_mknn,cka,dist_n \
  "${COMMON_ARGS[@]}" \
  "${BRIDGE_ARGS[@]}"
