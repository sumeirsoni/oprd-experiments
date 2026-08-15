#!/bin/bash
#SBATCH --job-name=oprd-official-task0
#SBATCH --partition=cml-furongh
#SBATCH --account=cml-furongh
#SBATCH --qos=cml-high_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200-sxm:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=72:00:00
#SBATCH --output=/cmlscratch/%u/oprd-experiments-main/cluster/logs/oprd_official_task0_%j.out
#SBATCH --export=NONE

set -euo pipefail

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ $# -eq 2 ]] || fail "usage: task0_official_stage2.sbatch RESULT_NAMESPACE PROJECT_REVISION"
readonly RESULT_NAMESPACE=$1
readonly PROJECT_REVISION=$2
[[ $RESULT_NAMESPACE =~ ^[a-z0-9][a-z0-9._-]{5,79}$ ]] ||
  fail "RESULT_NAMESPACE must be 6-80 lowercase letters, digits, dots, underscores, or hyphens"
[[ $PROJECT_REVISION =~ ^[0-9a-f]{40}$ ]] || fail "PROJECT_REVISION must be a full Git commit"
[[ $RESULT_NAMESPACE == official-task0-4h200-prod-r1 ]] ||
  fail "RESULT_NAMESPACE is not the one authorized production namespace"

readonly RUN_USER=${SLURM_JOB_USER:?SLURM_JOB_USER is required}
[[ ${SLURM_NTASKS:-} == 1 ]] || fail "production allocation must contain exactly one task"
[[ ${SLURM_JOB_NUM_NODES:-} == 1 ]] || fail "production allocation must contain exactly one node"
[[ ${SLURM_CPUS_PER_TASK:-} == 16 ]] || fail "production allocation must provide exactly 16 CPUs"
readonly SCRATCH_ROOT=/cmlscratch/$RUN_USER
readonly PROJECT_ROOT=$SCRATCH_ROOT/oprd-experiments-main
readonly OFFICIAL_ROOT=$SCRATCH_ROOT/oprd-official-stage2
readonly SOURCE=$OFFICIAL_ROOT/source/OPRD
readonly ENVIRONMENT=$OFFICIAL_ROOT/env-a63236c
readonly PYTHON=$ENVIRONMENT/bin/python
readonly READY_MARKER=$ENVIRONMENT/.oprd-official-ready.json
readonly CONTRACT=$PROJECT_ROOT/cluster/official_stage2/task0_contract.json
readonly CONTRACT_TOOL=$PROJECT_ROOT/cluster/official_stage2/contract.py
readonly EXECUTION=$PROJECT_ROOT/cluster/official_stage2/task0_execution_4h200.json
readonly VALIDATOR=$PROJECT_ROOT/cluster/official_stage2/validate_task0.py
readonly CAPTURE_BIN=$PROJECT_ROOT/cluster/official_stage2/capture_bin
readonly STAGE1_ARTIFACT=$PROJECT_ROOT/cluster/artifacts/oprd_bridge/qwen3_4b_to_1p7b/production/rank8_all_layers
readonly BRIDGE_ADAPTER=$OFFICIAL_ROOT/bridge/rank8_all_layers
readonly PROJECTOR_BANK=$BRIDGE_ADAPTER/ps_bank.pt
readonly MODEL_DIR=$OFFICIAL_ROOT/models
readonly RESULT_PARENT=$OFFICIAL_ROOT/runs
readonly RESULT_DIRECTORY=$RESULT_PARENT/$RESULT_NAMESPACE
readonly RESOLVED_STAGING=$OFFICIAL_ROOT/.resolved-${SLURM_JOB_ID}.json
readonly EXECUTION_STAGING=$OFFICIAL_ROOT/.execution-${SLURM_JOB_ID}.json
readonly PREFLIGHT_STAGING=$OFFICIAL_ROOT/.preflight-${SLURM_JOB_ID}.json
readonly HOME=$OFFICIAL_ROOT/home
readonly CUDA_HOME=/opt/common/cuda/cuda-12.8.1
readonly FLASHINFER_NVCC=$CUDA_HOME/bin/nvcc
readonly LD_LIBRARY_PATH=$CUDA_HOME/lib64
readonly RUNTIME_PATH=$ENVIRONMENT/bin:$CUDA_HOME/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/bin:/sbin
readonly HF_HOME=$OFFICIAL_ROOT/cache/huggingface
readonly VLLM_CACHE_ROOT=$OFFICIAL_ROOT/cache/vllm
readonly TRITON_CACHE_DIR=$OFFICIAL_ROOT/cache/triton
readonly FLASHINFER_WORKSPACE_BASE=$OFFICIAL_ROOT/cache/flashinfer
readonly RAY_TMPDIR=$OFFICIAL_ROOT/cache/ray/$SLURM_JOB_ID
readonly OUTLINES_CACHE_DIR=$OFFICIAL_ROOT/cache/outlines/$SLURM_JOB_ID
readonly PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
readonly PYTHONDONTWRITEBYTECODE=1

[[ -x $PYTHON ]] || fail "official environment is missing: $PYTHON"
[[ -r $EXECUTION ]] || fail "approved execution manifest is missing"
[[ -x $CAPTURE_BIN/python3 ]] || fail "Hydra capture wrapper is missing"
[[ -r $READY_MARKER ]] || fail "official environment marker is missing"
[[ -d $SOURCE && ! -L $SOURCE ]] || fail "official source is missing"
[[ -d $STAGE1_ARTIFACT && ! -L $STAGE1_ARTIFACT ]] || fail "canonical Stage-1 bridge is missing"
[[ -r $PROJECTOR_BANK ]] || fail "official projector bank is missing"
[[ -d $MODEL_DIR/Qwen3-1.7B-Base && ! -L $MODEL_DIR/Qwen3-1.7B-Base ]] ||
  fail "pinned student model is missing"
[[ -d $MODEL_DIR/Qwen3-4B && ! -L $MODEL_DIR/Qwen3-4B ]] ||
  fail "pinned teacher model is missing"
EXPECTED_UPDATES=$(
  "$PYTHON" - "$CONTRACT" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["training_data"]["expected_updates"])
PY
)
readonly EXPECTED_UPDATES
[[ $EXPECTED_UPDATES =~ ^[1-9][0-9]*$ ]] || fail "invalid expected update count"
[[ ! -e $RESULT_DIRECTORY ]] || fail "result namespace already exists: $RESULT_DIRECTORY"
[[ ! -e $RESOLVED_STAGING && ! -e $EXECUTION_STAGING && ! -e $PREFLIGHT_STAGING ]] ||
  fail "stale task-0 staging metadata exists"
mkdir -p "$RESULT_PARENT" "$HOME" "$RAY_TMPDIR" "$OUTLINES_CACHE_DIR"

[[ $(git -C "$PROJECT_ROOT" rev-parse HEAD) == "$PROJECT_REVISION" ]] ||
  fail "deployed project revision does not match PROJECT_REVISION"
[[ -z $(git -C "$PROJECT_ROOT" status --porcelain --untracked-files=all -- \
  cluster/official_stage2 cluster/oprd_bridge.py cluster/oprd_atomic_directory.py \
  cluster/nexus/task0_official_stage2.sbatch) ]] ||
  fail "project-side official task-0 trust files are not clean and tracked"

git -C "$SOURCE" diff --quiet --exit-code || fail "official source has unstaged changes"
git -C "$SOURCE" diff --cached --quiet --exit-code || fail "official source has staged changes"
[[ -z $(git -C "$SOURCE" status --porcelain --untracked-files=all) ]] ||
  fail "official source has untracked files"

"$PYTHON" "$CONTRACT_TOOL" --contract "$CONTRACT" check-environment >/dev/null
"$PYTHON" "$CONTRACT_TOOL" --contract "$CONTRACT" emit-resolved > "$RESOLVED_STAGING"
"$PYTHON" "$CONTRACT_TOOL" --contract "$CONTRACT" validate-resolved \
  --resolved "$RESOLVED_STAGING" >/dev/null
"$PYTHON" "$CONTRACT_TOOL" --contract "$CONTRACT" emit-execution \
  --execution "$EXECUTION" > "$EXECUTION_STAGING"

export PATH=$RUNTIME_PATH
export HOME PATH CUDA_HOME FLASHINFER_NVCC LD_LIBRARY_PATH HF_HOME
export VLLM_CACHE_ROOT TRITON_CACHE_DIR FLASHINFER_WORKSPACE_BASE
export RAY_TMPDIR OUTLINES_CACHE_DIR PYTORCH_CUDA_ALLOC_CONF PYTHONDONTWRITEBYTECODE
readonly SRUN_EXPORTS=HOME,PATH,CUDA_HOME,FLASHINFER_NVCC,LD_LIBRARY_PATH,HF_HOME,VLLM_CACHE_ROOT,TRITON_CACHE_DIR,FLASHINFER_WORKSPACE_BASE,RAY_TMPDIR,OUTLINES_CACHE_DIR,PYTORCH_CUDA_ALLOC_CONF,PYTHONDONTWRITEBYTECODE,SLURM_JOB_ID,SLURM_JOB_NUM_NODES,SLURM_JOB_NODELIST,CUDA_VISIBLE_DEVICES

srun --nodes=1 --ntasks=1 --export="$SRUN_EXPORTS" "$PYTHON" "$VALIDATOR" --contract "$CONTRACT" preflight \
  --official-root "$OFFICIAL_ROOT" \
  --project-root "$PROJECT_ROOT" \
  --project-revision "$PROJECT_REVISION" \
  --stage1-artifact "$STAGE1_ARTIFACT" \
  --resolved "$RESOLVED_STAGING" \
  --resolved-execution "$EXECUTION_STAGING" \
  --execution "$EXECUTION" \
  --result-directory "$RESULT_DIRECTORY" > "$PREFLIGHT_STAGING"

mkdir "$RESULT_DIRECTORY" || fail "cannot reserve result namespace"
mv "$RESOLVED_STAGING" "$RESULT_DIRECTORY/resolved-scientific-config.json"
mv "$EXECUTION_STAGING" "$RESULT_DIRECTORY/resolved-execution-topology.json"
mv "$PREFLIGHT_STAGING" "$RESULT_DIRECTORY/preflight.json"
readonly PROJECT_PATH=$RESULT_DIRECTORY
readonly BASH_ENV=$RESULT_DIRECTORY/bash-env
printf 'set -e\nset -o pipefail\nunset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES\n' > "$BASH_ENV"
readonly OPRD_REAL_PYTHON=$PYTHON
readonly OPRD_HYDRA_CAPTURE=$RESULT_DIRECTORY/actual-hydra.yaml
readonly OPRD_ARGV_CAPTURE=$RESULT_DIRECTORY/actual-python-argv.json
readonly OPRD_CONTRACT_TOOL=$CONTRACT_TOOL
readonly OPRD_CONTRACT=$CONTRACT
readonly OPRD_OFFICIAL_ROOT=$OFFICIAL_ROOT
readonly OPRD_EXECUTION_DEVIATION=$EXECUTION
readonly RUNTIME_LOG=$RESULT_DIRECTORY/official-runtime.log
export PATH=$CAPTURE_BIN:$RUNTIME_PATH
export OPRD_REAL_PYTHON OPRD_HYDRA_CAPTURE OPRD_ARGV_CAPTURE
export OPRD_CONTRACT_TOOL OPRD_CONTRACT OPRD_OFFICIAL_ROOT OPRD_EXECUTION_DEVIATION BASH_ENV

export MODEL_DIR PROJECT_PATH
export PROJECT_NAME=OPRD-Bridge-Official-Task0
export TRAIN_DATASET_NAME=DAPO-Math-17k-official-task0
export MINI_BATCH_SIZE=8
export TEMPERATURE=1.0
export TEACHER_TEMPERATURE=1.0
export REPETITION_PENALTY=1.0
export N_RESPONSES=2
export LOG_PROB_TOP_K=0
export TOP_K_STRATEGY=only_stu
export REWARD_WEIGHT_MODE=student_p
export LR_SCHEDULER=constant
export USE_KL=False
export ENABLE_FORMAT_REWARD=False
export MODEL_DTYPE=fp32
export IS_PLOT=False
export LOSS_AGG_MODE=token-mean
export ENABLE_ACTIVATION_OFFLOAD=False
export USE_REP_DISTILLATION=True
export REP_DISTILLATION_ONLY=True
export REP_DISTILLATION_COEF=10.0
export REP_DISTILLATION_POSITIONS=last_k
export REP_DISTILLATION_LAST_K=2000
export REP_DISTILLATION_FIRST_K=2000
export REP_DISTILLATION_LAYERS=all
export REP_PROJECTOR_MODE=low_rank
export REP_LOW_RANK=8
export REP_LOW_RANK_INIT_CHECKPOINT=$PROJECTOR_BANK
export REP_FREEZE_PS=True
export USE_ATT_DISTILLATION=False
export ATT_DISTILLATION_COEF=1.0
export ATT_DISTILLATION_LAYERS=all
export ATT_DISTILLATION_POSITIONS=last_k
export ATT_DISTILLATION_LAST_K=2000
export ATT_DISTILLATION_FIRST_K=2000
export ATT_DISTILLATION_MAX_KEY_LEN=4096
export ATT_DISTILLATION_LOSS=mse
export ATT_DISTILLATION_TEMPERATURE=1.0

readonly OFFICIAL_EXPORTS=$SRUN_EXPORTS,BASH_ENV,OPRD_REAL_PYTHON,OPRD_HYDRA_CAPTURE,OPRD_ARGV_CAPTURE,OPRD_CONTRACT_TOOL,OPRD_CONTRACT,OPRD_OFFICIAL_ROOT,OPRD_EXECUTION_DEVIATION,MODEL_DIR,PROJECT_PATH,PROJECT_NAME,TRAIN_DATASET_NAME,MINI_BATCH_SIZE,TEMPERATURE,TEACHER_TEMPERATURE,REPETITION_PENALTY,N_RESPONSES,LOG_PROB_TOP_K,TOP_K_STRATEGY,REWARD_WEIGHT_MODE,LR_SCHEDULER,USE_KL,ENABLE_FORMAT_REWARD,MODEL_DTYPE,IS_PLOT,LOSS_AGG_MODE,ENABLE_ACTIVATION_OFFLOAD,USE_REP_DISTILLATION,REP_DISTILLATION_ONLY,REP_DISTILLATION_COEF,REP_DISTILLATION_POSITIONS,REP_DISTILLATION_LAST_K,REP_DISTILLATION_FIRST_K,REP_DISTILLATION_LAYERS,REP_PROJECTOR_MODE,REP_LOW_RANK,REP_LOW_RANK_INIT_CHECKPOINT,REP_FREEZE_PS,USE_ATT_DISTILLATION,ATT_DISTILLATION_COEF,ATT_DISTILLATION_LAYERS,ATT_DISTILLATION_POSITIONS,ATT_DISTILLATION_LAST_K,ATT_DISTILLATION_FIRST_K,ATT_DISTILLATION_MAX_KEY_LEN,ATT_DISTILLATION_LOSS,ATT_DISTILLATION_TEMPERATURE

srun --nodes=1 --ntasks=1 --export="$OFFICIAL_EXPORTS" "$PYTHON" - <<'PY'
import torch
assert torch.cuda.device_count() == 4, torch.cuda.device_count()
devices = [torch.cuda.get_device_name(index) for index in range(4)]
assert all("H200" in device for device in devices), devices
print({"visible_gpus": 4, "devices": devices})
PY
srun --nodes=1 --ntasks=1 --export="$OFFICIAL_EXPORTS" bash -c \
  'cd "$1/verl" && exec bash "$1/low_rank_rep_distillation.sh"' \
  bash "$SOURCE" 2>&1 | tee "$RUNTIME_LOG"

CHECKPOINT=$(
  "$PYTHON" - "$RESULT_DIRECTORY" "$EXPECTED_UPDATES" <<'PY'
import pathlib, sys
step = int(sys.argv[2])
matches = list(pathlib.Path(sys.argv[1]).rglob(f"global_step_{step}"))
if len(matches) != 1:
    raise SystemExit(f"expected one global_step_{step} checkpoint, found {len(matches)}")
print(matches[0])
PY
)
readonly CHECKPOINT
"$PYTHON" "$VALIDATOR" --contract "$CONTRACT" shards \
  --actor "$CHECKPOINT/actor" --world-size 4 > "$RESULT_DIRECTORY/fsdp-shards.json"
readonly MERGED_CHECKPOINT=$RESULT_DIRECTORY/merged-hf
[[ ! -e $MERGED_CHECKPOINT ]] || fail "merged checkpoint destination already exists"
"$PYTHON" -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$CHECKPOINT/actor" \
  --target_dir "$MERGED_CHECKPOINT"

"$PYTHON" "$VALIDATOR" --contract "$CONTRACT" accept \
  --run-directory "$RESULT_DIRECTORY" \
  --project-root "$PROJECT_ROOT" \
  --execution "$EXECUTION" > "$RESULT_DIRECTORY/acceptance.json"
printf 'ACCEPTED_OFFICIAL_TASK0\n' > "$RESULT_DIRECTORY/COMPLETE"
printf 'official task 0 accepted: %s\n' "$RESULT_DIRECTORY"
