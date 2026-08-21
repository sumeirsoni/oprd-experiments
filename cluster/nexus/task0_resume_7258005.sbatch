#!/bin/bash
#SBATCH --job-name=oprd-task0-resume-7258005
#SBATCH --partition=cml-furongh
#SBATCH --account=cml-furongh
#SBATCH --qos=cml-high_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200-sxm:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=120:00:00
#SBATCH --output=/cmlscratch/%u/oprd-experiments-main/cluster/logs/oprd_task0_resume_7258005_%j.out
#SBATCH --export=NONE

set -euo pipefail
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ $# -eq 1 ]] || fail "usage: task0_resume_7258005.sbatch PROJECT_REVISION"
readonly PROJECT_REVISION=$1
[[ $PROJECT_REVISION =~ ^[0-9a-f]{40}$ ]] || fail "PROJECT_REVISION must be a full Git commit"
readonly RUN_USER=${SLURM_JOB_USER:?SLURM_JOB_USER is required}
[[ ${SLURM_ARRAY_JOB_ID:-} == "" && ${SLURM_ARRAY_TASK_ID:-} == "" ]] || fail "resume must not be an array job"
[[ ${SLURM_NTASKS:-} == 1 ]] || fail "resume allocation must contain exactly one task"
[[ ${SLURM_JOB_NUM_NODES:-} == 1 ]] || fail "resume allocation must contain exactly one node"
[[ ${SLURM_CPUS_PER_TASK:-} == 16 ]] || fail "resume allocation must provide exactly 16 CPUs"

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
readonly RESUME_MANIFEST=$PROJECT_ROOT/cluster/official_stage2/task0_resume_7258005.json
readonly VALIDATOR=$PROJECT_ROOT/cluster/official_stage2/validate_task0.py
readonly CAPTURE_BIN=$PROJECT_ROOT/cluster/official_stage2/capture_bin
readonly RESUME_INSTRUMENTATION=$PROJECT_ROOT/cluster/official_stage2/resume_instrumentation
readonly STAGE1_ARTIFACT=$PROJECT_ROOT/cluster/artifacts/oprd_bridge/qwen3_4b_to_1p7b/production/rank8_all_layers
readonly BRIDGE_ADAPTER=$OFFICIAL_ROOT/bridge/rank8_all_layers
readonly PROJECTOR_BANK=$BRIDGE_ADAPTER/ps_bank.pt
readonly MODEL_DIR=$OFFICIAL_ROOT/models
readonly RESULT_DIRECTORY=$OFFICIAL_ROOT/runs/official-task0-4h200-prod-r1
readonly CONTINUATION=$RESULT_DIRECTORY/continuations/resume-global-step-1600-r1
readonly SOURCE_CHECKPOINT=$RESULT_DIRECTORY/token_reward_direct_DAPO-Math-17k-official-task0_Qwen3-1.7B-Base_Qwen3-4B_16384-T_1.0-Tch_1.0-n_2-mbs_8-topk_0-topk_strategy_only_stu-rw_student_p-2026-08-16_02-58-29-OPD/global_step_1600
readonly CHECKPOINT_ROOT=$CONTINUATION/checkpoints
readonly FINAL_CHECKPOINT=$CHECKPOINT_ROOT/global_step_2239
readonly MERGED_CHECKPOINT=$CONTINUATION/merged-hf
readonly PREFLIGHT_STAGING=$RESULT_DIRECTORY/continuations/.resume-preflight-${SLURM_JOB_ID}.json
readonly RESUME_STAGING=$RESULT_DIRECTORY/continuations/.resolved-resume-${SLURM_JOB_ID}.json
readonly HOME=$OFFICIAL_ROOT/home
readonly CUDA_HOME=/opt/common/cuda/cuda-12.8.1
readonly FLASHINFER_NVCC=$CUDA_HOME/bin/nvcc
readonly LD_LIBRARY_PATH=$CUDA_HOME/lib64
readonly RUNTIME_PATH=$ENVIRONMENT/bin:$CUDA_HOME/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/bin:/sbin
readonly HF_HOME=$OFFICIAL_ROOT/cache/huggingface
readonly VLLM_CACHE_ROOT=$OFFICIAL_ROOT/cache/vllm
readonly TRITON_CACHE_DIR=$OFFICIAL_ROOT/cache/triton
readonly FLASHINFER_WORKSPACE_BASE=$OFFICIAL_ROOT/cache/flashinfer
readonly RAY_TMPDIR=/tmp/oprd-ray-$RUN_USER-$SLURM_JOB_ID
readonly OUTLINES_CACHE_DIR=$OFFICIAL_ROOT/cache/outlines/$SLURM_JOB_ID
readonly PYTHONDONTWRITEBYTECODE=1

[[ -x $PYTHON && -x $CAPTURE_BIN/python3 ]] || fail "official runtime or capture wrapper is missing"
[[ -r $READY_MARKER && -r $CONTRACT && -r $EXECUTION && -r $RESUME_MANIFEST ]] || fail "authorized metadata is missing"
[[ -d $SOURCE && ! -L $SOURCE && -r $PROJECTOR_BANK ]] || fail "authenticated source or bridge is missing"
[[ -d $MODEL_DIR/Qwen3-1.7B-Base && ! -L $MODEL_DIR/Qwen3-1.7B-Base ]] || fail "student model is missing"
[[ -d $MODEL_DIR/Qwen3-4B && ! -L $MODEL_DIR/Qwen3-4B ]] || fail "teacher model is missing"
[[ -d $RESULT_DIRECTORY && ! -L $RESULT_DIRECTORY ]] || fail "original namespace is missing or is a symlink"
[[ -d $SOURCE_CHECKPOINT && ! -L $SOURCE_CHECKPOINT ]] || fail "authorized source checkpoint is missing"
[[ ! -e $RESULT_DIRECTORY/COMPLETE && ! -e $RESULT_DIRECTORY/acceptance.json && ! -e $RESULT_DIRECTORY/merged-hf ]] || fail "original namespace is already accepted"
[[ ! -e $CONTINUATION && ! -e $PREFLIGHT_STAGING && ! -e $RESUME_STAGING ]] || fail "continuation or staging artifact already exists"
mkdir -p "$RESULT_DIRECTORY/continuations" "$HOME" "$RAY_TMPDIR" "$OUTLINES_CACHE_DIR"
trap 'rm -rf -- "$RAY_TMPDIR"; rm -f -- "$PREFLIGHT_STAGING" "$RESUME_STAGING"' EXIT

[[ $(git -C "$PROJECT_ROOT" rev-parse HEAD) == "$PROJECT_REVISION" ]] || fail "deployed project revision mismatch"
[[ -z $(git -C "$PROJECT_ROOT" status --porcelain --untracked-files=all -- cluster/official_stage2 cluster/oprd_bridge.py cluster/oprd_atomic_directory.py cluster/nexus/task0_resume_7258005.sbatch) ]] || fail "resume trust files are not clean and tracked"
git -C "$SOURCE" diff --quiet --exit-code || fail "official source has unstaged changes"
git -C "$SOURCE" diff --cached --quiet --exit-code || fail "official source has staged changes"
[[ -z $(git -C "$SOURCE" status --porcelain --untracked-files=all) ]] || fail "official source has untracked files"
df -Pk "$OFFICIAL_ROOT"; df -Pi "$OFFICIAL_ROOT"
"$PYTHON" "$CONTRACT_TOOL" --contract "$CONTRACT" check-environment >/dev/null
"$PYTHON" "$CONTRACT_TOOL" --contract "$CONTRACT" validate-execution --execution "$EXECUTION" >/dev/null
"$PYTHON" "$CONTRACT_TOOL" --contract "$CONTRACT" validate-resume --execution "$EXECUTION" --resume "$RESUME_MANIFEST" >/dev/null

# Read-only preflight authenticates every frozen source byte and exact dataloader state.
"$PYTHON" "$VALIDATOR" --contract "$CONTRACT" resume-preflight \
  --run-directory "$RESULT_DIRECTORY" \
  --project-root "$PROJECT_ROOT" \
  --project-revision "$PROJECT_REVISION" \
  --execution "$EXECUTION" \
  --resume "$RESUME_MANIFEST" > "$PREFLIGHT_STAGING"
"$PYTHON" "$CONTRACT_TOOL" --contract "$CONTRACT" emit-resume \
  --execution "$EXECUTION" --resume "$RESUME_MANIFEST" > "$RESUME_STAGING"
mkdir "$CONTINUATION" || fail "cannot reserve continuation directory"
"$PYTHON" "$VALIDATOR" --contract "$CONTRACT" resume-reservation \
  --run-directory "$RESULT_DIRECTORY" \
  --project-root "$PROJECT_ROOT" \
  --project-revision "$PROJECT_REVISION" \
  --execution "$EXECUTION" \
  --resume "$RESUME_MANIFEST" >/dev/null
mv "$PREFLIGHT_STAGING" "$CONTINUATION/resume-preflight.json"
mv "$RESUME_STAGING" "$CONTINUATION/resolved-resume-authorization.json"
mkdir "$CONTINUATION/logs" "$CONTINUATION/logs/terminal"

export PATH=$RUNTIME_PATH
export HOME PATH CUDA_HOME FLASHINFER_NVCC LD_LIBRARY_PATH HF_HOME VLLM_CACHE_ROOT TRITON_CACHE_DIR FLASHINFER_WORKSPACE_BASE RAY_TMPDIR OUTLINES_CACHE_DIR PYTHONDONTWRITEBYTECODE
readonly SRUN_EXPORTS=HOME,PATH,CUDA_HOME,FLASHINFER_NVCC,LD_LIBRARY_PATH,HF_HOME,VLLM_CACHE_ROOT,TRITON_CACHE_DIR,FLASHINFER_WORKSPACE_BASE,RAY_TMPDIR,OUTLINES_CACHE_DIR,PYTHONDONTWRITEBYTECODE,SLURM_JOB_ID,SLURM_JOB_NUM_NODES,SLURM_JOB_NODELIST,CUDA_VISIBLE_DEVICES
readonly PROJECT_PATH=$CONTINUATION
readonly BASH_ENV=$CONTINUATION/bash-env
printf 'set -e\nset -o pipefail\nunset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES\n' > "$BASH_ENV"
readonly OPRD_REAL_PYTHON=$PYTHON
readonly OPRD_HYDRA_CAPTURE=$CONTINUATION/actual-hydra.yaml
readonly OPRD_ARGV_CAPTURE=$CONTINUATION/actual-python-argv.json
readonly OPRD_CONTRACT_TOOL=$CONTRACT_TOOL
readonly OPRD_CONTRACT=$CONTRACT
readonly OPRD_OFFICIAL_ROOT=$OFFICIAL_ROOT
readonly OPRD_EXECUTION_DEVIATION=$EXECUTION
readonly OPRD_RESUME_AUTHORIZATION=task0-7258005
readonly OPRD_RESUME_FROM_PATH=$SOURCE_CHECKPOINT
readonly OPRD_RESUME_LOCAL_DIR=$CHECKPOINT_ROOT
readonly OPRD_RESUME_HYDRA_DIR=$CONTINUATION/hydra-run
readonly OPRD_RESUME_MANIFEST=$RESUME_MANIFEST
readonly OPRD_RESUME_NAMESPACE=$RESULT_DIRECTORY
readonly OPRD_RESUME_CONTINUATION=$CONTINUATION
readonly RUNTIME_LOG=$CONTINUATION/official-runtime.log
readonly TERMINAL_LOG=$CONTINUATION/logs/terminal/resume.log
readonly ORCHESTRATION_LOG=$CONTINUATION/logs/orchestration.log
readonly OPRD_RESUME_TERMINAL_LOG=$TERMINAL_LOG
readonly OPRD_RESUME_DATALOADER_EVIDENCE=$CONTINUATION/dataloader-restoration.json
readonly OPRD_RESUME_DATA_PATH=$SOURCE_CHECKPOINT/data.pt
readonly OPRD_RESUME_DATA_SHA256=8e8ff1e795907a551f475725ef8e34be3e31a68deffe2d535522196901f23407
readonly OPRD_RESUME_DATA_STATE='{"snapshot_step":1600,"samples_yielded":12800,"sampler_iter_yielded":1600,"num_workers":8,"iterator_finished":false}'
readonly PYTHONPATH=$RESUME_INSTRUMENTATION
export PATH=$CAPTURE_BIN:$RUNTIME_PATH
export OPRD_REAL_PYTHON OPRD_HYDRA_CAPTURE OPRD_ARGV_CAPTURE OPRD_CONTRACT_TOOL OPRD_CONTRACT OPRD_OFFICIAL_ROOT OPRD_EXECUTION_DEVIATION OPRD_RESUME_AUTHORIZATION OPRD_RESUME_FROM_PATH OPRD_RESUME_LOCAL_DIR OPRD_RESUME_HYDRA_DIR OPRD_RESUME_MANIFEST OPRD_RESUME_NAMESPACE OPRD_RESUME_CONTINUATION OPRD_RESUME_TERMINAL_LOG OPRD_RESUME_DATALOADER_EVIDENCE OPRD_RESUME_DATA_PATH OPRD_RESUME_DATA_SHA256 OPRD_RESUME_DATA_STATE PYTHONPATH BASH_ENV

export MODEL_DIR PROJECT_PATH PROJECT_NAME=OPRD-Bridge-Official-Task0 TRAIN_DATASET_NAME=DAPO-Math-17k-official-task0 MINI_BATCH_SIZE=8 TEMPERATURE=1.0 TEACHER_TEMPERATURE=1.0 REPETITION_PENALTY=1.0 N_RESPONSES=2 LOG_PROB_TOP_K=0 TOP_K_STRATEGY=only_stu REWARD_WEIGHT_MODE=student_p LR_SCHEDULER=constant USE_KL=False ENABLE_FORMAT_REWARD=False MODEL_DTYPE=fp32 IS_PLOT=False LOSS_AGG_MODE=token-mean ENABLE_ACTIVATION_OFFLOAD=False USE_REP_DISTILLATION=True REP_DISTILLATION_ONLY=True REP_DISTILLATION_COEF=10.0 REP_DISTILLATION_POSITIONS=last_k REP_DISTILLATION_LAST_K=2000 REP_DISTILLATION_FIRST_K=2000 REP_DISTILLATION_LAYERS=all REP_PROJECTOR_MODE=low_rank REP_LOW_RANK=8 REP_LOW_RANK_INIT_CHECKPOINT=$PROJECTOR_BANK REP_FREEZE_PS=True USE_ATT_DISTILLATION=False ATT_DISTILLATION_COEF=1.0 ATT_DISTILLATION_LAYERS=all ATT_DISTILLATION_POSITIONS=last_k ATT_DISTILLATION_LAST_K=2000 ATT_DISTILLATION_FIRST_K=2000 ATT_DISTILLATION_MAX_KEY_LEN=4096 ATT_DISTILLATION_LOSS=mse ATT_DISTILLATION_TEMPERATURE=1.0
readonly OFFICIAL_EXPORTS=$SRUN_EXPORTS,BASH_ENV,OPRD_REAL_PYTHON,OPRD_HYDRA_CAPTURE,OPRD_ARGV_CAPTURE,OPRD_CONTRACT_TOOL,OPRD_CONTRACT,OPRD_OFFICIAL_ROOT,OPRD_EXECUTION_DEVIATION,OPRD_RESUME_AUTHORIZATION,OPRD_RESUME_FROM_PATH,OPRD_RESUME_LOCAL_DIR,OPRD_RESUME_HYDRA_DIR,OPRD_RESUME_MANIFEST,OPRD_RESUME_NAMESPACE,OPRD_RESUME_CONTINUATION,OPRD_RESUME_TERMINAL_LOG,OPRD_RESUME_DATALOADER_EVIDENCE,OPRD_RESUME_DATA_PATH,OPRD_RESUME_DATA_SHA256,OPRD_RESUME_DATA_STATE,PYTHONPATH,MODEL_DIR,PROJECT_PATH,PROJECT_NAME,TRAIN_DATASET_NAME,MINI_BATCH_SIZE,TEMPERATURE,TEACHER_TEMPERATURE,REPETITION_PENALTY,N_RESPONSES,LOG_PROB_TOP_K,TOP_K_STRATEGY,REWARD_WEIGHT_MODE,LR_SCHEDULER,USE_KL,ENABLE_FORMAT_REWARD,MODEL_DTYPE,IS_PLOT,LOSS_AGG_MODE,ENABLE_ACTIVATION_OFFLOAD,USE_REP_DISTILLATION,REP_DISTILLATION_ONLY,REP_DISTILLATION_COEF,REP_DISTILLATION_POSITIONS,REP_DISTILLATION_LAST_K,REP_DISTILLATION_FIRST_K,REP_DISTILLATION_LAYERS,REP_PROJECTOR_MODE,REP_LOW_RANK,REP_LOW_RANK_INIT_CHECKPOINT,REP_FREEZE_PS,USE_ATT_DISTILLATION,ATT_DISTILLATION_COEF,ATT_DISTILLATION_LAYERS,ATT_DISTILLATION_POSITIONS,ATT_DISTILLATION_LAST_K,ATT_DISTILLATION_FIRST_K,ATT_DISTILLATION_MAX_KEY_LEN,ATT_DISTILLATION_LOSS,ATT_DISTILLATION_TEMPERATURE

srun --nodes=1 --ntasks=1 --export="$OFFICIAL_EXPORTS" "$PYTHON" - <<'PY'
import torch
assert torch.cuda.device_count() == 4, torch.cuda.device_count()
devices=[torch.cuda.get_device_name(i) for i in range(4)]; assert all("H200" in x for x in devices), devices
print({"visible_gpus":4,"devices":devices})
PY
srun --nodes=1 --ntasks=1 --export="$OFFICIAL_EXPORTS" bash -c 'cd "$1/verl" && exec bash "$1/low_rank_rep_distillation.sh"' bash "$SOURCE" 2>&1 | tee "$RUNTIME_LOG" "$ORCHESTRATION_LOG"

# Validate and merge only the exact authorized final checkpoint.
"$PYTHON" "$VALIDATOR" --contract "$CONTRACT" shards \
  --actor "$FINAL_CHECKPOINT/actor" --world-size 4 > "$CONTINUATION/fsdp-shards.json"
[[ ! -e $MERGED_CHECKPOINT ]] || fail "merge destination already exists"
"$PYTHON" -m verl.model_merger merge \
  --backend fsdp --local_dir "$FINAL_CHECKPOINT/actor" --target_dir "$MERGED_CHECKPOINT"

readonly CONTINUATION_ACCEPTANCE_STAGING=$CONTINUATION/.acceptance-${SLURM_JOB_ID}.json
readonly ROOT_ACCEPTANCE_STAGING=$RESULT_DIRECTORY/.acceptance-${SLURM_JOB_ID}.json
"$PYTHON" "$VALIDATOR" --contract "$CONTRACT" resume-accept \
  --run-directory "$RESULT_DIRECTORY" \
  --project-root "$PROJECT_ROOT" \
  --project-revision "$PROJECT_REVISION" \
  --execution "$EXECUTION" \
  --resume "$RESUME_MANIFEST" > "$CONTINUATION_ACCEPTANCE_STAGING"
"$PYTHON" "$VALIDATOR" publish-file \
  --source "$CONTINUATION_ACCEPTANCE_STAGING" \
  --destination "$CONTINUATION/acceptance.json" >/dev/null
rm -f -- "$CONTINUATION_ACCEPTANCE_STAGING"
"$PYTHON" "$VALIDATOR" --contract "$CONTRACT" resume-composite-accept \
  --run-directory "$RESULT_DIRECTORY" \
  --project-root "$PROJECT_ROOT" \
  --project-revision "$PROJECT_REVISION" \
  --execution "$EXECUTION" \
  --resume "$RESUME_MANIFEST" > "$ROOT_ACCEPTANCE_STAGING"
"$PYTHON" "$VALIDATOR" publish-file \
  --source "$ROOT_ACCEPTANCE_STAGING" \
  --destination "$RESULT_DIRECTORY/acceptance.json" >/dev/null
rm -f -- "$ROOT_ACCEPTANCE_STAGING"
"$PYTHON" "$VALIDATOR" publish-complete \
  --destination "$RESULT_DIRECTORY/COMPLETE" >/dev/null
printf 'official task 0 continuation accepted: %s\n' "$CONTINUATION"
