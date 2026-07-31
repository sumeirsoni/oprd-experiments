#!/bin/bash
#SBATCH --job-name=oprd-official-smoke
#SBATCH --partition=cml-furongh
#SBATCH --account=cml-furongh
#SBATCH --qos=cml-high_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200-sxm:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=/cmlscratch/%u/oprd-experiments-main/cluster/logs/oprd_official_smoke_%j.out
#SBATCH --export=NONE

set -euo pipefail
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ $# -eq 2 ]] || fail "usage: official_stage2_smoke.sbatch RESULT_NAMESPACE ROWS"
readonly RESULT_NAMESPACE=$1
readonly ROWS=$2
[[ $RESULT_NAMESPACE =~ ^[a-z0-9][a-z0-9._-]{5,79}$ ]] || fail "invalid result namespace"
[[ $ROWS == 16 || $ROWS == 80 ]] || fail "ROWS must be 16 for smoke or 80 for canary"

readonly RUN_USER=${SLURM_JOB_USER:?SLURM_JOB_USER is required}
readonly SCRATCH_ROOT=/cmlscratch/$RUN_USER
readonly PROJECT_ROOT=$SCRATCH_ROOT/oprd-experiments-main
readonly OFFICIAL_ROOT=$SCRATCH_ROOT/oprd-official-stage2
readonly SOURCE=$OFFICIAL_ROOT/source/OPRD
readonly ENVIRONMENT=$OFFICIAL_ROOT/env-a63236c
readonly PYTHON=$ENVIRONMENT/bin/python
readonly MODEL_DIR=$OFFICIAL_ROOT/models
readonly PROJECTOR_BANK=$OFFICIAL_ROOT/bridge/rank8_all_layers/ps_bank.pt
readonly RUN_DIRECTORY=$OFFICIAL_ROOT/smokes/$RESULT_NAMESPACE
readonly WORK_DIRECTORY=$RUN_DIRECTORY/work
readonly PROJECT_PATH=$RUN_DIRECTORY/output
readonly HOME=$OFFICIAL_ROOT/home
readonly CUDA_HOME=/opt/common/cuda/cuda-12.8.1
readonly LD_LIBRARY_PATH=$CUDA_HOME/lib64
readonly PATH=$ENVIRONMENT/bin:$CUDA_HOME/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/bin:/sbin
readonly HF_HOME=$OFFICIAL_ROOT/cache/huggingface
readonly VLLM_CACHE_ROOT=$OFFICIAL_ROOT/cache/vllm
readonly TRITON_CACHE_DIR=$OFFICIAL_ROOT/cache/triton
readonly FLASHINFER_WORKSPACE_BASE=$OFFICIAL_ROOT/cache/flashinfer
readonly PYTHONDONTWRITEBYTECODE=1

[[ -x $PYTHON && -r $PROJECTOR_BANK ]] || fail "official environment or bridge is missing"
[[ ! -e $RUN_DIRECTORY ]] || fail "smoke namespace already exists: $RUN_DIRECTORY"
mkdir -p "$WORK_DIRECTORY/verl" "$WORK_DIRECTORY/datasets/test_data/AMC23" "$PROJECT_PATH"
ln -s "$SOURCE/verl/verl" "$WORK_DIRECTORY/verl/verl"
"$PYTHON" - "$SOURCE" "$WORK_DIRECTORY" "$ROWS" <<'PY'
import pathlib, sys
import pyarrow.parquet as pq
source, work, rows = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), int(sys.argv[3])
train = pq.read_table(source / "datasets/dapo-math-17k.parquet").slice(0, rows)
pq.write_table(train, work / "datasets/dapo-math-17k.parquet")
validation = pq.read_table(source / "datasets/test_data/AMC23/test.parquet").slice(0, 1)
pq.write_table(validation, work / "datasets/test_data/AMC23/test.parquet")
PY

export HOME PATH CUDA_HOME LD_LIBRARY_PATH HF_HOME VLLM_CACHE_ROOT
export TRITON_CACHE_DIR FLASHINFER_WORKSPACE_BASE PYTHONDONTWRITEBYTECODE
export MODEL_DIR PROJECT_PATH
export PROJECT_NAME=OPRD-Bridge-Official-Distributed-Smoke
export TRAIN_DATASET_NAME=DAPO-Math-17k-official-smoke-$ROWS
export MINI_BATCH_SIZE=8 TEMPERATURE=1.0 TEACHER_TEMPERATURE=1.0 REPETITION_PENALTY=1.0
export N_RESPONSES=2 LOG_PROB_TOP_K=0 TOP_K_STRATEGY=only_stu REWARD_WEIGHT_MODE=student_p
export LR_SCHEDULER=constant USE_KL=False ENABLE_FORMAT_REWARD=False MODEL_DTYPE=fp32
export IS_PLOT=False LOSS_AGG_MODE=token-mean ENABLE_ACTIVATION_OFFLOAD=False
export USE_REP_DISTILLATION=True REP_DISTILLATION_ONLY=True REP_DISTILLATION_COEF=10.0
export REP_DISTILLATION_POSITIONS=last_k REP_DISTILLATION_LAST_K=2000 REP_DISTILLATION_FIRST_K=2000
export REP_DISTILLATION_LAYERS=all REP_PROJECTOR_MODE=low_rank REP_LOW_RANK=8
export REP_LOW_RANK_INIT_CHECKPOINT=$PROJECTOR_BANK REP_FREEZE_PS=True
export USE_ATT_DISTILLATION=False ATT_DISTILLATION_COEF=1.0 ATT_DISTILLATION_LAYERS=all
export ATT_DISTILLATION_POSITIONS=last_k ATT_DISTILLATION_LAST_K=2000 ATT_DISTILLATION_FIRST_K=2000
export ATT_DISTILLATION_MAX_KEY_LEN=4096 ATT_DISTILLATION_LOSS=mse ATT_DISTILLATION_TEMPERATURE=1.0
readonly EXPORTS=HOME,PATH,CUDA_HOME,LD_LIBRARY_PATH,HF_HOME,VLLM_CACHE_ROOT,TRITON_CACHE_DIR,FLASHINFER_WORKSPACE_BASE,PYTHONDONTWRITEBYTECODE,MODEL_DIR,PROJECT_PATH,PROJECT_NAME,TRAIN_DATASET_NAME,MINI_BATCH_SIZE,TEMPERATURE,TEACHER_TEMPERATURE,REPETITION_PENALTY,N_RESPONSES,LOG_PROB_TOP_K,TOP_K_STRATEGY,REWARD_WEIGHT_MODE,LR_SCHEDULER,USE_KL,ENABLE_FORMAT_REWARD,MODEL_DTYPE,IS_PLOT,LOSS_AGG_MODE,ENABLE_ACTIVATION_OFFLOAD,USE_REP_DISTILLATION,REP_DISTILLATION_ONLY,REP_DISTILLATION_COEF,REP_DISTILLATION_POSITIONS,REP_DISTILLATION_LAST_K,REP_DISTILLATION_FIRST_K,REP_DISTILLATION_LAYERS,REP_PROJECTOR_MODE,REP_LOW_RANK,REP_LOW_RANK_INIT_CHECKPOINT,REP_FREEZE_PS,USE_ATT_DISTILLATION,ATT_DISTILLATION_COEF,ATT_DISTILLATION_LAYERS,ATT_DISTILLATION_POSITIONS,ATT_DISTILLATION_LAST_K,ATT_DISTILLATION_FIRST_K,ATT_DISTILLATION_MAX_KEY_LEN,ATT_DISTILLATION_LOSS,ATT_DISTILLATION_TEMPERATURE,SLURM_JOB_ID,CUDA_VISIBLE_DEVICES

srun --export="$EXPORTS" bash -c \
  'cd "$1/verl" && exec bash "$2/low_rank_rep_distillation.sh"' \
  bash "$WORK_DIRECTORY" "$SOURCE"
printf '{"status":"complete","rows":%s,"expected_updates":%s,"world_size":8}\n' \
  "$ROWS" "$((ROWS / 8))" > "$RUN_DIRECTORY/smoke-result.json"
printf 'OFFICIAL_STAGE2_SMOKE_COMPLETE\n' > "$RUN_DIRECTORY/COMPLETE"
