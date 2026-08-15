#!/bin/bash
#SBATCH --job-name=oprd-official-smoke
#SBATCH --partition=cml-furongh
#SBATCH --account=cml-furongh
#SBATCH --qos=cml-high_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200-sxm:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=12:00:00
#SBATCH --output=/cmlscratch/%u/oprd-experiments-main/cluster/logs/oprd_official_smoke_%j.out
#SBATCH --export=NONE

set -euo pipefail
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ $# -eq 2 || $# -eq 3 ]] ||
  fail "usage: official_stage2_smoke.sbatch RESULT_NAMESPACE ROWS [WORLD_SIZE]"
readonly RESULT_NAMESPACE=$1
readonly ROWS=$2
readonly WORLD_SIZE=${3:-8}
[[ $RESULT_NAMESPACE =~ ^[a-z0-9][a-z0-9._-]{5,79}$ ]] || fail "invalid result namespace"
[[ $ROWS == 16 || $ROWS == 80 ]] || fail "ROWS must be 16 for smoke or 80 for canary"
[[ $WORLD_SIZE == 4 || $WORLD_SIZE == 8 ]] || fail "WORLD_SIZE must be 4 or 8"

readonly RUN_USER=${SLURM_JOB_USER:?SLURM_JOB_USER is required}
readonly SCRATCH_ROOT=/cmlscratch/$RUN_USER
readonly PROJECT_ROOT=$SCRATCH_ROOT/oprd-experiments-main
readonly OFFICIAL_ROOT=$SCRATCH_ROOT/oprd-official-stage2
readonly SOURCE=$OFFICIAL_ROOT/source/OPRD
readonly ENVIRONMENT=$OFFICIAL_ROOT/env-a63236c
readonly PYTHON=$ENVIRONMENT/bin/python
readonly CAPTURE_BIN=$PROJECT_ROOT/cluster/official_stage2/capture_bin
readonly CONTRACT=$PROJECT_ROOT/cluster/official_stage2/task0_contract.json
readonly CONTRACT_TOOL=$PROJECT_ROOT/cluster/official_stage2/contract.py
readonly MODEL_DIR=$OFFICIAL_ROOT/models
readonly PROJECTOR_BANK=$OFFICIAL_ROOT/bridge/rank8_all_layers/ps_bank.pt
readonly RUN_DIRECTORY=$OFFICIAL_ROOT/smokes/$RESULT_NAMESPACE
readonly WORK_DIRECTORY=$RUN_DIRECTORY/work
readonly PROJECT_PATH=$RUN_DIRECTORY/output
readonly HOME=$OFFICIAL_ROOT/home
readonly CUDA_HOME=/opt/common/cuda/cuda-12.8.1
readonly LD_LIBRARY_PATH=$CUDA_HOME/lib64
readonly RUNTIME_PATH=$ENVIRONMENT/bin:$CUDA_HOME/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/bin:/sbin
readonly HF_HOME=$OFFICIAL_ROOT/cache/huggingface
readonly VLLM_CACHE_ROOT=$OFFICIAL_ROOT/cache/vllm
readonly TRITON_CACHE_DIR=$OFFICIAL_ROOT/cache/triton
readonly FLASHINFER_WORKSPACE_BASE=$OFFICIAL_ROOT/cache/flashinfer
readonly PYTHONDONTWRITEBYTECODE=1
readonly BASH_ENV=$WORK_DIRECTORY/bash-env
readonly OPRD_REAL_PYTHON=$PYTHON
readonly OPRD_HYDRA_CAPTURE=$RUN_DIRECTORY/actual-hydra.yaml
readonly OPRD_ARGV_CAPTURE=$RUN_DIRECTORY/actual-python-argv.json
readonly OPRD_CONTRACT_TOOL=$CONTRACT_TOOL
readonly OPRD_CONTRACT=$CONTRACT
readonly OPRD_OFFICIAL_ROOT=$OFFICIAL_ROOT
readonly OPRD_EXPECTED_WORLD_SIZE=$WORLD_SIZE
readonly RUNTIME_LOG=$RUN_DIRECTORY/official-runtime.log

[[ -x $PYTHON && -x $CAPTURE_BIN/python3 && -r $PROJECTOR_BANK ]] ||
  fail "official environment, capture wrapper, or bridge is missing"
[[ ! -e $RUN_DIRECTORY ]] || fail "smoke namespace already exists: $RUN_DIRECTORY"
mkdir -p "$WORK_DIRECTORY/verl" "$WORK_DIRECTORY/datasets/test_data/AMC23" "$PROJECT_PATH"
printf 'set -e\nset -o pipefail\nunset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES\n' > "$BASH_ENV"
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

export PATH=$CAPTURE_BIN:$RUNTIME_PATH
export HOME PATH CUDA_HOME LD_LIBRARY_PATH HF_HOME VLLM_CACHE_ROOT BASH_ENV
export TRITON_CACHE_DIR FLASHINFER_WORKSPACE_BASE PYTHONDONTWRITEBYTECODE
export OPRD_REAL_PYTHON OPRD_HYDRA_CAPTURE OPRD_ARGV_CAPTURE
export OPRD_CONTRACT_TOOL OPRD_CONTRACT OPRD_OFFICIAL_ROOT OPRD_EXPECTED_WORLD_SIZE
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
readonly EXPORTS=HOME,PATH,CUDA_HOME,LD_LIBRARY_PATH,HF_HOME,VLLM_CACHE_ROOT,TRITON_CACHE_DIR,FLASHINFER_WORKSPACE_BASE,PYTHONDONTWRITEBYTECODE,BASH_ENV,OPRD_REAL_PYTHON,OPRD_HYDRA_CAPTURE,OPRD_ARGV_CAPTURE,OPRD_CONTRACT_TOOL,OPRD_CONTRACT,OPRD_OFFICIAL_ROOT,OPRD_EXPECTED_WORLD_SIZE,MODEL_DIR,PROJECT_PATH,PROJECT_NAME,TRAIN_DATASET_NAME,MINI_BATCH_SIZE,TEMPERATURE,TEACHER_TEMPERATURE,REPETITION_PENALTY,N_RESPONSES,LOG_PROB_TOP_K,TOP_K_STRATEGY,REWARD_WEIGHT_MODE,LR_SCHEDULER,USE_KL,ENABLE_FORMAT_REWARD,MODEL_DTYPE,IS_PLOT,LOSS_AGG_MODE,ENABLE_ACTIVATION_OFFLOAD,USE_REP_DISTILLATION,REP_DISTILLATION_ONLY,REP_DISTILLATION_COEF,REP_DISTILLATION_POSITIONS,REP_DISTILLATION_LAST_K,REP_DISTILLATION_FIRST_K,REP_DISTILLATION_LAYERS,REP_PROJECTOR_MODE,REP_LOW_RANK,REP_LOW_RANK_INIT_CHECKPOINT,REP_FREEZE_PS,USE_ATT_DISTILLATION,ATT_DISTILLATION_COEF,ATT_DISTILLATION_LAYERS,ATT_DISTILLATION_POSITIONS,ATT_DISTILLATION_LAST_K,ATT_DISTILLATION_FIRST_K,ATT_DISTILLATION_MAX_KEY_LEN,ATT_DISTILLATION_LOSS,ATT_DISTILLATION_TEMPERATURE,SLURM_JOB_ID,CUDA_VISIBLE_DEVICES

srun --export="$EXPORTS" "$PYTHON" - "$WORLD_SIZE" <<'PY'
import sys
import torch
world_size = int(sys.argv[1])
assert torch.cuda.device_count() == world_size, torch.cuda.device_count()
print({
    "visible_gpus": torch.cuda.device_count(),
    "devices": [torch.cuda.get_device_name(i) for i in range(world_size)],
})
PY
srun --export="$EXPORTS" bash -c \
  'cd "$1/verl" && exec bash "$2/low_rank_rep_distillation.sh"' \
  bash "$WORK_DIRECTORY" "$SOURCE" 2>&1 | tee "$RUNTIME_LOG"
"$PYTHON" - "$RUN_DIRECTORY" "$PROJECT_PATH" "$RUNTIME_LOG" "$ROWS" "$WORLD_SIZE" <<'PY'
import hashlib
import json
import pathlib
import sys
run = pathlib.Path(sys.argv[1])
project = pathlib.Path(sys.argv[2])
runtime_log = pathlib.Path(sys.argv[3])
rows = int(sys.argv[4])
world_size = int(sys.argv[5])
expected_updates = rows // 8
hydra = run / "actual-hydra.yaml"
argv = run / "actual-python-argv.json"
if not hydra.is_file() or not argv.is_file() or not runtime_log.is_file():
    raise SystemExit("actual Hydra, argv, or runtime capture is missing")
logs = sorted((project / "logs" / "terminal").glob("*.log"))
if len(logs) != 1:
    raise SystemExit(f"expected one official terminal log, found {len(logs)}")
terminal_text = logs[0].read_text(encoding="utf-8", errors="replace")
for marker in ("Traceback", "RayActorError", "CUDA out of memory", "NCCL error"):
    if marker in terminal_text:
        raise SystemExit(f"terminal log contains failure marker: {marker}")
step_marker = f"training/global_step:{expected_updates}"
if step_marker not in terminal_text:
    raise SystemExit(f"terminal log is missing final step marker: {step_marker}")
runtime_text = runtime_log.read_text(encoding="utf-8", errors="replace")
for marker in ("RayActorError", "ActorDiedError", "CUDA out of memory", "NCCL error"):
    if marker in runtime_text:
        raise SystemExit(f"runtime log contains failure marker: {marker}")
cleanup_warning = "RuntimeError: DataLoader worker ("
cleanup_warning_count = sum(
    cleanup_warning in line and "is killed by signal: Killed" in line
    for line in runtime_text.splitlines()
)
traceback_count = runtime_text.count("Traceback (most recent call last):")
if traceback_count != cleanup_warning_count:
    raise SystemExit(
        "runtime traceback count is not explained by the known DataLoader shutdown warning: "
        f"tracebacks={traceback_count}, cleanup_warnings={cleanup_warning_count}"
    )
steps = list(project.rglob(f"global_step_{expected_updates}"))
if len(steps) != 1:
    raise SystemExit(f"expected one final checkpoint, found {len(steps)}")
actor = steps[0] / "actor"
for kind in ("model", "optim", "extra_state"):
    pattern = f"{kind}_world_size_{world_size}_rank_*.pt"
    files = list(actor.glob(pattern))
    if len(files) != world_size:
        raise SystemExit(
            f"expected {world_size} checkpoint shards for {pattern}, found {len(files)}"
        )
if not (actor / "huggingface" / "config.json").is_file():
    raise SystemExit("checkpoint Hugging Face config is missing")
def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
result = {
    "status": "complete",
    "rows": rows,
    "expected_updates": expected_updates,
    "world_size": world_size,
    "topology_deviation_from_official": world_size != 8,
    "checkpoint": str(steps[0]),
    "checkpoint_shards_per_kind": world_size,
    "cleanup_warning_count": cleanup_warning_count,
    "actual_hydra_sha256": sha256(hydra),
    "actual_argv_sha256": sha256(argv),
    "runtime_log": str(runtime_log),
    "runtime_log_sha256": sha256(runtime_log),
    "terminal_log": str(logs[0]),
    "terminal_log_sha256": sha256(logs[0]),
}
(run / "smoke-result.json").write_text(
    json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
PY
printf 'OFFICIAL_STAGE2_SMOKE_COMPLETE\n' > "$RUN_DIRECTORY/COMPLETE"
