#!/bin/bash
#SBATCH --job-name=oprd-official-eval
#SBATCH --partition=cml-furongh
#SBATCH --account=cml-furongh
#SBATCH --qos=cml-high_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h200-sxm:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=72:00:00
#SBATCH --array=0-5
#SBATCH --output=/cmlscratch/%u/oprd-experiments-main/cluster/logs/oprd_official_eval_%A_%a.out
#SBATCH --export=NONE

set -euo pipefail

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[[ $# -eq 2 ]] || fail "usage: evaluate_official_stage2.sbatch TASK0_RUN_DIRECTORY EVAL_NAMESPACE"
readonly TASK0_RUN_INPUT=$1
readonly EVAL_NAMESPACE=$2
[[ $EVAL_NAMESPACE =~ ^[a-z0-9][a-z0-9._-]{5,79}$ ]] || fail "invalid evaluation namespace"
readonly INDEX=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
readonly BENCHMARKS=(aime24 aime25 aimo aime24 aime25 aimo)
readonly TRACKS=(published_offline_exact published_offline_exact published_offline_exact corrected_seeded_single_suffix corrected_seeded_single_suffix corrected_seeded_single_suffix)
readonly BENCHMARK=${BENCHMARKS[$INDEX]}
readonly TRACK=${TRACKS[$INDEX]}

readonly RUN_USER=${SLURM_JOB_USER:?SLURM_JOB_USER is required}
readonly SCRATCH_ROOT=/cmlscratch/$RUN_USER
readonly PROJECT_ROOT=$SCRATCH_ROOT/oprd-experiments-main
readonly OFFICIAL_ROOT=$SCRATCH_ROOT/oprd-official-stage2
TASK0_RUN=$(realpath -e -- "$TASK0_RUN_INPUT") || fail "cannot resolve task-0 run"
readonly TASK0_RUN
case "$TASK0_RUN/" in
  "$OFFICIAL_ROOT/runs/"*) ;;
  *) fail "task-0 run is outside the official runs root" ;;
esac
[[ $(dirname -- "$TASK0_RUN") == "$OFFICIAL_ROOT/runs" ]] ||
  fail "task-0 run must be an immediate child of the official runs root"
readonly SOURCE=$OFFICIAL_ROOT/source/OPRD
readonly ENVIRONMENT=$OFFICIAL_ROOT/env-a63236c
readonly PYTHON=$ENVIRONMENT/bin/python
readonly GENERATOR=$PROJECT_ROOT/cluster/official_stage2/generate_eval.py
readonly SCORER=$PROJECT_ROOT/cluster/official_stage2/aime_eval.py
readonly VALIDATOR=$PROJECT_ROOT/cluster/official_stage2/validate_task0.py
readonly INPUT_VALIDATOR=$PROJECT_ROOT/cluster/official_stage2/prepare_inputs.py
readonly CONTRACT=$PROJECT_ROOT/cluster/official_stage2/task0_contract.json
readonly ACCEPTANCE=$TASK0_RUN/acceptance.json
readonly DESTINATION=$OFFICIAL_ROOT/evaluations/$EVAL_NAMESPACE/$TRACK/$BENCHMARK
readonly SCORE=$DESTINATION/score-v1.json
readonly HOME=$OFFICIAL_ROOT/home
readonly CUDA_HOME=/opt/common/cuda/cuda-12.8.1
readonly LD_LIBRARY_PATH=$CUDA_HOME/lib64
readonly PATH=$ENVIRONMENT/bin:$CUDA_HOME/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/bin:/sbin
readonly HF_HOME=$OFFICIAL_ROOT/cache/huggingface
readonly VLLM_CACHE_ROOT=$OFFICIAL_ROOT/cache/vllm
readonly TRITON_CACHE_DIR=$OFFICIAL_ROOT/cache/triton
readonly FLASHINFER_WORKSPACE_BASE=$OFFICIAL_ROOT/cache/flashinfer
readonly PYTHONDONTWRITEBYTECODE=1

[[ -x $PYTHON ]] || fail "official environment is missing"
[[ -r $ACCEPTANCE ]] || fail "accepted task-0 report is missing"
[[ $(<"$TASK0_RUN/COMPLETE") == ACCEPTED_OFFICIAL_TASK0 ]] ||
  fail "task-0 COMPLETE marker is missing or invalid"
EXPECTED_PROJECT_REVISION=$(
  "$PYTHON" - "$ACCEPTANCE" <<'PY'
import json, pathlib, sys
print(json.loads(pathlib.Path(sys.argv[1]).read_text())["project_revision"])
PY
)
readonly EXPECTED_PROJECT_REVISION
[[ $(git -C "$PROJECT_ROOT" rev-parse HEAD) == "$EXPECTED_PROJECT_REVISION" ]] ||
  fail "evaluation project revision differs from accepted training"
[[ -z $(git -C "$PROJECT_ROOT" status --porcelain --untracked-files=all -- \
  cluster/official_stage2 cluster/oprd_bridge.py cluster/oprd_atomic_directory.py \
  cluster/nexus/task0_official_stage2.sbatch cluster/nexus/evaluate_official_stage2.sbatch) ]] ||
  fail "project-side official evaluation trust files are not clean and tracked"
[[ ! -e $DESTINATION ]] || fail "evaluation destination already exists: $DESTINATION"
REVALIDATED=$OFFICIAL_ROOT/.eval-revalidated-${SLURM_JOB_ID}-${INDEX}.json
SOURCE_REVALIDATED=$OFFICIAL_ROOT/.eval-source-revalidated-${SLURM_JOB_ID}-${INDEX}.json
readonly REVALIDATED SOURCE_REVALIDATED
[[ ! -e $REVALIDATED && ! -e $SOURCE_REVALIDATED ]] ||
  fail "stale evaluation validation output exists"
cleanup() { rm -f -- "$REVALIDATED" "$SOURCE_REVALIDATED"; }
trap cleanup EXIT
"$PYTHON" "$VALIDATOR" --contract "$CONTRACT" accept \
  --run-directory "$TASK0_RUN" \
  --project-root "$PROJECT_ROOT" > "$REVALIDATED"
cmp -s "$REVALIDATED" "$ACCEPTANCE" || fail "task-0 acceptance report no longer authenticates the run"
"$PYTHON" "$INPUT_VALIDATOR" --contract "$CONTRACT" source \
  --source "$SOURCE" \
  --manifest "$SOURCE_REVALIDATED" >/dev/null
cmp -s "$SOURCE_REVALIDATED" "$OFFICIAL_ROOT/source/manifest.json" ||
  fail "official evaluation source no longer matches its authenticated manifest"
readarray -t CHECKPOINT_FIELDS < <(
  "$PYTHON" - "$ACCEPTANCE" <<'PY'
import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text())
assert value["status"] == "ACCEPTED_OFFICIAL_TASK0", value["status"]
print(value["merged_checkpoint"])
print(value["merged_checkpoint_tree_sha256"])
PY
)
readonly CHECKPOINT=${CHECKPOINT_FIELDS[0]}
readonly CHECKPOINT_SHA256=${CHECKPOINT_FIELDS[1]}
[[ -d $CHECKPOINT ]] || fail "accepted checkpoint is missing: $CHECKPOINT"

export HOME PATH CUDA_HOME LD_LIBRARY_PATH HF_HOME VLLM_CACHE_ROOT
export TRITON_CACHE_DIR FLASHINFER_WORKSPACE_BASE PYTHONDONTWRITEBYTECODE
readonly EXPORTS=HOME,PATH,CUDA_HOME,LD_LIBRARY_PATH,HF_HOME,VLLM_CACHE_ROOT,TRITON_CACHE_DIR,FLASHINFER_WORKSPACE_BASE,PYTHONDONTWRITEBYTECODE,SLURM_JOB_ID,SLURM_JOB_NUM_NODES,SLURM_JOB_NODELIST,CUDA_VISIBLE_DEVICES

srun --export="$EXPORTS" "$PYTHON" "$GENERATOR" \
  --contract "$CONTRACT" \
  --official-source "$SOURCE" \
  --checkpoint "$CHECKPOINT" \
  --checkpoint-sha256 "$CHECKPOINT_SHA256" \
  --benchmark "$BENCHMARK" \
  --track "$TRACK" \
  --destination "$DESTINATION"
"$PYTHON" "$SCORER" score \
  --manifest "$DESTINATION/manifest.json" \
  --official-source "$SOURCE" \
  --official-commit a63236c990b1bfa9551e9d020719490d3ecd93e8 \
  --output "$SCORE"
printf 'official evaluation complete: %s\n' "$DESTINATION"
