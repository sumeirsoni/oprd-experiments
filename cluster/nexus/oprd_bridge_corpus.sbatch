#!/bin/bash
#SBATCH --job-name=oprd-bridge-corpus
#SBATCH --partition=cml-furongh
#SBATCH --account=cml-furongh
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:h200-sxm:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=/cmlscratch/%u/oprd-experiments/cluster/logs/oprd_bridge_corpus_%j.out

# Initialize one seed-independent bridge build and generate its frozen rollout corpus.
set -euo pipefail

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
check_space() {
  local available_kb
  available_kb=$(df -Pk "$SCRATCH_ROOT" | awk 'NR == 2 {print $4}')
  [[ "$available_kb" =~ ^[0-9]+$ ]] || fail "could not determine free space on $SCRATCH_ROOT"
  (( available_kb >= MIN_FREE_GIB * 1024 * 1024 )) ||
    fail "$SCRATCH_ROOT has less than ${MIN_FREE_GIB} GiB free"
}

source /etc/profile.d/modules.sh 2>/dev/null || fail "cannot initialize environment modules"
module use /opt/local/stow/.modulefiles
module load Python3/3.12.7 || fail "cannot load Python3/3.12.7"
module use /opt/common/.modulefiles
module load cuda/12.8.1 || fail "cannot load cuda/12.8.1"
module load ffmpeg/7.1 || module load ffmpeg/6.1 || fail "cannot load a supported FFmpeg module"
command -v ffmpeg >/dev/null 2>&1 || fail "ffmpeg is unavailable after module load"

SCRATCH_ROOT=${SCRATCH_ROOT:-/cmlscratch/$USER}
ROOT=${ROOT:-$SCRATCH_ROOT/oprd-experiments}
CLUSTER_DIR=$ROOT/cluster
NEXUS_DIR=$CLUSTER_DIR/nexus
PYTHON=${PYTHON:-$ROOT/venv/bin/python}
OPS=$NEXUS_DIR/oprd_bridge_ops.py
BRIDGE_BUILDER=${BRIDGE_BUILDER:-$CLUSTER_DIR/build_oprd_bridge.py}
BRIDGE_BUILD_DIR=${BRIDGE_BUILD_DIR:?submit_oprd_bridge.sh must export BRIDGE_BUILD_DIR}
BRIDGE_ARTIFACT=${BRIDGE_ARTIFACT:?submit_oprd_bridge.sh must export BRIDGE_ARTIFACT}
BRIDGE_HF_HOME=${BRIDGE_HF_HOME:?submit_oprd_bridge.sh must export BRIDGE_HF_HOME}
BRIDGE_RUN_ID=${BRIDGE_RUN_ID:?submit_oprd_bridge.sh must export BRIDGE_RUN_ID}
JOB_RECORD=${JOB_RECORD:?submit_oprd_bridge.sh must export JOB_RECORD}
BRIDGE_SMOKE=${BRIDGE_SMOKE:-1}
MIN_FREE_GIB=${CORPUS_MIN_FREE_GIB:-50}
ENV_READY=$ROOT/venv/.oprd-env-ready.json

[[ "$BRIDGE_SMOKE" == 0 || "$BRIDGE_SMOKE" == 1 ]] || fail "BRIDGE_SMOKE must be 0 or 1"
[[ -d "$SCRATCH_ROOT" && -w "$SCRATCH_ROOT" ]] || fail "scratch is unavailable or not writable: $SCRATCH_ROOT"
[[ -d "$CLUSTER_DIR" ]] || fail "cluster directory is missing: $CLUSTER_DIR"
[[ -x "$PYTHON" ]] || fail "venv Python is missing or not executable: $PYTHON"
[[ -f "$BRIDGE_BUILDER" ]] || fail "bridge builder is missing: $BRIDGE_BUILDER"
[[ -r "$ENV_READY" ]] || fail "environment-ready marker is missing: $ENV_READY"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' || fail "venv Python is not 3.12"
case "$BRIDGE_BUILD_DIR/" in "$SCRATCH_ROOT"/*) ;; *) fail "build directory must be on scratch: $BRIDGE_BUILD_DIR" ;; esac
case "$BRIDGE_ARTIFACT/" in "$SCRATCH_ROOT"/*) ;; *) fail "artifact must be on scratch: $BRIDGE_ARTIFACT" ;; esac
case "$BRIDGE_HF_HOME/" in "$SCRATCH_ROOT"/*) ;; *) fail "HF cache must be on scratch: $BRIDGE_HF_HOME" ;; esac
[[ ! -e "$BRIDGE_ARTIFACT" ]] || fail "refusing to overwrite canonical artifact: $BRIDGE_ARTIFACT"
"$PYTHON" "$OPS" check-environment --marker "$ENV_READY" --requirements-file "$CLUSTER_DIR/requirements.txt" --python "$PYTHON"
"$PYTHON" "$OPS" check-runtime --scratch "$SCRATCH_ROOT" --path "$BRIDGE_BUILD_DIR" --path "$BRIDGE_HF_HOME" --artifact "$BRIDGE_ARTIFACT" --run-id "$BRIDGE_RUN_ID" --record "$JOB_RECORD"
check_space

export HF_HOME="$BRIDGE_HF_HOME"
mkdir -p "$CLUSTER_DIR/logs" "$(dirname "$BRIDGE_BUILD_DIR")" "$(dirname "$BRIDGE_ARTIFACT")" "$HF_HOME"
cd "$CLUSTER_DIR"

TEACHER=${TEACHER:-Qwen/Qwen3-4B}
STUDENT=${STUDENT:-Qwen/Qwen3-1.7B-Base}
TOKENIZER=${TOKENIZER:-$STUDENT}
PROMPTS=${PROMPTS:-BytedTsinghua-SIA/DAPO-Math-17k}

INIT_ARGS=(
  init
  --work-dir "$BRIDGE_BUILD_DIR"
  --teacher-model "$TEACHER"
  --student-model "$STUDENT"
  --tokenizer "$TOKENIZER"
  --dataset "$PROMPTS"
  --artifact-dir "$BRIDGE_ARTIFACT"
)
if [[ "$BRIDGE_SMOKE" == 1 ]]; then
  INIT_ARGS+=(--smoke)
fi

srun "$PYTHON" "$BRIDGE_BUILDER" "${INIT_ARGS[@]}"
srun "$PYTHON" "$BRIDGE_BUILDER" generate-corpus \
  --work-dir "$BRIDGE_BUILD_DIR"

printf 'CORPUS_OK workspace=%s artifact=%s smoke=%s\n' \
  "$BRIDGE_BUILD_DIR" "$BRIDGE_ARTIFACT" "$BRIDGE_SMOKE"
