#!/bin/bash
# Submit, reconcile, or resume the OPRD bridge build and external-artifact smoke.
# This script never submits the full Tier-2 matrix or production Stage-2 seeds.
set -euo pipefail

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

usage() {
  cat <<'EOF'
Usage:
  submit_oprd_bridge.sh
  submit_oprd_bridge.sh --preflight-only
  submit_oprd_bridge.sh --status RECORD.json
  submit_oprd_bridge.sh --resume RECORD.json
  submit_oprd_bridge.sh --release-claim RECORD.json
  submit_oprd_bridge.sh --adopt-job RECORD.json PHASE JOB_ID
EOF
}

absolute_path() {
  "$PYTHON" - "$1" <<'PY'
import pathlib
import sys
print(pathlib.Path(sys.argv[1]).expanduser().resolve())
PY
}

file_sha256() {
  "$PYTHON" - "$1" <<'PY'
import hashlib
import pathlib
import sys
print(hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest())
PY
}

compute_source_digest() {
  "$PYTHON" - \
    "$CLUSTER_DIR/build_oprd_bridge.py" \
    "$CLUSTER_DIR/oprd_bridge.py" \
    "$CLUSTER_DIR/train_harness.py" \
    "$CLUSTER_DIR/requirements.txt" \
    "$NEXUS_DIR/oprd_bridge_ops.py" \
    "$NEXUS_DIR/submit_oprd_bridge.sh" \
    "$NEXUS_DIR/oprd_bridge_corpus.sbatch" \
    "$NEXUS_DIR/oprd_bridge_teacher.sbatch" \
    "$NEXUS_DIR/oprd_bridge_student_groups.sbatch" \
    "$NEXUS_DIR/oprd_bridge_publish.sbatch" \
    "$NEXUS_DIR/oprd_bridge_stage2_smoke.sbatch" <<'PY'
import hashlib
import pathlib
import sys
digest = hashlib.sha256()
for raw in sys.argv[1:]:
    path = pathlib.Path(raw).resolve(strict=True)
    name = path.name.encode()
    payload = path.read_bytes()
    digest.update(len(name).to_bytes(8, "big"))
    digest.update(name)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)
print(digest.hexdigest())
PY
}

load_runtime_modules() {
  local modules_init=${MODULES_INIT:-/etc/profile.d/modules.sh}
  source "$modules_init" 2>/dev/null || fail "cannot initialize environment modules"
  module use /opt/local/stow/.modulefiles
  module load Python3/3.12.7 || fail "cannot load Python3/3.12.7"
  module use /opt/common/.modulefiles
  module load cuda/12.8.1 || fail "cannot load cuda/12.8.1"
  module load ffmpeg/7.1 || module load ffmpeg/6.1 ||
    fail "cannot load a supported FFmpeg module"
  command -v ffmpeg >/dev/null 2>&1 || fail "ffmpeg is unavailable after module load"
}

submit_job() {
  local output job_id
  output=$(sbatch --parsable "$@")
  job_id=${output%%;*}
  [[ "$job_id" =~ ^[0-9]+$ ]] || fail "sbatch returned an invalid job ID: $output"
  printf '%s\n' "$job_id"
}

run_preflight() {
  local minimum_bytes
  local quota_args=()
  minimum_bytes=$((SUBMIT_MIN_FREE_GIB * 1024 * 1024 * 1024))
  if [[ "$ALLOW_QUOTA_UNAVAILABLE" == 1 ]]; then
    quota_args+=(--allow-quota-unavailable)
  fi
  "$PYTHON" "$OPS" preflight \
    --scratch "$SCRATCH_ROOT" \
    --hf-home "$BRIDGE_HF_HOME" \
    --python "$PYTHON" \
    --environment-marker "$ENV_READY" \
    --requirements-file "$CLUSTER_DIR/requirements.txt" \
    --import torch \
    --import numpy \
    --import safetensors \
    --import transformers \
    --import datasets \
    --import vllm \
    --script "$NEXUS_DIR/oprd_bridge_corpus.sbatch" \
    --script "$NEXUS_DIR/oprd_bridge_teacher.sbatch" \
    --script "$NEXUS_DIR/oprd_bridge_student_groups.sbatch" \
    --script "$NEXUS_DIR/oprd_bridge_publish.sbatch" \
    --script "$NEXUS_DIR/oprd_bridge_stage2_smoke.sbatch" \
    --writable-path "$ROOT" \
    --writable-path "$BRIDGE_BUILD_PARENT" \
    --writable-path "$(dirname "$BRIDGE_ARTIFACT")" \
    --writable-path "$JOB_RECORD_DIR" \
    --writable-path "$BRIDGE_HF_HOME" \
    --min-free-bytes "$minimum_bytes" \
    --min-free-inodes "$SUBMIT_MIN_FREE_INODES" \
    --quota-user "$USER" \
    "${quota_args[@]}"
}

load_record_configuration() {
  local record=$1 verify_source=${2:-1} line record_output
  record_output=$("$PYTHON" - "$record" <<'PY'
import json
import pathlib
import sys
record = json.loads(pathlib.Path(sys.argv[1]).read_text())
config = record.get("configuration", {})
keys = (
    "root",
    "python",
    "bridge_builder",
    "hf_home",
    "bridge_smoke",
    "allow_paper_scale",
    "bridge_pair",
    "bridge_rank",
    "teacher",
    "student",
    "tokenizer",
    "prompts",
    "submit_min_free_gib",
    "submit_min_free_inodes",
    "allow_quota_unavailable",
    "environment_marker_sha256",
    "source_bundle_sha256",
)
values = [
    record.get("scratch"),
    record.get("build_dir"),
    record.get("artifact_dir"),
    record.get("run_id"),
    *[config.get(key) for key in keys],
]
if any(not isinstance(value, str) or not value for value in values):
    raise SystemExit("record has incomplete immutable configuration")
print("\n".join(values))
PY
  ) || fail "cannot load immutable configuration from $record"
  RECORD_VALUES=()
  while IFS= read -r line; do
    RECORD_VALUES[${#RECORD_VALUES[@]}]=$line
  done <<<"$record_output"
  [[ ${#RECORD_VALUES[@]} -eq 21 ]] || fail "record returned an unexpected configuration field count"
  SCRATCH_ROOT=${RECORD_VALUES[0]}
  BRIDGE_BUILD_DIR=${RECORD_VALUES[1]}
  BRIDGE_ARTIFACT=${RECORD_VALUES[2]}
  BRIDGE_RUN_ID=${RECORD_VALUES[3]}
  ROOT=${RECORD_VALUES[4]}
  PYTHON=${RECORD_VALUES[5]}
  BRIDGE_BUILDER=${RECORD_VALUES[6]}
  BRIDGE_HF_HOME=${RECORD_VALUES[7]}
  BRIDGE_SMOKE=${RECORD_VALUES[8]}
  ALLOW_PAPER_SCALE=${RECORD_VALUES[9]}
  BRIDGE_PAIR=${RECORD_VALUES[10]}
  BRIDGE_RANK=${RECORD_VALUES[11]}
  TEACHER=${RECORD_VALUES[12]}
  STUDENT=${RECORD_VALUES[13]}
  TOKENIZER=${RECORD_VALUES[14]}
  PROMPTS=${RECORD_VALUES[15]}
  SUBMIT_MIN_FREE_GIB=${RECORD_VALUES[16]}
  SUBMIT_MIN_FREE_INODES=${RECORD_VALUES[17]}
  ALLOW_QUOTA_UNAVAILABLE=${RECORD_VALUES[18]}
  RECORDED_ENVIRONMENT_MARKER_SHA256=${RECORD_VALUES[19]}
  RECORDED_SOURCE_BUNDLE_SHA256=${RECORD_VALUES[20]}
  CLUSTER_DIR=$ROOT/cluster
  NEXUS_DIR=$CLUSTER_DIR/nexus
  OPS=$NEXUS_DIR/oprd_bridge_ops.py
  BRIDGE_BUILD_PARENT=$(dirname "$BRIDGE_BUILD_DIR")
  JOB_RECORD_DIR=$(dirname "$record")
  ENV_READY=$ROOT/venv/.oprd-env-ready.json
  if [[ "$verify_source" == 1 ]]; then
    CURRENT_ENVIRONMENT_MARKER_SHA256=$(file_sha256 "$ENV_READY") ||
      fail "cannot hash recorded environment marker"
    [[ "$CURRENT_ENVIRONMENT_MARKER_SHA256" == "$RECORDED_ENVIRONMENT_MARKER_SHA256" ]] ||
      fail "environment-ready marker changed since initial submission"
    CURRENT_SOURCE_BUNDLE_SHA256=$(compute_source_digest) ||
      fail "cannot hash recorded bridge source bundle"
    [[ "$CURRENT_SOURCE_BUNDLE_SHA256" == "$RECORDED_SOURCE_BUNDLE_SHA256" ]] ||
      fail "bridge source bundle changed since initial submission"
  fi
}

release_successful_claim() {
  local record=$1 status claim_dir line
  status=$("$PYTHON" - "$record" <<'PY'
import json
import pathlib
import sys
record = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(record.get("dag_status", ""))
print(record.get("claim_dir", ""))
PY
  ) || fail "cannot inspect reconciled record"
  STATUS_VALUES=()
  while IFS= read -r line; do
    STATUS_VALUES[${#STATUS_VALUES[@]}]=$line
  done <<<"$status"
  [[ ${STATUS_VALUES[0]:-} == succeeded ]] || return 0
  claim_dir=${STATUS_VALUES[1]:-}
  if [[ -d "$claim_dir" ]]; then
    "$PYTHON" "$OPS" release-claim \
      --scratch "$SCRATCH_ROOT" \
      --artifact "$BRIDGE_ARTIFACT" \
      --run-id "$BRIDGE_RUN_ID" \
      --record "$record"
  fi
}

MODE=fresh
RECORD_ARGUMENT=
ADOPT_PHASE=
ADOPT_JOB_ID=
case ${1:-} in
  "") ;;
  --preflight-only)
    [[ $# -eq 1 ]] || { usage >&2; exit 2; }
    MODE=preflight
    ;;
  --status|--resume|--release-claim)
    [[ $# -eq 2 ]] || { usage >&2; exit 2; }
    MODE=${1#--}
    RECORD_ARGUMENT=$2
    ;;
  --adopt-job)
    [[ $# -eq 4 ]] || { usage >&2; exit 2; }
    MODE=adopt-job
    RECORD_ARGUMENT=$2
    ADOPT_PHASE=$3
    ADOPT_JOB_ID=$4
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

SCRATCH_ROOT=${SCRATCH_ROOT:-/cmlscratch/$USER}
ROOT=${ROOT:-$SCRATCH_ROOT/oprd-experiments}
CLUSTER_DIR=$ROOT/cluster
NEXUS_DIR=$CLUSTER_DIR/nexus
PYTHON=${PYTHON:-$ROOT/venv/bin/python}
OPS=$NEXUS_DIR/oprd_bridge_ops.py

if [[ "$MODE" == status || "$MODE" == resume || "$MODE" == release-claim || "$MODE" == adopt-job ]]; then
  [[ -x "$PYTHON" ]] || fail "venv Python is missing or not executable: $PYTHON"
  RECORD_ARGUMENT=$(absolute_path "$RECORD_ARGUMENT")
fi

if [[ "$MODE" == status ]]; then
  set +e
  STATUS_OUTPUT=$("$PYTHON" "$OPS" status --record "$RECORD_ARGUMENT")
  STATUS_RC=$?
  set -e
  printf '%s\n' "$STATUS_OUTPUT"
  load_record_configuration "$RECORD_ARGUMENT" 0
  release_successful_claim "$RECORD_ARGUMENT"
  exit "$STATUS_RC"
fi

if [[ "$MODE" == release-claim ]]; then
  load_record_configuration "$RECORD_ARGUMENT" 0
  "$PYTHON" "$OPS" release-claim \
    --scratch "$SCRATCH_ROOT" \
    --artifact "$BRIDGE_ARTIFACT" \
    --run-id "$BRIDGE_RUN_ID" \
    --record "$RECORD_ARGUMENT"
  exit 0
fi

if [[ "$MODE" == adopt-job ]]; then
  load_record_configuration "$RECORD_ARGUMENT" 0
  "$PYTHON" "$OPS" record-job \
    --record "$RECORD_ARGUMENT" \
    --phase "$ADOPT_PHASE" \
    --job-id "$ADOPT_JOB_ID"
  printf 'Adopted Slurm job %s for pending phase %s. Run --status next.\n' \
    "$ADOPT_JOB_ID" "$ADOPT_PHASE"
  exit 0
fi

BRIDGE_GROUP_COUNT=7
BRIDGE_PAIR=${BRIDGE_PAIR:-qwen3_4b_to_1p7b}
BRIDGE_RANK=${BRIDGE_RANK:-8}
BRIDGE_SMOKE=${BRIDGE_SMOKE:-1}
ALLOW_PAPER_SCALE=${ALLOW_PAPER_SCALE:-0}
TEACHER=${TEACHER:-Qwen/Qwen3-4B}
STUDENT=${STUDENT:-Qwen/Qwen3-1.7B-Base}
TOKENIZER=${TOKENIZER:-$STUDENT}
PROMPTS=${PROMPTS:-BytedTsinghua-SIA/DAPO-Math-17k}
BRIDGE_HF_HOME=${BRIDGE_HF_HOME:-$SCRATCH_ROOT/hf-cache}
BRIDGE_BUILDER=${BRIDGE_BUILDER:-$CLUSTER_DIR/build_oprd_bridge.py}
BRIDGE_RUN_ID=${BRIDGE_RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}
BRIDGE_BUILD_PARENT=${BRIDGE_BUILD_PARENT:-$CLUSTER_DIR/artifacts/oprd_bridge/.builds}
SUBMIT_MIN_FREE_GIB=${SUBMIT_MIN_FREE_GIB:-50}
SUBMIT_MIN_FREE_INODES=${SUBMIT_MIN_FREE_INODES:-10000}
ALLOW_QUOTA_UNAVAILABLE=${ALLOW_QUOTA_UNAVAILABLE:-0}
JOB_RECORD_DIR=${JOB_RECORD_DIR:-$CLUSTER_DIR/job_ids}
ENV_READY=$ROOT/venv/.oprd-env-ready.json

if [[ "$MODE" == resume ]]; then
  load_record_configuration "$RECORD_ARGUMENT"
  JOB_RECORD=$RECORD_ARGUMENT
else
  ARTIFACT_VARIANT=production
  [[ "$BRIDGE_SMOKE" == 1 ]] && ARTIFACT_VARIANT=smoke
  BRIDGE_BUILD_DIR=${BRIDGE_BUILD_DIR:-$BRIDGE_BUILD_PARENT/$BRIDGE_RUN_ID}
  BRIDGE_ARTIFACT=${BRIDGE_ARTIFACT:-$CLUSTER_DIR/artifacts/oprd_bridge/$BRIDGE_PAIR/$ARTIFACT_VARIANT/rank${BRIDGE_RANK}_all_layers}
  JOB_RECORD=${JOB_RECORD:-$JOB_RECORD_DIR/oprd_bridge_$BRIDGE_RUN_ID.json}
fi

[[ "$BRIDGE_SMOKE" == 0 || "$BRIDGE_SMOKE" == 1 ]] || fail "BRIDGE_SMOKE must be 0 or 1"
[[ "$TEACHER" == Qwen/Qwen3-4B ]] || fail "this DAG is fixed to teacher Qwen/Qwen3-4B"
[[ "$STUDENT" == Qwen/Qwen3-1.7B-Base ]] || fail "this DAG is fixed to student Qwen/Qwen3-1.7B-Base"
[[ "$TOKENIZER" == Qwen/Qwen3-1.7B-Base ]] || fail "this DAG is fixed to the student tokenizer"
[[ "$PROMPTS" == BytedTsinghua-SIA/DAPO-Math-17k ]] || fail "this DAG is fixed to DAPO-Math-17k"
if [[ "$BRIDGE_SMOKE" == 0 && "$ALLOW_PAPER_SCALE" != 1 ]]; then
  fail "paper-scale submission requires ALLOW_PAPER_SCALE=1 after storage, quota, and cost approval"
fi
[[ "$BRIDGE_RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]] || fail "BRIDGE_RUN_ID contains unsafe characters"
[[ "$SUBMIT_MIN_FREE_GIB" =~ ^[0-9]+$ ]] || fail "SUBMIT_MIN_FREE_GIB must be a nonnegative integer"
[[ "$SUBMIT_MIN_FREE_INODES" =~ ^[0-9]+$ ]] || fail "SUBMIT_MIN_FREE_INODES must be a nonnegative integer"
[[ "$ALLOW_QUOTA_UNAVAILABLE" == 0 || "$ALLOW_QUOTA_UNAVAILABLE" == 1 ]] || fail "ALLOW_QUOTA_UNAVAILABLE must be 0 or 1"
[[ -d "$SCRATCH_ROOT" && -w "$SCRATCH_ROOT" ]] || fail "scratch is unavailable or not writable: $SCRATCH_ROOT"
[[ -d "$CLUSTER_DIR" ]] || fail "cluster directory is missing: $CLUSTER_DIR"
[[ -x "$PYTHON" ]] || fail "venv Python is missing or not executable: $PYTHON"
[[ -f "$OPS" ]] || fail "operations helper is missing: $OPS"
[[ -f "$BRIDGE_BUILDER" ]] || fail "bridge builder is missing: $BRIDGE_BUILDER"
command -v sbatch >/dev/null 2>&1 || fail "sbatch is not available"
command -v sacct >/dev/null 2>&1 || fail "sacct is not available"
SOURCE_BUNDLE_SHA256=$(compute_source_digest) || fail "cannot hash bridge source bundle"

load_runtime_modules
run_preflight

if [[ "$MODE" == preflight ]]; then
  printf 'PREFLIGHT_OK build=%s artifact=%s hf_home=%s\n' \
    "$BRIDGE_BUILD_DIR" "$BRIDGE_ARTIFACT" "$BRIDGE_HF_HOME"
  exit 0
fi

mkdir -p "$CLUSTER_DIR/logs" "$BRIDGE_BUILD_PARENT" "$(dirname "$BRIDGE_ARTIFACT")" "$JOB_RECORD_DIR" "$BRIDGE_HF_HOME"
umask 077
ENVIRONMENT_MARKER_SHA256=$(file_sha256 "$ENV_READY") || fail "cannot hash environment-ready marker"

PHASES=(
  corpus
  teacher
  student-group-0
  student-group-1
  student-group-2
  student-group-3
  student-group-4
  student-group-5
  student-group-6
  publish
  stage2-smoke
)
SCRIPTS=(
  "$NEXUS_DIR/oprd_bridge_corpus.sbatch"
  "$NEXUS_DIR/oprd_bridge_teacher.sbatch"
  "$NEXUS_DIR/oprd_bridge_student_groups.sbatch"
  "$NEXUS_DIR/oprd_bridge_student_groups.sbatch"
  "$NEXUS_DIR/oprd_bridge_student_groups.sbatch"
  "$NEXUS_DIR/oprd_bridge_student_groups.sbatch"
  "$NEXUS_DIR/oprd_bridge_student_groups.sbatch"
  "$NEXUS_DIR/oprd_bridge_student_groups.sbatch"
  "$NEXUS_DIR/oprd_bridge_student_groups.sbatch"
  "$NEXUS_DIR/oprd_bridge_publish.sbatch"
  "$NEXUS_DIR/oprd_bridge_stage2_smoke.sbatch"
)
[[ ${#PHASES[@]} -eq 11 && ${#SCRIPTS[@]} -eq 11 ]] || fail "internal DAG phase count changed"

RECORD_INITIALIZED=0
FRESH_CLAIM_CREATED=0
SUBMITTED_COUNT=0
SUBMISSION_INTENT=0
record_failure() {
  local rc=$?
  trap - EXIT
  if [[ "$RECORD_INITIALIZED" == 1 ]]; then
    "$PYTHON" "$OPS" mark-submission-failed \
      --record "$JOB_RECORD" \
      --error "submitter exited with status $rc" >/dev/null 2>&1 || true
  fi
  if [[ "$FRESH_CLAIM_CREATED" == 1 && "$SUBMITTED_COUNT" == 0 && "$SUBMISSION_INTENT" == 0 ]]; then
    "$PYTHON" "$OPS" release-claim \
      --scratch "$SCRATCH_ROOT" \
      --artifact "$BRIDGE_ARTIFACT" \
      --run-id "$BRIDGE_RUN_ID" \
      --record "$JOB_RECORD" >/dev/null 2>&1 || true
  fi
  printf 'Submission stopped. Job record: %s\n' "$JOB_RECORD" >&2
  exit "$rc"
}
trap record_failure EXIT

if [[ "$MODE" == fresh ]]; then
  [[ ! -e "$BRIDGE_BUILD_DIR" ]] || fail "build directory already exists: $BRIDGE_BUILD_DIR"
  [[ ! -e "$BRIDGE_ARTIFACT" ]] || fail "canonical artifact already exists: $BRIDGE_ARTIFACT"
  [[ ! -e "$JOB_RECORD" ]] || fail "job record already exists: $JOB_RECORD"
  "$PYTHON" "$OPS" init-record \
    --scratch "$SCRATCH_ROOT" \
    --record "$JOB_RECORD" \
    --run-id "$BRIDGE_RUN_ID" \
    --build-dir "$BRIDGE_BUILD_DIR" \
    --artifact "$BRIDGE_ARTIFACT" \
    --config "root=$ROOT" \
    --config "python=$PYTHON" \
    --config "bridge_builder=$BRIDGE_BUILDER" \
    --config "hf_home=$BRIDGE_HF_HOME" \
    --config "bridge_smoke=$BRIDGE_SMOKE" \
    --config "allow_paper_scale=$ALLOW_PAPER_SCALE" \
    --config "bridge_pair=$BRIDGE_PAIR" \
    --config "bridge_rank=$BRIDGE_RANK" \
    --config "teacher=$TEACHER" \
    --config "student=$STUDENT" \
    --config "tokenizer=$TOKENIZER" \
    --config "prompts=$PROMPTS" \
    --config "submit_min_free_gib=$SUBMIT_MIN_FREE_GIB" \
    --config "submit_min_free_inodes=$SUBMIT_MIN_FREE_INODES" \
    --config "allow_quota_unavailable=$ALLOW_QUOTA_UNAVAILABLE" \
    --config "environment_marker_sha256=$ENVIRONMENT_MARKER_SHA256" \
    --config "source_bundle_sha256=$SOURCE_BUNDLE_SHA256" >/dev/null
  RECORD_INITIALIZED=1
  "$PYTHON" "$OPS" claim \
    --scratch "$SCRATCH_ROOT" \
    --artifact "$BRIDGE_ARTIFACT" \
    --run-id "$BRIDGE_RUN_ID" \
    --record "$JOB_RECORD" >/dev/null
  FRESH_CLAIM_CREATED=1
  START_INDEX=0
  ln -sfn "$(basename "$JOB_RECORD")" "$JOB_RECORD_DIR/oprd_bridge_latest.json"
else
  RECORD_INITIALIZED=1
  [[ -d "$BRIDGE_BUILD_DIR" ]] || fail "resume build directory is missing: $BRIDGE_BUILD_DIR"
  RESUME_JSON=$("$PYTHON" "$OPS" resume-index --record "$JOB_RECORD")
  START_INDEX=$(printf '%s' "$RESUME_JSON" | "$PYTHON" -c 'import json,sys; print(json.load(sys.stdin)["resume_index"])')
  [[ "$START_INDEX" =~ ^[0-9]+$ && "$START_INDEX" -le 11 ]] || fail "invalid resume index: $START_INDEX"
  if [[ "$START_INDEX" == 11 ]]; then
    trap - EXIT
    release_successful_claim "$JOB_RECORD"
    printf 'DAG is already complete: %s\n' "$JOB_RECORD"
    exit 0
  fi
  CLAIM_ARGS=(
    claim
    --scratch "$SCRATCH_ROOT"
    --artifact "$BRIDGE_ARTIFACT"
    --run-id "$BRIDGE_RUN_ID"
    --record "$JOB_RECORD"
  )
  if (( START_INDEX >= 9 )) && [[ -d "$BRIDGE_ARTIFACT" ]]; then
    CLAIM_ARGS+=(--allow-existing-artifact --build-dir "$BRIDGE_BUILD_DIR")
  fi
  "$PYTHON" "$OPS" "${CLAIM_ARGS[@]}" >/dev/null
  printf 'Resuming at phase %s from record %s\n' "${PHASES[$START_INDEX]}" "$JOB_RECORD"
fi

EXPORTS="SCRATCH_ROOT=$SCRATCH_ROOT,ROOT=$ROOT,PYTHON=$PYTHON,BRIDGE_BUILDER=$BRIDGE_BUILDER,BRIDGE_BUILD_DIR=$BRIDGE_BUILD_DIR,BRIDGE_ARTIFACT=$BRIDGE_ARTIFACT,BRIDGE_HF_HOME=$BRIDGE_HF_HOME,HF_HOME=$BRIDGE_HF_HOME,BRIDGE_RUN_ID=$BRIDGE_RUN_ID,JOB_RECORD=$JOB_RECORD,BRIDGE_SMOKE=$BRIDGE_SMOKE,BRIDGE_PAIR=$BRIDGE_PAIR,BRIDGE_RANK=$BRIDGE_RANK,TEACHER=$TEACHER,STUDENT=$STUDENT,TOKENIZER=$TOKENIZER,PROMPTS=$PROMPTS"

PREVIOUS_JOB_ID=
for ((index = START_INDEX; index < ${#PHASES[@]}; index++)); do
  phase=${PHASES[$index]}
  script=${SCRIPTS[$index]}
  PHASE_EXPORTS=$EXPORTS
  if (( index >= 2 && index <= 8 )); then
    PHASE_EXPORTS="$EXPORTS,GROUP_INDEX=$((index - 2))"
  fi
  SBATCH_ARGS=(--export="$PHASE_EXPORTS" --comment="oprd:$BRIDGE_RUN_ID:$phase")
  if [[ -n "$PREVIOUS_JOB_ID" ]]; then
    SBATCH_ARGS+=(--dependency="afterok:$PREVIOUS_JOB_ID")
  fi
  INTENT_ARGS=(
    begin-job
    --record "$JOB_RECORD"
    --phase "$phase"
    --script "$script"
  )
  if [[ -n "$PREVIOUS_JOB_ID" ]]; then
    INTENT_ARGS+=(--dependency-job-id "$PREVIOUS_JOB_ID")
  fi
  "$PYTHON" "$OPS" "${INTENT_ARGS[@]}" >/dev/null
  SUBMISSION_INTENT=1
  JOB_ID=$(submit_job "${SBATCH_ARGS[@]}" "$script")
  SUBMITTED_COUNT=$((SUBMITTED_COUNT + 1))
  RECORD_ARGS=(
    record-job
    --record "$JOB_RECORD"
    --phase "$phase"
    --job-id "$JOB_ID"
    --script "$script"
  )
  if [[ -n "$PREVIOUS_JOB_ID" ]]; then
    RECORD_ARGS+=(--dependency-job-id "$PREVIOUS_JOB_ID")
  fi
  "$PYTHON" "$OPS" "${RECORD_ARGS[@]}" >/dev/null
  SUBMISSION_INTENT=0
  if [[ -n "$PREVIOUS_JOB_ID" ]]; then
    printf 'Submitted %s job %s afterok:%s\n' "$phase" "$JOB_ID" "$PREVIOUS_JOB_ID"
  else
    printf 'Submitted %s job %s\n' "$phase" "$JOB_ID"
  fi
  PREVIOUS_JOB_ID=$JOB_ID
done

"$PYTHON" "$OPS" mark-submitted --record "$JOB_RECORD" >/dev/null
trap - EXIT
printf 'Job record: %s\n' "$JOB_RECORD"
printf 'Submission is complete; DAG terminal status is not yet known.\n'
printf 'Run: %s --status %q\n' "$0" "$JOB_RECORD"
printf 'DAG stops after the Stage-2 smoke. Production Tier-2 seeds require separate approval.\n'
