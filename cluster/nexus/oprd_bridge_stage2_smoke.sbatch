#!/bin/bash
#SBATCH --job-name=oprd-bridge-s2-smoke
#SBATCH --partition=cml-furongh
#SBATCH --account=cml-furongh
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:h200-sxm:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/cmlscratch/%u/oprd-experiments/cluster/logs/oprd_bridge_stage2_smoke_%j.out

# Two-step backward smoke that consumes the verified external artifact.
set -euo pipefail

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

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
ATOMIC_DIRECTORY=$CLUSTER_DIR/oprd_atomic_directory.py
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
BRIDGE_BUILD_DIR=${BRIDGE_BUILD_DIR:?submit_oprd_bridge.sh must export BRIDGE_BUILD_DIR}
BRIDGE_ARTIFACT=${BRIDGE_ARTIFACT:?submit_oprd_bridge.sh must export BRIDGE_ARTIFACT}
BRIDGE_HF_HOME=${BRIDGE_HF_HOME:?submit_oprd_bridge.sh must export BRIDGE_HF_HOME}
BRIDGE_RUN_ID=${BRIDGE_RUN_ID:?submit_oprd_bridge.sh must export BRIDGE_RUN_ID}
JOB_RECORD=${JOB_RECORD:?submit_oprd_bridge.sh must export JOB_RECORD}
SMOKE_OUT=$BRIDGE_BUILD_DIR/stage2-backward-smoke
CONTRACT=$BRIDGE_BUILD_DIR/contract.json
VERIFY=$BRIDGE_BUILD_DIR/verify.json
ENV_READY=$ROOT/venv/.oprd-env-ready.json

[[ -x "$PYTHON" ]] || fail "venv Python is missing or not executable: $PYTHON"
[[ -r "$ATOMIC_DIRECTORY" ]] ||
  fail "atomic directory helper is missing or unreadable: $ATOMIC_DIRECTORY"
[[ -r "$ENV_READY" ]] || fail "environment-ready marker is missing: $ENV_READY"
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 12), sys.version' || fail "venv Python is not 3.12"
[[ -r "$CONTRACT" ]] || fail "bridge contract is missing: $CONTRACT"
[[ -r "$VERIFY" ]] || fail "bridge verification report is missing: $VERIFY"
case "$SMOKE_OUT/" in "$SCRATCH_ROOT"/*) ;; *) fail "smoke output must be on scratch: $SMOKE_OUT" ;; esac
case "$BRIDGE_HF_HOME/" in "$SCRATCH_ROOT"/*) ;; *) fail "HF cache must be on scratch: $BRIDGE_HF_HOME" ;; esac
case "$VLLM_CACHE_ROOT/" in "$SCRATCH_ROOT"/*) ;; *) fail "vLLM cache must be on scratch: $VLLM_CACHE_ROOT" ;; esac
case "$TRITON_CACHE_DIR/" in "$SCRATCH_ROOT"/*) ;; *) fail "Triton cache must be on scratch: $TRITON_CACHE_DIR" ;; esac
case "$FLASHINFER_WORKSPACE_BASE/" in "$SCRATCH_ROOT"/*) ;; *) fail "FlashInfer workspace must be on scratch: $FLASHINFER_WORKSPACE_BASE" ;; esac
"$PYTHON" "$OPS" check-runtime \
  --scratch "$SCRATCH_ROOT" \
  --path "$BRIDGE_BUILD_DIR" \
  --path "$BRIDGE_HF_HOME" \
  --path "$VLLM_CACHE_ROOT" \
  --path "$TRITON_CACHE_DIR" \
  --path "$FLASHINFER_WORKSPACE_BASE" \
  --path "$SMOKE_OUT" \
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
"$PYTHON" "$OPS" validate-published --build-dir "$BRIDGE_BUILD_DIR" --artifact "$BRIDGE_ARTIFACT"

BRIDGE_METADATA=$("$PYTHON" - "$CONTRACT" "$VERIFY" "$BRIDGE_ARTIFACT/build_config.json" <<'PY'
import json
import pathlib
import sys

contract = json.loads(pathlib.Path(sys.argv[1]).read_text())
verify = json.loads(pathlib.Path(sys.argv[2]).read_text())
artifact_contract = json.loads(pathlib.Path(sys.argv[3]).read_text())
if artifact_contract != contract:
    raise SystemExit("workspace contract does not match artifact build_config.json")
if verify.get("status") != "ok":
    raise SystemExit("verify.json does not report status=ok")
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
    verify["artifact_manifest_sha256"],
]
if any(not isinstance(value, str) or not value for value in values):
    raise SystemExit("bridge contract or verification report has an empty identity field")
print("\n".join(values))
PY
) || fail "bridge contract and verification report validation failed"
mapfile -t BRIDGE_VALUES <<<"$BRIDGE_METADATA"
[[ ${#BRIDGE_VALUES[@]} -eq 12 ]] || fail "unexpected bridge metadata field count"

cd "$CLUSTER_DIR"
export HF_HOME="$BRIDGE_HF_HOME"
mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" \
  "$FLASHINFER_WORKSPACE_BASE" "$(dirname "$SMOKE_OUT")"
SMOKE_RESTART_COUNT=${SLURM_RESTART_COUNT:-0}
SMOKE_ATTEMPT_ID=${SLURM_JOB_ID:-$$}.$SMOKE_RESTART_COUNT
SMOKE_STAGING=$BRIDGE_BUILD_DIR/.stage2-backward-smoke.$SMOKE_ATTEMPT_ID
SMOKE_QUARANTINE=$BRIDGE_BUILD_DIR/.stage2-backward-smoke.invalid.$SMOKE_ATTEMPT_ID
"$PYTHON" - "$SMOKE_OUT" "$SMOKE_STAGING" "$SMOKE_QUARANTINE" <<'PY'
import pathlib
import sys
paths = [pathlib.Path(value).resolve(strict=False) for value in sys.argv[1:]]
canonical, staging, quarantine = paths
if staging == canonical or quarantine == canonical or staging == quarantine:
    raise SystemExit("smoke output, staging, and quarantine paths must be distinct")
if staging.parent != canonical.parent or quarantine.parent != canonical.parent:
    raise SystemExit("smoke staging and quarantine must be siblings of canonical output")
PY
"$PYTHON" "$OPS" check-runtime \
  --scratch "$SCRATCH_ROOT" \
  --path "$SMOKE_STAGING" \
  --path "$SMOKE_QUARANTINE" \
  --artifact "$BRIDGE_ARTIFACT" \
  --run-id "$BRIDGE_RUN_ID" \
  --record "$JOB_RECORD" \
  --cuda-home "$BRIDGE_CUDA_HOME" \
  --cuda-nvcc "$BRIDGE_CUDA_NVCC" \
  --runtime-ld-library-path "$BRIDGE_RUNTIME_LD_LIBRARY_PATH" \
  --flashinfer-workspace-base "$FLASHINFER_WORKSPACE_BASE"
[[ ! -e "$SMOKE_STAGING" ]] || fail "smoke staging already exists: $SMOKE_STAGING"
[[ ! -e "$SMOKE_QUARANTINE" ]] || fail "smoke quarantine already exists: $SMOKE_QUARANTINE"
if [[ -e "$SMOKE_OUT" ]]; then
  [[ -d "$SMOKE_OUT" && ! -L "$SMOKE_OUT" ]] ||
    fail "published smoke output is not a real directory: $SMOKE_OUT"
  if "$PYTHON" "$OPS" validate-success --build-dir "$BRIDGE_BUILD_DIR" --artifact "$BRIDGE_ARTIFACT"; then
    printf 'STAGE2_SMOKE_ALREADY_VALID out=%s manifest=%s\n' "$SMOKE_OUT" "${BRIDGE_VALUES[11]}"
    exit 0
  fi
  "$PYTHON" "$ATOMIC_DIRECTORY" move \
    --source "$SMOKE_OUT" \
    --destination "$SMOKE_QUARANTINE" \
    --trusted-root "$SCRATCH_ROOT" ||
    fail "cannot quarantine invalid canonical smoke output"
fi

"$PYTHON" - "$SCRATCH_ROOT" "$FLASHINFER_WORKSPACE_BASE" <<'PY'
import pathlib
import sys

scratch = pathlib.Path(sys.argv[1]).resolve(strict=True)
workspace = pathlib.Path(sys.argv[2]).resolve(strict=True)
workspace.relative_to(scratch)
from flashinfer.sampling import get_sampling_module

get_sampling_module()
from flashinfer.jit import env as jit_env

for name in ("FLASHINFER_CACHE_DIR", "FLASHINFER_WORKSPACE_DIR", "FLASHINFER_JIT_DIR"):
    pathlib.Path(getattr(jit_env, name)).resolve(strict=False).relative_to(workspace)
print(f"FLASHINFER_RUNTIME_OK workspace={workspace}")
PY

srun "$PYTHON" train_harness.py \
  --loss oprd_bridge \
  --teacher "${BRIDGE_VALUES[2]}" \
  --teacher-revision "${BRIDGE_VALUES[3]}" \
  --student "${BRIDGE_VALUES[0]}" \
  --student-revision "${BRIDGE_VALUES[1]}" \
  --tokenizer "${BRIDGE_VALUES[4]}" \
  --tokenizer-revision "${BRIDGE_VALUES[5]}" \
  --prompts "${BRIDGE_VALUES[6]}" \
  --dataset-revision "${BRIDGE_VALUES[7]}" \
  --dataset-split "${BRIDGE_VALUES[8]}" \
  --prompt-key "${BRIDGE_VALUES[9]}" \
  --n-prompts 8 \
  --supervised-layers all \
  --bridge-artifact "$BRIDGE_ARTIFACT" \
  --bridge-manifest-sha256 "${BRIDGE_VALUES[11]}" \
  --bridge-stage1-contract-sha256 "${BRIDGE_VALUES[10]}" \
  --steps 2 \
  --batch-tokens 2048 \
  --lr 1e-5 \
  --seed 0 \
  --rollout-tokens 4096 \
  --supervise-last-k 2000 \
  --require-finished \
  --rollout-refresh 0 \
  --max-tokens 512 \
  --backend vllm \
  --out "$SMOKE_STAGING"

"$PYTHON" - "$SMOKE_STAGING" "${BRIDGE_VALUES[11]}" "${BRIDGE_VALUES[7]}" <<'PY'
import hashlib
import json
import math
import os
import pathlib
import sys
import tempfile

import torch


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


out = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
expected_dataset_revision = sys.argv[3]
results_path = out / "results.json"
state_path = out / "train_state_last.pt"
results = json.loads(results_path.read_text())
state = torch.load(state_path, map_location="cpu")
losses = results.get("losses", [])
if len(losses) != 2:
    raise SystemExit("Stage-2 smoke did not record exactly two optimizer losses")
if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in losses):
    raise SystemExit("Stage-2 smoke recorded a non-finite optimizer loss")
if state.get("losses") != losses:
    raise SystemExit("training state losses do not match results.json")
student = state.get("student")
if not isinstance(student, dict) or not student:
    raise SystemExit("training state has no student tensors")
for name, tensor in student.items():
    if not torch.is_tensor(tensor) or not bool(torch.isfinite(tensor).all().item()):
        raise SystemExit(f"training state has a non-finite student tensor: {name}")
if results.get("bridge_reference", {}).get("manifest_sha256") != expected:
    raise SystemExit("results.json has the wrong bridge manifest identity")
if results.get("start_step") != 0 or results.get("resumed_from") is not None:
    raise SystemExit("Stage-2 smoke must be a fresh uninterrupted two-step run")
expected_stage2_identity = {
    "prompts": "BytedTsinghua-SIA/DAPO-Math-17k",
    "dataset_revision": expected_dataset_revision,
    "dataset_split": "train",
    "prompt_key": "prompt",
    "n_prompts": 8,
    "seed": 0,
    "rollout_tokens": 4096,
    "max_tokens": 512,
    "supervise_last_k": 2000,
    "require_finished": True,
    "rollout_refresh": 0,
    "batch_tokens": 2048,
    "learning_rate": 1e-5,
    "bridge_loss_coefficient": 10.0,
    "backend": "vllm",
}
actual_stage2_identity = results["bridge_reference"].get("stage2_identity")
if actual_stage2_identity != expected_stage2_identity:
    raise SystemExit("results.json has the wrong Stage-2 smoke identity")
if state.get("step") != 2:
    raise SystemExit("training state did not commit step 2")
if state.get("bridge_reference", {}).get("manifest_sha256") != expected:
    raise SystemExit("training state has the wrong bridge manifest identity")
if state.get("bridge_reference") != results.get("bridge_reference"):
    raise SystemExit("training state and results disagree on bridge identity")
stage2_identity_sha256 = hashlib.sha256(
    json.dumps(
        actual_stage2_identity, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
).hexdigest()
marker = {
    "schema_version": 2,
    "status": "ok",
    "artifact_manifest_sha256": expected,
    "stage2_identity_sha256": stage2_identity_sha256,
    "fresh_uninterrupted": True,
    "state_step": 2,
    "student_tensors_finite": True,
    "results_sha256": sha256_file(results_path),
    "train_state_last_sha256": sha256_file(state_path),
}
destination = out / "SMOKE_COMPLETE.json"
if destination.exists():
    existing = json.loads(destination.read_text())
    if existing != marker:
        raise SystemExit("existing SMOKE_COMPLETE.json does not match current outputs")
else:
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=out)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(marker, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        pathlib.Path(temporary_name).unlink(missing_ok=True)
print(f"STAGE2_SMOKE_OK out={out} manifest={expected} marker={destination}")
PY

"$PYTHON" "$ATOMIC_DIRECTORY" move \
  --source "$SMOKE_STAGING" \
  --destination "$SMOKE_OUT" \
  --trusted-root "$SCRATCH_ROOT" ||
  fail "cannot atomically publish validated Stage-2 smoke output"
"$PYTHON" "$OPS" validate-success \
  --build-dir "$BRIDGE_BUILD_DIR" \
  --artifact "$BRIDGE_ARTIFACT" ||
  fail "published Stage-2 smoke output failed canonical validation"
printf 'STAGE2_SMOKE_PUBLISHED out=%s manifest=%s\n' "$SMOKE_OUT" "${BRIDGE_VALUES[11]}"
