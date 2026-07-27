#!/bin/bash
#SBATCH --job-name=oprd-t2-corrected
#SBATCH --partition=cml-furongh
#SBATCH --account=cml-furongh
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:h200-sxm:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --array=0-35%4
#SBATCH --output=/cmlscratch/%u/oprd-experiments/cluster/logs/t2_corrected_%A_%a.out

# Corrected all-layer Tier 2. OPRD tasks consume one independently verified artifact.
set -euo pipefail

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

source /etc/profile.d/modules.sh 2>/dev/null || true
module use /opt/local/stow/.modulefiles
module load Python3/3.12.3
module use /opt/common/.modulefiles
module load cuda/12.8.1
module load ffmpeg/7.1 || module load ffmpeg/6.1 || true

SCRATCH_ROOT=${SCRATCH_ROOT:-/cmlscratch/$USER}
ROOT=${ROOT:-$SCRATCH_ROOT/oprd-experiments}
CLUSTER_DIR=$ROOT/cluster
PYTHON=${PYTHON:-$ROOT/venv/bin/python}
cd "$CLUSTER_DIR"
mkdir -p logs runs
export HF_HOME=${HF_HOME:-$SCRATCH_ROOT/hf-cache}
[[ -x "$PYTHON" ]] || fail "venv Python is missing or not executable: $PYTHON"

ARMS=(oprd_bridge linear_map_mse minilmv2_selfrel ln_dist soft_knn_kl cka_gram)
PAIRS=(qwen3_4b_to_1p7b phi4mini_to_qwen3_1p7b)
SEEDS=(0 1 2)
i=$SLURM_ARRAY_TASK_ID
ARM=${ARMS[$((i % 6))]}
PAIR=${PAIRS[$(((i / 6) % 2))]}
SEED=${SEEDS[$((i / 12))]}

SUPERVISED_LAYERS=${SUPERVISED_LAYERS:-all}
ROLLOUT_TOKENS=${ROLLOUT_TOKENS:-16000}
SUPERVISE_LAST_K=${SUPERVISE_LAST_K:-2000}
EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-24000}
STEPS=${STEPS:-2000}
RESULT_NS=${RESULT_NS:-t2_corrected_all}
PROMPTS=${PROMPTS:-BytedTsinghua-SIA/DAPO-Math-17k}
if [[ "${SKIP_ARMS:-}" == *"$ARM"* ]]; then echo "arm $ARM skipped by SKIP_ARMS"; exit 0; fi
SKIP_PAIRS=${SKIP_PAIRS-"phi4mini_to_qwen3_1p7b"}
if [[ "$SKIP_PAIRS" == *"$PAIR"* ]]; then echo "pair $PAIR skipped by SKIP_PAIRS"; exit 0; fi

case $PAIR in
  qwen3_4b_to_1p7b) TEACHER=Qwen/Qwen3-4B; STUDENT=Qwen/Qwen3-1.7B-Base ;;
  phi4mini_to_qwen3_1p7b) TEACHER=microsoft/Phi-4-mini-reasoning; STUDENT=Qwen/Qwen3-1.7B-Base ;;
  *) fail "unknown model pair: $PAIR" ;;
esac

BRIDGE_BUILD_DIR=${BRIDGE_BUILD_DIR:?Tier-2 requires BRIDGE_BUILD_DIR for pinned identity}
CONTRACT=$BRIDGE_BUILD_DIR/contract.json
[[ -r "$CONTRACT" ]] || fail "bridge contract is missing: $CONTRACT"

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
[[ "${CONTRACT_VALUES[0]}" == "$STUDENT" ]] || fail "contract student does not match Tier-2 pair"
[[ "${CONTRACT_VALUES[2]}" == "$TEACHER" ]] || fail "contract teacher does not match Tier-2 pair"
PROMPTS=${CONTRACT_VALUES[6]}
COMMON_IDENTITY_ARGS=(
  --student-revision "${CONTRACT_VALUES[1]}"
  --teacher-revision "${CONTRACT_VALUES[3]}"
  --tokenizer "${CONTRACT_VALUES[4]}"
  --tokenizer-revision "${CONTRACT_VALUES[5]}"
  --dataset-revision "${CONTRACT_VALUES[7]}"
  --dataset-split "${CONTRACT_VALUES[8]}"
  --prompt-key "${CONTRACT_VALUES[9]}"
)
EXPECTED_PRODUCTION_CONTRACT_SHA256=$("$PYTHON" -c \
  'from oprd_bridge import stage1_contract_sha256; print(stage1_contract_sha256())')
[[ "${CONTRACT_VALUES[10]}" == "$EXPECTED_PRODUCTION_CONTRACT_SHA256" ]] ||
  fail "production Tier-2 refuses a non-production Stage-1 contract"

BRIDGE_ARGS=()
if [[ "$ARM" == oprd_bridge ]]; then
  BRIDGE_ARTIFACT=${BRIDGE_ARTIFACT:?oprd_bridge requires BRIDGE_ARTIFACT}
  VERIFY=$BRIDGE_BUILD_DIR/verify.json
  [[ -r "$VERIFY" ]] || fail "bridge verification report is missing: $VERIFY"
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
  BRIDGE_ARGS=(
    --bridge-artifact "$BRIDGE_ARTIFACT"
    --bridge-manifest-sha256 "$BRIDGE_MANIFEST_SHA256"
    --bridge-stage1-contract-sha256 "${CONTRACT_VALUES[10]}"
  )
fi

srun "$PYTHON" train_harness.py \
  --loss "$ARM" \
  --teacher "$TEACHER" --student "$STUDENT" \
  --prompts "$PROMPTS" --n-prompts 4000 \
  --supervised-layers "$SUPERVISED_LAYERS" \
  --steps "$STEPS" --batch-tokens 16384 --lr 1e-5 --seed "$SEED" \
  --rollout-tokens "$ROLLOUT_TOKENS" \
  --supervise-last-k "$SUPERVISE_LAST_K" \
  --require-finished \
  --max-tokens "$EVAL_MAX_TOKENS" \
  --eval aime24,aime25 --eval-avg-at 16 \
  --diagnostics calibrated_mknn,cka,dist_n \
  "${COMMON_IDENTITY_ARGS[@]}" \
  "${BRIDGE_ARGS[@]}" \
  --out "runs/${RESULT_NS}_${PAIR}_${ARM}_s${SEED}"
