#!/bin/bash
#SBATCH --job-name=oprd-t2-corrected
#SBATCH --partition=gpu            # EDIT: your partition
##SBATCH --account=REPLACE_ME     # EDIT: uncomment + set your allocation account
#SBATCH --gres=gpu:a100:1          # EDIT if your site names GPUs differently (4B teacher + 1.7B student)
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --array=0-35               # 6 arms x 2 pairs x 3 seeds
#SBATCH --output=logs/t2_corrected_%A_%a.out

# Tier 2 - corrected all-layer cross-architecture bridge bake-off, 3 seeds.
# Every representation arm supervises all student transformer blocks; unequal depths use
# OPRD Eq. 11 proportional mapping. Long rollouts + final-2k supervision replace the old,
# confounded 2k-truncated protocol. Smoke/profile one task before submitting the full array.
# The cross-tokenizer Phi pair remains skipped by default.

set -euo pipefail

# ---- EDIT: environment ----------------------------------------------------
# Compute nodes need python + torch/transformers/datasets/vllm on PATH and
# (usually) internet or a pre-populated HF cache. Typical setup:
#   module load cuda python            # your site's module names
#   source $HOME/oprd-venv/bin/activate
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}   # point at scratch if home quota is small
# ---------------------------------------------------------------------------

ARMS=(oprd_bridge linear_map_mse minilmv2_selfrel ln_dist soft_knn_kl cka_gram)
PAIRS=(qwen3_4b_to_1p7b phi4mini_to_qwen3_1p7b)
SEEDS=(0 1 2)
i=$SLURM_ARRAY_TASK_ID
ARM=${ARMS[$((i % 6))]}
PAIR=${PAIRS[$(((i / 6) % 2))]}
SEED=${SEEDS[$((i / 12))]}

# ---- corrected paper-comparable protocol -----------------------------------
SUPERVISED_LAYERS=${SUPERVISED_LAYERS:-all}
ROLLOUT_TOKENS=${ROLLOUT_TOKENS:-16000}
SUPERVISE_LAST_K=${SUPERVISE_LAST_K:-2000}
EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-24000}
STEPS=${STEPS:-2000}
RESULT_NS=${RESULT_NS:-t2_corrected_all}
PROMPTS=${PROMPTS:-BytedTsinghua-SIA/DAPO-Math-17k}
DATASET_SPLIT=${DATASET_SPLIT:-train}
PROMPT_KEY=${PROMPT_KEY:-prompt}
# M1 NO-GO: soft_knn_kl (identity) is control-only; drop it entirely via SKIP_ARMS if tight:
if [[ "${SKIP_ARMS:-}" == *"$ARM"* ]]; then echo "arm $ARM skipped by SKIP_ARMS"; exit 0; fi
# cross-tokenizer pair skipped BY DEFAULT — the harness does not support it (no OT bridge yet).
# Unset SKIP_PAIRS only after implementing cross-tokenizer alignment.
SKIP_PAIRS=${SKIP_PAIRS-"phi4mini_to_qwen3_1p7b"}
if [[ "$SKIP_PAIRS" == *"$PAIR"* ]]; then echo "pair $PAIR skipped by SKIP_PAIRS"; exit 0; fi

case $PAIR in
  qwen3_4b_to_1p7b)   TEACHER=Qwen/Qwen3-4B;                 STUDENT=Qwen/Qwen3-1.7B-Base ;;
  phi4mini_to_qwen3_1p7b) TEACHER=microsoft/Phi-4-mini-reasoning; STUDENT=Qwen/Qwen3-1.7B-Base ;;
esac

: "${STUDENT_REVISION:?Tier-2 requires STUDENT_REVISION for every arm}"
: "${TEACHER_REVISION:?Tier-2 requires TEACHER_REVISION for every arm}"
: "${TOKENIZER:?Tier-2 requires TOKENIZER for every arm}"
: "${TOKENIZER_REVISION:?Tier-2 requires TOKENIZER_REVISION for every arm}"
: "${DATASET_REVISION:?Tier-2 requires DATASET_REVISION for every arm}"
COMMON_IDENTITY_ARGS=(
  --student-revision "$STUDENT_REVISION"
  --teacher-revision "$TEACHER_REVISION"
  --tokenizer "$TOKENIZER"
  --tokenizer-revision "$TOKENIZER_REVISION"
  --dataset-revision "$DATASET_REVISION"
  --dataset-split "$DATASET_SPLIT"
  --prompt-key "$PROMPT_KEY"
)

BRIDGE_ARGS=()
if [[ "$ARM" == oprd_bridge ]]; then
  : "${BRIDGE_ARTIFACT:?oprd_bridge requires BRIDGE_ARTIFACT}"
  : "${BRIDGE_MANIFEST_SHA256:?oprd_bridge requires BRIDGE_MANIFEST_SHA256}"
  : "${BRIDGE_STAGE1_CONTRACT_SHA256:?oprd_bridge requires BRIDGE_STAGE1_CONTRACT_SHA256}"
  EXPECTED_PRODUCTION_CONTRACT_SHA256=$(python -c \
    'from oprd_bridge import stage1_contract_sha256; print(stage1_contract_sha256())')
  [[ "$BRIDGE_STAGE1_CONTRACT_SHA256" == "$EXPECTED_PRODUCTION_CONTRACT_SHA256" ]] || {
    printf 'ERROR: production Tier-2 refuses a non-production Stage-1 contract\n' >&2
    exit 1
  }
  BRIDGE_ARGS=(
    --bridge-artifact "$BRIDGE_ARTIFACT"
    --bridge-manifest-sha256 "$BRIDGE_MANIFEST_SHA256"
    --bridge-stage1-contract-sha256 "$BRIDGE_STAGE1_CONTRACT_SHA256"
  )
fi

srun python train_harness.py \
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
