#!/bin/bash
#SBATCH --job-name=oprd-t2-bridge
#SBATCH --partition=gpu            # EDIT: your partition
##SBATCH --account=REPLACE_ME     # EDIT: uncomment + set your allocation account
#SBATCH --gres=gpu:a100:1          # EDIT if your site names GPUs differently (4B teacher + 1.7B student)
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --array=0-35               # 6 arms x 2 pairs x 3 seeds
#SBATCH --output=logs/t2_bridge_%A_%a.out

# Tier 2 — six-arm cross-architecture bridge bake-off (adversarial review §5.7),
# OPRD-Bridge Table 4 (Qwen3-4B -> Qwen3-1.7B-Base, shared vocab) and Table 6
# (Phi-4-mini-reasoning -> Qwen3-1.7B-Base, cross-tokenizer) setups, 3 seeds.
# STATUS: never submitted to a real queue — smoke one task first (see cluster/README.md).
# NOTE: with the default SKIP_PAIRS below, 18 of 36 tasks exit immediately (phi pair
# unsupported), so the as-shipped submission is ~180 GPU-h, not 360.

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

# ---- set from the ACTUAL Tier-0 verdicts, 2026-07-11 (see cluster/README.md) ----
SUPERVISED_LAYER_FRACS=${SUPERVISED_LAYER_FRACS:-"0.25,0.5"}
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

srun python train_harness.py \
  --loss "$ARM" \
  --teacher "$TEACHER" --student "$STUDENT" \
  --prompts BytedTsinghua-SIA/DAPO-Math-17k --n-prompts 4000 \
  --supervised-layer-fracs "$SUPERVISED_LAYER_FRACS" \
  --steps 2000 --batch-tokens 16384 --lr 1e-5 --seed "$SEED" \
  --eval aime24,aime25 --eval-avg-at 16 \
  --diagnostics calibrated_mknn,cka,dist_n \
  --out "runs/t2_${PAIR}_${ARM}_s${SEED}"
