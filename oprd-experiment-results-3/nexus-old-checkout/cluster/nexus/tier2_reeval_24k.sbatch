#!/bin/bash
#SBATCH --job-name=oprd-t2-reeval
#SBATCH --partition=cml-furongh
#SBATCH --account=cml-furongh
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:h200-sxm:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-18%8
#SBATCH --output=/cmlscratch/%u/oprd-experiments/cluster/logs/t2_reeval24k_%A_%a.out

# Tier 2 re-eval at --max-tokens 24000 (was 2000 at train time; tier2-results.md
# called for 12k — bumped to 24k). Tasks 0-17: eval-only on trained ckpts.
# Task 18: untrained Qwen3-1.7B-Base baseline (--steps 0).
set -euo pipefail
# vLLM/flashinfer JIT needs nvcc (same preamble as working tier2 train jobs)
source /etc/profile.d/modules.sh 2>/dev/null || true
module use /opt/local/stow/.modulefiles
module load Python3/3.12.3
module use /opt/common/.modulefiles
module load cuda/12.8.1
module load ffmpeg/7.1 || module load ffmpeg/6.1 || true
ROOT=/cmlscratch/$USER/oprd-experiments
cd "$ROOT/cluster"
mkdir -p logs runs
export HF_HOME=/cmlscratch/$USER/hf-cache
# shellcheck disable=SC1091
source "$ROOT/venv/bin/activate"

ARMS=(oprd_bridge linear_map_mse minilmv2_selfrel ln_dist soft_knn_kl cka_gram)
SEEDS=(0 1 2)
TEACHER=Qwen/Qwen3-4B
STUDENT=Qwen/Qwen3-1.7B-Base
PAIR=qwen3_4b_to_1p7b
MAX_TOKENS=24000
i=$SLURM_ARRAY_TASK_ID

if [[ "$i" -eq 18 ]]; then
  OUT="runs/t2_${PAIR}_untrained_baseline"
  echo "=== untrained baseline -> $OUT (max-tokens=$MAX_TOKENS) ==="
  # Dump init weights with a cheap --steps 0 pass (short max-tokens so refresh is tiny),
  # then eval-only at the real 24k budget (same path as the 18 trained ckpts).
  srun python train_harness.py \
    --loss soft_knn_kl \
    --teacher "$TEACHER" --student "$STUDENT" \
    --prompts BytedTsinghua-SIA/DAPO-Math-17k --n-prompts 8 \
    --supervised-layer-fracs 0.25,0.5 \
    --steps 0 --batch-tokens 2048 --lr 1e-5 --seed 0 \
    --max-tokens 512 \
    --out "$OUT"
  srun python train_harness.py \
    --eval-only \
    --loss soft_knn_kl \
    --teacher "$TEACHER" --student "$STUDENT" \
    --prompts BytedTsinghua-SIA/DAPO-Math-17k --n-prompts 8 \
    --supervised-layer-fracs 0.25,0.5 \
    --seed 0 \
    --max-tokens "$MAX_TOKENS" \
    --eval aime24,aime25 --eval-avg-at 16 \
    --out "$OUT"
  rm -rf "$OUT/tmp_policy"
  exit 0
fi

ARM=${ARMS[$((i % 6))]}
SEED=${SEEDS[$((i / 6))]}
OUT="runs/t2_${PAIR}_${ARM}_s${SEED}"
CKPT="$OUT/ckpt_final.pt"
if [[ ! -f "$CKPT" ]]; then
  echo "MISSING ckpt: $CKPT" >&2
  exit 1
fi

echo "=== eval-only $OUT arm=$ARM seed=$SEED max-tokens=$MAX_TOKENS ==="
# Drop stale vLLM dumps before eval (each is ~6.5G; filled scratch last attempt).
rm -rf "$OUT/tmp_policy"
srun python train_harness.py \
  --eval-only \
  --loss "$ARM" \
  --teacher "$TEACHER" --student "$STUDENT" \
  --prompts BytedTsinghua-SIA/DAPO-Math-17k --n-prompts 8 \
  --supervised-layer-fracs 0.25,0.5 \
  --seed "$SEED" \
  --max-tokens "$MAX_TOKENS" \
  --eval aime24,aime25 --eval-avg-at 16 \
  --out "$OUT"
rm -rf "$OUT/tmp_policy"

echo "REEVAL_OK $OUT"
