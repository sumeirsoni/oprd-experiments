#!/bin/bash
#SBATCH --job-name=oprd-t1-tau
#SBATCH --partition=gpu            # EDIT: your partition
#SBATCH --gres=gpu:a100:1          # 1x A100-80GB per run (1.5B student + 1.5B frozen teacher)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-23               # 8 tau values x 3 seeds
#SBATCH --output=logs/t1_tau_%A_%a.out

# Tier 1 — temperature-interpolated local relation loss (adversarial review §5.5).
# One PKT-style loss: softmax-over-similarities to top-k window neighbors,
# teacher->student matched by KL, tau swept high (≈RKD/LSP distance) to low
# (≈hard neighbor identity ≈ ARH). 500 steps each, same-arch 1.5B pair.
# STATUS: UNTESTED-ON-CLUSTER — written on a Mac, syntax-checked only. See README.

set -euo pipefail

TAUS=(0.02 0.05 0.1 0.2 0.5 1.0 2.0 5.0)
SEEDS=(0 1 2)
TAU=${TAUS[$((SLURM_ARRAY_TASK_ID % 8))]}
SEED=${SEEDS[$((SLURM_ARRAY_TASK_ID / 8))]}

# ---- set from the ACTUAL Tier-0 verdicts, 2026-07-11 (see cluster/README.md) ----
# M1: relational signal peaks at 25-50% depth on both pairs; final layer is the weakest
SUPERVISED_LAYER_FRACS=${SUPERVISED_LAYER_FRACS:-"0.25,0.5"}
# M1 NO-GO: identity is output-readable -> identity-only runs as control arm only
IDENTITY_ARM_ROLE=${IDENTITY_ARM_ROLE:-"control"}

TEACHER=hbx/JustRL-DeepSeek-1.5B
STUDENT=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

srun python train_harness.py \
  --loss pkt_tau --tau "$TAU" --topk 10 \
  --teacher "$TEACHER" --student "$STUDENT" \
  --prompts BytedTsinghua-SIA/DAPO-Math-17k --n-prompts 2000 \
  --supervised-layer-fracs "$SUPERVISED_LAYER_FRACS" \
  --steps 500 --batch-tokens 16384 --lr 1e-5 --seed "$SEED" \
  --track-stable-rank \
  --out "runs/t1_tau${TAU}_s${SEED}"
