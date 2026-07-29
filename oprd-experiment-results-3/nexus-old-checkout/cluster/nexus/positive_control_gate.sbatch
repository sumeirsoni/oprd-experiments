#!/bin/bash
#SBATCH --job-name=oprd-pc-gate
#SBATCH --partition=cml-furongh
#SBATCH --account=cml-furongh
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:h200-sxm:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --array=0-6%4
#SBATCH --output=/cmlscratch/%u/oprd-experiments/cluster/logs/pc_gate_%A_%a.out

# Positive-control thin gate (same-arch 1.5B), fixed rollout protocol.
# Tasks 0-2: opd_topk seeds 0/1/2
# Tasks 3-5: minilmv2_selfrel mid-layer seeds 0/1/2
# Task 6:    untrained student baseline @ 24k eval
#
# Protocol: long rollouts + last-k mask + drop cap-hitters (faithful last-k),
# eval at 24k. If opd_topk cannot beat baseline, stop — protocol cannot distill.
set -euo pipefail
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

TEACHER=hbx/JustRL-DeepSeek-1.5B
STUDENT=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
PAIR=justRL_to_r1_1p5b
SUPERVISED_LAYER_FRACS=${SUPERVISED_LAYER_FRACS:-"0.25,0.5"}
ROLLOUT_TOKENS=${ROLLOUT_TOKENS:-16000}
SUPERVISE_LAST_K=${SUPERVISE_LAST_K:-2000}
EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-24000}
STEPS=${STEPS:-2000}
i=$SLURM_ARRAY_TASK_ID

if [[ "$i" -eq 6 ]]; then
  OUT="runs/pc_${PAIR}_untrained_baseline"
  echo "=== untrained baseline -> $OUT (max-tokens=$EVAL_MAX_TOKENS) ==="
  srun python train_harness.py \
    --loss opd_topk \
    --teacher "$TEACHER" --student "$STUDENT" \
    --prompts BytedTsinghua-SIA/DAPO-Math-17k --n-prompts 8 \
    --supervised-layer-fracs "$SUPERVISED_LAYER_FRACS" \
    --steps 0 --batch-tokens 2048 --lr 1e-5 --seed 0 \
    --max-tokens 512 \
    --out "$OUT"
  srun python train_harness.py \
    --eval-only \
    --loss opd_topk \
    --teacher "$TEACHER" --student "$STUDENT" \
    --prompts BytedTsinghua-SIA/DAPO-Math-17k --n-prompts 8 \
    --supervised-layer-fracs "$SUPERVISED_LAYER_FRACS" \
    --seed 0 \
    --max-tokens "$EVAL_MAX_TOKENS" \
    --eval aime24,aime25 --eval-avg-at 16 \
    --out "$OUT"
  rm -rf "$OUT/tmp_policy"
  exit 0
fi

ARMS=(opd_topk opd_topk opd_topk minilmv2_selfrel minilmv2_selfrel minilmv2_selfrel)
SEEDS=(0 1 2 0 1 2)
ARM=${ARMS[$i]}
SEED=${SEEDS[$i]}
OUT="runs/pc_${PAIR}_${ARM}_s${SEED}"

echo "=== train $OUT arm=$ARM seed=$SEED rollout=$ROLLOUT_TOKENS last_k=$SUPERVISE_LAST_K require-finished ==="
rm -rf "$OUT/tmp_policy"
srun python train_harness.py \
  --loss "$ARM" \
  --teacher "$TEACHER" --student "$STUDENT" \
  --prompts BytedTsinghua-SIA/DAPO-Math-17k --n-prompts 2000 \
  --supervised-layer-fracs "$SUPERVISED_LAYER_FRACS" \
  --steps "$STEPS" --batch-tokens 16384 --lr 1e-5 --seed "$SEED" \
  --rollout-tokens "$ROLLOUT_TOKENS" \
  --supervise-last-k "$SUPERVISE_LAST_K" \
  --require-finished \
  --max-tokens "$EVAL_MAX_TOKENS" \
  --eval aime24,aime25 --eval-avg-at 16 \
  --track-stable-rank \
  --out "$OUT"
rm -rf "$OUT/tmp_policy" "$OUT/ckpt_last.pt"

echo "PC_GATE_OK $OUT"
