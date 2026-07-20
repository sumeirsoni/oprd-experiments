#!/bin/bash
#SBATCH --job-name=oprd-pc-resume
#SBATCH --partition=cml-furongh
#SBATCH --account=cml-furongh
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:h200-sxm:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/cmlscratch/%u/oprd-experiments/cluster/logs/pc_resume_%j.out

# Resume a timed-out PC-gate train from ckpt_last, finish remaining steps, then 24k eval.
# Example (minilm s0 timed out at ~1930; ckpt_last @ 1900):
#   OUT=runs/pc_justRL_to_r1_1p5b_minilmv2_selfrel_s0 ARM=minilmv2_selfrel SEED=0 \
#   START_STEP=1900 sbatch --export=ALL,OUT,ARM,SEED,START_STEP nexus/pc_gate_resume.sbatch
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

TEACHER=${TEACHER:-hbx/JustRL-DeepSeek-1.5B}
STUDENT=${STUDENT:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}
OUT=${OUT:?set OUT}
ARM=${ARM:?set ARM}
SEED=${SEED:?set SEED}
START_STEP=${START_STEP:?set START_STEP}
STEPS=${STEPS:-2000}
ROLLOUT_TOKENS=${ROLLOUT_TOKENS:-16000}
SUPERVISE_LAST_K=${SUPERVISE_LAST_K:-2000}
EVAL_MAX_TOKENS=${EVAL_MAX_TOKENS:-24000}
SUPERVISED_LAYER_FRACS=${SUPERVISED_LAYER_FRACS:-"0.25,0.5"}

test -f "$OUT/ckpt_last.pt" || { echo "MISSING $OUT/ckpt_last.pt" >&2; exit 1; }
echo "=== resume $OUT arm=$ARM seed=$SEED start=$START_STEP/$STEPS ==="
rm -rf "$OUT/tmp_policy"
srun python train_harness.py \
  --loss "$ARM" \
  --teacher "$TEACHER" --student "$STUDENT" \
  --prompts BytedTsinghua-SIA/DAPO-Math-17k --n-prompts 2000 \
  --supervised-layer-fracs "$SUPERVISED_LAYER_FRACS" \
  --steps "$STEPS" --batch-tokens 16384 --lr 1e-5 --seed "$SEED" \
  --resume-from "$OUT/ckpt_last.pt" \
  --start-step "$START_STEP" \
  --rollout-tokens "$ROLLOUT_TOKENS" \
  --supervise-last-k "$SUPERVISE_LAST_K" \
  --require-finished \
  --max-tokens "$EVAL_MAX_TOKENS" \
  --eval aime24,aime25 --eval-avg-at 16 \
  --track-stable-rank \
  --out "$OUT"
rm -rf "$OUT/tmp_policy" "$OUT/ckpt_last.pt"
echo "PC_RESUME_OK $OUT"
