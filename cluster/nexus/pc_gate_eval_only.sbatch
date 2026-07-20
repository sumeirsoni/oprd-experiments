#!/bin/bash
#SBATCH --job-name=oprd-pc-eval
#SBATCH --partition=cml-furongh
#SBATCH --account=cml-furongh
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:h200-sxm:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/cmlscratch/%u/oprd-experiments/cluster/logs/pc_eval_%j.out

# Eval-only rescue for a PC-gate run that trained but died in AIME (e.g. poller
# pruned tmp_policy after results.json was written pre-eval).
# Usage: OUT=runs/pc_... ARM=opd_topk SEED=1 sbatch --export=ALL,OUT,ARM,SEED nexus/pc_gate_eval_only.sbatch
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
OUT=${OUT:?set OUT=runs/pc_...}
ARM=${ARM:?set ARM}
SEED=${SEED:?set SEED}
MAX_TOKENS=${MAX_TOKENS:-24000}

test -f "$OUT/ckpt_final.pt" || { echo "MISSING $OUT/ckpt_final.pt" >&2; exit 1; }
echo "=== eval-only $OUT arm=$ARM seed=$SEED max-tokens=$MAX_TOKENS ==="
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
echo "PC_EVAL_OK $OUT"
