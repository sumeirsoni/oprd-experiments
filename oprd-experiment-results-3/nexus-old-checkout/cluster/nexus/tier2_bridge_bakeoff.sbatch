#!/bin/bash
#SBATCH --job-name=oprd-t2-bridge
#SBATCH --partition=cml-furongh
#SBATCH --account=cml-furongh
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:h200-sxm:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --array=0-35%8
#SBATCH --output=/cmlscratch/%u/oprd-experiments/cluster/logs/t2_bridge_%A_%a.out

# Tier 2 — bridge bake-off. Phi pair skipped by default. Shares 8-GPU QoS budget with Tier 1.
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

ARMS=(oprd_bridge linear_map_mse minilmv2_selfrel ln_dist soft_knn_kl cka_gram)
PAIRS=(qwen3_4b_to_1p7b phi4mini_to_qwen3_1p7b)
SEEDS=(0 1 2)
i=$SLURM_ARRAY_TASK_ID
ARM=${ARMS[$((i % 6))]}
PAIR=${PAIRS[$(((i / 6) % 2))]}
SEED=${SEEDS[$((i / 12))]}

SUPERVISED_LAYER_FRACS=${SUPERVISED_LAYER_FRACS:-"0.25,0.5"}
if [[ "${SKIP_ARMS:-}" == *"$ARM"* ]]; then echo "arm $ARM skipped by SKIP_ARMS"; exit 0; fi
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
