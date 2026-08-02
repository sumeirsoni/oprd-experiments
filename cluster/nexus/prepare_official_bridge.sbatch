#!/bin/bash
#SBATCH --job-name=oprd-official-bridge
#SBATCH --partition=cml-furongh
#SBATCH --account=cml-furongh
#SBATCH --qos=cml-high_long
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/cmlscratch/%u/oprd-experiments-main/cluster/logs/oprd_official_bridge_%j.out
#SBATCH --export=NONE

set -euo pipefail

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

readonly RUN_USER=${SLURM_JOB_USER:?SLURM_JOB_USER is required}
readonly SCRATCH_ROOT=/cmlscratch/$RUN_USER
readonly PROJECT_ROOT=$SCRATCH_ROOT/oprd-experiments-main
readonly OFFICIAL_ROOT=$SCRATCH_ROOT/oprd-official-stage2
readonly PYTHON=$PROJECT_ROOT/venv/bin/python
readonly TOOL=$PROJECT_ROOT/cluster/official_stage2/convert_bridge_to_official.py
readonly CONTRACT=$PROJECT_ROOT/cluster/official_stage2/task0_contract.json
readonly SOURCE_ARTIFACT=$PROJECT_ROOT/cluster/artifacts/oprd_bridge/qwen3_4b_to_1p7b/production/rank8_all_layers
readonly DESTINATION=$OFFICIAL_ROOT/bridge/rank8_all_layers
readonly FAILED_DESTINATION=$OFFICIAL_ROOT/failed/bridge-${SLURM_JOB_ID}/rank8_all_layers.previous
readonly PATH=$PROJECT_ROOT/venv/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/bin:/sbin

[[ $# -eq 0 ]] || fail "prepare_official_bridge.sbatch accepts no arguments"
[[ -x $PYTHON ]] || fail "project Python is unavailable: $PYTHON"
[[ -d $SOURCE_ARTIFACT && ! -L $SOURCE_ARTIFACT ]] ||
  fail "production Stage-1 bridge is missing: $SOURCE_ARTIFACT"
[[ ! -e $FAILED_DESTINATION ]] || fail "bridge preservation destination already exists"
if [[ -e $DESTINATION ]]; then
  mkdir -p "$(dirname "$FAILED_DESTINATION")"
  mv "$DESTINATION" "$FAILED_DESTINATION"
fi
mkdir -p "$OFFICIAL_ROOT/bridge"
"$PYTHON" "$TOOL" \
  --contract "$CONTRACT" \
  --artifact "$SOURCE_ARTIFACT" \
  --destination "$DESTINATION"
printf 'official bridge adapter published: %s\n' "$DESTINATION"
