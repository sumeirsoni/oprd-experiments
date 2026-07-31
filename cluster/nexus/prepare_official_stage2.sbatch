#!/bin/bash
#SBATCH --job-name=oprd-official-inputs
#SBATCH --partition=cml-furongh
#SBATCH --account=cml-furongh
#SBATCH --qos=cml-high_long
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/cmlscratch/%u/oprd-experiments-main/cluster/logs/oprd_official_inputs_%j.out
#SBATCH --export=NONE

set -euo pipefail

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

readonly RUN_USER=${SLURM_JOB_USER:?SLURM_JOB_USER is required}
readonly SCRATCH_ROOT=/cmlscratch/$RUN_USER
readonly PROJECT_ROOT=$SCRATCH_ROOT/oprd-experiments-main
readonly OFFICIAL_ROOT=$SCRATCH_ROOT/oprd-official-stage2
readonly DESTINATION=$OFFICIAL_ROOT/source
readonly TOOL=$PROJECT_ROOT/cluster/official_stage2/prepare_inputs.py
readonly CONTRACT=$PROJECT_ROOT/cluster/official_stage2/task0_contract.json
readonly ATOMIC_DIRECTORY=$PROJECT_ROOT/cluster/oprd_atomic_directory.py
readonly PYTHON=$PROJECT_ROOT/venv/bin/python
readonly OFFICIAL_COMMIT=a63236c990b1bfa9551e9d020719490d3ecd93e8
readonly OFFICIAL_REPOSITORY=https://github.com/ShenzhiYang2000/OPRD.git
readonly PATH=/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/bin:/sbin

[[ $# -eq 0 ]] || fail "prepare_official_stage2.sbatch accepts no arguments"
[[ -x $PYTHON ]] || fail "project Python is unavailable: $PYTHON"
[[ -r $TOOL && -r $CONTRACT && -r $ATOMIC_DIRECTORY ]] ||
  fail "official input tooling is incomplete"
[[ ! -e $DESTINATION ]] || fail "official source destination already exists: $DESTINATION"
mkdir -p "$OFFICIAL_ROOT"
STAGING=$(mktemp -d "$OFFICIAL_ROOT/.source-staging.XXXXXX")
readonly STAGING
cleanup() { test ! -e "$STAGING" || rm -rf -- "$STAGING"; }
trap cleanup EXIT

git clone --no-checkout "$OFFICIAL_REPOSITORY" "$STAGING/OPRD"
git -C "$STAGING/OPRD" checkout --detach "$OFFICIAL_COMMIT"
test "$(git -C "$STAGING/OPRD" rev-parse HEAD)" = "$OFFICIAL_COMMIT" ||
  fail "official checkout did not resolve the pinned commit"
"$PYTHON" "$TOOL" --contract "$CONTRACT" source \
  --source "$STAGING/OPRD" \
  --manifest "$STAGING/manifest.json"
"$PYTHON" "$TOOL" --contract "$CONTRACT" dataset \
  --parquet "$STAGING/OPRD/datasets/dapo-math-17k.parquet" \
  --manifest "$STAGING/dapo-manifest.json"
"$PYTHON" "$ATOMIC_DIRECTORY" move \
  --trusted-root "$OFFICIAL_ROOT" \
  --source "$STAGING" \
  --destination "$DESTINATION"
trap - EXIT
printf 'official source and DAPO inputs published: %s\n' "$DESTINATION"
