#!/bin/bash
#SBATCH --job-name=oprd-official-models
#SBATCH --partition=cml-furongh
#SBATCH --account=cml-furongh
#SBATCH --qos=cml-high_long
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/cmlscratch/%u/oprd-experiments-main/cluster/logs/oprd_official_models_%j.out
#SBATCH --export=NONE

set -euo pipefail

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

readonly RUN_USER=${SLURM_JOB_USER:?SLURM_JOB_USER is required}
readonly SCRATCH_ROOT=/cmlscratch/$RUN_USER
readonly PROJECT_ROOT=$SCRATCH_ROOT/oprd-experiments-main
readonly OFFICIAL_ROOT=$SCRATCH_ROOT/oprd-official-stage2
readonly DESTINATION=$OFFICIAL_ROOT/models
readonly PYTHON=$PROJECT_ROOT/venv/bin/python
readonly ATOMIC_DIRECTORY=$PROJECT_ROOT/cluster/oprd_atomic_directory.py
readonly VALIDATOR=$PROJECT_ROOT/cluster/official_stage2/validate_task0.py
readonly CONTRACT=$PROJECT_ROOT/cluster/official_stage2/task0_contract.json
readonly HF_HOME=$OFFICIAL_ROOT/cache/huggingface
readonly PATH=$PROJECT_ROOT/venv/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/bin:/sbin

[[ $# -eq 0 ]] || fail "prepare_official_models.sbatch accepts no arguments"
[[ -x $PYTHON ]] || fail "project Python is unavailable: $PYTHON"
[[ ! -e $DESTINATION ]] || fail "official model destination already exists: $DESTINATION"
mkdir -p "$OFFICIAL_ROOT" "$HF_HOME"
export HF_HOME
STAGING=$(mktemp -d "$OFFICIAL_ROOT/.models-staging.XXXXXX")
readonly STAGING
cleanup() { test ! -e "$STAGING" || rm -rf -- "$STAGING"; }
trap cleanup EXIT

"$PYTHON" - "$STAGING" <<'PY'
import hashlib
import json
import os
import pathlib
import shutil
import sys
from huggingface_hub import snapshot_download

root = pathlib.Path(sys.argv[1])
models = [
    ("Qwen/Qwen3-1.7B-Base", "ea980cb0a6c2ae4b936e82123acc929f1cec04c1", "Qwen3-1.7B-Base"),
    ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c", "Qwen3-4B"),
]
manifest = {"schema_version": 1, "artifact_kind": "official_oprd_models", "models": {}}
for model_id, revision, directory_name in models:
    destination = root / directory_name
    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=destination,
    )
    shutil.rmtree(destination / ".cache", ignore_errors=True)
    files = []
    for path in sorted(destination.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        digest_object = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest_object.update(chunk)
        digest = digest_object.hexdigest()
        files.append({
            "path": path.relative_to(destination).as_posix(),
            "size": path.stat().st_size,
            "sha256": digest,
        })
    tree = hashlib.sha256(
        b"".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            for record in files
        )
    ).hexdigest()
    manifest["models"][directory_name] = {
        "model_id": model_id,
        "revision": revision,
        "tree_sha256": tree,
        "files": files,
    }
(root / "manifest.json").write_text(
    json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
PY
"$PYTHON" "$VALIDATOR" --contract "$CONTRACT" models --models-root "$STAGING"

"$PYTHON" "$ATOMIC_DIRECTORY" move \
  --trusted-root "$OFFICIAL_ROOT" \
  --source "$STAGING" \
  --destination "$DESTINATION"
trap - EXIT
printf 'pinned official models published: %s\n' "$DESTINATION"
