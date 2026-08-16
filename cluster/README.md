# OPRD Tier 1-2 cluster scripts

## Current protocol status

The immutable Qwen3-4B to Qwen3-1.7B Stage-1 bridge and its external-artifact backward path have
run successfully on Nexus. The completed single-H200 task-0 namespace `t2_paper_a4d7f92` is retained
as an infrastructure pilot, but it is **not** a paper reproduction. Its Stage-2 harness reused cached
rollouts, sampled one response per prompt, filtered cap hitters, traversed a prompt subset, and used
project-specific optimizer and evaluation settings.

A separate official-code-faithful task-0 path now lives under `official_stage2/` and the matching
`nexus/*official_stage2*` jobs. It pins official OPRD commit
`a63236c990b1bfa9551e9d020719490d3ecd93e8` and requires:

- a historical one-node, eight-GPU scientific contract, with the authorized production execution
  limited to four H200s by `official_stage2/task0_execution_4h200.json`;
- the ordered 17,917-row official DAPO parquet, batch 8, drop-last, one epoch, and 2,239 updates;
- two fresh responses per prompt at temperature 1.0, top-p 1.0, and a 16,384-token cap;
- raw chat with Qwen thinking disabled, cap-hitter retention, and no rollout reuse;
- all-layer rank-8 frozen bridge loss over the final 2,000 valid response tokens;
- AdamW at `1e-5`, weight decay `0.01`, clip threshold `1000`, constant LR, and zero warmup.

The original `train_harness.py` remains a controls/debugging runner. Legacy array task 0 now fails
closed and directs operators to `nexus/task0_official_stage2.sbatch`. Tasks 1-26 remain blocked.
The eight-H200 smoke and four-H200 canary passed. One fixed four-H200 full task-0 production run is
now authorized through the tracked execution manifest. Evaluation and tasks 1-26 remain unauthorized.

The historical sections below describe the Stage-1 recovery and project-control harness. Where they
call the old Tier-2 wrapper a paper matrix, the current protocol status above takes precedence.

## Official Stage-2 readiness workflow

The new workflow uses a distinct scratch root, environment, source checkout, models, bridge adapter,
and run namespaces under `/cmlscratch/$USER/oprd-official-stage2`. It never mutates the existing
bridge/control environment.

```bash
# Preparation only. Each destination is fresh and no-overwrite.
sbatch cluster/nexus/prepare_official_stage2.sbatch
sbatch cluster/nexus/prepare_official_models.sbatch
sbatch cluster/nexus/prepare_official_bridge.sbatch

# After all three preparation jobs validate and publish their manifests:
sbatch cluster/nexus/setup_official_stage2.sbatch

# Eight-GPU distributed smoke, then a documented four-GPU resource canary.
sbatch cluster/nexus/official_stage2_smoke.sbatch official-smoke-<revision> 16
sbatch --gres=gpu:h200-sxm:4 --cpus-per-task=16 --mem=300G \
  cluster/nexus/official_stage2_smoke.sbatch official-canary-<revision>-4gpu 80 4
```

The smoke and canary run the actual official verl/Ray/FSDP/vLLM path against ordered DAPO prefixes.
The smoke uses the pinned one-node, eight-GPU topology. The canary intentionally uses four H200s as
a documented topology deviation while preserving batch 8, two responses per prompt, optimizer and
loss settings, precision, and update count; its capture wrapper changes only
`trainer.n_gpus_per_node`. The pinned official launcher requested 64 CPUs, while Nexus
`cml-high_long` permits at most 32. Jobs do not request Slurm `--exclusive`. These runs are not paper
results. The wrapper records the full runtime stream and permits the known Ray shutdown warning
`DataLoader worker ... is killed by signal: Killed` only when it is the sole traceback, the final
global-step metric is present, and the complete model, optimizer, and extra-state shard set exists
for the selected world size. Any other traceback fails the job. Review logs, memory, scratch growth,
counter behavior, and projected full-run cost. That review authorized the fixed four-H200 task-0
execution while retaining eight GPUs as the historical contract topology.

The dedicated production command is:

```bash
sbatch cluster/nexus/task0_official_stage2.sbatch \
  official-task0-4h200-prod-r1 <full-clean-project-git-commit>
```

This is a fixed non-array job requesting four H200s, 16 CPUs, and 300 GiB RAM; do not override its
resources or topology at submission. It cannot submit tasks 1-26. It authenticates the unchanged
historical scientific contract plus the four-H200 execution manifest, then captures and validates
the actual resolved Hydra configuration before training. Batch 8, two responses, FP32, optimizer,
and loss remain unchanged. It saves every 200 updates but retains only the latest actor checkpoint,
preserving resume capability within the 200 GiB scratch allocation. It writes `COMPLETE` only after finding `global_step_2239`, proving exact
four-rank model/optimizer/extra-state shards, merging them into an authenticated Hugging Face
checkpoint, rejecting later steps and unexplained failure markers, and writing an
`ACCEPTED_OFFICIAL_TASK0` report.

Evaluation is independently authorized and submitted only against that accepted checkpoint:

```bash
sbatch cluster/nexus/evaluate_official_stage2.sbatch \
  /cmlscratch/$USER/oprd-official-stage2/runs/official-task0-4h200-prod-r1 \
  official-eval-<revision>
```

The six evaluation jobs cover AIME24, AIME25, and AIMO under two non-pooled tracks:
`published_offline_exact` preserves the pinned evaluator's prompt and unseeded-request behavior;
`corrected_seeded_single_suffix` uses exactly one instruction suffix and deterministic request seeds.
Both preserve raw response text and authoritative vLLM finish metadata in checksummed JSONL, then
run the pinned official grader as a standalone score layer. Rescoring validates the generation
manifest and does not need model loading or dataset downloads.

## Cluster smokes

The original plumbing smoke below includes a rollout refresh at step 5 and a 2-sample eval,
so it checks vLLM-after-optimizer-state and per-request seed diversity:

```bash
python train_harness.py --loss pkt_tau --tau 0.1 --steps 8 --n-prompts 8 \
  --teacher hbx/JustRL-DeepSeek-1.5B --student deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --prompts BytedTsinghua-SIA/DAPO-Math-17k --supervised-layer-fracs 0.25,0.5 \
  --batch-tokens 2048 --max-tokens 512 --rollout-refresh 5 --seed 0 \
  --eval aime24 --eval-avg-at 2 --out runs/smoke
```

First run downloads ~7GB of models + datasets.

That smoke does **not** validate the external bridge path. The bridge workflow is now:

```text
init/generate corpus -> teacher PCA/targets -> groups 0..6 in order
  -> staged reload + atomic publish -> full verify -> two-step external-artifact backward smoke
```

`nexus/submit_oprd_bridge.sh` defaults to the builder's eight-prompt smoke contract, submits
seven individually chained group jobs, and stops after `oprd_bridge_stage2_smoke.sbatch`. It does
not submit the 36-task Tier-2 matrix or the three production OPRD seeds. The submitter supports a
preflight-only mode, atomic artifact claims, JSON submission records, terminal status
reconciliation, and suffix resume after a failed phase. A paper-scale build additionally requires
`BRIDGE_SMOKE=0 ALLOW_PAPER_SCALE=1`, which should be set only after the realistic storage smoke,
quota check, and cost approval.

Pass criteria: exact 28-to-36 map, all seven groups complete in numeric order, the staged artifact
fully reloads before `COMPLETE`, every recorded Slurm job completes with exit code zero,
`verify.json` reports the manifest digest, and the same digest appears in `results.json`,
`train_state_last.pt`, and the checksummed `SMOKE_COMPLETE.json` marker. Stage 2 writes into a
sibling attempt directory and moves it to `stage2-backward-smoke` only after finished-rollout,
finite-loss, finite-weight, exact invocation identity, checksum validation, and a staging-directory
`fsync`. The canonical result is validated again after promotion. A killed attempt remains available
for inspection; a suffix resume runs a fresh deterministic two-step attempt instead of pretending a
weight-only checkpoint is continuation-equivalent. A pre-existing invalid canonical smoke directory
is atomically quarantined before the fresh retry.

Artifact publication, invalid-smoke quarantine, and final smoke promotion use the shared
`oprd_atomic_directory.py` helper. It tries the platform-native no-replace rename first. If the
filesystem reports that operation as unsupported, including NFSv3 `EINVAL`, it reserves the exact
destination with an empty mode-0700 directory and then performs a plain sibling rename. The empty
reservation can be briefly visible before the rename. The trusted parent chain must be owned by the
effective user with group and world write bits clear, so only the owner can write publication names.
Validation and cleanup are inode-based: cleanup removes only the exact empty reservation it created,
and any changed inode or uncertain postcondition is reported as ambiguous and fails closed without
unsafe cleanup.

## Environment

Submit `nexus/setup_env.sbatch` as a separate prerequisite. Furong login and compute nodes expose
different node-local Python 3.12 modules, so setup bootstraps `uv==0.9.8` once and installs a
scratch-managed CPython 3.12.7 shared by setup, preflight, status, resume, and every DAG phase. It
loads CUDA 12.8.1, installs a bundled FFmpeg executable from `imageio-ffmpeg==0.6.0`, installs
vLLM before the remaining requirement floors, runs `pip check`, and atomically writes the schema-2
`venv/.oprd-env-ready.json` marker. The marker authenticates Python, packages including
`flashinfer-python`, FFmpeg, the exact CUDA root and `nvcc`, the exact runtime library search path,
and hashes of `nvcc` and the resolved `libcudart.so.12`. A pre-existing module-backed venv is
preserved as a sibling backup before migration. Do not submit the bridge DAG until setup reaches
`COMPLETED`, prints `SETUP_OK`, and the marker's `requirements_sha256` matches
`cluster/requirements.txt`.

Two independent native checks run before model work. The existing GCC/ld probe compiles a
CUDA-independent shared object and catches missing compiler or linker executables. The CUDA probe
uses the exact recorded `nvcc`, links a no-RPATH shared object against `libcudart.so.12`, inspects
its ELF dependencies, and loads both the SONAME and generated object in a clean configured-Python
subprocess. Setup additionally loads FlashInfer's real sampling module under the configured scratch
workspace. Corpus and Stage-2 repeat that cached-module load before model weights.

These checks cover failures observed during bridge smoke validation. Run
`20260728T065705Z_145055` failed because the explicit Slurm export omitted the deterministic
runtime `PATH`, leaving GCC unable to find `ld`. After that fix, run
`20260728T165135Z_2088398` loaded the model and completed `torch.compile`, then failed because
FlashInfer used `/nfshomes/.../.cache/flashinfer` and EngineCore could not resolve
`libcudart.so.12`. Publish job 7161597 later reached the NFSv3 artifact parent but native
`renameat2(RENAME_NOREPLACE)` returned `EINVAL`. Preserve that failed run, including its record,
build, claim, and logs. Because the atomic publication helper changes the authenticated source
bundle, the run is not resumable: reconcile it, release only its exact inactive claim after explicit
approval, and submit a fresh DAG with a new run ID. The existing schema-2 environment marker remains
valid, so no setup rerun is expected unless requirements or authenticated environment inputs change.

The submitter uses an explicit Slurm export allowlist rather than `--export=ALL`. Every new record
freezes and every phase receives the same `PATH`, `CUDA_HOME`, `FLASHINFER_NVCC`,
`LD_LIBRARY_PATH`, and cache locations. Module loading remains site-compatible, but each script
validates the module's CUDA root and then overwrites runtime values with the authenticated record.
Resume restores recorded values and ignores conflicting caller environment values.

Four independent caches must remain under scratch and are exported before Python starts:

- `BRIDGE_HF_HOME`, exported as `HF_HOME`, for models and datasets;
- `VLLM_CACHE_ROOT` for vLLM compilation and runtime caches;
- `TRITON_CACHE_DIR` for Triton kernel artifacts;
- `FLASHINFER_WORKSPACE_BASE` for `.cache/flashinfer/<version>/<arch>` JIT state.

Models and datasets download at job runtime unless the cache is pre-populated; the required
repositories are public and ungated. Preflight checks canonical paths, all writable cache roots,
schema-2 environment identity, installed package versions, FFmpeg identity, both native probes,
free blocks, free inodes, and quota headroom. It also requires the existing artifact publication
parent beneath scratch and performs a real atomic-directory move and collision probe there, reporting
`native_noreplace` or `reserved_plain_rename` as the selected method. Any helper, cleanup, ownership,
or ambiguity failure stops submission before record creation, claim acquisition, or `sbatch`.
Preflight does not warm FlashInfer on the login node because login and compute nodes have different
glibc versions while the FlashInfer cache key does not encode that ABI difference. If the site
exposes no parseable quota command, preflight fails closed. `ALLOW_QUOTA_UNAVAILABLE=1` is an explicit
override for use only after a manual block and inode quota check.

**Disk:** each trained task writes ~7-10GB under `runs/<task>/` (two bf16 checkpoints + a
transient fp32 `tmp_policy/` dump per generation round, deleted after each round). The 27-task
paper matrix therefore needs substantially more scratch than the generic 50-GiB bridge preflight
gate. Confirm capacity for the production bridge, caches, transient model copies, failed attempts,
and roughly 24 trained outputs before submission. Do not reclaim space by deleting preserved runs
or caches without separate approval.

**Walltime/failure behavior:** the Nexus paper wrapper requests 72 hours because related 36-hour
runs timed out. Treat 72 hours as an initial ceiling, not a measured estimate: run task 0 alone and
review its elapsed time and scratch high-water mark before the matrix. Training results
(`results.json` with losses/diagnostics) are written BEFORE eval starts, so a walltime kill during
eval loses only the eval - rerun it from the checkpoint with `--eval-only --eval aime24,aime25
--out runs/<same-dir>`. Weight-level
mid-training rescue is supported with `--resume-from` / `--start-step`. Every 100 steps and at
normal completion, `train_state_last.pt` atomically stores synchronized student/map
weights, loss history, and the immutable bridge reference when applicable. `oprd_bridge` training
and resume both require an explicit `--bridge-artifact`, manifest SHA-256, and pinned model/tokenizer
revisions. Resume rejects a changed manifest, Stage-2 dataset identity, rollout protocol, or core
optimization settings, but permits relocating byte-identical artifact contents. No run-local
`bridge.pt` is written or searched. Optimizer moments
are not restored, so this remains weight-level recovery rather than continuation-equivalent replay.

## Known limitation: cross-tokenizer pair

The phi4mini→Qwen pair is **not implemented** (needs the anchor-fingerprint OT bridge, review
§5.8; token-aligned losses are undefined across tokenizers). It is not part of the Nexus paper
matrix. That matrix is deliberately limited to Qwen3-4B→Qwen3-1.7B so every array task performs a
meaningful run under one pinned tokenizer identity.

## Files

- `tier1_tau_sweep.sbatch` - 24 tasks (8 tau values x 3 seeds), same-architecture 1.5B pair.
- `official_stage2/` - canonical official task-0 contract, source/data authentication, bridge adapter,
  task acceptance checks, immutable evaluation artifacts, eight-worker generation, and rescoring.
- `nexus/task0_official_stage2.sbatch` - dedicated non-array official task-0 launcher.
- `nexus/official_stage2_smoke.sbatch` - historical eight-GPU smoke and documented four-GPU canary on ordered DAPO prefixes.
- `nexus/evaluate_official_stage2.sbatch` - separately authorized dual-track AIME24/AIME25/AIMO evaluation.
- `nexus/tier2_bridge_bakeoff.sbatch` - legacy project-control array. Task 0 is disabled because this
  cached-rollout harness is not the official OPRD Stage-2 protocol. Other tasks remain unsubmitted.
- `oprd_bridge.py` - authoritative layer mapping, `BridgeBank`, contract digest, and immutable
  safetensors artifact publication/loading.
- `build_oprd_bridge.py` - restartable seed-independent Stage-1 builder with explicit phases.
- `nexus/submit_oprd_bridge.sh` - ordered smoke/build DAG that stops after verification and the
  two-step external-artifact Stage-2 smoke.
- `nexus/oprd_bridge_ops.py` - preflight, artifact claims, atomic job records, terminal status,
  authenticated success checks, and safe suffix-resume decisions.
- `train_harness.py` - project controls and debugging harness. It validates the external bridge and
  OPRD layer map, but its rollout lifecycle is not accepted as the official task-0 implementation.

### Bridge procedure

- `oprd_bridge` follows the paper-scale two-stage recipe with a separate Stage-1 build: exactly
  2,000 attempted DAPO-Math-17K prompts for production, a fixed 1,800/200 split, rank-8 teacher
  PCA capped at 16,384 shared rows, seven serial groups of four student layers, 20 AdamW epochs,
  and one earliest strictly best global epoch shared by all 28 projectors. Empty and cap-terminated
  outputs remain in the corpus. The publisher writes safetensors and sidecars into a sibling
  directory, validates checksums and identities, reloads the full staged bridge, compares tensor
  values, then writes `COMPLETE` and atomically renames. Stage 2 never fits or searches for a bridge;
  every seed receives the same explicit artifact and records its manifest SHA-256.
- `linear_map_mse` deliberately remains a separate jointly trained full-rank contrast. It uses a
  seeded bounded least-squares calibration sample (`--linear-map-init-rows`, default 512), then
  checkpoints the map with the student.
- The other representation arms and the `opd_topk` positive control require no bridge artifact,
  but the Nexus Tier-2 wrapper still pins them to the same model, tokenizer, dataset revision, split,
  and prompt key. The untrained baseline loads only the pinned student and tokenizer, performs no
  training rollout or teacher prefill, and uses the same three stochastic evaluation seeds.
- `requirements.txt` contains the Linux/CUDA dependency floors.

## How the ACTUAL Tier-0/3 verdicts (../tier0-results.md, ../tier3-results.md) set the knobs

| Measured verdict | Knob setting (now the default) |
|---|---|
| M1 **NO-GO**, both pairs: row-space carries ≥99% of mKNN; relational signal richest at 25–50% depth, dead at final layer | Retained as a later layer-diagnostic hypothesis. The primary Tier-2 objective bake-off now fixes every arm to literal all-student-layer supervision, matching OPRD's reported Bridge setup and removing layer choice as a confound. |
| M1 no-go (identity output-readable) | `soft_knn_kl` (identity) runs as **control only** - keep it in the fixed paper matrix for the control row and never headline it. If allocation is insufficient, revise and record the matrix rather than silently dropping an arm through the submission environment. |
| M2: identity ranked first, local-distance ≈ global second (both pairs) — but M1 overrides identity-only as a treatment | LN-Dist stays the lead relational arm; no knob change |
| M3 **AMBIGUOUS**, rollout-level trend weakly negative for geometry | a functional-bridge (`_fbridge`) arm variant is recommended by the Tier-0 verdict but is **deferred — not implemented in this run**; treat current geometric-arm wins as provisional until it exists |
| M4: OPRD rel. variance 0.389 vs OPD 0.857; the power table in `tier0-results.md` requires substantially more independent training seeds for small effects | Three seeds make this an exploratory objective comparison. Report uncertainty and do not claim that the design is powered for a two-point difference. |
| Tier 3 pilot (1.5B, ../tier3-results.md): row-only ≥ full functionally, null-only hurts; **mid-layer supervision won KL while an eight-depth band won top-1** | Motivates a later layer sweep on the strongest bridge objectives; it no longer changes the layer coverage of the primary Tier-2 objective comparison. |

## Allocation estimate

The historical ~10 GPU-hour/task estimate came from two-layer, short-rollout Tier 2 and is not
valid for the corrected workflow. Before approving paper-scale Stage 1 or the three OPRD seeds,
measure corpus throughput, teacher passes, one realistic four-layer cache, restart behavior,
scratch high-water mark, Stage-2 throughput, and evaluation time. Use those measurements for the
cost proposal; do not extrapolate from the old run.

## Submission

```bash
# From the repository root on Nexus.
mkdir -p cluster/logs cluster/runs
sbatch cluster/nexus/setup_env.sbatch
# Wait for terminal success and venv/.oprd-env-ready.json.
# Preflight probes publication semantics in the existing cluster directory on the
# same scratch filesystem. Fresh submission creates the canonical artifact parent.
BRIDGE_SMOKE=1 cluster/nexus/submit_oprd_bridge.sh --preflight-only
BRIDGE_SMOKE=1 cluster/nexus/submit_oprd_bridge.sh

# Use the exact record printed by the submit command.
cluster/nexus/submit_oprd_bridge.sh --status cluster/job_ids/<record>.json

# After a failed terminal phase, status must show no active jobs before resume.
cluster/nexus/submit_oprd_bridge.sh --resume cluster/job_ids/<record>.json
```

`--preflight-only` never acquires a claim or submits work. Fresh submission atomically claims the
canonical artifact path and records each submission intent, job ID, dependency, script, and source
bundle digest. Runtime jobs recheck the claim, canonical scratch containment, environment marker,
and exact source bundle before mutation. `--status` never submits work; it reconciles `sacct` and
authenticates final outputs. Missing or unknown accounting rows remain unresolved and cannot be
resumed or released. `--resume` reuses the original build, artifact, cache, and claim and submits
only the failed or missing suffix. An abandoned claim can be released only explicitly and only
when the record has no active, unresolved, or pending jobs. The submitter writes a durable phase
intent before each `sbatch` call and tags the Slurm job with `oprd:<run-id>:<phase>`. If the shell is
killed after Slurm accepts a job but before its ID is recorded, locate that exact tagged job and run
`submit_oprd_bridge.sh --adopt-job RECORD PHASE JOB_ID` before status or resume.

Resume requires the original authenticated source-bundle digest. If a failure requires changing a
runtime script or operations helper, preserve the failed build and record, explicitly release its
inactive claim, and submit a fresh smoke with a new run ID. Never bypass the source-digest check to
reuse an incompatible record.

Production Tier 2 remains a staged decision. Preparing and validating the wrapper does not
authorize either the production bridge build or a Tier-2 `sbatch` call.

Before the production bridge build, increase scratch capacity and recheck free bytes, free inodes,
and quota headroom. Do not lower the storage gate or delete preserved evidence to make the check
pass. Once capacity and the paper-scale build are explicitly approved, build from the exact deployed
revision:

```bash
repo=/cmlscratch/$USER/oprd-experiments-main
cd "$repo"
env \
  SCRATCH_ROOT=/cmlscratch/$USER \
  ROOT="$repo" \
  BRIDGE_SMOKE=0 \
  ALLOW_PAPER_SCALE=1 \
  ALLOW_QUOTA_UNAVAILABLE=1 \
  "$repo/cluster/nexus/submit_oprd_bridge.sh"
```

Reconcile the printed record to terminal success. Then set `RECORD` to that exact record and read
the build and artifact paths from it rather than guessing them:

```bash
RECORD="$repo/cluster/job_ids/<printed-record>.json"
mapfile -t production_paths < <(
  "$repo/venv/bin/python" - "$RECORD" <<'PY'
import json
import pathlib
import sys

record = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(record["build_dir"])
print(record["artifact_dir"])
PY
)
BUILD_DIR=${production_paths[0]}
ARTIFACT_DIR=${production_paths[1]}
"$repo/venv/bin/python" "$repo/cluster/nexus/oprd_bridge_ops.py" validate-published \
  --build-dir "$BUILD_DIR" \
  --artifact "$ARTIFACT_DIR"
```

The Nexus Tier-2 wrapper accepts exactly those two validated paths plus a fresh result namespace.
Choose the final paper namespace before profiling. Task 0 is `oprd_bridge`, seed 0, and its successful
output becomes the first member of the final matrix rather than a disposable duplicate:

```bash
RESULT_NS="t2_paper_<revision>"
sbatch --array=0 \
  "$repo/cluster/nexus/tier2_bridge_bakeoff.sbatch" \
  "$BUILD_DIR" "$ARTIFACT_DIR" "$RESULT_NS"
```

Review its terminal state, elapsed time, memory, scratch high-water mark, checkpoint, finite losses,
bridge digest, and completed AIME results. If it passes, keep the exact source revision, artifact,
protocol, and namespace fixed, then submit only tasks 1 through 26 after a separate matrix decision:

```bash
sbatch --array=1-26%4 \
  "$repo/cluster/nexus/tier2_bridge_bakeoff.sbatch" \
  "$BUILD_DIR" "$ARTIFACT_DIR" "$RESULT_NS"
```

If task 0 fails or requires any code or protocol change, abandon that namespace and restart the
corrected matrix under a new namespace. Do not mix a corrected remainder with the old profile output.

The fixed matrix contains 27 tasks: nine labeled rows across seeds 0, 1, and 2. It records the exact
source revision and Slurm identity in each run configuration, rejects an existing output directory,
and takes no scientific settings from the submission shell.
