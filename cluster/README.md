# OPRD Tier 1–2 cluster scripts

**STATUS:** the long-rollout CUDA/vLLM/Slurm path passed the 1.5B positive-control gate.
The recovered offline all-layer Stage-1 builder now publishes one immutable external artifact.
The first recovered bridge smoke failed before corpus generation because its explicit Slurm export
omitted the system toolchain path. A corrected run reached publish job 7161597, where the NFSv3
artifact filesystem returned `EINVAL` for native `renameat2(RENAME_NOREPLACE)`; no artifact was
published. The failed run is preserved, and the authenticated helper change requires a fresh DAG
before any paper-scale build or Tier-2 seed.

Verified:
- Loss arms end-to-end on Apple MPS (Qwen2.5-0.5B): generation → teacher prefill →
  training steps → diagnostics, finite losses. Also: rollout refresh, AIME eval
  (with per-sample seeds → genuinely distinct Avg@N samples), `--eval-only`, `--track-stable-rank`.
- CUDA/vLLM/Slurm long-rollout protocol at 1.5B: 16k rollouts, final-2k supervision,
  naturally finished traces, checkpoint/eval rescues, and 24k AIME evaluation.
- An independent adversarial code review confirmed KL directions, gradient flow, masking, and
  token alignment empirically, and its findings (vLLM API drift, in-process engine OOM,
  answer-normalization bug, degenerate Avg@N seeding) are fixed in the current version.

**Still untested:**
- real-model external-artifact Stage 1 and Stage 2 on Qwen3-4B to Qwen3-1.7B;
- realistic four-layer cache size, scratch high-water mark, and checkpoint restart behavior;
- all-layer CKA and full-rank `linear_map_mse` peak GPU memory.

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

**Disk:** each task writes ~7-10GB under `runs/<task>/` (two bf16 checkpoints + a transient
fp32 `tmp_policy/` dump per generation round, deleted after each round). Put the clone (or at
least `runs/`) on scratch, not home.

**Walltime/failure behavior:** training results (`results.json` with losses/diagnostics) are
written BEFORE eval starts, so a walltime kill during eval loses only the eval — rerun it from
the checkpoint with `--eval-only --eval aime24,aime25 --out runs/<same-dir>`. Weight-level
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
§5.8; token-aligned losses are undefined across tokenizers — the harness exits with a clear
error). The tier2 sbatch **skips it by default** via `SKIP_PAIRS`; its 18 array tasks exit 0
in seconds. Pairs interleave every 6 task ids, so do NOT try to exclude them by trimming the
array range. To enable later (after the bridge exists): `SKIP_PAIRS= sbatch tier2_bridge_bakeoff.sbatch`
(set it EMPTY — unsetting it restores the default skip).

## Files

- `tier1_tau_sweep.sbatch` - 24 tasks (8 tau values x 3 seeds), same-architecture 1.5B pair.
- `tier2_bridge_bakeoff.sbatch` - corrected all-layer 36-task matrix. Every runnable arm requires
  the same pinned model, tokenizer, and dataset identities; only `oprd_bridge` additionally requires
  the external artifact and manifest digest. The cross-tokenizer Phi pair remains skipped.
- `oprd_bridge.py` - authoritative layer mapping, `BridgeBank`, contract digest, and immutable
  safetensors artifact publication/loading.
- `build_oprd_bridge.py` - restartable seed-independent Stage-1 builder with explicit phases.
- `nexus/submit_oprd_bridge.sh` - ordered smoke/build DAG that stops after verification and the
  two-step external-artifact Stage-2 smoke.
- `nexus/oprd_bridge_ops.py` - preflight, artifact claims, atomic job records, terminal status,
  authenticated success checks, and safe suffix-resume decisions.
- `train_harness.py` - Stage-2 harness. `--supervised-layers all` maps unequal depths with OPRD
  Eq. 11 and `oprd_bridge` validates its artifact before model generation.

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
- The other five arms are projector-free by construction and require no bridge artifact, but the
  Tier-2 wrappers still pin them to the same model, tokenizer, dataset revision, split, and prompt key.
- `requirements.txt` contains the Linux/CUDA dependency floors.

## How the ACTUAL Tier-0/3 verdicts (../tier0-results.md, ../tier3-results.md) set the knobs

| Measured verdict | Knob setting (now the default) |
|---|---|
| M1 **NO-GO**, both pairs: row-space carries ≥99% of mKNN; relational signal richest at 25–50% depth, dead at final layer | Retained as a later layer-diagnostic hypothesis. The primary Tier-2 objective bake-off now fixes every arm to literal all-student-layer supervision, matching OPRD's reported Bridge setup and removing layer choice as a confound. |
| M1 no-go (identity output-readable) | `soft_knn_kl` (identity) runs as **control only** — keep it in ARMS for the control row, never headline it; drop it entirely with `SKIP_ARMS=soft_knn_kl sbatch tier2_bridge_bakeoff.sbatch` if allocation is tight |
| M2: identity ranked first, local-distance ≈ global second (both pairs) — but M1 overrides identity-only as a treatment | LN-Dist stays the lead relational arm; no knob change |
| M3 **AMBIGUOUS**, rollout-level trend weakly negative for geometry | a functional-bridge (`_fbridge`) arm variant is recommended by the Tier-0 verdict but is **deferred — not implemented in this run**; treat current geometric-arm wins as provisional until it exists |
| M4: OPRD rel. variance 0.389 vs OPD 0.857; power table (tier0-results.md) says 2-pt effects need ~53–82 Avg@16 *eval samples* per arm; 1-pt needs 200–700 | `--eval-avg-at 16` × 3 training seeds ≈ 48 samples/arm — adequate for 2-pt effects only; never claim 1-pt differences from this design |
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

Production Tier 2 is prohibited. Do not submit `tier2_bridge_bakeoff.sbatch` for `oprd_bridge`
until the external-artifact smoke passes and the artifact path, manifest digest, revisions, quota,
and cost have been reviewed and separate production approval is recorded.
The portable Tier-2 wrapper requires `STUDENT_REVISION`, `TEACHER_REVISION`, `TOKENIZER`,
`TOKENIZER_REVISION`, `DATASET_REVISION`, `DATASET_SPLIT`, and `PROMPT_KEY` for every runnable
arm. OPRD tasks additionally require `BRIDGE_ARTIFACT`, `BRIDGE_MANIFEST_SHA256`, and
`BRIDGE_STAGE1_CONTRACT_SHA256`; the production wrapper rejects the smoke-contract digest. The
Nexus wrapper derives the shared identity from `contract.json` and the artifact digest from
`verify.json`.
