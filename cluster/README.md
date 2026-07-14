# OPRD Tier 1–2 cluster scripts

**STATUS: harness fully implemented; verified locally + by independent adversarial review;
cluster (CUDA/vLLM/Slurm) paths untested by construction — run the smoke below first.**

Verified (2026-07-13):
- **All 7 loss arms** end-to-end on Apple MPS (Qwen2.5-0.5B): generation → teacher prefill →
  training steps → diagnostics, finite decreasing losses. Also: rollout refresh, AIME eval
  (with per-sample seeds → genuinely distinct Avg@N samples), `--eval-only`, `--track-stable-rank`.
- An independent adversarial code review confirmed KL directions, gradient flow, masking, and
  token alignment empirically, and its findings (vLLM API drift, in-process engine OOM,
  answer-normalization bug, degenerate Avg@N seeding) are fixed in the current version.

**UNTESTED** (no CUDA machine available where this was built):
- the vLLM worker subprocess (written against current vLLM docs: `TokensPrompt`, per-request
  `SamplingParams`; main process offloads models+optimizer to CPU before spawning);
- CUDA memory at 1.5–4B scale and real `--batch-tokens 16384`;
- the sbatch files on a real queue.

## Cluster smoke — run this ONE task before submitting any array

Includes a rollout refresh at step 5 so vLLM-after-optimizer-state is exercised (the failure
mode a refresh-free smoke cannot see), and a 2-sample eval so you can verify the two
completions per problem differ (if they're identical, per-request seeding is broken — stop):

```bash
python train_harness.py --loss pkt_tau --tau 0.1 --steps 8 --n-prompts 8 \
  --teacher hbx/JustRL-DeepSeek-1.5B --student deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --prompts BytedTsinghua-SIA/DAPO-Math-17k --supervised-layer-fracs 0.25,0.5 \
  --batch-tokens 2048 --max-tokens 512 --rollout-refresh 5 --seed 0 \
  --eval aime24 --eval-avg-at 2 --out runs/smoke
```

First run downloads ~7GB of models + datasets — budget that before the "~10 min" estimate.

## Environment

`pip install vllm` first (it pins its own torch), then `pip install -r requirements.txt`
(this directory — NOT `tier0/requirements.txt`, which is the Mac stack and pins mlx).
Each sbatch file has a marked `EDIT: environment` block for your module loads / venv
activation, and exports `HF_HOME` — **point it at scratch**: models + datasets download at
job runtime (compute nodes need internet or a pre-populated cache; all 5 models and 3
datasets are public and ungated, no HF_TOKEN needed), and ~24 concurrent tasks share it.

**Disk:** each task writes ~7-10GB under `runs/<task>/` (two bf16 checkpoints + a transient
fp32 `tmp_policy/` dump per generation round, deleted after each round). Put the clone (or at
least `runs/`) on scratch, not home.

**Walltime/failure behavior:** training results (`results.json` with losses/diagnostics) are
written BEFORE eval starts, so a walltime kill during eval loses only the eval — rerun it from
the checkpoint with `--eval-only --eval aime24,aime25 --out runs/<same-dir>`. There is no
mid-training resume; a task killed mid-training must rerun.

## Known limitation: cross-tokenizer pair

The phi4mini→Qwen pair is **not implemented** (needs the anchor-fingerprint OT bridge, review
§5.8; token-aligned losses are undefined across tokenizers — the harness exits with a clear
error). The tier2 sbatch **skips it by default** via `SKIP_PAIRS`; its 18 array tasks exit 0
in seconds. Pairs interleave every 6 task ids, so do NOT try to exclude them by trimming the
array range. To enable later (after the bridge exists): `SKIP_PAIRS= sbatch tier2_bridge_bakeoff.sbatch`
(set it EMPTY — unsetting it restores the default skip).

## Files

- `tier1_tau_sweep.sbatch` — 24 tasks (8 τ × 3 seeds), same-arch 1.5B pair, 500 steps.
- `tier2_bridge_bakeoff.sbatch` — 36 tasks (6 arms × 2 pairs × 3 seeds); 18 run as shipped
  (phi pair auto-skipped), 2000 steps each.
- `train_harness.py` — the full shared training harness (all arms implemented).

### Bridge procedure (per-arm answer to "does tier 2 follow the paper's two-phase recipe?")

- `oprd_bridge` — **yes, faithfully**: Stage 1 fits the rank-8 teacher PCA and the student-side
  linear projector with **both backbones frozen** (paper Eq. 12–13; we solve the convex Stage-1
  objective in closed form instead of the paper's 20-epoch AdamW — exact optimum, same procedure),
  then freezes the bridge permanently; Stage 2 distills L2-normalized projections (Eq. 14).
  `--bridge-rank` (default 8) exposes the paper's r.
- `linear_map_mse` — **deliberately not**: joint map+student training IS the TinyBERT convention
  this arm represents; comparing it against `oprd_bridge` is exactly the frozen-vs-joint bridge
  question. It is warm-started by frozen-backbone least squares so a failure reflects joint
  training, not random initialization.
- the other five arms are **projector-free by construction** (dimension-free losses) — there is
  no bridge to warm up; that's their design point.
- `requirements.txt` — Linux/CUDA dependency floors.

## How the ACTUAL Tier-0/3 verdicts (../tier0-results.md, ../tier3-results.md) set the knobs

| Measured verdict | Knob setting (now the default) |
|---|---|
| M1 **NO-GO**, both pairs: row-space carries ≥99% of mKNN; relational signal richest at 25–50% depth, dead at final layer | `SUPERVISED_LAYER_FRACS=0.25,0.5` |
| M1 no-go (identity output-readable) | `soft_knn_kl` (identity) runs as **control only** — keep it in ARMS for the control row, never headline it; drop it entirely with `SKIP_ARMS=soft_knn_kl sbatch tier2_bridge_bakeoff.sbatch` if allocation is tight |
| M2: identity ranked first, local-distance ≈ global second (both pairs) — but M1 overrides identity-only as a treatment | LN-Dist stays the lead relational arm; no knob change |
| M3 **AMBIGUOUS**, rollout-level trend weakly negative for geometry | a functional-bridge (`_fbridge`) arm variant is recommended by the Tier-0 verdict but is **deferred — not implemented in this run**; treat current geometric-arm wins as provisional until it exists |
| M4: OPRD rel. variance 0.389 vs OPD 0.857; power table (tier0-results.md) says 2-pt effects need ~53–82 Avg@16 *eval samples* per arm; 1-pt needs 200–700 | `--eval-avg-at 16` × 3 training seeds ≈ 48 samples/arm — adequate for 2-pt effects only; never claim 1-pt differences from this design |
| Tier 3 pilot (1.5B, ../tier3-results.md): row-only ≥ full functionally, null-only hurts; **mid-layer (depth 50%) supervision beat all final-layer arms on KL** (all-layer won on top-1 — metric-dependent) | confirms `SUPERVISED_LAYER_FRACS=0.25,0.5`; the KL/top-1 disagreement is exactly what the AIME eval here adjudicates |

## Allocation estimate (A100-80GB hours, rough)

| Job | Real tasks | Est. hrs/task | Total |
|---|---|---|---|
| Tier 1 τ-sweep | 24 | ~4 (500 steps + rollouts) | ~96 GPU-h |
| Tier 2 bake-off (as shipped, phi pair skipped) | 18 | ~10 (2000 steps + AIME Avg@16 eval) | ~180 GPU-h |
| **Total as shipped** | | | **~280 GPU-h ≈ 12 A100-days** |

(If the phi pair is later enabled, tier 2 doubles to ~360 GPU-h.) Estimates are extrapolated,
not measured — treat the first completed task's wall-clock as the real number. To halve tier 2:
`sbatch --array=0-23 tier2_bridge_bakeoff.sbatch` runs seeds 0–1 only (seed = task_id/12);
top up winners later with `--array=24-35`.

## Submission

```bash
mkdir -p logs runs
sbatch tier1_tau_sweep.sbatch          # defaults already encode the Tier-0/3 verdicts
sbatch tier2_bridge_bakeoff.sbatch     # phi pair auto-skips; SKIP_ARMS=soft_knn_kl ... to drop the identity control
```
