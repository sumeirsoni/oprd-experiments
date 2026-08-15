# OPRD relational-distillation experiments

Empirical program probing **On-Policy Representation Distillation** (OPRD, arXiv 2606.06021):
which representational structure actually transfers, where the useful signal lives relative to
the LM head, and which cross-architecture bridge objective deserves training compute.

Tiers 0/3, the Tier-1 sweep, the original confounded Tier-2 run, and the positive-control gate
are complete. The immutable Qwen3-4B to Qwen3-1.7B bridge also passed a single-H200 Stage-2 pilot,
but that cached-rollout run is not a paper reproduction. A separate official-code-faithful task 0
is packaged in [`cluster/official_stage2/`](cluster/official_stage2/). Its historical scientific
contract uses one eight-GPU node; the explicitly authorized production execution uses four H200s
under a tracked topology-deviation manifest backed by successful eight-GPU smoke and four-GPU
canary evidence. Ordered one-epoch DAPO traversal, batch 8, two fresh responses per prompt, FP32,
optimizer, loss, and the frozen all-layer rank-8 bridge remain unchanged. Evaluation and tasks 1-26
remain blocked until task 0 is accepted and separately reviewed.

## Results so far (read these two files first)

| Doc | What it contains |
|---|---|
| [`tier0-results.md`](tier0-results.md) | Four no-training kill-switch measurements. Headline: neighbor identity is **output-readable** (row-space carries ≥99% of the mKNN signal, both model pairs) → identity-only arms demoted to control. OPRD's gradient-variance advantage confirmed (0.39 vs 0.86). Seed-budget power table. |
| [`tier3-results.md`](tier3-results.md) | Causal follow-up at 1.5B: row-space-only supervision is the only final-layer loss that improves functional agreement; null-space supervision matches full-MSE geometrically but *hurts* function. Mid-layer supervision beats every final-layer variant on KL; all-layer beats it on top-1 (metric disagreement → needs real AIME eval on the cluster). |

Supporting docs: [`oprd-adversarial-review.md`](oprd-adversarial-review.md) (the research review
that defined the tiers), `tier0-execution.plan.md` / `tier3-local.plan.md` (execution contracts,
including deviations), [`evidence/`](evidence/) (paper digests).

## Run Tier 1-2 on the UMD cluster

Full detail is in [`cluster/README.md`](cluster/README.md). The external bridge smoke uses an
explicit setup, preflight, submission, and terminal reconciliation sequence:

```bash
# 1. Put the clone on scratch.
git clone <this-repo> /cmlscratch/$USER/oprd-experiments
cd /cmlscratch/$USER/oprd-experiments
mkdir -p cluster/logs

# 2. Create and validate the pinned Python 3.12 CUDA environment on an H200.
sbatch cluster/nexus/setup_env.sbatch
# Wait for SETUP_OK and venv/.oprd-env-ready.json before continuing.

# 3. Validate paths, the environment marker, scratch blocks and inodes, quota reporting,
#    Slurm commands, and the unclaimed canonical smoke artifact.
BRIDGE_SMOKE=1 cluster/nexus/submit_oprd_bridge.sh --preflight-only

# 4. Submit only the eight-prompt bridge smoke DAG. It stops after the two-step
#    external-artifact Stage-2 backward smoke and never submits production Tier 2.
BRIDGE_SMOKE=1 cluster/nexus/submit_oprd_bridge.sh

# 5. Reconcile every recorded Slurm job and authenticate all final outputs.
cluster/nexus/submit_oprd_bridge.sh --status cluster/job_ids/<record>.json
```

The old Tier-2 wrapper remains available only for project controls and now rejects task 0. Official
task 0 has its own contract, dependency environment, bridge-format adapter, source/data manifests,
eight-GPU smoke, accepted four-GPU canary, fixed four-H200 non-array production launcher, and
dual-track offline evaluation. Full task-0 training is authorized through the tracked execution
manifest; evaluation remains separately gated. The resource gates, scientific settings, and
acceptance rules are documented in `cluster/README.md`.

## Repo map

```
cluster/            Slurm scripts + fully implemented training harness  <- partner starts here
tier0-results.md    Tier 0 verdicts (go/no-go table on top)
tier3-results.md    Tier 3 causal results + layer-ablation addendum
tier0/scripts/      full measurement + training pipeline (Mac/MPS; reusable pieces for CUDA)
tier0/prompts.jsonl the exact 150 DAPO-Math prompts (seed 0) — reuse for comparability
tier0/rollouts/     the exact rollouts all measurements used
tier0/results/      raw JSON behind every number in the results docs
evidence/           digests of OPRD + rival papers, with the review's caveats
```

## Not in the repo (too large)

- `tier0/hidden/` (15 GB teacher/student hidden-state memmaps) and `tier0/t3/runs/` (46 GB
  checkpoints). Regenerate with `tier0/scripts/gen_rollouts.py` → `extract_hidden.py` →
  `t3_train.py` (≈2 h on an M3 Max, faster on CUDA — swap the mlx generation for vLLM), or ask
  Christian for a transfer.

## Environment

Measurements ran on: `torch 2.13.0, transformers 5.2.0, numpy 2.5.1, scipy 1.18.0`
(plus `mlx-lm 0.31.3` for Mac-only generation — replace with vLLM on the cluster).
`tier0/requirements.txt` pins these.
