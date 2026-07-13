# OPRD relational-distillation experiments

Empirical program probing **On-Policy Representation Distillation** (OPRD, arXiv 2606.06021):
which representational structure actually transfers, where the useful signal lives relative to
the LM head, and which cross-architecture bridge objective deserves training compute.

Everything that runs without a cluster is **already done** (Tiers 0 and 3, on a MacBook).
What's left — the Tier 1–2 training arms — is packaged in [`cluster/`](cluster/) for Slurm.

## Results so far (read these two files first)

| Doc | What it contains |
|---|---|
| [`tier0-results.md`](tier0-results.md) | Four no-training kill-switch measurements. Headline: neighbor identity is **output-readable** (row-space carries ≥99% of the mKNN signal, both model pairs) → identity-only arms demoted to control. OPRD's gradient-variance advantage confirmed (0.39 vs 0.86). Seed-budget power table. |
| [`tier3-results.md`](tier3-results.md) | Causal follow-up at 1.5B: row-space-only supervision is the only final-layer loss that improves functional agreement; null-space supervision matches full-MSE geometrically but *hurts* function. Mid-layer supervision beats every final-layer variant on KL; all-layer beats it on top-1 (metric disagreement → needs real AIME eval on the cluster). |

Supporting docs: [`oprd-adversarial-review.md`](oprd-adversarial-review.md) (the research review
that defined the tiers), `tier0-execution.plan.md` / `tier3-local.plan.md` (execution contracts,
including deviations), [`evidence/`](evidence/) (paper digests).

## ⚡ Run Tier 1–2 on the UMD cluster (5 steps)

Full detail in [`cluster/README.md`](cluster/README.md). Short version:

```bash
# 1. clone + enter
git clone <this-repo> && cd <repo>/cluster

# 2. point the sbatch headers at your allocation (2 lines per file)
#    edit in tier1_tau_sweep.sbatch and tier2_bridge_bakeoff.sbatch:
#      #SBATCH --partition=<your gpu partition>
#      ##SBATCH --account=<your account>        <- uncomment + fill
#    and adjust `module load cuda python` to the site's module names

# 3. implement the TODOs in train_harness.py (loss arms are fully specified in
#    its docstrings + cluster/README.md; the CLI contract is already wired)

# 4. smoke-test ONE task before submitting arrays
python train_harness.py --loss pkt_tau --tau 0.1 --steps 5 --n-prompts 8 \
  --teacher hbx/JustRL-DeepSeek-1.5B --student deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --supervised-layer-fracs 0.25,0.5 --seed 0 --out runs/smoke

# 5. submit
mkdir -p logs runs
sbatch tier1_tau_sweep.sbatch          # 24 tasks, ~96 GPU-h total
sbatch tier2_bridge_bakeoff.sbatch     # 36 tasks, ~360 GPU-h total
```

**The sbatch defaults already encode the Tier 0/3 verdicts** (supervised layer band 0.25–0.5,
identity-only as control). The verdict→knob mapping is a table in `cluster/README.md`.
Everything in `cluster/` is marked **UNTESTED-ON-CLUSTER** — written on a Mac, syntax-checked,
never submitted. Budget ~450 A100-hours for the full matrix; halve by running Tier 2 at 2 seeds
and topping up winners (seed budget per the M4 power table in `tier0-results.md`).

## Repo map

```
cluster/            Slurm scripts + harness stub for Tier 1–2  <- partner starts here
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
