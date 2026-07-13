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
# 1. clone + enter (put the clone — or at least cluster/runs/ — on SCRATCH: ~7-10GB/task)
git clone <this-repo> && cd <repo>/cluster

# 2. env: pip install vllm first (it pins its own torch), then
pip install -r requirements.txt        # cluster/requirements.txt — NOT tier0's (that's the Mac stack)

# 3. edit the marked blocks in BOTH sbatch files:
#    - "#SBATCH --partition" and the commented "--account" line
#    - the "EDIT: environment" block (your module loads / venv activation;
#      point HF_HOME at scratch — models+datasets download at job runtime)

# 4. smoke-test ONE task before submitting arrays — this exact command, it exercises
#    the vLLM-after-optimizer path and eval-sample diversity that a shorter smoke misses:
python train_harness.py --loss pkt_tau --tau 0.1 --steps 8 --n-prompts 8 \
  --teacher hbx/JustRL-DeepSeek-1.5B --student deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --prompts BytedTsinghua-SIA/DAPO-Math-17k --supervised-layer-fracs 0.25,0.5 \
  --batch-tokens 2048 --max-tokens 512 --rollout-refresh 5 --seed 0 \
  --eval aime24 --eval-avg-at 2 --out runs/smoke

# 5. submit
mkdir -p logs runs
sbatch tier1_tau_sweep.sbatch          # 24 tasks, ~96 GPU-h
sbatch tier2_bridge_bakeoff.sbatch     # 18 real tasks as shipped (phi pair auto-skips), ~180 GPU-h
```

**The sbatch defaults already encode the Tier 0/3 verdicts** (supervised layer band 0.25–0.5,
identity-only as control). The verdict→knob mapping, environment details, disk/walltime notes,
and what is verified vs. untested (vLLM/CUDA/Slurm paths can only be validated by the step-4
smoke) are all in `cluster/README.md`. Total as shipped ≈ **280 A100-hours**; halving and
top-up strategies are documented there.

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
