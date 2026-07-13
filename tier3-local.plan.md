# Tier 3 local pilot — W_head causal decomposition + layer-band ablation (1.5B, Mac)

**Scope agreed 2026-07-12:** Tier 3 (review avenues 9–10) + OPRD future-work #3 (adaptive layer
selection), targeted at workshop level. This is the **1.5B local pilot** of the review's 8B spec —
also serves as the de-risking pilot for the cluster version. Deviations from spec are recorded, not
hidden.

## Design

Student: DeepSeek-R1-Distill-Qwen-1.5B, fp32 + gradient checkpointing, SGD (no momentum,
grad-clip 1.0), 500 steps, batch = 1 rollout ≤ 1000 gen tokens. Teacher targets: **precomputed**
justrl hidden states from `tier0/hidden/same_arch/` (layers 14, 28) — teacher never loaded during
training. Data: rollouts 0–79 train / 80–99 held-out (same prompts.jsonl, seed-0 subset).

Projectors: SVD of the student's own initial `lm_head` (float64, CPU), r at 90% spectral energy
(M1 measured r90 = 1140 / 1536). Cached once, shared by all runs.

### Arms (× seeds)

| Arm | Loss | Tests |
|---|---|---|
| `full` | MSE(s28, t28) | baseline (= final-layer band arm for exp #3) |
| `row` | MSE(P_row s28, P_row t28) | avenue 9: row-space-only supervision |
| `null` | MSE(P_null s28, P_null t28) | avenue 9: null-space-only supervision |
| `opd` | CE on rollout tokens | OPD reference + enables avenue-10-lite subspace overlap |
| `mid` | MSE(s14, t14) | future-work #3 lite: mid-band (M1's richest layer) vs final |

2 seeds per arm first (10 runs); top up seed 2 on `full/row/null` if the night allows.
Eval after each run (separate process): KL(teacher‖student), top-1 agreement, MSE row/null
components on the 20 held-out rollouts; baseline = Tier 0 `functional.npz` (pre-training student).
Avenue-10-lite (CPU, after all runs): per-param top-16 update-subspace principal angles OPD vs
OPRD arms + escape-component energy in W_head top/bottom singular directions.

### Verdict rules

- **Avenue 9:** if `row` ≈ `full` on functional metrics and `null` ≈ untrained baseline → the
  OPRD signal is head-readable (confirms M1 causally; OPD-equivalence story). If `null` moves
  functional metrics materially → head-bottleneck story survives; name the effect size.
- **Exp #3:** if `mid` ≥ `full` functionally → adaptive layer selection is a real Tier-1 knob.

## Deviations from review spec (carry into any write-up)

1. 1.5B same-arch pair, not 8B/4B.
2. Single-round on-policy: rollouts fixed from the initial student (no regeneration during training).
3. SGD 500 steps, not the review's full recipe; proxy functional metrics (KL/top-1), no AIME eval.
4. Raw MSE (consistent with Tier 0 M4), not L2-norm MSE.

## Resource safety (same contract as tier0 — MANDATORY)

One GPU job at a time; guard.sh running before anything; `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.85`,
low 0.6; caffeinate on every job; checkpoint every 100 steps (resumable); **smoke gate first**
(5 steps + eval on 2 rollouts, verify peak memory <20GB and s/step) — no campaign until it passes.
Budget guard: stop launching new runs after 08:00; write up whatever exists.

## Artifacts

`tier0/t3/`: `projector.npz`, `runs/<arm>_s<seed>/` (ckpt_final.pt bf16, losses.json),
`results/t3_eval_*.json`, `results/t3_subspace.json`, logs in `tier0/logs/t3_*.log`.
Final deliverable: `tier3-results.md`.
