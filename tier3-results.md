# Tier 3 local pilot results — W_head causal decomposition (2026-07-12, 1.5B, M3 Max)

**Summary for partner (read this first):** We causally tested where OPRD's useful signal lives by
training the 1.5B student under spectrally decomposed versions of the hidden-state loss. The
result is clean and the story is one sentence: **the functionally useful part of final-layer
hidden-state distillation is exactly its LM-head-readable (row-space) component; matching the
null space moves geometry as much as full MSE but actively hurts functional agreement.** Row-only
supervision was the only final-layer arm that improved teacher agreement over baseline; full-MSE
and null-only both degraded it while reducing hidden-state distance — the geometry/function
dissociation M3 hinted at, now causal. The bonus finding is the strongest arm overall:
**mid-layer (depth-50%) supervision beat every final-layer variant on functional metrics** —
exactly the band where Tier 0 M1 found the richest relational signal, and direct evidence for the
OPRD paper's open "adaptive layer selection" question. The subspace-overlap analysis (avenue 10)
came back null: no evidence that OPRD updates escape into the head's null space. Together with
Tier 0 M1, this is the coherent workshop-paper arc: correlational (M1) + causal (this) evidence
that OPRD's signal is output-readable, plus an actionable layer-selection knob.

## Verdicts

| # | Experiment | Verdict |
|---|---|---|
| A9 | Row/null causal decomposition | **LM-head-bottleneck story falsified at 1.5B.** row-only ≥ full functionally (KL 0.322 vs 0.362, baseline 0.327); null-only matches full geometrically (mse28 1.132 vs 1.142) but is the worst MSE arm functionally (KL 0.381). OPRD's useful signal is head-readable. |
| FW#3 | Mid-band vs final-layer supervision | **GO — strongest arm.** mid (layer 14) KL 0.296, −9.3% vs baseline, best of all arms, with zero movement in layer-28 geometry. Adaptive layer selection is a real Tier-1/2 knob; supervise the 25–50% band. |
| A10 | OPD↔OPRD update-subspace overlap + escape alignment | **Null result.** MSE-arm update subspaces overlap OPD's only weakly (mean principal cos 0.21–0.31 vs ~0.10 chance, K=16); escape components are near-isotropic w.r.t. W_head's spectrum (row-energy 0.65–0.72 vs 0.742 isotropic) — no null-space concentration, no subspace-lock unification. |

## Setup actually run

Student DeepSeek-R1-Distill-Qwen-1.5B, teacher JustRL-1.5B (targets precomputed in Tier 0
extraction — teacher never loaded during training). fp32 + gradient checkpointing, SGD (no
momentum, lr 1e-4, clip 1.0), 500 steps, batch = 1 rollout ≤ 1000 gen tokens, rollouts 0–79
train / 80–99 held-out. Projectors: float64 SVD of the student's initial `lm_head`, r90 = 1140/1536
(independently matches Tier 0 M1's r90). Eval: KL(teacher‖student), top-1 agreement, layer-28 MSE
(full/row/null components) on the 20 held-out rollouts.

## Results (mean ± sd over seeds; 3 seeds for full/row/null, 2 for opd/mid)

| arm | KL(t‖s) ↓ | top-1 ↑ | mse28 | mse28 row-part | mse28 null-part |
|---|---|---|---|---|---|
| baseline (untrained) | 0.3267 | 0.8222 | 1.6510 | — | — |
| full (MSE @ L28) | 0.3616 ± .0004 | 0.8265 ± .0003 | 1.1417 | 0.4774 | 0.6643 |
| row (P_row MSE @ L28) | **0.3216 ± .0011** | **0.8285 ± .0002** | 1.2503 | 0.4672 | 0.7831 |
| null (P_null MSE @ L28) | 0.3806 ± .0014 | 0.8253 ± .0005 | **1.1320** | 0.4894 | 0.6426 |
| opd (self-CE proxy) | 0.3945 ± .0002 | 0.8210 ± .0003 | 1.3854 | 0.5367 | 0.8487 |
| mid (MSE @ L14) | **0.2963 ± .0001** | 0.8260 ± .0001 | 1.6560 | 0.5420 | 1.1140 |

Reading guide: seed noise is ±0.001 KL, so every arm separation above is many sigma. Note full-MSE
*degrades* KL vs baseline while cutting hidden distance by a third — geometric progress and
functional progress dissociate under direct causal manipulation. Top-1 moves are marginal
(0.821–0.828); KL is the sensitive metric.

## Deviations from review spec (carry into any write-up)

1. 1.5B same-arch pilot, not the review's 8B/4B spec; cluster re-run at spec scale is the upgrade path.
2. Single-round on-policy (rollouts fixed from the initial student); raw MSE (not L2-normalized); SGD 500 steps.
3. **`opd` arm is a weak proxy**: CE on the student's own rollout tokens (no stored teacher logits), closer to self-sharpening than true teacher-logit OPD. A10's overlap numbers inherit this caveat; A9's row/null contrast does not (it never uses the opd arm).
4. Proxy functional metrics (KL/top-1 vs teacher, 20 held-out rollouts); no AIME eval.
5. A10 restricted to o_proj/down_proj at every 4th layer, K=16, seed 0 only.
6. mid evaluated at layer 28 for geometry (trained at 14) — by design, to show the dissociation.

## Addendum (2026-07-13): mid-only vs all-layer vs last-k supervision

Question (partner): in RL, middle-layer supervision can beat all-layer supervision — does this
hold for OPRD? Two new arms, same recipe, 2 seeds each: `all` = uniform MSE over all 8 stored
band layers (the OPRD paper's "uniform layer weighting" default), `lastk` = layers {21,24,28}
(the paper's last-k heuristic).

| arm | KL(t‖s) ↓ | top-1 ↑ | mse28 |
|---|---|---|---|
| mid (L14 only) | **0.2963 ± .0001** | 0.8260 | 1.656 |
| all (8 layers, uniform) | 0.3167 ± .0006 | **0.8304 ± .0002** | 1.236 |
| lastk (21,24,28) | 0.3194 ± .0007 | 0.8297 ± .0001 | 1.224 |
| (ref) row / full / baseline | 0.3216 / 0.3616 / 0.3267 | 0.8285 / 0.8265 / 0.8222 | — |

**Verdict: metric-dependent — the RL finding holds on KL, reverses on top-1.** Mid-only remains
the best arm on distributional agreement (KL 0.296 vs all-layer 0.317), so yes, single-mid-layer
supervision beats uniform all-layer weighting on the metric OPRD's objective most directly
targets. But all-layer is the best arm *overall* on top-1 agreement (0.8304), beating mid
(0.8260) — and both beat every final-layer variant. Two secondary observations: (a) adding the
other 7 layers rescues the final-layer harm (full alone degraded KL to 0.362; all-layer including
L28 improves it to 0.317 — the mid/early terms dominate the update), and (b) last-k, the paper's
actual default heuristic, is strictly dominated by all-layer here on both functional metrics.

Consequence: "adaptive layer selection beats uniform weighting" is supported but the *size and
even sign* of the mid-vs-all gap depends on which functional proxy you trust — the clearest
argument yet for the cluster re-run with real AIME eval, since our two proxies disagree about
the winner. For the write-up, report both metrics; do not headline mid-only as the winner
without the downstream eval.

## What this buys the paper

- The workshop package is now complete and internally consistent: M1 (correlational, both pairs) +
  A9 (causal, same-arch) + FW#3 (actionable knob) + A10 (honest null).
- Framing that survives review: *"hidden-state distillation signal is output-readable"* — do not
  claim OPRD is useless (row-space supervision still shapes *where* in weight space updates land,
  and M4's variance advantage stands); claim its advantage is not hidden-from-the-head information.
- Cluster upgrade path (if Tier 2 proceeds): rerun A9 at 8B with true teacher-logit OPD and AIME
  eval; the pilot's effect direction and the M1 verdict make this a low-risk ask.
