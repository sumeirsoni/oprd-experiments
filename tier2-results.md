# Tier 2 results: cross-width bridge bake-off (UMD cluster, 2026-07-15)

18/18 runs (6 arms x 3 seeds), Qwen3-4B -> Qwen3-1.7B-Base, 2000 steps, layers 0.25/0.5,
harness commit 1f55ead (faithful two-phase oprd_bridge). Raw data in
`oprd-experiment-results-2/runs/`. Phi pair skipped as shipped (no OT bridge).

## Avg@16 (%), 3 seeds, EVAL CAPPED AT 2000 TOKENS (see caveat)

| arm | AIME24 | AIME25 | mknn k10 (s0) | dist4 (s0) | note |
|---|---|---|---|---|---|
| minilmv2_selfrel | 2.57 +- 0.35 | 0.42 +- 0.00 | 0.712 | 0.679 | best on every metric |
| cka_gram | 0.69 +- 0.26 | 0.00 | 0.659 | 0.672 | healthy diversity |
| soft_knn_kl (control) | 0.21 +- 0.17 | 0.07 | 0.580 | 0.164 | |
| ln_dist | 0.07 +- 0.10 | 0.00 | 0.503 | 0.050 | degenerate text (dist4) |
| oprd_bridge | 0.00 | 0.00 | 0.567 | 0.245 | paper recipe, faithful |
| linear_map_mse | 0.00 | 0.00 | 0.504 | 0.110 | degenerate text (dist4) |

Paper Table 4 reference (different eval budget, see caveat): untrained student 0.6/0.2,
OPRD-Bridge r=8 5.0/3.8, OPD top-1 8.5/6.3.

## Dominant caveat: eval token budget

Eval generation used the default --max-tokens 2000. The paper's students average 5.3k-14.5k
tokens per response, so most completions here likely truncate before \boxed{} and grade as
wrong regardless of quality. Every accuracy above is structurally deflated and the table
must NOT be read as a final ranking or compared to the paper's numbers. Fix in flight:
re-eval all 18 checkpoints via --eval-only --max-tokens 12000 plus an untrained-student
baseline (--steps 0) under the identical protocol.

## What is real signal already

1. mknn (teacher-student representational alignment after training) tracks the AIME
   ordering: minilmv2 tops both. Consistent with tier 0/3 (alignment where it matters is
   head-readable and functional).
2. ln_dist and linear_map_mse collapsed into repetitive text (dist4 0.05-0.11 vs 0.67 for
   the healthy arms). Those failures are training outcomes, not eval artifacts.
3. The faithful oprd_bridge arm hit the paper's own stage-1/2 internal diagnostics
   (projection cosine ~95% at r=8) yet still shows weak generation quality, echoing the
   paper's own Table 4 result (OPRD-Bridge < OPD top-1 on Avg@16).

## Agreed next steps (order matters)

1. Re-eval all 18 ckpts + untrained baseline at 12k eval tokens (~15-20 H200-h). No
   decisions before this lands.
2. Winner-only all-layer variant (3 seeds), not a full-matrix rerun (~180 GPU-h saved).
3. Mid-vs-all-layer on AIME, same-arch, no bridge (4 arms x 3 seeds, ~50 GPU-h): still on,
   more valuable now that t2 shows alignment tracking AIME while t3 left mid-vs-all
   metric-dependent.
4. Open gap for the write-up: no first-hand OPD baseline arm in the cluster harness; add
   one if we want the "vs OPD" comparison under our own protocol.
