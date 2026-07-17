# Tier 2 results: cross-width bridge bake-off (UMD cluster, 2026-07-15/16)

18/18 runs (6 arms x 3 seeds), Qwen3-4B -> Qwen3-1.7B-Base, 2000 steps, layers 0.25/0.5,
harness commit 1f55ead (faithful two-phase oprd_bridge). Raw data in
`~/oprd-experiment-results/runs/`. Phi pair skipped as shipped (no OT bridge).

24k re-eval: job 7101591, harness with `answered_rate`/`truncation_rate` (1c02b90),
`--max-tokens 24000`, Avg@16. All 18 ckpts + untrained baseline completed (19/19).

## Avg@16 (%), 3 seeds — PRIMARY: max-tokens 24000

| arm | AIME24 | AIME25 | ans rate | trunc rate | note |
|---|---|---|---|---|---|
| **untrained baseline** | **3.12** | **2.50** | 0.76 | 0.19 | no distill; best overall |
| minilmv2_selfrel | 2.71 +- 0.62 | 0.42 +- 0.00 | 0.63 | 0.30 | only arm near baseline on AIME24 |
| cka_gram | 0.69 +- 0.32 | 0.00 | 0.53 | 0.24 | |
| soft_knn_kl (control) | 0.21 +- 0.21 | 0.07 +- 0.12 | 0.39 | 0.41 | |
| ln_dist | 0.07 +- 0.12 | 0.00 | 0.28 | 0.51 | mostly unanswered |
| oprd_bridge | 0.00 | 0.00 | 0.18 | 0.41 | paper recipe; s0 trunc 0.72 |
| linear_map_mse | 0.00 | 0.00 | 0.11 | 0.83 | near-total truncation |

Paper Table 4 reference (their protocol/budget): untrained student 0.6/0.2,
OPRD-Bridge r=8 5.0/3.8, OPD top-1 8.5/6.3. Our baseline is higher than theirs on this
extractor/prompting; trained OPRD-Bridge still does not leave the floor.

## Legacy table: max-tokens 2000 (superseded)

| arm | AIME24 | AIME25 | mknn k10 (s0) | dist4 (s0) | note |
|---|---|---|---|---|---|
| minilmv2_selfrel | 2.57 +- 0.35 | 0.42 +- 0.00 | 0.712 | 0.679 | best on every metric |
| cka_gram | 0.69 +- 0.26 | 0.00 | 0.659 | 0.672 | healthy diversity |
| soft_knn_kl (control) | 0.21 +- 0.17 | 0.07 | 0.580 | 0.164 | |
| ln_dist | 0.07 +- 0.10 | 0.00 | 0.503 | 0.050 | degenerate text (dist4) |
| oprd_bridge | 0.00 | 0.00 | 0.567 | 0.245 | paper recipe, faithful |
| linear_map_mse | 0.00 | 0.00 | 0.504 | 0.110 | degenerate text (dist4) |

## Eval budget findings (24k)

1. **Truncation still binds** for every arm (trunc 0.19–0.83). 24k did not make the
   budget irrelevant; longer CoT arms (esp. linear_map / ln_dist / oprd_bridge s0) still
   often hit the cap without `\boxed{}`.
2. **Raising the budget did not reorder the bake-off.** Ranking is essentially unchanged
   vs 2k; absolute AIME stays tiny.
3. **Untrained baseline ≥ most trained arms.** Distillation here is not helping Avg@16
   under our protocol — minilmv2 is the only arm competitive with init, and still loses
   on AIME25 (0.42 vs 2.50).
4. mknn from training still tracks the AIME ordering among trained arms (minilmv2 tops
   both). Training failures (ln_dist / linear_map collapse) remain training outcomes.

## Analysis of the 24k table (Fable, 2026-07-17)

1. **Conditional accuracy: the AIME24 gap is failure-to-answer, not wrong answers.**
   acc/answered_rate: baseline 3.12/0.76 = 4.1% per answered attempt; minilmv2
   2.71/0.63 = 4.3%. Identical. Training's primary casualty is concluding behavior
   (answered_rate 0.76 -> 0.63..0.11, roughly ordered by how hard each loss pulls the
   student). AIME25 shows a real conditional gap as well.
2. **Suspected protocol defect (mine): training rollouts were 2000 tokens TOTAL, the
   paper's are long with a last-2000-token loss mask.** Every arm supervised the student
   exclusively on truncated never-concluding traces; the untrained baseline never saw
   that data and answers most. This confounds "distillation <= untrained": the arm-vs-arm
   ranking may survive (shared data), but the vs-baseline claim cannot be made from this
   run. Fixed in harness 7d8aa3b (--rollout-tokens 12000 --supervise-last-k 2000).
3. The faithful oprd_bridge hit the paper's internal diagnostic (~95% projection cosine)
   and still scored 0.00 with answered_rate 0.18. If that repeats under the fixed
   rollout protocol, it is a genuine empirical challenge to Table 4; before then, no.

## Revised next steps (supersedes the list above; order matters)

1. **Positive control first (new `opd_topk` arm, harness 7d8aa3b):** same-arch 1.5B pair,
   opd_topk + minilmv2 + oprd_bridge (+ mid/all layer variants, absorbing the planned
   mid-vs-all AIME experiment), all under the fixed protocol
   (--rollout-tokens 12000 --supervise-last-k 2000, eval --max-tokens 24000). If opd_topk
   cannot beat the untrained baseline, the protocol cannot distill and nothing else is
   interpretable. ~6 arms x 3 seeds x ~4h ~= 70 GPU-h.
2. Full t2 rerun under the fixed protocol (~180+ GPU-h, longer rollouts cost more):
   now justified in principle because the current t2 is confounded, but gate it on the
   positive control passing.
3. Winner-only all-layer t2 variant: folded into step 1's layer variants; revisit after.
