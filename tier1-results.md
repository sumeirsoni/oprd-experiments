# Tier 1 results: τ-sweep, same-arch 1.5B pair (UMD cluster, 2026-07-14)

Runs: 24/24 complete (8 τ × 3 seeds), JustRL-1.5B → R1-Distill-Qwen-1.5B, pkt_tau loss on
layers {7, 14}, 500 steps each. Raw data in `oprd-experiment-results/runs/` (pulled from
`/cmlscratch/ssoni11/oprd-experiments`; checkpoints remain on cluster scratch).
Note: tier 1 records loss trajectories and a stable-rank probe by design. Functional ranking
comes from tier 2's AIME eval, not from this sweep.

## Numbers

| τ | final/initial loss (mean ± sd) | stable rank, layer 14 (mean ± sd) |
|---|---|---|
| 0.02 | 0.0163 ± 0.0014 | 20.79 ± 0.72 |
| 0.05 | 0.0169 ± 0.0020 | 20.43 ± 0.44 |
| 0.1 | 0.0173 ± 0.0019 | 20.99 ± 1.01 |
| 0.2 | 0.0189 ± 0.0028 | 20.62 ± 0.75 |
| 0.5 | 0.0257 ± 0.0054 | 20.78 ± 0.87 |
| 1.0 | 0.0292 ± 0.0015 | 20.78 ± 0.96 |
| 2.0 | 0.0395 ± 0.0029 | 20.53 ± 0.63 |
| 5.0 | 0.0616 ± 0.0052 | 20.58 ± 1.17 |

Untrained-student baseline (same model, same layer, computed from the Tier-0 extraction
pool, 40k tokens): stable rank **21.99**.

## Verdicts

1. **The whole distance↔identity interpolation trains stably.** Every τ from 0.02 (≈ hard
   neighbor identity, the ARH-style endpoint) to 5.0 (≈ RKD/LSP distance endpoint) optimizes
   cleanly, with the final loss at 1.6-6.2% of its initial value and tight seed agreement.
2. **No representation collapse anywhere, including the identity endpoint.** Stable rank is
   flat at 20.4-21.0 across the entire sweep vs 21.99 untrained: a uniform ~5% contraction
   with no τ-dependence. The review's concern that identity-flavored losses might collapse
   the representation is not realized at 500 steps / 1.5B.
3. Relative loss drop is monotone in τ (sharper targets are matched more completely). Loss
   values across τ are not directly comparable in magnitude (the objective changes with τ);
   the comparable quantity is the stable-rank probe, which is what stays flat.

## Caveats

- **Loss scale varies ~τ² across the sweep** (KL over softmax(cos/τ) with bounded cosines:
  at τ=5.0 losses sit at 1e-6 to 1e-7 and print as 0.00000 at 5 decimals; they are never
  exactly zero, verified against raw results.json). Since lr is fixed, gradient magnitude
  shrinks with τ² as well, so high-τ arms take smaller effective steps: "same 500 steps"
  is not "same training progress" across τ. Any τ-vs-τ comparison should either normalize
  the loss by τ² or tune lr per τ; the stable-rank conclusion is unaffected.

- Stable-rank baseline uses Mac-generated rollouts, trained probes use cluster rollouts
  (same model, layer, and pool-size ballpark). Treat the ~5% contraction as approximate.
- 500 steps is a short-horizon probe. Collapse dynamics over 5-10k steps remain untested.
- No downstream eval by design; τ selection for any Tier-2 follow-up should wait for the
  t2 AIME table and, if a relational arm wins there, a small τ×eval grid on the winner.
