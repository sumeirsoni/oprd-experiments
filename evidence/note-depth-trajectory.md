# Note: per-token depth-trajectory distillation (partner idea, 2026-07-20)

Idea: supervise how a single token's hidden state evolves ACROSS LAYERS (depth), instead of
relations across tokens at a fixed layer. Assessed via literature sweep + an empirical probe
on the tier0 same-arch states (`tier0/scripts/depth_traj_probe.py`,
`tier0/results/depth_traj_probe.json`).

## Prior art (ranked by closeness)

1. **FDD** (Gong et al., ACL 2025 Long, 2025.acl-long.1125) — closest. Depth-as-ODE-time;
   per token matches the depth trajectory and its first-order derivative (adjacent-layer
   finite differences, cosine distance). Aligns in **vocabulary space via each model's LM
   head** explicitly because raw hidden dims differ. Proportional layer mapping.
2. **DistillLens** (2602.13567) — aligns per-layer logit-lens prediction trajectories
   (vocab space again).
3. **MTA** (2605.01374) — depth-adaptive granularity along the layer trajectory; relational
   structure still within-layer.
4. **FSP** (Yim, CVPR 2017) — cross-layer Gram as distilled object; CNN ancestor, no LLM port.
5. PHF (2606.29340) itself includes an adjacent-layer relation term — one foot in depth.

**Novelty verdict: partially explored.** The concept (per-token depth trajectories as KD
targets) is taken by FDD/DistillLens. The open delta: doing it on **raw hidden-state geometry
via a per-token L×L cross-layer cosine Gram — projector-free, dimension-free**, exactly where
FDD punted to LM-head projection. Frame any pursuit directly against FDD.

## Probe results (8k tokens, layers {4..28}, JustRL vs R1-distill)

- Teacher-student per-token depth-Gram correlation **0.995**, but shuffled-token null
  **0.973**: ~97% of the depth-Gram is token-independent global structure (adjacent layers
  similar by residual-stream construction). A naive loss on raw depth-Grams would mostly
  match a population-mean object, not token-specific structure. Calibration (subtract mean
  Gram, match residuals) is mandatory — same lesson as ARH's calibrated-metric critique.
- Incremental signal exists but is modest: trajectory agreement predicts token KL after
  controlling final-layer cosine at partial Spearman **−0.099** (p≈1e-18).
- **Sign flip worth attention:** mid-layer (L14) cosine correlates **+0.16** with functional
  agreement; final-layer (L28) cosine **−0.39**; the depth-Gram (−0.37) inherits the final
  layer's sign. Consistent with tier3 (mid supervision wins KL) and M1 (mid-band signal).
- Measured norm growth L4→L28 ≈ **2.07×**: any unnormalized variant (naive subtraction,
  magnitude matching) is dominated by late-layer norms. Cosine/per-layer normalization only.

## Recommendation

Not a bake-off arm yet. If pursued, in order:
1. Calibrated depth-Gram probe (tier-0 style, CPU, hours): subtract population-mean Gram,
   test whether residual trajectory agreement predicts functional agreement beyond
   mid+final cosines. Kill-switch before any GPU.
2. If it survives: loss = match mean-centered depth-Gram residuals, mid-band layer pairs
   weighted (long-range pairs, not near-diagonal). Dimension-free → drops into the
   cross-width bake-off unchanged.
3. Variant worth testing alongside: supervise residual UPDATE directions (Δh between
   mid-band layers) — combines the depth idea with the mid-layer finding; nearest to FDD's
   derivative term, so needs the FDD comparison in the write-up.
