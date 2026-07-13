---
title: "OPRD Literature Review: Contrastive Objectives, Position Bias, and Geometry"
type: comparison
created: 2026-07-08
updated: 2026-07-08
tags:
  - self-distillation
  - contrastive-learning
  - representation-learning
  - optimization
sources:
  - "[[on-policy-representation-distillation]]"
  - "[[on-the-position-bias-of-on-policy-distillation]]"
  - "[[on-the-geometry-of-on-policy-distillation]]"
  - "[[contrastive-representation-distillation]]"
  - "[[codir]]"
  - "[[distiller]]"
  - "[[learning-beyond-teacher]]"
  - "[[entropy-aware-opd]]"
  - "[[tip-token-importance-opd]]"
  - "[[fire-opd]]"
  - "[[selectkd]]"
  - "[[phf]]"
aliases:
  - "OPRD Literature Review"
  - "Contrastive OPRD Review"
---

# OPRD Literature Review: Contrastive Objectives, Position Bias, and Geometry

## Research Frame

[[on-policy-representation-distillation|OPRD]] establishes that hidden-state supervision can beat output-space OPD by avoiding sampling variance and the LM-head bottleneck. The next experimental step should test whether OPRD can be improved or better understood along three axes:

- **Objective**: replace or augment pointwise hidden-state MSE with contrastive or relational representation objectives.
- **Position**: test whether OPRD has the same prefix/suffix asymmetry as output-space OPD.
- **Geometry**: measure whether OPRD exhibits OPD-style early subspace locking or changes the update trajectory.

## Contrastive OPRD

[[contrastive-representation-distillation|CRD]] and [[codir|CoDIR]] are the main precedents. They argue that pointwise hidden losses treat dimensions independently and miss relational structure. [[distiller|Distiller]] broadens this into a general KD design space where intermediate representation loss and layer mapping are major performance drivers.

### Candidate Objectives

- **MSE baseline**: OPRD's current all-layer hidden-state MSE.
- **Cosine / normalized MSE**: cheaper coordinate-alignment controls.
- **InfoNCE / CRD**: positive = teacher/student representation of same rollout object; negatives = other prompts, trajectories, or segments.
- **MI-style objectives**: [[distiller|Distiller]] suggests MI-alpha-style losses as a middle ground between pointwise and contrastive objectives.
- **Transition geometry**: [[phf|PHF]] suggests matching hidden-state displacements and Gram structure instead of hidden coordinates.

### Negative Sampling Risk

The central danger is false negatives. In fixed-image or GLUE-style settings, different examples are usually safe negatives. In reasoning, two different rollouts may solve the same problem, share a valid proof step, or differ only superficially. Contrastive OPRD should start conservatively:

- in-batch negatives from different prompts;
- optionally restrict negatives to different final answers or verifier outcomes;
- avoid same-prompt negatives until correctness or trajectory clustering is available;
- compare in-batch negatives against a memory bank only after measuring representation staleness under on-policy training.

### Hypothesis

> A contrastive or transition-level OPRD objective will improve over MSE when teacher/student hidden states have similar relational geometry but unstable coordinates. It will hurt if negatives are semantically false or stale.

## Position-Aware OPRD

[[on-the-position-bias-of-on-policy-distillation|Position Bias OPD]] shows that uniform OPD wastes supervision on late drifted suffixes. [[entropy-aware-opd|EOPD]], [[tip-token-importance-opd|TIP]], [[fire-opd|FiRe-OPD]], and [[selectkd|SelecTKD]] show that token value depends on teacher uncertainty, student uncertainty, teacher-student disagreement, trajectory reliability, and verifier acceptance.

### Diagnostics

Measure OPRD hidden loss and downstream gradient signal by position:

- prefix vs middle vs suffix OPRD-only training;
- per-position hidden MSE/cosine gap by layer;
- per-position output KL alongside hidden discrepancy;
- teacher entropy and student entropy by position;
- cumulative prefix drift as in IW-OPD;
- trajectory-level teacher likelihood as in FiRe-OPD.

### Weighting Candidates

- **IW-OPD-style prefix weight**: use cumulative output discrepancy to downweight drifted suffixes.
- **Hidden discrepancy weight**: emphasize positions with large hidden gaps, optionally capped to avoid noisy outliers.
- **TIP-style Soft-OR**: combine student entropy with hidden or output teacher-student disagreement.
- **FiRe-style soft weight**: multiply teacher confidence, student confusion, and hidden discrepancy, normalized per trajectory.
- **SelecTKD-style verifier gate**: apply a binary or residual weight around the OPRD loss.

### Hypothesis

> OPRD may show a different position profile from output-space OPD. Because OPRD's original design supervises the last 2000 response tokens and reports late-stage representation divergence, useful hidden-state signal may persist later than output KL. The right experiment is not to assume prefix weighting wins, but to compare prefix, suffix, last-k, and adaptive weights directly.

## Geometry of OPRD

[[on-the-geometry-of-on-policy-distillation|OPD Geometry]] shows that OPD rapidly locks into a low-dimensional update subspace. OPRD adds a hidden-state objective that sees directions invisible to the LM head, so it may alter this geometry.

### Diagnostics to Reuse

- update sparsity under bf16-aware thresholds;
- principal-angle rotation of weight singular subspaces;
- normalized spectral shift;
- update-mask overlap with principal and low-magnitude masks;
- stable rank of cumulative updates;
- Frobenius norm to separate low-rank structure from small-update artifacts;
- Hill tail estimate for spectral shape;
- subspace similarity between early checkpoints and final update subspace;
- rank-constrained training using an early top-k update subspace.

### Additional OPRD-Specific Diagnostics

- layer-wise hidden alignment vs layer-wise parameter movement;
- output KL improvement vs hidden cosine/MSE improvement;
- stable rank for `OPD`, `OPRD`, `OPD + OPRD`, `ExOPD`, and `contrastive OPRD`;
- whether OPRD's hidden objective expands the locked subspace or creates a distinct low-rank channel;
- whether token/position weighting changes spectral shape, unlike token sparsification in OPD Geometry.

### Hypothesis

> If OPD subspace locking is driven mainly by on-policy reverse-KL objective structure, OPRD should shift the trajectory because its hidden-state gradients expose LM-head-null directions. If the lock is driven mainly by the student rollout distribution and pretrained geometry, OPRD may preserve the lock while changing update scale and functional outcome.

## Recommended Experiment Sequence

1. **Reproduce OPRD baseline diagnostics**: standard OPD, OPRD, and OPD + OPRD on the same checkpoints with position-stratified hidden/output losses.
2. **Position probes before new objectives**: prefix-only, suffix-only, last-k, IW-style weighting, and hidden-discrepancy weighting.
3. **Geometry pass**: stable rank, subspace similarity, and rank-constrained training for OPD vs OPRD.
4. **Conservative contrastive OPRD**: in-batch different-prompt negatives only; compare token-level, segment-pooled, and transition-level variants.
5. **Combined best variant**: apply the strongest position weighting to the strongest representation objective, then re-run geometry diagnostics.

## Current Best Guess

The literature suggests the most promising first variant is **soft position-aware OPRD**, not fully contrastive OPRD. FiRe-OPD and TIP indicate that weighting is a low-risk improvement to uniform token supervision, while CRD/CoDIR raise harder negative-sampling risks under on-policy reasoning. A safe path is: diagnose OPRD position bias, add soft hidden/output hybrid weights, then introduce contrastive or transition objectives once positive/negative definitions are empirically grounded.

> [!open-question]
> Does the hidden-state gap peak late because reasoning representations diverge near solution synthesis, or does it merely reflect accumulated prefix drift where teacher hidden targets become less reliable?

> [!open-question]
> Can a transition-level objective inspired by [[phf|PHF]] preserve OPRD's zero-variance benefit while reducing coordinate-matching brittleness?
