# Adversarial sweep: PRH / ARH / representation-alignment reliability

Angle: does ARH disprove PRH, or refine measurement while leaving convergence intact? Is any PRH-motivated method (like OPRD-Bridge) shown to fail in practice? Are CKA/mutual-kNN/CKNNA reliable or gameable?

---

## 1. Platonic Representation Hypothesis — exact original claims

**Huh, Cheung, Wang, Isola. "Position: The Platonic Representation Hypothesis." ICML 2024 (oral).**
- https://arxiv.org/abs/2405.07987 / https://proceedings.mlr.press/v235/huh24a.html / https://phillipi.github.io/prh/

Claim (as stated by the authors, per search + prior knowledge): representations across differently-trained neural nets (different architectures, objectives, even modalities) are converging over time/scale toward a shared statistical model of reality — a "platonic representation." Evidence is a survey of increasing cross-model alignment (mostly measured via mutual-kNN and related metrics) as models get larger/more capable. They explicitly hedge: convergence is a *trend*, not universal — they discuss selective pressures toward it (task/data diversity, capacity, simplicity bias) and note where it should break down (specialist/narrow-task models, weight decay artifacts, non-SGD optimization, "edge of stability" effects, input heterogeneity). This scoping ("may be broken by X") is directly in the original paper's own limitations discussion, per search summaries of the position paper — I was not able to pull an exact quoted sentence for each item, so treat that specific list as **unverified paraphrase**, not a direct quote (direction: context).

Load-bearing point for OPRD-Bridge: PRH's *empirical* evidence base for cross-modal/cross-architecture convergence leaned heavily on mutual-nearest-neighbor–style local metrics, evaluated mostly on modest-size probe sets — this becomes the target of both critiques below.

---

## 2. ARH does NOT disprove PRH — it recalibrates and narrows it (refinement, not refutation)

**Gröger, Wen, Brbić. "Revisiting the Platonic Representation Hypothesis: An Aristotelian View." ICML 2026.**
- https://arxiv.org/abs/2602.14486 (v1 HTML: https://arxiv.org/html/2602.14486v1)
- code: https://github.com/mlbio-epfl/aristotelian ; project page: https://brbiclab.epfl.ch/projects/aristotelian/

Direct quotes pulled from the paper text via WebFetch:
> "After calibration, the previously reported convergence in global metrics largely disappears, suggesting it was driven primarily by width and depth confounders, whereas local neighborhood-based metrics retain significant cross-modal alignment."

> ARH statement: "Neural networks, trained with different objectives on different data and modalities, converge to shared local neighborhood relationships."

> On distances vs. identity: "models increasingly agree on which points are neighbors, but do not agree on the pairwise distances, since CKA-RBF with a small bandwidth shows no alignment after calibration."

Mechanism of the artifact (from search summaries, consistent across multiple sources):
- **Width confound**: raw similarity scores are inflated by embedding dimensionality relative to sample size — nontrivial "chance similarity" baseline that grows with width.
- **Depth confound**: standard practice reports max/top-k similarity over many layer-pairs; the more comparisons, the higher the expected max even under a true null. Classic "look-elsewhere effect."
- Fix: permutation-based null calibration — compare observed similarity to an empirical null built by permuting sample correspondence; stabilizes by K≈50-200 permutations.

**Verdict on the adversarial question: ARH refines, does not refute, PRH.** The paper's own framing is "Revisiting" (not "Refuting"/"Disproving"). It keeps a convergence claim — just relocates it from global spectral/geometric alignment (CKA, Procrustes, RV — measures of *how far apart* points are and how the whole space rotates/scales) to local topological alignment (mutual-kNN, CKNNA — measures of *who is near whom*). This is exactly the framing in the research plan and should be represented that way: ARH is a downgrade of PRH's claimed strength/locus, not a kill shot. **Direction: supports-plan** (the plan already frames it this way correctly) but also **refutes** any stronger claim that "ARH disproves PRH" if the plan or anyone downstream overstates it.

Caveat found in the null-calibration mechanics: the calibration is itself a design choice (permutation of "sample correspondence") — this is a legitimate methodological improvement, but it is not the only possible null model, and the paper does not (per available text) address whether their specific null is itself sensitive to width/depth in second-order ways. Flag as an open methodological question, not a resolved one. (direction: context)

---

## 3. A second, independent line of evidence undermining PRH — NOT via calibration, via evaluation-regime fragility

**"Back into Plato's Cave: Examining Cross-modal Representational Convergence at Scale."** arXiv:2604.18572
- https://arxiv.org/abs/2604.18572

This is a *different* critique mechanism than ARH's width/depth calibration — it attacks PRH's evidence base by scaling the evaluation set:
> "alignment is measured using mutual nearest neighbors on small datasets (≈1K samples) and degrades substantially as the dataset is scaled to millions of samples."
> "the reported trend of stronger language models increasingly aligning with vision does not appear to hold for newer models."
> alignment that remains "reflects coarse semantic overlap rather than consistent fine-grained structure."
> "Models trained on different modalities may learn equally rich representations of the world, just not the same one."

Important: this paper attacks the exact metric family (mutual-kNN) that ARH says *survives* calibration and is the one the research plan wants to build its relational loss on. Two things to reconcile:
- ARH's claim "mutual-kNN survives width/depth calibration" is about *architectural* confounds at fixed eval-set size.
- Plato's Cave's claim "mutual-kNN alignment degrades at eval-set scale" is about *sample-size* confounds, largely orthogonal to ARH's fix.
- Together they suggest mutual-kNN/CKNNA convergence claims are fragile along **two separate axes** (arch scale, data scale), not just one. A relational loss (proposal B/E) built on "top-k teacher neighbors within a token window" should be read as operating on a *small, local* neighbor set per anchor token — which sidesteps the large-eval-set degradation somewhat (it's not claiming global corpus-scale neighbor structure, just local-window neighbor structure) — but the plan should not cite ARH's "mutual-kNN survives" result as if it were established at the scale/setting OPRD will actually operate in (per-token, few-thousand-token rollout windows) without re-checking. **Direction: refutes-plan** (as a scope caveat — the survival of mutual-kNN convergence is evaluation-regime-dependent, not a settled fact transferable to arbitrary local-window use).

---

## 4. CKA/Procrustes-type metrics: independently shown to be gameable, pre-dating ARH

**"Reliability of CKA as a Similarity Measure in Deep Learning."** arXiv:2210.16156
- https://arxiv.org/abs/2210.16156
> CKA is "highly sensitive to a large class of simple transformations, which can naturally occur in the context of modern machine learning" and "the CKA value can be easily manipulated without substantial changes to the functional behaviour of the models." Authors "call for caution when leveraging activation alignment metrics."

**"Correcting Biased Centered Kernel Alignment Measures in Biological and Artificial Neural Networks."** arXiv:2405.01012
- https://arxiv.org/abs/2405.01012 (found via search; not independently WebFetched — treat specifics as **unverified**, direction: context)
- Per search snippet: adding isotropic noise to token-mean representations, or breaking semantic correspondence via kernel shuffling, *increases* alignment under (biased) CKA despite no real correspondence; debiased CKA does not show this artifact.

This is independent, earlier (2022/2024) confirmation that global spectral metrics like CKA are gameable — consistent with, and predating, ARH's width/depth critique. Good prior-art context: the community already knew CKA was untrustworthy before ARH; ARH's contribution is specifically the *systematic, parametric* width/depth confound and the calibration fix, not the first "CKA can be fooled" result. **Direction: prior-art.**

---

## 5. Is the surviving "local neighbor identity" signal itself explained away by something even more reductive? — Linear Representation Hypothesis

**"Representation Alignment Rests on Linear Structure."** arXiv:2605.28870
- https://arxiv.org/abs/2605.28870
> "Platonic alignment arises from the universal relationship between objects and attributes, which is encoded linearly in representations." Sparse-autoencoder-extracted linear object-attribute features "often exhibit stronger cross-modal alignment than their dense counterparts."

This is an adjacent, competing successor hypothesis to ARH (not the same authors) proposing that whatever convergence remains (including plausibly the local-neighbor-identity signal ARH finds) may itself be reducible to shared *linear* object-attribute encoding rather than some richer relational/topological structure. If true, this is a threat to the framing that "local neighbor identity is the fundamental unit of convergence" — it might instead be a downstream consequence of linear feature sharing, and a well-designed linear-probe-based method could be just as good as, or the actual reason for, kNN-based relational losses working. **Direction: context/refutes-plan** — worth at minimum acknowledging in related work; doesn't kill the ARH-loss proposal but complicates the theoretical story ("we align on relational/topological structure per ARH" vs. "we align on shared linear directions per LRH, and kNN just happens to pick that up").

---

## 6. Most load-bearing finding for OPRD specifically: representational convergence does NOT imply reasoning transfer, and it is weakest exactly where OPRD applies its loss

**"Convergence Without Understanding: When Language Models Agree on Representations but Disagree on Reasoning."** arXiv:2605.23315
- https://arxiv.org/abs/2605.23315 (https://arxiv.org/pdf/2605.23315, https://arxiv.org/html/2605.23315v1)
- Methodology: 16 models, 8 families, 1.5B–72B params, 800 reasoning problems, CKA + causal ablations.

Three findings, all directly adversarial to the OPRD/OPRD-Bridge premise:

1. **Difficulty inversion**: models converge *more* on problems they collectively **fail** (CKA = 0.897) than on problems they **solve** (CKA = 0.830). I.e., representational agreement is anti-correlated with the thing you actually want (correct reasoning).
2. **Generation gap**: pre-decision (input-processing) representations align strongly (CKA = 0.875); post-decision representations — i.e., representations produced *during generation*, exactly the region OPRD targets ("student-generated rollouts... last ~2000 response tokens") — diverge sharply (CKA = 0.274).
3. **Epiphenomenal correctness**: shared cross-model information is linearly decodable (66% transfer accuracy) but has almost no causal effect on the model's own predictions (1.5%–5.5% flip rate under ablation). Aligned representations are largely *not what the model is using* to decide its output.

**This is the single most important adversarial finding in this sweep for the overall OPRD research plan**, not just the PRH/ARH sub-question. OPRD's entire mechanism is: match teacher/student hidden states (MSE) on the student's own generated response tokens, on the theory that if hidden states align, downstream behavior/reasoning will follow (avoiding the "LM-head bottleneck"). Convergence-Without-Understanding says exactly the opposite regime is where representations diverge most (generation/post-decision) and that even where representations *do* align, that alignment is often causally inert. Two implications for the plan:
- It is independent, direct evidence (not just a theoretical PRH/ARH argument) that pointwise hidden-state MSE matching on response tokens may be matching a representational property that is *not the one driving the student's reasoning* — strengthening the case for proposal (B)'s move away from raw pointwise matching, but also raising the bar: even a *relational* (ARH-style, local-neighbor) loss on those same post-decision tokens could suffer the same problem if the issue is causal irrelevance rather than metric choice (global vs. local). A neighbor-identity-preserving loss on causally-inert dimensions of the representation is still matching something causally inert.
- It suggests an experiment the plan is currently missing: a causal-influence / ablation check (à la this paper's flip-rate protocol) on whatever the OPRD or ARH-loss training actually changes in the student, rather than assuming hidden-state similarity (of any flavor, pointwise or relational) is the right proxy target. This is a concrete, citable reason to add a causal-ablation diagnostic to (A3/D)'s update-geometry work.

**Direction: refutes-plan** (targets a load-bearing assumption shared by OPRD and by the plan's proposed relational replacement — that hidden-state alignment, however measured, is what carries reasoning capability across the response-token region).

Caveat: this paper studies *already-trained, independently-trained* models being compared post hoc, not models being *actively trained* to align via a distillation loss. It's evidence that *naturally arising* alignment doesn't imply shared reasoning; it is not direct evidence that *training to force* local-neighbor alignment (which is what proposal B/E would do) fails to transfer reasoning — that's a different causal direction and remains an open empirical question the plan should test, not assume answered either way. Flag this nuance explicitly when citing it.

---

## 7. Does any work show a PRH-motivated *method* (not just the geometric claim) failing in practice?

I could not find a paper that specifically evaluates an OPRD-Bridge-style "frozen teacher PCA + learned linear map + MSE in rank-r space" cross-architecture distillation method and shows it underperforming. The closest indirect evidence:
- Convergence-Without-Understanding (above) — general representational-alignment-doesn't-transfer-reasoning result, not a distillation-method ablation specifically.
- Back into Plato's Cave (above) — shows PRH's evidentiary base for cross-modal PCA/linear-map-style alignment is much weaker at scale than originally reported, undercutting the *motivation* for PCA-projector-based bridges, but again not a direct method failure.
- No direct hit on "PCA-projector distillation fails cross-architecture" specifically for LLM reasoning distillation. **Mark this as an open gap**: the plan's claim that "does ARH really undermine the PRH motivation of OPRD-Bridge" is currently argued from first principles (ARH says global spectral/PCA-style alignment is the confounded part) rather than from a direct empirical failure case of PCA-projector bridging. That argument is sound as a theoretical critique (PCA captures global spectral structure, which is exactly what ARH shows loses its convergence signal after calibration) but the plan should not claim there is direct empirical evidence of OPRD-Bridge failing — there isn't, in this sweep. **Direction: context** (gap, not a finding either way).

---

## Bottom-line answers to the two explicit adversarial questions

**Q1: Does ARH disprove PRH, or refine measurement while leaving the convergence claim intact?**
Refines, does not disprove. ARH's own framing ("Revisiting") and quoted results keep a (narrower, relocated) convergence claim alive: local neighbor identity, not global geometry, not local distances. Treat "ARH kills PRH" as an overstatement the plan should avoid; treat "ARH kills the *specific* global-spectral-alignment version of PRH that OPRD-Bridge's PCA-projector implicitly relies on" as well-supported.

**Q2: Does ARH really undermine the PRH motivation of OPRD-Bridge?**
Yes, at the level of theoretical motivation — OPRD-Bridge's PCA projector + MSE-in-rank-r-space is precisely a global-spectral-alignment mechanism (linear subspace matching), which is the part ARH shows loses its calibrated convergence signal. Switching to a local-neighbor-identity loss (proposal E) is the theoretically correct response to ARH. But: (a) mutual-kNN's own survival is evaluation-regime-fragile per "Back into Plato's Cave" and should be re-verified at the plan's actual scale/window rather than assumed from ARH's setting; (b) "Convergence Without Understanding" is a more severe threat than ARH to the *whole* hidden-state-matching paradigm (pointwise or relational) because it shows alignment is weakest and least causally relevant exactly in the post-decision/generation-token region OPRD operates on — this threatens proposal B/E just as much as it threatens OPRD's original MSE loss, and the plan currently has no experiment designed to rule this out (recommend adding a causal-ablation/flip-rate diagnostic).

---

## Sources list (all fetched or search-confirmed real; none invented)
- Huh et al., "Position: The Platonic Representation Hypothesis," ICML 2024 — https://arxiv.org/abs/2405.07987 , https://proceedings.mlr.press/v235/huh24a.html , https://phillipi.github.io/prh/
- Gröger, Wen, Brbić, "Revisiting the Platonic Representation Hypothesis: An Aristotelian View," ICML 2026 — https://arxiv.org/abs/2602.14486 , https://github.com/mlbio-epfl/aristotelian
- "Back into Plato's Cave: Examining Cross-modal Representational Convergence at Scale" — https://arxiv.org/abs/2604.18572
- "Reliability of CKA as a Similarity Measure in Deep Learning" — https://arxiv.org/abs/2210.16156
- "Correcting Biased Centered Kernel Alignment Measures in Biological and Artificial Neural Networks" — https://arxiv.org/abs/2405.01012 (search-snippet only, not directly WebFetched)
- "Representation Alignment Rests on Linear Structure" — https://arxiv.org/abs/2605.28870
- "Convergence Without Understanding: When Language Models Agree on Representations but Disagree on Reasoning" — https://arxiv.org/abs/2605.23315
- (context, not deeply pursued) "Is One Layer Enough? Training A Single Transformer Layer Can Match Full-Parameter RL Training" — https://arxiv.org/abs/2607.01232 — already cited in the plan for a different sub-question (layer selectivity), confirmed real and on-topic for A2, not re-litigated here since this sweep's scope is PRH/ARH.
