# Adversarial sweep: geometry of fine-tuning updates / low-rank bias / subspace locking (2024-2026)

Task: is "OPD locks a low-rank subspace" already established elsewhere, making an OPRD-specific
replication (plan items A3/D) incremental? Also: what prior art threatens the PRH motivation /
novelty of OPRD-Bridge and the proposed ARH-Bridge (E)?

Bottom line up front: **yes, substantially incremental as currently scoped.** The exact diagnostic
suite proposed in A3/D (stable rank, early subspace locking, rank-constrained training) has already
been run on OPD itself (not an analogous method — the literal same method) in the paper the plan
cites as motivation. The phenomenon-class ("training collapses into a low-dim subspace early")
predates all LLM RL work by ~7 years. The defensible novel slice is narrow: whether OPRD's
hidden-state MSE gradient (dense, deterministic, not sampled-token KL) locks a *different* subspace
than OPD's, and whether OPD's locked subspace and OPRD's overlap. Nobody has shown that. Everything
else in A3/D is reconfirmation.

For OPRD-Bridge/ARH-Bridge: the cross-architecture low-rank/relational representation-matching
space is more crowded than the plan implies. At least one paper (LoRi) already does cross-arch
reasoning-trajectory distillation with global second-order-statistic matching in a shared low-rank
subspace — close in spirit to what ARH-Bridge proposes to do differently. And a CKA/Gram-based
relational loss (the plan's "PRH-flavored control," expected to be a weak strawman) already beat
output-only matching in a real RL-post-trained-model distillation setting. That raises the bar for
ARH-Bridge: it needs to beat the Gram-matching control, not just raw pointwise MSE.

---

## 1. Direct hit: "On the Geometry of On-Policy Distillation" already runs A3/D on OPD

**On the Geometry of On-Policy Distillation**, arXiv 2606.07082 (June 2026)
https://arxiv.org/abs/2606.07082 / https://arxiv.org/html/2606.07082

Verbatim abstract (confirmed via fetch of the abs page):
> "On-policy distillation (OPD) is increasingly used to improve large language model reasoning, but
> its training dynamics remain poorly understood. We characterize the trajectory of OPD updates in
> parameter space and compare it with supervised fine-tuning (SFT) and reinforcement learning with
> verifiable rewards (RLVR). A suite of parameter-space diagnostics consistently places OPD in a
> relaxed off-principal regime: compared with SFT, its updates affect fewer weights and avoid
> principal directions more strongly, while compared with RLVR, they remain less tightly
> constrained. Beyond this static localization, OPD exhibits subspace locking: its cumulative
> updates rapidly enter a narrow low-dimensional channel. Constraining training to the update
> subspace formed early in training preserves OPD performance but substantially degrades SFT,
> indicating that the locked subspace is functionally sufficient for OPD. Control experiments
> further show that sparsifying the update tokens and shifting rollout generation off-policy
> preserve the rank dynamics, whereas mixing the OPD objective with RLVR changes them. Overall,
> these results suggest that OPD is not merely an intermediate point between SFT and RLVR, but
> induces its own update geometry in parameter space."

Additional detail from a full-text fetch (metric definition and rank-16 experiment, moderate
confidence — obtained via summarization tool, not hand-verified against raw PDF, but consistent
with the verified abstract):
- Stable rank metric: `srank(ΔW_t) = ||ΔW_t||_F² / ||ΔW_t||_op²`.
- "OPD stays within a narrow low-rank band throughout training. SFT progressively expands its
  update subspace, while RLVR contracts toward a low-dimensional endpoint."
- "OPD aligns with its final V₁₆ subspace from the first measured checkpoint, whereas SFT and RLVR
  converge more gradually" — i.e. rank-16 early-subspace lock, measured from the *first* checkpoint.
- Rank-16 projection: "leaves OPD intact but degrades SFT" — this **is** the rank-constrained
  training experiment (D) already run, on OPD, at rank 16, with SFT as the negative control.
- Additional control experiments already performed: token sparsification, off-policy rollout shift
  (rank dynamics preserved), OPD+RLVR objective mixing (rank dynamics change).

**Implication for A3/D as scoped in the plan:** the plan cites this paper as *cf.* motivation, but
its own list of ingredients — stable rank, early subspace locking, rank-constrained training — is
not a novel diagnostic suite to invent, it is the exact suite this paper already ran, on the exact
method (OPD) the plan wants to compare against. Running it again on OPD replicates a known result.
The paper explicitly frames itself as *already* characterizing OPD vs SFT vs RLVR; it does not
touch OPRD (hidden-state MSE) at all — that is the actual gap. If A3/D is pitched as "does OPD lock
a subspace," it is answered. If pitched as "does OPRD's hidden-state-MSE gradient lock a subspace,
and is it the *same* subspace as OPD's," that is unaddressed and is the only defensible framing.

---

## 2. RLVR subspace/off-principal localization is model-conditioned, not recipe-specific — weakens the "discovery" framing

**The Path Not Taken: RLVR Provably Learns Off the Principals**, arXiv 2511.08567 (Nov 2025)
https://arxiv.org/abs/2511.08567

Verbatim abstract (confirmed):
> "Reinforcement Learning with Verifiable Rewards (RLVR) reliably improves the reasoning
> performance of large language models, yet it appears to modify only a small fraction of
> parameters. We revisit this paradox and show that sparsity is a surface artifact of a
> model-conditioned optimization bias: for a fixed pretrained model, updates consistently localize
> to preferred parameter regions, highly consistent across runs and largely invariant to datasets
> and RL recipes. We mechanistically explain these dynamics with a Three-Gate Theory: Gate I (KL
> Anchor) imposes a KL-constrained update; Gate II (Model Geometry) steers the step off principal
> directions into low-curvature, spectrum-preserving subspaces; and Gate III (Precision) hides
> micro-updates in non-preferred regions, making the off-principal bias appear as sparsity. ...
> RLVR learns off principal directions in weight space, achieving gains via minimal spectral drift,
> reduced principal-subspace rotation, and off-principal update alignment. In contrast, SFT targets
> principal weights, distorts the spectrum, and even lags RLVR. ... we show that RL operates in a
> distinct optimization regime from SFT, so directly adapting SFT-era parameter-efficient
> fine-tuning (PEFT) methods can be flawed..."

Key adversarial point: the paper's central claim is that off-principal, low-curvature update
localization is **"largely invariant to datasets and RL recipes"** and driven by **Gate II (Model
Geometry)** — a property of the *pretrained model*, not of the specific training objective. If
subspace/off-principal locking is mostly predictable from base-model geometry rather than from
which post-training method is used, then "discovering" that OPRD also locks a subspace is close to
an a priori expectation, not a surprising empirical result attributable to OPRD's design. The
interesting, non-obvious question the plan should target is not *whether* OPRD locks a subspace but
*which* subspace, and whether it's the same one RLVR/OPD/SFT settle into on the same base model —
i.e., subspace-overlap analysis, which neither this paper nor 2606.07082 does.

Also directly corroborates 2606.07082's SFT-vs-non-SFT spectral finding from an independent paper
and independent mechanism story (Three-Gate Theory vs. stable-rank measurement) — two separate
groups converging on "SFT targets/distorts principal directions, RL-family methods don't." This
makes the SFT-vs-{RL,OPD} spectral asymmetry a fairly robust, already-multiply-replicated
background fact, not something A3 needs to re-establish from scratch.

## 3. The phenomenon class itself ("training subspace collapse") predates all of this by ~7 years

**Gradient Descent Happens in a Tiny Subspace**, Gur-Ari, Roberts, Dyer, arXiv 1812.04754 (2018)
https://arxiv.org/abs/1812.04754

Core finding (from search-result synthesis, well-established/classic result, not re-verified via
fresh fetch but widely cited and consistent with community knowledge): in a variety of deep
learning training scenarios, the gradient rapidly (within a short period of training) enters and
stays within a small subspace spanned by the top eigenvectors of the loss Hessian, roughly matching
the number of output classes; this subspace is preserved over long stretches of training.

Adversarial implication: "early subspace locking during neural-net training" is not new to
LLM post-training at all — it's a long-standing general SGD phenomenon. What is specific and
publishable about 2606.07082 / 2511.08567 is not that locking happens, but *which* method locks
into what kind of subspace (principal vs. off-principal) and whether the locked subspace is
functionally sufficient (rank-constrained training preserves performance). A3/D needs to be pitched
at that level of specificity for OPRD or it will read as rediscovering Gur-Ari et al. under a new
acronym.

## 4. General "RLVR updates are low-rank/sparse and reuse pretrained directions" — further background, not novel

**On Predictability of Reinforcement Learning Dynamics for Large Language Models**, arXiv 2510.00553
https://arxiv.org/pdf/2510.00553

From search synthesis: RLVR updates concentrate in a few regions with sparse, low-rank structure;
RL fine-tuning is biased toward staying close to the base model and reinforces signals already
present at pretraining rather than introducing new directions. Consistent with, and additional
independent support for, the off-principal/low-rank RLVR literature above. Not specific to
OPD/OPRD — general RLVR-updates-are-low-rank finding, by 2025 already well-populated
(see also GeoRA arXiv 2601.09361, and "Evaluating Parameter Efficient Methods for RLVR" arXiv
2512.23165, both surfaced in search but not independently fetched/verified here — flagging as
context only, not confirmed claims).

---

## 5. Cross-architecture / representation-space low-rank distillation prior art — bears on OPRD-Bridge and ARH-Bridge specifically

**LoRi: Low-Rank Distillation for Implicit Reasoning**, arXiv 2606.05315 (June 2026)
https://arxiv.org/abs/2606.05315

Verbatim abstract (confirmed):
> "Implicit chain-of-thought (iCoT) methods aim to internalize reasoning in large language models,
> but often underperform explicit CoT prompting. We empirically find that hidden-state reasoning
> trajectories exhibit low-rank structure. Motivated by this observation, we propose a low-rank
> distillation framework that transfers reasoning by aligning teacher and student trajectories in a
> shared low-rank tensor subspace using first- and second-order statistics. ... We evaluate the
> method across multiple model families, including LLaMA and Qwen, at different scales on
> mathematical reasoning benchmarks. Our approach consistently improves performance, especially on
> challenging multi-step tasks, approaching explicit CoT accuracy and outperforming prior iCoT
> distillation methods."

This is close prior art for both OPRD-Bridge and ARH-Bridge: cross-model-family (LLaMA/Qwen),
reasoning-trajectory, low-rank-subspace alignment, using **first- and second-order statistics**
(i.e., closer to covariance/Gram-style global structure than raw pointwise MSE, but also not the
same as ARH's local-neighbor-identity proposal). The plan should position ARH-Bridge explicitly
against LoRi's second-order-statistic matching, not just against naive MSE — LoRi already shows a
non-pointwise cross-architecture alignment loss winning on math reasoning benchmarks with LLaMA/Qwen,
which is very close to the plan's own target setting (DAPO-Math, AIME/AIMO, ~1.5B math students).

**A Token is Worth over 1,000 Tokens: Efficient Knowledge Distillation through Low-Rank Clone (LRC)**,
arXiv 2505.12781, NeurIPS 2025 Spotlight
https://arxiv.org/abs/2505.12781

From search synthesis + abstract fetch attempt (could not get full technical detail on
architecture-mismatch handling — flagging as context/partially-verified): LRC trains low-rank
projection matrices that (1) compress teacher weights into the student's shape (soft pruning) and
(2) align student activations, including FFN signals, with the teacher's ("activation clone"),
achieving 1000x+ training-token efficiency vs. training from scratch. This is structurally close to
OPRD-Bridge's "frozen teacher PCA projector + learned student linear map + MSE in rank-r space,"
published roughly 13 months before OPRD. Could not fully verify whether LRC's projector is frozen
post-hoc (PCA-style) vs. jointly learned, or whether it handles genuinely independent
architectures (as opposed to compressing one teacher down to a designed-in-advance student shape) —
this distinction matters for how directly it undercuts OPRD-Bridge's novelty and should be checked
against the full LRC paper before treating this as settled prior art.

**Demystifying Low-Rank Knowledge Distillation in Large Language Models: Convergence,
Generalization, and Information-Theoretic Guarantees**, arXiv 2603.22355 (March 2026)
https://arxiv.org/abs/2603.22355

Verbatim abstract confirms this is a *theory* paper built directly on top of LRC: proves
convergence rate O(1/√T) for low-rank-projected activation-cloning distillation, generalization
bounds scaling as O(r(m+n)/√n) in the projection rank r, an information-theoretic account of why
activation cloning maximizes teacher-student mutual information, and a principled optimal-rank
rule r* = O(√n). This matters because it supplies rigorous theoretical grounding for exactly the
"low-rank projected representation matching" family of methods that OPRD-Bridge belongs to —
independent of the Platonic Representation Hypothesis. If OPRD-Bridge's justification leans on PRH
as its main theoretical story, reviewers can point to this paper as an existing, more rigorous,
PRH-agnostic theoretical foundation for the same mechanism, making the PRH framing feel like an
after-the-fact narrative rather than a load-bearing justification.

**What Should Feature Distillation Transfer in LLMs? A Task-Tangent Geometry View** (aka Flex-KD),
arXiv 2507.10155 (July 2025 — predates OPRD by ~11 months)
https://arxiv.org/abs/2507.10155

Verbatim-adjacent abstract excerpt (confirmed via fetch): "Existing approaches typically rely on
direct feature matching or learned projections, implicitly treating representations as objects with
intrinsic meaning. However, the relevance of a representation dimension is determined solely by how
it affects the model's output." Their conclusion, per fetch: pointwise/raw hidden-state matching is
explicitly rejected in favor of preserving "dominant directions of functional contribution" —
a functional-geometry framing, not relational/neighbor-based, but a direct predecessor critique of
"just MSE the hidden states" as a distillation objective, published well before OPRD. Directly
relevant prior art for the plan's item (B): the claim "pointwise hidden MSE might not be the right
objective" is not novel to the ARH framing — it was already argued on different (functional-
relevance) grounds a year earlier. The plan should cite/distinguish from Flex-KD explicitly:
Flex-KD's argument is about *which dimensions matter* (task-relevance), ARH's is about *what
structure to match* (local neighbor identity vs. absolute position); these are compatible but
distinct critiques and the plan currently doesn't address the overlap.

**Beyond Output Matching: Preserving Internal Geometry in NVFP4 LLM Distillation**, arXiv 2606.05682
(June 2026)
https://arxiv.org/abs/2606.05682

Verbatim abstract (confirmed):
> "... we first provide a representation level diagnosis of QAD: output matching alone can mask
> internal degradation, because many intermediate activation geometries can yield similar
> teacher-aligned logits. Using CKA, we show that KL-only QAD can reduce layerwise representational
> similarity relative to the BF16 teacher, with especially severe drift in RL-post-trained models.
> This drift correlates with downstream bottlenecks on reasoning and coding tasks, suggesting that
> low bit recovery requires preserving internal geometry rather than matching outputs alone.
> Motivated by this finding, we propose CKA-QAD, a CKA-guided representational alignment method for
> NVFP4 QAD... The method adds a lightweight regularizer that preserves internal representational
> geometry during distillation by aligning layerwise Gram matrices through CKA. Across Nemotron 3
> Nano and Qwen3-4B-Thinking-2507, CKA-QAD substantially improves representational alignment and
> improves downstream reasoning and coding accuracy..."

This is the single most load-bearing adversarial finding for the plan's item (B)/(E) design choice.
Two points:
1. It directly supports the plan's premise that **output-only matching (i.e., OPD-style reverse-KL)
   under-preserves internal representation geometry**, "with especially severe drift in
   RL-post-trained models" — good independent support for the plan's OPD-vs-OPRD motivation, in a
   distillation (not pure RL) setting, on reasoning/coding tasks. This is `supports-plan` evidence.
2. But the fix that already works in this closely analogous setting is **CKA-guided Gram-matrix
   alignment** — exactly the kind of global/relational, PRH-flavored measure the plan proposes to
   demote to "control only, expected to underperform" per ARH's claim that global spectral measures
   (CKA, Procrustes, RV) lose cross-model scaling signal after null-calibration. Here, in a real
   RL-post-trained-LLM distillation task, plain CKA/Gram alignment (no local-neighbor refinement)
   already delivers measurable downstream reasoning/coding gains over output-only matching. This
   directly threatens the plan's assumption that the full-Gram control is a weak strawman: it is a
   documented working method in an adjacent setting. ARH-Bridge (E) and the local-neighbor loss (B)
   need to beat this Gram/CKA baseline, not just an MSE-only baseline, to justify the added
   complexity of "local neighbor identity, no distances." This is `refutes-plan` evidence against
   the framing (not necessarily against ARH's correctness, but against treating Gram-matching as an
   obviously-losing control).

Caveat: this is quantization-aware distillation (QAD) with a frozen same-architecture
teacher/student pair (bit-width mismatch, not width/depth architecture mismatch), so it's not a
direct cross-architecture test — but it is the closest empirical analogue found for "does a global
relational (Gram/CKA) loss beat pointwise/output-only matching in RL-post-trained reasoning
distillation," and the answer there is yes.

---

## Sources list

- On the Geometry of On-Policy Distillation — https://arxiv.org/abs/2606.07082 (abstract verified)
- The Path Not Taken: RLVR Provably Learns Off the Principals — https://arxiv.org/abs/2511.08567 (abstract verified)
- Gradient Descent Happens in a Tiny Subspace (Gur-Ari, Roberts, Dyer) — https://arxiv.org/abs/1812.04754 (classic result, not independently re-fetched; widely known/cited, treat claim as high-confidence but source-verification here is via search synthesis only)
- On Predictability of Reinforcement Learning Dynamics for LLMs — https://arxiv.org/pdf/2510.00553 (search synthesis only, not independently fetched/verified — treat as context)
- LoRi: Low-Rank Distillation for Implicit Reasoning — https://arxiv.org/abs/2606.05315 (abstract verified)
- A Token is Worth over 1,000 Tokens (Low-Rank Clone) — https://arxiv.org/abs/2505.12781 (search synthesis + partial fetch; architecture-mismatch handling not independently confirmed)
- Demystifying Low-Rank Knowledge Distillation in LLMs — https://arxiv.org/abs/2603.22355 (abstract verified)
- What Should Feature Distillation Transfer in LLMs? (Flex-KD) — https://arxiv.org/abs/2507.10155 (abstract/conclusion verified via fetch)
- Beyond Output Matching: Preserving Internal Geometry in NVFP4 LLM Distillation — https://arxiv.org/abs/2606.05682 (abstract verified)
- Self-Policy Distillation via Capability-Selective Subspace Projection — https://arxiv.org/abs/2605.22675 (abstract verified; noted as tangential — single-model self-distillation, not teacher-student, low relevance beyond showing gradient-derived low-rank subspaces are a broader post-training design pattern)
- (context only, not independently verified) GeoRA arXiv 2601.09361, "Evaluating Parameter Efficient Methods for RLVR" arXiv 2512.23165 — surfaced in search, general RLVR-is-compatible-with-low-rank-updates literature, not fetched.

## Notes on verification confidence

Abstracts marked "verified" were fetched directly from the arXiv `/abs/` page and quoted close to
verbatim (WebFetch summarizes through a small model even for abstract pages, so treat as high- but
not 100%-confidence transcription; recommend a final human/manual cross-check of exact wording
before quoting in a paper draft). Findings marked "search synthesis only" or "not independently
fetched" should be re-verified against the primary source before being relied on for a go/no-go
compute decision.
