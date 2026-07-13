# Adversarial sweep: known failure modes of hidden-state/feature distillation for LLMs
Date: 2026-07-10. Target: kill weak claims / find redundant experiments in the OPRD relational-loss plan
(`/Users/christianyoon/Desktop/Projects/oprd/relational_oprd_experiments_7e321945.plan.md`).

---

## 1. MOST LOAD-BEARING FIND: PHF already tests "relational hidden-state objective vs pointwise MSE" in the exact OPSD-for-reasoning setting

**PHF: Privileged Hidden Flow for On-Policy Self-Distillation**, arXiv:2606.29340 (2026)
https://arxiv.org/abs/2606.29340

Full abstract (verified via WebFetch, quoted):
> "On-policy self-distillation (OPSD) trains a reasoning model on rollouts sampled from its own policy by matching a privileged teacher that also sees verified reference solutions. Existing OPSD objectives supervise only the output distribution... We propose Privileged Hidden Flow (PHF), which additionally distills how a privileged teacher's hidden states move along the same rollout. **Rather than forcing each student hidden vector to match the teacher vector at the same token position**, PHF aligns token-to-token transition directions and trajectory geometry over selected generated positions. The all-layer recipe also includes an adjacent-layer relation computed from these same transitions, **without pointwise hidden-state imitation**. Under the same 100-step training schedule, PHF improves the Average@12 aggregate over our reproduced OPSD baseline on Qwen3-1.7B, 4B, and 8B, with observed gains of about +2.2, +1.5, and +1.7 points... **Ablations distinguish the fixed PHF recipe from pointwise hidden-state matching, single-channel transition losses, and layer-subset choices**, supporting PHF as a compact hidden-flow extension to OPSD."

Why this matters for the plan:
- This is a *relational* (not pointwise-MSE) hidden-state objective — transition directions + adjacent-layer relations, explicitly invariant to shared trajectory offsets and orthogonal transformations of transition directions — beating pointwise hidden-state MSE, **in Qwen3 1.7B/4B/8B reasoning-model on-policy self-distillation**. That is close to the plan's exact target setting (~1.5B math-reasoning students, on-policy rollouts).
- It already ran the ablation the plan's item (B) wants to run: pointwise MSE vs. relational objective. Result: relational wins, modestly (+1.5–2.2 pts).
- It already ran a **layer-subset ablation** (item A2's territory) in this setting.
- Difference from the plan's proposed ARH-flavored local-neighbor-identity loss: PHF's relational signal is *temporal* (transition geometry along the generation trajectory, single trajectory per hidden state), not *cross-token / cross-neighbor identity* within a window/batch as ARH's mutual-kNN framing implies. So the plan's proposed loss is not literally identical to PHF's, but the plan should (a) cite PHF, (b) explicitly argue why ARH-style local-neighbor-identity (which-tokens-are-near-which, batch/window relational) is different from and possibly complementary to PHF's trajectory-geometry relational loss, and (c) consider PHF's relational objective as a required baseline, not just plain pointwise MSE, when testing "does a relational loss beat pointwise MSE" — otherwise a reviewer will ask "isn't this just PHF again?"
- Risk to kill: running (B) with *only* pointwise-MSE-vs-neighbor-loss and no PHF-style baseline invites a "you didn't compare to the closest prior art" rejection.

Caveat: PHF is OPSD (self-distillation, teacher = privileged version of same model with access to reference solution) not cross-model OPRD/OPD. Setting is adjacent, not identical — worth noting explicitly as a limitation of this citation, but it is close enough that the novelty claim in (B) needs sharpening.

---

## 2. Direct evidence for whitening/scale mismatch as a real, measured failure mode

**A Study on Hidden Layer Distillation for Large Language Model Pre-Training**, arXiv:2605.11513 (Google DeepMind, 2026)
https://arxiv.org/html/2605.11513v1

Verified quotes (via WebFetch):
> "the mean squared magnitude of the teacher... reaches approximately 3×10⁵, whereas the pre-trained Gemma3 270M student... only reaches ≈1×10⁵." — concrete, measured activation-scale mismatch across a teacher/student pair, addressed only via L2 normalization (not solved architecturally).
> "HLDF's improvement over KD occurs predominantly in regimes where KD itself performs poorly, while in regimes where KD is already efficient the gain becomes negligible."
> "HLD does not consistently outperform a well-tuned logit-based KD baseline when training budgets are strictly equalized."

Numbers: C4 perplexity, 123M student: HLDF 3.000 vs KD 3.005 (a 0.005 improvement — noise-level). HellaSwag 123M: HLDF 0.626 vs KD 0.622.

This paper **deliberately uses single median-layer matching, not all-layer MSE**, and still only gets a marginal, condition-dependent win over logit-based KD. This is direct counter-evidence to any framing that hidden-state distillation straightforwardly beats output-space distillation — it's regime-dependent and the effect size is small once KD is well-tuned. Relevant caveat for OPRD's own claimed win over OPD: is the OPD baseline in OPRD's paper well-tuned? (Not something I could verify from the abstract fetch of OPRD; flagging as a question the plan's authors should audit in the OPRD paper itself, not something I can confirm from outside.)

---

## 3. Direct evidence that all-layer MSE is not obviously better than final-layer-only, and can hurt via optimization complexity

**Task-Based Flexible Feature Distillation for LLMs / Flex-KD**, arXiv:2507.10155 (2025)
https://arxiv.org/html/2507.10155v2 (also v1: https://arxiv.org/html/2507.10155v1)

Verified quotes:
> "applying Flex-KD solely on the final hidden layer is sufficient to achieve strong performance, leading to the best performance/efficiency balance" — in contrast with methods such as CKA which require distillation across all hidden layers.
> "multi-layer distillation introduces a more complex optimization process, whereas last-layer distillation is simpler and more efficient."

Numeric evidence: final-layer distillation beat multi-layer configs on XSum (Fig. 6a); beat CKA baseline on IMDB classification (95.09% vs 94.65%) and XSum ROUGE-L (30.45 vs 29.20).

This directly bears on OPRD's design choice to match "all layers." It's evidence (from a different, non-on-policy, non-reasoning setting — instruction-following/summarization/classification, not math RL rollouts) that all-layer matching is not a free lunch and single/selected-layer approaches are a legitimate, evidenced alternative — which is exactly what the plan's A2 (layer-selectivity) proposes to test for OPD/OPRD. Good news for the plan: this is prior art *motivating* A2, not redundant with it (Flex-KD is not on-policy, not RL, not reasoning-specific), so A2 remains a reasonable, non-redundant experiment, but the plan should cite Flex-KD as the generalization hypothesis it's testing in a new (on-policy reasoning) regime rather than presenting "all-layer might be suboptimal" as a novel idea.

---

## 4. "On the Geometry of On-Policy Distillation" already runs much of proposed diagnostic (A3/D)

**On the Geometry of On-Policy Distillation**, arXiv:2606.07082 — already cited in the plan.
https://arxiv.org/pdf/2606.07082

WebFetch summary (paraphrased, PDF was not perfectly extractable so treat with lower confidence than the quote-level items above):
- Identifies **early rank collapse**: student representations rapidly lose effective dimensionality early in OPD training, before stabilizing (non-monotonic stable-rank curve).
- Explores **rank-constrained training** explicitly, finding moderate rank constraints can improve stability without hurting performance.
- Does track stable-rank metrics over training steps as an empirical diagnostic — i.e., the specific measurement apparatus the plan's A3/D wants to build already exists in this paper.
- Per my fetch, open questions the paper itself leaves open: theoretical explanation of *why* rank collapse happens, optimal rank-schedules across scales, and generalization beyond the on-policy setting studied.

Implication: A3/D as currently scoped ("stable rank, early subspace locking, rank-constrained training [for OPD/OPRD]") substantially overlaps with an existing paper's method and headline finding. This does not make the diagnostic worthless for the plan's purposes — the novel contribution would have to be specifically applying/contrasting it on **OPRD's hidden-state MSE objective vs. OPD's output-space objective**, since the existing paper's "OPD" framing was not confirmed by my fetch to include hidden-state-space OPRD-style variants. But the plan should not present stable-rank / early-rank-collapse tracking as a new method — it should explicitly position itself as "does the known OPD rank-collapse phenomenon also occur in OPRD's representation space, and does relational (ARH-style) loss avoid it" — a narrower, still-plausible claim. As scoped in the draft plan text I was given, this distinction is not sharp; recommend narrowing before spending compute. (Confidence: medium — my PDF extraction was imperfect; recommend the plan authors read the actual paper directly before finalizing A3/D scope.)

---

## 5. Position-bias and layer-concentration priors are real and already well-documented (grounds A1/A2, doesn't kill them, but confirms plan's premises are correctly sourced)

**On the Position Bias of On-Policy Distillation**, arXiv:2606.22600 (already cited by the plan)
https://arxiv.org/abs/2606.22600 — confirmed via search: "OPD using only the first 30% of tokens can perform comparably to using all tokens, whereas OPD using only the last 30% of tokens barely learns anything" — root cause is student rollouts diverging further from teacher distribution at later positions; proposes closed-form IW-OPD token weighting. This is solid grounding for A1 (position-bias probes) and for item C's "soft position weights," and item C is honest that it overlaps with IW-OPD. Note: this paper's fix is specifically for output-space KL — its applicability to hidden-state MSE targets is untested by this paper (an actual gap the plan could fill non-redundantly).

**Is One Layer Enough? Training A Single Transformer Layer Can Match Full-Parameter RL Training**, arXiv:2607.01232 (already cited)
https://arxiv.org/abs/2607.01232 — confirmed: RL-training (not distillation) gains concentrate in a small subset of middle transformer layers, consistent across 7 Qwen3/Qwen2.5 models, 3 RL algorithms, math/code/agentic domains. This is an RL-training result about which layers *matter for updates*, not about which layers carry good *distillation targets* — the plan's own framing ("cf. ... an RL-training result") is accurate and appropriately cautious; this is prior art motivating A2 by analogy, not a distillation result that would make A2 redundant.

---

## 6. Output-space OPD failure-mode literature exists and is NOT about hidden states — useful contrast/context, some redundancy risk for cheaper baselines

**Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes**, arXiv:2603.25562
https://arxiv.org/html/2603.25562v2

Confirmed failure modes (output-space, token-level OPD, not hidden-state):
1. Imbalanced token-level supervision — "most sampled tokens receive negative reward... optimization dominated by a small subset of locally positive tokens."
2. Unreliable teacher guidance on drifted student prefixes — "tokens with high teacher probability can remain rewarded even after the trajectory has drifted into repetition, self-resetting reasoning, or other meaningless continuations."
3. Tokenizer/special-token mismatch across models.

Fix proposed: **teacher top-K local support matching** (still output/logit-space, not hidden-state) — +5.0% single-task, +19.8% multi-task math score over masked baseline.

This paper does **not** touch hidden-state distillation at all, so it's not direct counter-evidence to OPRD/relational-loss claims. But it is a warning: a much cheaper output-space fix (top-K local support matching) already recovers large gains on math reasoning from OPD's known failure modes. Before the plan spends compute on hidden-state relational losses, top-K local support matching (or IW-OPD) should arguably be included as a cheap output-space baseline in any headline comparison table — otherwise "OPRD-with-relational-loss beats OPD" claims risk being an apples-to-oranges comparison against an under-tuned OPD baseline rather than the best available cheap fix.

**Escaping the KL Agreement Trap in On-Policy Distillation**, arXiv:2606.09471 — confirmed via search: identifies a "low-KL agreement trap" where student drifts into a degenerate prefix the teacher locally agrees with, producing weak supervision; proposes an online termination rule (KAT), +2.66% avg@k / +3.43% pass@k on 4 math benchmarks. Again output-space, not hidden-state — relevant context for A1/A3 framing of "why does OPD fail on drifted rollouts," not a hidden-state result.

**AsyncOPD: How Stale Can On-Policy Distillation Be?**, arXiv:2606.24143 — studies staleness of the *rollout policy* relative to the learner in asynchronous training pipelines (a systems/infra staleness axis), not staleness of hidden-state targets under a fixed synchronous schedule. Finding: "forward KL remains robust to stale rollouts, whereas reverse KL is more vulnerable because it is student-weighted." This is a different axis from what the task asked about ("on-policy target staleness" in the hidden-state-MSE sense) — flagging as **context only**, not directly transferable evidence about whether OPRD's hidden-state targets go stale under sampling drift within a single (synchronous) training run. I could not find a paper directly measuring representation-target staleness for OPRD/hidden-state on-policy distillation specifically — this looks like an actual gap, i.e., a legitimately non-redundant thing the plan could measure.

---

## 7. Independent support for OPRD's core "LM head is a bottleneck" premise (not a knock-down, but worth citing precisely)

**Lost in Backpropagation: The LM Head is a Gradient Bottleneck**, arXiv:2603.10145 (2026)
https://arxiv.org/html/2603.10145v1

Quote (from search synthesis, treat as paraphrase not verbatim since I did not WebFetch the full text directly): "95-99% of the gradient norm is suppressed by the output layer" — independent, differently-motivated (pretraining optimization, not distillation) evidence that the LM head is a genuine gradient bottleneck. This is **prior-art support** for OPRD's own motivating claim (bypassing the LM head is valuable) — useful for the plan to cite as independent confirmation, separate from OPRD's own paper. Direction: supports-plan (supports OPRD's premise, which the new plan builds on). Confidence: medium — I have not verified this exact percentage figure via direct fetch of the PDF, only via WebSearch's synthesized snippet; recommend the plan authors verify the number themselves before citing it precisely.

---

## 8. Cross-architecture relational KD prior art (bears on item E, ARH-Bridge)

Vision-domain "Neighborhood Relation-Based Knowledge Distillation" (NRKD), ScienceDirect / Neural Networks 2025 — **could not verify**: WebFetch was blocked (HTTP 403, paywalled). Only a WebSearch snippet title/description was available: "transfers local structure knowledge by constructing similarity matrices using feature representations of the teacher model for neighbor selection (K-nearest neighbors for each sample are selected from a mini-batch)." This is the same *general idea* as the plan's item B/E (match local neighbor identity, not absolute position), but in **vision/image classification, not LLMs, not on-policy, not reasoning**. Marking direction=context because I could not independently verify the actual claims/results behind the paywall — but flagging it because if it holds up, it means "local-neighbor-identity relational distillation" as a general KD technique is not novel in ML broadly (it has vision precedent going back further too — classic Relational KD, Park et al. CVPR 2019, and CRD, which the plan already correctly excludes as contrastive/negatives-based). The plan's actual novelty claim should be narrowly scoped to: applying an ARH-motivated, non-contrastive local-neighbor-identity loss to **on-policy LLM reasoning distillation**, and to the **cross-architecture bridge** setting — not "inventing relational distillation" in general. This is worth being explicit about in the writeup to preempt a "this is just RKD/NRKD" reviewer comment.

---

## Bottom line for the adversarial question

1. **Kill or narrow immediately**: A3/D as scoped substantially overlaps with "On the Geometry of On-Policy Distillation" (2606.07082)'s own method (stable rank + early rank collapse + rank-constrained training). Must reframe as "does this OPD-space phenomenon transfer to OPRD's representation-MSE objective, and does the relational loss avoid it" — not present the diagnostic itself as new.
2. **Sharpen before running**: item B's core comparison (relational vs. pointwise MSE hidden-state loss) has a close, quantified precedent in PHF (2606.29340) in an adjacent (OPSD, not OPD/OPRD) reasoning setting, where the relational objective already won by +1.5–2.2 pts with all the same intuition (pointwise imitation is unnecessarily rigid). The plan needs to (a) cite PHF, (b) explain how ARH-style local-neighbor-identity differs from PHF's trajectory-geometry relational objective, (c) consider including a PHF-style transition-geometry baseline alongside plain pointwise MSE, or the "relational beats pointwise" finding will read as confirmatory-only, not novel.
3. **Add a cheap baseline before claiming OPRD/relational-loss wins**: top-K local support matching (2603.25562) and IW-OPD (2606.22600) are cheap output-space fixes that already substantially close OPD's known failure modes on math reasoning. Any headline "hidden-state/relational beats OPD" table should include these, or risk comparing against a strawman OPD baseline.
4. **Real, uncontested known failure modes to cite for framing**: activation-scale/whitening mismatch is measured and real (2605.11513: ~3x magnitude gap, addressed only by L2-norm hack); all-layer MSE is not obviously better than final/selected-layer matching and can add optimization complexity for little gain (2507.10155); hidden-state distillation's advantage over well-tuned logit-based KD is regime-dependent and can vanish (2605.11513, "does not consistently outperform... when training budgets are strictly equalized").
5. **Genuine gap** (non-redundant, worth doing): I found no paper directly measuring *representation-target staleness* for on-policy hidden-state distillation specifically (as opposed to policy/rollout staleness in async pipelines, which AsyncOPD covers). If OPRD's hidden-state targets are computed on frozen-teacher-forward-passes over student rollouts within a single synchronous step, "staleness" per se may not even apply the way it does in async OPD — the plan should double check this concept transfers before building an experiment around it.
6. **Preempt the "you reinvented RKD" critique**: local-neighbor-identity relational losses have vision-domain precedent (NRKD, and classical RKD/CRD which the plan already excludes) — frame novelty as "ARH-motivated, applied to on-policy LLM reasoning distillation + cross-architecture bridge," not as a new loss family per se.

## Sources list
- OPRD (target paper): https://arxiv.org/abs/2606.06021
- PHF: Privileged Hidden Flow: https://arxiv.org/abs/2606.29340
- A Study on Hidden Layer Distillation for LLM Pre-Training: https://arxiv.org/html/2605.11513v1
- Task-Based Flexible Feature Distillation for LLMs (Flex-KD): https://arxiv.org/html/2507.10155v2
- On the Geometry of On-Policy Distillation: https://arxiv.org/pdf/2606.07082
- On the Position Bias of On-Policy Distillation: https://arxiv.org/abs/2606.22600
- Is One Layer Enough?: https://arxiv.org/abs/2607.01232
- Revisiting On-Policy Distillation: Empirical Failure Modes and Simple Fixes: https://arxiv.org/html/2603.25562v2
- Escaping the KL Agreement Trap in On-Policy Distillation: https://arxiv.org/pdf/2606.09471
- AsyncOPD: How Stale Can On-Policy Distillation Be?: https://arxiv.org/pdf/2606.24143
- Lost in Backpropagation: The LM Head is a Gradient Bottleneck: https://arxiv.org/html/2603.10145v1
- OISD: On-Policy Internal Self-Distillation of Language Models: https://arxiv.org/abs/2605.29089 (abstract not independently re-verified beyond WebSearch synthesis + github-list description; treat as medium confidence)
- NRKD (Neighborhood Relation-Based KD, vision domain): https://www.sciencedirect.com/science/article/abs/pii/S0893608025003089 (403, unverified — title/description only)
- awesome-on-policy-distillation list (used to find real arxiv IDs for PHF/OISD, cross-check against hallucination): https://github.com/chrisliu298/awesome-on-policy-distillation
