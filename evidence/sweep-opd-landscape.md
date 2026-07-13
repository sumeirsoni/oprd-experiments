# Adversarial sweep: on-policy distillation (OPD) landscape 2023-2026, prior art for OPRD / relational-OPRD plan

Goal: find prior art that makes proposed experiments (A1/A2/A3/D/B/C/E in the draft plan) redundant, and evidence
against the plan's/OPRD's framing assumptions. Sources below are real search/fetch results; quotes are as returned
by WebFetch (treat as paraphrase-with-quotes, not guaranteed byte-exact OCR of the PDF, since several were fetched
from PDF and the tool flagged extraction limits — flagged inline where confidence is lower).

---

## 1. Canonical / foundational OPD lineage (context — establishes what's "already known")

### GKD — Agarwal et al., "On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes"
- arXiv 2306.13649 (2023), OpenReview https://openreview.net/forum?id=3zKtaqxLhW, DeepMind pub page.
- Founding OPD paper: student samples its own rollouts, teacher scores them, generalized JSD loss family,
  integrates with RLHF. **Output-space only** — no hidden-state supervision at all.
- Direction: context / prior-art. This is the lineage OPRD is extending; nothing here anticipates representation-space
  distillation, so OPRD's basic "lift OPD into hidden-state space" framing is *not* contradicted by the founding paper.

### Thinking Machines Lab, "On-Policy Distillation" blog (Tinker cookbook)
- https://thinkingmachines.ai/blog/on-policy-distillation/ ; recipes at
  https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/distillation
- Reverse KL exclusively: "Our reward function minimizes the reverse KL, which pushes the student to approximate
  the teacher's behavior" — described as mode-seeking and "unhackable." 70% AIME'24 at ~9-30x lower compute than
  off-policy distillation. **Output-space only**, no feature/hidden-state matching mentioned anywhere.
- Caveats reported: mixing ≥30% chat data doesn't fully preserve IFEval; even zero-KL-in-expectation training on a
  model's own samples degrades performance due to finite-batch distribution shift; results depend on teacher policy
  being within student support.
- Direction: context. Confirms the field's most-cited applied reference for OPD never touches representations —
  supports OPRD's premise that hidden-state OPD is not mainstream practice, but see §2 for closer academic prior art
  that already did move into representation space before OPRD's June 2026 submission.

### Reverse-KL vs forward-KL ablations (DistiLLM, DistiLLM-2, MiniLLM, ICLR 2025 blogpost)
- MiniLLM arXiv 2306.08543; DistiLLM (2024) introduces skew-KL OPD-style objective; DistiLLM-2 (2025) adds a
  contrastive extension; ICLR Blogposts 2025 "On LLM Knowledge Distillation - A Comparison between Forward KL and
  Reverse KL" (https://d2jud02ci9yv69.cloudfront.net/2025-04-28-llm-knowledge-distil-157/).
- Consistent finding across this line: reverse KL beats forward KL because of mode-seeking vs mode-covering behavior,
  reverse KL avoids the student overestimating low-probability teacher regions.
- Direction: context. A plain reverse-vs-forward KL ablation is not novel by 2026 standards — well trodden. Not
  directly one of the plan's proposed experiments, but flag if any sub-experiment reduces to "reverse KL is better,"
  that alone is not a contribution anymore.

---

## 2. Prior art that already does on-policy + intermediate-representation supervision (BEFORE or independent of OPRD)
### — the single most load-bearing finding for this sweep —

### OISD — "On-Policy Internal Self-Distillation of Language Models"
- arXiv 2605.29089 (dated ~one week *before* OPRD's 2606.06021).
- **Self-distillation within one model**: final layer acts as internal teacher for intermediate layers, during
  rollout + GRPO optimization. Two mechanisms: (1) "Logit Alignment" transferring high-level reasoning behavior from
  final to intermediate layers, (2) "Attention Alignment" enforcing "consistent attention patterns... from the final
  layer to the selected intermediate layer." Loss: "signed advantage-weighted Jensen-Shannon alignment," explicitly
  *not* plain MSE. Claims "substantial and consistent improvements over strong reasoning RL baselines across four
  mathematical reasoning tasks," entirely "without requiring external privileged information."
- Direction: refutes-plan / prior-art. This already combines on-policy rollouts with intermediate-layer supervision
  and already uses a non-pointwise-MSE (JS-divergence + attention-map) alignment mechanism, one week before OPRD.
  It weakens OPRD's implicit "first to move OPD into representation space" framing (OPRD's claim is narrower —
  cross-model teacher/student MSE — but the research plan's authors should read this before claiming any
  "hidden-state supervision on-policy" territory as OPRD-original). It's not cross-architecture teacher-student, so
  it doesn't directly bear on OPRD-Bridge/PRH/ARH questions, but it is very close prior art for "does non-pointwise
  representation supervision on rollouts help" — i.e., partially pre-answers the *general* premise behind plan
  item (B), independent of the ARH framing.

### PHF — "Privileged Hidden Flow for On-Policy Self-Distillation"
- arXiv 2606.29340 (dated ~3.5 weeks after OPRD; likely written in response to / concurrent with it).
- Also **self-distillation** (privileged teacher = same architecture/weights as student, given extra context —
  not a separate larger/cross-architecture model, per available metadata; PDF text extraction was degraded, treat
  this specific point as medium-confidence, not byte-verified quote).
- Method is explicitly **relational, not pointwise**: "aligns token-to-token transition directions and trajectory
  geometry over selected generated positions," with "an adjacent-layer relation computed from these same
  transitions, without pointwise hidden-state imitation." States the objective "is exactly invariant to shared
  trajectory offsets; its local geometry term is also invariant to orthogonal transformations of transition
  directions." Tested on Qwen3 1.7B/4B/8B, gains of "+2.2, +1.5, +1.7" points on Average@12 over 100 steps.
  Explicitly motivated against pointwise output-only OPSD: "existing approaches supervise only the output
  distribution... without directly supervising the internal computation that produced that distribution."
- Direction: **refutes-plan (strongest single finding of this sweep)**. This is near-verbatim the mechanism the
  draft plan proposes as novel in item (B) — "replace pointwise hidden MSE with a relational loss" — except PHF's
  relational invariant is transition-geometry / orthogonal-invariance (a transport/geometry framing), not ARH's
  local-neighbor-identity-via-softmax-KL framing. The plan's B is differentiable from PHF (neighbor ranking/kNN
  identity vs. trajectory-transport invariance are different mathematical objects and different theoretical
  motivations — ARH vs. a generic "don't overfit to pointwise position" argument), but the plan MUST cite and
  explicitly contrast against PHF or a reviewer will call (B) a re-derivation of already-published work. It also
  means the core empirical question "does *any* relational hidden-state loss beat pointwise MSE for on-policy
  distillation" already has a documented "yes" (PHF's own ablation implied by its framing, not independently
  verified by us here) — raising the bar for (B) to show the *specific* ARH/neighbor-identity invariant matters,
  not just "any relational loss beats MSE."
- Caveat: same-architecture only (Qwen3 family, varying sizes but likely same tokenizer/arch), so this does not
  address cross-architecture width/depth mismatch — leaves room for proposal (E)/ARH-Bridge's cross-architecture
  claim, but weakens the "relational loss is the novel part" angle generically.

### DEAR — "Finding the Evidence" (token-selection for OPD via hidden-state similarity)
- arXiv 2606.22830.
- Not a representation *loss*; a **token-selection** mechanism using hidden states: identifies "decision" tokens via
  student entropy, and "evidence" tokens via "hidden-state cosine similarity to decision anchors, boosted by
  teacher-student divergence." Reports up to +2.5pp math, +5.7pp code over standard OPD across three student-teacher
  configurations.
- Direction: prior-art / partially refutes-plan item (C). Shows hidden-state-informed token/position weighting for
  OPD losses is already an active, validated idea — reduces the novelty of "use representational signal to
  soft-weight which tokens/positions get OPD supervision" in isolation. Plan's (C) is about weighting the *existing*
  OPD loss with position/trajectory soft weights (IW-OPD/TIP/FiRe-style, not hidden-state cosine to decision
  anchors) so it's not identical, but DEAR is close enough that (C) needs to cite it and differentiate the weighting
  signal (position/trajectory-based vs. hidden-state-cosine-based).

### "A Study on Hidden Layer Distillation for Large Language Model Pre-Training" (arXiv 2605.11513) — context only
- Found via search; hidden-layer distillation but for **pretraining**, not on-policy rollouts. Relevant only as
  general prior art that plain hidden-layer MSE/cosine matching is a known technique outside the on-policy setting;
  did not fetch full text, flagged as unverified beyond title/abstract snippet — direction: context.

### "Improving Language Model Distillation through Hidden State Matching" (OpenReview, https://openreview.net/forum?id=IcVSKhVpKu) — context only
- Search snippet only, not independently fetched/verified: reportedly uses cosine loss on hidden states, notes this
  "restricts the dimensionality of the student to the teacher's, severely limiting compression ratio," and mentions
  CKA as an alternative for mismatched dimensionality. Relevant as a possible citation for why OPRD-Bridge needs a
  projector (dimensionality mismatch problem is real and previously documented) but NOT verified as on-policy.
  Direction: context (unverified, cite with caution).

---

## 3. Update-geometry / stable-rank prior art (bears directly on plan items A3/D)

### "On the Geometry of On-Policy Distillation" — arXiv 2606.07082 (already known to the plan's authors)
- Confirmed via search only in this sweep (already cited in the plan); not independently re-verified here beyond
  what the plan already states. Direction: context (already in plan).

### "Dense Supervision, Sparse Updates: On the Sparsity and Geometry of On-Policy Distillation" — arXiv 2606.13657
- Close companion/possibly same-lab paper to the above, NOT currently cited in the draft plan.
- Findings (from abstract): OPD weight updates are "numerically full-rank but spectrally concentrated," lying
  "mostly away from the principal singular subspaces of the source weights," disproportionately hitting
  near-zero-weight coordinates. Updates are "small and coordinate-sparse... distributed across layers, with FFN
  modules showing the largest relative changes." Critically: **"training only the discovered subnetwork nearly
  recovers full-training performance"** — i.e., they already ran a rank/subnetwork-constrained training experiment
  and it worked. Also finds AdamW > SGD for OPD (unlike prior RLVR findings) because "dense teacher supervision
  preserves useful momentum structure and heterogeneous second-moment scales." Analysis spans "several language and
  vision-language model pairs."
- Direction: **refutes-plan (item A3/D)**. This paper already runs a version of "rank-constrained training" as a
  diagnostic/validation for OPD and already finds a low-rank subnetwork recovers full performance — this is close
  to (though not identical to — it's *weight*-update geometry/sparsity, not necessarily *hidden-representation*
  stable-rank/subspace-locking dynamics) what A3/D proposes. The plan's A3/D needs to (a) explicitly read this paper
  and its companion before running, (b) clarify whether its stable-rank diagnostic is about weight-update spectra
  (already covered here) or activation/representation-space stable rank across training (which would be a distinct,
  less-covered question and worth doing), and (c) not re-derive the subnetwork-recovers-performance result without
  citing it as a replication/extension rather than a novel finding.

---

## 4. Position bias / layer selectivity (bears on plan item A1/A2)

### "On the Position Bias of On-Policy Distillation" — arXiv 2606.22600 (already cited in plan)
- Confirms plan's characterization: "OPD using only the first 30% of tokens can perform comparably to using all
  tokens, whereas OPD using only the last 30% of tokens barely learns anything." Proposes IW-OPD
  (importance-weighted OPD, weight ∝ accumulated student-teacher prefix discrepancy), up to +6.9 AIME25 points.
- Direction: prior-art (already known to plan). Position bias in **output-space OPD** is now a well-established,
  multiply-replicated finding (see next two entries) — treat "position bias exists in OPD" itself as no longer
  novel; the open, apparently-still-unaddressed question is whether it applies identically to OPRD's
  representation-space loss (denser, deterministic per-position MSE) — that comparison (plan's A1) does appear to
  still be open territory, not found duplicated anywhere in this sweep.

### "When Are Teacher Tokens Reliable? Position-Weighted On-Policy Self-Distillation for Reasoning" — arXiv 2605.21606
- Finds within-sequence position is "the strongest tested predictor of privileged-teacher reliability, rather than
  ambiguous teacher entropy."
- Direction: prior-art. Second independent paper on OPD position-reliability — the sub-area is crowded (at least 3
  papers: IW-OPD, this one, and Blockwise Policy-Drift Gating below). A1 as "does position bias exist in OPD" is
  redundant; A1 should be scoped tightly as "OPD vs. OPRD *comparison*," which none of these three papers do.

### "Blockwise Policy-Drift Gating for On-Policy Distillation" — arXiv 2606.24084
- "Reweights OPD position losses with student-only block-level policy-drift gates... without altering teacher
  targets." Third paper in the position-reliability cluster.
- Direction: prior-art, same caveat as above.

### "Is One Layer Enough? Training A Single Transformer Layer Can Match Full-Parameter RL Training"
- arXiv 2607.01232 (matches plan's characterization).
- Confirmed: this is an **RL training** result (GRPO/GiGPO/Dr.GRPO), not a distillation result. "RL gains are
  highly concentrated in a small subset of, and in many cases even a single, transformer layers," concentrated in
  the middle of the stack, tested across 7 models / 3 RL algorithms / math+code+decision-making domains.
- Direction: context, confirms plan's own characterization is accurate ("an RL-training result," not an OPD/OPRD
  result). This means A2 (layer-selectivity for OPD/OPRD specifically) is NOT already answered by this paper — it's
  suggestive analogy only, genuinely open for the distillation setting. No layer-selectivity study specific to OPD
  or OPRD was found in this sweep; A2 looks like real open territory.

---

## 5. Evidence against OPRD's own framing / motivation (useful ammunition for adversarial review of OPRD itself)

### "Rethinking On-Policy Distillation of Large Language Models" — arXiv 2604.13016
- Predates OPRD by ~2 months. Argues OPD success requires (a) "compatible thinking patterns" between student and
  teacher — "challenges assumptions that any capable teacher suffices regardless of architecture similarity" — and
  (b) that the teacher must offer "genuinely novel capabilities... beyond what the student has seen during
  training." Flags a scalability worry: "OPD's apparent free lunch of dense token-level reward comes at a cost,
  raising the question of whether OPD can scale to long-horizon distillation." Reports that successful OPD works
  through "a small shared token set that concentrates most of the probability mass (97%-99%)," and proposes fixes
  "off-policy cold start and teacher-aligned prompt selection" for failing cases.
- Direction: refutes-plan / context (ammunition against OPRD's own motivation, not the new plan directly). The
  "97-99% probability mass on a small token set" finding is worth weaponizing against OPRD's "denser signal is
  strictly better" framing — if effective OPD learning already concentrates on a tiny high-agreement token set,
  OPRD's claimed advantage from *density* (layers × positions × hidden-dim scalars) may be doing something other
  than what OPRD's paper claims (e.g., stabilizing optimization rather than adding new information) — worth an
  explicit ablation baseline: does OPRD's gain survive if the OPD baseline is also restricted/weighted to the
  same high-agreement token set this paper identifies? If not run, OPRD's OPD baseline may be an artificially weak
  strawman.

### "Escaping the KL Agreement Trap in On-Policy Distillation" — arXiv 2606.09471
- Identifies a distinct failure mode of reverse-KL OPD, independent of OPRD's stated "sampling variance + LM-head
  bottleneck" story: "when the student drifts into an unrecoverable prefix, the teacher may locally agree with the
  degraded state, producing low reverse KL but little corrective training signal." Proposes KAT (trap-termination
  filtering), giving "+2.66% avg@k, +3.43% pass@k, -59.73% avg rollout length."
- Direction: **refutes-plan (methodological control gap)**. This is a *third, independent* explanation for why
  vanilla output-space OPD underperforms — not variance, not the LM-head bottleneck, but locally-agreeing-but-wrong
  trajectories. If the plan's OPD baselines (used throughout A1-A3, and as the comparison point for OPRD everywhere)
  don't control for the KL-agreement trap (e.g. via KAT-style filtering or equivalent), any claimed OPRD/relational
  advantage over "OPD" risks being confounded by this well-documented, fixable OPD pathology rather than by
  anything fundamental about output- vs. representation-space supervision. Recommend adding a KAT-filtered OPD
  baseline (or citing why it's out of scope) before attributing OPD's weakness to variance/bottleneck alone.

---

## 6. ARH paper — verified against the plan's characterization

### "Revisiting the Platonic Representation Hypothesis: An Aristotelian View" — arXiv 2602.14486
- Verbatim abstract obtained via WebFetch: "...We introduce a permutation-based null-calibration framework...
  We revisit the Platonic Representation Hypothesis with our calibration framework, which reveals a nuanced
  picture: the apparent convergence reported by global spectral measures largely disappears after calibration,
  while local neighborhood similarity, but not local distances, retains significant agreement across different
  modalities. Based on these findings, we propose the Aristotelian Representation Hypothesis: representations in
  neural networks are converging to shared local neighborhood relationships."
- Direction: supports-plan (confirms the plan's own characterization of ARH is accurate, not an exaggeration).
  This is a faithful restatement, not embellished — good sign for the plan's factual grounding on this specific
  citation. Does NOT independently verify ARH's claims are correct (out of scope for this sweep — that would need
  a separate replication-focused search), only that the plan is quoting it correctly.

---

## 7. Cross-architecture / PRH motivation for OPRD-Bridge

- Confirmed via search snippets only (not independently fetched in full): OPRD-Bridge "exploits the observation
  that heterogeneous models share a low-rank representational structure," using a frozen projector pair; example
  cited: "a rank-8 linear subspace achieves 95% cosine similarity between the projected representations" of a
  Qwen3-4B teacher and a student of different depth/width. Results: "OPRD-Vanilla closes the student-teacher gap
  where every output-space baseline plateaus, while training 1.44x faster and using up to 54% less memory."
- No pre-OPRD paper was found in this sweep that does cross-architecture (width/depth-mismatched) on-policy
  representation distillation via a PCA/projector approach — PHF and OISD above are both same-architecture. This
  narrower claim (cross-arch on-policy representation transfer) appears to be OPRD's actual novel territory, not
  contradicted by anything found here. This is exactly the terrain proposal (E) wants to attack with an
  ARH-neighbor-identity loss instead of PCA+MSE — genuinely appears open, not redundant, PROVIDED the plan engages
  with PHF/OISD in its related work rather than ignoring them.
- Direction: context (no refuting prior art found for the cross-architecture piece specifically).

---

## Bottom line for the plan's authors

1. **Kill/rewrite risk for (B):** PHF (2606.29340) already runs a relational (non-pointwise, transition-geometry)
   hidden-state loss for on-policy distillation and reports gains over pointwise baselines. (B) is not "first
   relational loss for on-policy representation distillation" — it must be reframed as "first *ARH-motivated
   local-neighbor-identity* loss," and run a head-to-head against PHF's geometry-invariant loss, not just against
   plain MSE, or reviewers will merge the two lines.
2. **Kill/rewrite risk for (A3/D):** "Dense Supervision, Sparse Updates" (2606.13657) already does rank/subnetwork-
   constrained training for OPD and finds a low-rank subnetwork recovers full performance. Scope A3/D narrowly to
   activation/representation-space stable-rank dynamics (not weight-update sparsity, which is already covered), and
   cite this paper explicitly.
3. **Weaken/scope for (A1):** "Position bias exists in OPD" is now a 3-paper-deep established fact (2606.22600,
   2605.21606, 2606.24084). Only the OPD-vs-OPRD *comparison* is novel — make sure the writeup doesn't present
   "OPD has position bias" itself as a finding.
4. **(A2) appears genuinely open** — no OPD/OPRD-specific layer-selectivity study found; "Is One Layer Enough?" is
   RL-training-specific as the plan already correctly notes.
5. **(C) has a close cousin in DEAR** (hidden-state-cosine-based token selection) — differentiate soft
   position/trajectory weighting from DEAR's decision/evidence-anchor selection explicitly.
6. **(E)/OPRD-Bridge cross-architecture claim appears not directly contradicted** by any prior art found — PHF and
   OISD are both same-architecture self-distillation, so the width/depth-mismatch terrain seems still open. This is
   the strongest place to spend compute per this sweep.
7. **Add a KAT-style (2606.09471) and/or high-agreement-token-restricted (2604.13016) OPD baseline control** before
   attributing any OPRD/relational advantage to representation-space supervision per se — two independent papers
   suggest vanilla OPD baselines can be artificially weak for reasons unrelated to OPRD's stated motivation
   (sampling variance / LM-head bottleneck), which would inflate apparent gains from any hidden-state-based method.

## Confidence / verification notes
- All arXiv IDs and direct quotes above came from live WebSearch/WebFetch results in this session (dated "today,"
  2026-07-10 per system context). Several PDF fetches (PHF 2606.29340 in particular) returned degraded/compressed
  text per the tool's own caveats — treat PHF's exact wording as medium-confidence paraphrase, not a verified
  byte-exact quote, though the substantive mechanism description (transition-direction + adjacent-layer relation,
  invariance properties) was returned as directly quoted text by WebFetch and is likely reliable.
- Did not independently fetch/verify: "A Study on Hidden Layer Distillation for LLM Pre-Training" (2605.11513),
  "Improving LM Distillation through Hidden State Matching" (OpenReview), or the full taxonomy of "A Survey of
  On-Policy Distillation for LLMs" (2604.00626) beyond its abstract fragment — flagged as context-only, cite with
  caution / re-verify before relying on them.
- Did not find any paper that runs the ARH-style local-neighbor-identity/CKNNA-flavored loss specifically (as
  opposed to PHF's transition-geometry loss or classic Relational KD's distance/angle loss) inside an on-policy LLM
  distillation setting — this specific combination appears to be genuinely unclaimed territory as of this sweep.
