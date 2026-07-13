# Adversarial sweep: token-level weighting/selection in LLM distillation & SFT (2025-2026)

Scope: EOPD, TIP, FiRe, SelecTKD, IW-OPD, position weighting, 80/20 RLVR token-selection,
and whether ANY of these use hidden-state losses instead of output KL. Goal: assess how
crowded this space is relative to the draft plan's A1 (position-bias probes), A2 (layer
selectivity), C (soft position/trajectory weights), and B/E (relational hidden-state loss).

## Headline verdict

**This space is saturated, and saturated specifically on the axis the plan's A1/C
experiments target (position weighting for OPD).** There is already a paper that does
almost exactly A1+C combined — "On the Position Bias of On-Policy Distillation" (IW-OPD)
and "When Are Teacher Tokens Reliable? Position-Weighted On-Policy Self-Distillation"
(PW-OPSD) — both purely in output-KL space, both published before this plan. Meanwhile
the intersection the plan treats as a gap (token-selection/weighting COMBINED with
hidden-state/relational losses) is thinner but **not empty**: at least three 2026 papers
(SRA, PHF, DEAR) already combine hidden-state or relational signals with token-level
weighting in on-policy/self-distillation settings. None of them use ARH-style local-kNN
identity matching specifically, which is a genuine remaining gap — but the plan should
not describe "hidden-state loss + token selection" as novel; it should differentiate
sharply from SRA/PHF/DEAR on the *specific relational operator* (local-neighbor-identity
vs. cosine/geometric-distance-preserving vs. attention-pooled span alignment).

---

## 1. How crowded is token-weighting/selection in OPD overall?

Extremely. Two separate community-maintained "awesome" lists exist
(`chrisliu298/awesome-on-policy-distillation`, `thinkwee/AwesomeOPD`), plus at least two
survey/meta papers as of mid-2026:
- "A Survey of On-Policy Distillation for Large Language Models" — arXiv 2604.00626
- "A Formula-Driven Survey and Research Agenda for On-Policy Distillation" (Bowen Zhang) — arXiv 2606.22793

The formula-driven survey explicitly nests token-weighting/selection as a "gate or
weight" variable in its general OPD framework, i.e. it's already a recognized, named
axis of the design space, not an open question. Its own quoted open-problems list does
NOT include "combine token weighting with hidden-state losses" as a named gap — the
closest it gets is noting "hidden-state support" and "representation supports" appear
only in an appendix table (Table 11) with no detailed treatment. That's weak positive
evidence for a gap, but it's an omission, not a stated open problem — treat cautiously.

Papers found doing token-level weighting/selection in OPD (output-KL space unless noted):

| Method | arXiv | Selection signal | Loss space |
|---|---|---|---|
| TIP | 2604.14084 | student entropy + teacher/student divergence (2-axis) | output KL |
| EOPD | 2603.07079 | teacher token-level entropy, forward-KL where uncertain | output KL |
| EGAD | 2605.01732 | entropy curriculum + adaptive temperature + dual-branch | output KL |
| SCOPE | 2604.10688 | routes by trajectory correctness (teacher-ppl / student-ppl weighting) | output KL/MLE |
| Rock Tokens ("Cornerstones or Stumbling Blocks") | 2605.09253 | persistent high-loss "rock tokens" masked | output KL |
| FiRe-OPD | 2606.02684 | trajectory filter (teacher log-prob) + soft token reweight | output KL |
| ReNIO | 2606.23104 | reweights trajectories by student-teacher log-ratio over "pivotal" tokens | output KL |
| IW-OPD ("On the Position Bias of OPD") | 2606.22600 | **position**: weight by accumulated student/teacher prefix discrepancy | output KL |
| PW-OPSD ("When Are Teacher Tokens Reliable?") | 2605.21606 | **position**: sigmoid weight on normalized token index, tuned against entropy | output KL |
| Blockwise Policy-Drift Gating | 2606.24084 | block-level student-only policy-drift gates on position | output KL |
| DEAR ("Finding the Evidence") | 2606.22830 | decision tokens (entropy) + evidence tokens (**hidden-state cosine similarity**) | **hidden-state relational + output** |
| SEAD | 2606.28562 | joint teacher-student entropy as "competence probe" | output KL |
| SelecTKD | 2510.24021 | propose-and-verify (speculative-decoding-style accept/reject) | output KL |
| Token Cleaning (SFT) | (semantic scholar) | fine-grained data selection for SFT, not OPD | output/loss-masking |

That's 13+ independent token-weighting/selection papers in OPD/KD alone from
~Oct 2025–July 2026, on top of the RLVR-side literature (below). This is not a niche.

## 2. Position weighting specifically — kills or narrows plan items A1 and C

Two papers make the plan's A1 (position-bias probes for OPD) largely non-novel as an
*existence* claim, though there's room to differentiate on *diagnosis*:

**"On the Position Bias of On-Policy Distillation" (IW-OPD)** — arXiv 2606.22600.
Finding: "OPD using only the first 30% of tokens can perform comparably to using all
tokens, whereas OPD using only the last 30% of tokens barely learns anything" — later
rollout tokens drift off-teacher and provide degraded supervision. Fix: weight tokens by
accumulated student/teacher prefix KL discrepancy, upweighting early / downweighting
late. Results: up to +6.9 AIME-2025, tested same-size and cross-scale. This is a
head-on collision with the plan's A1 motivation (position bias in OPD) — the plan needs
to either (a) show OPRD's hidden-state loss has DIFFERENT position bias than OPD's (the
actually interesting comparative claim, since sampling-variance explanations for
position bias shouldn't apply the same way to a deterministic MSE objective), or (b)
drop A1 as redundant.

**"When Are Teacher Tokens Reliable? Position-Weighted On-Policy Self-Distillation"
(PW-OPSD)** — arXiv 2605.21606. This is the more dangerous one for the plan's framing:
it directly pits position against entropy as a reliability predictor and reports
**position wins**: "an oriented within-sequence position score is the strongest tested
predictor of teacher-token reliability, reaching AUROC 0.83 [0.66, 0.95]; local
uncertainty scores are at most 0.57." It also implements almost exactly the plan's item
C ("soft position/trajectory token weights"): `w(t) = w_min + (1-w_min)·σ((r-τ)/s)`, a
sigmoid ramp over normalized position, tuned on Qwen3-4B and cross-validated on
DeepSeek-R1-Distill-Llama-8B and Olmo-3-7B-Think, MATH-500/AIME24/AIME25/HMMT25. Gains
are modest (+1.0–1.1pp Avg@12) but the mechanism is essentially IW-OPD's rival, applied
to self-distillation. **Neither paper touches hidden-state losses** — both stay in
output-KL space and neither compares against OPRD-style MSE. That's the actual opening:
"does position-weighted supervision transfer to hidden-state MSE, and does OPRD have the
SAME position-reliability profile as OPD, given that OPRD's objective is deterministic
(no sampling variance) and per-layer rather than per-logit?" is a real, currently-untested
question — but the plan's proposal text should cite and explicitly contrast against
IW-OPD/PW-OPSD rather than presenting position-bias probing as its own contribution.

## 3. Layer selectivity (A2) — RL result exists (already known), no clean OPD analog found

Confirmed the "Is One Layer Enough?" RL-training result (arXiv 2607.01232) is the paper
already cited in context; this sweep did not turn up a distinct token-weighting paper
that ALSO does layer selectivity for OPD/OPRD specifically — layer selectivity and
token-weighting are being pursued by disjoint sets of authors right now. That's a mild
point in favor of A2 as a real gap, but low confidence since this sweep's search terms
targeted token-weighting, not layer selection — treat as context, not a cleared gap.

## 4. RLVR-side 80/20 token selection — real, large, but a different mechanism

**"Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement
Learning for LLM Reasoning"** — arXiv 2506.01939, NeurIPS 2025. Restricting policy-
gradient updates to the ~20% highest-entropy "forking" tokens matches full-gradient RLVR
on Qwen3-8B and *beats* full-gradient on Qwen3-32B (+11.04 AIME25, +7.71 AIME24) and
Qwen3-14B. This is RL (GRPO-style), not distillation — no teacher, no hidden states — so
it's motivating context for "high-entropy tokens carry disproportionate learning signal"
but is NOT a hidden-state-loss result and doesn't directly support or refute OPRD/ARH
claims. Relevant mainly as: (a) prior art the plan should cite for why entropy-based
token selection is a well-established prior in the *RL* literature (raises the bar for
what counts as a novel entropy-based finding in OPD), and (b) a useful null result to
contrast with PW-OPSD's finding that position beats entropy in OPD — i.e., the
entropy-forking-token story that works for RLVR does NOT straightforwardly transfer to
OPD reliability-weighting, which is itself a citable tension worth flagging in the plan.

## 5. Did ANY of these test hidden-state losses instead of output KL? — Yes, three, and they matter a lot

This is the most important finding for the plan's B/E items. The scan initially returned
"no" for TIP/EOPD/Rock-Tokens/IW-OPD/PW-OPSD/FiRe/SCOPE/SelecTKD (all output-KL only,
confirmed via fetch), but three papers DO combine token-level/relational signals with
hidden-state losses in on-policy or distillation settings, and are direct prior art for
plan items B ("relational loss matching local neighbor identity") and E ("ARH-Bridge"):

### SRA — Span Representation Alignment for LLM Distillation, arXiv 2605.01205
Not token-level but span-level (LCS-aligned across tokenizers), with THREE loss terms:
(1) cosine hidden-state loss weighted by teacher-attention-pooled span importance
(`ℒ_HS^Span = Σ w_i^sp · ℒ_cos(...)`, weights = attention from last token to span,
aggregated across heads — i.e., an entropy/attention-based *weighting*, not neighbor
selection); (2) a **geometric regularizer that preserves pairwise DISTANCES between span
centers** (`ℒ_Geo = Σ (d(C_i^S,C_j^S) − d(C_i^T,C_j^T))²`); (3) span-level output KL.
Cross-architecture: teacher (Qwen 1.5-1.8B / 2.5-7B / Mistral-7B) → student (GPT-2
120M-1.5B, TinyLLaMA-1.1B, OPT-2.7B) via a learned projection matrix W, i.e. structurally
similar to OPRD-Bridge's "frozen PCA + learned student map" but using LCS-aligned spans
instead of PCA rank-r. Beats DSKD/ULD/MinED/MultilevelOT baselines by ~1.2-2.6 ROUGE-L
on Dolly/S-NI/VicunaEval/DialogSum/SelfInst. **Adversarial point: SRA's geometric
regularizer explicitly preserves pairwise DISTANCES — exactly the invariant ARH claims
does NOT survive width/depth null-calibration** ("local distances do not [survive], only
who-is-near-whom"). If ARH is right, SRA's ℒ_Geo term is optimizing a signal that's
mostly noise once teacher/student differ in width/depth, which is a genuine, citable,
falsifiable prediction the plan could test directly: does an ARH-flavored local-identity
loss beat SRA's ℒ_Geo distance-preserving loss on cross-architecture transfer, especially
after null-calibration controls? That is a sharper, more novel experiment than the
plan's current framing and should probably replace or sharpen item E. Also note SRA
never discusses PRH/ARH, CKA, Procrustes, or mutual-kNN at all — it's a physics-analogy
(MPDS/"particle dynamics") framing, not a representational-similarity framing — so OPRD's
authors and ARH's authors likely don't know about it either. Worth flagging as prior art
in both directions.

### PHF — Privileged Hidden Flow for On-Policy Self-Distillation, arXiv 2606.29340
Directly the closest thing to "OPRD but relational." Instead of pointwise MSE at matched
positions, PHF aligns **token-to-token transition directions** ("how the privileged
teacher's hidden states move along the rollout"), with the transport objective explicitly
built to be invariant to shared trajectory offsets and orthogonal transformations —
i.e., closer in spirit to a local/relative-geometry loss than to OPRD's pointwise MSE,
though it's about temporal (rollout-position) transitions, not spatial (neighbor-in-
representation-space) relations, so it's not the same operator as the plan's proposed
local-kNN loss. Tested on Qwen 1.7B/4B/8B, +1.5-2.2 Avg@12 over reproduced OPSD baselines
at matched 100-step budgets. **This directly weakens the plan's implicit claim that
pointwise MSE (OPRD's actual loss) is under-scrutinized as a design choice** — someone
has already shown a non-pointwise alternative beats reproduced OPSD/OPD-style baselines
in a comparable self-distillation setting. The plan should cite PHF as evidence FOR
"relational > pointwise" (supports the B/E direction) but must clarify PHF's relational
axis (temporal/transition) is orthogonal to ARH's (spatial/neighbor-identity), so the two
are complementary, not redundant — but this needs to be argued explicitly, not assumed.

### DEAR — "Finding the Evidence: Discovering Decision-Supporting Tokens," arXiv 2606.22830
Splits tokens into "decision" tokens (found via student entropy — same as TIP/EOPD) and
"evidence" tokens, found via **hidden-state cosine similarity to decision anchors**,
boosted by teacher-student divergence. This is explicitly relational (neighbor-in-
hidden-space) token SELECTION used to decide where to apply an output-KL loss — i.e., it
uses relational hidden-state structure as a selection signal but not as the loss target
itself (the actual distillation loss stays output-KL, per the fetched summary; the
"hidden-state relational" component is only the discovery mechanism for which tokens to
supervise). +2.5pp competition math, +5.7pp code gen over standard OPD. This narrows but
doesn't kill plan item B: DEAR proves neighbor-similarity-in-hidden-space is already
being used operationally in OPD, but as a token-selection heuristic, not as the training
objective. The plan's proposal (match local neighbor IDENTITY via soft top-k KL, as the
actual loss) is still a different mechanism, but the plan should cite DEAR to show
"hidden-state neighbor relations predict useful tokens" is not itself new — only using
that relation AS the loss is.

## 6. Older / adjacent prior art worth flagging

- **Relational Knowledge Distillation** (Park et al., CVPR 2019, arXiv 1904.05068) — the
  foundational vision-KD paper for pairwise-distance-preserving and angle-preserving
  relational losses. Not LLM/OPD-specific and not found via the OPD-specific searches,
  but it's the direct ancestor of SRA's ℒ_Geo term and conceptually of any "relational
  loss" framing in this plan. Worth a one-line citation for lineage, and worth noting: RKD
  matches *distances/angles*, which is exactly the ARH-refuted invariant, whereas the
  plan's proposed local-neighbor-identity loss is closer to what ARH says DOES survive.
  This is useful ammunition for positioning: "our loss differs from classical RKD/SRA
  precisely on the axis ARH predicts matters."

## 7. Direct answers to the assignment's two questions

**How crowded is 2025-2026 token-weighting/selection in distillation?** Very crowded —
13+ papers in OPD alone, two survey papers, two community-maintained lists, plus the
RLVR-side 80/20 literature. Position-weighting (A1/C) is the most crowded specific
sub-axis, with two papers (IW-OPD, PW-OPSD) essentially pre-empting the plan's framing.
Entropy-based weighting (EOPD/TIP/EGAD/SEAD) is also crowded. The plan should assume any
"we tried position weights" or "we tried entropy weights" result will need explicit
differentiation from IW-OPD/PW-OPSD/TIP or it will read as replication.

**Did any test hidden-state losses rather than output KL?** Mostly no (10+ papers stay in
output-KL space), but three did meaningfully touch hidden-state/relational signals: SRA
(hidden-state cosine + distance-preserving regularizer + span-level KL, cross-arch via
projection), PHF (temporal-transition relational hidden-state loss, self-distillation),
and DEAR (hidden-state cosine similarity used for token selection, output-KL loss). None
of the three implement an ARH-style local-neighbor-identity / soft-kNN-KL loss. That
specific operator looks like a genuine gap — but "hidden-state loss + token-level
signal" as a category is not a gap, and the plan's framing should be tightened
accordingly: the novel bet is specifically "local-neighbor-identity matching, not
distance-preserving (SRA) and not transition-direction (PHF) and not selection-only
(DEAR)," which is a narrower and more defensible claim than the plan currently states.

## Sources (all fetched/searched this session)

- TIP: https://arxiv.org/abs/2604.14084
- EOPD: https://arxiv.org/pdf/2603.07079
- EGAD: https://arxiv.org/pdf/2605.01732
- SCOPE: https://arxiv.org/abs/2604.10688
- Rock Tokens: https://arxiv.org/pdf/2605.09253
- FiRe-OPD: https://arxiv.org/abs/2606.02684
- ReNIO: https://arxiv.org/pdf/2606.23104
- IW-OPD / Position Bias of OPD: https://arxiv.org/abs/2606.22600
- PW-OPSD / "When Are Teacher Tokens Reliable?": https://arxiv.org/html/2605.21606
- Blockwise Policy-Drift Gating: (listed via awesome-on-policy-distillation, arXiv 2606.24084)
- DEAR / "Finding the Evidence": https://arxiv.org/abs/2606.22830
- SEAD: (listed via awesome-on-policy-distillation, arXiv 2606.28562)
- SelecTKD: https://arxiv.org/abs/2510.24021
- SRA: https://arxiv.org/html/2605.01205 (also https://arxiv.org/pdf/2605.01205)
- PHF: https://arxiv.org/abs/2606.29340
- 80/20 RLVR: https://arxiv.org/abs/2506.01939 (NeurIPS 2025)
- Survey (formula-driven): https://arxiv.org/html/2606.22793
- Survey (general OPD): https://arxiv.org/pdf/2604.00626
- awesome-on-policy-distillation: https://github.com/chrisliu298/awesome-on-policy-distillation
- AwesomeOPD (second list, not deeply explored): https://github.com/thinkwee/AwesomeOPD
- Relational Knowledge Distillation (classical prior art): https://arxiv.org/abs/1904.05068
- OPRD (target paper, for cross-reference): https://arxiv.org/abs/2606.06021

## Caveats / what I could NOT verify

- Several fetches returned WebFetch's paraphrase of PDF/HTML content, not primary-source
  full text (arXiv HTML rendering is lossy for equations in some cases). Loss formulas
  quoted for SRA are as extracted by the fetch tool, not independently re-derived —
  moderate confidence, not full-text-verified against the PDF directly.
- Did not verify SEAD (2606.28562) or Blockwise Policy-Drift Gating (2606.24084) beyond
  the one-line descriptions surfaced via the GitHub awesome-list fetch — no direct
  primary-source check. Marked as context/lower-confidence in the table.
- Did not find a paper doing BOTH layer-selectivity AND token-weighting for OPD/OPRD —
  absence-of-evidence, not evidence-of-absence; the search targeted token-weighting
  terms, not layer-selection terms, so this should not be read as a cleared gap for A2.
- Did not independently confirm IW-OPD's exact weighting formula (only got conceptual
  description, not the equation) — flagged in-line above.
