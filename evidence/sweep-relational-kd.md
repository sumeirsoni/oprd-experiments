# Adversarial sweep: relational vs. pointwise KD objectives (prior art + replication failures)

Scope: RKD, CRD, CoDIR, similarity-preserving KD (SP), PKT, DIST, MiniLM/MiniLMv2,
TinyBERT hidden-MSE lineage, plus "does distillation really work" replication angle,
plus 2025-2026 LLM-specific hidden-state / CKA distillation work. Goal: find
evidence AGAINST the draft plan's implicit assumption that a relational objective
(local neighbor identity, no negatives) will beat pointwise MSE, and find prior art
that makes proposed experiments (esp. B, E / ARH-Bridge) redundant.

---

## 1. Headline adversarial finding: RKD (the closest classical analog to the plan's
   proposed local-relational loss) *underperforms vanilla KD* in the standard
   benchmark, averaged over 5 seeds

Source: Tian, Krishnan, Isola, "Contrastive Representation Distillation" (CRD),
ICLR 2020. arXiv: https://arxiv.org/abs/1910.10699 (full text via
https://ar5iv.labs.arxiv.org/html/1910.10699). Benchmark code/table also at
https://github.com/HobbitLong/RepDistiller (CIFAR-100, "Average over 5 runs").

Table 1 numbers (student accuracy, %, same-architecture teacher/student pairs):

| Pair | KD (vanilla) | RKD (relational, distance+angle) | PKT (probabilistic relational) | CRD (contrastive) |
|---|---|---|---|---|
| WRN-40-2 → WRN-16-2 | 74.92 | 73.35 (**-1.57**) | 74.54 | 75.48 |
| resnet110 → resnet20 | 70.67 | 69.25 (**-1.42**) | 70.25 | 71.46 |
| resnet32x4 → resnet8x4 | 73.33 | 71.90 (**-1.43**) | 73.64 | 75.51 |

Per the CRD paper itself: CRD is "the only method to always outperform KD"; other
feature/relational methods (including RKD, and per RepDistiller's own README,
Attention Transfer) are inconsistent and sometimes *worse* than plain KD
(e.g. AT: 59.40% vs KD 67.37% on vgg13→MobileNetV2, a cross-architecture pair).

**Why this matters for the plan**: RKD's pairwise distance/angle loss is the
closest published analog to a "relational, no-negatives" objective. It is the
purest test of "does preserving *relational structure* (not identity, not
contrastive) beat pointwise matching" — and in the most-cited apples-to-apples
benchmark, it loses to vanilla KD on 3/3 sampled pairs, averaged over 5 runs (not
cherry-picked single runs). The one relational-family method that *reliably* wins
(CRD) is explicitly contrastive with negatives — which the plan's item (B)
explicitly rules out ("NOT InfoNCE/CRD contrastive, no negatives"). This is a
direct, load-bearing counter-example the plan needs to address: the version of
"relational" it wants to keep (no negatives) is the version with the weaker track
record; the version with the strong track record is the version it's rejecting.
CAVEAT: this is vision/CNN, small-scale, classification-only, not LLM/reasoning —
flagging as *prior art / cautionary base rate*, not a direct refutation of the
LLM setting.

---

## 2. PKT (Probabilistic Knowledge Transfer) is close prior art for exactly the
   mechanism proposed in plan item (B) — soft KL-over-neighbor-similarity — and
   it roughly ties vanilla KD, doesn't clearly beat it

Source: Passalis & Tefas, "Learning Deep Representations with Probabilistic
Knowledge Transfer," ECCV 2018. https://arxiv.org/abs/1803.10837, code
https://github.com/passalis/probabilistic_kt

Mechanism: convert pairwise (cosine) similarities within a batch into a
conditional probability distribution per sample (softmax-like normalization,
same spirit as SNE/t-SNE), then match teacher's and student's distributions via
KL divergence. Explicitly designed to work when "the dimensionality of layers
used for transfer does not match" — i.e., it already handles the exact
width-mismatch problem OPRD-Bridge uses PCA+linear-map to solve, without any PCA
projector, via a soft-neighbor-KL loss.

This is materially close to the plan's proposed "(B) soft local softmax/KL over
top-k teacher neighbors" — the main structural differences are (i) PKT computes
similarity over the whole minibatch (heterogeneous samples), not a local token
window within one rollout, and (ii) PKT is not restricted to top-k neighbors (it
is a dense/full softmax, not sparse top-k). The plan's "local" (token-window,
top-k neighbors) framing IS a genuine point of novelty vs. PKT, but the "soft-KL
over similarity-derived distribution, no negatives, handles dimension mismatch"
core idea is not new — it is 2018 vision literature. The plan should cite PKT and
explain precisely why *local top-k within a rollout* is expected to behave
differently than *global dense softmax over a batch of unrelated images* (the
ARH "local neighbor identity survives, global doesn't" framing is actually a
good argument for why top-k local should behave differently — but the plan
should make that argument explicit rather than treat the loss family as novel).

Empirically PKT is close to vanilla KD in the table above (74.54 vs 74.92, 70.25
vs 70.67, 73.64 vs 73.33 — beats KD once, loses twice, all within ~1pt) — i.e.
even the closest prior-art analog to the proposed loss does not show a
convincing win over pointwise KD in the standard benchmark.

---

## 3. Similarity-Preserving KD (Tung & Mori, ICCV 2019) — another "relational,
   no negatives" precursor, pairwise-similarity-matrix matching, not obviously
   dominant either

Source: https://arxiv.org/abs/1907.09682, CVF:
https://openaccess.thecvf.com/content_ICCV_2019/papers/Tung_Similarity-Preserving_Knowledge_Distillation_ICCV_2019_paper.pdf

"Guides training such that input pairs that produce similar (dissimilar)
activations in the teacher produce similar (dissimilar) activations in the
student... the student is not required to mimic the representation space of the
teacher, but rather to preserve pairwise similarities in its own representation
space." This is the general family (SP, RKD, PKT, DIST) the plan's (B)/(E)
belongs to — "preserve relational structure, don't match absolute values." It
predates and is architecturally similar to the plan's framing; not cited in the
plan draft as far as I can tell. Not independently benchmarked here beyond its
inclusion (as "SP") in the RepDistiller comparison table, where per the
RepDistiller README, several relational/structural methods (SP, VID, CC, RKD)
cluster near or below CRD/vanilla-KD, with none reliably dominant except CRD.

---

## 4. DIST ("Knowledge Distillation from a Stronger Teacher," NeurIPS 2022) —
   relational (Pearson-correlation) objective built *specifically* for the
   large-teacher-student-gap regime, i.e. structurally the closest match to the
   OPD "stronger teacher on student's own rollouts" setting

Source: Huang et al., NeurIPS 2022. https://proceedings.neurips.cc/paper_files/paper/2022/file/da669dfd3c36c93905a17ddba01eef06-Paper-Conference.pdf,
OpenReview https://openreview.net/forum?id=157Usp_kbi

Key claim: exact-match losses (KL on outputs) "disturb training" when teacher and
student are far apart in capability — DIST instead preserves *relations* (inter-
class and inter-instance Pearson correlation) rather than exact values, "relaxing"
the matching requirement. This is functionally the output-space (logit-level)
analog of the OPRD critique of OPD (reverse-KL / output matching struggles under
large teacher-student capability gaps and distribution mismatch) — DIST is prior
art specifically motivating "loosen exact/pointwise matching → use relational/
correlation matching" for exactly the "stronger teacher" regime the plan cares
about, but at the logit level, not hidden-state level, and not on LLM reasoning
rollouts. Should be cited; also suggests a cheap ablation (Pearson-correlation
loss on hidden states, à la DIST but at the representation level) as a much
simpler baseline than a full local-kNN-softmax-KL loss before spending compute
on (B).

---

## 5. MiniLM / MiniLMv2 — YES, this is prior art for "relational beats pointwise
   specifically in the cross-architecture / width-mismatch case," predating and
   arguably subsuming the motivation for OPRD-Bridge / ARH-Bridge

Sources: Wang et al., "MiniLM: Deep Self-Attention Distillation for
Task-Agnostic Compression of Pretrained Transformers," NeurIPS 2020,
https://arxiv.org/abs/2002.10957; Wang et al., "MiniLMv2: Multi-Head
Self-Attention Relation Distillation for Compressing Pretrained Transformers,"
Findings of ACL 2021, https://arxiv.org/abs/2012.15828,
https://aclanthology.org/2021.findings-acl.188.pdf

MiniLMv2 explicitly generalizes MiniLM to a *relation-only* distillation
objective: it distills scaled dot-product "relations" (Q-Q, K-K, V-V pairwise
similarity matrices within self-attention) from teacher to student, and its
headline architectural benefit is: "there is no restriction in terms of the
number of student's attention heads, while most previous work has to guarantee
the same head number between teacher and student." That is precisely the
width/head-count mismatch problem OPRD-Bridge solves via PCA projection — MiniLM
solved an analogous mismatch via a relational (self-similarity) loss instead of
any projector, back in 2020/2021. It's evidence *for* the plan's ARH-Bridge
instinct (relation-only losses can dissolve dimension-matching problems that a
projector otherwise has to solve) but it's also prior art that undercuts the
novelty claim: "relational objectives handle cross-architecture width mismatch
without a learned projector" is not a new idea — it shipped in production-scale
LM distillation five years ago. The plan should position ARH-Bridge explicitly
relative to MiniLMv2 (what's different: token-window local kNN over residual-
stream hidden states across full depth vs. self-attention QKV relation matrices
per layer; different notion of "relation") rather than motivate it purely from
ARH/PRH theory as if it were unprecedented. Caveat: MiniLM(v2) is same *family*
(Transformer-to-Transformer, BERT-style encoders), not tested across genuinely
different architectures (different depth AND width AND positional scheme AND
model family simultaneously) — so it doesn't fully settle the cross-architecture
case the plan cares about (reasoning-LLM decoder-only, potentially different
families).

---

## 6. TinyBERT — the width/depth-mismatch problem is *already solved with plain
   pointwise MSE* plus a learned linear map, no relational loss needed, in the
   BERT-lineage setting

Source: Jiao et al., "TinyBERT: Distilling BERT for Natural Language
Understanding," Findings of EMNLP 2020, https://aclanthology.org/2020.findings-emnlp.372.pdf

TinyBERT's hidden-state loss is literally MSE(student_hidden · W_map,
teacher_hidden), where W_map is a learned linear projection handling the
dimensionality mismatch — i.e., "learned student linear map + MSE," structurally
identical in spirit to what OPRD-Bridge does (frozen teacher PCA + learned
student linear map + MSE in rank-r space), just without the PCA step. TinyBERT is
widely used, reasonably strong empirically, and uses no relational component at
all. This is a direct, well-known counter-example to "you need relational
structure to bridge architecture mismatch" — plain pointwise MSE + a linear
adapter is an established, working baseline for cross-width distillation. Any
ARH-Bridge experiment needs to beat *this* baseline (MSE + learned linear map,
no PCA, no relational loss), not just OPRD-Bridge's PCA+MSE variant, to make a
convincing case. Caveat: TinyBERT student/teacher are same architecture family
(BERT-to-BERT-shaped), same depth-mapping-lite scheme — doesn't test extreme
cross-family mismatch, and it's encoder MLM objective, not on-policy reasoning
rollouts.

---

## 7. CoDIR — explicit theoretical critique of L2/MSE hidden-state matching that
   the plan should probably cite (motivates why pointwise MSE could be
   insufficient), but its fix is contrastive (negatives), which the plan
   excludes

Source: Sun, Yu, Song, Liu, Yu, Zhou, "Contrastive Distillation on Intermediate
Representations for Language Model Compression," EMNLP 2020,
https://arxiv.org/abs/2009.14167, https://aclanthology.org/2020.emnlp-main.36/

Direct quote (per search extraction): "L2 [MSE] objective by design assumes that
all the dimensions of hidden representations are independent, failing to capture
important structural knowledge in the intermediate layers." This is a clean,
citable articulation of exactly the pointwise-MSE weakness the plan is trying to
motivate for OPRD — but CoDIR's actual mechanism (pull together positive
teacher/student pairs, push apart negatives, i.e., InfoNCE-style contrastive) is
precisely the family the plan explicitly rules out for item (B). Same pattern as
CRD/RKD above: the literature's best-performing "beyond MSE" hidden-state losses
for LM compression are consistently the contrastive ones, and the plan is
deliberately avoiding that family. This is a real strategic risk the plan should
name up front, not just an incidental detail — the adversarial reading is "every
time someone tried a non-contrastive relational alternative to MSE (RKD, PKT,
SP), it roughly tied or lost to pointwise baselines; every time the relational
alternative reliably won (CRD, CoDIR, MiniLMv2-self-attention), it either used
negatives/contrastive structure or exploited an architecture-specific relation
(attention patterns) rather than a generic embedding-neighbor KL." The plan's
proposed local-neighbor-softmax-KL-no-negatives loss (B) sits in the
under-performing bucket by the closest available analogy (PKT), not the
over-performing bucket.

---

## 8. Two 2025-2026 papers doing almost exactly "hidden-state relational loss
   for LLMs," using CKA (a *global* spectral measure — the kind ARH says loses
   cross-model scaling signal) — and reporting it *works*, including for
   RL-post-trained reasoning models

### 8a. CKA-QAD ("Beyond Output Matching: Preserving Internal Geometry in NVFP4
LLM Distillation," 2026). arXiv: https://arxiv.org/abs/2606.05682

- Setting: quantization-aware distillation (student = same architecture as
  teacher, just NVFP4 vs BF16 precision) — NOT cross-architecture, but IS
  applied to reasoning-RL-trained models (Nemotron 3 Nano, Qwen3-4B-Thinking-2507).
- Method: KL(logits) + CKA-based regularizer aligning layerwise Gram matrices
  ("preserves internal representational geometry... aligning layerwise Gram
  matrices through CKA").
- Finding: "KL-only QAD can reduce layerwise representational similarity
  relative to the BF16 teacher, with especially severe drift in RL-post-trained
  models," and CKA-QAD "substantially improves representational alignment and
  improves downstream reasoning and coding accuracy with modest training
  overhead."
- **Adversarial tension with the plan's ARH framing**: this is a full-Gram,
  global-spectral (CKA) relational loss — exactly the "PRH-flavored control"
  category the plan says should be weak/uninformative per ARH's null-calibration
  argument — yet it is reported to produce real downstream reasoning-accuracy
  gains on RL-trained models. Two ways to reconcile: (i) ARH's claim is about
  CKA's value as a *diagnostic/comparison* metric across models of different
  scale (whether CKA similarity predicts scaling/quality), not about whether a
  CKA-shaped *training loss* transfers useful gradient signal — these are
  different claims and the plan should not conflate them; (ii) alternatively,
  this could be genuine counter-evidence that global spectral alignment still
  carries a useful training signal even in a regime (RL-post-trained reasoning
  models) ARH considers most likely to show representational drift. Either way,
  the plan should explicitly address this paper before asserting the "full-Gram
  matching kept only as a PRH-flavored control [i.e., expected weak]" framing —
  it is not obviously weak in the closest published real-world test on
  reasoning models. Caveat: same-architecture (quantization only), not
  cross-arch; not on-policy-rollout distillation; small set of 2 models.

### 8b. "Improving Language Model Distillation through Hidden State Matching,"
ICLR 2025. OpenReview: https://openreview.net/forum?id=IcVSKhVpKu (PDF blocked
by bot-check in this fetch; findings below via search-engine synthesis + ICLR
listing, not independently read in full — flagged as lower-confidence, verify
before citing quantitatively)

- Method: CKA to match hidden states of **different dimensionality** across
  encoder-only (BERT) and encoder-decoder (BART/mBART/T5) architectures —
  explicitly framed as solving the problem that "cosine loss typically
  restricts the dimensionality of the student to the teacher's, severely
  limiting the compression ratio."
- This is directly analogous to OPRD-Bridge's width-mismatch problem, solved via
  global CKA rather than PCA+MSE or local-kNN. If this result holds up, it is
  additional prior art that a *global* spectral method already handles
  cross-architecture width mismatch reasonably well for LM distillation,
  further raising the bar for what a local-neighbor loss (ARH-Bridge) needs to
  demonstrate beyond it. **Flagging direction as "context," not fully verified**
  — I could not get past OpenReview's bot-check to read the actual results
  table; recommend the plan authors pull the PDF directly
  (https://openreview.net/attachment?id=IcVSKhVpKu&name=pdf) before relying on
  this.

---

## 9. Replication / "fancy losses don't beat well-tuned vanilla" angle

### 9a. Stanton, Izmailov, Kirichenko, Alemi, Wilson, "Does Knowledge
Distillation Really Work?", NeurIPS 2021. https://arxiv.org/abs/2106.05945

Not specifically about relational-vs-pointwise, but the load-bearing general
finding: "there often remains a surprisingly large discrepancy between the
predictive distributions of the teacher and student, even when the student has
the capacity to perfectly match the teacher... difficulties in optimization are
a key reason." Relevant adversarial framing for the whole plan: before
attributing OPD's weaknesses to *where* supervision lands (output space vs.
hidden space) or *what structure* it preserves (pointwise vs. relational), this
paper is a reminder that optimization difficulty/fidelity gaps in distillation
are pervasive and not fully explained by objective choice — any A/B result
between OPD/OPRD/relational variants should control for optimization
confounds (LR, steps, init) before attributing gaps to representational theory.
Direction: context (general caution, not a direct refutation of any specific
plan item).

### 9b. VanillaKD, "Revisit the Power of Vanilla Knowledge Distillation: from
Small Scale to Large Scale," NeurIPS 2023. https://arxiv.org/abs/2305.15781

Finding (per search synthesis): with proper training recipes/hyperparameters at
scale, plain vanilla KD (logit matching) is "competitive with" or beats many
more elaborate feature/relational distillation methods that were previously
claimed to be superior on small-scale ImageNet benchmarks — i.e., much of the
literature's claimed superiority of fancy losses over vanilla KD may be a
training-recipe/small-scale artifact rather than a fundamental property of the
loss. This is a direct, citable version of the "well-tuned vanilla KD beats
fancy losses" adversarial claim the task asked me to hunt for. Strongly
supports being skeptical of any small-scale (~1.5B student) result in the plan
that shows a relational loss beating pointwise MSE by a modest margin — the
effect could evaporate with a better-tuned MSE baseline or at larger scale.
Recommend the plan explicitly budget for a "well-tuned vanilla OPRD (MSE)"
sweep (LR, layer subset, token window, weighting) before concluding a relational
variant wins.

### 9c. Object detection KD literature note (weaker, context only)

General survey-level observation found in search (not independently verified
against a specific paper): "knowledge distillation has made great success in
image classification, but only attains trivial improvement in object
detection" — i.e., there is a known task-dependence to whether feature/
relational KD helps at all, which is a mild caution against assuming a result on
math-reasoning (single task family) generalizes. Direction: context, low
confidence, not a specific verified citation — flagging honestly rather than
inventing a source.

---

## 10. Bottom line / synthesis for the researchers

1. **The core empirical base rate is unfavorable to the plan's specific loss
   design.** In the most standard, most-cited relational-vs-pointwise benchmark
   (CRD/RepDistiller, CIFAR-100, 5-seed averages), the *no-negatives* relational
   methods (RKD, PKT, SP) roughly tie or lose to vanilla KD; the *only* method
   that reliably wins uses negatives (CRD) — which the plan explicitly excludes.
   This should be treated as a strong prior against "local neighbor identity, no
   negatives" (item B) winning by a large margin, not proof it will lose in the
   LLM/on-policy setting, but a reason to run a cheap, well-tuned MSE baseline
   sweep before investing heavily in the relational variant.

2. **MiniLMv2 and TinyBERT are real, load-bearing prior art the plan should cite
   and differentiate from**, not novel-territory claims. MiniLMv2 already shows
   a *relational* (self-attention) loss dissolving a head-count/width mismatch
   without any projector — partial support for ARH-Bridge's instinct, but also
   evidence the core idea predates ARH/PRH theorizing by years. TinyBERT shows
   the width-mismatch problem is *already solvable* with plain pointwise MSE + a
   learned linear map (no PCA, no relational loss) — this is the baseline
   ARH-Bridge and OPRD-Bridge both need to clear, and it's cheaper than either.

3. **PKT (2018) is close-enough prior art to the specific mechanism in item (B)**
   (soft KL over similarity-derived distribution, dimension-agnostic) that the
   plan should cite it and articulate precisely why *local top-k within a token
   window* should behave differently than PKT's *global dense batch softmax* —
   the ARH "local survives, global doesn't" argument is actually a good reason,
   but it needs to be stated, because the empirical PKT result (parity with KD,
   not a clear win) undercuts assuming novelty = advantage.

4. **The CKA-QAD paper (2606.05682) is a genuine complication for the ARH
   framing**, not just supportive prior art: a *global* spectral (CKA/Gram)
   relational loss reportedly improves reasoning accuracy in RL-post-trained
   LLMs, the closest published real test of "does representational-geometry
   preservation help distillation of reasoning models" that I found. The plan's
   framing of full-Gram/PRH-style matching as an expected-to-be-weak control
   needs to engage with this result directly, or clarify that ARH's claim is
   about CKA as a cross-model-scale diagnostic, not CKA as a training loss —
   these are not the same claim and conflating them is a risk.

5. **VanillaKD (2305.15781) is a strong general-purpose adversarial citation**:
   much of the "fancy loss beats vanilla" literature may be a small-scale/
   under-tuned-baseline artifact. Any experiment in the plan claiming a
   relational win over pointwise MSE needs a genuinely well-tuned MSE baseline
   (layer subset, token window, LR) or the result is not trustworthy at face
   value.

## Sources (all fetched/verified via WebSearch or WebFetch this session)

- CRD: https://arxiv.org/abs/1910.10699 / https://ar5iv.labs.arxiv.org/html/1910.10699
- RepDistiller benchmark: https://github.com/HobbitLong/RepDistiller
- RKD (original): https://arxiv.org/pdf/1904.05068 (CVPR 2019) — referenced via CRD's benchmarking, not independently re-read in full this session
- PKT: https://arxiv.org/abs/1803.10837, https://github.com/passalis/probabilistic_kt
- Similarity-Preserving KD: https://arxiv.org/abs/1907.09682
- DIST: https://proceedings.neurips.cc/paper_files/paper/2022/file/da669dfd3c36c93905a17ddba01eef06-Paper-Conference.pdf, https://openreview.net/forum?id=157Usp_kbi
- MiniLM: https://arxiv.org/abs/2002.10957
- MiniLMv2: https://arxiv.org/abs/2012.15828, https://aclanthology.org/2021.findings-acl.188.pdf
- TinyBERT: https://aclanthology.org/2020.findings-emnlp.372.pdf
- CoDIR: https://arxiv.org/abs/2009.14167, https://aclanthology.org/2020.emnlp-main.36/
- CKA-QAD: https://arxiv.org/abs/2606.05682
- Hidden State Matching (CKA, ICLR 2025): https://openreview.net/forum?id=IcVSKhVpKu (PDF not independently read — bot-check blocked; low-confidence, flagged)
- Does KD Really Work: https://arxiv.org/abs/2106.05945
- VanillaKD: https://arxiv.org/abs/2305.15781
- Position bias OPD (context, already known to researchers per prompt): https://arxiv.org/abs/2606.22600
