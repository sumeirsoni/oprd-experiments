# Adversarial sweep: neighborhood/ranking losses over representations

Angle: soft nearest-neighbor loss, NKD/neighborhood KD, local-structure-preserving GNN
distillation, soft-rank/differentiable sorting, collapse failure modes, temperature/window
sensitivity. Adversarial question: does matching only NEIGHBOR IDENTITY (not distances)
give enough constraint to transfer capability, or does it underdetermine the representation?

Bottom line up front: the single most load-bearing finding is that **the ARH paper itself
never proposes or tests local-neighborhood similarity as a training loss** — it is used
purely as a post-hoc measurement/diagnostic metric, with k fixed at 10 and essentially no
sensitivity analysis. Everything in plan components B and E (ARH-aligned relational loss,
ARH-Bridge) is an extrapolation from a measurement finding to a training prescription that
the source paper does not make and does not validate. That extrapolation is exactly where
prior art (RKD, LSP, CRD, dimensional-collapse literature) suggests real risk.

---

## 1. ARH paper itself (arxiv 2602.14486, Gröger, Wen, Brbić 2026) — primary source check

- URL: https://arxiv.org/abs/2602.14486 (HTML: https://arxiv.org/html/2602.14486v2)
- Verbatim abstract: "...the apparent convergence reported by global spectral measures
  largely disappears after calibration, while local neighborhood similarity, but not local
  distances, retains significant agreement across different modalities. Based on these
  findings, we propose the Aristotelian Representation Hypothesis: representations in
  neural networks are converging to shared local neighborhood relationships."
- Direct quote on distances vs identity: "Models increasingly agree on which points are
  neighbors, but do not agree on the pairwise distances, since CKA-RBF with a small
  bandwidth shows no alignment after calibration."
- **Metric is measurement-only.** Quote: "Neighborhood-based approaches, such as mutual
  k-Nearest Neighbors (mKNN)...capture local topological consistency rather than global
  alignment" — used exclusively to *evaluate* convergence across independently-trained
  models, never as a loss term to *train toward*. I could not find any passage proposing
  mKNN/CKNNA as an optimization objective.
- k is fixed at k=10 throughout their experiments; the paper states the theoretical mKNN
  baseline scales as O(k/n) but offers "minimal systematic analysis of k-sensitivity" in
  the main text (per my fetch of the HTML version — I was not able to verify a dedicated
  k-sweep ablation exists; flagging this as unverified rather than confirmed-absent).
- Explicit open question from their own limitations: "Why representations converge in
  local neighborhoods but not in global geometry remains the key open question." They do
  **not** claim local-neighborhood convergence implies functional equivalence or
  sufficiency for transfer — that inferential leap is the plan's, not theirs.
- Direction: **refutes-plan / context.** The plan's B/E components treat ARH as license to
  build a local-neighborhood *training* loss, but the paper that coined ARH only measured
  neighbor agreement between already-independently-trained, already-capable models. It
  says nothing about whether optimizing toward that signature during training would
  produce a capable model. This is the crux risk for the whole angle and should be stated
  explicitly in the plan, not assumed away.

## 2. Soft Nearest Neighbor Loss (Frosst, Papernot, Hinton, ICML 2019)

- URL: https://arxiv.org/abs/1902.01889 (PDF: http://proceedings.mlr.press/v97/frosst19a/frosst19a.pdf)
- This is the direct technical ancestor of the plan's "soft local softmax/KL over top-k
  teacher neighbors" idea — SNNL computes a softmax over pairwise distances within a batch
  at temperature T and is fully differentiable, i.e., exactly the "soft" family the plan
  wants (not hard top-k / not InfoNCE).
- Key finding from the paper: SNNL is normally used to *analyze* entanglement of class
  manifolds, and — counter to naive intuition — the authors found that *maximizing*
  entanglement in hidden (not final) layers, i.e., encouraging same-class points to *not*
  be trivially separated locally, improved generalization and calibration. This is a
  nontrivial, somewhat counter-intuitive result about what "matching local neighbor
  structure" does to a network, and cuts against a simple story where locking in local
  neighbor identity early/aggressively is unambiguously good — the original authors found
  the interesting regime was the opposite direction (loosening, not tightening, local
  structure) in some layers.
- No direct citation found of SNNL used as a **teacher→student distillation loss** in the
  original paper — it is a self-regularizer, not cross-model matching. I found a citing
  continual-learning paper ("A soft nearest-neighbor framework for continual
  semi-supervised learning," arxiv 2212.05102) that repurposes soft-NN structure as a
  teacher-student regularizer across time (old model as teacher for new model), which is
  the closest verified prior art to what the plan proposes, just in a different setting
  (class-incremental vision, not LLM hidden states).
- Direction: **prior-art.** Establishes that the exact loss family (softmax-over-distances,
  temperature-controlled) is well studied for measuring/encouraging entanglement, with a
  known non-monotonic relationship between "local structure sharpness" and downstream
  quality — i.e., temperature and which-layers-to-apply-it-to are first-order design
  choices with documented surprising effects, not free parameters to wave away.

## 3. Relational Knowledge Distillation — RKD (Park et al., CVPR 2019)

- URL: https://arxiv.org/abs/1904.05068
- RKD is the most directly relevant non-contrastive relational-KD prior art. It matches
  **pairwise Euclidean distances and triplet angles** between teacher and student feature
  spaces — i.e., it explicitly keeps distance/angle information, not just rank/identity.
  It outperformed naive pointwise (individual feature) KD baselines of its era.
- Adversarial point: RKD is direct prior art for "relational loss beats pointwise MSE" —
  which is good news for the plan's general thesis (B) that relational > pointwise. But
  RKD's success came from **keeping metric information** (distances, angles), which is
  precisely the information the plan's ARH-aligned loss (B) proposes to *discard* in favor
  of identity-only. This is a real tension: the one clear historical case study for
  "relational distillation beats pointwise" used distance-preserving relations, not
  identity-only/rank-only relations. The plan should not assume identity-only inherits
  RKD's win — that's an unverified, distinct hypothesis and deserves its own ablation
  against an RKD-style (distance-preserving) relational baseline, not just against
  pointwise MSE.
- Direction: **refutes-plan** (on the identity-only design choice specifically) /
  **prior-art** (on the general "relational > pointwise" claim).

## 4. Local Structure Preserving (LSP) distillation for GCNs (Yang et al., CVPR 2020)

- URL: https://arxiv.org/abs/2003.10477
- Distills a *distribution of pairwise distances to neighboring nodes* from teacher to
  student GCN ("local structure around each node is defined as the set of parameterized
  pairwise distances to its neighboring nodes in the latent feature space"), matched via
  distributional divergence.
- Same pattern as RKD: the one well-cited "neighborhood-based" distillation method in the
  graph domain preserves **distances to neighbors**, not merely neighbor identity/rank.
  This is a second independent data point suggesting that, in practice, when people have
  tried "local neighborhood" distillation losses and gotten it to work, they kept distance
  magnitude in the loss rather than throwing it away.
- Direction: **refutes-plan** (identity-only design choice) — worth naming explicitly as
  the "LSP-style" ablation arm the ARH-aligned loss should be tested against, since it's
  the closest thing to a fair distance-preserving control that isn't full-Gram PRH-style
  MSE.

## 5. Contrastive Representation Distillation — CRD (Tian et al., ICLR 2020)

- URL: https://arxiv.org/abs/1910.10699 (found via RepDistiller repo, HobbitLong/RepDistiller)
- Uses InfoNCE-style contrastive loss with explicit negatives to align teacher/student
  representations; widely reported as a strong, sometimes state-of-the-art, feature-KD
  baseline in vision distillation benchmarks.
- The plan explicitly excludes InfoNCE/CRD-style contrastive losses with negatives ("full
  Gram matching kept only as a PRH-flavored control... explicitly NOT InfoNCE/CRD"). That
  is a defensible scope cut, but it also means the plan is choosing not to compare against
  the strongest and most battle-tested relational-loss family in the KD literature. If the
  local-softmax loss (B) underperforms, reviewers/collaborators will immediately ask "did
  you try CRD" — worth at minimum a citation-level acknowledgment of why negatives are
  excluded (compute/memory cost of storing cross-batch negatives at LLM hidden-state scale
  is real, and negatives may not make sense with only 1 teacher + top-k-per-token-window
  neighbors) rather than leaving it implicit.
- Direction: **context** — not evidence for or against the plan's specific mechanism, but
  a known adjacent baseline whose absence should be justified in the writeup, not just
  assumed obvious.

## 6. Dimensional/representation collapse in similarity-softmax objectives

- "Understanding Dimensional Collapse in Contrastive Self-Supervised Learning" (Jing et
  al.) — https://arxiv.org/abs/2110.09348 — shows InfoNCE-family losses have collapse as a
  literal stationary point of the loss (linear embeddings give zero gradient at collapse),
  driven by (1) strong augmentation and (2) implicit low-rank bias of the optimizer.
- Relevant because the plan's proposed local-softmax/KL loss (B) is in the same mathematical
  family as InfoNCE/SNNL (softmax over similarities at a temperature). The specific failure
  mode to worry about for OPRD's setting: if the student's local neighborhoods collapse to
  low effective rank (a handful of directions dominate visible similarity), the loss can be
  driven toward zero by trivial, low-information solutions in exactly the way SNNL-style
  losses are known to reward that are not the intended "capability transfer."
  Effective-rank / stable-rank diagnostics (which the plan's component A3/D already
  proposes for OPD/OPRD's update geometry) should also be applied to the *representations*
  produced by any B/E relational loss, not just to the update matrices — collapse could
  hide in either place.
- Direction: **refutes-plan** in the sense of "identifies an unaddressed risk," not a
  direct empirical refutation of OPRD specifically. Treat as a required diagnostic
  (effective/stable rank of student hidden states under the relational loss, at multiple
  temperatures) rather than an assumed non-issue.

## 7. mutual-kNN / CKNNA sensitivity as a *metric*, independent of ARH paper

- Platonic Representation Hypothesis paper (Huh et al. 2024, arxiv 2405.07987) originally
  defines CKNNA. Secondary summaries (I could not extract the primary PDF text directly —
  flagging this as **unverified against primary source**, sourced from search-engine
  summaries only) describe CKNNA as monotonically increasing with k while roughly
  preserving the *ordering* of model-pair alignments — i.e., the metric's absolute value is
  k-dependent even if relative rankings are more stable. This is consistent with, and
  reinforces, the ARH paper's own admission that they fixed k=10 without extensive
  sensitivity sweeps.
- Practical implication for plan component B/E: if the evaluation metric itself is
  k-sensitive, a loss trained with a specific top-k/window size could look like it "wins"
  on the very metric whose k was tuned to match training, which would be circular. The plan
  should evaluate trained models across a *range* of k/window values distinct from the
  training-time k, not just report the training-matched setting.
- Direction: **refutes-plan** (methodological risk) / **context** (secondary-source only,
  should double check against Huh et al. primary text before citing in a paper).

## 8. Ordinal embedding theory — does neighbor-identity/ranking determine geometry?

- "Some theory for ordinal embedding" (Arias-Castro) — https://arxiv.org/abs/1501.02861.
  Abstract confirms the paper proves large-sample consistency and convergence rates for
  embedding points from triple/quadruple *ordinal* (comparison-only, no distance values)
  constraints, and bounds the number of comparisons needed for consistency.
- I was **not able to extract the specific sample-complexity numbers or the precise
  identifiability conditions** from the abstract/fetch (marking this unverified in detail).
  The qualitative takeaway that is well-established in the ordinal-embedding/NCA/CKNNA
  literature broadly: rank-only constraints CAN in principle pin down a Euclidean geometry
  up to similarity transform (rotation/scale/translation), but *only* in the large-sample,
  dense-comparison regime — sparse or small-k neighbor sets are a classical
  underdetermined regime (many non-isometric embeddings satisfy the same small set of
  ordinal constraints).
- Direct relevance to the plan: component B proposes "top-k teacher neighbors within a
  token window" (finite k, finite window, ~2000 tokens). This is exactly the sparse-
  comparison regime where ordinal-embedding theory says identity-only constraints are
  weakest/most underdetermined. This is a legitimate theoretical grounding for the
  adversarial question itself — the plan should treat k and window size as a swept
  hyperparameter with an explicit hypothesis ("larger k / wider window → better-determined
  target → better transfer, up to some compute-cost ceiling"), rather than picking one k
  and one window and calling it "the ARH-aligned loss."
- Direction: **context** (theoretical framing, not a direct empirical result about LLMs or
  distillation) but load-bearing for justifying the k/window sensitivity sweep the plan
  should run.

## 9. Representational similarity ≠ functional equivalence (general caution)

- Klabunde et al., "Similarity of Neural Network Models: A Survey of Functional and
  Representational Measures," ACM Computing Surveys 2025 — search-engine summary indicates
  the survey's central caution is that high representational similarity does not imply
  functional equivalence, and that two models can match on representational measures while
  differing in downstream behavior (**unverified against primary text — flagging**).
- Braun et al. 2025 reportedly give an analytical deep-linear-network treatment showing
  functional and representational similarity can be decoupled even in controlled settings
  (**unverified against primary text — flagging, could not locate exact arXiv ID in this
  sweep**).
- Direction: **context / refutes-plan-motivation**. This is the general-purpose version of
  finding #1: representational-alignment metrics (of any flavor — global CKA or local
  mutual-kNN) are measurement tools whose relationship to downstream capability transfer is
  itself an open empirical question, not settled theory. It reinforces that the plan's
  central bet (a loss that matches ARH's *measured* signature will *cause* capability
  transfer) needs to be treated as the primary hypothesis under test, not as a corollary of
  ARH being true.

## 10. Differentiable sorting / soft-rank as an alternative loss mechanism

- NeuralSort / SoftSort (Grover et al.), Fast Differentiable Sorting and Ranking (Blondel
  et al., arxiv 2002.08871), Cuturi et al. optimal-transport ranks (arxiv 1905.11885),
  PiRank (Swezey et al.) — established toolkit for making rank/order operations
  differentiable, used in learning-to-rank and (per one search hit) as a distillation loss
  from a ranking teacher to a pre-ranking student model.
- Relevant as an **alternative implementation** to softmax/KL-over-neighbors for the same
  underlying goal (matching who's-near-whom without matching distances): NeuralSort-style
  losses directly optimize a relaxed permutation matrix rather than a temperature-scaled
  softmax distribution. This is worth a footnote in the plan's method section as a design
  alternative with different temperature/bias tradeoffs (differentiable sorting relaxations
  have their own temperature-like relaxation parameter that trades bias for gradient
  variance, so this doesn't sidestep temperature sensitivity, it just moves it).
- Direction: **context** — not evidence for/against, but redundant-design-space awareness:
  if B's softmax-KL variant underperforms, a soft-rank variant is a natural next thing a
  reviewer will suggest, so it's cheap to at least cite it as considered/rejected with a
  reason rather than have it look like an unknown unknown.

---

## Net adversarial assessment for the parent plan

1. **Biggest risk, understated in the draft plan:** ARH (2602.14486) is a *measurement*
   result about independently-trained, already-capable models converging in local
   neighborhood structure. It is not a demonstrated *training* result. Nothing in the ARH
   paper, nor in the broader literature surfaced here, directly shows that optimizing a
   student toward a teacher's local-neighbor-identity signature during training is
   sufficient to transfer capability. This should be flagged as the paper's central novel
   empirical claim to establish, not a validated premise to build on.

2. **The plan's identity-only design choice is not supported by the closest working prior
   art.** Both RKD (vision KD) and LSP (GNN KD) — the two most-cited "relational, not
   pointwise" distillation methods that are genuinely analogous to what's proposed — kept
   distance/magnitude information rather than discarding it for pure rank/identity. If B's
   identity-only loss underperforms a distance-preserving relational loss (RKD/LSP-style),
   that would be a meaningful, citable negative result, not a corner case — recommend
   adding an RKD/LSP-style distance-preserving relational baseline as a required ablation
   arm, distinct from both pointwise MSE and full-Gram PRH-MSE.

3. **k/window size is a first-order hyperparameter with theoretical reasons (ordinal
   embedding underdetermination) and empirical reasons (CKNNA's own k-sensitivity, ARH's
   own unswept k=10) to expect large, possibly non-monotonic effects.** The current plan
   language ("top-k teacher neighbors within a token window") reads like a single fixed
   design choice; recommend explicitly framing k and window width as a swept axis with a
   stated hypothesis, and evaluating downstream capability (not just the alignment metric)
   across that sweep — otherwise the eventual result is fragile to reviewer-requested
   sensitivity checks.

4. **Collapse is a live risk for the softmax/KL-over-neighbors formulation specifically**
   because it's in the same loss family as SNNL/InfoNCE, both of which have documented
   temperature-dependent and rank-collapse pathologies. The plan already proposes stable-
   rank/subspace-locking diagnostics for the *optimizer geometry* (A3/D) — recommend
   explicitly extending those diagnostics to the *student representations produced under
   the relational loss*, since collapse there would look like "loss went down, alignment
   metric went up" while capability doesn't transfer — a failure mode that would be
   invisible without a rank check.

5. **CRD exclusion needs a stated reason, not silent omission** — it's the strongest
   adjacent contrastive baseline in the KD literature and its absence will be the first
   thing a critical reader asks about.

## Confidence / verification caveats

- Findings #1 (ARH primary quotes) and #4 (LSP) were verified against primary-source text
  (HTML fetch / abstract) directly.
- Findings #2 (SNNL), #3 (RKD) were verified against abstracts/search summaries with some
  primary-source confirmation (RKD PDF fetch partially succeeded); mechanism claims (what
  RKD matches) are corroborated by multiple independent search results so treated as
  reliable, but I did not get a clean full-text read of RKD's failure-mode discussion.
- Findings #7 (CKNNA k-sensitivity secondary description), #9 (Klabunde/Braun) are
  **search-engine-summary only** — I was not able to fetch and directly quote primary text
  due to PDF binary-extraction failures. These are flagged in-line as unverified and should
  be independently confirmed before being cited in any written report.
- Finding #8 (ordinal embedding theory) — abstract-level confirmation only; specific sample-
  complexity claims not verified.
