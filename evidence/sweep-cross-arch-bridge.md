# Adversarial sweep: cross-architecture / cross-tokenizer distillation bridges
Angle: learned projectors, PCA/whitening, CKA losses, optimal transport, universal logit distillation,
MiniLM relation transfer, vocab-mismatch distillation, 2024-2026. Question: does prior art already cover
neighborhood/relational bridges, and is a mutual-kNN bridge actually novel vs OPRD-Bridge (PCA+MSE)?

Date of sweep: 2026-07-10. All sources found via WebSearch/WebFetch; several fetches returned
binary-PDF-decode failures (noted below) and were substituted with WebSearch-derived summaries or
abstract-page fetches — treat those as lower-confidence than the ones with direct quotes.

---

## 1. Bridge design families that ALREADY exist (competing with OPRD-Bridge's PCA+MSE)

### 1a. PCA / linear-projector + pointwise loss (same family as OPRD-Bridge)
- OPRD itself (arXiv 2606.06021) — confirmed via direct fetch of arxiv.org/html/2606.06021:
  - Teacher side: frozen PCA projector `P_T(l) = TopPrincipalDirections(h_T(l) - mu_T(l), r)`.
  - Student side: learned linear map `P_S(l)` trained to minimize MSE to the (frozen) teacher projection.
  - Motivation quoted: "strong neural networks, regardless of architecture, are converging toward a
    shared statistical model of reality in their representation spaces" (citing Huh et al. 2024, PRH).
  - Reported cosine similarity between projected student/teacher reps reaches ~95% at r=8 across layer
    pairs (their own bridge-quality diagnostic, not an external alignment benchmark).
  - Cross-arch results: Qwen3-4B -> Qwen3-1.7B-Base, and cross-tokenizer Phi-4 -> Qwen3-1.7B (Table 4,
    §4.2.2). The related-work section (§6) lists categories ("Feature/Intermediate-Representation
    Distillation", "Hint Learning, Auxiliary Losses, and Distribution Matching") but — per the fetch —
    gives "limited detail on specific competing cross-architecture methods or their limitations." This
    is itself a finding: OPRD's own lit review does not appear to seriously engage the relational/CKA/OT
    bridge literature below, which is a gap the draft plan could legitimately call out as motivation, but
    also means the plan needs to do this diligence itself rather than trust OPRD's framing of "no
    alternatives."

### 1b. CKA as a training loss (not just a diagnostic), already handles width mismatch
- "Feature Structure Distillation with Centered Kernel Alignment in BERT Transferring" (arXiv 2204.08922,
  2022). Uses CKA-based Gram-matrix distillation for BERT compression — i.e., full pairwise-similarity
  ("Gram") matching between teacher and student, already applied to transformer LMs, already handling
  dimension mismatch by construction (CKA is dimension-agnostic). This is a direct, several-years-old
  precedent for the plan's "full-Gram matching kept only as a PRH-flavored control" — it is not a novel
  control condition, it's an existing competing method with its own literature.
- "Rethinking Centered Kernel Alignment in Knowledge Distillation" (IJCAI 2024 / arXiv 2401.11824).
  Explicitly reframes CKA loss as distilling the *centered similarity matrix* rather than raw features,
  citing training-instability issues with naive CKA losses and proposing fixes. Relevant because it
  already documents failure modes of global-Gram objectives in KD — this is exactly the kind of empirical
  problem the ARH story (global spectral measures are noisy/unstable) would predict, so it's usable as
  *supporting* context for the plan's motivation, not just a strawman control.
- Search snippet (via WebSearch, not independently fetched) on width-mismatch KD: "Traditional cosine
  loss restricts student dimensionality to the teacher's... but CKA can match hidden states of different
  dimensionality, allowing for smaller students and higher compression ratios" — general claim from
  survey-style pages (labelyourdata.com, OpenReview thread "Improving Language Model Distillation through
  Hidden State Matching"), not verified against a single primary paper; flagged lower-confidence.

### 1c. Optimal-transport bridges (a whole separate design family, at logit space)
- Universal Logit Distillation (ULD) — Boizard et al., "Towards Cross-Tokenizer Distillation: the
  Universal Logit Distillation Loss for LLMs" (arXiv 2402.12030). OT-based loss operating on logit
  distributions, explicitly built to cross architecture/tokenizer boundaries without shared vocabulary.
- "Multi-Level Optimal Transport for Universal Cross-Tokenizer Knowledge Distillation on Language
  Models" (arXiv 2412.14528, AAAI 2025 oral). Confirmed via WebFetch of the abstract page: operates on
  **logit distributions, not hidden states** — token-level OT (Sinkhorn) + sequence-level OT, explicitly
  claims to beat MSE/CKA-based hidden-state alignment approaches by "eliminating the need for dimensional
  or token-by-token correspondence." This directly overlaps the "optimal-transport alignment" item in the
  task brief and is a mature (AAAI oral) alternative bridge design that OPRD-Bridge should be benchmarked
  against, not just PCA+MSE vs the proposed relational loss.
- "COT2ALIGN: Cross-Chain-of-Thought Distillation via Optimal Transport Alignment for LLMs with
  Different Tokenizers" (Le et al., cited in search results, not independently fetched) — OT applied
  specifically to reasoning-trace/CoT distillation across tokenizers, which is uncomfortably close to
  this plan's own math-reasoning cross-tokenizer setting. Needs a direct read before claiming ARH-Bridge
  is addressing an open problem in CoT-distillation-across-tokenizers; it may not be.
- "Cross-Tokenizer LLM Distillation through a Byte-Level Interface" (arXiv 2604.07466) — a third,
  non-OT/non-hidden-state approach (byte-level interface) to the same vocab-mismatch problem, found but
  not fetched in depth; flag as another 2026 competitor design worth a differentiation paragraph.

### 1d. Cross-attention / learned "soft correspondence" bridges (closest existing thing to a relational bridge)
- DSKD — "Dual-Space Knowledge Distillation for Large Language Models" (EMNLP 2024, arXiv 2406.17328;
  code: github.com/songmzhang/DSKD). General white-box KD framework for both same- and cross-tokenizer
  LLM distillation via a **cross-model attention (CMA)** mechanism that aligns representations without
  requiring a shared vocabulary or fixed linear projector.
- "Dual-Space Knowledge Distillation with Key-Query Matching for Large Language Models with Vocabulary
  Mismatch" (arXiv 2603.22056) — builds on DSKD-CMA, adds adversarial (GA) learning to fix mismatched
  key/query distributions between teacher and student attention spaces. Confirmed via WebFetch: "modest
  but consistent ROUGE-L gains... particularly on out-of-distribution data (+0.37 on average)."
  **This is the single closest already-published thing to a "relational" cross-architecture/cross-
  tokenizer LLM bridge**: it computes soft correspondences (attention/key-query) between teacher and
  student token representations rather than a fixed linear/PCA map. It is NOT phrased in kNN/ARH
  language and doesn't cite PRH/ARH, but structurally it is a learned soft-alignment bridge. The modest
  effect size (+0.37 ROUGE-L average, OOD) is a useful realistic prior for how much a smarter bridge
  design might move the needle — worth citing when setting expected-effect-size priors for ARH-Bridge.
- "DWA-KD: Dual-Space Weighting and Time-Warped Alignment for Cross-Tokenizer Knowledge Distillation"
  (arXiv 2602.21669) — combines dual-space token weighting (relevant to item C, soft position weights)
  with time-warped (i.e., non-1:1, alignment-search-based) token correspondence for cross-tokenizer KD.
  This is also close prior art for the plan's item (C) soft position/trajectory weighting AND for the
  cross-tokenizer bridge problem simultaneously — worth reading in full before claiming novelty on either
  axis.
- "CTPD: Cross Tokenizer Preference Distillation" (arXiv 2601.11865) — found, not fetched; likely
  relevant to on-policy / preference-based cross-tokenizer distillation, adjacent to OPRD's on-policy
  framing. Flag for a closer read.

---

## 2. Relational / neighborhood distillation for heterogeneous architectures — prior art directly on point

This is the most important thread for "is a mutual-kNN bridge actually novel."

- **Relational Knowledge Distillation (RKD)**, Park et al., CVPR 2019. Foundational: transfers
  *mutual relations* of data examples (pairwise distances, angles) instead of pointwise feature/output
  matching. Distance-wise loss = Huber loss on the difference of pairwise-distance matrices; angle-wise
  loss on triplet angles. This is structurally a "relational, not pointwise" loss from 2019 — six years
  before this plan. The draft's framing of "replace pointwise MSE with a relational loss" as a fresh idea
  needs to explicitly position itself against RKD and explain the delta (RKD: hard pairwise
  distance/angle over the *whole* batch/set, global; ARH-relational proposal: soft top-k neighbor KL,
  *local* window, identity not distance — i.e., borrowing RKD's "relations not points" thesis but
  changing which relational statistic is used, motivated by ARH's specific claim that identity survives
  but distance doesn't).
- Direct search finding: "Directly aligning intermediate features between ViT and CNN models is
  suboptimal due to their different architectural priors, but RKD captures higher-order inter-sample
  dependencies, which are more transferable across heterogeneous architectures" — but also "Direct use of
  RKD in heterogeneous distillation causes over-amplification of the role of dark knowledge... which can
  contribute to the failure of the RKD method" (from search summaries of cross-architecture KD surveys,
  not independently verified against one paper — treat as directionally right, not a precise quote).
- **Multi-Level Decoupled Relational Distillation (MLDR-KD)** for Heterogeneous Architectures (arXiv
  2502.06189, Feb 2026). Confirmed via WebFetch of the HTML version with direct quotes:
  - Explicitly decouples logits/features into **class-wise relation** and **sample-wise relation**,
    computed via a "scaled product relation" + softmax, then aligned via **KL divergence** — i.e., a
    *soft, normalized, identity/neighborhood-flavored* relational objective, not a raw distance loss.
    This is structurally extremely close to the plan's proposed "soft local softmax/KL over top-k
    teacher neighbors": MLDR-KD already does softmax + KL over relation matrices for cross-architecture
    heterogeneous distillation (CNN/ViT/MLP-Mixer/Vision-Mamba).
  - Uses learned projectors ("feature maps at each stage of student are projected to logit space")
    combined with the relational KL loss — i.e., projector + relational loss together, same combination
    the plan proposes for ARH-Bridge, just in vision classification rather than LLM reasoning hidden
    states.
  - **This is the closest single piece of prior art to "ARH-Bridge" found in this sweep.** The plan
    should cite MLDR-KD directly and argue the delta is (a) domain — autoregressive LLM hidden states +
    on-policy rollouts vs. vision classification features, (b) locality — token-window-local top-k neighbors
    vs. MLDR-KD's whole-batch/whole-class relation matrices, (c) theoretical grounding — ARH's specific
    width/depth-null-calibration argument for *why* local identity should be preferred, which MLDR-KD does
    not invoke (they get to relational loss empirically/heuristically, not via a representational-geometry
    hypothesis). Without that differentiation a reviewer familiar with the KD-for-vision literature will
    read ARH-Bridge as "MLDR-KD reinvented for LLMs, with a geometry justification bolted on."
- **Information Theoretic Representation Distillation (IRD)** (arXiv 2112.00459) and the broader
  contrastive-relational family (CRD and successors) — mutual-information / contrastive relational
  transfer between teacher/student representations. The plan explicitly wants to avoid InfoNCE/CRD-style
  contrastive losses (no negatives) — good, because that literature is large and already well-mined; the
  plan's choice to use a no-negatives local-softmax-KL variant instead of CRD is a real differentiator,
  but only if stated explicitly, since "relational distillation" as a category is otherwise dominated by
  contrastive methods in people's prior expectations.
- **MiniLM / MiniLM v2** (Wang et al., 2020/2021; arXiv 2002.10957, 2012.15828). Confirmed via WebSearch
  summaries: transfers self-attention *relations* (query-key attention distributions, and value-value
  relation matrices) rather than raw hidden states, and MiniLM v2 is explicitly designed to be **hidden-
  size- and head-count-agnostic** — "get around the need to have projection matrices that make the hidden
  dimensions of the teacher and student the same by instead forming a value relation matrix that is
  hidden-size agnostic." This is a second major, very well-known (NeurIPS 2020) precedent showing that
  relation-transfer specifically *solves the width/depth-mismatch problem without a projector at all* —
  which is a direct, load-bearing counter to any claim that ARH-Bridge's relational approach is needed
  *because* PCA/linear projectors are awkward across widths. MiniLM already shows you can dodge the
  projector question entirely by transferring relations instead of features. If ARH-Bridge still uses a
  frozen-teacher projector (per the plan's item E description) on top of a relational loss, the plan
  should explain why it doesn't just drop the projector MiniLM-style, or that comparison will be an
  obvious ablation a reviewer asks for.

---

## 3. PRH / ARH theoretical layer: is there a THIRD position in the literature?

- **"Back into Plato's Cave: Examining Cross-modal Representational Convergence at Scale"** (arXiv
  2604.18572). Per WebSearch snippets (not independently fetched in full): finds mutual-kNN agreement is
  "highly sensitive to the evaluation regime: it drops sharply when moving from small galleries to
  million-scale datasets and degrades further under many-to-many cross-modal correspondences," while also
  reporting "local neighborhood overlap (e.g., mutual kNN) remains significantly aligned across modalities
  even after calibration, while global spectral 'convergence' mostly disappears" — this second half is
  consistent with / supportive of the ARH claim structure (local survives, global doesn't), but the first
  half is a genuine caveat: mutual-kNN itself is not scale-invariant and degrades at large gallery sizes /
  many-to-many settings. **This matters directly for A5/E**: if the student-generated rollout token pool
  used as the "neighbor gallery" for a local top-k loss is large (thousands of tokens), this paper's
  finding suggests the neighbor-identity signal itself may become unstable at that scale, independent of
  the PRH/ARH debate — worth a pilot check (does mutual-kNN agreement between teacher and a same-arch
  student stay stable as the candidate-neighbor pool grows to full-context-window size?) before building
  an entire loss around it.
- **"Representation Alignment Rests on Linear Structure"** (Bangachev, Bresler, Polyanskiy — arXiv
  2605.28870). Fetched the actual abstract text directly (verbatim, via WebFetch on the arxiv abstract
  page): the paper (1) proposes a Linear Representation Hypothesis component, using sparse autoencoders
  to show *stronger* cross-modal alignment than dense representations; (2) shows centering/normalization
  "improve cross-model alignment"; (3) shows a "strong and consistent positive correlation between word
  frequency and alignment in LLMs and text embedding models," i.e. ties representational noise to data
  scarcity. **Caveat on this finding**: an earlier WebFetch pass on the PDF (before I got the clean
  abstract) produced a summary claiming this paper "undermines the premise that mutual-kNN provides a
  fundamentally different or superior alignment signal" — that claim is NOT clearly supported by the
  actual abstract text and may have been the fetch model over-interpreting a passing mention of CKA/
  mutual-kNN in the paper's related-work framing. I could not fully verify the paper's stance on the
  specific local-vs-global ARH question from the abstract alone. **Treat as unverified / context, not a
  confirmed refutation of ARH.** What IS solid: this paper offers a *third* theoretical account of cross-
  model representational alignment (linear-structure + SAE sparsity + word-frequency-driven noise) that is
  orthogonal to both PRH's "convergence toward a shared statistical model" and ARH's "local survives,
  global doesn't" framing. The research plan's binary framing ("PRH vs ARH") is likely incomplete — there
  is at least one more active hypothesis in the 2026 alignment-metrology literature, and the plan's
  motivation section for E (ARH-Bridge) should not present this as a clean two-sided debate. Recommend the
  planning team actually read 2605.28870 in full (not just abstract) before citing it either way.

---

## 4. Direct answer to "is a mutual-kNN bridge actually novel?"

**Partial novelty, not clean novelty.** Breaking the proposal into pieces:

1. "Use relational/neighborhood structure instead of pointwise MSE for cross-architecture distillation" —
   **NOT novel** as a general strategy. RKD (2019), MiniLM/MiniLM-v2 (2020-2021), and MLDR-KD (Feb 2026)
   all already do this, the last one in a form (softmax + KL over class-wise/sample-wise relation
   matrices) that is structurally very close to the plan's "soft local softmax/KL over top-k teacher
   neighbors."
2. "Frame the choice via PRH/ARH representational-geometry theory (local identity survives width/depth
   null-calibration, global doesn't) rather than empirical trial-and-error" — this specific *justification*
   appears to be new; none of RKD/MiniLM/MLDR-KD invoke PRH/ARH or null-calibration arguments. This is
   probably the most defensible novel contribution of the plan's framing.
3. "Apply it to cross-architecture bridging for **autoregressive LLM hidden states on-policy reasoning
   rollouts**, specifically token-window-local top-k neighbors (not global/batch relations), with no
   negatives (unlike CRD/IRD)" — this specific combination (domain + locality + no-negatives) was not
   found in the literature searched. It is a real, narrow gap, but it is an incremental combination of
   existing pieces (RKD/MLDR-KD's relational-KL idea + ARH's locality argument + on-policy LLM
   distillation), not an unprecedented mechanism.
4. Mutual-kNN as a **metric** for measuring alignment (Huh et al. PRH 2024, Back into Plato's Cave, ARH
   itself) is well established. Mutual-kNN or kNN-identity **as a training loss** for cross-architecture
   distillation specifically — I could not find a paper that does exactly this for LLMs. It's the metric
   used elsewhere as an evaluation tool, not (as far as this sweep found) as a first-class training
   objective. That gap is real, but it is adjacent enough to RKD/MLDR-KD-style relational losses that
   reviewers are very likely to ask "how is this different from RKD/MiniLM-v2/MLDR-KD" and the plan does
   not currently have that comparison built in.

**Recommendation for the planning team:** before running (B) or (E), add explicit ablations against (i)
RKD-style distance/angle loss, (ii) MLDR-KD-style class/sample relation-KL loss, and (iii) MiniLM-v2-style
value-relation transfer (projector-free). If the ARH-motivated local top-k soft-KL loss doesn't beat all
three on the cross-architecture / cross-tokenizer setting, the "relational objective beats pointwise MSE"
claim survives but the "PRH/ARH theory materially informed a better design" claim does not — that's the
sharper, falsifiable version of the plan's own question, and it's currently missing as a comparison arm.

---

## Sources (URLs)

- OPRD (primary paper): https://arxiv.org/abs/2606.06021 / https://arxiv.org/html/2606.06021
- Relational Knowledge Distillation (RKD), CVPR 2019: https://openaccess.thecvf.com/content_CVPR_2019/papers/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.pdf
- MiniLM: https://arxiv.org/pdf/2002.10957
- MiniLMv2: https://arxiv.org/pdf/2012.15828
- Multi-Level Decoupled Relational Distillation for Heterogeneous Architectures: https://arxiv.org/html/2502.06189v1
- Cross-Architecture Distillation Made Simple with Redundancy Suppression: https://arxiv.org/abs/2507.21844
- Feature Structure Distillation with CKA in BERT Transferring: https://arxiv.org/pdf/2204.08922
- Rethinking Centered Kernel Alignment in Knowledge Distillation (IJCAI 2024): https://arxiv.org/pdf/2401.11824
- Universal Logit Distillation (ULD): https://arxiv.org/abs/2402.12030
- Multi-Level Optimal Transport for Universal Cross-Tokenizer KD (AAAI 2025 oral): https://arxiv.org/abs/2412.14528
- Cross-Tokenizer LLM Distillation through a Byte-Level Interface: https://arxiv.org/html/2604.07466v1
- DSKD (Dual-Space KD, EMNLP 2024): https://arxiv.org/html/2406.17328 ; https://github.com/songmzhang/DSKD
- DSKD with Key-Query Matching (vocab mismatch, GA extension): https://arxiv.org/abs/2603.22056
- DWA-KD (Dual-Space Weighting + Time-Warped Alignment): https://arxiv.org/html/2602.21669v1
- CTPD (Cross Tokenizer Preference Distillation): https://www.arxiv.org/pdf/2601.11865
- Information Theoretic Representation Distillation (IRD): https://arxiv.org/pdf/2112.00459
- Back into Plato's Cave (cross-modal mutual-kNN at scale): https://arxiv.org/html/2604.18572 / https://arxiv.org/pdf/2604.18572
- Representation Alignment Rests on Linear Structure: https://arxiv.org/abs/2605.28870 (also seen as https://arxiv.org/pdf/2605.28870)

## Fetches that failed to yield readable text (noted for transparency, not used as evidence)
- https://arxiv.org/pdf/2412.14528 — binary/PDF decode failure; substituted with the arxiv.org/abs page.
- https://arxiv.org/pdf/2606.06021 — binary/PDF decode failure on first attempt; substituted with the
  arxiv.org/html version, which worked and is the source for the OPRD-Bridge quotes above.
- https://aclanthology.org/2025.findings-acl.419.pdf ("Enhancing Cross-Tokenizer Knowledge Distillation
  with ...") — binary/PDF decode failure, could not recover title-suffix or method; not used.
