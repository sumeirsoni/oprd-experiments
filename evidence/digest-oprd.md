# Digest: OPRD — On-Policy Representation Distillation (arXiv 2606.06021v4, 20 Jun 2026)

Authors: Shenzhi Yang, Guangcheng Zhu, Bowen Song, Haobo Wang, Mingxuan Xia, Xing Zheng,
Yingfan Ma, Zhongqi Chen, Weiqiang Wang, Junbo Zhao, Gang Chen (Zhejiang Univ. / Ant Group).
Code: https://github.com/ShenzhiYang2000/OPRD. Status: "Work in progress" preprint, not peer-reviewed.

Source: full PDF fetched directly from arxiv.org/pdf/2606.06021 and read page-by-page (30 pages:
main body + notation + start of theoretical appendix). alphaxiv.org abs/pdf pages both returned
HTTP 403 (blocked); arxiv.org/abs page gave only an abstract-level summary via WebFetch's small
model. All quotes/numbers below are taken directly from the primary PDF text.

---

## 1. Exact loss functions

### 1.1 Background: OPD (baseline being replaced/augmented)
Trajectory-level reverse KL on student on-policy rollouts, decomposed into per-token terms (Eq. 2):
`L_OPD(θ) = E_{x~D_x, ŷ~π_θ(·|x)} [ Σ_t D_KL(p_t || q_t) ]`
Three variants differ only in which token subset `S_t ⊆ V` is scored per position:
- **Sampled-token OPD** (`S_t={ŷ_t}`, Eq. 3): single-sample log-ratio estimator, unbiased single-sample estimator of per-token reverse KL. Cheapest, most widely deployed.
- **Full-vocabulary OPD** (`S_t=V`, Eq. 4): exact per-position KL, O(BT|V|) memory — infeasible at scale.
- **Top-k OPD** (`S_t=TopK(p_t,k)`, Eq. 5): KL over student's top-k renormalized support; interpolates the two extremes. Typical k∈[4,64]; paper uses k=16 as its top-k baseline.

### 1.2 OPRD-Vanilla objective (Eq. 6) — the core contribution
```
L_OPRD(θ) = E_{x~D_x, ŷ~π_θ(·|x)} [
  (1/|L_layer|) Σ_{l∈L_layer} (1/Σ_t m_t) Σ_{t=1}^{T} m_t · (1/d) || h_θ,t^(l) − sg(h_T,t^(l)) ||_2^2
]
```
- `L_layer ⊆ {1,...,L}`: set of distilled layers (e.g. last layer, all layers, parity subset).
- `P(ŷ) ⊆ {1,...,T}`: supervised response positions, encoded as binary mask `m_t`.
- `sg(·)` = stop-gradient on teacher representation (teacher frozen).
- `1/d` normalizes across architectures with different hidden sizes.
- `1/Σ_t m_t` (position averaging) makes the loss invariant to `|P(ŷ)|` (invariant to how many positions are supervised).
- **Default/reported configuration**: `L_layer = {1,...,L}` (**all 28 layers** of the Qwen2.5-1.5B backbone used) and `P(ŷ)` = **last k=2000 response tokens** (i.e., "the suffix in which the chain-of-thought converges to a final answer"). This is stated explicitly: "For long chain-of-thought (CoT) responses common in mathematical reasoning, we typically set P(ŷ) to the last k response tokens and L_layer to all transformer layers." Main results (§4) are reported in the **OPRD-only setting (μ=0)**.
- Composable additively with any OPD variant at "essentially zero infrastructure cost" (Eq. 7): `L(θ) = L_OPD(θ) + μ·L_OPRD(θ), μ≥0`.

Two accompanying theorems (informal in main text, formal proofs in Appendix B, not fully extracted):
- **Theorem 1 (zero-variance gradient)**: `Var[g_OPRD | x,ŷ] = 0`, whereas `Var[g_OPD]` is proportional to variance of the per-position log-ratio over the sampled token and does **not** vanish as `p_t→q_t`; this is claimed as the mechanism behind OPD's late-stage plateau.
- **Theorem 2 (LM-head information bottleneck)**: defines the effective null space `N_W` of `W_head` (hidden perturbations that are softmax-invariant shifts). Shows `||h_θ−h_T||^2 / ℓ_out(h_θ,h_T) ≳ (σ_1/σ_d)^2` along the top singular direction of `W_head`'s null space — i.e., hidden deviations along low-singular-value directions of `W_head` can be arbitrarily large while producing zero output-space loss. This ratio is "typically very large for production LLMs due to the ill-conditioned singular spectrum of W_head."

### 1.3 OPRD-Bridge objective (cross-architecture) — two-stage pipeline
**Stage 1 (bridge construction, offline, both backbones frozen).** Layer correspondence via proportional-depth mapping (Eq. 11): `φ(l_S) = round((l_S−1)/(L_S−1) · (L_T−1)) + 1`. Teacher projector obtained via PCA of centered teacher hidden-state covariance per layer (Eq. 12): `P_T^(l_T) = TopPrincipalDirections(h_T^(l_T) − μ_T^(l_T), r) ∈ R^{r×d_T}` — computed once via eigendecomposition of the d_T×d_T covariance matrix (not a thin SVD of the full token matrix), subsampling ≤16,384 token rows, then **permanently frozen**. Student projector `P_S^(l_S) ∈ R^{r×d_S}` is a trained *linear* map minimizing (Eq. 13, trained for 20 epochs with AdamW, both backbones frozen):
```
L_bridge = Σ_l Σ_t || P_S^(l_S) h_{S,t}^(l_S) − P_T^(φ(l_S)) (h_{T,t}^(φ(l_S)) − μ_T^(φ(l_S))) ||_2^2
```
After convergence, `P_S` is also frozen.

**Stage 2 (bridge-based distillation, online, on-policy; bridge + teacher frozen).** Only the student backbone θ receives gradients, through the frozen bridge (Eq. 14–15):
```
ℓ^(l)(θ;ŷ) = (1/Σ_t m_t) Σ_t m_t || P_S^(l)(h_{θ,t}^(l)) − P_T^(φ(l))(h_{T,t}^(φ(l)) − μ_T^(φ(l))) ||_2
L_OPRD-Bridge(θ) = E[ (1/|L_layer|) Σ_l ℓ^(l)(θ;ŷ) ]
```
Both projected vectors are **L2-normalized before computing MSE**, making the loss scale-invariant. Same `m_t` mask as OPRD-Vanilla (default last k=2000 tokens). Composable with OPD via Eq. 7 exactly as OPRD-Vanilla. Default rank **r=8** (r=4 used only in the cross-tokenizer experiment).

### 1.4 PRH motivation (quote)
"The Platonic Representation Hypothesis [15=Huh et al. 2024] provides theoretical grounds to believe such a bridge should exist. It posits that strong neural networks, regardless of architecture, are converging toward a shared statistical model of reality in their representation spaces." The paper's own empirical support: pairing Qwen3-4B teacher (36 layers, d=2560) with Qwen3-1.7B-Base student (28 layers, d=2048), a **rank-8 linear subspace achieves 95% cosine similarity** between projected representations of the two models. Similarity **decreases with higher rank** (94.1% at r=32, 87.9% at r=256, 77.0% at full rank r=2048) — interpreted as "the shared structure concentrates in very few principal directions and... additional dimensions introduce noise rather than signal." Theorem 3 (informal) frames this as a rate-distortion optimal bias-variance trade-off: `E(r) = Σ_{i>r} λ_i ρ_i^2 (bias) + Σ_{i=1}^r λ_i(1−ρ_i^2) (variance)`, assuming teacher covariance eigenvalues `λ_i` and a per-direction cross-model alignment `ρ_i ∈[0,1]` that is *assumed* (and "empirically verified" only for this one pair) to decrease in i.

**Important scope caveat for this evidence**: the demonstration pair (Qwen3-4B/Qwen3-1.7B-Base) is **same-family, same-tokenizer, same lab/training pipeline** — i.e., not a strong test of "heterogeneous architectures" in the PRH sense (different training runs/data/init but same base recipe). The truly heterogeneous test (different tokenizer, Phi-4-mini-reasoning → Qwen3-1.7B-Base) is presented later (§4.2.4) but with much weaker absolute numbers (see §3 below). No CKA/mutual-kNN/CKNNA or any null-calibrated (random-projection-baseline) metric is reported anywhere in the paper — only raw/uncalibrated cosine similarity of the learned projections. This is directly relevant to ARH's critique that naive spectral-alignment numbers can look inflated absent width/depth null calibration; the OPRD-Bridge paper does not perform or address this calibration at all.

---

## 2. Baselines used throughout

- Frozen teacher, unmodified student (no distillation) — floor/ceiling references.
- **OPD top-1** (sampled-token reverse KL, Eq. 3).
- **OPD top-16** (Eq. 5, k=16).
- **OPRD-Vanilla** (μ=0, standalone).
- **OPD + μ·OPRD** composite, μ∈{0,1,10} (§4.1.5 ablation).
- For OPRD-Bridge: same OPD top-1 / OPD top-16 baselines (applicable cross-architecture because they only need a shared vocabulary), plus a two-stage "OPRD-Bridge → OPD top-1" composite.
- For cross-tokenizer: standard OPD is structurally inapplicable ("collapses") since it needs a shared vocabulary — so OPRD-Bridge's only real competitor there is the unmodified student, not a working OPD baseline.

No comparison against classical feature-distillation baselines (FitNets, TinyBERT, MiniLM), no contrastive/InfoNCE baseline, no CKA-only or full-Gram baseline, no ARH-style local-neighborhood metric baseline — all relevant to the plan's proposed controls being genuinely novel relative to this paper.

---

## 3. Effect sizes / numbers (what the paper SHOWS, separated from framing)

### 3.1 OPRD-Vanilla, same architecture (Table 2; Qwen2.5-1.5B backbone; JustRL-1.5B teacher, R1-distill-1.5B student — both already independently post-trained checkpoints on the same base model, i.e. a favorable, pre-aligned setting)
Avg@16 (%):
| Method | AIME24 | AIME25 | AIMO |
|---|---|---|---|
| Teacher (JustRL-1.5B) | 50.8 | 35.6 | 79.5 |
| Student (R1-distill-1.5B) | 32.9 | 21.9 | 62.2 |
| OPD top-1 | 42.3 | 33.5 | 77.0 |
| OPD top-16 | 47.1 | 34.0 | 76.5 |
| **OPRD-Vanilla** | **49.8** | 34.6 | **79.1** (underlined = within teacher's eval noise) |

OPRD gains vs. better OPD baseline: **+2.7 / +0.6 / +2.1** pts (AIME24/25/AIMO). Vs. unmodified student: +16.9/+12.7/+16.9. All three methods trained for only **500 optimizer steps**, single run each (no seeds/error bars reported anywhere in the paper for Table 2, 4, 5, or 6 — a real statistical-rigor gap given some deltas are as small as 0.5–0.6 pts).

Efficiency (Table 3, same 500-step budget, 8×A100 FSDP): OPRD **563 min** wall-clock vs 813/812 min for OPD top-1/top-16 (**1.44× speedup**); Δpeak actor-update memory **20.5 GB** vs 30.2/45.0 GB (**32%/54% reduction**). This is attributed to OPRD never materializing the `[B,T,|V|]` logits tensor.

Mechanistic/behavioral: OPRD converges to ~5,700-token mean responses vs ~7,000 for both OPD variants (shorter chains at higher accuracy — but note: no diversity metric, e.g. Dist-n, is reported for this same-architecture experiment, only for the cross-architecture one). OPD+OPRD composite sweep (Fig. 7, AIME24 only): μ=0 → 42.3, μ=1 → 47.7 (+5.4 over μ=0, already beats standalone OPD top-16's 47.1), μ=10 → 50.2 (+7.9 over μ=0). Position-divergence probe (Fig. 8): first-k teacher-student cosine similarity stays ≥97% for k≤1600 (peaks 97.69% at k=1600); last-k similarity starts at only 91.65% (k=50) and stays ≥4 pts below the first-k curve until k approaches full response length — i.e., **student-teacher divergence concentrates in the tail of the response**, which is the paper's own stated justification for its last-2000-token default.

### 3.2 OPRD-Bridge, cross-architecture, shared vocab (Table 4; Qwen3-4B teacher [36L, d=2560] → Qwen3-1.7B-Base student [28L, d=2048], r=8)
Avg@16 (%, primary accuracy metric):
| Method | AIME24 | AIME25 | AIMO | Avg |
|---|---|---|---|---|
| Teacher | 23.1 | 25.2 | 59.4 | 35.9 |
| Student | 0.6 | 0.2 | 7.5 | 2.8 |
| OPD top-1 | 8.5 | 6.3 | 31.0 | **15.3** |
| OPD top-16 | 6.7 | 4.0 | 20.5 | 11.7 |
| **OPRD-Bridge (r=8)** | 5.0 | 3.8 | 24.8 | **11.2** |
| OPRD-Bridge → OPD top-1 (composite) | 9.6 | 3.0 | 28.2 | 13.6 |

**On the primary Avg@16 metric, OPRD-Bridge (11.2) is worse than OPD top-1 (15.3) and roughly comparable to (slightly below) OPD top-16 (11.7).** Even the two-stage composite (13.6) does not beat standalone OPD top-1 (15.3) on Avg@16.

Best@16 (%, "capability ceiling"): OPD top-1 30.8, OPD top-16 27.1, OPRD-Bridge 30.8 (**exact tie** with OPD top-1), composite 33.9 (best overall). The paper's own framing: "the Avg@16 gap between OPRD-Bridge and OPD reflects a difference in *consistency*... not in *capability*." This is a claim, not a shown result on the metric that most directly reflects deployed single-sample accuracy.

Generation-quality (Table 5): OPRD-Bridge Dist-4g (lexical diversity) 22.6/17.2/21.3 (avg 20.4) vs OPD top-1 6.2/6.4/15.6 (avg 9.4) and OPD top-16 (avg 11.7) — substantially higher diversity, closer to teacher's own 24.0/25.4/41.6. Avg response length: OPRD-Bridge 5,909/5,234/4,855 (avg 5,333) vs OPD top-1 avg 14,550, i.e. **~2.7× shorter**.

**Bridge-rank ablation** (Fig. 12, §4.2.3): before Stage-1 training, projected representations are "essentially orthogonal at all ranks" (cosine sim ≤7.5%); after training, similarity peaks at r=8 (95.0%) and *declines monotonically* thereafter (94.1% at r=32, 87.9% at r=256, 77.0% at full rank 2048). Paper's own read: rank must be tuned to a narrow "sweet spot" (r=4–16, 93.7–95.0% plateau); this is presented as validating a low-pass-filter interpretation, but it is also a hyperparameter-sensitivity finding — no rank was ablated for downstream accuracy (Table 4), only for the internal cosine-similarity diagnostic.

### 3.3 Cross-tokenizer distillation (Table 6; Phi-4-mini-reasoning teacher [32L, d=3072, ~200K tiktoken vocab] → Qwen3-1.7B-Base student, r=4)
Standard OPD "collapses" (structurally inapplicable — no shared vocabulary). OPRD-Bridge: Avg@16 goes from student's 2.8% to **5.0%** (teacher: 50.0%); Best@16 from student's 4.5% to **21.3%** (teacher 67.7%, a "4.7× improvement" over student). This is the cleanest directional win for OPRD-Bridge, but (a) the comparison baseline is degenerate (OPD cannot run at all rather than underperforming), and (b) absolute numbers remain far below the teacher (5.0% vs 50.0% Avg@16 — a 45-point gap that OPRD-Bridge closes only modestly).

---

## 4. What OPRD-Bridge actually shows vs. claims (adversarial summary)

**Shows convincingly:**
- Cross-tokenizer distillation is *possible* via a learned low-rank representation bridge when the vocabulary-based channel is structurally unavailable (student moves off a near-zero floor).
- Within one same-family model pair (Qwen3-4B/1.7B), a very-low-rank (r≈8) linear subspace captures most of the *learnable* cosine alignment between projected hidden states, and that alignment number is non-monotonic in rank (peaks low, then degrades) — an interesting empirical regularity, but demonstrated on a single teacher-student pair only, no calibration baseline, no repeated pairs across architecture families.

**Does NOT show, despite framing:**
- That OPRD-Bridge beats output-space OPD on the metric that matters most for a single deployed sample (Avg@16) in the shared-vocabulary cross-architecture setting — it loses to OPD top-1 there (11.2 vs 15.3) and only ties (not beats) on Best@16.
- That the "shared low-rank structure" motivating the PRH framing generalizes beyond one same-family, same-tokenizer pair; the strongest PRH-style evidence (95% cosine sim at r=8) is drawn from Qwen3-4B vs Qwen3-1.7B-Base, models sharing lineage/tokenizer/training recipe — a much easier case than "heterogeneous architectures" in the PRH sense the paper invokes.
- Any test of whether the observed alignment is more than a null-calibration artifact — no CKA, no mutual-kNN/CKNNA, no random-projection or dimensionality-matched control is computed. This is exactly the methodological gap ARH-style calibration is designed to close, and this paper does not close it (nor claim to).
- That the PCA+MSE bridge choice is compared against any relational/local-neighborhood alternative — no such ablation exists in the paper, so Phase E's "does ARH-Bridge beat OPRD-Bridge" question is entirely open, not something this paper already answers either way.

---

## 5. Stated limitations / open questions (paper's own "Future Work," §7, verbatim-derived)

The paper has no separate "Limitations" section; limitations are framed either as motivating the method (Limitations 1–2 of OPD, §1, already captured above) or as forward-looking gaps in Conclusion/§7:
1. **"Beyond mathematical reasoning."** All experiments are long-CoT math only (AIME24/25, AIMO/AMC); "Whether OPRD's gains transfer to code generation, agentic interaction, and open-ended dialogue... remains an open question."
2. **Cross-modal distillation** flagged as future work, not attempted.
3. **"Adaptive layer and position selection."** Explicit admission: "We use uniform layer weighting and a simple last-k position heuristic. Adaptively weighting layers and positions based on where the student–teacher gap is largest or where the gradient signal is most informative could further sharpen supervision." — i.e., the paper never ablates layer-subset selection or non-uniform/soft position weighting; this is left entirely open.
4. **OPRSD (on-policy self-distillation) scaling to multi-turn/multi-task** — future work, not tested here.
5. **"Understanding the phase transition."** The PG-loss spike (Fig. 9) and associated entropy/overlap dynamics are observed but explicitly not mechanistically explained: "the precise mechanism is under active investigation."
6. **Attention-map distillation** not attempted — OPRD aligns hidden vectors only, not attention patterns.
7. **Alignment-aware pre-training** — speculative direction, would obviate the need for a bridge; not tested.
8. **"Tighter theoretical bounds."** Authors acknowledge their theorems are qualitative/asymptotic ("identifies the qualitative mechanisms"), not quantitative convergence-rate bounds.
9. Implicit/unstated limitations noticed on close reading:
   - No seeds / variance reporting anywhere in the main results tables (2, 4, 5, 6) — single run per method per benchmark.
   - Main same-architecture win (Table 2) is in an unusually favorable setting: teacher and student are independently-trained checkpoints of the **same base model** (Qwen2.5-1.5B) with, per the paper's own §3.3 admission, "a high degree of representational alignment" already — i.e., the exact regime where MSE-on-hidden-states is easiest, not a stress test.
   - Diversity metrics (Dist-4g) are reported for the cross-architecture experiment but never for the flagship same-architecture experiment, so it's unverifiable whether OPRD-Vanilla's shorter responses (Fig. 5) come at a diversity cost in that setting.
   - OPRD-Bridge's own headline accuracy result (Table 4) underperforms plain OPD top-1 on Avg@16, a fact stated in the table but reframed rhetorically as a "consistency, not capability" gap in the text — a claim the paper does not independently validate (e.g., no analysis of variance/entropy of the Avg@16 distribution to support "consistency" framing).
   - Bridge rank r is tuned per experiment (r=8 same-arch, r=4 cross-tokenizer) via an internal cosine-similarity diagnostic, not via downstream accuracy sweep — the accuracy-optimal rank is never established.
   - No comparison to any relational/contrastive/local-neighborhood objective (as the research plan itself notes) — this paper's PCA+MSE bridge is not benchmarked against any alternative bridge design.

---

## 6. Relevance to the draft plan (files read: relational_oprd_experiments_7e321945.plan.md, oprd-literature-review.md)

- **Loss/config match**: the plan's stated baselines ("Student SFT/base checkpoint," "Standard OPD," "OPRD (all-layer MSE, last-k tokens as in paper)," "OPD+OPRD") match this paper's actual defaults exactly (Eq. 6, all-28-layers, last k=2000, Eq. 7 composite) — the plan's baseline list is accurate and directly reusable.
- **Phase A1 (position bias)**: paper's Fig. 8 (first-k vs last-k cosine similarity) is a partial precedent but only measures *where teacher-student hidden states already disagree*, not a controlled prefix/middle/suffix-only training ablation and not compared against OPD's own position bias — Phase A1 is not redundant with this paper.
- **Phase A2 (layer selectivity)**: paper explicitly flags this as unexplored ("uniform layer weighting... could further sharpen supervision") — directly supports Phase A2 as a genuine gap, not redundant.
- **Phase A3/D (update geometry)**: paper performs zero geometry/stable-rank analysis — fully open, supports the plan.
- **Phase B (ARH-aligned relational loss)**: paper never tests any relational/contrastive/local-neighborhood alternative to pointwise MSE — open. Note the paper's own PCA+MSE motivation is explicitly "global"/PRH-flavored (shared low-rank *coordinate* subspace via PCA + coordinate MSE), which is exactly the kind of objective the plan classifies as PRH-flavored-control rather than ARH-aligned — consistent with the plan's framing.
- **Phase E (ARH-Bridge vs OPRD-Bridge)**: this is the highest-value target. The paper's own Table 4 shows OPRD-Bridge **losing to OPD top-1 on Avg@16** in the shared-vocab cross-architecture setting and only tying on Best@16 — meaning the bar OPRD-Bridge itself has actually cleared is lower than the paper's rhetoric suggests. This substantially de-risks Phase E: the plan does not need to beat a strong, already-validated result — it needs to beat a PCA+MSE baseline whose own paper shows a real accuracy shortfall against plain output-space OPD. That said, the cross-tokenizer comparison (Table 6) is the one place OPRD-Bridge looks unambiguously good, but only because OPD cannot run at all there — an ARH-Bridge candidate should be compared against OPRD-Bridge directly on both the shared-vocab (Table 4 style) and cross-tokenizer (Table 6 style) settings, not just against degenerate OPD.
- **PRH vs ARH question**: the paper's PRH evidence (95% cosine sim at r=8) comes from one same-family pair with no null-calibration control (no CKA, no mutual-kNN, no random-projection baseline) — this is precisely the gap ARH-style calibration targets, and the plan's premise that "OPRD-Bridge's motivation is PRH-flavored and under-tested against ARH's calibrated-local-structure claim" is well-supported by what this paper does (and does not) show.
