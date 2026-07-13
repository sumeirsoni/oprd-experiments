# Digest: "On the Geometry of On-Policy Distillation" (arXiv 2606.07082)

**Authors:** Zhennan Shen, Yanshu Li, Qingyu Yin, Chak Tou Leong, Zhilin Wang, Yanxu Chen, Rongduo Han, Sunbowen Lee, Yi R. Fung. HKUST / Brown / Zhejiang U / HK PolyU / USTC / BUPT / Nankai / BIT.
**Version:** v3, 14 Jun 2026. 17 pages, 8 figures + appendices A–G.

**Retrieval note:** alphaxiv.org returned HTTP 403. arxiv.org/abs and arxiv.org/pdf WebFetch summaries (which route through a secondary summarization model) contained fabricated specifics — one claimed "Llama 2 (7B, 13B)" as the tested models, which is **wrong**; the paper uses only the Qwen3 family. I re-fetched the PDF and read all 17 pages directly (via the Read tool's PDF extractor) rather than trusting the WebFetch summary. All facts below are taken from the primary-source page images/text, with page/section numbers.

---

## Core research questions (p.1)

- RQ1: Where does OPD lie in the parameter-space spectrum between SFT and RLVR?
- RQ2: What intrinsic update trajectory does OPD follow during training?
- RQ3: Which component of OPD controls this trajectory?

**Critical scope note:** "OPD" in this paper = standard **output-space** on-policy distillation — reverse-KL / token log-prob matching on student rollouts (Eq. 19: `L_OPD = Σ_t D_KL(p_θ^t || q_T^t)`, formalized via sampled-token log-prob gap `r_t = logπ_θ(a_t|s_t) − logπ_T(a_t|s_t)`, Appendix D/G.2). **The paper never defines or tests a hidden-state/representation-matching objective (i.e., nothing OPRD-shaped).** The three paradigms formalized throughout (Appendix D, Eq. 16–23) are SFT, RLVR (GRPO), and output-space OPD only.

## Section 3 — RQ1: OPD occupies a "relaxed off-principal regime"

Diagnostic suite adopted from Zhu et al. 2025 (PNT), applied to Qwen3-8B controlled comparison (SFT vs OPD vs RLVR, all sharing the same SFT-anchor init for OPD/RLVR):

- **bf16-aware update sparsity** `sp_bf16(W0,W+)` (Eq. 1): treats a weight as unchanged if the bf16 rounding gap is below tolerance η=1e-3. Results (Table 1, controlled Qwen3-8B): SFT unchanged=8.1%, OPD=51.6%, RLVR=77.2%. Stable across OPD "variant" runs (teacher/student scale, code domain, MoE teacher): 48.6%–57.1%.
- **Principal-angle rotation** (Eq. 2), rank k=512, CPU float64 SVD (explicitly needed — "GPU SVD can produce unstable angles for near-identical matrices," p.12): representative-layer top-k angle magnitudes — SFT >10°, OPD ~1°, RLVR <0.5° (p.5, Fig. 2).
- **Normalized spectral shift (NSS)** (Eq. 3): SFT ~1e-3, OPD ~1e-4, RLVR ~1e-5 (Fig. 2, p.4–5).
- **Update–mask overlap** (principal mask M_princ from rank-r=64 SVD reconstruction, α=0.5, vs low-magnitude mask M_low): global update density SFT 92.73%, OPD 46.28%, RLVR 27.76%; principal-mask overlap decreases monotonically SFT→OPD→RLVR, all below the 30% random baseline; low-magnitude overlap SFT 31.88%, OPD 53.59%, RLVR 73.48% (Fig. 3, p.5).

Conclusion: OPD sits consistently between SFT and RLVR on all four axes, biased toward RLVR — called the "relaxed off-principal regime." Section 3.3 gives a mechanistic "relaxed Three-Gate" account (KL anchor / model geometry / bf16 precision-realization gates) explaining this via update-covariance structure (Eq. 4–14): RLVR uses one scalar sequence-level coefficient a_t=A(y) shared across tokens; OPD uses heterogeneous per-token coefficients a_t from teacher–student disagreement — broadening directional support relative to RLVR while remaining geometry-steered.

## Section 4 — Subspace-locking finding (the load-bearing result)

**Setup:** track cumulative update ΔW_t = W_t − W0 across checkpoints (main config: Qwen3-8B student, Qwen3-32B teacher, dapo-math-17k, 299 training steps, checkpoints every 64 steps).

- **Stable rank** `srank(ΔW_t) = ||ΔW_t||_F² / ||ΔW_t||_op²` (Eq. 8/31), averaged over 144 matrices (36 layers × 4 attention-weight types **only** — QKVO; MLP/FFN weights are NOT included in the spectral trajectory analysis, per Appendix G.1 Table 8: "36 layers × 4 attention weights").
- Finding (Fig. 4a, p.7): "OPD stays within a narrow low-rank band throughout training [~18→27]. SFT progressively expands its update subspace [~50→75]. RLVR contracts toward a low-dimensional endpoint [~65→20]." Quote: **"OPD is not a temporal interpolation between the two: it enters the low-dimensional regime early and remains there."**
- Ruled out trivial small-update explanation via **Frobenius norm** (Eq. 32) — OPD accumulates substantially larger cumulative update than RLVR while ending with *comparable* stable rank (i.e., the low rank is not just "nothing moved") — and via **Hill tail estimate** (Eq. 33, auxiliary spectral-shape check) — OPD's tail exponent evolves mildly vs. SFT increasing sharply and RLVR decreasing (Fig. 4b–c).
- **Note for adversarial reading:** OPD's *final* stable rank (~20–27) ends up numerically close to RLVR's (~20). The distinguishing claim is about the *trajectory* (when/how the low-rank channel is entered and whether it's stable early), not that OPD's final rank is uniquely lower than RLVR's.

**Subspace emergence (§4.1, Fig. 5):** Similarity of top-16 right-singular subspace V_16(t) to the *final* V_16(t_end) (Eq. 9) — OPD reaches high similarity (~0.7+ from the first measured checkpoint, converging to 1.0) much earlier than SFT/RLVR, which converge "more gradually." I.e., OPD's channel is set early, not assembled late.

**Functional sufficiency (§4.2, Fig. 6):** Rank-16 projected training — extract V_16 from ΔW at ~20% training progress, then constrain all subsequent gradients to that subspace (`g ← g V_16 V_16^T`, Eq. 10) and resume training. Result: **OPD is "essentially unchanged"** under this rank-16 bottleneck (AIME 2024 avg@8, Fig. 6a — noisy but roughly flat/comparable trajectory vs. unconstrained). **SFT is substantially degraded** by the identical constraint (Fig. 6b — rank-16 SFT trajectory clearly worse than unconstrained SFT). Confirmed across 5 benchmarks (AIME 2024, AMC 2023, AIME 2025, Minerva, Beyond AIME) in Appendix F/Fig. 8 — same qualitative pattern; "OPD is consistently less affected by the rank-16 bottleneck than SFT."
- **Gap:** RLVR was NOT run through the rank-16 functional-sufficiency test (Fig. 6/8 only compare OPD vs SFT panels) — so we don't know if RLVR's own early-contracted subspace is similarly sufficient. Not claimed either way.

Implication box (p.8, verbatim): **"OPD subspace locking identifies an early-emerging update channel that is both persistent and sufficient for training. Future OPD implementations should therefore monitor and exploit this channel, rather than treating low-dimensionality as a post-hoc spectral statistic only."**

## Section 5 — RQ3: what controls the lock (directly relevant to token/objective sparsification question)

Three perturbations tested, holding the OPD pipeline otherwise fixed (Qwen3-8B/Qwen3-32B, dapo-math-17k, 300 steps; primary diagnostic = stable rank, auxiliary = Frobenius norm + Hill tail):

1. **Token sparsification** (top-KL retention or random retention, ρ ∈ {0.25, 0.50}): Fig. 7a / Fig. 9a — **"all sparsified variants closely track the OPD stable-rank trajectory. Even random 25% retention changes update scale more than spectral shape, indicating that the low-rank profile is not localized to a small set of high-KL tokens."** → **Confirms the plan's premise: token sparsification in this paper does NOT change spectral shape, only update magnitude/scale.**
2. **Off-policy rollouts** (teacher-generated rollouts instead of student-generated, ε=1.0 clip effectively inactive): Fig. 7b — stable-rank trajectory "nearly unchanged"; off-policy produces "a modestly larger update norm" but rank stays matched to on-policy OPD.
3. **Objective mixing** (interpolate OPD teacher-correction signal with RLVR advantage, `A^(α) = α·A_OPD + (1-α)·A_RLVR`, Eq. 34, α ∈ {1(pure OPD), 0.75, 0.5, 0.25, 0.05, 0.01, 0(pure RLVR)}): Fig. 7c / Fig. 9c — **clear split**: OPD-dominant mixtures (α≥0.5-ish) retain the OPD-like stable-rank trajectory; as the OPD component weakens (small α), "the trajectory no longer follows the baseline and enters a distinct spectral regime" (visibly flat/lower stable-rank curve, more RLVR-like).

Mechanistic explanation (Eq. 12–14): token gradients are sums `g_OPD = Σ_t J_t^T δ_t`; masking (token sparsification) primarily rescales the second moment (`E[g̃g̃^T] ≈ c·E[g_OPD g_OPD^T] + noise`) — i.e. preserves leading spectral directions. Objective mixing changes the covariance source itself (`Σ_α ≈ α²Σ_OPD + (1-α)²Σ_RLVR + α(1-α)Σ_cross`), which can change the dominant directions.

Implication box (p.8, verbatim): **"Subspace locking is objective-sensitive rather than runtime-induced. Effective OPD recipes should regulate objective-induced update geometry, not only token coverage or rollout generation."**

## Model scale actually used (Appendix C, Tables 2–7)

- **Shared backbone across all main trajectory/locking analyses (Sections 3–5, Figs 2–9): Qwen3-8B student, Qwen3-32B teacher.** This is the ONLY configuration used for the deep geometric trajectory diagnostics (stable rank curves, subspace-emergence, rank-16 projection, all three Section-5 control perturbations).
- **Robustness/variant runs (Table 1, Table 6) — used only for the bf16-sparsity check, not the full trajectory/locking analysis:** Qwen3-4B student w/ teachers {Qwen3-8B, Qwen3-14B, Qwen3-32B}; Qwen3-14B student w/ Qwen3-32B teacher; Qwen3-8B student w/ Qwen3-30B-A3B MoE teacher; code-domain variant (DeepCoder data / LiveCodeBench v5 eval); multi-seed. Sparsity stays in 48.6%–57.1% across all of these, supporting robustness of the SFT<OPD<RLVR *ordering*, but **the subspace-locking / rank-16-sufficiency claims themselves are never re-run outside the 8B/32B pair.**
- No model smaller than Qwen3-4B appears anywhere. **No 1.5B-scale results exist in this paper.**
- Training data: dapo-math-17k (math domain); DeepCoder for the code-domain variant. SFT anchor trained on Dolci-Think SFT data (Olmo 3). Evaluation: AIME 2024 avg@8 primary; AMC 2023, AIME 2025, Minerva, Beyond AIME as secondary (Appendix F); LiveCodeBench v5 for code variant.
- Hardware: H200 80GB GPUs, bf16 training, fp32 gradient accumulation/attention softmax.

## Whether hidden-state objectives were tested

**No.** Confirmed by direct reading of Appendix D ("Objectives and Three-Gate Extension"), which formalizes SFT (Eq. 16), RLVR (Eq. 17–18), and OPD (Eq. 19–20) — the OPD objective is strictly the token-level KL/log-prob-gap form; there is no hidden-state MSE, cosine, Gram, or any representation-space term anywhere in the objectives, diagnostics, or discussion. The "Discussion & Conclusion" (§6) frames future work as "geometry control" over token selection, rollout policy, and objective composition — it does not mention representation-level distillation (OPRD) at all, and does not cite the OPRD paper.

## Whether token sparsification changed spectral shape

**No — this is an explicit, load-bearing negative result.** Both top-KL and random token retention (25%/50%) preserve the OPD stable-rank trajectory almost exactly (Fig. 7a, Fig. 9a); the authors explicitly attribute the small residual differences to **update-scale/Frobenius-norm shifts, not spectral shape**. Only *objective composition* (mixing in RLVR) changes the spectral trajectory/shape. This is the single most reusable empirical fact for the plan.

## Stated limitations (Limitations section, p.9, verbatim)

1. "Our analysis focuses on controlled Qwen3-family reasoning settings. While this design isolates the effect of training paradigms, the observed geometry may vary across other model families, modalities, and task distributions."
2. "Our diagnostics characterize parameter-space trajectories from stored checkpoints. The Three-Gate account and covariance analysis should therefore be viewed as mechanistic explanations consistent with the evidence, rather than complete causal or formal theories of on-policy distillation (OPD) optimization."

## Unstated limitations / adversarial notes (mine, not the authors')

1. **Spectral/locking diagnostics are attention-weights-only** (36 layers × 4 attention weight types = 144 matrices, Appendix G.1 Table 8) — MLP/FFN weight matrices are excluded from the stable-rank / subspace-locking trajectory analysis entirely. If OPRD's hidden-state gradient acts differently on MLP vs attention output projections (plausible, since MLP is the dominant residual-stream writer in most transformer blocks), reusing "the same diagnostic suite" on OPRD (as the plan proposes) without also covering MLP matrices could miss exactly the layers where a hidden-state objective's marginal effect would show up.
2. **The deep trajectory/locking/rank-16-sufficiency results were only ever run at Qwen3-8B (student) / Qwen3-32B (teacher).** The plan's "OPD Geometry used 8B for clean SVD, but the diagnostics themselves are scale-agnostic" (plan file, Hybrid Scale Strategy) is the plan's own assumption — this paper provides zero evidence that the trajectory/subspace-locking findings hold at 4B, let alone the plan's default 1.5B. Sparsity-diagnostic robustness (Table 1) covers 4B/14B/MoE, but that is a single-scalar endpoint check, not the trajectory/locking machinery. Treat 1.5B geometry results as unvalidated extrapolation from the outset, not just a fallback if curves look noisy.
3. **Principal-angle diagnostics require CPU float64 SVD** because "GPU SVD can produce unstable angles for near-identical matrices" (p.12) — a real compute/engineering cost the plan's Phase A3/D budget doesn't mention; at 1.5B×several variants×many checkpoints this could be a nontrivial CPU-SVD bottleneck.
4. **RLVR was excluded from the rank-16 functional-sufficiency test** (Fig. 6/8 show only OPD vs SFT) — so "OPD's early channel is uniquely sufficient" is only benchmarked against SFT, not against RLVR's own (also low-rank, but late-contracting) channel. A skeptic could ask whether RLVR's channel is equally sufficient once it locks — paper doesn't say.
5. OPD's *final* stable rank converges close to RLVR's (~20–27 vs ~20) — the paper's novelty claim rests entirely on trajectory/timing (early lock, Fig. 5), not on a uniquely low endpoint rank. Any plan experiment claiming "OPD/OPRD preserves vs. escapes the lock" needs to specify whether it means final-rank comparison or early-subspace-similarity comparison — these are different claims and the paper is careful to use the latter.
6. Mechanistic account (Three-Gate, covariance decomposition, Eq. 12–14) is explicitly flagged by the authors themselves as "consistent with the evidence" rather than a proven causal theory — so the plan should not treat Eq. 13/14's "masking only rescales second moment" as a theorem transferable to OPRD's very different gradient structure (φ_t' from backprop through a hidden-state MSE term, not from a token log-prob gap) without new derivation/measurement.
7. Objective-mixing experiment interpolates OPD with **RLVR**, not with a hidden-state objective — so "objective composition breaks the lock" is evidence only about OPD+RLVR mixing. Extrapolating this to "OPD+OPRD mixing will also break/reshape the lock" (which is exactly what plan Phase A3/D hypothesizes) is a reasonable-sounding but *untested* analogy; the paper gives a mechanistic reason to expect it (different token-coefficient/covariance structure) but provides no direct evidence for a representation-space additive term.

## Direct relevance to the two draft plan files

`relational_oprd_experiments_7e321945.plan.md` explicitly cites this paper (`on-the-geometry-of-on-policy-distillation`) as the anchor for Phase A3 (geometry baseline at 1.5B: stable rank, Frobenius norm, early-to-final top-16 subspace similarity, rank-constrained training) and Phase D (geometry confirmation, decisive tests about locked-subspace freezing/orthogonal projection, and whether position weighting changes spectral shape "unlike token sparsification in OPD Geometry"). `oprd-literature-review.md` §"Geometry of OPRD" reuses the same diagnostic list almost verbatim (update sparsity, principal-angle rotation, NSS, update-mask overlap, stable rank, Frobenius norm, Hill tail, subspace similarity, rank-constrained training) and explicitly flags the open question of whether OPRD's hidden-state gradients (which see LM-head-null directions) will preserve or escape the OPD lock.

**This paper is a faithful, direct methodological precedent for Phase A3/D** — its diagnostic suite, equations, and rank-16 projection protocol can be copied almost exactly. Its core finding (early, persistent, functionally-sufficient low-rank channel for OPD; robust to token/rollout perturbation but broken by objective mixing) is real and load-bearing, not just a claim — it survives multiple controls (Frobenius norm ruling out trivial small-update artifact, Hill tail cross-check, 5-benchmark replication of functional sufficiency). But **none of it has been tested with a hidden-state objective or at anywhere near 1.5B scale** — so Phase A3/D is a genuine, non-redundant extension, not a re-run of existing results. The plan should (a) explicitly budget for CPU-SVD cost, (b) extend the 144-matrix diagnostic set to include MLP weights given OPRD's likely different locus of effect, (c) treat 1.5B as unvalidated rather than as a cheap default, and (d) recognize that "objective mixing breaks the lock" is currently only evidenced for OPD+RLVR, not OPD+OPRD — so Phase A3's OPD+OPRD comparison is testing a real open question, not confirming an established result.
