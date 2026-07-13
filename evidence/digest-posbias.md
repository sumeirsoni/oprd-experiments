# Digest: "On the Position Bias of On-Policy Distillation" (arXiv 2606.22600v3, 26 Jun 2026)

Authors: Yan Xie¹*, Sijie Zhu¹*, Tiansheng Wen², Bo Chen¹†, Yifei Wang³† (*equal contrib, †corresponding)
¹Xidian University, ²Georgia Tech, ³Amazon AGI SF Lab. Preprint (not yet peer-reviewed venue-stamped).

Source: full main text (pp.1–10) + references + start of Appendix A read directly from the arXiv PDF
(ground-truthed, not paraphrase-of-paraphrase — the WebFetch text-summary tool's first pass on this
paper was noticeably corrupted, see "Corrections" section below). Appendices B–F (extra experiment
details, additional ablations, broader-impact/limitations subsection, ExOPD-combination results)
were NOT read in full — only the main body (§1–§7) and Appendix A.1 (proof of Prop. 1) are digested
below in detail.

## 0. Corrections to first-pass automated summaries (important for calibration)

The first two WebFetch calls (alphaxiv.org → 403; a second-pass "detailed extraction" prompt against
arxiv.org/pdf) produced a summary containing several fabricated specifics: invented section/equation
numbers ("Equations 8–12" for IW-OPD, "Section 5.3" ablations — these turned out to be directionally
right but the actual eq. numbers are 8/9/11/13/14/15/16/17, and ablations really are in §5.3, so that
one was lucky), an invented author list, and vague hand-waving on the "Table 10-12" claim (there is no
Table 10-12; the real result tables are Tables 1–4). The arxiv.org/html WebFetch pass was substantially
more accurate (correct authors, correct model names, correct headline number 6.9) and is confirmed
correct against the primary-source PDF read below. **Bottom line: everything in this digest below is
taken from directly reading the PDF pages, not from the earlier lossy summarizer passes.**

## 1. Core Claims (with section refs)

1. **Position bias exists in OPD, and it's a prefix>>suffix asymmetry, not a middle-vs-edges effect.**
   (§3.1) Controlled study: fix token supervision budget at 30% of response length, vary which segment
   is supervised. "Prefix-30" (first 30% of response tokens) matches or exceeds full-token standard OPD;
   "Suffix-30" (last 30%) "yields substantially lower rewards throughout training" / "barely learns
   anything" (abstract). Verbatim: *"with the same 30% token budget, training on the prefix part of each
   response matches or exceeds full token Standard OPD, whereas training on the suffix part fails to
   learn effectively."* (Fig. 1 caption). This is an output-reward-space finding (RL reward curves), not
   a benchmark-accuracy finding per se — Fig 1a plots training reward, not AIME score, for this specific
   probe.

2. **Teacher–student gap persists locally even after OPD "converges."** (§3.1, Fig. 3a/3b) Mean
   token-level reverse KL decreases during OPD training but "plateaus at a non-zero residual" — divergence
   only drops ~20% even after reward saturates. Interpreted as evidence OPD only optimizes the student
   within a small local trust region, not that it drives KL to ~0.

3. **Theoretical account: OPD is a finite local-update-budget allocation problem.** (§3.2, Prop. 1)
   Formalizing OPD's implicit trust region as `min_q D_KL(q||π_T) s.t. D_KL(q||π_θ) ≤ ρ`, the closed-form
   optimal reweighted policy is `q*_α(y) = π_θ(y)/Z_α(θ) · (π_T(y)/π_θ(y))^α`, i.e., proportional to the
   teacher/student likelihood ratio raised to a power α∈(0,1) set by the budget ρ. Because student-sampled
   prefix probability under the teacher `π_T(y_<t)` drops much faster than under the student `π_θ(y_<t)`
   as t grows (student rollouts drift off the teacher-preferred path), the ratio `r_θ(y_<t)^α` shrinks at
   later positions — which the paper claims *is* the mechanistic origin of the observed prefix bias. This
   is a clean, falsifiable derivation but it is a **single strong assumption stacked on a single strong
   assumption**: (a) OPD training is well-modeled as a fixed-KL-radius trust-region problem at all, and
   (b) the teacher-conditional-prefix-likelihood-decay pattern shown in Fig. 3c (one student/teacher pair,
   Qwen3-0.6B/4B-Instruct) generalizes. No falsification attempt (e.g., does the claim hold under
   different rollout temperatures / RL algorithms) is reported in the main text.

4. **IW-OPD (Importance-Weighted OPD) reweights token loss by a stable, cumulative, unsigned
   teacher–student discrepancy statistic** derived from (but heavily engineered away from) the ideal
   ratio-based weight of Prop. 1/2. Final practical formula (Eq. 17, §4.2-IV):
   `r̃_t^{IW-OPD} = 1 + γ·(1 − Σ_{k<t}|A_k^OPD| / Σ_{k<T}|A_k^OPD|)`, where
   `A_k^OPD = −(log π_θ(y_k|y_<k) − log π_T(y_k|y_<k))` is the standard per-token OPD advantage, γ=0.5
   default. This is an **interpolation with standard OPD** (γ controls how much extra weight goes to
   teacher-compatible prefixes) — IW-OPD reduces exactly to standard OPD when the extra weighting term
   is removed (γ=0), which the paper states explicitly as a design virtue ("requires no additional
   teacher evaluations beyond standard OPD," §1). The weight is applied as a stop-gradient multiplicative
   coefficient on the OPD policy-gradient advantage (Eq. 12–13): `A_t^{IW-OPD} = −sg[r̃_t]·(log π_θ(y_t|y_<t) − log π_T(y_t|y_<t))`.

   Getting from the "ideal" ratio weight to this practical form required four stabilization tricks
   (§4.2, ablated in Table 4), each independently justified but stacked:
   I. log-space scaling of the ratio (avoids exponential blowup over long sequences) → Eq. 14
   II. **sign correction**: take Σ|A_k| instead of Σ A_k, because positive-advantage tokens (where
      teacher assigns *higher* prob than student — happens on some tokens) partially cancel the
      accumulated negative discrepancy and creates local reversals/instability (Fig. 4b/4c) → Eq. 15
   III. within-sample [0,1] normalization of the cumulative discrepancy → Eq. 16
   IV. interpolation (blend) with standard OPD via γ → Eq. 17 (final form actually used)

5. **IW-OPD improves both convergence speed and final accuracy, with effect size scaling with
   teacher–student mismatch and shrinking student size** (§5.2, Tables 1–3):
   - Headline: **+6.9 AIME25 points at step 10** ("IW-OPD₁₀"), Qwen3-4B student / Qwen3-30B-A3B-Instruct
     teacher (small-overlap, cross-scale setting): OPD₁₀=42.4 → IW-OPD₁₀=49.3.
   - At full convergence, same pair: OPD=48.0 → IW-OPD=49.7 (+1.7 AIME25); HMMT25 OPD=55.3→57.1(+1.8).
   - Qwen3-1.7B student, same teacher: AIME25 OPD=28.7→IW-OPD=29.5 (+0.8); HMMT25 +0.9.
   - Qwen3-0.6B student, same teacher: AIME25 OPD=17.8→19.3 (+1.5); HMMT25 +1.4.
   - Large-overlap teacher (Qwen3-4B-Instruct-2507, same family as students), Table 2: gains are
     generally smaller at large student sizes and larger at small student sizes — 4B student AIME25
     +0.4, 0.6B student AIME25 +1.9.
   - Large-scale stress test (Table 3): Qwen3-235B-A22B-Instruct-2507 teacher → Qwen3-30B-A3B student:
     AIME24 +1.3, AIME25 +2.2, HMMT25 +2.1, HE+ +1.4, MBPP+ +2.4. So the effect **does replicate at much
     larger absolute scale** (30B student, 235B teacher), which is a real strength — not just a 0.6B–4B
     toy-scale artifact.
   - Relative-gain trend as student shrinks (fixed 4B teacher, §5.2 "IW-OPD benefits more as students
     scale down"): final-average improvement over OPD grows from +1.0 (4B student) → +1.2 (1.7B) → +1.9
     (0.6B) points, i.e. ~+1.8%, +3.2%, +9.5% relative. At step 10 the relative gains are larger still
     (+2.0%, +3.5%, +10.8%).
   - All numbers are mean@32 accuracy, averaged over three random seeds (§5.1) — no variance/error bars
     reported in the tables shown, which is a real gap for effect-size trust (six-plus benchmark columns
     × 3 teacher regimes × 3 student sizes, no stated significance test).

6. **Ablations (§5.3, Table 4, AIME25, Qwen3-0.6B student)** isolate three independent design axes vs a
   43.3 standard-OPD baseline:
   - Weighting *shape*: fixed-ratio amplification +0.4, linear decay +0.7, manual position curriculum
     +1.5, vs. the paper's trajectory-adaptive cumulative-share rule **+5.6**. Conclusion: adaptivity to
     each trajectory's own discrepancy trace matters far more than any fixed decreasing-with-position
     shape.
   - Signed vs unsigned discrepancy: signed Σ A_k +2.6 vs unsigned Σ|A_k| **+5.6**.
   - Weight source: "ideal" ratio weight alone **−1.2** (worse than plain OPD!), ideal+OPD blend +0.6,
     surrogate-only +2.9, full surrogate+OPD blend (their method) **+5.6**. This is notable: the
     theoretically "correct" weight from Prop. 2 actually hurts when used unblended, and most of the
     gain comes from engineering choices (unsigned accumulation + blending), not from the ratio-based
     derivation per se. The derivation is better read as motivation/justification-after-the-fact than
     as a tight predictor of what works.

## 2. Scope: output-space only, no hidden states

Confirmed by direct read of the full main text (§2 Preliminaries through §7 Discussion): **this paper
never touches hidden-state representations at all**. Every objective, diagnostic, and ablation operates
purely on token-level log-probabilities / KL divergence between teacher and student *output*
distributions (Eqs. 1–6 define πθ, πT, standard OPD loss D_KL(πθ||πT) all in output-probability space).
"Position bias" is diagnosed via output KL (Fig. 3a/3b) and via reward curves under position-restricted
output-KL supervision (Fig. 1a). There is no hidden-state MSE, no representation-space probe, no mention
of intermediate layers anywhere in the 10 pages of main text or in the references list (references are
all OPD/KD/RLVR/credit-assignment papers, none representation-geometry papers). **This means the plan's
assumption that this paper establishes an OPD prefix bias that OPRD "may or may not inherit" is exactly
right as a research gap — the paper gives zero direct evidence either way about hidden-state position
bias.** It only shows the *output*-KL / reward asymmetry.

## 3. Scale

- Students: Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-30B-A3B (30B total / ~3B active MoE).
- Teachers: Qwen3-4B-Instruct-2507 (same-family/large-overlap), Qwen3-30B-A3B-Instruct-2507
  (cross-scale/small-overlap), Qwen3-235B-A22B-Instruct-2507 (large-scale stress test).
- Training data: DeepMath (difficulty≥6, ~57K problems) for math, Eurus-RL-Code (~25K problems) for code.
- Eval: AIME 2024, AIME 2025, HMMT 2025 (math, mean@32); HumanEval+, MBPP+ (code).
- The core position-bias diagnostic (Fig. 1, prefix-30/suffix-30) uses only ONE pair: Qwen3-0.6B
  student / Qwen3-4B-Instruct-2507 teacher. This is the paper's single evidentiary basis for the
  headline asymmetry claim — it is not replicated across teacher/student pairs in the main text (the
  larger sweep in Tables 1-3 is about IW-OPD final performance, not about re-verifying the prefix/suffix
  asymmetry itself at other scales).

## 4. Effect sizes — summary table

| Setting | Metric | OPD | IW-OPD | Δ |
|---|---|---|---|---|
| Qwen3-4B student / 30B-A3B teacher, step 10 | AIME25 | 42.4 | 49.3 | **+6.9** |
| Qwen3-4B / 30B-A3B, converged | AIME25 | 48.0 | 49.7 | +1.7 |
| Qwen3-0.6B / 4B teacher, converged | AIME25 | 17.1 | 19.0 | +1.9 |
| Qwen3-30B-A3B / 235B-A22B teacher | AIME25 | 56.7 | 58.9 | +2.2 |
| Qwen3-30B-A3B / 235B-A22B teacher | HMMT25 | 38.4 | 40.5 | +2.1 |
| Ablation, cumulative-share vs fixed-shape (AIME25, 0.6B) | — | 43.3 base | 48.9 | +5.6 |

## 5. Limitations (stated + unstated)

**Stated (§7 Discussion, closest thing to a limitations section — no dedicated "Limitations" heading in
main text; may exist in an unread appendix):**
- "Such bias is not unique to OPD. It may also appear in many methods involving two autoregressive
  sequence models... The issue is especially visible in OPD since the teacher–student distribution gap
  is not known in advance, so we cannot predefine where their trajectories remain compatible." — i.e.,
  the authors frame position bias as a fairly general phenomenon of on-policy two-model supervision, not
  OPD-specific, but do not test it in any other setting (no RL-with-KL-penalty ablation, no other
  distillation family).

**Unstated / my assessment:**
- No hidden-state analysis whatsoever (see §2) — cannot directly speak to whether OPRD (hidden-space
  matching) has the same or different position profile. This is a genuine open question the draft plan's
  A1 correctly identifies as needing new experiments, not literature lookup.
- No variance/CI reported across the stated 3 seeds in the main result tables — many Δs are ≤1–2 points
  (e.g., +0.4, +0.8, +1.0) which is within plausible seed noise for RL/RLVR-style training; only the
  step-10 +6.9 and the ablation deltas (+5.6, etc.) are large enough to likely be robust to seed noise.
- The mechanistic story (Prop. 1/2, trust-region projection) is elegant but per Table 4 ablations, the
  "ideal" theoretically-motivated weight alone actually *underperforms* plain OPD (−1.2), and most of the
  empirical win comes from an unsigned-accumulation + OPD-blend engineering combo that is only loosely
  connected to the derivation. So: treat the theory as good narrative/motivation, not as a tight
  causal explanation validated by the ablations.
- Only one prefix/suffix probe pair (0.6B/4B) backs the core asymmetry claim; not replicated at other
  scales in the main text.
- Everything is Qwen3-family, math+code RLVR-style domains; no cross-tokenizer, no non-reasoning domain,
  no vision/multimodal.
- No comparison to non-uniform-weighting alternatives from prior literature beyond the internal ablation
  variants (e.g., doesn't compare against TIP/FiRe/SelecTKD-style token weighting mentioned in the
  user's own literature review as related work — those are cited in §6 Related Work as related but not
  empirically benchmarked against IW-OPD).
- IW-OPD is a pure re-weighting of the *existing* OPD gradient signal — it adds no new information
  source (still just teacher/student output log-probs), so it cannot by construction fix the underlying
  cause of drift (student rollout leaving teacher support); it only reallocates a fixed budget toward
  currently-reliable tokens. This matters for the plan: IW-OPD is a control/reference method, not a
  ceiling — it doesn't test whether hidden states carry *reliable* late-position signal (OPRD's
  potential edge) versus just deprioritizing late tokens like IW-OPD does in output space.

## 6. Relevance to the draft plan

Directly cited as anchor source A1 hypothesis-motivator in
`relational_oprd_experiments_7e321945.plan.md` §Phase A1 and referenced in
`oprd-literature-review.md` ("Position-Aware OPRD" section, IW-OPD weighting candidate). The plan
correctly treats this paper as evidence *for OPD only* and explicitly flags that OPRD's position profile
is an open empirical question — this is fully justified: the paper provides **zero** hidden-state
evidence, so A1 (probing whether OPRD inherits/escapes OPD's prefix bias) is a genuinely new experiment,
not a redundant replication. The IW-OPD weighting formula (Eq. 17, cumulative unsigned |A_k| share) is a
concrete, cheap-to-implement candidate for Phase C's "IW-OPD-style cumulative prefix discrepancy" weight
— the plan's phrasing ("cumulative prefix discrepancy") matches the paper's mechanism closely and can be
reused nearly verbatim, with the caveat noted in the plan (§Phase C hypothesis) that this weight was
derived and validated for *output*-space OPD; whether it should be computed from output KL or a
hidden-space analogue when applied to OPRD is an open design choice the plan should make explicit rather
than assume. One adversarial note for the plan: the ablation table here (Table 4) shows the
"theoretically ideal" ratio-weight underperforms and most of the win is engineering (sign-correction +
blending) — the plan should not assume A1/Phase-C's borrowed IW-OPD machinery will transfer its
effect size to OPRD/hidden-space objectives just because the derivation is clean; the empirical result
is decoupled from the theory here, so it needs its own ablation once ported, not a citation-based
assumption of transfer.
