# Adversarial Deep-Research Review: Relational OPRD Plan

**Date:** 2026-07-10
**Method:** 54-agent research harness — 5 paper ingests, 8 adversarial prior-art sweeps, 30 refutation votes (10 claims × 3 lenses), 7 area-chair judgments, 9 proposed avenues, 1 completeness critic. Full evidence notes with quotes/URLs: `./evidence/` (digest-*.md, sweep-*.md; raw claim votes/judgments in `evidence/synth/`).

---

## 1. Headline answers

### Q1 (partner): "ARH disproves PRH, so OPRD-Bridge rests on a falsified premise"

**Overstated — refuted 3/3 votes, high confidence.** ARH (2602.14486, Gröger/Wen/Brbić, ICML 2026) is titled *"Revisiting"* and explicitly frames itself as a **refinement**, not a disproof: after width/depth null-calibration, global spectral convergence (CKA, Procrustes, RV) largely disappears, but local neighbor-identity convergence (mutual-kNN, CKNNA) **survives and increases with scale**. PRH survives in a narrower, topological form.

What *does* land: OPRD-Bridge's PCA+MSE is precisely a global-spectral/linear-subspace mechanism — the exact metric family ARH shows is confounded. And OPRD's own PRH evidence is weak on its face: one same-family, same-tokenizer model pair, 95% cosine at rank 8, **no null-calibration control of any kind**. So "the Bridge's motivating story is measurement-invalid" is a legitimate and publishable critique.

But four things block the stronger inference:

1. **ARH is measurement-only.** It never tests any alignment metric as a *training loss*. "This metric survives when measuring independently-trained models" → "train against it" is an analogy the plan makes, not a result ARH reports. Its permutation-null calibration also assumes sample exchangeability, which **breaks on token sequences** (ARH's own stated limitation) — the exact data regime OPRD needs.
2. **The surviving metric is itself fragile.** *Back into Plato's Cave* (2604.18572): mutual-kNN alignment degrades substantially as eval sets scale from ~1K to millions of samples, and the newer-models-align-more trend doesn't replicate. Calibrated effect sizes in ARH are small (~0.10–0.19).
3. **Live counterexample to "global-spectral is useless for training":** CKA-QAD / *Beyond Output Matching* (2606.05682) shows a plain global Gram/CKA loss **improving** reasoning recovery in RL-post-trained LLMs. Measurement-invalidity did not translate into training-uselessness. Full-Gram is a **live competitor, not a strawman control** — the plan currently treats it as a floor.
4. **The sharpest threat hits both sides equally.** *Convergence Without Understanding* (2605.23315, 16 models/8 families/800 problems): representational alignment is *higher* on problems models fail (CKA 0.897 vs 0.830), pre-decision representations align (0.875) while post-decision/generation representations diverge (0.274), and shared representational info is decodable (66%) but **causally near-inert (1.5–5.5% flip rate)** — weakest exactly in the generation-token region where OPRD applies its last-2000-token loss. This threatens pointwise MSE, PCA+MSE, *and* any relational replacement alike. No experiment in the plan currently tests causal relevance rather than assuming alignment transfers reasoning.

There is also a third theoretical position beyond the PRH/ARH binary — a linear-structure account (*Representation Alignment Rests on Linear Structure*, 2605.28870) — worth citing so the paper doesn't frame a two-horse race.

**Bottom line:** use ARH as motivation for a bridge ablation, never as entailment. Frame as: "OPRD-Bridge's justification is measurement-confounded (uncalibrated, single-pair); we test which relational property actually transfers." That's defensible. "PRH is disproven" will get shredded in review — including by the ARH authors.

### Q2 (partner): "Would a relational objective beat pointwise MSE for OPRD, especially cross-architecture?"

**As a general claim: refuted 3/3, high confidence. As a cross-architecture bet with the right design: genuinely open and worth taking — but not with the identity-only loss the plan specifies.**

Evidence against the general claim:

- **Base rate is bad.** In the standard controlled benchmark (CRD/RepDistiller, 5-seed CIFAR-100), the no-negatives relational family closest to the plan's mechanism — RKD, PKT, SP — **ties or loses to vanilla pointwise KD** (RKD −1.4 to −1.6 pts on 3/3 pairs). The only relational family that reliably wins (CRD/CoDIR) uses contrastive negatives, which the plan excludes.
- **PKT (ECCV 2018) is near-identical prior art** for the plan's soft-KL-over-similarity mechanism — and it only ties KD.
- **VanillaKD (NeurIPS 2023, 2305.15781):** much of the "fancy loss beats vanilla" literature is a tuning/scale artifact. Any relational win at 1.5B is presumptively an under-tuned-baseline artifact until a well-tuned MSE baseline is cleared.
- **TinyBERT** already solves cross-width hidden-state distillation with plain MSE + a learned linear map — a cheap baseline every bridge must beat.
- **Identity-only is the weakest family member.** Every documented no-negatives relational *win* (RKD distances/angles, LSP local distance distributions in GNNs) **kept distance information**. "Neighbor identity only, no distances" has zero positive training precedent anywhere. Ordinal-embedding theory (Arias-Castro, 1501.02861) says rank-only constraints pin down geometry only in dense-comparison regimes — the plan's sparse top-k/window is precisely the underdetermined case. And the loss family (softmax-over-similarity, SNNL/InfoNCE-adjacent) has documented dimensional-collapse stationary points (Jing et al., 2110.09348) and temperature pathologies.

Evidence for a narrower version:

- **PHF (2606.29340)** — the plan cites it but underweights it: a relational (transition-direction, orthogonal-invariant) hidden-state loss **already beats pointwise matching by +1.5–2.2 Avg@12 on on-policy Qwen3 1.7B/4B/8B self-distillation**. Relational-over-pointwise *can* win on-policy — on the temporal axis, not ARH's spatial-neighbor axis. PHF is a rival to beat, not background.
- **MiniLMv2 / MLDR-KD (2502.06189)**: relation-matrix losses dissolve width/head mismatch with no projector at all — structural (dimension-freeness) advantage cross-architecture, though neither shows accuracy superiority over tuned MSE+linear-map.
- **OPRD-Bridge's own bar is low.** Table 4: cross-arch shared-vocab, Bridge Avg@16 = 11.2 vs OPD top-1 = 15.3 (it *loses* on the primary metric; ties only Best@16). Cross-tokenizer it wins only because OPD structurally can't run.

**Bottom line:** the defensible thesis is not "relational > MSE." It is: *cross-architecture, where coordinates don't exist, which relational property — neighbor identity, local distances, or global spectrum — transfers capability?* That's a property-isolation study with a beatable published bar, and it's the strongest thing in the plan. Same-architecture, expect relational to tie a tuned MSE (hybrid MSE + small relational term is the only safe same-arch bet).

---

## 2. Claim verdict table (10 claims × 3 adversarial lenses)

| # | Claim | Verdict | Surviving (salvage) version |
|---|-------|---------|------------------------------|
| C1 | ARH disproves PRH → Bridge falsified | REFUTED (high) | ARH kills the *global-spectral* reading; PCA+MSE's justification is measurement-confounded, method not shown to fail |
| C2 | Relational > pointwise MSE, esp. cross-arch | REFUTED (high) | Specific mechanisms (negatives, attention relations, transition geometry, Gram) win in specific settings; identity-only unprecedented |
| C3 | CRD/CoDIR gains robust + transfer to on-policy LLM | REFUTED (high) | Real in offline vision/NLU; zero on-policy generative evidence; "negatives" undefined per-token on-policy |
| C4 | ARH methodologically sound + generalizes | REFUTED (high) | Sound *within* its regime (cross-modal, ~1K samples, i.i.d.); own paper disclaims sequential data; mKNN fragile at scale |
| C5 | ARH right ⇒ neighbor-identity bridge beats PCA+MSE | REFUTED (high) | ARH motivates the ablation; CKA-QAD is a direct counterexample to the inference |
| C6 | OPD prefix bias transfers to OPRD | REFUTED (med) | Genuinely open — mechanism (sampling variance) doesn't exist in OPRD; OPRD's own suffix-targeting default weakly suggests *reversal*. Best framing for A1 |
| C7 | RL middle-layer results transfer to OPD/OPRD | REFUTED (high) | Untested analogy; OPD geometry ≠ RLVR geometry; closest distillation datapoint (Flex-KD) says *final* layer, not middle |
| C8 | Soft token weighting is low-risk likely-positive | REFUTED (med/high) | Worth one ablation row; IW-OPD's own "ideal" weight *underperforms* uniform (−1.2) without stacked engineering; motivating mechanism absent in OPRD |
| C9 | Soft-kNN loss trains stably + preserves LM head | REFUTED (med/high) | Plausible with stabilizers, but collapse risk is real (InfoNCE-family) and LM-head decodability is completely untested — make both first-class diagnostics |
| C10 | Neighbor identity alone suffices to transfer reasoning | REFUTED (high) | No precedent; all relational wins kept distances; sparse-k regime is underdetermined; causal-inertness threat unaddressed |

Full per-lens reasoning with citations: `synth/claimVerdicts.json` in the evidence dir.

---

## 3. Experiment-by-experiment publishability verdicts

| Phase | Verdict | One-line |
|-------|---------|----------|
| A1 position probes | worth-doing-not-headline | Genuinely open (posbias paper is output-KL-only) but high confirmatory risk; only a *reversal* result is a contribution |
| A2 layer selectivity | worth-doing-not-headline (shrink) | Cut the 28-run freeze-C(k) scan; keep the one novel arm: **loss-supervision selectivity** (which layers enter L_OPRD) — zero prior art, attacks OPRD's all-layer default |
| A3 geometry @1.5B | worth-doing-not-headline (re-scope) | Descriptive battery is a re-run of 2606.07082; **1.5B principal angles are likely uninterpretable** (source needed CPU float64 SVD at 8B); novel slice = OPD-vs-OPRD subspace overlap |
| B relational objective | **needs-reframing** | "Relational beats MSE" is uninterpretable with the current control set; reframe as property isolation; best home is inside E |
| C soft weights | **needs-reframing** | Space is saturated (13+ OPD token-weighting papers); cut to one IW-OPD ablation row; retarget to "does OPRD escape the prefix-reliability profile" (ties to A1) |
| D geometry compose | **needs-reframing (merge into A3)** | Only the two interventions are novel (freeze-into / project-out of OPD's locked subspace); run once at 8B, include MLP matrices, gate on a confirmed B/C winner |
| E ARH-Bridge | **worth-doing — highest value in the plan** | No paper compares PCA+MSE vs a neighbor-identity bridge; OPRD-Bridge's Table-4 shortfall is an honest, beatable bar; but baselines must include the relational prior art |

### Key objections per phase (what a reviewer will actually say)

**A1.** (i) Confirmatory: OPD prefix bias is 3-papers-established (IW-OPD 2606.22600, PW-OPSD 2605.21606, blockwise drift gating 2606.24084) and OPRD already supervises the suffix by design — "OPRD uses the suffix" re-derives its own default. (ii) Not apples-to-apples unless both objectives run under *identical* prefix/middle/suffix masks, same rollouts/checkpoints/budget, reporting the same downstream benchmark with ≥3 seeds + CIs. Pre-register the falsifiable prediction: *OPRD reverses or flattens OPD's prefix bias because the sampling-variance mechanism (IW-OPD Prop. 1) doesn't exist for a deterministic MSE gradient*. That reversal, if found, is the real contribution — and it retroactively justifies OPRD's own design choice with a mechanism.

**A2.** Freeze-all-but-k C(k) is ill-posed for an objective that already updates all layers each step, costs ~28 runs, and the RL transfer prior is affirmatively contradicted (OPD geometry ≠ RLVR; Flex-KD 2507.10155 finds *final*-layer feature distillation sufficient/best). The loss-supervision-selectivity arm (all layers trainable, supervise only an early/middle/late/all band) is cheap (~5 runs), has no precedent anywhere, and is practice-changing if a 3-layer band ties all-layer MSE. Baseline against Flex-KD's final-layer finding, not the RL middle-layer prior.

**A3+D (merged).** Already run for OPD: the exact stable-rank/locking/rank-16 suite (2606.07082) *and* an independent rank-constrained OPD result (*Dense Supervision, Sparse Updates*, 2606.13657). "OPRD also locks a subspace" is the expected null (Gur-Ari 2018; *The Path Not Taken* 2511.08567 shows localization is model-conditioned and recipe-invariant). The only publishable deliverable: **principal angles between OPD's and OPRD's locked V₁₆ subspaces + cross-objective projection sufficiency** (constrain OPRD's gradient to OPD's early subspace — if accuracy drops to OPD level, OPRD's gains provably live outside OPD's channel, which would be the cleanest mechanistic validation of Theorem 2 available). Run at 8B/4B where SVD is interpretable, extend to MLP matrices (source suite was QKVO-only, yet hidden-state MSE gradients act heavily on MLP), and treat 1.5B curves as suggestive only.

**B.** Current controls (MSE-only, full-Gram, hybrid) cannot attribute a win: you can't tell whether *locality* or *identity-vs-distance* drove it. Two missing baselines make it interpretable: (i) a **distance-preserving local relational loss** (RKD/LSP-style over the same top-k window), (ii) **TinyBERT-style MSE + learned linear map**. PHF and MLDR-KD must be cited as rivals, and the full-Gram arm must be framed as a live competitor (CKA-QAD). Add representation stable-rank tracking and an LM-head-decodability check (does the student's own head still read the matched states?) as first-class diagnostics — collapse can hide while both loss and alignment metrics improve.

**C.** All four weighting schemes are pre-published in output-KL space; the mechanism motivating them (sampling-variance/trust-region drift) is the thing OPRD's Theorem 1 claims not to have. IW-OPD's own ablation shows the theoretically-ideal weight *hurts* (−1.2) and the gain comes from stacked stabilization engineering. One ablation row (IW-OPD cumulative unsigned discrepancy) on the winning objective; drop TIP/FiRe/PW-OPSD unless it clears.

**E.** Baseline gap is fatal as scoped: without RKD-distance, MLDR-KD relation-KL, MiniLMv2 projector-free, and a CKA/Gram arm, a win over PCA+MSE reads as "MLDR-KD reinvented for LLMs." Also: cross-arch 1.5B is a near-floor regime (student 2.8 Avg@16) — 1–4 pt deltas without seeds are uninterpretable; the cross-tokenizer setting (Phi-4-mini → Qwen3-1.7B) is the cleanest because the OPD baseline structurally can't run, making any competent bridge comparison the headline.

---

## 4. What the plan misses entirely (completeness critic)

1. **OPRD's own base result is unverified.** Single-run, no seeds/CIs anywhere, deltas as small as 0.5–0.6 pts, teacher/student are independently-trained checkpoints of the *same base model* (best-case alignment). Every phase downstream assumes this first digit is real. **The single most important pre-commit question: does any hidden-state objective beat a *well-tuned* OPD at 1.5B with ≥5 seeds and CIs?** Two papers give OPD-side confounds to control first: token-mass concentration (*Rethinking OPD*, 2604.13016) and the KL-agreement trap (2606.09471) — plus cheap output-space fixes (top-K support matching, 2603.25562) that recover large gains without touching hidden states.
2. **No power analysis.** AIME is ~30 problems; nobody has computed how many seeds resolve a 1–2 pt effect. This silently invalidates most proposed comparisons as scoped.
3. **Theorem 1 is invoked but never checked.** "Zero-variance gradient" holds *given fixed rollouts*; rollouts drift on-policy. Cross-step gradient variance is measurable and load-bearing for the A1/C reframings.
4. **Unsearched formal angles:** an information-theoretic (data-processing-inequality) formulation of the LM-head-bottleneck claim; optimization-dynamics comparison of MSE-regression vs reverse-KL (gradient SNR, conditioning) — the most likely *actual* reason one objective beats another; representation-target staleness under synchronous on-policy hidden-state matching (AsyncOPD 2606.24143 covers only rollout-policy staleness — this is a genuine unclaimed gap).
5. **Supporting ammunition the plan doesn't cite:** *Lost in Backpropagation: The LM Head is a Gradient Bottleneck* (2603.10145) independently supports OPRD's core premise — useful for the paper's motivation section.

---

## 5. Specific avenues (ranked, with kill-switches)

The most valuable reframe: **four of these are pure measurements, no training, hours on one GPU** — run them before committing any training compute.

### Tier 0 — no-training kill-switches (run first)

1. **Where does neighbor identity live: W_head row space or null space?** SVD the LM head once; compute teacher-student mutual-kNN per layer on rollout tokens after projecting hidden states onto full / top-r PCA / W_head row-space / W_head null-space; sweep k and eval-set size (1K→100K, testing the Plato's-Cave fragility directly). **Decisive:** if neighbor identity is fully recoverable from the row space, output-space OPD already sees it and B/E's premise dies; if it concentrates in the null space, you've pinpointed the causal niche for a hidden-state relational loss — that figure alone is a paper-worthy motivation.
2. **Local-distance vs identity vs global calibrated alignment** on the frozen pairs (same-arch 1.5B pair; Qwen3-4B/1.7B cross-width): width/depth-null-calibrated alignment via global-distance (RKD-style), identity-only (mKNN), and local-window top-k distance. Picks the Phase B/E loss *before* training; kills the identity-only arm early if local-distance dominates.
3. **Functional vs geometric alignment probe:** per generation token, correlate PCA-cosine bridge alignment and logit-lens KL (through a shared frozen readout / shared sub-vocab) with answer correctness and causal flip-rate (per 2605.23315's protocol). If geometric alignment doesn't track causality but functional does, every geometric bridge in the plan needs the functional variant; if both fail, the hidden-state-bridge premise itself is in doubt — a high-value negative.
4. **Power analysis + gradient-variance measurement:** seed-budget calculation for AIME/AIMO deltas; measure OPRD's actual cross-step gradient variance vs OPD's (tests Theorem 1 as used).

### Tier 1 — objective design (same-arch, cheap, decisive)

5. **Temperature-interpolated local relation loss** — the single best experiment for the partner's main question. One PKT-style loss: softmax-over-similarities to top-k window neighbors, matched teacher→student by KL, with temperature τ swept from high (preserves local distance distribution ≈ RKD/LSP) to low (peaked ≈ hard neighbor identity ≈ ARH). One knob spans the distance↔identity continuum; monotone improvement toward low τ confirms ARH's prescription, a peak at high τ vindicates distance-preservation, a flat plateau kills the distinction. Track representation stable-rank at every τ. (~24 runs at 500 steps, 1.5B.)
6. **Procrustes/CKA-invariant MSE ladder:** raw MSE → L2-norm MSE (OPRD default) → per-batch orthogonal-Procrustes-aligned MSE → CKA/log-CKA loss → whitened-CCA MSE (too-invariant control). Localizes exactly which nuisance invariance (rotation, scale, spectrum) carries the signal — the minimal ARH-motivated upgrade that never gambles on identity-only.

### Tier 2 — the flagship: cross-architecture bridge bake-off (revised E, absorbing B)

7. **Six-arm bridge comparison** on OPRD-Bridge's exact Table 4/6 setups (Qwen3-4B→1.7B-Base shared-vocab; Phi-4-mini-reasoning→Qwen3-1.7B cross-tokenizer), ≥3 seeds: PCA+MSE (paper recipe) / MSE+learned-linear-map (TinyBERT, no PCA) / **MiniLMv2-style projector-free token-token self-relation KL** / **LN-Dist: local-window top-k distance matching (Huber, dimension-free)** / soft neighbor-identity KL (the plan's arm) / CKA-Gram (live competitor per CKA-QAD). Diagnostics: calibrated mKNN + CKA control + downstream + Dist-n. This is the paper: nobody has run it, the published bar (11.2 vs 15.3) is honest and beatable, and whichever property wins is informative — including a pointwise win, which would be a clean empirical rebuttal of the ARH-as-training-prescription analogy.
8. **Anchor-fingerprint OT bridge for cross-tokenizer** (stretch arm): represent each token by its cosine vector to M shared anchor texts encoded by both frozen models; match fingerprint sets via Sinkhorn OT — no token alignment, no projector, no PCA. Offline retrieval sanity check first (do fingerprints recover known cross-model correspondences better than PCA?). Differentiates from ULD/MultiLevelOT (logit-space OT, 2412.14528) and DSKD-CMA (2603.22056) by operating on hidden-state kernel embeddings.

### Tier 3 — mechanistic (the deepest potential contribution)

9. **LM-head spectral decomposition of the OPRD gradient:** split the teacher-student residual into W_head row-space and null-space components; train full-MSE vs row-only vs null-only variants (3 arms, 500 steps, 3 seeds). Directly tests OPRD's Theorem 2 *causally*: null-only ≈ student floor while row-only ≈ full falsifies the LM-head-bottleneck story; the reverse confirms it. Nobody has done this, and it's the load-bearing claim of the whole OPRD program.
10. **OPD-vs-OPRD locked-subspace overlap + escape-alignment** (the merged A3/D deliverable): principal angles between the two V₁₆ subspaces; orthogonalize OPRD's residual subspace against OPD's and measure its energy in W_head's bottom vs top singular directions. If OPRD's escape component aligns with the head's null space, subspace-lock and LM-head-bottleneck unify into one mechanistic story — a genuinely novel result. 8B/4B, attention+MLP, float64 SVD.

---

## 6. Revised sequence (recommendation)

1. **Week 0 (no training):** avenues 1–4. Each can kill or redirect a whole phase for the cost of forward passes.
2. **Validate the first digit:** OPRD vs *well-tuned* OPD (top-K support matching, KAT-aware) at 1.5B, ≥5 seeds, CIs. If OPRD's edge dissolves, the program pivots to "when does hidden-state supervision help" — still publishable, differently framed.
3. **Flagship:** the six-arm cross-arch/cross-tokenizer bridge bake-off (Tier 2), with the τ-interpolated loss (Tier 1) as the same-arch companion that answers distance-vs-identity parametrically.
4. **Mechanistic chapter:** avenues 9–10 at 8B, gated on step 3 producing a real winner.
5. **Demoted to ablation rows/diagnostics:** A1 (controlled head-to-head, pre-registered reversal prediction), A2 (loss-supervision bands only), C (one IW row).

**What to publish:** one paper — *"Which representational structure transfers in cross-architecture on-policy distillation?"* — with the bridge bake-off as the headline, the W_head neighbor-localization measurement as motivation, the τ-sweep as the same-arch mechanism study, and geometry/position as supporting diagnostics. Running the plan's seven phases as peers would produce five footnote-grade results and no headline.

---

## 7. Load-bearing prior art the plan must engage (rivals, not citations)

| Paper | Why it matters |
|---|---|
| PHF, 2606.29340 | Relational on-policy hidden-state loss already beats pointwise (+1.5–2.2, Qwen3) — direct rival to B |
| OISD, 2605.29089 | On-policy intermediate-layer supervision ~1 wk before OPRD — weakens "first into representation space" framings |
| MLDR-KD, 2502.06189 | Softmax+KL over relation matrices for heterogeneous architectures — structurally near-identical to ARH-Bridge |
| MiniLMv2, 2012.15828 | Projector-free relation transfer dissolving width/head mismatch (2020) |
| TinyBERT, EMNLP 2020 | MSE + learned linear map solves cross-width — mandatory baseline |
| CKA-QAD, 2606.05682 | Global Gram loss helps reasoning recovery — kills the "Gram strawman" framing |
| VanillaKD, 2305.15781 | Fancy-loss wins are often tuning artifacts — demands a tuned-MSE baseline |
| Dense Supervision Sparse Updates, 2606.13657 | Rank-constrained OPD already exists — shrinks A3/D |
| The Path Not Taken, 2511.08567 | Update localization is model-conditioned — predicts "OPRD locks too" a priori |
| IW-OPD 2606.22600 / PW-OPSD 2605.21606 / DEAR 2606.22830 | Token/position weighting saturated; PW-OPSD: position beats entropy (AUROC 0.83 vs 0.57) |
| Flex-KD, 2507.10155 | Final-layer feature distillation sufficient — contradicts both all-layer default and middle-layer prior |
| Back into Plato's Cave, 2604.18572 | mKNN fragile at eval scale — sweep eval-set size in every alignment measurement |
| Convergence Without Understanding, 2605.23315 | Alignment causally near-inert in generation tokens — the threat every geometric objective must answer |
| Lost in Backpropagation, 2603.10145 | Independent support for the LM-head-bottleneck premise (pro-OPRD ammunition) |
| Low-Rank Clone 2505.12781 / LoRi 2606.05315 / Demystifying Low-Rank KD 2603.22355 | Low-rank projector distillation lineage predating OPRD-Bridge; PRH-independent theory exists |

**Caveats on this review:** the ARH digest was built from the abstract + five secondary sources (primary PDF fetch failed — verify section-level claims against the paper before citing); one ICLR 2025 hidden-state-matching paper (OpenReview IcVSKhVpKu) and the NRKD Neural Networks 2025 paper are unverified beyond search results; all agent-reported numbers should be spot-checked against primary PDFs before going into a manuscript.
