# Digest: "Is One Layer Enough? Training A Single Transformer Layer Can Match Full-Parameter RL Training"

arXiv 2607.01232v2 (2 Jul 2026). Zijian Zhang, Rizhen Hu, Athanasios Glentis, Dawei Li, Chung-Yiu Yau, Hongzhou Lin, Mingyi Hong. Univ. Minnesota / Peking Univ / Amazon (Lin's contribution stated as independent of Amazon).
Retrieved via: arxiv.org/html + arxiv.org/pdf full text (23 pp, including appendix). alphaxiv.org/abs/2607.01232 returned HTTP 403; arxiv.org/abs worked for abstract+authors only. Full paper successfully ingested from the HTML mirror and the PDF (all sections + all tables read directly, not just secondary summaries).

## 1. Core claim (verbatim, abstract)

> "training a single transformer layer can recover most of the gains achieved by full-parameter RL training, and in some cases even surpass it... high-contribution layers concentrate in the middle of the transformer stack, while layers near the input and output ends contribute substantially less."

This is a **descriptive, behavioral (task-score) finding about RLVR (RL with verifiable rewards), not a representation-geometry finding.** No CKA, no hidden-state similarity, no PRH/ARH-relevant metric appears anywhere in the paper.

## 2. C(k) methodology (§2.2, Eq. 3–4) — exact mechanics

- Model has L transformer decoder layers {θ_0,...,θ_{L-1}}, plus embedding θ_emb and LM head θ_head.
- **Single-layer training**: isolate one layer θ_k; update only θ_k via GRPO gradient descent, `θ_k ← θ_k − α∇_{θ_k} L_GRPO(θ)`, with `requires_grad=False` for every other parameter (embeddings, head, all other layers). The gradient is computed via full backprop through the whole network (so it "sees" all layers), but only θ_k's parameters are actually stepped. Repeated independently for every layer k ∈ {0,...,L−1} — i.e. one full RL training run per layer, not one combined run with per-layer diagnostics.
- **Layer contribution**:
  `C(k) = (S_k − S_base) / (S_full − S_base)`
  where S_k = in-domain benchmark avg of the layer-k-only-trained model, S_base = pretrained/base model score, S_full = standard full-parameter GRPO score. C=1.0 means single-layer training fully recovers full-parameter gain; C>1.0 means it surpasses it; C≈0 means the layer captures none of the RL improvement.
- A second "overall" variant `C_all(k)` substitutes the in-domain math score with an across-category average (Math+Code+Reasoning+Language); tracks C_math closely (Pearson r>0.6), used to argue the effect is general capability improvement, not objective-overfitting.
- **Fair-comparison protocol** (§3.1): full-parameter baseline LR is tuned per model over {1e-6,3e-6,5e-6,1e-5} and the *best* LR is used as the reference; the same LR is then applied to every single-layer run (so single-layer runs are not given a home-field-advantage LR). A learning-rate ablation (Appendix A.7, Table 12) reruns top-5/bottom-5 layers at 3x LR (1.5e-5 vs 5e-6) on Qwen3-1.7B and shows rankings don't flip (ΔC ≤0.02) — used to argue C(k) is an intrinsic property of the pretrained model, not an LR artifact.

## 3. Middle-layer concentration finding — numbers (§3, Tables 2–7, Figs. 1–6)

Tested: 7 models (2 families: Qwen3, Qwen2.5, incl. one DeepSeek-R1-distilled-into-Qwen checkpoint), 3 RL algorithms (GRPO, Dr. GRPO, GiGPO), 3 task domains (math reasoning, code generation as OOD probe, agentic decision-making/ALFWorld), 1.5B–8B params.

| Model | Algo/Task | Best layer (C) | Worst layer (C) | Layers ≥1.0 | Depth of best |
|---|---|---|---|---|---|
| Qwen3-1.7B-Base | GRPO/NuminaMath | L10, C=1.14 | L24, C=0.28 | 5/28 | ~36% |
| Qwen3-4B-Base | GRPO/NuminaMath | L16, C=1.06 | L2, C=0.66 | 4/36 | ~44% |
| Qwen3-8B-Base | GRPO/NuminaMath | L16, C=1.07 | L0, C=−0.51 | 4/36 | ~44% |
| Qwen2.5-Math-1.5B | Dr.GRPO/MATH | L14, C=1.01 | L23, C=0.42 | 2/28 | ~50% |
| Qwen2.5-1.5B-Instruct | GiGPO/ALFWorld | L14, C=1.02 | L24, C=0.25 | 1/8 tested | ~50% |
| Qwen2.5-3B-Instruct | GiGPO/ALFWorld | L18, C=1.01 | L4, C=0.17 | 1/11 tested | 50% |
| DeepSeek-Distilled-Qwen-7B | GRPO/Skywork-OR1 | L16, C=1.05 | L24, C=0.33 | 2/8 tested | ~57% |

Layers "around 40%–60% of network depth consistently achieve the highest contribution" across all seven models (§3 overview, direct quote). Best-to-worst spread is large (best layer often >2–4x the worst layer's C).

**Cross-dataset/task consistency** (§3.3, Fig. 3): per-layer C(k) rankings correlate Spearman ρ=0.76 (p<0.001) between NuminaMath-CoT and DeepScaleR (both math, same model, Qwen3-1.7B), and ρ=0.59 (p<0.001) between NuminaMath (math) and DeepCoder (code) — i.e. the same layers rank high even across task type, suggesting layer contribution is "an intrinsic property of the pretrained model" rather than data/task-specific.

**Weight-change-magnitude control** (§5.2, Fig. 9): under full-parameter training, ‖Δθ_k‖₂ is roughly uniform across layers (~0.5–0.8) despite the highly non-uniform C(k) — so "layer contribution reflects the *effectiveness* of a layer's parameter subspace... rather than the *magnitude* of parameter change." This is a directly reusable diagnostic (parameter-update-norm-by-layer) for the plan's geometry phase (A3/D), independent of C(k) itself.

## 4. Guided training strategies (§4) — the "practical payoff," narrower scope than the descriptive finding

Three strategies, **all evaluated only on the 3 Qwen3 base models, GRPO, NuminaMath-CoT/math benchmarks** (not on Qwen2.5, agentic, code, or the distilled model):

1. **Layer-adaptive LR** (§4.1): boost top-k layers to LR=1e-5, rest at base 5e-6. Best-10 boost: Qwen3-1.7B 53.70±0.40 vs 50.82±0.40 full baseline (+2.88, "43% of total RL gain"); Qwen3-4B 64.42±0.13 vs 62.97±0.78 (+1.45); Qwen3-8B 67.42±0.40 vs 66.43±0.40 (+0.99). Boosting the *worst* layers instead (control) degrades performance — used as evidence the effect is contribution-driven, not just "any extra LR helps."
2. **Layer-selective training** (§4.2): freeze all but best-k layers entirely. Qwen3-1.7B Only-B5 51.53±0.24 > full baseline 50.82±0.40; Qwen3-4B Only-B5 65.87±0.70 (+2.90, "27% of total RL gain"); Qwen3-8B Only-B10 69.11±0.10 (+2.68, "32% of total RL gain"). Training only the *worst* k layers is much worse (e.g. Only-W5 46.87 on 1.7B vs 50.82 full) — the clean asymmetry that supports the causal story.
3. **Heuristic middle-layer selection** (§4.3, profiling-free): just pick layers in [L/2−k/2, L/2+k/2] by *position*, no C(k) profiling needed. For k=5: layers 11–15 (1.7B), 15–19 (4B and 8B). Beats full-parameter baseline on all 3 scales (e.g. 4B: 64.80±0.30 vs 62.97±0.78 full, +1.83, "17% of total gain") though it underperforms the profiled Only-B5 selection — a cheap default.

**Diversity/majority-voting result** (§5.1, Fig. 8): top-7 high-contribution single-layer-trained models solve largely non-overlapping problem subsets (mean pairwise Jaccard similarity only 34.1% on OlympiadBench). Majority-voting their outputs reaches 33.6±0.91% vs best single layer-trained model 28.3±0.25% and full-parameter baseline 26.9±0.40%, and beats self-consistency voting over 7 samples from the *same* full-parameter model (31.3±1.11%) — i.e. structural diversity across layers beats sampling diversity. The paper explicitly flags this as "an analysis tool rather than a practical training strategy, since it requires independently training multiple layer-specific models" (§5.1).

## 5. Anything tested for distillation rather than RL?

**No.** Confirmed by direct read of all sections (Introduction, §2 Preliminaries/RLVR+GRPO, §3, §4, §5 Discussion, §6 Related Work, §7 Conclusion) — every experiment trains via RL (GRPO/Dr.GRPO/GiGPO) on verifiable-reward tasks. There is no SFT run, no KD run, no output-space or hidden-space distillation objective anywhere in the paper.

- DeepSeek-Distilled-Qwen-7B (one of the 7 models) *is itself a product of distillation* (distilled from DeepSeek-R1 into the Qwen architecture) — but in this paper it is only ever used as a base checkpoint that then undergoes further **RL** training (GRPO on Skywork-OR1); the paper studies RL's layer contribution on top of a distilled model, not distillation's own layer contribution. Framed by the authors purely as a generality check ("to further test generality beyond the Qwen3 and Qwen2.5 model families... a model distilled from DeepSeek-R1 into the Qwen architecture", §3.4).
- Related Work (§6) explicitly locates its contribution against SFT-era layer-importance work — LISA (Pan et al. 2024, random layer sampling during SFT), MISA (module-wise importance sampling for SFT), AdaGradSelect (gradient-guided layer selection for SFT), Nepal et al. 2025 (claims critical layers for math reasoning are set during pretraining and "invariant after post-training") — and states: **"These works primarily operate on inference-time analysis or the supervised finetuning setting. In contrast, our work focuses on the RL setting..."** This is the paper's own explicit RL-vs-everything-else framing; it does not claim its results transfer to SFT/KD/OPD/OPRD, and gives no evidence either way.

## 6. Stated scope limits / limitations (§7, verbatim)

> "Our work has several limitations. Our guided training strategies are validated only on mathematical reasoning; extending them to coding and agentic tasks is left for future work. Additionally, our layer contribution metric is defined with respect to a specific training configuration, and a deeper theoretical understanding of why middle layers are disproportionately important for RL adaptation remains an open question."

Decomposed:
1. **§4's practical strategies (adaptive LR, layer-selective training, middle-heuristic) are validated only on math**, despite the *descriptive* C(k) pattern (§3) being shown to generalize to code (DeepCoder, cross-task Spearman ρ=0.59) and agentic tasks (ALFWorld, §3.4) — code/agentic guided-training extensions are explicit future work, not done here.
2. **C(k) is configuration-dependent** (tied to a specific LR, batch size, KL coefficient, etc., per model) — partially defended by the Appendix A.7 LR-ablation but not fully resolved.
3. **No theoretical mechanism** for why middle layers dominate — an open question, not even hypothesized about mechanistically (no attention-pattern, gradient-flow, or representation-based explanation offered).

## 7. Unstated limitations (my read, adversarial)

- **Zero representation-geometry content.** No CKA, no hidden-state similarity/MSE, no subspace/rank analysis of *representations* (only of raw parameter-update magnitude, §5.2, and that's L2 norm not stable rank/SVD). The paper's "layer contribution" is entirely an output-behavior metric (task accuracy). It gives **no direct evidence** about whether middle layers are where hidden-state representations change most, only where RL-induced *task-score* improvement is most recoverable from isolated updates. Plan A2/A3 should not assume C(k)-style layer concentration implies anything about hidden-state MSE/CKA layer profiles — that mapping is unverified even within this paper's own RL setting, let alone for OPD/OPRD.
- **Parameter-update selectivity only**, never **loss-supervision selectivity**. This paper always freezes non-target layers' *parameters*; it never tests "all layers still trainable, but only some layers' hidden states are supervised by an extra loss term" (which is exactly the OPRD-relevant axis the draft plan's A2 wants to test — "distinguish parameter update selectivity vs loss supervision selectivity"). There is no precedent here for the latter; it's a genuine novel axis, not something this paper already answers.
- **Partial layer scans for 3/7 models** (Table 1: Qwen2.5-1.5B-Instruct, Qwen2.5-3B-Instruct, DeepSeek-Distilled-Qwen-7B trained on only a representative subset of layers, not a full scan) — the reported "best layer" for these three could be a local, not global, optimum; middle-concentration claim for those three rests on sparser sampling (as few as 8 layers probed for a 28-layer model).
- **§4 guided-strategy percentages ("43% of total RL gain" etc.) are somewhat fragile/cherry-pickable** — they're computed relative to already-small full-parameter gains (e.g. Qwen3-1.7B full RL gain over base is only 50.82−44.1≈6.7 points), so the same absolute Δ (+2.88) yields a large-looking percentage on a small base; the raw point-gains (~1–3 points on a 4-benchmark math average) are modest in absolute terms even though framed with large relative-gain language.
- **No comparison to standard parameter-efficient fine-tuning (LoRA, adapters) as a baseline** for the "guided training" strategies — the paper compares full RL vs single/selected-layer RL, but not vs. e.g. LoRA-RL, which is the more standard "train fewer parameters" alternative and would contextualize how special "one real layer" is versus any low-rank/parameter-efficient method.
- **All models are Qwen-family** (Qwen3, Qwen2.5, and a Qwen-architecture distillation target) — no cross-architecture-family generalization test (e.g. Llama, Gemma, Mistral) despite claiming a "structural property" of transformers generally.
- Majority-voting result (§5.1) is explicitly not a practical training method (requires training L separate single-layer models) — a nice diversity data point but not something to build an efficiency claim on.

## 8. Relevance to the draft plan (relational_oprd_experiments + oprd-literature-review)

The plan's **Phase A2** ("Mirror Is One Layer Enough?: freeze all but layer k, update only θ_k, compute C(k) vs full OPD... For OPRD, distinguish parameter update selectivity vs loss supervision selectivity") leans directly and heavily on this paper — it borrows the exact C(k) formula, the freeze-all-but-one-layer protocol, and the "middle-layer" hypothesis. This paper genuinely underwrites that transplant: the C(k) methodology is well-specified, reasonably robust (LR ablation, cross-dataset/task consistency checks), and — critically for the plan's novelty claim — **the paper itself explicitly restricts its claims to RL and never touches distillation**, so replicating C(k) for OPD/OPRD is not redundant with anything already published here.

Two things the plan should sharpen given this reading:
1. Its stated hypothesis "OPD middle-layer concentration exists but is flatter than RL" (Phase A2) has no supporting evidence from this paper one way or the other (this paper says nothing about OPD) — it's a pure guess and should be labeled as such, not implied to follow from RL literature.
2. Its "loss supervision selectivity" idea for OPRD (which layers enter L_OPRD while all layers remain trainable) has **no analog anywhere in this paper** — it's not a variant they tried, so the plan cannot cite this paper as precedent for that specific ablation; it should be framed as a new axis the plan is inventing, with correspondingly higher scrutiny/priority given scarce compute (per the "kill weak/redundant experiments" mandate). This is the more original — and more expensive to justify — half of Phase A2, and deserves a tighter minimum-viable version (e.g. one model, 2–3 layer-band configs) before scaling.

No connection to ARH/PRH/representation-geometry questions (Phase B/E) — this paper is silent on that axis entirely; it should not be cited as evidence for or against the relational-objective or ARH-Bridge hypotheses, only as a methodological precedent for the layer-selectivity diagnostic in Phase A2/D.
