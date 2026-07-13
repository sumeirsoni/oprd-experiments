# Tier 0 Execution Plan — overnight kill-switch measurements (M3 Max, 36GB)

**Executor:** fresh Claude session running autonomously overnight. This file is the contract — execute it top to bottom; do not re-litigate scope.
**Context:** `oprd-adversarial-review.md` §5 Tier 0 (avenues 1–4), evidence in `evidence/`. These are *measurements, no training* — they decide which Tier 1–2 training arms are worth the partner's UMD cluster allocation.
**Hard constraints:** everything runs locally on this Mac (Apple M3 Max, 36GB unified). No cluster access. Partner runs Tiers 1–3 later from Slurm scripts we produce (D2).

## Goal

Produce `tier0-results.md` containing four measurements, each ending in an explicit **verdict line** (go / no-go / ambiguous + what it implies for Tier 1–2 arms), plus Slurm-ready training scripts for the partner. A negative or ambiguous result is a valid deliverable; a crashed measurement is not — skip-and-continue, never block the night on one arm.

## Non-goals

- No training runs of any kind (M4's ~30 gradient steps are probes, not training).
- No causal activation-patching (flip-rate) sweep — correlational M3 only; patching is a named follow-up.
- No wiki edits, no changes to `oprd-adversarial-review.md`.

## Models and data (verify exact HF repo IDs before downloading — search HF, do not guess)

| Role | Model (intended) | Notes |
|---|---|---|
| Same-arch pair | JustRL-1.5B (teacher) / DeepSeek-R1-Distill-Qwen-1.5B (student) | OPRD paper's flagship pair; both Qwen2.5-1.5B-based |
| Cross-width pair | Qwen3-4B (teacher) / Qwen3-1.7B-Base (student) | OPRD-Bridge Table 4 setup |
| Cross-tokenizer (stretch, only if ahead of schedule) | microsoft/Phi-4-mini-reasoning (teacher) / Qwen3-1.7B-Base | OPRD-Bridge Table 6 setup |
| Smoke model | Qwen2.5-0.5B-Instruct | gate only |

- Prompts: DAPO-Math-17k (HF) — sample 150 prompts, fixed seed 0, saved to `tier0/prompts.jsonl` so every measurement uses identical prompts.
- AIME24/AIME25 problem sets for M4's power analysis (counts + pass-rate stats only; no eval run).
- If a checkpoint is gated/unavailable, substitute the nearest same-size open checkpoint, and record the substitution prominently in results (it weakens comparability to OPRD's tables).

## Environment

- `uv venv` in `tier0/`; PyTorch (MPS) + transformers + numpy/scipy. Use **mlx-lm for generation** (faster on M-series), PyTorch for hidden-state extraction and M4 backward passes. Check current mlx-lm API via context7/docs — do not write generation code from memory.
- All artifacts under `tier0/`: `rollouts/`, `hidden/`, `results/`, `logs/`. Hidden states: fp16 memmap `.npy` per (model, layer-band), **never** hold a full all-layer pool in RAM (~170MB per 2000-tok rollout at 1.5B; budget ≤20GB disk per model — subsample layers if needed: every 2nd layer is acceptable, record it).

## Phase 0 — Smoke gate (blocks everything; ~30–45 min)

Run the ENTIRE pipeline end-to-end on Qwen2.5-0.5B with 10 rollouts × 256 tokens: generation → extraction → M1 → M2 → M3 → M4 → results-md rendering. Every measurement must produce its output file and verdict line on toy data. **Do not launch full-scale work until this passes.** Fix forward; if a component can't be fixed in ~30 min, mark that measurement `SKIPPED (smoke failure: <reason>)` and continue.

## Resource safety — MANDATORY (added after the 2026-07-11 02:05 kernel panic)

The first attempt panicked the Mac: mlx generation (Qwen3-4B) and MPS extraction (two 1.5B models) ran **concurrently**, wired unified memory to the point that watchdogd couldn't check in for 91s, and the hardware watchdog hard-reset the machine (`panic-base+socd-2026-07-11-020542`). Rules:

1. **One GPU job at a time. Ever.** No mlx generation while a PyTorch/MPS job runs, and vice versa. Wait for each process to fully exit (memory released) before launching the next.
2. **Start the watchdog first:** `nohup tier0/scripts/guard.sh >/dev/null 2>&1 &` before any model job. It SIGTERMs tier0 jobs at critical memory pressure or <8% free — a killed measurement is recoverable (checkpointed), a panicked Mac is not.
3. **Allocator caps are in the scripts** (do not remove): `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7` in extract/M4, `mx.set_memory_limit(20GB)` in generation. Overload now raises a catchable OOM → follow the existing OOM ladder (halve batch → subsample layers → reduce N).
4. Also export `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7` in the shell that launches any ad-hoc torch work.
5. Launch every long-running job under `caffeinate -is <cmd>` — the panic log shows the machine slept mid-run last night; sleep/wake cycles during GPU work add risk and stall the timeline.
6. Other user workloads may coexist (browser streams, audio capture) but **no other local ML/GPU jobs** while tier0 runs.

## Phase 1 — Generation + extraction (STRICTLY SEQUENTIAL — see Resource safety)

Run one job at a time, in this order (a killed/OOM'd job resumes from its checkpoint):

1. Qwen3-4B: generate 100 rollouts × ≤2000 tokens on the shared prompts (~2–3 hr).
2. R1-distill-1.5B (student): 100 rollouts × ≤2000 tokens (~1 hr).
3. Qwen3-1.7B-Base (cross-width student): 100 rollouts × ≤2000 tokens.
4. Extraction, one pair at a time, only after all generation processes have exited: teacher + student **prefill** over the student's rollouts, capture hidden states (all layers or every-2nd), stream to memmap.
5. Compute teacher/student per-token top-1 agreement and per-position cosine while extracting (cheap, feeds M3).

Sequential Phase 1 costs ~1–2 hr of wall-clock vs the pipelined version; the timeline table absorbs this by cutting the cross-tokenizer stretch goal by default.

Token pool target: ≥150K supervised-region tokens per pair (last-2000-style mask). If generation is slower than expected, floor is 60 rollouts / 100K tokens — record actual N.

## Phase 2 — Measurements

### M1 — Where does neighbor identity live: W_head row space vs null space? (kill-switch for identity-only arms)

1. SVD of each model's LM head `W_head` (full, float64, CPU — small matrix). Define row-space projector `P_row` (top-r right singular dirs; r at 90% spectral energy, also report r ∈ {64, 256}) and null projector `P_null = I − P_row`.
2. For layer ℓ in a swept band (at minimum: 25%, 50%, 75% depth, final): compute teacher–student **mutual-kNN agreement** (k ∈ {5, 10, 30}) over pooled rollout tokens, on hidden states projected onto: full / top-r PCA / `P_row` / `P_null`. Tile similarity matmuls (≤8K-token blocks); never materialize N×N.
3. Eval-size sweep: N ∈ {1K, 10K, 50K, 100K} tokens (tests Plato's-Cave fragility directly).
4. Both pairs (same-arch and cross-width; cross-width uses each model's own head for its own states).

**Verdict rule:** if mKNN(`P_row`) ≥ ~90% of mKNN(full) across layers/k — neighbor identity is output-readable → **no-go for identity-only Tier 1/2 arms** (OPD already sees the signal). If mKNN(`P_null`) carries a substantial share — **go**, and name the layers where the null-space signal peaks (those are the layers Tier 1 should supervise). Report the fragility slope vs N either way.

### M2 — Which relational property survives calibration: local distance vs identity vs global?

On frozen pairs, over the same pools: (a) global-distance/RKD-style alignment, (b) identity-only mutual-kNN, (c) local-window top-k distance agreement (teacher's top-k neighbors, compare distance sub-vectors, Spearman + normalized L2). Each with a **permutation-null calibration** (≥100 shuffles of token correspondence; report calibrated effect = raw − null mean, flag if CI overlaps 0). Note honestly in results: token sequences violate exchangeability (ARH's own caveat) — use block permutations (shuffle whole rollouts, not tokens) as the primary null.

**Verdict rule:** ranks the three Phase-B/E loss families for Tier 1–2. If (c) local-distance dominates (b) identity — the plan's identity-only arm demotes to a control and LN-Dist becomes the lead relational arm.

### M3 — Functional vs geometric alignment (correlational only)

Per generation token: (a) geometric alignment = PCA-cosine between mapped teacher/student states; (b) functional alignment = KL between teacher and student next-token distributions, plus logit-lens KL at the M1 band layers; (c) answer correctness per rollout (extract boxed answers; grade against dataset answers where available, else teacher-student answer agreement — record which). Correlate (a) and (b) with correctness and with position; compare pre-decision vs generation-region tokens (2605.23315's split).

**Verdict rule:** if geometric alignment doesn't track correctness where functional does → every Tier 1–2 geometric arm needs the functional-bridge variant added; if both track → causal-inertness threat is weakened, say so.

### M4 — Gradient variance probe + power analysis

1. ~30 fwd+bwd steps on the 1.5B student (PyTorch MPS, batch=1 × 2000 tok, no optimizer step): compute OPRD-MSE gradient and OPD-style sampled-token gradient on the same rollouts; report cross-step/cross-rollout gradient cosine-similarity and variance for each objective (tests Theorem 1 as actually used).
2. Power analysis (pure stats): given AIME24/25 sizes (~30 problems) and Avg@16 binomial variance at plausible pass rates (10–50%), compute seeds required to resolve 1-pt and 2-pt effects at 80% power. Output a seeds-vs-effect-size table.

**Verdict rule:** the seeds table goes verbatim into the partner handoff — it sets the seed budget for every Tier 1–2 arm. If OPRD's gradient variance is *not* materially lower than OPD's across steps, flag that Theorem 1's practical relevance is weakened (affects A1/C framing).

## Phase 3 — Deliverables

- **D1 `tier0-results.md`:** per measurement — setup actually run (models, N, substitutions), key numbers/tables, plots optional, and the bolded verdict line. Top section: a 4-row go/no-go table mapping each verdict to its Tier 1–2 consequence. List every skip/deviation.
- **D2 `cluster/` Slurm scripts** for the partner: `tier1_tau_sweep.sbatch`, `tier2_bridge_bakeoff.sbatch` (+ shared training harness stub + README with allocation-size estimates). Parameterized by the M1/M2 verdicts (e.g., supervised-layer band, which relational arms are enabled). These must be *syntactically valid and self-documenting*; they cannot be run locally, so mark them UNTESTED-ON-CLUSTER in the README.
- **D3:** one-paragraph summary at the very top of D1 written for the partner (plain language, verdict-first).

## Failure policy

- Each measurement is independent: wrap in its own script; on failure, log to `tier0/logs/`, mark SKIPPED with reason in D1, continue.
- Checkpoint everything (rollouts and memmaps are resumable); on OOM, halve batch/block size once, then subsample layers, then reduce N — in that order, recording each reduction.
- Budget guard: if Phase 1 isn't done by hour 5, cut the cross-tokenizer stretch goal and cap N at the 100K floor.
- Never delete `evidence/` or edit the review/plan files.

## Timeline (target ~9 hr wall-clock)

| Hour | Work |
|---|---|
| 0–1 | env + downloads + smoke gate |
| 1–5 | Phase 1 sequential: gen 4B → gen 1.5B → gen 1.7B → extract per pair |
| 5–7 | M1, M2 (CPU/memmap, may overlap each other but not any GPU job); M4 after extraction |
| 7–8.5 | M1/M2 cross-width; M3 |
| 8.5–9.5 | D1–D3 writing, cleanup, final verdict table |
