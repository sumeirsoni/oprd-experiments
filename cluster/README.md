# OPRD Tier 1–2 cluster scripts

**STATUS: harness implemented and smoke-tested locally; cluster paths untested.**

What has actually been verified (2026-07-13, Apple MPS, Qwen2.5-0.5B pair):
- **all 7 loss arms** ran end-to-end (generation → teacher prefill → 2 training steps →
  diagnostics) with finite, sane losses;
- **rollout refresh** (regeneration from the current policy mid-training) and the
  **AIME eval plumbing** (dataset download, Avg@N generation, boxed-answer grading).

What is **UNTESTED** and needs a 1-task cluster smoke before submitting arrays:
- the vLLM generation backend (local smoke used the `hf` fallback);
- CUDA memory behavior at 1.5–4B scale and real `--batch-tokens 16384`;
- the sbatch files themselves (never submitted to a real queue — fix partition/account/modules).

Cluster smoke command (one GPU, ~5 min):
```bash
python train_harness.py --loss pkt_tau --tau 0.1 --steps 5 --n-prompts 8 \
  --teacher hbx/JustRL-DeepSeek-1.5B --student deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
  --prompts BytedTsinghua-SIA/DAPO-Math-17k --supervised-layer-fracs 0.25,0.5 \
  --batch-tokens 2048 --rollout-refresh 0 --seed 0 --out runs/smoke
```

Known limitation: **cross-tokenizer pairs are not implemented** (harness exits with a clear
error) — Tier 2's phi4mini pair needs the anchor-fingerprint OT bridge (review §5.8) or another
alignment. The tier2 sbatch therefore **skips the phi pair by default** (`SKIP_PAIRS` env; its
array tasks exit 0 immediately — note the pairs interleave every 6 task ids, so do NOT try to
exclude them by trimming the array range). Unset `SKIP_PAIRS` only once the bridge exists.

## Files

- `tier1_tau_sweep.sbatch` — 24 tasks (8 τ × 3 seeds), same-arch 1.5B pair, 500 steps.
- `tier2_bridge_bakeoff.sbatch` — 36 tasks (6 arms × 2 pairs × 3 seeds), 2000 steps.
- `train_harness.py` — shared CLI contract + loss-arm specifications (stub).

## How the ACTUAL Tier-0 verdicts (2026-07-11, ../tier0-results.md) set the knobs

| Measured verdict | Knob setting (now the default) |
|---|---|
| M1 **NO-GO**, both pairs: row-space carries ≥99% of mKNN; relational signal richest at 25–50% depth, dead at final layer | `SUPERVISED_LAYER_FRACS=0.25,0.5` |
| M1 no-go (identity output-readable) | `soft_knn_kl` (identity) runs as **control only** — keep it in ARMS for the control row, never headline it; `SKIP_ARMS=soft_knn_kl` to drop it entirely if allocation is tight |
| M2: identity ranked first, local-distance ≈ global second (both pairs) — but M1 overrides identity-only as a treatment | LN-Dist stays the lead relational arm; no knob change |
| M3 **AMBIGUOUS**, rollout-level trend weakly negative for geometry | functional-bridge variant is **required**, not optional — add `_fbridge` arms when implementing the harness |
| M4: OPRD rel. variance 0.389 vs OPD 0.857; power table says 2-pt effects need ~53–82 eval seeds/arm (pass 0.2–0.5, ρ=0.2), 1-pt needs 200–700 | keep 3 training seeds, use Avg@16 with ≥64 eval seeds; design Tier 1–2 to detect **≥2-pt effects only** |
| Tier 3 pilot (1.5B, ../tier3-results.md): row-only ≥ full functionally, null-only hurts; **mid-layer (depth 50%) supervision beat all final-layer arms** | confirms `SUPERVISED_LAYER_FRACS=0.25,0.5` causally; consider a `row_projected` ablation arm in Tier 1 (cheap, one projector matmul in the loss) |

## Allocation estimate (A100-80GB hours, rough)

| Job | Tasks | Est. hrs/task | Total |
|---|---|---|---|
| Tier 1 τ-sweep | 24 | ~4 (500 steps + rollouts) | ~96 GPU-h |
| Tier 2 bake-off | 36 | ~10 (2000 steps + AIME Avg@16 eval) | ~360 GPU-h |
| **Total** | | | **~450 GPU-h ≈ 19 A100-days** |

Halve by running Tier 2 at 2 seeds first and topping up the winner arms to 3+
(seed budget per M4's power table). Rollout generation dominates wall-clock;
a persistent vLLM server per node amortizes it across arms.

## Submission

```bash
mkdir -p logs runs
sbatch tier1_tau_sweep.sbatch                 # defaults already reflect Tier-0 verdicts
sbatch tier2_bridge_bakeoff.sbatch            # add SKIP_ARMS=soft_knn_kl to drop the identity control
```
