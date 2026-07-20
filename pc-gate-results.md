# Positive-control thin gate (UMD Nexus CML, 2026-07-18/20)

Same-arch 1.5B pair under the **fixed rollout protocol**, before any Tier-2 rerun.

- Teacher: `hbx/JustRL-DeepSeek-1.5B`
- Student: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- Layers: `--supervised-layer-fracs 0.25,0.5` → layers **{7, 14}**
- Train: 2000 steps, `--rollout-tokens 16000 --supervise-last-k 2000 --require-finished`
- Eval: AIME24/25 Avg@16, `--max-tokens 24000`
- Job array `7106521`; MiniLM s0 resume `7111775`; MiniLM s1 eval-only rescue `7112484`
- Raw `results.json` / `train.log` in `oprd-experiment-results-2/runs/pc_*`
  (checkpoints remain on `/cmlscratch/ssoni11/oprd-experiments/cluster/runs/`)

Harness: `opd_topk` + `--require-finished` (`7d8aa3b` / `29f273a`) plus later
`--resume-from` / `--start-step` for walltime rescues.

## Avg@16 (%), mean ± sd over seeds

| arm | AIME24 | AIME25 | ans rate (24/25) | trunc (24/25) |
|---|---|---|---|---|
| untrained baseline | 28.33 | 22.92 | 0.72 / 0.77 | 0.28 / 0.23 |
| minilmv2_selfrel (3 seeds) | 35.42 ± 1.16 | 26.46 ± 1.27 | 0.90 / 0.94 | 0.10 / 0.06 |
| **opd_topk (3 seeds)** | **48.89 ± 1.68** | **33.68 ± 0.48** | **0.95 / 0.93** | **0.05 / 0.07** |

### Per-seed

| run | AIME24 | AIME25 | ans24 | trunc24 |
|---|---|---|---|---|
| baseline | 28.33 | 22.92 | 0.717 | 0.281 |
| opd_topk s0 | 47.92 | 33.12 | 0.969 | 0.031 |
| opd_topk s1 | 50.83 | 33.96 | 0.938 | 0.062 |
| opd_topk s2 | 47.92 | 33.96 | 0.935 | 0.065 |
| minilmv2 s0 | 35.21 | 27.92 | 0.906 | 0.094 |
| minilmv2 s1 | 34.38 | 25.62 | 0.902 | 0.098 |
| minilmv2 s2 | 36.67 | 25.83 | 0.898 | 0.102 |

## Verdict

1. **Gate PASS.** `opd_topk` clearly beats the untrained baseline on both AIME24 and
   AIME25, with much higher answered rate and lower truncation. The fixed long-rollout /
   last-k / require-finished protocol **can distill**.
2. **MiniLM is a weak positive** vs baseline (~+7pt AIME24, ~+3.5pt AIME25) but far below
   output-level OPD. Mid-layer relational matching helps a little; it is not the main
   lever on this same-arch pair.
3. This unblocks a Tier-2 rerun under the same protocol (previously confounded by
   2k-token truncated training rollouts — see `tier2-results.md`).

## Ops notes / rescues

- Finish-frac under `--require-finished` stayed high (~0.85–0.95 of 64-rollout pools).
- Array walltime was 36h: MiniLM s0 timed out at ~1930/2000 (rescued from `ckpt_last` @
  1900); s1 finished train but timed out mid-eval (eval-only rescue). s2 completed in-job.
- Early poller bug: pruning `tmp_policy` when pre-eval `results.json` appeared killed the
  first `opd_topk` evals; fixed by pruning only after `eval` is present. Those three
  seeds were re-eval'd from `ckpt_final` (jobs 7107033/35/36).

## Caveats

- MiniLM (and other relational arms) still use `REL_WINDOW=512` with `hs[:512]` on the
  last-k span (prefix of the supervised 2k, not `hs[-512:]`). Revisit before strong
  claims about relational arms vs OPD.
- MiniLM training loss often logged as ~0 at 5 decimals (near-zero KL); functional gap vs
  `opd_topk` remains large, so loss≈0 ≠ “matched the useful signal.”
- Same-arch strong student; does not by itself validate cross-width OPRD-Bridge.
