"""Generate rollouts with mlx-lm batch_generate.

usage: gen_rollouts.py <model_key> <n_rollouts> <max_tokens> [--smoke]
Writes rollouts/<model_key>[_smoke].jsonl: {idx, prompt, gen_text}
"""
import json
import sys
import time

import mlx.core as mx

# cap mlx GPU memory so overload raises instead of wiring unified memory to the
# point of kernel starvation (2026-07-11 watchdog panic)
_MLX_CAP = 20 * 2**30
if hasattr(mx, "set_memory_limit"):
    mx.set_memory_limit(_MLX_CAP)
elif hasattr(mx, "metal") and hasattr(mx.metal, "set_memory_limit"):
    mx.metal.set_memory_limit(_MLX_CAP)

from mlx_lm import load
from mlx_lm.generate import batch_generate
from mlx_lm.sample_utils import make_sampler

from common import MODELS, SEED, build_prompt_ids, is_base_model, load_prompts, rollout_path, log

BATCH = 16


def main():
    model_key = sys.argv[1]
    n = int(sys.argv[2])
    max_tokens = int(sys.argv[3])
    smoke = "--smoke" in sys.argv
    repo = MODELS[model_key]
    is_base = is_base_model(repo)

    mx.random.seed(SEED)
    log(f"gen: loading {repo} (mlx)")
    model, tokenizer = load(repo)
    prompts = load_prompts()[:n]
    enc = [build_prompt_ids(tokenizer, p["prompt"], is_base) for p in prompts]

    sampler = make_sampler(temp=0.6, top_p=0.95)
    t0 = time.time()
    out_path = rollout_path(model_key, smoke)
    results = []
    for s in range(0, len(enc), BATCH):
        chunk = enc[s:s + BATCH]
        resp = batch_generate(model, tokenizer, chunk, max_tokens=max_tokens,
                              sampler=sampler, verbose=False)
        for j, text in enumerate(resp.texts):
            results.append({"idx": prompts[s + j]["idx"],
                            "prompt": prompts[s + j]["prompt"],
                            "gen_text": text})
        log(f"gen {model_key}: {min(s+BATCH,len(enc))}/{len(enc)} rollouts, "
            f"{time.time()-t0:.0f}s elapsed")
        # checkpoint after every batch
        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
    log(f"gen {model_key}: DONE {len(results)} rollouts in {time.time()-t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
