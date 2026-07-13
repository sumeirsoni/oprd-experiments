"""Sample 150 DAPO-Math-17k prompts (seed 0) -> tier0/prompts.jsonl. Also cache AIME sizes for M4."""
import json
import random

from datasets import load_dataset

from common import TIER0, SEED, save_json, log


def main():
    ds = load_dataset("BytedTsinghua-SIA/DAPO-Math-17k", split="train")
    log(f"DAPO-Math-17k rows: {len(ds)}; columns: {ds.column_names}")
    # dedupe by prompt text (the original repo has repeated rows), keep first answer
    seen = {}
    for ex in ds:
        p = ex["prompt"]
        if isinstance(p, list):  # chat format [{role, content}]
            text = "\n".join(m["content"] for m in p)
        else:
            text = str(p)
        if text not in seen:
            ans = ex.get("reward_model", {}) or {}
            gt = ans.get("ground_truth") if isinstance(ans, dict) else None
            seen[text] = gt
    keys = sorted(seen.keys())
    log(f"unique prompts: {len(keys)}")
    rng = random.Random(SEED)
    picks = rng.sample(keys, 150)
    with open(TIER0 / "prompts.jsonl", "w") as f:
        for i, text in enumerate(picks):
            f.write(json.dumps({"idx": i, "prompt": text, "answer": seen[text]}) + "\n")
    log("wrote tier0/prompts.jsonl (150 prompts, seed 0)")

    # AIME sizes for M4 power analysis (counts only)
    aime = {}
    for name, repo, split in [("aime24", "Maxwell-Jia/AIME_2024", "train"),
                              ("aime25", "MathArena/aime_2025", "train")]:
        try:
            d = load_dataset(repo, split=split)
            aime[name] = {"repo": repo, "n_problems": len(d)}
        except Exception as e:  # noqa: BLE001 — boundary: external download, record and continue
            aime[name] = {"repo": repo, "error": str(e)}
    save_json(TIER0 / "results" / "aime_sizes.json", aime)
    log(f"AIME sizes: {aime}")


if __name__ == "__main__":
    main()
