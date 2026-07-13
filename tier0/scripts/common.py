"""Shared config/helpers for tier0 measurements. All paths relative to tier0/."""
import json
import os
from pathlib import Path

import numpy as np

TIER0 = Path(__file__).resolve().parent.parent
ROLLOUTS = TIER0 / "rollouts"
HIDDEN = TIER0 / "hidden"
RESULTS = TIER0 / "results"
LOGS = TIER0 / "logs"

MODELS = {
    # key: (hf_repo, role notes)
    "smoke_teacher": "Qwen/Qwen2.5-0.5B-Instruct",
    "smoke_student": "Qwen/Qwen2.5-0.5B",
    "justrl": "hbx/JustRL-DeepSeek-1.5B",  # OPRD "JustRL-1.5B" — actual HF name
    "r1d15": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "qwen3_4b": "Qwen/Qwen3-4B",
    "qwen3_17b": "Qwen/Qwen3-1.7B-Base",
}

# pair key -> (teacher_key, student_key, rollout_source_key)
# extraction runs both models' prefill over the rollout_source's rollouts
PAIRS = {
    "smoke": ("smoke_teacher", "smoke_student", "smoke_student"),
    "same_arch": ("justrl", "r1d15", "r1d15"),
    "cross_width": ("qwen3_4b", "qwen3_17b", "qwen3_17b"),
}

BAND_FRACS = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
M1_FRACS = [0.25, 0.5, 0.75, 1.0]  # verdict layers (subset of band)
SEED = 0


def is_base_model(repo):
    return repo.endswith("-Base") or repo == "Qwen/Qwen2.5-0.5B"


def build_prompt_ids(tokenizer, prompt_text, is_base):
    """Token ids for a rollout prompt. Must match between generation and extraction."""
    instr = prompt_text + "\n\nPlease reason step by step, and put your final answer within \\boxed{}."
    if not is_base and getattr(tokenizer, "chat_template", None) is not None:
        out = tokenizer.apply_chat_template(
            [{"role": "user", "content": instr}], add_generation_prompt=True,
            tokenize=True)
        ids = out["input_ids"] if hasattr(out, "keys") else out
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return list(ids)
    return tokenizer.encode("Problem:\n" + instr + "\n\nSolution:\n")


def band_layers(num_hidden_layers):
    """Map fractional depths to 1-indexed hidden_states indices (0 = embeddings)."""
    idxs = sorted({max(1, round(f * num_hidden_layers)) for f in BAND_FRACS})
    return idxs


def load_prompts():
    with open(TIER0 / "prompts.jsonl") as f:
        return [json.loads(l) for l in f]


def rollout_path(model_key, smoke=False):
    return ROLLOUTS / f"{model_key}{'_smoke' if smoke else ''}.jsonl"


def load_rollouts(model_key, smoke=False):
    p = rollout_path(model_key, smoke)
    with open(p) as f:
        return [json.loads(l) for l in f]


def hidden_dir(pair, model_key, smoke=False):
    d = HIDDEN / f"{pair}{'_smoke' if smoke else ''}" / model_key
    d.mkdir(parents=True, exist_ok=True)
    return d


def open_memmap(path, shape, mode="r"):
    return np.memmap(path, dtype=np.float16, mode=mode, shape=tuple(shape))


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=float)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def log(msg):
    import datetime
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOGS / "run.log", "a") as f:
        f.write(line + "\n")
