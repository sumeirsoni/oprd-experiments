"""Tier 3 eval: functional + geometric metrics for a trained checkpoint on held-out rollouts.

usage: t3_eval.py <run_name>|baseline [--smoke]

Loads teacher (justrl) + student (r1d15 with t3/runs/<run>/ckpt_final.pt, or untrained for
'baseline'), both bf16, no grad. Held-out = extraction-order rollouts 80-99, first 1000 gen
tokens. Writes results/t3_eval_<run>.json.
"""
import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.85")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")

import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import HIDDEN, MODELS, RESULTS, TIER0, build_prompt_ids, load_json, load_rollouts, log, save_json

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
T3 = TIER0 / "t3"
MAX_TOK = 1000
N_TRAIN = 80
FINAL_L, MID_L = 28, 14
KL_CHUNK = 128


def kl_chunked(lp, lq):
    kls = []
    for s in range(0, lp.shape[0], KL_CHUNK):
        p = torch.log_softmax(lp[s:s + KL_CHUNK].float(), -1)
        q = torch.log_softmax(lq[s:s + KL_CHUNK].float(), -1)
        kls.append((p.exp() * (p - q)).sum(-1))
    return torch.cat(kls)


@torch.no_grad()
def main():
    run = sys.argv[1]
    smoke = "--smoke" in sys.argv
    log(f"t3 eval {run}: loading models bf16, device={DEV}")
    tm = AutoModelForCausalLM.from_pretrained(MODELS["justrl"], dtype=torch.bfloat16).to(DEV).eval()
    sm = AutoModelForCausalLM.from_pretrained(MODELS["r1d15"], dtype=torch.bfloat16).to(DEV).eval()
    if run != "baseline":
        st = torch.load(T3 / "runs" / run / "ckpt_final.pt", map_location="cpu")
        sm.load_state_dict({k: v.to(torch.bfloat16) for k, v in st.items()})
        log(f"t3 eval {run}: checkpoint loaded")
    tok = AutoTokenizer.from_pretrained(MODELS["r1d15"])

    z = np.load(T3 / "projector.npz")
    Vr = torch.tensor(z["Vr"]).to(DEV)

    meta = load_json(HIDDEN / "same_arch" / "meta.json")
    held = meta["index"][N_TRAIN:N_TRAIN + (2 if smoke else 20)]
    rollouts = {r["idx"]: r for r in load_rollouts("r1d15")}

    kls, top1s, mse_full, mse_row, mse_null, mse_mid = [], [], [], [], [], []
    for i, e in enumerate(held):
        r = rollouts[e["idx"]]
        pids = build_prompt_ids(tok, r["prompt"], False)
        gids = tok.encode(r["gen_text"], add_special_tokens=False)
        glen = min(e["len"], MAX_TOK, len(gids))
        ids = torch.tensor([list(pids) + gids[:glen]], device=DEV)
        gstart = len(pids)
        sl = slice(gstart - 1, gstart + glen - 1)
        hsl = slice(gstart, gstart + glen)

        to = tm(input_ids=ids, output_hidden_states=True, use_cache=False)
        so = sm(input_ids=ids, output_hidden_states=True, use_cache=False)
        tl, slg = to.logits[0, sl], so.logits[0, sl]
        kls.append(kl_chunked(tl, slg).mean().item())
        top1s.append((tl.argmax(-1) == slg.argmax(-1)).float().mean().item())

        th = to.hidden_states[FINAL_L][0, hsl].float()
        sh = so.hidden_states[FINAL_L][0, hsl].float()
        d = sh - th
        d_row = (d @ Vr.T) @ Vr
        mse_full.append((d**2).mean().item())
        mse_row.append((d_row**2).mean().item())
        mse_null.append(((d - d_row)**2).mean().item())
        dm = so.hidden_states[MID_L][0, hsl].float() - to.hidden_states[MID_L][0, hsl].float()
        mse_mid.append((dm**2).mean().item())
        del to, so
        if DEV == "mps":
            torch.mps.empty_cache()
        log(f"t3 eval {run}: {i+1}/{len(held)}")

    res = {"run": run, "n_heldout": len(held),
           "kl_ts_mean": float(np.mean(kls)), "top1_agree_mean": float(np.mean(top1s)),
           "mse28_full": float(np.mean(mse_full)), "mse28_row": float(np.mean(mse_row)),
           "mse28_null": float(np.mean(mse_null)), "mse14_full": float(np.mean(mse_mid)),
           "per_rollout_kl": kls}
    save_json(RESULTS / f"t3_eval_{run}.json", res)
    log(f"t3 eval {run}: DONE kl={res['kl_ts_mean']:.4f} top1={res['top1_agree_mean']:.4f} "
        f"mse28={res['mse28_full']:.5f}")


if __name__ == "__main__":
    main()
