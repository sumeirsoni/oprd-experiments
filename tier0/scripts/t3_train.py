"""Tier 3 local pilot: train the 1.5B student against precomputed teacher hidden states
under one of five losses (see tier3-local.plan.md).

usage: t3_train.py --arm {full,row,null,opd,mid} --seed S [--steps 500] [--smoke]

Teacher targets come from tier0/hidden/same_arch/justrl/layer{14,28}.f16.npy memmaps —
the teacher model is never loaded. Checkpoints every 100 steps for resume.
"""
import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.85")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import HIDDEN, MODELS, TIER0, build_prompt_ids, load_json, load_rollouts, log

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
STUDENT = MODELS["r1d15"]
T3 = TIER0 / "t3"
MAX_TOK = 1000
N_TRAIN = 80  # rollouts 0-79 train, 80-99 held-out (index order of meta.json)
FINAL_L, MID_L = 28, 14
BAND = [4, 7, 10, 14, 18, 21, 24, 28]  # layers with stored teacher targets
LASTK = [21, 24, 28]  # OPRD's last-k heuristic


def get_projector(model):
    """Row-space projector basis (r90 right-singular dirs of the initial lm_head), cached."""
    p = T3 / "projector.npz"
    if p.exists():
        z = np.load(p)
        return torch.tensor(z["Vr"]), int(z["r90"])
    W = model.lm_head.weight.detach().cpu().double()
    _, S, Vh = torch.linalg.svd(W, full_matrices=False)
    energy = (S**2).cumsum(0) / (S**2).sum()
    r90 = int((energy < 0.90).sum().item()) + 1
    Vr = Vh[:r90].float()  # (r90, hidden)
    T3.mkdir(parents=True, exist_ok=True)
    np.savez(p, Vr=Vr.numpy(), r90=r90)
    log(f"t3: projector computed, r90={r90}/{Vh.shape[0]}")
    return Vr, r90


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["full", "row", "null", "opd", "mid", "all", "lastk"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.steps = 5
    run = f"{a.arm}_s{a.seed}{'_smoke' if a.smoke else ''}"
    out = T3 / "runs" / run
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    log(f"t3 train {run}: loading {STUDENT} fp32 + grad ckpt, device={DEV}")
    model = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.float32)
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.to(DEV).train()
    tok = AutoTokenizer.from_pretrained(STUDENT)

    Vr, r90 = get_projector(model)
    Vr = Vr.to(DEV)

    meta = load_json(HIDDEN / "same_arch" / "meta.json")
    index = meta["index"]  # list of {idx, start, len} in extraction order
    rollouts = {r["idx"]: r for r in load_rollouts("r1d15")}
    train_entries = index[:N_TRAIN]
    if a.arm == "all":
        loss_layers = BAND
    elif a.arm == "lastk":
        loss_layers = LASTK
    elif a.arm == "mid":
        loss_layers = [MID_L]
    else:
        loss_layers = [FINAL_L]
    Hts = {li: np.load(HIDDEN / "same_arch" / "justrl" / f"layer{li}.f16.npy", mmap_mode="r")
           for li in loss_layers}

    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.0)
    resume = out / "ckpt_last.pt"
    step0 = 0
    if resume.exists():
        st = torch.load(resume, map_location=DEV)
        model.load_state_dict(st["model"])
        step0 = st["step"]
        log(f"t3 train {run}: resumed at step {step0}")

    order = np.random.permutation(np.repeat(np.arange(N_TRAIN), (a.steps // N_TRAIN) + 1))
    losses = []
    for step in range(step0, a.steps):
        e = train_entries[order[step]]
        r = rollouts[e["idx"]]
        pids = build_prompt_ids(tok, r["prompt"], False)
        gids = tok.encode(r["gen_text"], add_special_tokens=False)
        glen = min(e["len"], MAX_TOK, len(gids))
        ids = torch.tensor([list(pids) + gids[:glen]], device=DEV)
        gstart = len(pids)

        outm = model(input_ids=ids, output_hidden_states=(a.arm != "opd"))
        if a.arm == "opd":
            logits = outm.logits[0, gstart - 1:gstart + glen - 1]
            loss = torch.nn.functional.cross_entropy(logits, ids[0, gstart:gstart + glen])
        else:
            per_layer = []
            for li in loss_layers:
                s = outm.hidden_states[li][0, gstart:gstart + glen]
                t = torch.tensor(np.ascontiguousarray(Hts[li][e["start"]:e["start"] + glen]),
                                 dtype=torch.float32, device=DEV)
                if a.arm == "row":
                    s, t = s @ Vr.T, t @ Vr.T
                elif a.arm == "null":
                    s, t = s - (s @ Vr.T) @ Vr, t - (t @ Vr.T) @ Vr
                per_layer.append(torch.nn.functional.mse_loss(s, t))
            loss = torch.stack(per_layer).mean()  # uniform layer weighting (OPRD default)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(float(loss.item()))
        if DEV == "mps":
            torch.mps.empty_cache()
        if (step + 1) % 10 == 0 or a.smoke:
            mem = torch.mps.current_allocated_memory() / 2**30 if DEV == "mps" else 0
            log(f"t3 train {run}: step {step+1}/{a.steps} loss={loss.item():.5f} mem={mem:.1f}GB")
        if (step + 1) % 100 == 0:
            torch.save({"model": model.state_dict(), "step": step + 1}, resume)

    model_bf16 = {k: v.to(torch.bfloat16).cpu() for k, v in model.state_dict().items()}
    torch.save(model_bf16, out / "ckpt_final.pt")
    resume.unlink(missing_ok=True)
    with open(out / "losses.json", "w") as f:
        json.dump({"arm": a.arm, "seed": a.seed, "steps": a.steps, "lr": a.lr,
                   "r90": r90, "layers": loss_layers, "losses": losses}, f)
    log(f"t3 train {run}: DONE {a.steps} steps -> {out}")


if __name__ == "__main__":
    main()
