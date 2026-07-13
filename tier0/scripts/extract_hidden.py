"""Prefill teacher+student over the source model's rollouts; stream band-layer hidden
states (generated region only) to fp16 memmaps; compute per-token functional metrics.

usage: extract_hidden.py <pair> [--smoke] [--max-rollouts K]

Outputs under hidden/<pair>[_smoke]/:
  <model_key>/layer{i}.f16.npy   (n_gen_tokens, dim) fp16 memmap
  meta.json    {n_tokens, layers per model, dims, rollout index (id, start, len)}
  token_ids.npy (int32), rollout_id.npy (int32), pos_in_gen.npy (int32)
  functional.npz: kl_ts (teacher||student final), top1_agree, cos_final (same-dim only),
                  logitlens_kl (n_tokens, n_m1_layers) student logit-lens vs teacher final
"""
import os
import sys

# must be set before torch initializes MPS: fail with a catchable OOM at ~70% of
# the recommended working set instead of wiring memory until the kernel starves
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")  # must be <= high

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (HIDDEN, MODELS, PAIRS, M1_FRACS, band_layers, build_prompt_ids,
                    is_base_model, load_rollouts, log, save_json)

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
KL_CHUNK = 128


def load_model(repo):
    m = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.bfloat16).to(DEV).eval()
    return m


@torch.no_grad()
def forward(model, ids):
    out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
    return out.logits, out.hidden_states


def kl_div_chunked(logits_p, logits_q):
    """KL(p||q) per token, fp32, chunked over sequence."""
    kls = []
    for s in range(0, logits_p.shape[0], KL_CHUNK):
        p = torch.log_softmax(logits_p[s:s + KL_CHUNK].float(), -1)
        q = torch.log_softmax(logits_q[s:s + KL_CHUNK].float(), -1)
        kls.append((p.exp() * (p - q)).sum(-1))
    return torch.cat(kls)


def main():
    pair = sys.argv[1]
    smoke = "--smoke" in sys.argv
    max_rollouts = None
    if "--max-rollouts" in sys.argv:
        max_rollouts = int(sys.argv[sys.argv.index("--max-rollouts") + 1])
    tkey, skey, srckey = PAIRS[pair]
    trepo, srepo, srcrepo = MODELS[tkey], MODELS[skey], MODELS[srckey]
    outdir = HIDDEN / f"{pair}{'_smoke' if smoke else ''}"
    outdir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(srcrepo)
    is_base = is_base_model(srcrepo)
    rollouts = load_rollouts(srckey, smoke)
    if max_rollouts:
        rollouts = rollouts[:max_rollouts]

    # pass 1: tokenize, compute gen-region sizes
    seqs = []
    for r in rollouts:
        pids = build_prompt_ids(tok, r["prompt"], is_base)
        gids = tok.encode(r["gen_text"], add_special_tokens=False)
        if len(gids) < 8:
            continue
        seqs.append((r["idx"], list(pids), gids))
    n_tok = sum(len(g) for _, _, g in seqs)
    log(f"extract {pair}: {len(seqs)} rollouts, {n_tok} gen tokens, device={DEV}")

    log(f"extract: loading {trepo} + {srepo}")
    tm, sm = load_model(trepo), load_model(srepo)
    tcfg, scfg = tm.config, sm.config
    tlayers, slayers = band_layers(tcfg.num_hidden_layers), band_layers(scfg.num_hidden_layers)
    m1_s = sorted({max(1, round(f * scfg.num_hidden_layers)) for f in M1_FRACS})
    same_dim = tcfg.hidden_size == scfg.hidden_size

    maps = {}
    for key, layers, dim in [(tkey, tlayers, tcfg.hidden_size), (skey, slayers, scfg.hidden_size)]:
        d = outdir / key
        d.mkdir(parents=True, exist_ok=True)
        for li in layers:
            maps[(key, li)] = np.lib.format.open_memmap(
                d / f"layer{li}.f16.npy", mode="w+", dtype=np.float16, shape=(n_tok, dim))

    token_ids = np.zeros(n_tok, np.int32)
    rollout_id = np.zeros(n_tok, np.int32)
    pos_in_gen = np.zeros(n_tok, np.int32)
    kl_ts = np.zeros(n_tok, np.float32)
    top1_agree = np.zeros(n_tok, np.float32)
    cos_final = np.full(n_tok, np.nan, np.float32)
    ll_kl = np.zeros((n_tok, len(m1_s)), np.float32)

    index = []
    off = 0
    for ri, (idx, pids, gids) in enumerate(seqs):
        ids = torch.tensor([pids + gids], device=DEV)
        gstart, glen = len(pids), len(gids)
        # positions whose OUTPUT predicts a gen token: gstart-1 .. gstart+glen-2
        sl = slice(gstart - 1, gstart + glen - 1)
        tlog, thid = forward(tm, ids)
        slog, shid = forward(sm, ids)
        tlog, slog = tlog[0, sl], slog[0, sl]

        kl_ts[off:off + glen] = kl_div_chunked(tlog, slog).cpu().numpy()
        top1_agree[off:off + glen] = (tlog.argmax(-1) == slog.argmax(-1)).float().cpu().numpy()

        # hidden states at the positions of gen tokens themselves
        hsl = slice(gstart, gstart + glen)
        for li in tlayers:
            maps[(tkey, li)][off:off + glen] = thid[li][0, hsl].to(torch.float16).cpu().numpy()
        for li in slayers:
            maps[(skey, li)][off:off + glen] = shid[li][0, hsl].to(torch.float16).cpu().numpy()
        if same_dim:
            a, b = thid[tcfg.num_hidden_layers][0, hsl].float(), shid[scfg.num_hidden_layers][0, hsl].float()
            cos_final[off:off + glen] = torch.nn.functional.cosine_similarity(a, b, -1).cpu().numpy()

        # student logit-lens KL vs teacher final, at M1 layers
        with torch.no_grad():
            for j, li in enumerate(m1_s):
                h = shid[li][0, sl]
                ll = sm.lm_head(sm.model.norm(h))
                ll_kl[off:off + glen, j] = kl_div_chunked(tlog, ll).cpu().numpy()

        token_ids[off:off + glen] = gids
        rollout_id[off:off + glen] = idx
        pos_in_gen[off:off + glen] = np.arange(glen)
        index.append({"idx": idx, "start": off, "len": glen})
        off += glen
        if (ri + 1) % 10 == 0 or ri + 1 == len(seqs):
            log(f"extract {pair}: {ri+1}/{len(seqs)} rollouts, {off} tokens")

    for m in maps.values():
        m.flush()
    np.save(outdir / "token_ids.npy", token_ids)
    np.save(outdir / "rollout_id.npy", rollout_id)
    np.save(outdir / "pos_in_gen.npy", pos_in_gen)
    np.savez(outdir / "functional.npz", kl_ts=kl_ts, top1_agree=top1_agree,
             cos_final=cos_final, logitlens_kl=ll_kl, m1_student_layers=np.array(m1_s))
    save_json(outdir / "meta.json", {
        "pair": pair, "teacher": trepo, "student": srepo, "source": srcrepo,
        "n_tokens": int(off), "same_dim": same_dim,
        "layers": {tkey: tlayers, skey: slayers},
        "dims": {tkey: tcfg.hidden_size, skey: scfg.hidden_size},
        "num_hidden_layers": {tkey: tcfg.num_hidden_layers, skey: scfg.num_hidden_layers},
        "index": index})
    log(f"extract {pair}: DONE, {off} tokens -> {outdir}")


if __name__ == "__main__":
    main()
