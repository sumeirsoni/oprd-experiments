"""M4: gradient variance probe (OPRD-MSE vs OPD sampled-token CE) + seed power analysis.

usage: m4_gradient.py <pair> [--smoke] [--steps N]
Writes results/m4_<pair>[_smoke].json
"""
import os
import sys

# fail with a catchable MPS OOM instead of wiring memory until the kernel starves
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")  # must be <= high

import numpy as np
import torch
from scipy import stats as st

from common import (HIDDEN, MODELS, PAIRS, RESULTS, TIER0, build_prompt_ids,
                    is_base_model, load_json, load_rollouts, log, save_json)

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MAX_TOK = 1000  # halved from 2000 per OOM ladder (MPS backward peak on long rollouts); recorded in D1


def grad_stats(G):
    """G: (steps, P) float32. Mean pairwise cosine + relative variance."""
    Gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-12)
    C = Gn @ Gn.T
    iu = np.triu_indices(len(G), 1)
    mean_cos = float(C[iu].mean())
    g_mean = G.mean(0)
    rel_var = float(1.0 - (np.linalg.norm(g_mean) ** 2) / (np.mean(np.linalg.norm(G, axis=1) ** 2) + 1e-12))
    return {"mean_pairwise_cosine": mean_cos, "relative_variance": rel_var,
            "mean_grad_norm": float(np.linalg.norm(G, axis=1).mean())}


def main():
    pair = sys.argv[1]
    smoke = "--smoke" in sys.argv
    steps = 30
    if "--steps" in sys.argv:
        steps = int(sys.argv[sys.argv.index("--steps") + 1])
    tkey, skey, srckey = PAIRS[pair]
    hdir = HIDDEN / f"{pair}{'_smoke' if smoke else ''}"
    meta = load_json(hdir / "meta.json")
    if meta["dims"][tkey] != meta["dims"][skey]:
        raise SystemExit("m4 requires same-dim pair (OPRD-MSE direct)")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    srepo, srcrepo = MODELS[skey], MODELS[srckey]
    tok = AutoTokenizer.from_pretrained(srcrepo)
    sm = AutoModelForCausalLM.from_pretrained(srepo, dtype=torch.float32).to(DEV)
    sm.train()
    for p in sm.parameters():
        p.requires_grad_(False)
    mid = sm.config.num_hidden_layers // 2
    block = sm.model.layers[mid]
    for p in block.parameters():
        p.requires_grad_(True)
    P = sum(p.numel() for p in block.parameters() if p.requires_grad)
    log(f"m4 {pair}: probing block layer {mid}, {P/1e6:.1f}M params, {steps} steps")

    final_l = meta["num_hidden_layers"][skey]
    t_final_l = meta["num_hidden_layers"][tkey]
    Hteach = np.load(hdir / tkey / f"layer{t_final_l}.f16.npy", mmap_mode="r")
    index = {e["idx"]: e for e in meta["index"]}
    rollouts = [r for r in load_rollouts(srckey, smoke) if r["idx"] in index][:steps]
    is_base = is_base_model(srcrepo)

    G_mse = np.zeros((len(rollouts), P), np.float32)
    G_ce = np.zeros((len(rollouts), P), np.float32)

    def collect(into):
        o = 0
        for p in block.parameters():
            if p.requires_grad:
                g = p.grad
                into[o:o + p.numel()] = (g if g is not None else torch.zeros_like(p)).detach().flatten().cpu().numpy()
                o += p.numel()
        block.zero_grad(set_to_none=True)

    for si, r in enumerate(rollouts):
        pids = build_prompt_ids(tok, r["prompt"], is_base)
        gids = tok.encode(r["gen_text"], add_special_tokens=False)
        e = index[r["idx"]]
        glen = min(e["len"], MAX_TOK, len(gids))
        ids = torch.tensor([list(pids) + gids[:glen]], device=DEV)
        gstart = len(pids)

        th = torch.tensor(np.ascontiguousarray(Hteach[e["start"]:e["start"] + glen]),
                          dtype=torch.float32, device=DEV)

        out = sm(input_ids=ids, output_hidden_states=True, use_cache=False)
        sh = out.hidden_states[final_l][0, gstart:gstart + glen]
        loss_mse = torch.nn.functional.mse_loss(sh, th)
        loss_mse.backward(retain_graph=True)
        collect(G_mse[si])

        logits = out.logits[0, gstart - 1:gstart + glen - 1]
        tgt = ids[0, gstart:gstart + glen]
        loss_ce = torch.nn.functional.cross_entropy(logits, tgt)
        loss_ce.backward()
        collect(G_ce[si])
        del out, sh, logits
        if DEV == "mps":
            torch.mps.empty_cache()
        log(f"m4 {pair}: step {si+1}/{len(rollouts)} mse={loss_mse.item():.4f} ce={loss_ce.item():.4f}")

    res = {"pair": pair, "block_layer": mid, "n_params_probed": int(P), "steps": len(rollouts),
           "oprd_mse": grad_stats(G_mse), "opd_ce": grad_stats(G_ce)}

    # power analysis: seeds to resolve 1pt / 2pt effects on AIME Avg@16
    try:
        aime = load_json(TIER0 / "results" / "aime_sizes.json")
        n_problems = int(aime.get("aime24", {}).get("n_problems", 30) or 30)
    except FileNotFoundError:
        n_problems = 30
    z_a, z_b = st.norm.ppf(0.975), st.norm.ppf(0.80)
    table = []
    for p in [0.10, 0.20, 0.30, 0.40, 0.50]:
        for rho in [0.0, 0.2, 0.5]:
            # per-seed variance of mean pass rate over n_problems x 16 samples,
            # with within-problem correlation rho across the 16 samples
            var_seed = p * (1 - p) * (1 + 15 * rho) / (n_problems * 16)
            for delta in [0.01, 0.02]:
                n = 2 * (z_a + z_b) ** 2 * var_seed / delta ** 2
                table.append({"pass_rate": p, "rho_within_problem": rho, "effect_pts": delta * 100,
                              "seeds_per_arm": int(np.ceil(n))})
    res["power_analysis"] = {"n_problems": n_problems, "design": "two-sample, 80% power, alpha=.05 two-sided, Avg@16",
                             "table": table}

    v_mse, v_ce = res["oprd_mse"]["relative_variance"], res["opd_ce"]["relative_variance"]
    if v_mse < 0.8 * v_ce:
        verdict = (f"OPRD gradient variance materially lower than OPD ({v_mse:.3f} vs {v_ce:.3f} "
                   "relative variance) - Theorem 1's variance claim holds as used.")
    else:
        verdict = (f"OPRD gradient variance NOT materially lower than OPD ({v_mse:.3f} vs {v_ce:.3f}) "
                   "- Theorem 1's practical relevance weakened (affects A1/C framing).")
    res["summary"] = {"verdict": verdict}
    save_json(RESULTS / f"m4_{pair}{'_smoke' if smoke else ''}.json", res)
    log(f"m4 {pair}: DONE. {verdict}")


if __name__ == "__main__":
    main()
