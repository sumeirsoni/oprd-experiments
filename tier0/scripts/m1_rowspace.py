"""M1: does neighbor identity live in W_head row space or null space?

usage: m1_rowspace.py <pair> [--smoke]
Writes results/m1_<pair>[_smoke].json
"""
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from common import (HIDDEN, MODELS, PAIRS, M1_FRACS, RESULTS, SEED, load_json, log,
                    save_json)

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
KS = [5, 10, 30]
KMAX = 30
BLOCK = 8192


def head_right_singvecs(repo):
    """Right singular vectors + spectral energy of W_head, float64 CPU via gram eigh."""
    m = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32)
    W = m.get_output_embeddings().weight.detach().double().numpy()  # (V, d)
    del m
    G = W.T @ W
    evals, evecs = np.linalg.eigh(G)  # ascending
    evals, evecs = evals[::-1], evecs[:, ::-1]
    energy = np.cumsum(evals) / evals.sum()
    r90 = int(np.searchsorted(energy, 0.90) + 1)
    return evecs, r90  # columns = right singular dirs, descending


def topk_neighbors(X):
    """Top-KMAX cosine neighbors (self excluded). X: (N, d) float16 torch on DEV."""
    X = torch.nn.functional.normalize(X.float(), dim=1).half()
    N = X.shape[0]
    idxs = torch.zeros(N, KMAX, dtype=torch.long)
    for s in range(0, N, BLOCK):
        b = X[s:s + BLOCK]
        vals = torch.full((b.shape[0], KMAX), -2.0, device=DEV)
        ids = torch.zeros(b.shape[0], KMAX, dtype=torch.long, device=DEV)
        for cs in range(0, N, BLOCK):
            sim = (b @ X[cs:cs + BLOCK].T).float()
            if cs == s:
                sim.fill_diagonal_(-2.0)
            elif cs < s + b.shape[0] and s < cs + sim.shape[1]:
                for i in range(sim.shape[0]):
                    j = s + i - cs
                    if 0 <= j < sim.shape[1]:
                        sim[i, j] = -2.0
            allv = torch.cat([vals, sim], 1)
            alli = torch.cat([ids, torch.arange(cs, cs + sim.shape[1], device=DEV).expand(sim.shape[0], -1)], 1)
            vals, pos = allv.topk(KMAX, dim=1)
            ids = alli.gather(1, pos)
        idxs[s:s + BLOCK] = ids.cpu()
    return idxs.numpy()


def mutual_knn_agreement(idx_t, idx_s):
    out = {}
    for k in KS:
        a, b = idx_t[:, :k], idx_s[:, :k]
        inter = np.array([len(np.intersect1d(a[i], b[i], assume_unique=True)) for i in range(a.shape[0])])
        out[k] = float(inter.mean() / k)
    return out


def project(H, basis):
    """Coordinates of H in subspace spanned by basis columns. H torch (N,d) on DEV."""
    return H @ basis


def main():
    pair = sys.argv[1]
    smoke = "--smoke" in sys.argv
    tkey, skey, _ = PAIRS[pair]
    hdir = HIDDEN / f"{pair}{'_smoke' if smoke else ''}"
    meta = load_json(hdir / "meta.json")
    n_tok = meta["n_tokens"]
    rng = np.random.default_rng(SEED)

    Ns = [n for n in [1000, 10000, 50000, 100000] if n <= n_tok]
    if n_tok < 100000 and n_tok not in Ns:
        Ns.append(n_tok)
    Nmax = max(Ns)
    sel = np.sort(rng.choice(n_tok, Nmax, replace=False))

    # per-model: head SVD + projector bases
    bases, r90s = {}, {}
    for key in (tkey, skey):
        evecs, r90 = head_right_singvecs(MODELS[key])
        d = evecs.shape[0]
        r90s[key] = r90
        B = {}
        B["row_r90"] = evecs[:, :r90]
        B["null_r90"] = evecs[:, r90:]
        for r in (64, 256):
            if r < d:
                B["row_%d" % r] = evecs[:, :r]
        bases[key] = {k: torch.tensor(np.ascontiguousarray(v), dtype=torch.float32, device=DEV)
                      for k, v in B.items()}
        log(f"m1 {pair}: {key} head d={d} r90={r90}")

    fracs_layers = {}
    for key in (tkey, skey):
        nl = meta["num_hidden_layers"][key]
        fracs_layers[key] = {f: max(1, round(f * nl)) for f in M1_FRACS}

    results = {"pair": pair, "r90": r90s, "Ns": Ns, "n_pool": n_tok, "cells": []}
    projections = ["full", "pca_r90", "row_r90", "null_r90", "row_64", "row_256"]

    for frac in M1_FRACS:
        # load layer states for the max-N sample once per model
        H = {}
        for key in (tkey, skey):
            li = fracs_layers[key][frac]
            arr = np.load(hdir / key / f"layer{li}.f16.npy", mmap_mode="r")
            H[key] = torch.tensor(np.ascontiguousarray(arr[sel]), dtype=torch.float32, device=DEV)
        for proj in projections:
            # neighbor indices per model at Nmax done per N (kNN depends on pool size)
            for N in Ns:
                if proj not in ("full", "row_r90", "null_r90") and N != Nmax:
                    continue  # N-sweep only for the three verdict projections
                nbrs = {}
                ok = True
                for key in (tkey, skey):
                    h = H[key][:N]
                    if proj == "full":
                        X = h
                    elif proj == "pca_r90":
                        r = min(r90s[key], h.shape[1], N - 1)
                        hc = h - h.mean(0)
                        _, _, V = torch.linalg.svd(hc.cpu(), full_matrices=False)
                        X = hc @ V[:r].T.to(DEV)
                    else:
                        if proj not in bases[key]:
                            ok = False
                            break
                        X = project(h, bases[key][proj])
                    nbrs[key] = topk_neighbors(X.half())
                if not ok:
                    continue
                agree = mutual_knn_agreement(nbrs[tkey], nbrs[skey])
                results["cells"].append({"frac": frac, "proj": proj, "N": N, "mknn": agree})
                log(f"m1 {pair}: frac={frac} proj={proj} N={N} mknn={agree}")
        del H
        if DEV == "mps":
            torch.mps.empty_cache()

    # verdict computation
    def get(frac, proj, N, k=10):
        for c in results["cells"]:
            if c["frac"] == frac and c["proj"] == proj and c["N"] == N:
                return c["mknn"][str(k)] if str(k) in c["mknn"] else c["mknn"][k]
        return None

    Nv = Nmax
    ratios_row, null_shares = [], []
    for frac in M1_FRACS:
        full, row, null = get(frac, "full", Nv), get(frac, "row_r90", Nv), get(frac, "null_r90", Nv)
        if full and full > 0:
            ratios_row.append(row / full)
            null_shares.append(null / full)
    # fragility: slope of full-space mknn (k=10) vs log10 N
    xs, ys = [], []
    for N in Ns:
        v = get(0.5, "full", N)
        if v is not None:
            xs.append(np.log10(N)), ys.append(v)
    slope = float(np.polyfit(xs, ys, 1)[0]) if len(xs) > 1 else None

    row_ratio_min = float(min(ratios_row)) if ratios_row else None
    null_share_max = float(max(null_shares)) if null_shares else None
    if row_ratio_min is not None and row_ratio_min >= 0.90:
        verdict = ("NO-GO for identity-only arms: mKNN(P_row) >= 90% of mKNN(full) at all "
                   "swept layers - neighbor identity is output-readable, OPD already sees it.")
    elif null_share_max is not None and null_share_max >= 0.5:
        verdict = "GO: substantial null-space share of neighbor identity."
    else:
        verdict = "AMBIGUOUS: row-space does not dominate everywhere but null-space share is modest."
    results["summary"] = {"row_ratio_min": row_ratio_min, "null_share_max": null_share_max,
                          "fragility_slope_per_decade": slope, "verdict": verdict}
    save_json(RESULTS / f"m1_{pair}{'_smoke' if smoke else ''}.json", results)
    log(f"m1 {pair}: DONE. {verdict}")


if __name__ == "__main__":
    main()
