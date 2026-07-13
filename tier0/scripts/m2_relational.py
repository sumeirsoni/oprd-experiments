"""M2: which relational property survives calibration - global distance vs identity vs
local-window distance? Block-permutation null (shuffle rollout correspondence).

usage: m2_relational.py <pair> [--smoke]
Writes results/m2_<pair>[_smoke].json
"""
import sys

import numpy as np
import torch
from scipy import stats

from common import HIDDEN, PAIRS, RESULTS, SEED, load_json, log, save_json

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
N_SAMPLE = 8000       # tokens for distance matrices
N_ANCHORS = 2000      # anchors for local-window metric
K_LOCAL = 10
N_PERMS = 100
K_ID = 10


def pdist_cos(X):
    Xn = torch.nn.functional.normalize(X, dim=1)
    return 1 - Xn @ Xn.T


def metrics(Ht, Hs, rng):
    """All three relational metrics for corresponding rows of Ht, Hs."""
    n = Ht.shape[0]
    idx = rng.choice(n, min(N_SAMPLE, n), replace=False)
    Dt, Ds = pdist_cos(Ht[idx]), pdist_cos(Hs[idx])
    iu = torch.triu_indices(len(idx), len(idx), offset=1, device=Dt.device)
    dt, ds = Dt[iu[0], iu[1]], Ds[iu[0], iu[1]]
    # (a) global distance alignment (RKD-style): Pearson r over pairwise distances
    dt_, ds_ = dt - dt.mean(), ds - ds.mean()
    global_r = float((dt_ @ ds_) / (dt_.norm() * ds_.norm() + 1e-9))

    # (b) identity: mutual-kNN (within the same sample)
    kt = (-Dt).topk(K_ID + 1, dim=1).indices[:, 1:]
    ks = (-Ds).topk(K_ID + 1, dim=1).indices[:, 1:]
    kt_np, ks_np = kt.cpu().numpy(), ks.cpu().numpy()
    inter = np.array([len(np.intersect1d(kt_np[i], ks_np[i])) for i in range(len(idx))])
    mknn = float(inter.mean() / K_ID)

    # (c) local window: teacher's top-k neighbors, compare distance sub-vectors
    anchors = rng.choice(len(idx), min(N_ANCHORS, len(idx)), replace=False)
    sp, l2 = [], []
    Dt_np, Ds_np = Dt.cpu().numpy(), Ds.cpu().numpy()
    for a in anchors:
        nb = kt_np[a]
        vt, vs = Dt_np[a, nb], Ds_np[a, nb]
        if vt.std() > 1e-9 and vs.std() > 1e-9:
            sp.append(stats.spearmanr(vt, vs).statistic)
        l2.append(np.linalg.norm(vt / (np.linalg.norm(vt) + 1e-9) - vs / (np.linalg.norm(vs) + 1e-9)))
    return {"global_r": global_r, "identity_mknn": mknn,
            "local_spearman": float(np.mean(sp)) if sp else None,
            "local_nl2": float(np.mean(l2))}


def block_permute(rollout_id, rng):
    """Permute rollout correspondence: student token i gets teacher token from a
    different rollout at the same within-rollout offset (truncated to min length)."""
    rids = np.unique(rollout_id)
    perm = rng.permutation(rids)
    starts = {r: np.flatnonzero(rollout_id == r) for r in rids}
    src, dst = [], []
    for r, pr in zip(rids, perm):
        a, b = starts[r], starts[pr]
        m = min(len(a), len(b))
        dst.append(a[:m])
        src.append(b[:m])
    return np.concatenate(dst), np.concatenate(src)


def main():
    pair = sys.argv[1]
    smoke = "--smoke" in sys.argv
    tkey, skey, _ = PAIRS[pair]
    hdir = HIDDEN / f"{pair}{'_smoke' if smoke else ''}"
    meta = load_json(hdir / "meta.json")
    rng = np.random.default_rng(SEED)
    rollout_id = np.load(hdir / "rollout_id.npy")

    # mid-depth layer (0.5 frac) as the primary M2 layer + final
    out = {"pair": pair, "layers": {}}
    for frac in (0.5, 1.0):
        Ht_l = max(1, round(frac * meta["num_hidden_layers"][tkey]))
        Hs_l = max(1, round(frac * meta["num_hidden_layers"][skey]))
        Ht = torch.tensor(np.ascontiguousarray(
            np.load(hdir / tkey / f"layer{Ht_l}.f16.npy", mmap_mode="r")), dtype=torch.float32, device=DEV)
        Hs = torch.tensor(np.ascontiguousarray(
            np.load(hdir / skey / f"layer{Hs_l}.f16.npy", mmap_mode="r")), dtype=torch.float32, device=DEV)

        raw = metrics(Ht, Hs, np.random.default_rng(SEED))
        nulls = {k: [] for k in raw}
        for p in range(N_PERMS):
            prng = np.random.default_rng(1000 + p)
            dst, src = block_permute(rollout_id, prng)
            m = metrics(Ht[dst], Hs[src], np.random.default_rng(SEED))
            for k, v in m.items():
                if v is not None:
                    nulls[k].append(v)
            if (p + 1) % 25 == 0:
                log(f"m2 {pair} frac={frac}: perm {p+1}/{N_PERMS}")
        layer_res = {}
        for k, v in raw.items():
            nl = np.array(nulls[k])
            cal = v - nl.mean() if v is not None else None
            lo, hi = np.percentile(nl, [2.5, 97.5])
            # CI on calibrated effect ~ raw - null distribution
            sig = bool(v is not None and (v < lo or v > hi))
            layer_res[k] = {"raw": v, "null_mean": float(nl.mean()), "null_sd": float(nl.std()),
                            "calibrated": float(cal) if cal is not None else None,
                            "null_ci95": [float(lo), float(hi)], "significant": sig}
        out["layers"][str(frac)] = layer_res
        log(f"m2 {pair} frac={frac}: {layer_res}")
        del Ht, Hs
        if DEV == "mps":
            torch.mps.empty_cache()

    # verdict: rank calibrated effects (normalize by null sd -> z-scores), mid layer
    mid = out["layers"]["0.5"]
    def z(k):
        c = mid[k]
        return abs(c["calibrated"]) / (c["null_sd"] + 1e-9) if c["calibrated"] is not None else 0.0
    zs = {"global": z("global_r"), "identity": z("identity_mknn"),
          "local_distance": z("local_spearman")}
    ranked = sorted(zs, key=zs.get, reverse=True)
    if zs["local_distance"] > zs["identity"]:
        verdict = ("local-distance dominates identity (calibrated z: "
                   f"{zs}) - identity-only arm demotes to control; LN-Dist leads.")
    else:
        verdict = f"identity >= local-distance (calibrated z: {zs}) - identity arm stays live."
    out["summary"] = {"calibrated_z_mid_layer": zs, "ranking": ranked, "verdict": verdict,
                      "note": "block permutation (whole rollouts) as primary null; "
                              "token-level exchangeability violated per ARH caveat"}
    save_json(RESULTS / f"m2_{pair}{'_smoke' if smoke else ''}.json", out)
    log(f"m2 {pair}: DONE. {verdict}")


if __name__ == "__main__":
    main()
