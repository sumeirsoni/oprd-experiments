"""Probe: does per-token DEPTH-trajectory agreement (teacher vs student) carry signal,
and does it predict functional agreement better than single-layer state similarity?

Uses tier0 same_arch extraction (8 layers x 198k tokens, both models) + functional.npz.
Per token t and model m: depth-Gram G_m(t) = pairwise cosines between h_l(t) for
l in the 8 stored layers (28 upper-tri entries) — dimension-free by construction.
Also: consecutive-layer cosine profile (7 dims) and norm profile (8 dims).

Outputs results/depth_traj_probe.json:
  - traj agreement (teacher vs student Gram correlation) vs shuffled-token null
  - Spearman of [traj agreement, final-layer cos, mid-layer cos] vs per-token -KL / top1
"""
import json

import numpy as np
from scipy import stats as st

from common import HIDDEN, RESULTS, log

LAYERS = [4, 7, 10, 14, 18, 21, 24, 28]
N = 8000
SEED = 0


def load_states(model_key, idx):
    hs = []
    for li in LAYERS:
        X = np.load(HIDDEN / "same_arch" / model_key / f"layer{li}.f16.npy", mmap_mode="r")
        hs.append(np.asarray(X[idx], dtype=np.float32))
    return np.stack(hs)  # (L, N, d)


def depth_gram(h):
    """h: (L, N, d) -> per-token upper-tri pairwise layer cosines (N, L*(L-1)/2)."""
    hn = h / (np.linalg.norm(h, axis=-1, keepdims=True) + 1e-8)
    L, n, _ = h.shape
    iu = np.triu_indices(L, k=1)
    G = np.einsum("lnd,mnd->nlm", hn, hn)  # (N, L, L)
    return G[:, iu[0], iu[1]]


def rowcorr(A, B):
    """Per-row Pearson corr between matching rows of A and B: (N,)"""
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    return (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1) + 1e-8)


def main():
    rng = np.random.default_rng(SEED)
    fz = np.load(HIDDEN / "same_arch" / "functional.npz")
    n_total = fz["kl_ts"].shape[0]
    idx = np.sort(rng.choice(n_total, N, replace=False))
    kl, top1 = fz["kl_ts"][idx], fz["top1_agree"][idx]

    log("depth probe: loading states")
    ht, hs = load_states("justrl", idx), load_states("r1d15", idx)

    Gt, Gs = depth_gram(ht), depth_gram(hs)
    traj_agree = rowcorr(Gt, Gs)
    perm = rng.permutation(N)
    null_agree = rowcorr(Gt, Gs[perm])

    def cos(a, b):
        return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)

    cos_final = cos(ht[-1], hs[-1])          # layer 28
    cos_mid = cos(ht[3], hs[3])              # layer 14

    # consecutive-layer cosine profiles (the "naive subtraction/cosine" variant)
    def consec(h):
        hn = h / (np.linalg.norm(h, axis=-1, keepdims=True) + 1e-8)
        return np.stack([(hn[i] * hn[i + 1]).sum(-1) for i in range(len(LAYERS) - 1)], 1)

    ct, cs = consec(ht), consec(hs)
    consec_agree = rowcorr(ct, cs)
    norm_profile_t = np.linalg.norm(ht, axis=-1).T  # (N, L)

    res = {
        "n_tokens": N, "layers": LAYERS,
        "traj_gram_agreement": {
            "mean": float(traj_agree.mean()), "sd": float(traj_agree.std()),
            "null_mean": float(null_agree.mean()), "null_sd": float(null_agree.std()),
        },
        "consec_cos_agreement": {
            "mean": float(consec_agree.mean()), "null_mean": float(rowcorr(ct, cs[perm]).mean()),
        },
        "teacher_norm_growth_l4_to_l28": float(
            (norm_profile_t[:, -1] / norm_profile_t[:, 0]).mean()),
        "predicts_functional": {},
    }
    for name, x in [("traj_gram_agree", traj_agree), ("consec_cos_agree", consec_agree),
                    ("cos_final_l28", cos_final), ("cos_mid_l14", cos_mid)]:
        r_kl = st.spearmanr(x, -kl)
        r_t1 = st.spearmanr(x, top1)
        res["predicts_functional"][name] = {
            "spearman_vs_negKL": float(r_kl.statistic), "p_kl": float(r_kl.pvalue),
            "spearman_vs_top1": float(r_t1.statistic), "p_t1": float(r_t1.pvalue),
        }
    # incremental value: does traj agreement predict -KL after controlling cos_final?
    resid = traj_agree - np.polyval(np.polyfit(cos_final, traj_agree, 1), cos_final)
    r_part = st.spearmanr(resid, -kl)
    res["partial_traj_vs_negKL_given_cosfinal"] = {
        "spearman": float(r_part.statistic), "p": float(r_part.pvalue)}

    with open(RESULTS / "depth_traj_probe.json", "w") as f:
        json.dump(res, f, indent=1)
    log(f"depth probe: traj agree {res['traj_gram_agreement']['mean']:.3f} "
        f"(null {res['traj_gram_agreement']['null_mean']:.3f}) -> results/depth_traj_probe.json")


if __name__ == "__main__":
    main()
