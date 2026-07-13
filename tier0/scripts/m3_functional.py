"""M3: functional vs geometric alignment (correlational only).

usage: m3_functional.py <pair> [--smoke]
Writes results/m3_<pair>[_smoke].json
"""
import re
import sys

import numpy as np
import torch
from scipy import stats

from common import HIDDEN, PAIRS, RESULTS, SEED, load_json, load_prompts, load_rollouts, log, save_json

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
PCA_R = 128


def extract_boxed(text):
    m = list(re.finditer(r"\\boxed\{", text))
    if not m:
        return None
    s = m[-1].end()
    depth, i = 1, s
    while i < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
    return text[s:i - 1].strip()


def norm_ans(a):
    if a is None:
        return None
    a = str(a).strip().replace(" ", "").replace("\\!", "").replace("\\,", "")
    a = a.replace("\\left", "").replace("\\right", "").rstrip(".")
    try:
        return str(float(a))
    except ValueError:
        return a


def geo_cosine(Ht, Hs, rng):
    """Per-token PCA-cosine: own-PCA to PCA_R dims, Procrustes-map teacher->student
    on a fit split, cosine on all tokens (fit mask returned)."""
    n = Ht.shape[0]
    fit = np.zeros(n, bool)
    fit[rng.choice(n, n // 2, replace=False)] = True
    out = {}
    for key, H in (("t", Ht), ("s", Hs)):
        Hc = H - H[fit].mean(0)
        # PCA basis from fit split
        cov_src = Hc[fit]
        _, _, V = torch.linalg.svd(cov_src.cpu(), full_matrices=False)
        r = min(PCA_R, V.shape[0])
        out[key] = Hc @ V[:r].T.to(DEV)
    A, B = out["t"], out["s"]
    r = min(A.shape[1], B.shape[1])
    A, B = A[:, :r], B[:, :r]
    # orthogonal Procrustes on fit split
    M = A[fit].T @ B[fit]
    U, _, Vt = torch.linalg.svd(M.cpu())
    R = (U @ Vt).to(DEV)
    cos = torch.nn.functional.cosine_similarity(A @ R, B, dim=-1)
    return cos.cpu().numpy(), fit


def corr_block(name, x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 10 or x[ok].std() == 0 or y[ok].std() == 0:
        return {name: None}
    r = stats.spearmanr(x[ok], y[ok])
    return {name: {"spearman": float(r.statistic), "p": float(r.pvalue), "n": int(ok.sum())}}


def main():
    pair = sys.argv[1]
    smoke = "--smoke" in sys.argv
    tkey, skey, srckey = PAIRS[pair]
    hdir = HIDDEN / f"{pair}{'_smoke' if smoke else ''}"
    meta = load_json(hdir / "meta.json")
    fn = np.load(hdir / "functional.npz")
    rollout_id = np.load(hdir / "rollout_id.npy")
    pos = np.load(hdir / "pos_in_gen.npy")
    rng = np.random.default_rng(SEED)

    kl, top1 = fn["kl_ts"], fn["top1_agree"]
    out = {"pair": pair, "n_tokens": int(len(kl))}

    # geometric alignment at mid + final student depth
    geo = {}
    for frac in (0.5, 1.0):
        lt = max(1, round(frac * meta["num_hidden_layers"][tkey]))
        ls = max(1, round(frac * meta["num_hidden_layers"][skey]))
        Ht = torch.tensor(np.ascontiguousarray(np.load(hdir / tkey / f"layer{lt}.f16.npy", mmap_mode="r")),
                          dtype=torch.float32, device=DEV)
        Hs = torch.tensor(np.ascontiguousarray(np.load(hdir / skey / f"layer{ls}.f16.npy", mmap_mode="r")),
                          dtype=torch.float32, device=DEV)
        geo[frac], _ = geo_cosine(Ht, Hs, rng)
        del Ht, Hs
        if DEV == "mps":
            torch.mps.empty_cache()

    # correctness per rollout (dataset ground truth; fall back to None)
    answers = {p["idx"]: p.get("answer") for p in load_prompts()}
    rollouts = load_rollouts(srckey, smoke)
    correct = {}
    graded = 0
    for r in rollouts:
        gt = norm_ans(answers.get(r["idx"]))
        pred = norm_ans(extract_boxed(r["gen_text"]))
        if gt is not None and pred is not None:
            correct[r["idx"]] = float(pred == gt)
            graded += 1
        else:
            correct[r["idx"]] = np.nan
    out["grading"] = {"method": "dataset ground_truth vs \\boxed{} extraction",
                      "graded_rollouts": graded, "total": len(rollouts)}

    tok_correct = np.array([correct.get(int(r), np.nan) for r in rollout_id])

    # boxed split: reasoning region vs answer-emission region (proxy for
    # pre-decision vs generation-region split of 2605.23315)
    boxed_pos = {}
    src_rolls = {r["idx"]: r for r in rollouts}
    # approximate: fraction position of '\boxed' in text -> fraction of tokens
    for idx, r in src_rolls.items():
        t = r["gen_text"]
        k = t.rfind("\\boxed")
        boxed_pos[idx] = (k / max(1, len(t))) if k >= 0 else np.nan
    rl_len = {}
    for rid in np.unique(rollout_id):
        rl_len[rid] = pos[rollout_id == rid].max() + 1
    frac_pos = pos / np.array([rl_len[r] for r in rollout_id])
    pre_dec = np.array([frac_pos[i] < boxed_pos.get(int(rollout_id[i]), np.nan)
                        for i in range(len(pos))], dtype=float)

    # correlations
    corrs = {}
    for frac in (0.5, 1.0):
        g = geo[frac]
        corrs.update(corr_block(f"geo{frac}_vs_negKL", g, -kl))
        corrs.update(corr_block(f"geo{frac}_vs_top1", g, top1))
        corrs.update(corr_block(f"geo{frac}_vs_pos", g, frac_pos))
    corrs.update(corr_block("KL_vs_pos", kl, frac_pos))

    # per-rollout means vs correctness
    rids = np.unique(rollout_id)
    def per_rollout(v):
        return np.array([np.nanmean(v[rollout_id == r]) for r in rids])
    rc = np.array([correct.get(int(r), np.nan) for r in rids])
    ok = np.isfinite(rc)
    roll_corr = {}
    for name, v in [("geo0.5", geo[0.5]), ("geo1.0", geo[1.0]), ("negKL", -kl), ("top1", top1)]:
        pv = per_rollout(v)[ok]
        if ok.sum() >= 8 and np.unique(rc[ok]).size > 1 and pv.std() > 0:
            r_ = stats.pointbiserialr(rc[ok], pv)
            roll_corr[name] = {"pointbiserial": float(r_.statistic), "p": float(r_.pvalue),
                               "n": int(ok.sum())}
        else:
            roll_corr[name] = None

    # region split means
    region = {}
    for name, v in [("geo1.0", geo[1.0]), ("KL", kl)]:
        m = np.isfinite(pre_dec)
        region[name] = {"pre_decision_mean": float(np.nanmean(v[m & (pre_dec == 1)])),
                        "answer_region_mean": float(np.nanmean(v[m & (pre_dec == 0)]))}

    out.update({"token_correlations": corrs, "rollout_vs_correctness": roll_corr,
                "region_split": region,
                "logitlens_kl_mean_by_layer": {str(l): float(fn["logitlens_kl"][:, j].mean())
                                               for j, l in enumerate(fn["m1_student_layers"])}})

    # verdict
    def tracks(d):
        return d is not None and d["p"] < 0.05 and abs(d["pointbiserial"]) > 0.1
    f_tracks = tracks(roll_corr.get("negKL")) or tracks(roll_corr.get("top1"))
    g_tracks = tracks(roll_corr.get("geo1.0")) or tracks(roll_corr.get("geo0.5"))
    if f_tracks and not g_tracks:
        verdict = ("geometric alignment does NOT track correctness where functional does - "
                   "every Tier 1-2 geometric arm needs the functional-bridge variant.")
    elif f_tracks and g_tracks:
        verdict = "both geometric and functional track correctness - causal-inertness threat weakened."
    elif not f_tracks and not g_tracks:
        verdict = "AMBIGUOUS: neither alignment tracks correctness at this N (low power or ceiling)."
    else:
        verdict = "geometric tracks but functional does not - unexpected; inspect before acting."
    out["summary"] = {"functional_tracks": bool(f_tracks), "geometric_tracks": bool(g_tracks),
                      "verdict": verdict}
    save_json(RESULTS / f"m3_{pair}{'_smoke' if smoke else ''}.json", out)
    log(f"m3 {pair}: DONE. {verdict}")


if __name__ == "__main__":
    main()
