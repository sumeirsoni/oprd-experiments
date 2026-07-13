"""Avenue-10-lite: update-subspace overlap between OPD and OPRD arms + escape-component
energy in W_head's top vs bottom singular directions. CPU only, run after the campaign.

usage: t3_analysis.py

For residual-stream-facing 2D params (o_proj, down_proj, every 4th layer + last), per run:
delta = ckpt_final - base weights; U16 = top-16 left-singular basis of delta (hidden-dim side).
Reports, per (layer, param): mean principal-angle cosine between OPD's U16 and each MSE arm's
U16 (seed 0); and for each MSE arm, the energy fraction of its OPD-orthogonalized escape
component that lies in W_head's row space (top-r90 dirs) vs null space.
Writes results/t3_subspace.json.
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM

from common import MODELS, RESULTS, TIER0, log, save_json

T3 = TIER0 / "t3"
ARMS = ["full", "row", "null", "mid"]  # compared against opd
K = 16


def top_left_sv(delta, k=K):
    U, S, _ = torch.linalg.svd(delta.float(), full_matrices=False)
    return U[:, :k], S


def principal_cosines(A, B):
    """A, B: (d, k) orthonormal bases -> singular values of A^T B."""
    return torch.linalg.svdvals(A.T @ B)


def main():
    log("t3 analysis: loading base weights (cpu)")
    base = AutoModelForCausalLM.from_pretrained(MODELS["r1d15"], dtype=torch.float32)
    bsd = base.state_dict()
    n_layers = base.config.num_hidden_layers
    layers = sorted(set(list(range(0, n_layers, 4)) + [n_layers - 1]))
    params = [f"model.layers.{li}.self_attn.o_proj.weight" for li in layers] + \
             [f"model.layers.{li}.mlp.down_proj.weight" for li in layers]

    z = np.load(T3 / "projector.npz")
    Vr = torch.tensor(z["Vr"]).float()  # (r90, hidden) row-space basis of W_head

    ckpts = {}
    for arm in ARMS + ["opd"]:
        p = T3 / "runs" / f"{arm}_s0" / "ckpt_final.pt"
        if p.exists():
            ckpts[arm] = torch.load(p, map_location="cpu")
        else:
            log(f"t3 analysis: missing {p}, skipping arm {arm}")

    if "opd" not in ckpts:
        log("t3 analysis: no opd checkpoint — cannot compute overlaps, aborting")
        return

    out = {"k": K, "params": [], "arm_means": {}}
    per_arm_cos, per_arm_rowfrac = {a: [] for a in ARMS}, {a: [] for a in ARMS}
    for pname in params:
        row = {"param": pname}
        d_opd = ckpts["opd"][pname].float() - bsd[pname]
        U_opd, _ = top_left_sv(d_opd)
        for arm in ARMS:
            if arm not in ckpts:
                continue
            d = ckpts[arm][pname].float() - bsd[pname]
            U, S = top_left_sv(d)
            cos = principal_cosines(U_opd, U)
            # escape: components of arm's update basis orthogonal to OPD's
            esc = U - U_opd @ (U_opd.T @ U)
            esc_energy = (esc**2).sum()
            esc_row = ((Vr @ esc)**2).sum()  # only valid when hidden-dim side (d==1536)
            row[arm] = {
                "mean_principal_cos": float(cos.mean()),
                "top_principal_cos": float(cos[0]),
                "escape_row_energy_frac": float(esc_row / esc_energy) if esc_energy > 0 else None,
                "delta_spectral_top1_frac": float(S[0]**2 / (S**2).sum()),
            }
            per_arm_cos[arm].append(row[arm]["mean_principal_cos"])
            if row[arm]["escape_row_energy_frac"] is not None:
                per_arm_rowfrac[arm].append(row[arm]["escape_row_energy_frac"])
        out["params"].append(row)
        log(f"t3 analysis: {pname} done")

    r90 = Vr.shape[0]
    hid = Vr.shape[1]
    expected_row_frac = r90 / hid  # energy fraction under isotropy
    for arm in ARMS:
        if per_arm_cos[arm]:
            out["arm_means"][arm] = {
                "mean_principal_cos_vs_opd": float(np.mean(per_arm_cos[arm])),
                "escape_row_energy_frac": float(np.mean(per_arm_rowfrac[arm])),
                "isotropic_expectation": expected_row_frac,
            }
    out["note"] = ("escape_row_energy_frac >> isotropic_expectation means the arm's "
                   "OPD-orthogonal update component still lives in W_head row space "
                   "(head-readable); << means it escapes into the null space.")
    save_json(RESULTS / "t3_subspace.json", out)
    log(f"t3 analysis: DONE -> results/t3_subspace.json; arm means: {out['arm_means']}")


if __name__ == "__main__":
    main()
