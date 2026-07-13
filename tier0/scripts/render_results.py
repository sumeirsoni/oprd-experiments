"""Render results/*.json into a markdown report.

usage: render_results.py <out_path> [--smoke]
"""
import sys
from pathlib import Path

from common import RESULTS, load_json


def cell(pair, m, smoke):
    p = RESULTS / f"{m}_{pair}{'_smoke' if smoke else ''}.json"
    if not p.exists():
        return None
    return load_json(p)


def main():
    out_path = Path(sys.argv[1])
    smoke = "--smoke" in sys.argv
    pairs = ["smoke"] if smoke else ["same_arch", "cross_width"]
    lines = []
    W = lines.append

    W("## Go/no-go verdict table\n")
    W("| Measurement | Pair | Verdict |")
    W("|---|---|---|")
    for m in ["m1", "m2", "m3", "m4"]:
        for pair in pairs:
            r = cell(pair, m, smoke)
            v = r["summary"]["verdict"] if r and "summary" in r else "SKIPPED (no result file)"
            W(f"| {m.upper()} | {pair} | {v} |")
    W("")

    for m in ["m1", "m2", "m3", "m4"]:
        for pair in pairs:
            r = cell(pair, m, smoke)
            if not r:
                W(f"## {m.upper()} ({pair}) - SKIPPED\n")
                continue
            W(f"## {m.upper()} ({pair})\n")
            if m == "m1":
                W(f"- pool={r['n_pool']} tokens, Ns={r['Ns']}, r90={r['r90']}")
                W("\n| layer frac | proj | N | mKNN k=5 | k=10 | k=30 |")
                W("|---|---|---|---|---|---|")
                for c in r["cells"]:
                    k = c["mknn"]
                    W(f"| {c['frac']} | {c['proj']} | {c['N']} | "
                      f"{k.get('5', k.get(5)):.3f} | {k.get('10', k.get(10)):.3f} | {k.get('30', k.get(30)):.3f} |")
                s = r["summary"]
                W(f"\nrow_ratio_min={s['row_ratio_min']}, null_share_max={s['null_share_max']}, "
                  f"fragility_slope={s['fragility_slope_per_decade']}")
            elif m == "m2":
                for frac, lr in r["layers"].items():
                    W(f"\n**layer frac {frac}** (raw / null mean / calibrated / sig):\n")
                    W("| metric | raw | null | calibrated | significant |")
                    W("|---|---|---|---|---|")
                    for k, v in lr.items():
                        W(f"| {k} | {v['raw']:.4f} | {v['null_mean']:.4f} | "
                          f"{v['calibrated']:.4f} | {v['significant']} |")
                W(f"\ncalibrated z (mid layer): {r['summary']['calibrated_z_mid_layer']}")
                W(f"note: {r['summary']['note']}")
            elif m == "m3":
                W(f"- n_tokens={r['n_tokens']}, grading: {r['grading']}")
                W(f"- token correlations: {r['token_correlations']}")
                W(f"- rollout-level vs correctness: {r['rollout_vs_correctness']}")
                W(f"- region split: {r['region_split']}")
                W(f"- logit-lens KL by student layer: {r['logitlens_kl_mean_by_layer']}")
            elif m == "m4":
                W(f"- block layer {r['block_layer']}, {r['n_params_probed']/1e6:.1f}M params, {r['steps']} steps")
                W(f"- OPRD-MSE: {r['oprd_mse']}")
                W(f"- OPD-CE:   {r['opd_ce']}")
                pa = r["power_analysis"]
                W(f"\n**Power analysis** ({pa['design']}, n_problems={pa['n_problems']}):\n")
                W("| pass rate | rho | effect (pts) | seeds/arm |")
                W("|---|---|---|---|")
                for t in pa["table"]:
                    W(f"| {t['pass_rate']} | {t['rho_within_problem']} | {t['effect_pts']:.0f} | {t['seeds_per_arm']} |")
            W(f"\n**Verdict:** {r['summary']['verdict']}\n")

    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
