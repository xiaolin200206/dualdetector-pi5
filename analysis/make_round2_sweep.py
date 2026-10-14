#!/usr/bin/env python3
"""
make_round2_sweep.py - tables and figure for the second round and the first (period x offset) sweep.

  python3 -I make_round2_sweep.py --round1 results/tables/runs.csv \
      --round2 data/campaign2 --sweep data/sweep1 --out results

Round 2 (campaign2): one run of each fixed-period condition, 60 min, steady window minutes 20-60,
power sampled with randomised intervals.  Sweep (sweep1): release period x offset, two blocks,
30 min per run, steady window minutes 10-30, default threading.

Prediction used for the sweep: with clean (non-overlapped) latencies L_leaf and L_pest, the two
detectors do not overlap if the pest release offset o satisfies  o >= L_leaf  and  o + L_pest <= P.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_results as M  # noqa: E402  (also sets the plot style)

PERIODS = [1.0, 1.5, 2.0]
FRACS = [0.0, 0.25, 0.5]
PCOL = {1.0: "#9E9AC8", 1.5: "#6A51A3", 2.0: "#2D004B"}   # sequential, distinct from the overlap/clear colours


def sweep_label(p, f):
    return f"p{int(round(p * 10)):02d}_o{int(round(f * 100)):02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round1", required=True, help="results/tables/runs.csv of round 1")
    ap.add_argument("--round2", required=True)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    td = os.path.join(a.out, "tables")
    fd = os.path.join(a.out, "figures")
    os.makedirs(td, exist_ok=True)
    os.makedirs(fd, exist_ok=True)

    # ---------------------------------------------------------------- round 2
    r1 = pd.read_csv(a.round1)
    r2, _ = M.load_runs(a.round2, 20, 60)
    r2 = r2.drop(columns=["dir"])
    r2.to_csv(os.path.join(td, "round2_runs.csv"), index=False)
    keep = ["leaf_median_ms", "pest_median_ms", "leaf_iqr_ms", "pest_iqr_ms", "compute_duty", "time_to_both_ms",
            "temp_mean_c", "fan_rpm_mean", "power_mean_w", "energy_per_inference_j",
            "power_sampling_gap", "power_sampling_ok", "power_idle_w"]
    comp = []
    for c in ["sync", "phase", "sync_t2", "phase_t2", "sync_ss", "phase_ss"]:
        x = r2[r2.condition == c].iloc[0]
        s = r1[r1.condition == c]
        row = {"condition": c}
        for k in keep:
            row[k + "_r2"] = x[k]
        for k in ["leaf_median_ms", "pest_median_ms", "temp_mean_c", "compute_duty", "fan_rpm_mean"]:
            row[k + "_r1_mean"] = s[k].mean()
            row[k + "_r1_min"] = s[k].min()
            row[k + "_r1_max"] = s[k].max()
        comp.append(row)
    comp = pd.DataFrame(comp)
    comp.to_csv(os.path.join(td, "round2_vs_round1.csv"), index=False)

    pw = []
    r2i = r2.set_index("condition")
    for ca, cb in [("sync", "phase"), ("sync_t2", "phase_t2"), ("sync_ss", "phase_ss")]:
        pa, pb = r2i.loc[ca, "power_mean_w"], r2i.loc[cb, "power_mean_w"]
        ea, eb = r2i.loc[ca, "energy_per_inference_j"], r2i.loc[cb, "energy_per_inference_j"]
        ta, tb = r2i.loc[ca, "temp_mean_c"], r2i.loc[cb, "temp_mean_c"]
        pw.append({"a": ca, "b": cb, "power_a_w": pa, "power_b_w": pb, "power_diff_w": pa - pb,
                   "power_diff_pct": 100 * (pa / pb - 1), "energy_a_j": ea, "energy_b_j": eb,
                   "energy_diff_pct": 100 * (ea / eb - 1), "temp_diff_c": ta - tb})
    pd.DataFrame(pw).to_csv(os.path.join(td, "round2_power_contrasts.csv"), index=False)

    # ------------------------------------------------------------------ sweep
    sw, sdirs = M.load_runs(a.sweep, 10, 30)
    sw["period"] = sw.condition.str.extract(r"p(\d+)_")[0].astype(int) / 10
    sw["frac"] = sw.condition.str.extract(r"_o(\d+)")[0].astype(int) / 100
    sw["offset_s"] = sw.period * sw.frac
    sw = sw.drop(columns=["dir"]).sort_values(["period", "frac", "rep"])
    sw.to_csv(os.path.join(td, "sweep_runs.csv"), index=False)

    clean = r1[r1.condition == "phase"]
    L_leaf, L_pest = clean.leaf_median_ms.mean() / 1000, clean.pest_median_ms.mean() / 1000
    rows = []
    for p in PERIODS:
        for f in FRACS:
            s = sw[(sw.period == p) & (sw.frac == f)]
            o = p * f
            pred_overlap = not (o >= L_leaf and o + L_pest <= p)
            row = {"period_s": p, "offset_frac": f, "offset_s": o, "n": len(s),
                   "predicted_overlap": pred_overlap,
                   "observed_overlap_any": bool((s.leaf_overlapped_frac > 0.5).any() or
                                                (s.leaf_overlap_share_mean > 0.02).any())}
            for k in ["leaf_median_ms", "pest_median_ms", "leaf_iqr_ms", "pest_iqr_ms",
                      "leaf_overlapped_frac", "leaf_overlap_share_mean", "compute_duty", "temp_mean_c",
                      "power_mean_w", "time_to_both_ms", "leaf_p99_ms", "pest_p99_ms"]:
                v = s[k].astype(float).dropna()
                row[k + "_mean"] = v.mean() if len(v) else np.nan
                row[k + "_min"] = v.min() if len(v) else np.nan
                row[k + "_max"] = v.max() if len(v) else np.nan
                row[k + "_n"] = len(v)
            rows.append(row)
    st = pd.DataFrame(rows)
    st.to_csv(os.path.join(td, "sweep_conditions.csv"), index=False)

    # sync minus clean (offset 0.5 P) at each period, same work rate
    cr = []
    for p in PERIODS:
        s0 = sw[(sw.period == p) & (sw.frac == 0.0)].set_index("rep")
        s5 = sw[(sw.period == p) & (sw.frac == 0.5)].set_index("rep")
        for m in ["leaf_median_ms", "pest_median_ms", "temp_mean_c", "power_mean_w"]:
            d = (s0[m] - s5[m]).dropna()
            cr.append({"period_s": p, "metric": m, "n_pairs": len(d), "diff_mean": d.mean(),
                       "diff_min": d.min(), "diff_max": d.max(),
                       "same_sign": int(max((d > 0).sum(), (d < 0).sum()))})
    pd.DataFrame(cr).to_csv(os.path.join(td, "sweep_sync_vs_half.csv"), index=False)

    # figure 7 (sweep)
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.5), constrained_layout=True)
    for p in PERIODS:
        s = sw[sw.period == p]
        g = s.groupby("offset_s")
        for j, (m, ttl) in enumerate([("leaf_median_ms", "a  Leaf detector"),
                                       ("pest_median_ms", "b  Pest detector")]):
            ax[j].scatter(s.offset_s, s[m], s=14, color=PCOL[p], alpha=0.55, zorder=3)
            ax[j].plot(g[m].mean().index, g[m].mean().values, color=PCOL[p], lw=1.1, zorder=2,
                       label=f"$P$ = {p:.1f} s")
            ax[j].set_title(ttl, loc="left")
            ax[j].set_xlabel("Pest release offset (s)")
            ax[j].set_ylabel("Median latency (ms)")
    ax[0].axvline(L_leaf, color=M.C_GR, ls="--", lw=0.8)
    ax[0].text(L_leaf + 0.02, ax[0].get_ylim()[1] * 0.97, "overlap-free leaf\nmedian (%.0f ms)" % (L_leaf * 1000),
               va="top", fontsize=7, color=M.C_GR)
    ax[0].legend(frameon=False, loc="center right")
    ok = sw.dropna(subset=["power_mean_w"])
    for p in PERIODS:
        s = ok[ok.period == p]
        ax[2].scatter(s.compute_duty, s.power_mean_w, s=18, color=PCOL[p], label=f"period {p:g} s", zorder=3)
    ax[2].set_title("c  PMIC power", loc="left")
    ax[2].set_xlabel("Compute duty")
    ax[2].set_ylabel("Mean PMIC power (W)")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fd, "fig07_sweep." + ext))
    plt.close(fig)

    facts = {"sweep_runs": int(len(sw)), "sweep_power_valid": int(sw.power_sampling_ok.sum()),
             "round2_runs": int(len(r2)), "round2_power_valid": int(r2.power_sampling_ok.sum()),
             "L_leaf_clean_s": L_leaf, "L_pest_clean_s": L_pest,
             "prediction_matches": int((st.predicted_overlap == st.observed_overlap_any).sum()),
             "prediction_total": int(len(st)),
             "sweep_cutoffs": int(sw.cutoffs.sum()), "round2_cutoffs": int(r2.cutoffs.sum()),
             "sweep_temp_max_c": float(sw.temp_max_c.max()), "round2_temp_max_c": float(r2.temp_max_c.max()),
             "sweep_arm_below_max_max": float(sw.arm_below_max_frac.max()),
             "round2_arm_below_max_max": float(r2.arm_below_max_frac.max()),
             "sweep_throttled_now_max": float(sw.throttled_now_frac.max()),
             "round2_throttled_now_max": float(r2.throttled_now_frac.max())}
    json.dump(facts, open(os.path.join(td, "facts2.json"), "w"), indent=1)
    print(json.dumps(facts, indent=1))
    print(st[["period_s", "offset_s", "predicted_overlap", "observed_overlap_any", "leaf_median_ms_mean",
              "leaf_overlap_share_mean_mean"]].round(3).to_string())


if __name__ == "__main__":
    main()
