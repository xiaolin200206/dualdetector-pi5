#!/usr/bin/env python3
"""
make_campaign3_sweep2.py - tables for the fixed-period serial baseline (campaign3) and the
dense offset sweep (sweep2), both added after the first three sets had been analysed.

  python3 -I make_campaign3_sweep2.py --round1 results/tables/runs.csv --sweep1 results/tables/sweep_runs.csv \
      --campaign3 data/campaign3 --sweep2 data/sweep2 --out results

campaign3: serialp, serialp_t2, serialp_ss; 3 reps, 30 min, steady window minutes 10-30.
sweep2:    P = 1.5 s, pest offsets 0.39, 0.42, 0.45, 0.50, 0.60 s; 2 reps, 30 min, window 10-30.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_results as M  # noqa: E402

C3 = ["serialp", "serialp_t2", "serialp_ss"]
PAIR = {"serialp": ("sync", "phase"), "serialp_t2": ("sync_t2", "phase_t2"), "serialp_ss": ("sync_ss", "phase_ss")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round1", required=True)
    ap.add_argument("--sweep1", required=True)
    ap.add_argument("--campaign3", required=True)
    ap.add_argument("--sweep2", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    td = os.path.join(a.out, "tables")
    os.makedirs(td, exist_ok=True)

    r1 = pd.read_csv(a.round1)
    s1 = pd.read_csv(a.sweep1)

    # ---------------------------------------------------------------- campaign3
    c3, _ = M.load_runs(a.campaign3, 10, 30)
    c3 = c3.drop(columns=["dir"])
    c3.to_csv(os.path.join(td, "campaign3_runs.csv"), index=False)
    met = ["leaf_median_ms", "leaf_iqr_ms", "leaf_p99_ms", "pest_median_ms", "pest_iqr_ms", "pest_p99_ms",
           "time_to_both_ms", "time_to_both_p90_ms", "compute_duty", "leaf_rate_per_min", "temp_mean_c",
           "temp_max_c", "power_mean_w", "power_sampling_gap", "leaf_late_p99_ms"]
    M.ORDER[:] = C3
    cond3 = M.cond_table(c3, met)
    # comparison with the main-campaign sync/phase means (minutes 20-60) at the same work rate
    rows = []
    for c in C3:
        s, p = PAIR[c]
        x = cond3[cond3.condition == c].iloc[0]
        rows.append({"serial_condition": c, "sync_condition": s, "phase_condition": p,
                     "serial_both_ms": x.time_to_both_ms_mean, "serial_both_min": x.time_to_both_ms_min,
                     "serial_both_max": x.time_to_both_ms_max,
                     "sync_both_ms": r1[r1.condition == s].time_to_both_ms.mean(),
                     "phase_both_ms": r1[r1.condition == p].time_to_both_ms.mean(),
                     "serial_leaf_median_ms": x.leaf_median_ms_mean, "serial_pest_median_ms": x.pest_median_ms_mean,
                     "phase_leaf_median_ms": r1[r1.condition == p].leaf_median_ms.mean(),
                     "phase_pest_median_ms": r1[r1.condition == p].pest_median_ms.mean()})
    comp = pd.DataFrame(rows)
    comp["sync_vs_serial_pct"] = 100 * (comp.sync_both_ms / comp.serial_both_ms - 1)
    cond3.to_csv(os.path.join(td, "campaign3_conditions.csv"), index=False)
    comp.to_csv(os.path.join(td, "campaign3_vs_round1.csv"), index=False)

    # ---------------------------------------------------------------- sweep2
    s2, _ = M.load_runs(a.sweep2, 10, 30)
    s2 = s2.drop(columns=["dir"])
    s2["offset_s"] = s2.run.str.extract(r"_s(\d{3})_")[0].astype(float) / 1000
    s2.to_csv(os.path.join(td, "sweep2_runs.csv"), index=False)
    # combined offset table at P = 1.5 s: sweep1 (0, 0.375, 0.75) + sweep2
    s1p = s1[s1.period == 1.5].copy()
    s1p["offset_s"] = s1p.frac * 1.5
    s1p["source"] = "sweep1"
    s2["source"] = "sweep2"
    keep = ["run", "source", "offset_s", "leaf_median_ms", "leaf_p99_ms", "pest_median_ms", "pest_p99_ms",
            "leaf_overlapped_frac", "leaf_overlap_share_mean", "time_to_both_ms", "time_to_both_p90_ms",
            "temp_mean_c", "power_mean_w", "power_sampling_gap"]
    allp = pd.concat([s1p[keep], s2[keep]]).sort_values(["offset_s", "run"])
    allp.to_csv(os.path.join(td, "offset_p15_runs.csv"), index=False)
    rows = []
    for o, g in allp.groupby("offset_s"):
        row = {"offset_s": o, "n": len(g), "source": ",".join(sorted(set(g.source)))}
        for k in keep[3:]:
            v = g[k].astype(float).dropna()
            row[k + "_mean"] = v.mean() if len(v) else np.nan
            row[k + "_min"] = v.min() if len(v) else np.nan
            row[k + "_max"] = v.max() if len(v) else np.nan
            row[k + "_n"] = len(v)
        rows.append(row)
    off = pd.DataFrame(rows)
    off.to_csv(os.path.join(td, "offset_p15_conditions.csv"), index=False)

    L_leaf = r1[r1.condition == "phase"].leaf_median_ms.mean() / 1000
    facts = {
        "campaign3_runs": int(len(c3)), "sweep2_runs": int(len(s2)),
        "serialp_both_ms": float(comp.loc[0, "serial_both_ms"]),
        "serialp_both_min": float(comp.loc[0, "serial_both_min"]), "serialp_both_max": float(comp.loc[0, "serial_both_max"]),
        "serialp_t2_both_ms": float(comp.loc[1, "serial_both_ms"]),
        "serialp_ss_both_ms": float(comp.loc[2, "serial_both_ms"]),
        "L_leaf_s": L_leaf,
        "contact_stops_between_s": [0.39, 0.42],
        "touched_frac_at_390": float(s2[s2.offset_s == 0.39].leaf_overlapped_frac.mean()),
        "share_at_390": float(s2[s2.offset_s == 0.39].leaf_overlap_share_mean.mean()),
        "touched_frac_at_420_max": float(s2[s2.offset_s == 0.42].leaf_overlapped_frac.max()),
        "both_at_420_ms": float(s2[s2.offset_s == 0.42].time_to_both_ms.mean()),
        "leaf_p99_at_390_max": float(s2[s2.offset_s == 0.39].leaf_p99_ms.max()),
        "leaf_p99_at_420_max": float(s2[s2.offset_s == 0.42].leaf_p99_ms.max()),
        "power_valid_campaign3": int(c3.power_sampling_ok.sum()), "power_valid_sweep2": int(s2.power_sampling_ok.sum()),
        "cutoffs": int(c3.cutoffs.sum() + s2.cutoffs.sum()), "slots_skipped": int(c3.slots_skipped.sum() + s2.slots_skipped.sum()),
        "temp_max_c": float(max(c3.temp_max_c.max(), s2.temp_max_c.max())),
    }
    json.dump(facts, open(os.path.join(td, "facts3.json"), "w"), indent=1)
    print(comp.round(1).to_string())
    print(off[["offset_s", "n", "leaf_median_ms_mean", "leaf_p99_ms_max", "leaf_overlapped_frac_mean",
               "leaf_overlap_share_mean_mean", "time_to_both_ms_mean"]].round(3).to_string())


if __name__ == "__main__":
    main()
