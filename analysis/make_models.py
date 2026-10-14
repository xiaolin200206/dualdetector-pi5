#!/usr/bin/env python3
"""
make_models.py - the timing relations of the paper, checked against the logs.

  python3 -I analysis/make_models.py --tables results/tables --campaign1 data/campaign1 \
      --replay data/replay1 --out results

Run after make_results.py, make_round2_sweep.py, make_campaign3_sweep2.py and make_replay.py.

Eq. (1), non-overlap condition, and the share of leaf inferences overlapped at an offset o.
   A pest inference released o after the leaf release overlaps the leaf inference if the leaf
   inference is still running, i.e. if L_leaf > o. With the overlap-free leaf latency distribution
   of the main campaign (phased runs, minutes 20-60), the expected share of overlapped leaf
   inferences is P(L_leaf > o). Eq. (1) with medians is o >= median(L_leaf) and
   o + median(L_pest) <= P; the same with p99 is the conservative form.

Eq. (2), overlap probability of two independent loops, (L_leaf + L_pest) / P_other.
   Checked on the main-campaign runs it was written from, and out of sample on the two
   independent-timer runs of the replay test (overlap-free latencies taken from the replay runs
   with o = 0.75 s, and alternatively from the camera runs).

Eq. (3), time from image onset to both answers (fixed-period policies, replay test).
   The leaf detector is released w after the onset, w ~ U(0, P); the pest detector at the first
   pest release after the onset (leaf release + o, modulo P), or, for fixed-period serial, when
   the leaf inference returns. Each answer arrives one per-call latency after its release.
   Evaluated with (a) per-call medians of the same replay run (a consistency check) and
   (b) per-call medians of the camera runs of the same policy (an out-of-sample prediction).

Also writes the share of images that had both answers within a given dwell time.

Outputs: tables/offset_rule.csv, tables/overlap_estimate.csv, tables/onset_model.csv,
tables/dwell_coverage.csv, tables/wait_distribution.csv, tables/facts_models.json, figures/fig08_replay.(png|pdf).
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyse as A  # noqa: E402
import make_results as M  # noqa: E402  (plot style and colours)

P = 1.5
WARMUP_S = 120.0
OFFSET = {"rp_sync": 0.0, "rp_phase42": 0.42, "rp_phase75": 0.75, "rp_serialp": None}
LAB = {"rp_indep": "Independent timers", "rp_serialp": "Fixed-period serial",
       "rp_sync": "Synchronous", "rp_phase42": "Phased, $o$ = 0.42 s", "rp_phase75": "Phased, $o$ = 0.75 s"}
STY = {"rp_indep": (M.C_GR, "-"), "rp_serialp": (M.C_GR, "--"), "rp_sync": (M.C_OV, "-"),
       "rp_phase42": (M.C_CL, "-"), "rp_phase75": (M.C_CL, ":")}
MK = {"rp_serialp": "s", "rp_phase42": "o", "rp_phase75": "D", "rp_sync": "^"}
DWELLS = (1.0, 1.25, 1.5, 2.0, 2.5)


def steady_inferences(run_dir, t_from_s, t_to_s=None):
    meta = json.load(open(os.path.join(run_dir, "meta.json")))
    inf = pd.read_csv(os.path.join(run_dir, "inferences.csv"))
    t0 = meta["run_start_s"]
    keep = inf.t_start >= t0 + t_from_s
    if t_to_s is not None:
        keep &= inf.t_start < t0 + t_to_s
    return inf[keep]


# ---------------------------------------------------------------- Eq. (1)
def offset_rule(tables, campaign1):
    lat = []
    for d in sorted(glob.glob(os.path.join(campaign1, "phase_r*"))):
        if "__failed" in d:
            continue
        inf = steady_inferences(d, 20 * 60, 60 * 60)
        lat.append(inf[inf.model == "leaf"].latency_ms.values / 1000)
    leaf = np.concatenate(lat)
    c = pd.read_csv(os.path.join(tables, "conditions.csv")).set_index("condition").loc["phase"]
    l_med, p_med = c.leaf_median_ms_mean / 1000, c.pest_median_ms_mean / 1000
    l_99, p_99 = c.leaf_p99_ms_mean / 1000, c.pest_p99_ms_mean / 1000
    s1 = pd.read_csv(os.path.join(tables, "sweep_conditions.csv"))
    s1 = s1[["period_s", "offset_s", "leaf_overlapped_frac_mean", "leaf_overlapped_frac_max"]].assign(sweep=1)
    s2 = pd.read_csv(os.path.join(tables, "offset_p15_conditions.csv"))
    s2 = s2[s2.source == "sweep2"][["offset_s", "leaf_overlapped_frac_mean", "leaf_overlapped_frac_max"]]
    s2 = s2.assign(period_s=1.5, sweep=2)
    t = pd.concat([s1, s2], ignore_index=True).sort_values(["period_s", "offset_s"])
    rows = []
    for x in t.itertuples():
        o, p = x.offset_s, x.period_s
        rows.append({"period_s": p, "offset_s": o, "sweep": x.sweep,
                     "predicted_median_rule": "overlap" if not (o >= l_med and o + p_med <= p) else "none",
                     "predicted_p99_rule": "overlap" if not (o >= l_99 and o + p_99 <= p) else "none",
                     "predicted_overlapped_frac": float((leaf > o).mean()) if o > 0 else 1.0,
                     "observed_overlapped_frac": x.leaf_overlapped_frac_mean,
                     "observed_overlapped_frac_max": x.leaf_overlapped_frac_max})
    return pd.DataFrame(rows), dict(L_leaf_median=l_med, L_pest_median=p_med, L_leaf_p99=l_99, L_pest_p99=p_99,
                                    n_leaf=int(leaf.size))


# ---------------------------------------------------------------- Eq. (2)
def overlap_estimate(tables, replay_dir):
    r = pd.read_csv(os.path.join(tables, "runs.csv"))
    ph = r[r.condition == "phase"]
    l_leaf, l_pest = ph.leaf_median_ms.mean() / 1000, ph.pest_median_ms.mean() / 1000
    rows = []
    for x in r[r.condition == "indep"].itertuples():
        rows.append({"set": "main campaign", "run": x.run, "latency_source": "main campaign, phased",
                     "leaf_period_s": x.leaf_period_s, "pest_period_s": x.pest_period_s,
                     "leaf_predicted": (l_leaf + l_pest) / x.pest_period_s, "leaf_observed": x.leaf_overlapped_frac,
                     "pest_predicted": (l_leaf + l_pest) / x.leaf_period_s, "pest_observed": x.pest_overlapped_frac})
    # replay: latencies of the overlap-free replay runs (o = 0.75 s)
    rl, rp = [], []
    for d in sorted(glob.glob(os.path.join(replay_dir, "rp_phase75_r*"))):
        inf = steady_inferences(d, WARMUP_S)
        rl.append(inf[inf.model == "leaf"].latency_ms.median() / 1000)
        rp.append(inf[inf.model == "pest"].latency_ms.median() / 1000)
    rl, rp = float(np.mean(rl)), float(np.mean(rp))
    for d in sorted(glob.glob(os.path.join(replay_dir, "rp_indep_r*"))):
        inf = steady_inferences(d, WARMUP_S)
        lf = inf[inf.model == "leaf"].sort_values("t_start")
        pe = inf[inf.model == "pest"].sort_values("t_start")
        pl, pp = float(np.diff(lf.t_start).mean()), float(np.diff(pe.t_start).mean())
        ol = A.overlap_with(lf[["t_start", "t_end"]].values, pe[["t_start", "t_end"]].values) > 0
        op = A.overlap_with(pe[["t_start", "t_end"]].values, lf[["t_start", "t_end"]].values) > 0
        for src, a, b in (("replay, phased 0.75 s", rl, rp), ("main campaign, phased", l_leaf, l_pest)):
            rows.append({"set": "replay", "run": os.path.basename(d), "latency_source": src,
                         "leaf_period_s": pl, "pest_period_s": pp,
                         "leaf_predicted": (a + b) / pp, "leaf_observed": float(ol.mean()),
                         "pest_predicted": (a + b) / pl, "pest_observed": float(op.mean())})
    sparse = {f"predicted_at_period_{p:g}s": (l_leaf + l_pest) / p for p in (10.0, 60.0)}
    return pd.DataFrame(rows), l_leaf, l_pest, sparse


# ---------------------------------------------------------------- Eq. (3)
def predict(ll, lp, cond):
    w = np.linspace(0.0, P, 30001, endpoint=False)          # wait until the next leaf release
    leaf = w + ll
    if OFFSET[cond] is None:                                 # serial: pest starts when leaf returns
        pest = w + ll + lp
    else:
        o = OFFSET[cond]
        wp = w + o - P
        wp = np.where(wp >= 0, wp, w + o)                     # first pest release after the onset
        pest = wp + lp
    return float(np.median(np.maximum(leaf, pest))), float(np.median(leaf))


def camera_latencies(tables):
    c = pd.read_csv(os.path.join(tables, "conditions.csv")).set_index("condition")
    s3 = pd.read_csv(os.path.join(tables, "campaign3_conditions.csv")).set_index("condition")
    o = pd.read_csv(os.path.join(tables, "offset_p15_conditions.csv")).set_index("offset_s")
    return {"rp_serialp": (s3.loc["serialp"].leaf_median_ms_mean, s3.loc["serialp"].pest_median_ms_mean),
            "rp_phase42": (o.loc[0.42].leaf_median_ms_mean, o.loc[0.42].pest_median_ms_mean),
            "rp_phase75": (c.loc["phase"].leaf_median_ms_mean, c.loc["phase"].pest_median_ms_mean),
            "rp_sync": (c.loc["sync"].leaf_median_ms_mean, c.loc["sync"].pest_median_ms_mean)}


def onset_model(tables, replay_dir, presentations):
    cam = camera_latencies(tables)
    rows = []
    for run in sorted(presentations.run.unique()):
        cond = run.rsplit("_r", 1)[0]
        if cond not in OFFSET:
            continue
        inf = steady_inferences(os.path.join(replay_dir, run), WARMUP_S)
        ll = inf[inf.model == "leaf"].latency_ms.median() / 1000
        lp = inf[inf.model == "pest"].latency_ms.median() / 1000
        both_same, leaf_same = predict(ll, lp, cond)
        both_cam, leaf_cam = predict(cam[cond][0] / 1000, cam[cond][1] / 1000, cond)
        ob = presentations[presentations.run == run]
        rows.append({"run": run, "condition": cond, "leaf_latency_s": ll, "pest_latency_s": lp,
                     "both_observed_median_s": float(ob.both_answers_s.median()),
                     "both_predicted_same_run_s": both_same, "both_predicted_camera_s": both_cam,
                     "leaf_observed_median_s": float(ob.leaf_first_answer_s.median()),
                     "leaf_predicted_same_run_s": leaf_same, "leaf_predicted_camera_s": leaf_cam,
                     "both_observed_max_s": float(ob.both_answers_s.max())})
    t = pd.DataFrame(rows)
    t["error_same_run_s"] = t.both_predicted_same_run_s - t.both_observed_median_s
    t["error_camera_s"] = t.both_predicted_camera_s - t.both_observed_median_s
    return t


def wait_distribution(replay_dir):
    """Empirical wait w from each image onset to the next leaf release (fixed-period runs)."""
    rows = []
    for d in sorted(glob.glob(os.path.join(replay_dir, "rp_*_r[0-9]"))):
        cond = os.path.basename(d).rsplit("_r", 1)[0]
        if cond not in OFFSET:
            continue
        meta = json.load(open(os.path.join(d, "meta.json")))
        inf = pd.read_csv(os.path.join(d, "inferences.csv"))
        rep = pd.read_csv(os.path.join(d, "replay.csv"))
        rep = rep[(rep.onset >= meta["run_start_s"] + WARMUP_S) & (rep.offset <= meta["run_end_s"])]
        rel = np.sort(inf[inf.model == "leaf"].t_sched.unique())
        k = np.searchsorted(rel, rep.onset.values)
        w = rel[k[k < len(rel)]] - rep.onset.values[k < len(rel)]
        hist = np.histogram(w, bins=6, range=(0, P))[0]
        rows.append({"run": os.path.basename(d), "n": int(w.size), "w_median_s": float(np.median(w)),
                     "w_mean_s": float(np.mean(w)), "bin_min": int(hist.min()), "bin_max": int(hist.max())})
    return pd.DataFrame(rows)


def dwell_coverage(presentations):
    p = presentations.assign(condition=presentations.run.str.rsplit("_r", n=1).str[0])
    rows = []
    for c, g in p.groupby("condition"):
        r = {"condition": c, "presentations": int(len(g))}
        for d in DWELLS:
            r[f"both_within_{d:g}s"] = float((g.both_answers_s <= d).mean())
        r["both_max_s"] = float(g.both_answers_s.max())
        rows.append(r)
    return pd.DataFrame(rows)


def figure(presentations, om, out):
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.8), gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axs[0]
    for c in ("rp_indep", "rp_serialp", "rp_phase42", "rp_phase75", "rp_sync"):
        v = np.sort(presentations[presentations.run.str.rsplit("_r", n=1).str[0] == c].both_answers_s.values)
        col, ls = STY[c]
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=col, ls=ls, lw=1.3, label=LAB[c])
    ax.set_xlabel("Time from image onset to both answers (s)")
    ax.set_ylabel("Cumulative share of images")
    ax.set_title("a  Replay, two runs per policy", loc="left")
    ax.set_xlim(0.4, 2.5)
    ax.legend(frameon=False, loc="lower right", fontsize=6.5)
    ax = axs[1]
    for c in ("rp_serialp", "rp_phase42", "rp_phase75", "rp_sync"):
        s = om[om.condition == c]
        col, _ = STY[c]
        ax.scatter(s.both_predicted_same_run_s, s.both_observed_median_s, s=24, color=col, marker=MK[c],
                   label=LAB[c], zorder=3)
        ax.scatter(s.both_predicted_camera_s, s.both_observed_median_s, s=24, marker=MK[c], zorder=3,
                   facecolors="none", edgecolors=col, linewidths=1)
    lo, hi = 1.15, 1.55
    ax.plot([lo, hi], [lo, hi], color=M.C_GR, ls="--", lw=0.8, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Predicted median (s)")
    ax.set_ylabel("Observed median (s)")
    ax.set_title("b  Eq. (3)", loc="left")
    from matplotlib.lines import Line2D
    h, l = ax.get_legend_handles_labels()
    h += [Line2D([], [], ls="", marker="o", color=M.C_GR, ms=4.5),
          Line2D([], [], ls="", marker="o", markerfacecolor="none", markeredgecolor=M.C_GR, ms=4.5)]
    l += ["latencies of the same run", "latencies of the camera runs"]
    ax.legend(h, l, frameon=False, loc="upper left", fontsize=6.0)
    fig.tight_layout()
    fig.savefig(out + ".png")
    fig.savefig(out + ".pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", required=True, help="results/tables written by the other scripts")
    ap.add_argument("--campaign1", required=True, help="folder with the main-campaign runs (data/campaign1)")
    ap.add_argument("--replay", required=True, help="folder with the replay runs (data/replay1)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    td, fd = os.path.join(a.out, "tables"), os.path.join(a.out, "figures")
    os.makedirs(td, exist_ok=True)
    os.makedirs(fd, exist_ok=True)

    rule, lat = offset_rule(a.tables, a.campaign1)
    rule.to_csv(os.path.join(td, "offset_rule.csv"), index=False)

    est, l_leaf, l_pest, sparse = overlap_estimate(a.tables, a.replay)
    est.to_csv(os.path.join(td, "overlap_estimate.csv"), index=False)

    pres = pd.read_csv(os.path.join(a.tables, "replay_presentations.csv"))
    om = onset_model(a.tables, a.replay, pres)
    om.to_csv(os.path.join(td, "onset_model.csv"), index=False)
    dw = dwell_coverage(pres)
    dw.to_csv(os.path.join(td, "dwell_coverage.csv"), index=False)
    figure(pres, om, os.path.join(fd, "fig08_replay"))
    wd = wait_distribution(a.replay)
    wd.to_csv(os.path.join(td, "wait_distribution.csv"), index=False)

    main_est = est[est.set == "main campaign"]
    rep_est = est[(est.set == "replay") & (est.latency_source == "replay, phased 0.75 s")]
    facts = {**{k: float(v) if not isinstance(v, int) else v for k, v in lat.items()},
             "leaf_overlap_predicted_main": [float(main_est.leaf_predicted.min()), float(main_est.leaf_predicted.max())],
             "pest_overlap_predicted_main": [float(main_est.pest_predicted.min()), float(main_est.pest_predicted.max())],
             "replay_leaf_overlap_predicted": [float(rep_est.leaf_predicted.min()), float(rep_est.leaf_predicted.max())],
             "replay_leaf_overlap_observed": [float(rep_est.leaf_observed.min()), float(rep_est.leaf_observed.max())],
             "replay_pest_overlap_predicted": [float(rep_est.pest_predicted.min()), float(rep_est.pest_predicted.max())],
             "replay_pest_overlap_observed": [float(rep_est.pest_observed.min()), float(rep_est.pest_observed.max())],
             **sparse,
             "wait_median_s": [float(wd.w_median_s.min()), float(wd.w_median_s.max())],
             "onset_abs_error_same_run_max_s": float(om.error_same_run_s.abs().max()),
             "onset_error_camera_min_s": float(om.error_camera_s.min()),
             "onset_error_camera_max_s": float(om.error_camera_s.max())}
    with open(os.path.join(td, "facts_models.json"), "w") as f:
        json.dump(facts, f, indent=1)
    pd.set_option("display.width", 200)
    for t in (rule, est, om, dw):
        print(t.round(4).to_string())
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
