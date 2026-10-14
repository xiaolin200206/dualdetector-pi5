#!/usr/bin/env python3
"""
make_results.py - main campaign: Tables 4-5 and Figs 2-6 of the manuscript, and the per-run tables used by the later scripts.

  python3 -I make_results.py CAMPAIGN_DIR OUT_DIR [--warmup-min 20] [--max-min 60]

CAMPAIGN_DIR holds one directory per run (inferences.csv, system.csv, power.csv,
events.csv, meta.json). Runs whose directory name contains "__failed" are ignored.
Only the first --max-min minutes of each run are used, and of those the last
(max-min - warmup-min) minutes for the steady-state statistics, so that all runs
are compared over the same window even if a run was longer.
"""
import argparse
import glob
import json
import os

# reproducible PDF output (no creation date in the files)
os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyse as A  # noqa: E402

ORDER = ["indep", "serial", "sync", "phase", "sync_t2", "phase_t2", "sync_ss", "phase_ss",
         "serialp", "serialp_t2", "serialp_ss"]
LABEL = {
    "indep": "Independent timers", "serial": "Serialised cycle",
    "sync": "Synchronous", "phase": "Phased",
    "sync_t2": "Synchronous, 2 threads", "phase_t2": "Phased, 2 threads",
    "sync_ss": "Synchronous, s + s", "phase_ss": "Phased, s + s",
    "serialp": "Serial, fixed period", "serialp_t2": "Serial, fixed period, 2 threads",
    "serialp_ss": "Serial, fixed period, s+s",
}
PAIRS = [("sync", "phase"), ("sync_t2", "phase_t2"), ("sync_ss", "phase_ss")]
C_OV, C_CL, C_GR, C_OR = "#D55E00", "#0072B2", "#6B6B6B", "#E69F00"   # overlap, clear, legacy, accent

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.spines.top": False,
    "axes.spines.right": False, "axes.linewidth": 0.7, "figure.dpi": 150,
    "savefig.dpi": 300, "font.family": "DejaVu Sans", "axes.grid": True,
    "grid.color": "#E3E3E3", "grid.linewidth": 0.5, "axes.axisbelow": True,
    "text.hinting": "no_hinting",
})


def load_runs(camp, warmup, maxmin):
    rows, dirs = [], {}
    for d in sorted(glob.glob(os.path.join(camp, "*"))):
        if "__failed" in d or not os.path.exists(os.path.join(d, "meta.json")):
            continue
        r = A.analyse_run(d, warmup, 2.0, maxmin)
        r["rep"] = int(r["run"].rsplit("_r", 1)[1])
        r["dir"] = d
        rows.append(r)
        dirs[r["run"]] = d
    df = pd.DataFrame(rows)
    df = df[df.status == "ok"].copy()
    return df, dirs


def run_series(d, maxmin):
    meta = json.load(open(os.path.join(d, "meta.json")))
    t0 = meta["run_start_s"]
    inf = pd.read_csv(os.path.join(d, "inferences.csv"))
    inf["ts"] = inf.t_start - t0
    inf["te"] = inf.t_end - t0
    inf = inf[inf.te <= maxmin * 60].copy()
    sysr = pd.read_csv(os.path.join(d, "system.csv"))
    sysr = sysr[sysr.phase == "run"].copy()
    sysr["m"] = (sysr.t_s - t0) / 60
    return meta, inf, sysr[sysr.m <= maxmin]


def spectral_peak_period(d, warmup, maxmin, lo=3.0, hi=30.0, dt=0.05):
    """Dominant period (s) of the leaf latency series between lo and hi seconds.
    The latency series is interpolated onto a regular grid, Hann-windowed, and the
    largest periodogram peak in [1/hi, 1/lo] Hz is returned."""
    meta, inf, _ = run_series(d, maxmin)
    L = inf[(inf.model == "leaf") & (inf.ts >= warmup * 60)].sort_values("ts")
    grid = np.arange(L.ts.iloc[0], L.ts.iloc[-1], dt)
    x = np.interp(grid, L.ts.values, L.latency_ms.values)
    x = (x - x.mean()) * np.hanning(len(x))
    p = np.abs(np.fft.rfft(x)) ** 2
    fr = np.fft.rfftfreq(len(x), dt)
    sel = (fr > 1 / hi) & (fr < 1 / lo)
    return float(1 / fr[np.argmax(p * sel)])


def extras(df, dirs, warmup, maxmin):
    """Per-run extras not in analyse.py: time-to-both for sync slots, drift."""
    d1, d2 = [], []
    for _, r in df.iterrows():
        meta, inf, sysr = run_series(r["dir"], maxmin)
        # drift of die temperature between the two halves of the steady window
        mid = (warmup + maxmin) / 2
        a = sysr[(sysr.m >= warmup) & (sysr.m < mid)].temp_c.mean()
        b = sysr[(sysr.m >= mid) & (sysr.m <= maxmin)].temp_c.mean()
        d1.append(a)
        d2.append(b)
    df["leaf_latency_peak_period_s"] = [
        spectral_peak_period(r["dir"], warmup, maxmin) if r["policy"] == "indep" else np.nan
        for _, r in df.iterrows()]
    df["temp_first_half"] = d1
    df["temp_second_half"] = d2
    df["temp_drift_c"] = df.temp_second_half - df.temp_first_half
    return df


def cond_table(df, metrics):
    out = []
    for c in [c for c in ORDER if c in set(df.condition)]:
        s = df[df.condition == c]
        row = {"condition": c, "n": len(s)}
        for m in metrics:
            v = s[m].astype(float).dropna()
            row[m + "_mean"] = v.mean() if len(v) else np.nan
            row[m + "_sd"] = v.std(ddof=1) if len(v) > 1 else np.nan
            row[m + "_min"] = v.min() if len(v) else np.nan
            row[m + "_max"] = v.max() if len(v) else np.nan
        out.append(row)
    return pd.DataFrame(out)


def contrasts(df, pairs, metrics):
    """Paired by replicate (block): first minus second."""
    rows = []
    for a, b in pairs:
        for m in metrics:
            x = df[df.condition == a].set_index("rep")[m]
            y = df[df.condition == b].set_index("rep")[m]
            d = (x - y).dropna()
            if not len(d):
                continue
            rows.append({"a": a, "b": b, "metric": m, "n_pairs": len(d),
                         "diff_mean": d.mean(), "diff_min": d.min(), "diff_max": d.max(),
                         "same_sign": int(max((d > 0).sum(), (d < 0).sum())),
                         "ratio_mean": (x / y).dropna().mean()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures

def fig_timeline(df, maxmin, out):
    """Fig. 2: 14 s of real inference intervals for four policies."""
    pol = [("indep", "Independent timers"), ("serial", "Serialised cycle"),
           ("sync", "Synchronous release"), ("phase", "Phased release")]
    t_a = 30 * 60
    win = 14
    fig, axs = plt.subplots(4, 1, figsize=(7.1, 3.5), sharex=True)
    for ax, (c, name) in zip(axs, pol):
        r = df[(df.condition == c) & (df.rep == 1)].iloc[0]
        meta, inf, _ = run_series(r["dir"], maxmin)
        w = inf[(inf.ts >= t_a) & (inf.ts < t_a + win)]
        for y, m, col in ((1, "leaf", C_CL), (0, "pest", C_OR)):
            s = w[w.model == m]
            ax.barh([y] * len(s), s.te - s.ts, left=s.ts - t_a, height=0.7, color=col,
                    edgecolor="white", linewidth=0.4)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["pest", "leaf"])
        ax.set_ylim(-0.6, 1.6)
        ax.set_title(name, loc="left", fontsize=8, pad=2)
        ax.grid(axis="y", visible=False)
    axs[-1].set_xlabel("Time within a 14 s excerpt of minute 30 (s)")
    axs[-1].set_xlim(0, win)
    fig.tight_layout(h_pad=0.4)
    fig.savefig(out + ".png")
    fig.savefig(out + ".pdf")
    plt.close(fig)


def fig_indep(df, maxmin, warmup, out):
    """Fig. 3: leaf latency under independent timers; overlap dose-response."""
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6), gridspec_kw={"width_ratios": [1.5, 1.15, 1]})
    r = df[(df.condition == "indep") & (df.rep == 1)].iloc[0]
    meta, inf, _ = run_series(r["dir"], maxmin)
    leaf = inf[inf.model == "leaf"].sort_values("ts")
    pest = inf[inf.model == "pest"].sort_values("ts")
    t_a = 30 * 60
    w = leaf[(leaf.ts >= t_a) & (leaf.ts < t_a + 60)]
    ov = A.overlap_with(leaf[["t_start", "t_end"]].values, pest[["t_start", "t_end"]].values) > 0
    leaf = leaf.assign(ov=ov)
    w = leaf[(leaf.ts >= t_a) & (leaf.ts < t_a + 60)]
    ax = axs[0]
    ax.plot(w.ts - t_a, w.latency_ms, color=C_GR, lw=0.8, zorder=1)
    ax.scatter(w.ts[~w.ov] - t_a, w.latency_ms[~w.ov], s=11, color=C_CL, label="no overlap", zorder=2)
    ax.scatter(w.ts[w.ov] - t_a, w.latency_ms[w.ov], s=11, color=C_OV, marker="s", label="overlaps a pest inference", zorder=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Leaf latency (ms)")
    ax.set_title("a  One minute under independent timers", loc="left")
    ax.legend(frameon=False, loc="upper left", ncol=1, handletextpad=0.3)
    ax.set_ylim(300, max(900, w.latency_ms.max() * 1.1))
    # b: distribution all indep runs
    ax = axs[1]
    allv = []
    for _, r in df[df.condition == "indep"].iterrows():
        meta, inf, _ = run_series(r["dir"], maxmin)
        inf = inf[inf.ts >= warmup * 60]
        l = inf[inf.model == "leaf"].sort_values("ts")
        p = inf[inf.model == "pest"].sort_values("ts")
        o = A.overlap_with(l[["t_start", "t_end"]].values, p[["t_start", "t_end"]].values)
        share = o / np.maximum((l.t_end - l.t_start).values, 1e-9)
        allv.append(pd.DataFrame({"lat": l.latency_ms.values, "share": share}))
    a = pd.concat(allv)
    bins = [-0.001, 1e-9, 0.25, 0.5, 0.75, 1.0001]
    lab = ["0", "0–25", "25–50", "50–75", ">75"]  # percent of the leaf inference overlapped
    a["bin"] = pd.cut(a.share, bins=bins, labels=lab)
    g = a.groupby("bin", observed=True).lat
    med = g.median()
    q1, q3 = g.quantile(.25), g.quantile(.75)
    n = g.size()
    x = np.arange(len(med))
    ax.errorbar(x, med.values, yerr=[med - q1, q3 - med], fmt="o", color=C_OV, ms=4, capsize=2, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(med.index.astype(str), fontsize=6.5)
    ax.tick_params(axis="x", pad=1)
    ax.set_xlabel("Share of the leaf inference\noverlapped by pest (%)")
    ax.set_ylabel("Leaf latency (ms)")
    ax.set_title("b  Dose–response", loc="left")
    # c: episode spacing vs beat
    ax = axs[2]
    sp = df[df.condition == "indep"][["leaf_latency_peak_period_s", "beat_predicted_s"]].astype(float)
    ax.scatter(sp.beat_predicted_s, sp.leaf_latency_peak_period_s, s=22, color=C_OV, zorder=3)
    lo, hi = 9.4, 10.2
    ax.plot([lo, hi], [lo, hi], color=C_GR, lw=0.8, ls="--", zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Beat period from the\nmeasured periods (s)")
    ax.set_ylabel("Spectral peak of the\nleaf latency series (s)")
    ax.set_title("c  Beat period", loc="left")
    fig.tight_layout()
    fig.savefig(out + ".png")
    fig.savefig(out + ".pdf")
    plt.close(fig)
    return a


def fig_pairs(df, out):
    """Fig. 4: median latency, overlap on vs off, three configurations."""
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.6), sharey=False)
    cfg = [("sync", "phase", "s + n\ndefault"),
           ("sync_t2", "phase_t2", "s + n\n2 threads"),
           ("sync_ss", "phase_ss", "s + s\ndefault")]
    for ax, m, t in zip(axs, ("leaf_median_ms", "pest_median_ms"), ("a  Leaf detector", "b  Pest detector")):
        for i, (s, p, _) in enumerate(cfg):
            for j, (c, col, mk) in enumerate(((p, C_CL, "o"), (s, C_OV, "s"))):
                v = df[df.condition == c][m].astype(float).values
                x = i + (j - 0.5) * 0.34
                ax.bar(x, v.mean(), width=0.3, color=col, alpha=0.9, hatch="" if j == 0 else "////",
                       edgecolor="white" if j == 0 else "white", linewidth=0.5,
                       label=("Phased (no overlap)" if j == 0 else "Synchronous (full overlap)") if i == 0 else None)
                ax.scatter([x] * len(v), v, s=9, color="black", zorder=3, marker=mk if False else "o")
                ax.text(x, v.max() + 18, f"{v.mean():.0f}", ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(3))
        ax.set_xticklabels([c[2] for c in cfg])
        ax.set_ylabel("Median latency (ms)")
        ax.set_title(t, loc="left")
        ax.set_ylim(0, 1250)
        ax.grid(axis="x", visible=False)
    axs[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out + ".png")
    fig.savefig(out + ".pdf")
    plt.close(fig)


def fig_ecdf(df, maxmin, warmup, out):
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 3.1), sharey=True)
    sty = {"indep": (C_GR, "-"), "serial": (C_GR, "--"), "sync": (C_OV, "-"), "phase": (C_CL, "-"),
           "sync_t2": (C_OV, "--"), "phase_t2": (C_CL, "--"), "sync_ss": (C_OV, ":"), "phase_ss": (C_CL, ":")}
    for ax, m, t in zip(axs, ("leaf", "pest"), ("a  Leaf detector", "b  Pest detector")):
        for c in ORDER:
            if c not in sty or not (df.condition == c).any():
                continue
            vals = []
            for _, r in df[df.condition == c].iterrows():
                meta, inf, _ = run_series(r["dir"], maxmin)
                vals.append(inf[(inf.model == m) & (inf.ts >= warmup * 60)].latency_ms.values)
            v = np.sort(np.concatenate(vals))
            col, ls = sty[c]
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=col, ls=ls, lw=1.3, label=LABEL[c])
        ax.set_xlabel("Latency (ms)")
        ax.set_title(t, loc="left")
        ax.set_xlim(100 if m == "pest" else 300, 1300)
    axs[0].set_ylabel("Cumulative share of inferences")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=4, fontsize=6.8, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(out + ".png")
    fig.savefig(out + ".pdf")
    plt.close(fig)


def fig_thermal(df, maxmin, out):
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 3.0))
    cols = {"indep": C_GR, "serial": C_GR, "sync": C_OV, "phase": C_CL, "sync_t2": C_OV,
            "phase_t2": C_CL, "sync_ss": C_OV, "phase_ss": C_CL}
    mk = {"indep": "o", "serial": "s", "sync": "o", "phase": "o", "sync_t2": "^", "phase_t2": "^",
          "sync_ss": "D", "phase_ss": "D"}
    filled = lambda c: c.startswith("sync") or c in ("indep",)
    ax = axs[0]
    for c in ("sync", "phase", "sync_ss", "phase_ss"):
        ms = []
        for _, r in df[df.condition == c].iterrows():
            meta, inf, s = run_series(r["dir"], maxmin)
            s = s.copy()
            s["mb"] = np.floor(s.m)
            ms.append(s.groupby("mb").temp_c.mean())
        mm = pd.concat(ms, axis=1).mean(axis=1)
        ax.plot(mm.index + 0.5, mm.values, color=cols[c], lw=1.3,
                ls="-" if "ss" not in c else ":", label=LABEL[c])
    ax.set_xlabel("Minutes since start")
    ax.set_ylabel("Die temperature (°C)")
    ax.set_title("a  Warm-up", loc="left")
    ax.legend(frameon=False, fontsize=6.0, loc="center", bbox_to_anchor=(0.6, 0.41), ncol=2,
              handlelength=1.6, columnspacing=0.6)
    ax = axs[1]
    for c in [c for c in ORDER if c in cols]:
        s = df[df.condition == c]
        ax.scatter(s.compute_duty, s.temp_mean_c, s=22, color=cols[c], marker=mk[c],
                   facecolors=cols[c] if filled(c) else "none", linewidths=1, zorder=3)
    ax.set_xlabel("Compute duty, $D$")
    ax.set_ylabel("Steady die temperature (°C)")
    ax.set_title("b  One point per run", loc="left")
    ax = axs[2]
    for c in [c for c in ORDER if c in cols]:
        s = df[df.condition == c]
        ax.scatter(s.temp_mean_c, s.fan_rpm_mean, s=22, color=cols[c], marker=mk[c],
                   facecolors=cols[c] if filled(c) else "none", linewidths=1, zorder=3)
    ax.set_xlabel("Steady die temperature (°C)")
    ax.set_ylabel("Fan speed (min$^{-1}$)")
    ax.set_title("c  Fan response", loc="left")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], ls="", marker=mk[c], ms=5, mew=1, color=cols[c],
                 markerfacecolor=cols[c] if filled(c) else "none", label=LABEL[c])
          for c in ORDER if c in cols]
    fig.legend(handles=hs, frameon=False, fontsize=6.3, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out + ".png")
    fig.savefig(out + ".pdf")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign")
    ap.add_argument("out")
    ap.add_argument("--warmup-min", type=float, default=20)
    ap.add_argument("--max-min", type=float, default=60)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "figures"), exist_ok=True)
    df, dirs = load_runs(a.campaign, a.warmup_min, a.max_min)
    df = extras(df, dirs, a.warmup_min, a.max_min)
    df.drop(columns=["dir"]).to_csv(os.path.join(a.out, "tables", "runs.csv"), index=False)

    M = ["leaf_median_ms", "leaf_iqr_ms", "leaf_p90_ms", "leaf_p99_ms", "pest_median_ms", "pest_iqr_ms",
         "pest_p90_ms", "pest_p99_ms", "leaf_overlapped_frac", "pest_overlapped_frac", "compute_duty",
         "leaf_rate_per_min", "pest_rate_per_min", "time_to_both_ms", "temp_mean_c", "temp_rise_c",
         "temp_max_c", "temp_sd_c", "idle_temp_c", "fan_rpm_mean", "capture_fps_mean", "cpu_pct_mean",
         "temp_drift_c", "power_mean_w", "energy_per_inference_j", "leaf_median_overlapped_ms",
         "leaf_median_clear_ms", "pest_median_overlapped_ms", "pest_median_clear_ms",
         "beat_predicted_s", "leaf_latency_peak_period_s", "leaf_episode_spacing_median_s",
         "time_to_both_p90_ms", "leaf_frame_age_median_ms", "pest_frame_age_median_ms",
         "leaf_late_median_ms", "leaf_late_p99_ms"]
    M = [m for m in M if m in df.columns]
    cond_table(df, M).to_csv(os.path.join(a.out, "tables", "conditions.csv"), index=False)
    cm = ["leaf_median_ms", "pest_median_ms", "leaf_p90_ms", "pest_p90_ms", "leaf_iqr_ms", "pest_iqr_ms",
          "compute_duty", "temp_mean_c", "temp_max_c", "fan_rpm_mean", "capture_fps_mean", "cpu_pct_mean"]
    c1 = contrasts(df, PAIRS, cm)
    c2 = contrasts(df, [("phase_t2", "phase"), ("indep", "phase"), ("serial", "phase"),
                        ("sync_ss", "sync"), ("phase_ss", "phase")], cm)
    pd.concat([c1, c2]).to_csv(os.path.join(a.out, "tables", "contrasts.csv"), index=False)
    fd = os.path.join(a.out, "figures")
    fig_timeline(df, a.max_min, os.path.join(fd, "fig02_timelines"))
    dose = fig_indep(df, a.max_min, a.warmup_min, os.path.join(fd, "fig03_independent"))
    g = dose.groupby(pd.cut(dose.share, [-0.001, 1e-9, 0.25, 0.5, 0.75, 1.0001],
                            labels=["0", "0-25", "25-50", "50-75", ">75"]), observed=True).lat
    pd.DataFrame({"n": g.size(), "median_ms": g.median(), "q1": g.quantile(.25), "q3": g.quantile(.75)}
                 ).to_csv(os.path.join(a.out, "tables", "indep_dose_response.csv"))
    fig_pairs(df, os.path.join(fd, "fig04_overlap_on_off"))
    fig_ecdf(df, a.max_min, a.warmup_min, os.path.join(fd, "fig05_latency_ecdf"))
    fig_thermal(df, a.max_min, os.path.join(fd, "fig06_thermal"))
    # whole-campaign facts
    facts = {"n_runs": int(len(df)), "n_inferences": int(df.leaf_n.sum() + df.pest_n.sum()),
             "cutoffs": int(df.cutoffs.sum()), "throttled_now_max": float(df.throttled_now_frac.max()),
             "arm_below_max_max": float(df.arm_below_max_frac.max()),
             "temp_max_c": float(df.temp_max_c.max()),
             "slots_skipped": int(df.slots_skipped.sum()),
             "spearman_temp_vs_duty": float(df[["compute_duty", "temp_mean_c"]].corr(method="spearman").iloc[0, 1]),
             "power_valid_runs": int(df.power_sampling_ok.sum()) if "power_sampling_ok" in df else 0}
    json.dump(facts, open(os.path.join(a.out, "tables", "facts.json"), "w"), indent=1)
    print(json.dumps(facts, indent=1))


if __name__ == "__main__":
    main()
