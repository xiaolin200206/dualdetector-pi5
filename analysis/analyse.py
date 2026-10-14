#!/usr/bin/env python3
"""
analyse.py - summarise a dual_bench campaign.

  python3 analyse.py campaign1                 # a run_matrix output directory
  python3 analyse.py campaign1/sync_r1 ...     # or individual run directories

Writes <first dir>/runs_summary.csv (one row per run) and
<first dir>/conditions_summary.csv (mean, sd and n over replicates), and prints
the contrasts the study is built on.

Definitions (stated here so that the paper can cite them verbatim)
  overlap       an inference is "overlapped" if its [t_start, t_end] interval
                intersects any interval of the other detector; overlap_frac is
                the intersected share of its own duration.
  compute duty  sum of inference durations / run duration (can exceed 1 when
                the detectors overlap); busy fraction is the union instead.
  slow          latency > 1.5 x the run's own median for that detector.
  episode       a maximal run of consecutive slow leaf inferences whose starts
                are no more than --episode-gap s apart; spacing is the interval
                between successive episode onsets.
  beat          1 / |1/P_leaf - 1/P_pest| with P the realised mean start-to-
                start periods; reported for every policy, meaningful for indep.
  steady window thermal and power statistics exclude the first --warmup-min
                minutes of the run.
"""

import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def load_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def col(rows, key, cond=None):
    return np.array([fnum(r[key]) for r in rows if cond is None or cond(r)], dtype=float)


def nanstat(fn, a):
    a = a[~np.isnan(a)]
    return float(fn(a)) if a.size else math.nan


def overlap_with(a_iv, b_iv):
    """For each interval in a_iv, total intersection (s) with intervals b_iv.
    Both are (n, 2) arrays sorted by start."""
    out = np.zeros(len(a_iv))
    if len(b_iv) == 0:
        return out
    j0 = 0
    for i, (s, e) in enumerate(a_iv):
        while j0 < len(b_iv) and b_iv[j0, 1] <= s:
            j0 += 1
        j = j0
        tot = 0.0
        while j < len(b_iv) and b_iv[j, 0] < e:
            tot += max(0.0, min(e, b_iv[j, 1]) - max(s, b_iv[j, 0]))
            j += 1
        out[i] = tot
    return out


def union_length(iv):
    if len(iv) == 0:
        return 0.0
    iv = iv[np.argsort(iv[:, 0])]
    tot, cs, ce = 0.0, iv[0, 0], iv[0, 1]
    for s, e in iv[1:]:
        if s > ce:
            tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return tot + ce - cs


POWER_GAP_MAX = 0.03     # largest tolerated |sampled - true| busy share


def inside(iv, t):
    """True where time t falls inside one of the sorted intervals iv."""
    i = np.searchsorted(iv[:, 0], t, side="right") - 1
    ok = i >= 0
    out = np.zeros(len(t), bool)
    out[ok] = t[ok] <= iv[i[ok], 1]
    return out


def episodes(starts, slow, gap):
    onsets, last = [], None
    for t, s in zip(starts, slow):
        if not s:
            continue
        if last is None or t - last > gap:
            onsets.append(t)
        last = t
    return onsets


def latency_stats(lat, prefix):
    d = {}
    if lat.size == 0:
        return d
    q1, med, q3 = np.percentile(lat, [25, 50, 75])
    d[f"{prefix}_n"] = int(lat.size)
    d[f"{prefix}_median_ms"] = med
    d[f"{prefix}_iqr_ms"] = q3 - q1
    d[f"{prefix}_p90_ms"] = float(np.percentile(lat, 90))
    d[f"{prefix}_p99_ms"] = float(np.percentile(lat, 99))
    d[f"{prefix}_mean_ms"] = float(lat.mean())
    d[f"{prefix}_sd_ms"] = float(lat.std(ddof=1)) if lat.size > 1 else math.nan
    d[f"{prefix}_slow_frac"] = float((lat > 1.5 * med).mean())
    return d


def analyse_run(rundir, warmup_min, episode_gap, max_min=None):
    meta = json.load(open(os.path.join(rundir, "meta.json")))
    label = meta.get("label", os.path.basename(rundir))
    cond = label.rsplit("_r", 1)[0] if "_r" in label else label
    r = {"run": label, "condition": cond, "status": meta.get("status"),
         "policy": meta["args"].get("policy"),
         "ort_threads": meta["args"].get("ort_threads"),
         "camera": meta.get("camera"), "origin_wall": meta.get("origin_wall"),
         "governor": (meta.get("versions") or {}).get("governor")}
    t0 = fnum(meta.get("run_start_s"))
    t1 = fnum(meta.get("run_end_s"))
    if math.isnan(t0) or math.isnan(t1):
        r["note"] = "run did not start or did not finish"
        return r
    if max_min is not None and (t1 - t0) > max_min * 60:
        t1 = t0 + max_min * 60          # analyse the common window only
    dur = t1 - t0
    ts = t0 + warmup_min * 60
    r["duration_min"] = dur / 60

    # All inference statistics use the steady window [ts, t1]: inferences that
    # started at or after warmup and ended by t1. The same window is used for the
    # thermal and power statistics below.
    inf = [x for x in load_csv(os.path.join(rundir, "inferences.csv"))
           if fnum(x["t_end"]) <= t1 and fnum(x["t_start"]) >= ts]
    win = t1 - ts
    iv, starts, lat = {}, {}, {}
    for m in ("leaf", "pest"):
        rows = [x for x in inf if x["model"] == m]
        a = np.array([[fnum(x["t_start"]), fnum(x["t_end"])] for x in rows]).reshape(-1, 2)
        a = a[np.argsort(a[:, 0])] if len(a) else a
        iv[m] = a
        starts[m] = a[:, 0] if len(a) else np.array([])
        lat[m] = (a[:, 1] - a[:, 0]) * 1000 if len(a) else np.array([])
        r[f"{m}_rate_per_min"] = len(a) / win * 60
        r.update(latency_stats(lat[m], m))
        fa = np.array([fnum(x["frame_age_ms"]) for x in rows])
        fa = fa[~np.isnan(fa)]
        r[f"{m}_frame_age_median_ms"] = float(np.median(fa)) if fa.size else math.nan
        r[f"{m}_frame_age_p90_ms"] = float(np.percentile(fa, 90)) if fa.size else math.nan
        if len(a) > 2:
            r[f"{m}_period_s"] = float(np.diff(starts[m]).mean())

    # Time to both answers, from the common release instant (fixed-period policies,
    # paired by t_sched) or from the start of the leaf inference (serialised cycle,
    # paired by idx). Phased release counts the pest offset as waiting time.
    # Pairs are matched by slot index (idx); the origin is the leaf release instant
    # (t_sched) for the fixed-period policies and the leaf start for the serialised cycle.
    key = "idx" if r["policy"] in ("sync", "phase", "serial", "serialp") else None
    if key:
        pairs = {}
        for x in inf:
            k = x[key]
            if k:
                pairs.setdefault(k, {})[x["model"]] = x
        both = []
        for k, p in pairs.items():
            if "leaf" in p and "pest" in p:
                origin = fnum(p["leaf"]["t_sched"]) if r["policy"] != "serial" else fnum(p["leaf"]["t_start"])
                if math.isnan(origin):
                    continue
                both.append((max(fnum(p["leaf"]["t_end"]), fnum(p["pest"]["t_end"])) - origin) * 1000)
        if both:
            both = np.array(both)
            r["time_to_both_ms"] = float(np.median(both))
            r["time_to_both_p90_ms"] = float(np.percentile(both, 90))
            r["time_to_both_n"] = int(both.size)
        # release lateness of the leaf detector (fixed-period policies only)
        if r["policy"] in ("sync", "phase", "serialp"):
            late = np.array([fnum(x["late_ms"]) for x in inf if x["model"] == "leaf"])
            late = late[~np.isnan(late)]
            if late.size:
                r["leaf_late_median_ms"] = float(np.median(late))
                r["leaf_late_p99_ms"] = float(np.percentile(late, 99))

    for m, o in (("leaf", "pest"), ("pest", "leaf")):
        if len(iv[m]) == 0:
            continue
        ov = overlap_with(iv[m], iv[o])
        frac = ov / np.maximum(iv[m][:, 1] - iv[m][:, 0], 1e-9)
        hit = ov > 0
        r[f"{m}_overlapped_frac"] = float(hit.mean())
        r[f"{m}_overlap_share_mean"] = float(frac.mean())
        r[f"{m}_median_overlapped_ms"] = float(np.median(lat[m][hit])) if hit.any() else math.nan
        r[f"{m}_median_clear_ms"] = float(np.median(lat[m][~hit])) if (~hit).any() else math.nan

    allv = np.vstack([iv["leaf"], iv["pest"]]) if len(iv["leaf"]) + len(iv["pest"]) else np.zeros((0, 2))
    r["compute_duty"] = float((allv[:, 1] - allv[:, 0]).sum() / win) if len(allv) else math.nan
    r["busy_frac"] = union_length(allv) / win if len(allv) else math.nan

    if lat["leaf"].size:
        slow = lat["leaf"] > 1.5 * np.median(lat["leaf"])
        on = episodes(starts["leaf"], slow, episode_gap)
        r["leaf_episodes"] = len(on)
        r["leaf_episode_spacing_median_s"] = float(np.median(np.diff(on))) if len(on) > 2 else math.nan
    pl, pp = r.get("leaf_period_s"), r.get("pest_period_s")
    if pl and pp and abs(1 / pl - 1 / pp) > 1e-9:
        r["beat_predicted_s"] = 1 / abs(1 / pl - 1 / pp)

    sysr = load_csv(os.path.join(rundir, "system.csv"))
    idle = [x for x in sysr if x["phase"] == "idle"]
    run = [x for x in sysr if x["phase"] == "run" and ts <= fnum(x["t_s"]) <= t1]
    r["idle_temp_c"] = nanstat(np.mean, col(idle, "temp_c"))
    r["ambient_c"] = nanstat(np.mean, col(sysr, "ambient_c"))
    temp = col(run, "temp_c")
    r["temp_mean_c"] = nanstat(np.mean, temp)
    r["temp_sd_c"] = nanstat(lambda a: a.std(ddof=1) if a.size > 1 else math.nan, temp)
    r["temp_p99_c"] = nanstat(lambda a: np.percentile(a, 99), temp)
    r["temp_max_c"] = nanstat(np.max, temp)
    r["temp_rise_c"] = r["temp_mean_c"] - r["idle_temp_c"]
    fan = col(run, "fan_rpm")
    r["fan_rpm_mean"] = nanstat(np.mean, fan)
    r["fan_on_frac"] = nanstat(lambda a: (a > 0).mean(), fan)
    arm = col(run, "arm_mhz")
    r["arm_mhz_mean"] = nanstat(np.mean, arm)
    r["arm_below_max_frac"] = nanstat(lambda a: (a < 0.98 * np.nanmax(arm)).mean(), arm)
    live = [x["throttled_now"] for x in run if x["throttled_now"]]
    r["throttled_now_frac"] = (sum(int(v, 16) != 0 for v in live) / len(live)) if live else math.nan
    r["cpu_pct_mean"] = nanstat(np.mean, col(run, "cpu_pct"))
    r["capture_fps_mean"] = nanstat(np.mean, col(run, "capture_fps"))

    ev = load_csv(os.path.join(rundir, "events.csv"))
    r["cutoffs"] = sum(1 for x in ev if x["event"] == "THERMAL_CUTOFF"
                       and t0 <= fnum(x["t_s"]) <= t1)
    r["slots_skipped"] = sum(int(n.split("=")[1]) for n in
                             (x["note"] for x in ev if x["event"] == "SLOT_SKIPPED"))

    pw = load_csv(os.path.join(rundir, "power.csv"))
    if pw:
        w_idle = col([x for x in pw if x["phase"] == "idle"], "total_w")
        pwr = [x for x in pw if x["phase"] == "run" and ts <= fnum(x["t_s"]) <= t1]
        w_run = col(pwr, "total_w")
        r["power_idle_w"] = nanstat(np.mean, w_idle)
        # Sampling check. The PMIC is read at instants. If those instants are
        # locked to the release schedule, the share of samples that land inside
        # an inference differs from the share of time spent inside one, and the
        # mean is biased. Compare the two for each detector.
        tt = col(pwr, "t_s")
        grid = np.arange(ts, t1, 0.01)
        worst = 0.0
        for m in ("leaf", "pest"):
            if len(iv[m]) and tt.size:
                worst = max(worst, abs(inside(iv[m], tt).mean() - inside(iv[m], grid).mean()))
        r["power_sampling_gap"] = worst
        r["power_sampling_ok"] = int(worst <= POWER_GAP_MAX)
        mean_w = nanstat(np.mean, w_run)
        r["power_mean_w_raw"] = mean_w
        r["power_mean_w"] = mean_w if worst <= POWER_GAP_MAX else math.nan
        rate = (len(iv["leaf"]) + len(iv["pest"])) / win
        r["energy_per_inference_j"] = r["power_mean_w"] / rate if rate else math.nan
    return r


KEY = ["leaf_median_ms", "leaf_iqr_ms", "leaf_p90_ms", "leaf_slow_frac",
       "pest_median_ms", "pest_iqr_ms", "pest_p90_ms",
       "leaf_overlapped_frac", "compute_duty", "leaf_rate_per_min", "pest_rate_per_min",
       "temp_mean_c", "temp_rise_c", "temp_max_c", "fan_rpm_mean", "arm_mhz_mean",
       "cutoffs", "power_mean_w", "energy_per_inference_j"]


def aggregate(rows):
    by = {}
    for r in rows:
        if r.get("status") == "ok" and "duration_min" in r:
            by.setdefault(r["condition"], []).append(r)
    keys = sorted({k for r in rows for k, v in r.items() if isinstance(v, (int, float))
                   and not isinstance(v, bool)})
    out = []
    for c, rs in by.items():
        d = {"condition": c, "n_runs": len(rs)}
        for k in keys:
            v = np.array([fnum(x.get(k)) for x in rs])
            v = v[~np.isnan(v)]
            d[f"{k}_mean"] = float(v.mean()) if v.size else math.nan
            d[f"{k}_sd"] = float(v.std(ddof=1)) if v.size > 1 else math.nan
        out.append(d)
    return out


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def fmt(m, s):
    if math.isnan(m):
        return "—"
    return f"{m:.4g}" if math.isnan(s) else f"{m:.4g}±{s:.2g}"


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dirs", nargs="+")
    p.add_argument("--warmup-min", type=float, default=30.0)
    p.add_argument("--episode-gap", type=float, default=2.0)
    p.add_argument("--max-min", type=float, default=None,
                   help="analyse only the first N minutes of each run (common window)")
    a = p.parse_args()

    rundirs = []
    for d in a.dirs:
        if os.path.exists(os.path.join(d, "meta.json")):
            rundirs.append(d)
        else:
            rundirs += sorted(x for x in glob.glob(os.path.join(d, "*"))
                              if os.path.exists(os.path.join(x, "meta.json"))
                              and "__failed" not in x)
    if not rundirs:
        sys.exit("no run directories found")
    rows = [analyse_run(d, a.warmup_min, a.episode_gap, a.max_min) for d in rundirs]
    agg = aggregate(rows)
    base = a.dirs[0] if not os.path.exists(os.path.join(a.dirs[0], "meta.json")) \
        else os.path.dirname(os.path.abspath(a.dirs[0]))
    write_csv(os.path.join(base, "runs_summary.csv"), rows)
    write_csv(os.path.join(base, "conditions_summary.csv"), agg)

    bad = [r["run"] for r in rows if r.get("status") != "ok"]
    if bad:
        print(f"excluded (status not ok): {bad}")
    govs = {r.get("governor") for r in rows}
    if len(govs) > 1:
        print(f"WARNING: runs used different CPU governors: {govs}")
    unreliable = [r["run"] for r in rows if r.get("power_sampling_ok") == 0]
    if unreliable:
        print("power set to NaN (PMIC samples locked to the schedule, see "
              f"power_sampling_gap): {unreliable}")
    if any(r.get("camera") == "none" for r in rows):
        print("WARNING: synthetic-frame runs present; not reportable.")
    order = [c for c in ("indep", "serial", "sync", "phase", "sync_t2", "phase_t2")
             if any(g["condition"] == c for g in agg)]
    order += [g["condition"] for g in agg if g["condition"] not in order]
    g = {x["condition"]: x for x in agg}
    print(f"\n{'metric':28s}" + "".join(f"{c:>16s}" for c in order))
    print(f"{'n_runs':28s}" + "".join(f"{g[c]['n_runs']:>16d}" for c in order))
    for k in KEY:
        print(f"{k:28s}" + "".join(f"{fmt(g[c].get(k + '_mean', math.nan), g[c].get(k + '_sd', math.nan)):>16s}"
                                   for c in order))

    for x, y in (("sync", "phase"), ("sync_t2", "phase_t2")):
        if x in g and y in g:
            print(f"\n{x} - {y}  (same work rate, overlap on vs off)")
            for k in ("leaf_p90_ms", "leaf_iqr_ms", "pest_p90_ms", "temp_mean_c",
                      "power_mean_w", "compute_duty"):
                d = g[x].get(k + "_mean", math.nan) - g[y].get(k + "_mean", math.nan)
                print(f"  {k:24s} {d:+.4g}")
    for r in rows:
        if r.get("policy") == "indep" and "leaf_median_overlapped_ms" in r:
            print(f"\n{r['run']}: leaf median overlapped {r['leaf_median_overlapped_ms']:.1f} ms vs "
                  f"clear {r.get('leaf_median_clear_ms', math.nan):.1f} ms; episode spacing "
                  f"{r.get('leaf_episode_spacing_median_s', math.nan):.1f} s vs beat "
                  f"{r.get('beat_predicted_s', math.nan):.1f} s")
    print(f"\nwrote {os.path.join(base, 'runs_summary.csv')} and conditions_summary.csv")


if __name__ == "__main__":
    main()
