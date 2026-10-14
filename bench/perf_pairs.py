#!/usr/bin/env python3
"""
perf_pairs.py - cache and memory-bus counters during overlap on and off.

Runs dual_bench.py for each condition and, from minute --perf-start to the end of the run,
records system-wide Cortex-A76 PMU counters every 10 s with `perf stat -a -I`.
Counters (six programmable plus cycles, so nothing is multiplexed):
  instructions, l2d_cache_refill, l3d_cache_refill, ll_cache_rd, ll_cache_miss_rd, bus_access

Needs:  sudo sysctl -w kernel.perf_event_paranoid=-1   (resets at reboot)

  python3 perf_pairs.py --dry-run
  python3 perf_pairs.py --outdir perf1 -- --leaf-model leaf_s.onnx --pest-model pest_n.onnx --camera usb

Counters are system-wide (all four cores, including the OS), so absolute values include
background activity. Compare conditions, not absolute values. PMIC sampling is off.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

import run_matrix as rm

EVENTS = "cycles,instructions,l2d_cache_refill,l3d_cache_refill,ll_cache_rd,ll_cache_miss_rd,bus_access"
CONDS = {
    "sync":     ["--policy", "sync", "--period", "1.5"],
    "phase":    ["--policy", "phase", "--period", "1.5", "--phase-frac", "0.5"],
    "sync_t2":  ["--policy", "sync", "--period", "1.5", "--ort-threads", "2"],
    "phase_t2": ["--policy", "phase", "--period", "1.5", "--phase-frac", "0.5",
                 "--ort-threads", "2"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--outdir", default="perf1")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261013)
    ap.add_argument("--duration-min", type=float, default=20.0)
    ap.add_argument("--perf-start-min", type=float, default=8.0,
                    help="minutes after the run's measured start (after the idle phase)")
    ap.add_argument("--idle-s", type=float, default=60.0)
    ap.add_argument("--cool-min", type=float, default=5.0)
    ap.add_argument("--cool-max", type=float, default=12.0)
    ap.add_argument("--cool-margin", type=float, default=2.0)
    ap.add_argument("--baseline-s", type=float, default=120.0)
    ap.add_argument("--perf-bin", default="perf")
    ap.add_argument("--dry-run", action="store_true")
    a, extra = ap.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]

    order = rm.plan(list(CONDS), a.reps, a.seed)
    if a.dry_run:
        for i, r in enumerate(order, 1):
            print(f"{i:2d}. {r['label']:12s} {' '.join(CONDS[r['cond']])}")
        print(f"\n{len(order)} runs, perf from minute {a.perf_start_min} to {a.duration_min}.")
        return 0

    os.makedirs(a.outdir, exist_ok=True)
    mpath = os.path.join(a.outdir, "manifest.json")
    if os.path.exists(mpath):
        man = json.load(open(mpath))
        if [r["label"] for r in man["order"]] != [r["label"] for r in order]:
            sys.exit("manifest.json exists with a different plan; use another --outdir.")
    else:
        man = {"created": datetime.now().isoformat(timespec="seconds"),
               "args": vars(a), "extra": extra, "events": EVENTS, "order": order}

    def save():
        tmp = mpath + ".tmp"
        json.dump(man, open(tmp, "w"), indent=2)
        os.replace(tmp, mpath)

    if man.get("idle_baseline_c") is None and a.baseline_s > 0:
        man["idle_baseline_c"] = rm.idle_baseline(a.baseline_s)
    target = (man["idle_baseline_c"] + a.cool_margin
              if man.get("idle_baseline_c") is not None else None)
    save()

    first = True
    for i, r in enumerate(man["order"], 1):
        if r.get("status") == "ok":
            continue
        rundir = os.path.join(a.outdir, r["label"])
        if os.path.exists(rundir):
            n = 1
            while os.path.exists(f"{rundir}__failed{n}"):
                n += 1
            os.rename(rundir, f"{rundir}__failed{n}")
        if not first:
            el, t, why = rm.cooldown(a, target)
            r["cooldown_min"], r["cooldown_stop"] = round(el, 2), why
            rm.log(f"cool-down {el:.1f} min, die {t} C ({why})")
        first = False
        cmd = [sys.executable, rm.BENCH, *CONDS[r["cond"]], "--label", r["label"],
               "--outdir", a.outdir, "--duration-min", str(a.duration_min),
               "--idle-s", str(a.idle_s), *extra]
        r["cmd"], r["status"] = cmd, "running"
        r["started"] = datetime.now().isoformat(timespec="seconds")
        save()
        rm.log(f"[{i}/{len(order)}] {r['label']}")
        t0 = time.time()
        bench = subprocess.Popen(cmd)
        wait = a.idle_s + 20 + a.perf_start_min * 60     # +20 s: model loading
        perf_s = max(10, int(a.duration_min * 60 - a.perf_start_min * 60 - 30))
        try:
            while time.time() - t0 < wait and bench.poll() is None:
                time.sleep(2)
            pcmd = [a.perf_bin, "stat", "-a", "-I", "10000", "-x", ",", "-e", EVENTS,
                    "-o", os.path.join(a.outdir, f"{r['label']}_perf.csv"),
                    "--", "sleep", str(perf_s)]
            r["perf_started_after_s"] = round(time.time() - t0, 1)
            r["perf_cmd"] = pcmd
            prc = subprocess.call(pcmd) if bench.poll() is None else -1
            rc = bench.wait()
        except KeyboardInterrupt:
            bench.terminate()
            r["status"] = "interrupted"
            save()
            return 130
        r["status"] = "ok" if (rc == 0 and prc == 0) else f"failed(bench={rc},perf={prc})"
        r["finished"] = datetime.now().isoformat(timespec="seconds")
        save()
        rm.log(f"{r['label']}: {r['status']}")
    failed = [r["label"] for r in man["order"] if r.get("status") != "ok"]
    rm.log("all runs complete" if not failed else f"incomplete: {failed}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
