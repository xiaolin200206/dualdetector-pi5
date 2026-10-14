#!/usr/bin/env python3
"""
run_matrix.py - run the full condition matrix of dual_bench.py, unattended.

Design
  * Block randomisation: every block contains each condition once, in an order
    drawn from --seed, and blocks are repeated --reps times. Slow drift in room
    temperature over the session is therefore spread across conditions rather
    than loaded onto whichever condition happened to run last.
  * Cool-down between runs: at least --cool-min minutes, then until the die is
    within --cool-margin C of the idle temperature measured before the first
    run, or --cool-max minutes, whichever comes first. Recorded per run.
  * Resumable: progress is kept in <outdir>/manifest.json. Re-running the same
    command skips completed runs and retries failed ones (the failed directory
    is renamed, never deleted).
  * PMIC power sampling is on for every run by default, so its small overhead
    is identical across conditions. --no-pmic turns it off for every run.

Everything after `--` is passed to dual_bench.py unchanged (models, imgsz...).

  python3 run_matrix.py --dry-run
  python3 run_matrix.py --outdir campaign1 -- --leaf-model leaf.onnx --pest-model pest.onnx
  python3 run_matrix.py --outdir smoke --duration-min 1 --idle-s 5 --cool-min 0 \\
      --cool-max 0 --reps 1 -- --backend fake --allow-no-camera
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timedelta

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(HERE, "dual_bench.py")

CONDITIONS = {
    # legacy policies, replicating the original scheduler
    "indep":    ["--policy", "indep"],
    "serial":   ["--policy", "serial"],
    # fixed-rate pair: same work, overlap 100 % vs 0 %
    "sync":     ["--policy", "sync", "--period", "1.5"],
    "phase":    ["--policy", "phase", "--period", "1.5", "--phase-frac", "0.5"],
    # the same pair with two ONNX Runtime threads per detector (4 in total on
    # 4 cores): if the sync-vs-phase gap survives, it is not oversubscription
    "sync_t2":  ["--policy", "sync", "--period", "1.5", "--ort-threads", "2"],
    "phase_t2": ["--policy", "phase", "--period", "1.5", "--phase-frac", "0.5",
                 "--ort-threads", "2"],
    # backbone-matched pair (pest detector swapped for its s-size counterpart):
    # does the overlap penalty depend on the pair's composition?
    # Enabled only when --pest-s-model is given.
    "sync_ss":  ["--policy", "sync", "--period", "1.5"],
    "phase_ss": ["--policy", "phase", "--period", "1.5", "--phase-frac", "0.5"],
    # fixed-period serial: the same work rate as sync/phase, pest started the
    # moment leaf returns. Added after the main campaign (campaign3).
    "serialp":    ["--policy", "serialp", "--period", "1.5"],
    "serialp_t2": ["--policy", "serialp", "--period", "1.5", "--ort-threads", "2"],
    "serialp_ss": ["--policy", "serialp", "--period", "1.5"],
}
MATCHED = {"sync_ss", "phase_ss", "serialp_ss"}


def read_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return float(f.read()) / 1000.0
    except Exception:
        return None


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


def plan(conds, reps, seed):
    order = []
    for r in range(1, reps + 1):
        block = list(conds)
        random.Random(seed * 1000 + r).shuffle(block)
        order += [{"cond": c, "rep": r, "label": f"{c}_r{r}"} for c in block]
    return order


def idle_baseline(seconds):
    vals = []
    t_end = time.time() + seconds
    while time.time() < t_end:
        t = read_temp()
        if t is not None:
            vals.append(t)
        time.sleep(5)
    return sum(vals) / len(vals) if vals else None


def cooldown(a, target):
    t0 = time.time()
    while True:
        el = (time.time() - t0) / 60
        t = read_temp()
        if el >= a.cool_max:
            return el, t, "max"
        if el >= a.cool_min and (target is None or t is None or t <= target):
            return el, t, "target" if target is not None else "min"
        time.sleep(10)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--outdir", default=os.path.join(HERE, "campaign"))
    p.add_argument("--conditions", default=None,
                   help="comma list; default: all, the *_ss pair only with --pest-s-model")
    p.add_argument("--pest-s-model", default=None,
                   help="s-size pest detector for the backbone-matched conditions")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--seed", type=int, default=20261009)
    p.add_argument("--duration-min", type=float, default=60.0)
    p.add_argument("--idle-s", type=float, default=60.0)
    p.add_argument("--cool-min", type=float, default=10.0)
    p.add_argument("--cool-max", type=float, default=25.0)
    p.add_argument("--cool-margin", type=float, default=2.0)
    p.add_argument("--baseline-s", type=float, default=120.0,
                   help="idle sampling before the first run, sets the cool-down target")
    p.add_argument("--no-pmic", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a, extra = p.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]

    if a.conditions is None:
        conds = [c for c in CONDITIONS if a.pest_s_model or c not in MATCHED]
    else:
        conds = [c.strip() for c in a.conditions.split(",") if c.strip()]
    if any(c in MATCHED for c in conds) and not a.pest_s_model:
        p.error("the *_ss conditions need --pest-s-model")
    bad = [c for c in conds if c not in CONDITIONS]
    if bad:
        p.error(f"unknown condition(s): {bad}; known: {list(CONDITIONS)}")

    order = plan(conds, a.reps, a.seed)
    per_run = a.duration_min + a.idle_s / 60 + (a.cool_min + a.cool_max) / 2
    eta = timedelta(minutes=per_run * len(order) + a.baseline_s / 60)
    if a.dry_run:
        for i, r in enumerate(order, 1):
            print(f"{i:2d}. {r['label']:12s} {' '.join(CONDITIONS[r['cond']])}")
        print(f"\n{len(order)} runs, roughly {eta} including cool-downs.")
        return 0

    os.makedirs(a.outdir, exist_ok=True)
    mpath = os.path.join(a.outdir, "manifest.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            man = json.load(f)
        if [r["label"] for r in man["order"]] != [r["label"] for r in order]:
            sys.exit("manifest.json exists with a different plan; use another --outdir.")
        log(f"resuming: {sum(r.get('status') == 'ok' for r in man['order'])} of "
            f"{len(order)} runs already complete")
    else:
        man = {"created": datetime.now().isoformat(timespec="seconds"),
               "args": vars(a), "extra": extra, "order": order}

    def save():
        tmp = mpath + ".tmp"
        with open(tmp, "w") as f:
            json.dump(man, f, indent=2)
        os.replace(tmp, mpath)

    if man.get("idle_baseline_c") is None and a.baseline_s > 0:
        log(f"idle baseline, {a.baseline_s:.0f} s ...")
        man["idle_baseline_c"] = idle_baseline(a.baseline_s)
        man["idle_baseline_wall"] = datetime.now().isoformat(timespec="seconds")
    target = (man["idle_baseline_c"] + a.cool_margin
              if man.get("idle_baseline_c") is not None else None)
    save()
    log(f"{len(order)} runs, roughly {eta}; cool-down target "
        f"{'n/a' if target is None else f'{target:.1f} C'}")

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
            r.setdefault("failed_dirs", []).append(f"{r['label']}__failed{n}")
        if not first:
            el, t, why = cooldown(a, target)
            r["cooldown_min"], r["cooldown_end_c"], r["cooldown_stop"] = round(el, 2), t, why
            log(f"cool-down {el:.1f} min, die {t} C ({why})")
        first = False
        cmd = [sys.executable, BENCH, *CONDITIONS[r["cond"]],
               "--label", r["label"], "--outdir", a.outdir,
               "--duration-min", str(a.duration_min), "--idle-s", str(a.idle_s),
               *([] if a.no_pmic else ["--pmic"]), *extra,
               # after `extra`, so it overrides any --pest-model given there
               *(["--pest-model", os.path.abspath(a.pest_s_model)]
                 if r["cond"] in MATCHED else [])]
        r["cmd"] = cmd
        r["started"] = datetime.now().isoformat(timespec="seconds")
        r["status"] = "running"
        save()
        log(f"[{i}/{len(order)}] {r['label']}: {' '.join(CONDITIONS[r['cond']])}")
        try:
            rc = subprocess.call(cmd)
        except KeyboardInterrupt:
            r["status"] = "interrupted"
            save()
            log("interrupted; re-run the same command to resume")
            return 130
        r["status"] = "ok" if rc == 0 else f"failed(rc={rc})"
        r["finished"] = datetime.now().isoformat(timespec="seconds")
        save()
        log(f"{r['label']}: {r['status']}")

    failed = [r["label"] for r in man["order"] if r.get("status") != "ok"]
    log("all runs complete" if not failed else f"incomplete: {failed}; re-run to retry")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
