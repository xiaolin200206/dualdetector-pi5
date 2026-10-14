#!/usr/bin/env python3
"""
replay_matrix.py - replay test: the Sabah images are shown one at a time in place of the camera,
and five schedules are compared on how quickly each detector answers after an image appears.

  python3 replay_matrix.py --dry-run
  python3 replay_matrix.py --outdir replay1 --duration-min 20 -- \
      --replay sabah --dwell 2.5 --gap 1.0 --leaf-model leaf_s.onnx --pest-model pest_n.onnx --pmic-jitter 0.5

Conditions: independent timers, fixed-period serial, synchronous, phased 0.75 s, phased 0.42 s
(all at P = 1.5 s where applicable). Block-randomised, resumable, same cool-down logic as run_matrix.py.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta

import run_matrix as rm

CONDITIONS = {
    "rp_indep":   ["--policy", "indep"],
    "rp_serialp": ["--policy", "serialp", "--period", "1.5"],
    "rp_sync":    ["--policy", "sync", "--period", "1.5"],
    "rp_phase75": ["--policy", "phase", "--period", "1.5", "--phase-s", "0.75"],
    "rp_phase42": ["--policy", "phase", "--period", "1.5", "--phase-s", "0.42"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--outdir", default="replay1")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--duration-min", type=float, default=20.0)
    ap.add_argument("--idle-s", type=float, default=60.0)
    ap.add_argument("--cool-min", type=float, default=3.0)
    ap.add_argument("--cool-max", type=float, default=8.0)
    ap.add_argument("--cool-margin", type=float, default=2.0)
    ap.add_argument("--baseline-s", type=float, default=120.0)
    ap.add_argument("--no-pmic", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a, extra = ap.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    if not a.dry_run and "--replay" not in extra:
        sys.exit("pass --replay FOLDER after '--'")

    order = rm.plan(list(CONDITIONS), a.reps, a.seed)
    per_run = a.duration_min + a.idle_s / 60 + (a.cool_min + a.cool_max) / 2
    eta = timedelta(minutes=per_run * len(order) + a.baseline_s / 60)
    if a.dry_run:
        for i, r in enumerate(order, 1):
            print(f"{i:2d}. {r['label']:14s} {' '.join(CONDITIONS[r['cond']])}")
        print(f"\n{len(order)} runs, roughly {eta} including cool-downs.")
        return 0

    os.makedirs(a.outdir, exist_ok=True)
    mpath = os.path.join(a.outdir, "manifest.json")
    if os.path.exists(mpath):
        man = json.load(open(mpath))
        if [r["label"] for r in man["order"]] != [r["label"] for r in order]:
            sys.exit("manifest.json exists with a different plan; use another --outdir.")
        rm.log(f"resuming: {sum(r.get('status') == 'ok' for r in man['order'])} of "
               f"{len(order)} runs already complete")
    else:
        man = {"created": datetime.now().isoformat(timespec="seconds"),
               "args": vars(a), "extra": extra, "order": order}

    def save():
        tmp = mpath + ".tmp"
        json.dump(man, open(tmp, "w"), indent=2)
        os.replace(tmp, mpath)

    if man.get("idle_baseline_c") is None and a.baseline_s > 0:
        rm.log(f"idle baseline, {a.baseline_s:.0f} s ...")
        man["idle_baseline_c"] = rm.idle_baseline(a.baseline_s)
    target = (man["idle_baseline_c"] + a.cool_margin
              if man.get("idle_baseline_c") is not None else None)
    save()
    rm.log(f"{len(order)} runs, roughly {eta}; cool-down target "
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
            el, t, why = rm.cooldown(a, target)
            r["cooldown_min"], r["cooldown_end_c"], r["cooldown_stop"] = round(el, 2), t, why
            rm.log(f"cool-down {el:.1f} min, die {t} C ({why})")
        first = False
        cmd = [sys.executable, rm.BENCH, *CONDITIONS[r["cond"]],
               "--label", r["label"], "--outdir", a.outdir,
               "--duration-min", str(a.duration_min), "--idle-s", str(a.idle_s),
               "--seed", str(a.seed + r["rep"]),
               *([] if a.no_pmic else ["--pmic"]), *extra]
        r["cmd"] = cmd
        r["started"] = datetime.now().isoformat(timespec="seconds")
        r["status"] = "running"
        save()
        rm.log(f"[{i}/{len(order)}] {r['label']}: {' '.join(CONDITIONS[r['cond']])}")
        try:
            rc = subprocess.call(cmd)
        except KeyboardInterrupt:
            r["status"] = "interrupted"
            save()
            rm.log("interrupted; re-run the same command to resume")
            return 130
        r["status"] = "ok" if rc == 0 else f"failed(rc={rc})"
        r["finished"] = datetime.now().isoformat(timespec="seconds")
        save()
        rm.log(f"{r['label']}: {r['status']}")

    failed = [r["label"] for r in man["order"] if r.get("status") != "ok"]
    rm.log("all runs complete" if not failed else f"incomplete: {failed}; re-run to retry")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
