#!/usr/bin/env python3
"""
sweep_matrix.py - operating-point sweep: release period x offset, default threading.

For each release period P and offset fraction f the pest detector is released f*P after
the leaf detector (f = 0 is synchronous release). The sweep maps where overlap starts.
Runs are block-randomised, resumable and use the same cool-down logic as run_matrix.py.

  python3 sweep_matrix.py --dry-run
  python3 sweep_matrix.py --outdir sweep1 --duration-min 30 -- \
      --leaf-model leaf_s.onnx --pest-model pest_n.onnx --camera usb --pmic-jitter 0.5

Rep 1 is a complete set of nine runs; stop after it if time is short.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta

import run_matrix as rm

PERIODS = [1.0, 1.5, 2.0]
FRACS = [0.0, 0.25, 0.5]

# Dense sweep (sweep2): P = 1.5 s, offsets in seconds between the 0.375 s and 0.75 s
# settings of sweep1, to locate where contact between the detectors stops.
DENSE_PERIOD = 1.5
DENSE_OFFSETS_S = [0.39, 0.42, 0.45, 0.50, 0.60]


def label(p, f):
    return f"p{int(round(p * 10)):02d}_o{int(round(f * 100)):02d}"


def label_s(p, o):
    return f"p{int(round(p * 10)):02d}_s{int(round(o * 1000)):03d}"


def conditions(dense=False):
    out = {}
    if dense:
        for o in DENSE_OFFSETS_S:
            out[label_s(DENSE_PERIOD, o)] = ["--policy", "phase", "--period", str(DENSE_PERIOD),
                                             "--phase-s", str(o)]
        return out
    for p in PERIODS:
        for f in FRACS:
            if f == 0.0:
                args = ["--policy", "sync", "--period", str(p)]
            else:
                args = ["--policy", "phase", "--period", str(p), "--phase-frac", str(f)]
            out[label(p, f)] = args
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--outdir", default="sweep1")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--duration-min", type=float, default=30.0)
    ap.add_argument("--idle-s", type=float, default=60.0)
    ap.add_argument("--cool-min", type=float, default=5.0)
    ap.add_argument("--cool-max", type=float, default=12.0)
    ap.add_argument("--cool-margin", type=float, default=2.0)
    ap.add_argument("--baseline-s", type=float, default=120.0)
    ap.add_argument("--no-pmic", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--dense", action="store_true",
                    help="dense offset sweep at P = 1.5 s (labels p15_sNNN, offset in ms)")
    a, extra = ap.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]

    conds = conditions(a.dense)
    order = rm.plan(list(conds), a.reps, a.seed)
    per_run = a.duration_min + a.idle_s / 60 + (a.cool_min + a.cool_max) / 2
    eta = timedelta(minutes=per_run * len(order) + a.baseline_s / 60)
    if a.dry_run:
        for i, r in enumerate(order, 1):
            print(f"{i:2d}. {r['label']:12s} {' '.join(conds[r['cond']])}")
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
        cmd = [sys.executable, rm.BENCH, *conds[r["cond"]],
               "--label", r["label"], "--outdir", a.outdir,
               "--duration-min", str(a.duration_min), "--idle-s", str(a.idle_s),
               *([] if a.no_pmic else ["--pmic"]), *extra]
        r["cmd"] = cmd
        r["started"] = datetime.now().isoformat(timespec="seconds")
        r["status"] = "running"
        save()
        rm.log(f"[{i}/{len(order)}] {r['label']}: {' '.join(conds[r['cond']])}")
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
