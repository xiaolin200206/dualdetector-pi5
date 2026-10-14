#!/usr/bin/env python3
"""
make_replay.py - replay test: how quickly does each schedule answer after an image appears?

  python3 -I make_replay.py --runs data/replay1 --images data/sabah_labels --out results

Ground truth is the YOLO label file next to each replayed image (classes present in the image).
Class mapping (Sabah ids -> this paper's models) is fixed below.
Per presentation and per model: whether any inference used it, the time from its onset to the first
answer, the time to the first correct answer (a detected class present in the label file), and misses.
Per presentation: whether both models answered and when both had.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

# Sabah label ids: 0 Algal, 1 Leaf_rot, 2 Phomopsis, 3 Psyllid, 4 Psyllid_damage, 5 leaf_hopper_damage
LEAF_GT = {0, 1, 2}
PEST_GT = {3, 4, 5}
# model class id -> Sabah id (None = class not in the Sabah label set)
LEAF_MAP = {0: 0, 1: 1, 2: 2, 3: None, 4: None}            # Algal, Leaf_rot, Phomopsis, Pink, Root
PEST_MAP = {0: 5, 1: 3, 2: 4, 3: None, 4: None, 5: None, 6: None}  # leafhopper, Psyllid, Psyllid_damage, ...
MAPS = {"leaf": LEAF_MAP, "pest": PEST_MAP}
GTS = {"leaf": LEAF_GT, "pest": PEST_GT}


def labels_for(images_dir):
    """Ground truth per image file name.

    Accepts either the image set itself (images/ + labels/ or labels beside the images)
    or a labels-only folder (data/sabah_labels: *.txt + images.txt listing the image names),
    so the replay tables reproduce without redistributing the images.
    """
    gt = {}
    imgs = [p for p in glob.glob(os.path.join(images_dir, "**", "*"), recursive=True)
            if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    if not imgs and os.path.exists(os.path.join(images_dir, "images.txt")):
        imgs = [os.path.join(images_dir, n.strip()) for n in open(os.path.join(images_dir, "images.txt")) if n.strip()]
    for p in imgs:
        lab = os.path.splitext(p)[0] + ".txt"
        if not os.path.exists(lab):
            lab = lab.replace("images", "labels")
        cls = set()
        if os.path.exists(lab):
            for line in open(lab):
                if line.strip():
                    cls.add(int(line.split()[0]))
        gt[os.path.basename(p)] = cls
    return gt


def parse_dets(s):
    out = []
    if isinstance(s, str) and s:
        for tok in s.split(";"):
            c, p = tok.split(":")
            out.append((int(c), float(p)))
    return out


def analyse_run(d, gt, warmup_min):
    meta = json.load(open(os.path.join(d, "meta.json")))
    t0 = meta["run_start_s"]
    rep = pd.read_csv(os.path.join(d, "replay.csv"))
    inf = pd.read_csv(os.path.join(d, "inferences.csv"))
    inf = inf[inf.t_start >= t0 + warmup_min * 60]
    rep = rep[(rep.onset >= t0 + warmup_min * 60) & (rep.offset <= meta["run_end_s"])]
    by_pres = {m: defaultdict(list) for m in ("leaf", "pest")}
    for r in inf.itertuples():
        if r.frame_id > 0:
            by_pres[r.model][int(r.frame_id)].append(r)
    rows = []
    for p in rep.itertuples():
        g = gt.get(p.image, set())
        row = {"presentation": p.presentation, "image": p.image, "onset": p.onset}
        firsts = {}
        for m in ("leaf", "pest"):
            rs = sorted(by_pres[m].get(p.presentation, []), key=lambda r: r.t_end)
            target = bool(g & GTS[m])
            row[f"{m}_target"] = target
            row[f"{m}_n_inf"] = len(rs)
            row[f"{m}_first_answer_s"] = (rs[0].t_end - p.onset) if rs else np.nan
            firsts[m] = rs[0].t_end if rs else None
            correct_t = np.nan
            false_alarm = False
            for r in rs:
                dets = parse_dets(r.dets)
                mapped = {MAPS[m][c] for c, _ in dets}
                if target and (mapped & g):
                    correct_t = r.t_end - p.onset
                    break
                if not target and dets:
                    false_alarm = True
            row[f"{m}_first_correct_s"] = correct_t
            row[f"{m}_miss"] = target and np.isnan(correct_t)
            row[f"{m}_no_inference"] = len(rs) == 0
            row[f"{m}_false_alarm"] = (not target) and false_alarm
        both = firsts["leaf"] is not None and firsts["pest"] is not None
        row["both_answered"] = both
        row["both_answers_s"] = (max(firsts["leaf"], firsts["pest"]) - p.onset) if both else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    label = meta.get("label", os.path.basename(d))
    out = {"run": label, "condition": label.rsplit("_r", 1)[0], "presentations": len(df)}
    # context: governor, clock, temperature and per-call latency over the same window
    out["governor"] = meta.get("versions", {}).get("governor", "")
    sysd = pd.read_csv(os.path.join(d, "system.csv"))
    w = sysd[(sysd.t_s >= t0 + warmup_min * 60) & (sysd.t_s <= meta["run_end_s"])]
    out["clock_below_2400_frac"] = float((w.arm_mhz < 2400).mean())
    out["temp_mean_c"] = float(w.temp_c.mean())
    out["throttled_any"] = bool((w.throttled_now.astype(str) != "0x0").any())
    for mname in ("leaf", "pest"):
        out[f"{mname}_latency_median_ms"] = float(inf[inf.model == mname].latency_ms.median())
        out[f"{mname}_n_inf"] = int((inf.model == mname).sum())
    for m in ("leaf", "pest"):
        t = df[df[f"{m}_target"]]
        out[f"{m}_targets"] = len(t)
        out[f"{m}_first_answer_median_s"] = df[f"{m}_first_answer_s"].median()
        out[f"{m}_first_answer_p90_s"] = df[f"{m}_first_answer_s"].quantile(0.9)
        out[f"{m}_no_inference_frac"] = df[f"{m}_no_inference"].mean()
        out[f"{m}_first_correct_median_s"] = t[f"{m}_first_correct_s"].median()
        out[f"{m}_miss_frac"] = t[f"{m}_miss"].mean() if len(t) else np.nan
        nt = df[~df[f"{m}_target"]]
        out[f"{m}_false_alarm_frac"] = nt[f"{m}_false_alarm"].mean() if len(nt) else np.nan
        out[f"{m}_inf_per_presentation"] = df[f"{m}_n_inf"].mean()
    out["both_answered_frac"] = df.both_answered.mean()
    out["both_answers_median_s"] = df.both_answers_s.median()
    out["both_answers_p90_s"] = df.both_answers_s.quantile(0.9)
    return out, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup-min", type=float, default=2.0)
    a = ap.parse_args()
    td = os.path.join(a.out, "tables")
    os.makedirs(td, exist_ok=True)
    gt = labels_for(a.images)
    rows, per = [], []
    for d in sorted(glob.glob(os.path.join(a.runs, "*_r?"))):
        if "__failed" in d or not os.path.exists(os.path.join(d, "replay.csv")):
            continue
        r, df = analyse_run(d, gt, a.warmup_min)
        rows.append(r)
        df.insert(0, "run", r["run"])
        per.append(df)
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(td, "replay_runs.csv"), index=False)
    pd.concat(per).to_csv(os.path.join(td, "replay_presentations.csv"), index=False)
    num = runs.select_dtypes("number").columns
    cond = runs.groupby("condition")[num].agg(["mean", "min", "max"])
    cond.columns = [f"{a}_{b}" for a, b in cond.columns]
    cond.to_csv(os.path.join(td, "replay_conditions.csv"))
    L, P = GTS["leaf"], GTS["pest"]
    facts = {"images": len(gt), "boxes_total": None,
             "images_leaf_target": sum(bool(g & L) for g in gt.values()),
             "images_pest_target": sum(bool(g & P) for g in gt.values()),
             "images_both": sum(bool(g & L) and bool(g & P) for g in gt.values()),
             "images_neither": sum(not (g & L) and not (g & P) for g in gt.values()),
             "runs": len(runs), "warmup_min": a.warmup_min,
             "governor": dict(zip(runs.run, runs.governor)),
             "clock_below_2400_frac_max": float(runs.clock_below_2400_frac.max()),
             "throttled_any": bool(runs.throttled_any.any())}
    nb = 0
    for n_ in gt:
        lab = os.path.join(a.images, "labels", os.path.splitext(n_)[0] + ".txt")
        if not os.path.exists(lab):
            lab = os.path.join(a.images, os.path.splitext(n_)[0] + ".txt")
        if os.path.exists(lab):
            nb += sum(1 for line in open(lab) if line.strip())
    facts["boxes_total"] = nb
    json.dump(facts, open(os.path.join(td, "facts_replay.json"), "w"), indent=1)
    show = ["leaf_first_answer_median_s", "pest_first_answer_median_s", "both_answers_median_s",
            "leaf_no_inference_frac", "pest_no_inference_frac", "leaf_miss_frac", "pest_miss_frac",
            "leaf_false_alarm_frac", "pest_false_alarm_frac", "leaf_targets", "pest_targets"]
    print(runs[["run"] + show].round(3).to_string())


if __name__ == "__main__":
    main()
