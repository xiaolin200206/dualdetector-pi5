#!/usr/bin/env python3
"""
02_build.py - merge, group, split and project the two datasets.

Reads the output of 01_inspect.py. Writes, in --out:
  leaf/  pest/        YOLO datasets (images/{train,val}, labels/{train,val}, data.yaml)
  split.csv           one row per kept image: group, split, source files, classes
  dropped.csv         copies removed as near-identical variants of a kept image
  audit.md / .json    leak checks across the new boundary and per-class counts

Procedure (the order matters, and each step is reported):
  1. class names are mapped through --class-map (rename / drop / domain);
  2. byte-identical files are collapsed into one image, label sets unioned;
  3. images are grouped by export stem, by perceptual hash (dihedral, <= --group-phash)
     and by capture burst (same name prefix, counter gap <= --burst-gap);
  4. inside each group, copies sharing an export stem and within --dedupe-phash
     of a kept copy are dropped (offline-augmented variants); --keep-variants keeps them;
  5. whole groups are assigned to train or val, greedily, rarest class first;
  6. the merged split is projected onto a leaf set and a pest set. An image enters a
     domain's set only if it was annotated for that domain, so no image can be in
     training for one detector and in validation for the other;
  7. the new boundary is audited and the build refuses to finish if anything leaks.

  python 02_build.py --inspect inspect_out --out built --class-map class_map.yaml
"""

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import DSU, _popcount, load_yaml, near_pairs, union_boxes, video_key

DOMAINS = ("leaf", "pest")


def load_inspect(d):
    d = Path(d)
    with open(d / "inventory.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["idx"] = int(r["idx"])
    H = np.load(d / "hashes.npy")
    lab = json.load(open(d / "labels.json", encoding="utf-8"))
    labels = {int(k): v for k, v in lab["labels"].items()}
    return rows, H, labels, lab["names"]


def build_class_map(names, cmap_path):
    cm = load_yaml(cmap_path) if cmap_path else {}
    rename = cm.get("rename") or {}
    drop = set(cm.get("drop") or [])
    dom_over = cm.get("domain") or {}
    mapping, order = {}, {d: [] for d in DOMAINS}
    for ds in DOMAINS:
        for c in names[ds]:
            if c in drop:
                mapping[(ds, c)] = None
                continue
            m = rename.get(c, c)
            if m in drop:
                mapping[(ds, c)] = None
                continue
            dom = dom_over.get(m, ds)
            if dom not in DOMAINS:
                sys.exit(f"class_map domain for {m!r} must be leaf or pest")
            mapping[(ds, c)] = (m, dom)
            if m not in order[dom]:
                order[dom].append(m)
    # every merged name must live in exactly one domain
    where = defaultdict(set)
    for v in mapping.values():
        if v:
            where[v[0]].add(v[1])
    clash = {k: v for k, v in where.items() if len(v) > 1}
    if clash:
        sys.exit(f"class names in both domains: {clash}. Assign them with `domain:` in the class map.")
    for d in DOMAINS:
        order[d] = [c for c in order[d] if c in where and d in where[c]]
    low = Counter(c.lower() for d in DOMAINS for c in order[d])
    case = [c for d in DOMAINS for c in order[d] if low[c.lower()] > 1]
    if case:
        sys.exit(f"case-variant class names after mapping: {case}. "
                 "Merge them with `rename:` in the class map, or confirm they are distinct "
                 "by renaming one of them.")
    return mapping, order


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inspect", default="inspect_out")
    p.add_argument("--out", default="built")
    p.add_argument("--class-map", default=None)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--group-phash", type=int, default=2)
    p.add_argument("--burst-gap", type=int, default=3)
    p.add_argument("--dedupe-phash", type=int, default=10)
    p.add_argument("--keep-variants", action="store_true")
    p.add_argument("--audit-near", type=int, default=10)
    p.add_argument("--max-group-frac", type=float, default=0.05,
                   help="warn if one group exceeds this share of images (chain merging)")
    a = p.parse_args()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        sys.exit(f"{out} exists and is not empty; choose a new --out")
    rows, H, labels, names = load_inspect(a.inspect)
    mapping, order = build_class_map(names, a.class_map)
    log = []

    # 1-2. map classes, collapse byte-identical files into units -----------------
    by_sha = defaultdict(list)
    for r in rows:
        by_sha[r["sha256"]].append(r["idx"])
    units = []
    for sha, members in by_sha.items():
        lists, ann = [], set()
        for i in members:
            ds = rows[i]["dataset"]
            ann.add(ds)
            if ds == "bg":              # unannotated negatives count for both detectors
                ann.update(DOMAINS)
            mapped = []
            for b in labels[i]:
                m = mapping[(ds, b[0])]
                if m:
                    mapped.append((m[0], *b[1:]))
                    ann.add(m[1])
            lists.append(mapped)
        rep = sorted(members, key=lambda i: (rows[i]["split"] not in ("val", "test"),
                                            int(rows[i]["aug_marker"]), rows[i]["relpath"]))[0]
        units.append({"uid": len(units), "members": members, "rep": rep,
                      "boxes": union_boxes(lists), "ann": ann})
    log.append(f"files {len(rows)} -> {len(units)} byte-distinct images")
    UH = np.array([H[u["rep"]] for u in units], dtype=np.uint64)

    # 3. grouping ------------------------------------------------------------------
    dsu = DSU(len(units))
    by_stem = defaultdict(list)
    for u in units:
        for i in u["members"]:
            by_stem[rows[i]["export_stem"]].append(u["uid"])
    for lst in by_stem.values():
        for x in lst[1:]:
            dsu.union(lst[0], x)
    n_stem = len(units) - len(dsu.groups())
    by_video = defaultdict(set)
    for u in units:
        for i in u["members"]:
            v = video_key(rows[i]["export_stem"])
            if v:
                by_video[v].add(u["uid"])
    for lst in by_video.values():
        lst = sorted(lst)
        for x in lst[1:]:
            dsu.union(lst[0], x)
    log.append(f"video sequences (frames sharing a video id): {sum(len(v) > 1 for v in by_video.values())} "
               f"covering {sum(len(v) for v in by_video.values() if len(v) > 1)} images")
    for i, j, d in near_pairs(UH, a.group_phash):
        dsu.union(i, j)
    if a.burst_gap > 0:
        byp = defaultdict(list)
        for u in units:
            r = rows[u["rep"]]
            if r["burst_prefix"]:
                byp[r["burst_prefix"].split("::", 1)[-1]].append((int(r["burst_counter"]), u["uid"]))
        for lst in byp.values():
            lst.sort()
            for (c1, i1), (c2, i2) in zip(lst, lst[1:]):
                if c2 - c1 <= a.burst_gap:
                    dsu.union(i1, i2)
    groups = dsu.groups()
    gid = {}
    for g, mem in enumerate(groups):
        for x in mem:
            gid[x] = g
    big = max(map(len, groups))
    log.append(f"groups: {len(groups)} (stem merges {n_stem}); largest group {big} images")
    if big > a.max_group_frac * len(units):
        msg = (f"WARNING: largest group holds {big} of {len(units)} images. A long run of "
               "consecutive camera counters chains into one group; it will stay in train. "
               "Check it in split.csv; --burst-gap 1 narrows the rule.")
        print(msg)
        log.append(msg)

    # 4. drop near-identical variants inside a stem ---------------------------------
    dropped = []
    keep = set(range(len(units)))
    if not a.keep_variants:
        for stem, lst in by_stem.items():
            lst = sorted(set(lst), key=lambda x: (rows[units[x]["rep"]]["split"] not in ("val", "test"),
                                                  int(rows[units[x]["rep"]]["aug_marker"]),
                                                  rows[units[x]["rep"]]["relpath"]))
            kept = []
            for x in lst:
                if x not in keep:
                    continue
                dmin = min((int(_popcount(UH[x] ^ UH[k, 0]).min()) for k in kept), default=99)
                if dmin <= a.dedupe_phash:
                    keep.discard(x)
                    dropped.append((x, stem, dmin))
                else:
                    kept.append(x)
    log.append(f"variants dropped: {len(dropped)}; images kept: {len(keep)}")

    # 5. group-level split ------------------------------------------------------------
    gclass = defaultdict(Counter)
    gsize = Counter()
    for x in keep:
        gsize[gid[x]] += 1
        for b in units[x]["boxes"]:
            gclass[gid[x]][b[0]] += 1
    total = Counter()
    for c in gclass.values():
        total.update(c)
    want = {c: a.val_frac * n for c, n in total.items()}
    cap = a.val_frac * len(keep) * 1.15
    rng = random.Random(a.seed)
    glist = list(gsize)
    rng.shuffle(glist)
    labelled = sorted([g for g in glist if gclass[g]],
                      key=lambda g: min(total[c] for c in gclass[g]))
    background = [g for g in glist if not gclass[g]]
    split, vcount, vimg = {}, Counter(), 0
    for g in labelled:
        need = sum(max(0.0, want[c] - vcount[c]) / total[c] for c in gclass[g])
        over = sum(max(0.0, vcount[c] + n - want[c]) / total[c] for c, n in gclass[g].items())
        if vimg + gsize[g] <= cap and need > over:
            split[g] = "val"
            vcount.update(gclass[g])
            vimg += gsize[g]
        else:
            split[g] = "train"
    for g in background:
        if vimg + gsize[g] <= a.val_frac * len(keep):
            split[g] = "val"
            vimg += gsize[g]
        else:
            split[g] = "train"
    log.append(f"val images {vimg} of {len(keep)} ({vimg / len(keep):.1%})")

    # 6. project and write ---------------------------------------------------------
    out.mkdir(parents=True, exist_ok=True)
    counts = {d: defaultdict(Counter) for d in DOMAINS}
    imgcounts = {d: defaultdict(Counter) for d in DOMAINS}
    nimg = {d: Counter() for d in DOMAINS}
    for d in DOMAINS:
        for s in ("train", "val"):
            (out / d / "images" / s).mkdir(parents=True, exist_ok=True)
            (out / d / "labels" / s).mkdir(parents=True, exist_ok=True)
        cidx = {c: k for k, c in enumerate(order[d])}
        for x in sorted(keep):
            u = units[x]
            if d not in u["ann"]:
                continue
            s = split[gid[x]]
            src = Path(rows[u["rep"]]["abspath"])
            name = f"u{x:05d}_{src.name}"
            shutil.copy2(src, out / d / "images" / s / name)
            lines = []
            for b in u["boxes"]:
                if b[0] in cidx:
                    lines.append(f"{cidx[b[0]]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}")
                    counts[d][b[0]][s] += 1
            for c in {b[0] for b in u["boxes"] if b[0] in cidx}:
                imgcounts[d][c][s] += 1
            (out / d / "labels" / s / (Path(name).stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            nimg[d][s] += 1
        (out / d / "data.yaml").write_text(
            f"path: {(out / d).resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
            f"names:\n" + "".join(f"  {k}: {c}\n" for k, c in enumerate(order[d])),
            encoding="utf-8")

    with open(out / "split.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["uid", "group", "split", "domains", "classes", "export_stems", "files"])
        for x in sorted(keep):
            u = units[x]
            w.writerow([x, gid[x], split[gid[x]], "|".join(sorted(u["ann"])),
                        "|".join(sorted({b[0] for b in u["boxes"]})),
                        "|".join(sorted({rows[i]["export_stem"] for i in u["members"]})),
                        "|".join(rows[i]["dataset"] + ":" + rows[i]["relpath"] for i in u["members"])])
    with open(out / "dropped.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["uid", "export_stem", "phash_to_kept", "files"])
        for x, stem, dmin in dropped:
            w.writerow([x, stem, dmin, "|".join(rows[i]["dataset"] + ":" + rows[i]["relpath"]
                                                for i in units[x]["members"])])

    # 7. audit -----------------------------------------------------------------------
    sp = {x: split[gid[x]] for x in keep}
    kl = sorted(keep)
    fails = {}
    stem_sp = defaultdict(set)
    for x in kl:
        for i in units[x]["members"]:
            stem_sp[rows[i]["export_stem"]].add(sp[x])
    fails["export_stem_across_split"] = sum(len(v) > 1 for v in stem_sp.values())
    sha_sp = defaultdict(set)
    for x in kl:
        for i in units[x]["members"]:
            sha_sp[rows[i]["sha256"]].add(sp[x])
    fails["sha256_across_split"] = sum(len(v) > 1 for v in sha_sp.values())
    vid_sp = defaultdict(set)
    for x in kl:
        for i in units[x]["members"]:
            v = video_key(rows[i]["export_stem"])
            if v:
                vid_sp[v].add(sp[x])
    fails["video_across_split"] = sum(len(v) > 1 for v in vid_sp.values())
    KH = UH[kl]
    near = near_pairs(KH, a.audit_near)
    cross = [(kl[i], kl[j], d) for i, j, d in near if sp[kl[i]] != sp[kl[j]]]
    fails["phash_le_group_threshold_across_split"] = sum(d <= a.group_phash for _, _, d in cross)
    if a.burst_gap > 0:
        bl = 0
        byp = defaultdict(list)
        for x in kl:
            r = rows[units[x]["rep"]]
            if r["burst_prefix"]:
                byp[r["burst_prefix"].split("::", 1)[-1]].append((int(r["burst_counter"]), x))
        for lst in byp.values():
            lst.sort()
            for (c1, i1), (c2, i2) in zip(lst, lst[1:]):
                if c2 - c1 <= a.burst_gap and sp[i1] != sp[i2]:
                    bl += 1
        fails["burst_link_across_split"] = bl
    info = {"cross_split_pairs_within_audit_near": len(cross),
            "cross_split_pair_distances": dict(Counter(d for _, _, d in cross))}

    A = ["# Build audit\n", *[f"- {x}" for x in log], ""]
    A.append("## Leak checks across the new boundary (all must be 0)\n")
    A += [f"- {k}: **{v}**" for k, v in fails.items()]
    A.append(f"- informational: cross-split pairs within pHash {a.audit_near}: "
             f"{info['cross_split_pairs_within_audit_near']} {info['cross_split_pair_distances']} "
             "(listed in audit_cross_pairs.csv; look at them)\n")
    for d in DOMAINS:
        A.append(f"## {d}: {nimg[d]['train']} train / {nimg[d]['val']} val images\n")
        A.append("| class | train inst. | val inst. | val share | train img. | val img. |")
        A.append("|---|---|---|---|---|---|")
        for c in order[d]:
            t, v = counts[d][c]["train"], counts[d][c]["val"]
            ti, vi = imgcounts[d][c]["train"], imgcounts[d][c]["val"]
            flag = " ⚠" if v < 10 or vi < 5 else ""
            A.append(f"| {c} | {t} | {v} | {v / max(t + v, 1):.0%}{flag} | {ti} | {vi} |")
        A.append("")
    A.append("## Five largest groups (all stay together)\n")
    gm = defaultdict(list)
    for x in kl:
        gm[gid[x]].append(x)
    for g, mem in sorted(gm.items(), key=lambda t: -len(t[1]))[:5]:
        cl = Counter(c for x in mem for c in {b[0] for b in units[x]["boxes"]} or {"(background)"})
        ex = sorted(rows[units[x]["rep"]]["relpath"] for x in mem)[:3]
        A.append(f"- group {g}: {len(mem)} images, {split[g]}; image classes {dict(cl.most_common(6))}; e.g. {ex}")
    A.append("")
    (out / "audit.md").write_text("\n".join(A), encoding="utf-8")
    json.dump({"log": log, "fails": fails, "info": info,
               "counts": {d: {c: dict(v) for c, v in counts[d].items()} for d in DOMAINS},
               "images": {d: dict(nimg[d]) for d in DOMAINS}, "args": vars(a)},
              open(out / "audit.json", "w", encoding="utf-8"), indent=2)
    with open(out / "audit_cross_pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["d", "uid_a", "split_a", "file_a", "uid_b", "split_b", "file_b"])
        for x, y, d in sorted(cross, key=lambda t: t[2]):
            w.writerow([d, x, sp[x], rows[units[x]["rep"]]["relpath"],
                        y, sp[y], rows[units[y]["rep"]]["relpath"]])

    print("\n".join(log))
    for k, v in fails.items():
        print(f"{k}: {v}")
    print(f"-> {out / 'audit.md'}")
    if any(fails.values()):
        sys.exit("LEAK CHECK FAILED: do not train on this build.")


if __name__ == "__main__":
    main()
