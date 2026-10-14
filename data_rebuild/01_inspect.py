#!/usr/bin/env python3
"""
01_inspect.py - read-only inventory of the two pre-merge datasets.

Nothing is modified. Produces, in --out:
  report.md      what is in each dataset and what is wrong with it
  inventory.csv  one row per image
  hashes.npy     dihedral perceptual hashes (reused by 02_build.py)
  labels.json    parsed boxes per image, with class NAMES (reused by 02_build.py)
  pairs.csv      every image pair within perceptual distance --near, for eyeballing

  python 01_inspect.py --leaf "D:\\...\\Detection_leaf_disease" --pest "D:\\...\\Detection_pest_disease"
"""

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from common import (DSU, aug_marker, burst_key, class_names_from_yaml, export_stem,
                    find_images, find_yamls, has_rf, label_path_for, near_pairs,
                    orig_split, phash_dihedral, read_labels, sha256)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--leaf", required=True)
    p.add_argument("--pest", required=True)
    p.add_argument("--background", default=None,
                   help="optional folder of unannotated negatives (counts for both domains)")
    p.add_argument("--out", default="inspect_out")
    p.add_argument("--near", type=int, default=10, help="report pairs up to this Hamming distance")
    p.add_argument("--burst-gap", type=int, default=3)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    datasets = {"leaf": Path(a.leaf), "pest": Path(a.pest)}
    if a.background:
        datasets["bg"] = Path(a.background)
    # A folder may hold several Roboflow exports (one per project); each image takes
    # its class names from the nearest data.yaml above it.
    names, yaml_notes, yaml_of_dir = {}, {}, {}
    for ds, root in datasets.items():
        if not root.is_dir():
            sys.exit(f"not a directory: {root}")
        ys = find_yamls(root)
        if not ys and ds != "bg":
            n_lab = sum(1 for _ in root.rglob("*.txt"))
            sys.exit(f"no data.yaml with class names under {root} ({n_lab} .txt files). "
                     "This looks like raw, unannotated images. Point --leaf/--pest at the "
                     "annotated YOLO exports (unzipped Roboflow downloads) instead.")
        names[ds] = []
        for yp, n in ys:
            yaml_of_dir[yp.parent.resolve()] = n
            names[ds] += [c for c in n if c not in names[ds]]
        yaml_notes[ds] = [(str(p.relative_to(root)), n) for p, n in ys]

    def export_of(img, root):
        d = img.parent.resolve()
        rootr = root.resolve()
        while True:
            if d in yaml_of_dir:
                return d, yaml_of_dir[d]
            if d == rootr or d.parent == d:
                return None, []
            d = d.parent

    rows, H, labels = [], [], {}
    t0 = time.time()
    for ds, root in datasets.items():
        imgs = find_images(root)
        print(f"{ds}: {len(imgs)} images")
        for k, img in enumerate(imgs, 1):
            lp = label_path_for(img)
            exp_dir, enames = export_of(img, root)
            exp = "" if exp_dir is None else str(exp_dir.relative_to(root.resolve()))
            if ds == "bg" and lp is None:
                boxes, issues = [], []
            else:
                boxes, issues = read_labels(lp, len(enames))
            size, hs = phash_dihedral(img)
            stem = export_stem(img)
            bk = burst_key(stem)
            idx = len(rows)
            rows.append({
                "idx": idx, "dataset": ds, "split": orig_split(img, root),
                "relpath": str(img.relative_to(root)), "abspath": str(img.resolve()),
                "label_path": "" if lp is None else str(lp.resolve()),
                "sha256": sha256(img), "w": size[0], "h": size[1],
                "n_boxes": len(boxes),
                "export": exp,
                "classes": "|".join(sorted({enames[b[0]] for b in boxes})),
                "issues": "|".join(sorted(set(issues))),
                "export_stem": stem, "has_rf": int(has_rf(img)), "aug_marker": int(aug_marker(img)),
                "burst_prefix": "" if bk is None else f"{ds}/{exp}::{bk[0]}",
                "burst_counter": "" if bk is None else bk[1]})
            labels[idx] = [[enames[b[0]], *b[1:]] for b in boxes]
            H.append(hs)
            if k % 200 == 0:
                print(f"  {k}/{len(imgs)}  ({time.time() - t0:.0f} s)")
    if not rows:
        sys.exit("no images found under either folder")
    H = np.array(H, dtype=np.uint64)

    with open(out / "inventory.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.save(out / "hashes.npy", H)
    with open(out / "labels.json", "w", encoding="utf-8") as f:
        json.dump({"names": names, "labels": labels}, f)

    print("near-duplicate search ...")
    pairs = near_pairs(H, a.near)
    with open(out / "pairs.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["d", "i", "j", "dataset_i", "split_i", "file_i", "dataset_j", "split_j", "file_j"])
        for i, j, d in sorted(pairs, key=lambda x: x[2]):
            w.writerow([d, i, j, rows[i]["dataset"], rows[i]["split"], rows[i]["relpath"],
                        rows[j]["dataset"], rows[j]["split"], rows[j]["relpath"]])

    # ------------------------------------------------------------ report
    R = []
    say = R.append
    say("# Pre-merge dataset inspection\n")
    for ds, root in datasets.items():
        say(f"## {ds}: `{root}`\n")
        say(f"- yaml files with class names: {len(yaml_notes[ds])}")
        for rel, n in yaml_notes[ds]:
            say(f"  - `{rel}`: {len(n)} classes {n}")
        if len({tuple(n) for _, n in yaml_notes[ds]}) > 1:
            say("  - several exports with different class lists: each image uses the "
                "data.yaml of its own export")
        sub = [r for r in rows if r["dataset"] == ds]
        sc = Counter(r["split"] for r in sub)
        say(f"- images: {len(sub)}  by original split: {dict(sc)}")
        say(f"- image sizes (top 5): {Counter((r['w'], r['h']) for r in sub).most_common(5)}")
        say(f"- with `.rf.` marker: {sum(r['has_rf'] for r in sub)}; "
            f"`aug` in name: {sum(r['aug_marker'] for r in sub)}")
        iss = Counter(x for r in sub for x in r["issues"].split("|") if x)
        say(f"- label issues: {dict(iss) or 'none'}")
        say(f"- images with zero boxes: {sum(r['n_boxes'] == 0 for r in sub)}")
        inst = defaultdict(Counter)
        for r in sub:
            for b in labels[r["idx"]]:
                inst[b[0]][r["split"]] += 1
        say("\n| class | " + " | ".join(sorted(sc)) + " | total | images |")
        say("|---|" + "---|" * (len(sc) + 2))
        for c in names[ds]:
            nimg = sum(1 for r in sub if c in r["classes"].split("|"))
            say(f"| {c} | " + " | ".join(str(inst[c][s]) for s in sorted(sc))
                + f" | {sum(inst[c].values())} | {nimg} |")
        stems = Counter(r["export_stem"] for r in sub)
        multi = {s: n for s, n in stems.items() if n > 1}
        say(f"\n- distinct export stems: {len(stems)}; stems with >1 file: {len(multi)} "
            f"covering {sum(multi.values())} files")
        cross = 0
        for s in multi:
            if len({r["split"] for r in sub if r["export_stem"] == s}) > 1:
                cross += 1
        say(f"- stems whose files sit in more than one ORIGINAL split "
            f"(augment-before-split leak in the source export): **{cross}**")
        say("- 15 random filenames:")
        for r in random.Random(0).sample(sub, min(15, len(sub))):
            say(f"  - `{r['relpath']}`")
        say("")

    say("## Across both datasets\n")
    ln = {ds: {c.lower(): c for c in n} for ds, n in names.items()}
    shared = sorted(set(ln["leaf"]) & set(ln["pest"]))
    say(f"- class names present in both (case-insensitive): {[ (ln['leaf'][s], ln['pest'][s]) for s in shared ] or 'none'}")
    for ds, n in names.items():
        low = Counter(c.lower() for c in n)
        d = [c for c in n if low[c.lower()] > 1]
        if d:
            say(f"- **{ds} has case-variant duplicate classes: {d}**")

    by_sha = defaultdict(list)
    for r in rows:
        by_sha[r["sha256"]].append(r)
    dup = [g for g in by_sha.values() if len(g) > 1]
    say(f"- byte-identical groups: {len(dup)} covering {sum(len(g) for g in dup)} files")
    say(f"  - spanning both datasets: {sum(len({r['dataset'] for r in g}) > 1 for g in dup)}")
    say(f"  - spanning original splits within one dataset: "
        f"{sum(any(len({r['split'] for r in g if r['dataset'] == ds}) > 1 for ds in datasets) for g in dup)}")
    conflict = 0
    for g in dup:
        for ds in datasets:
            ls = [tuple(sorted(map(tuple, labels[r['idx']]))) for r in g if r["dataset"] == ds]
            if len(set(ls)) > 1:
                conflict += 1
    say(f"  - groups where copies in one dataset carry DIFFERENT labels: {conflict}")

    dist = Counter(d for _, _, d in pairs)
    say(f"- perceptual pairs (dihedral pHash) by distance <= {a.near}: "
        f"{dict(sorted(dist.items()))}  (all listed in pairs.csv)")
    xds = sum(1 for i, j, d in pairs if rows[i]["dataset"] != rows[j]["dataset"])
    say(f"  - of which across datasets: {xds}")

    if a.burst_gap > 0:
        # bursts are links between DIFFERENT export stems; copies of one stem are
        # already reported above
        stem_of = {}
        for r in rows:
            stem_of.setdefault((r["dataset"], r["export_stem"]), r)
        dsu = DSU(len(rows))
        byp = defaultdict(list)
        for (ds, stem), r in stem_of.items():
            if r["burst_prefix"]:
                byp[(ds, r["burst_prefix"])].append((int(r["burst_counter"]), r["idx"]))
        for lst in byp.values():
            lst.sort()
            for (c1, i1), (c2, i2) in zip(lst, lst[1:]):
                if c2 - c1 <= a.burst_gap:
                    dsu.union(i1, i2)
        bg = [g for g in dsu.groups() if len(g) > 1]
        def splits_of(g):
            return {r["split"] for i in g for r in rows
                    if r["dataset"] == rows[i]["dataset"] and r["export_stem"] == rows[i]["export_stem"]}
        strad = sum(len(splits_of(g)) > 1 for g in bg)
        say(f"- burst groups (different export stems, same name prefix, counter gap <= {a.burst_gap}): "
            f"{len(bg)} groups of {sum(map(len, bg))} stems; straddling original splits: {strad}")
        top = Counter(r["burst_prefix"] for r in rows if r["burst_prefix"]).most_common(10)
        say(f"  - most common name prefixes: {top}")

    (out / "report.md").write_text("\n".join(R), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f} s -> {out / 'report.md'}")


if __name__ == "__main__":
    main()
