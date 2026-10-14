#!/usr/bin/env python3
"""
05_check_external.py - is an external image set (e.g. the Sabah orchards) independent of the
images used to train and validate this paper's detectors?

  python 05_check_external.py --external PATH/sabah --exports PATH/disease_640 PATH/pest_640 --out sabah_check

Compares every external image with every image in the exports (train and val alike):
  * SHA-256: byte-identical files
  * 64-bit DCT perceptual hash under the 8 dihedral transforms (same code as the rebuild),
    reporting the nearest export image for each external image and counting pairs at
    Hamming distance <= 2 (the rebuild's grouping threshold) and <= 10 (its caution threshold)
Also inventories the external set: images, label files, boxes per class.
Writes <out>/nearest.csv, <out>/matches.csv and <out>/summary.json, and prints the summary.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import _popcount, phash_dihedral, sha256  # noqa: E402

IMG = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def images_under(root):
    return sorted(p for p in Path(root).rglob("*") if p.suffix.lower() in IMG and p.is_file())


def label_for(img):
    cands = [img.with_suffix(".txt"),
             Path(str(img.parent).replace("images", "labels")) / (img.stem + ".txt")]
    for c in cands:
        if c.exists():
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", required=True)
    ap.add_argument("--exports", nargs="+", required=True)
    ap.add_argument("--out", default="external_check")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    ext = images_under(a.external)
    exp = []
    for e in a.exports:
        exp += images_under(e)
    print(f"external images: {len(ext)}; export images: {len(exp)}")
    if not ext or not exp:
        sys.exit("nothing to compare")

    # inventory of the external set
    classes = Counter()
    n_lab = 0
    n_empty = 0
    for p in ext:
        lab = label_for(p)
        if lab is None:
            continue
        n_lab += 1
        rows = [r.split() for r in lab.read_text().splitlines() if r.strip()]
        if not rows:
            n_empty += 1
        for r in rows:
            classes[r[0]] += 1

    print("hashing exports ...")
    exp_sha = {}
    exp_h = []
    for i, p in enumerate(exp):
        exp_sha.setdefault(sha256(p), []).append(str(p))
        _, hs = phash_dihedral(p)
        exp_h.append(hs)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(exp)}")
    H = np.array(exp_h, dtype=np.uint64)       # (n_exp, 8)
    base = H[:, 0]

    print("hashing external ...")
    rows = []
    sha_hits = []
    for i, p in enumerate(ext):
        s = sha256(p)
        if s in exp_sha:
            sha_hits.append((str(p), exp_sha[s]))
        _, hs = phash_dihedral(p)
        hs = np.array(hs, dtype=np.uint64)     # (8,)
        d = _popcount(hs[:, None] ^ base[None, :]).min(axis=0)   # (n_exp,)
        j = int(d.argmin())
        rows.append({"external": str(p), "nearest_export": str(exp[j]), "hamming": int(d[j]),
                     "sha256_identical": s in exp_sha})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(ext)}")

    import csv
    with open(out / "nearest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    matches = [r for r in rows if r["hamming"] <= 10 or r["sha256_identical"]]
    with open(out / "matches.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(matches)

    ham = np.array([r["hamming"] for r in rows])
    summary = {
        "external_images": len(ext), "export_images": len(exp),
        "external_with_label_file": n_lab, "external_label_files_empty": n_empty,
        "external_boxes_per_class_id": dict(sorted(classes.items())),
        "sha256_identical": len(sha_hits),
        "phash_le_2": int((ham <= 2).sum()), "phash_le_10": int((ham <= 10).sum()),
        "phash_min": int(ham.min()), "phash_median": float(np.median(ham)),
        "verdict": ("INDEPENDENT: no byte-identical file and no pair within Hamming 10"
                    if not sha_hits and (ham > 10).all() else
                    "NOT INDEPENDENT: see matches.csv"),
    }
    json.dump(summary, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
