"""Shared helpers for the dataset rebuild: discovery, labels, hashing, grouping."""

import hashlib
import re
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = {"train": "train", "valid": "val", "val": "val", "test": "test"}


# ---------------------------------------------------------------- discovery

def load_yaml(path):
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        raise SystemExit("PyYAML is required (it is installed with ultralytics).")


def class_names_from_yaml(d):
    names = d.get("names")
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=lambda x: int(x))]
    if isinstance(names, list):
        return list(names)
    return None


def find_yamls(root):
    out = []
    for p in sorted(Path(root).rglob("*.y*ml")):
        try:
            d = load_yaml(p)
        except Exception:
            continue
        if isinstance(d, dict) and class_names_from_yaml(d):
            out.append((p, class_names_from_yaml(d)))
    return out


def find_images(root):
    root = Path(root)
    return sorted(p for p in root.rglob("*")
                  if p.suffix.lower() in IMG_EXT
                  and not any(part.startswith(".") for part in p.relative_to(root).parts))


def label_path_for(img):
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            cand = Path(*parts[:i], "labels", *parts[i + 1:]).with_suffix(".txt")
            if cand.exists():
                return cand
    cand = img.with_suffix(".txt")
    return cand if cand.exists() else None


def orig_split(img, root):
    for part in img.relative_to(root).parts[:-1]:
        if part.lower() in SPLIT_NAMES:
            return SPLIT_NAMES[part.lower()]
    return "unknown"


# ---------------------------------------------------------------- names

RF_RE = re.compile(r"\.rf\.[0-9a-fA-F]+$")
EXT_TAIL_RE = re.compile(r"_(jpe?g|png|bmp|webp|heic)$", re.I)


def export_stem(img):
    """Pre-export filename preserved by Roboflow before '.rf.<hash>'.
    'IMG_1234_jpg.rf.abcd' -> 'IMG_1234'. Files without the marker keep their stem."""
    s = RF_RE.sub("", img.stem)
    return EXT_TAIL_RE.sub("", s)


def has_rf(img):
    return bool(RF_RE.search(img.stem))


AUG_RE = re.compile(r"(^|[_\-.])aug", re.I)


def aug_marker(img):
    return bool(AUG_RE.search(img.stem))


DIGITS_RE = re.compile(r"\d+")


def burst_key(stem):
    """(prefix, counter): the last digit run is the counter, everything else
    the prefix. 'IMG_1234' -> ('IMG_#', 1234). None when there is no number."""
    runs = list(DIGITS_RE.finditer(stem))
    if not runs:
        return None
    m = runs[-1]
    if len(m.group()) > 9:          # a long run is an ID, not a frame counter
        return None
    return stem[:m.start()] + "#" + stem[m.end():], int(m.group())


VIDEO_RE = re.compile(r"^(.*?\d.*?)[_\-. ]*frame[_\-. ]*\d+", re.I)


def video_key(stem):
    """Frames extracted from one video share everything before 'frame<N>'.
    'durian_phomopsis__img_9976_frame01434' -> 'durian_phomopsis__img_9976'.
    Only when that prefix carries a number (a video id); a bare 'frame_0012'
    gives None, because unrelated videos would otherwise be pooled."""
    m = VIDEO_RE.match(stem)
    return m.group(1).lower() if m else None


# ---------------------------------------------------------------- labels

def read_labels(path, n_classes):
    """Returns (boxes, issues). boxes: list of (cls, cx, cy, w, h).
    Polygon (segmentation) lines are converted to their bounding box."""
    boxes, issues = [], []
    if path is None:
        return boxes, ["missing_label_file"]
    for ln, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        t = line.split()
        if not t:
            continue
        try:
            c = int(float(t[0]))
            v = [float(x) for x in t[1:]]
        except ValueError:
            issues.append(f"unparsable_line:{ln}")
            continue
        if len(v) == 4:
            cx, cy, w, h = v
        elif len(v) >= 6 and len(v) % 2 == 0:
            xs, ys = v[0::2], v[1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
            issues.append("polygon")
        else:
            issues.append(f"bad_token_count:{ln}")
            continue
        if not (0 <= c < n_classes):
            issues.append(f"class_out_of_range:{c}")
            continue
        if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1.0001 and 0 < h <= 1.0001):
            issues.append("coord_out_of_range")
        boxes.append((c, cx, cy, w, h))
    return boxes, issues


def iou(a, b):
    ax0, ay0, ax1, ay1 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx0, by0, bx1, by1 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    u = a[3] * a[4] + b[3] * b[4] - inter
    return inter / u if u > 0 else 0.0


def union_boxes(lists, thr=0.9):
    """Union of several label sets for one image; same-class boxes with
    IoU > thr are treated as the same object."""
    out = []
    for boxes in lists:
        for b in boxes:
            if not any(o[0] == b[0] and iou(o, b) > thr for o in out):
                out.append(b)
    return out


# ---------------------------------------------------------------- hashing

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_N = 32
_C = np.cos(np.pi * (2 * np.arange(_N)[None, :] + 1) * np.arange(_N)[:, None] / (2 * _N))


def _phash_of(a):
    d = _C @ a @ _C.T
    low = d[:8, :8].flatten()
    bits = low > np.median(low[1:])
    return int(np.packbits(bits).view(">u8")[0])


def phash_dihedral(path):
    """64-bit DCT perceptual hash under the eight dihedral transforms.
    Returns (size, [8 hashes]); index 0 is the untransformed image."""
    with Image.open(path) as im:
        size = im.size
        a = np.asarray(im.convert("L").resize((_N, _N), Image.LANCZOS), dtype=np.float64)
    vs = []
    for k in range(4):
        r = np.rot90(a, k)
        vs += [_phash_of(r), _phash_of(np.fliplr(r))]
    return size, vs


def _popcount(x):
    if hasattr(np, "bitwise_count"):
        return np.bitwise_count(x)
    t = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    return t[x.view(np.uint8)].reshape(x.shape + (8,)).sum(-1)


def near_pairs(H, max_d, chunk=256):
    """H: (n, 8) uint64 dihedral hashes. Returns list of (i, j, d), i < j,
    where d is the minimum Hamming distance between image i under any
    transform and image j untransformed."""
    n = len(H)
    base = H[:, 0]
    out = []
    for s in range(0, n, chunk):
        blk = H[s:s + chunk]                                   # (b, 8)
        x = blk[:, :, None] ^ base[None, None, :]              # (b, 8, n)
        d = _popcount(x).min(axis=1)                           # (b, n)
        ii, jj = np.nonzero(d <= max_d)
        for i, j in zip(ii + s, jj):
            if i < j:
                out.append((int(i), int(j), int(d[i - s, j])))
    return out


# ---------------------------------------------------------------- union-find

class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)

    def groups(self):
        g = {}
        for i in range(len(self.p)):
            g.setdefault(self.find(i), []).append(i)
        return list(g.values())
